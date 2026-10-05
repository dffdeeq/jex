import asyncio

import pytest
import torch

from jex.engine import AsyncEngine
from jex.head import DecisionHead, HeadConfig
from jex.model import JexModel
from jex.schema import SchemaError


def _close(a, b, tol=1e-4):
    if a["type"] == "noul":
        return abs(a["noul"] - b["noul"]) < tol
    return all(abs(a["probabilities"][k] - b["probabilities"][k]) < tol for k in a["probabilities"])


@pytest.fixture(params=["verbalizer", "head"])
def model(request, tiny_backbone):
    if request.param == "verbalizer":
        return JexModel(tiny_backbone)
    torch.manual_seed(0)
    head = DecisionHead(HeadConfig(d_backbone=tiny_backbone.hidden_size, d_model=32, n_layers=1, n_heads=2))
    return JexModel(tiny_backbone, head, temperatures={"choice": 1.5})


def test_engine_matches_direct_predict(model, payload):
    direct = model.predict(payload)

    async def go():
        async with AsyncEngine(model) as eng:
            return await eng.evaluate(payload)

    via_engine = asyncio.run(go())
    for name, ans in direct["answers"].items():
        assert _close(ans, via_engine["answers"][name])


def test_concurrent_questions_from_many_states(model, payload):
    states = [payload["state"], "second state", {"k": "third"}]
    q = {"type": "noul", "instructions": "Is it about billing?"}

    async def go():
        async with AsyncEngine(model, max_wait_ms=5) as eng:
            # interleave questions about different states; all are in flight at once
            futs = [eng.ask(s, f"q{i}", q) for i in range(4) for s in states]
            answers = await asyncio.gather(*futs)
            # a late question about a known state reuses the cached prefix
            late = await eng.ask(states[0], "late", q)
            return answers, late, eng.stats

    answers, late, stats = asyncio.run(go())
    assert len(answers) == 12
    expected = [model.predict({"state": s, "questions": {"q": q}})["answers"]["q"] for s in states]
    for i, a in enumerate(answers):
        assert _close(a, expected[i % 3])
    assert _close(late, expected[0])
    assert stats.prefills == 3 and stats.cache_hits >= 1
    assert stats.questions == 13


def test_stream_yields_every_answer(model, payload):
    async def go():
        async with AsyncEngine(model) as eng:
            return [name async for name, _ in eng.stream(payload)]

    assert sorted(asyncio.run(go())) == sorted(payload["questions"])


def test_invalid_question_fails_fast(model):
    async def go():
        async with AsyncEngine(model) as eng:
            with pytest.raises(SchemaError):
                eng.submit("s", "bad", {"type": "score", "instructions": "x", "criteria": ["one"]})
            # the engine keeps serving afterwards
            return await eng.ask("s", "ok", {"type": "noul", "instructions": "Is it?"})

    assert asyncio.run(go())["type"] == "noul"


def test_live_session_sees_only_earlier_events(model):
    start = "Game log. Turn 1: the player enters the cave."
    event = " Turn 2: a dragon attacks and the player's health drops to 5 percent."
    q = {"type": "noul", "instructions": "Is the player in danger?"}

    async def go():
        async with AsyncEngine(model) as eng:
            await eng.open_session("game", start)
            # queued back to back: the first question must not see the event
            before = eng.submit(None, "before", q, session="game")
            appended = eng.append("game", event)
            after = eng.submit(None, "after", q, session="game")
            (_, a_before), _, (_, a_after) = await asyncio.gather(before, appended, after)
            stats = eng.stats
            await eng.close_session("game")
            with pytest.raises(KeyError):
                await eng.ask_session("game", "gone", q)
            return a_before, a_after, stats

    a_before, a_after, stats = asyncio.run(go())
    expect_before = model.predict({"state": start, "questions": {"q": q}})["answers"]["q"]
    expect_after = model.predict({"state": start + event, "questions": {"q": q}})["answers"]["q"]
    assert _close(a_before, expect_before)
    assert _close(a_after, expect_after)
    assert stats.prefills == 1 and stats.appended_tokens > 0
