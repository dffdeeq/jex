"""Asynchronous question engine.

Questions are independent units of work. A caller can submit one question at
a time, at any moment, about any state, and await its own answer; it never
waits for other questions and nothing is generated token by token.

How it works:

* every state is prefilled once and its key/values are kept in an LRU cache;
  a question about a known state costs only its own ~50-100 tokens;
* a single worker drains the queue continuously: whatever questions arrived
  while the previous pass was running are packed, per state, as isolated
  branches into one forward pass (continuous micro-batching);
* every answer resolves its own future as soon as its pass finishes, so
  results can be streamed in completion order;
* a *session* is a live state that grows (chat, logs, game events): appends
  compute only the new tokens, and questions see exactly the events that were
  appended before them.

The model runs in a worker thread, so the event loop keeps accepting
questions while a pass is computing.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .backbone import StateCache
from .model import JexModel
from .schema import Question, parse_question, parse_request, render_state


@dataclass
class _Pending:
    state_key: str
    state: Any
    question: Question
    future: asyncio.Future
    submitted: float = field(default_factory=time.perf_counter)


@dataclass
class _SessionOp:
    """Open a session or append text to it; ordered with the questions."""

    key: str
    kind: str  # "open" | "append" | "close"
    payload: Any
    future: asyncio.Future


@dataclass
class EngineStats:
    passes: int = 0
    questions: int = 0
    prefills: int = 0
    cache_hits: int = 0
    appended_tokens: int = 0
    latencies: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)


def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


class AsyncEngine:
    def __init__(
        self,
        model: JexModel,
        max_batch_questions: int = 64,
        max_wait_ms: float = 1.0,
        cache_size: int = 64,
    ):
        self.model = model
        self.max_batch = max_batch_questions
        self.max_wait = max_wait_ms / 1000
        self.cache_size = cache_size
        self._cache: OrderedDict[str, StateCache] = OrderedDict()
        self._sessions: dict[str, StateCache] = {}  # pinned, never evicted
        self._queue: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self.stats = EngineStats()

    # --------------------------------------------------------------- lifecycle

    async def start(self) -> "AsyncEngine":
        if self._worker is None:
            self._queue = asyncio.Queue()
            self._worker = asyncio.create_task(self._run())
        return self

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def __aenter__(self) -> "AsyncEngine":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    # ------------------------------------------------------------------ public

    def _put(self, item) -> asyncio.Future:
        if self._queue is None:
            raise RuntimeError("engine is not started")
        self._queue.put_nowait(item)
        return item.future

    def submit(self, state: Any, name: str, spec: dict[str, Any], session: str | None = None) -> asyncio.Future:
        """Enqueue one typed question; returns a future of (name, answer JSON).
        With ``session`` the question is asked about that live state instead."""
        question = parse_question(name, spec)
        key = _session_key(session) if session is not None else hashlib.sha1(render_state(state).encode()).hexdigest()
        fut = asyncio.get_running_loop().create_future()
        return self._put(_Pending(key, state, question, fut))

    async def ask(self, state: Any, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        _, answer = await self.submit(state, name, spec)
        return answer

    async def stream(self, payload: dict[str, Any]) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield (question name, answer) in completion order."""
        request = parse_request(payload)
        futs = [self.submit(request.state, q.name, payload["questions"][q.name]) for q in request.questions]
        for done in asyncio.as_completed(futs):
            yield await done

    async def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Jev-compatible request/response, answered through the async queue."""
        request = parse_request(payload)
        answers = await asyncio.gather(
            *(self.submit(request.state, q.name, payload["questions"][q.name]) for q in request.questions)
        )
        return {"model": self.model.name, "answers": dict(answers), "usage": {"output_tokens": 0}}

    # ---------------------------------------------------------------- sessions

    def _session_op(self, session_id: str, kind: str, payload: Any = None) -> asyncio.Future:
        fut = asyncio.get_running_loop().create_future()
        return self._put(_SessionOp(_session_key(session_id), kind, payload, fut))

    # Session ops are enqueued when called (not when awaited), so their order
    # relative to submitted questions is exactly the call order.

    def open_session(self, session_id: str, state: Any) -> asyncio.Future:
        return self._session_op(session_id, "open", state)

    def append(self, session_id: str, text: str) -> asyncio.Future:
        """Add events to a live state; later questions see them, earlier ones do not."""
        return self._session_op(session_id, "append", text)

    def close_session(self, session_id: str) -> asyncio.Future:
        return self._session_op(session_id, "close")

    async def ask_session(self, session_id: str, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        _, answer = await self.submit(None, name, spec, session=session_id)
        return answer

    # ------------------------------------------------------------------ worker

    async def _collect(self) -> list:
        batch = [await self._queue.get()]
        deadline = time.perf_counter() + self.max_wait
        while len(batch) < self.max_batch:
            try:
                batch.append(self._queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), remaining))
            except asyncio.TimeoutError:
                break
        return batch

    async def _run(self) -> None:
        while True:
            batch = await self._collect()
            groups: OrderedDict[str, list[_Pending]] = OrderedDict()
            for item in batch:
                if isinstance(item, _SessionOp):
                    # Questions queued before this op must not see its effect.
                    if item.key in groups:
                        await self._answer(groups.pop(item.key))
                    await self._apply(item)
                else:
                    groups.setdefault(item.state_key, []).append(item)
            for items in groups.values():
                await self._answer(items)

    async def _apply(self, op: _SessionOp) -> None:
        try:
            await asyncio.to_thread(self._apply_sync, op)
        except Exception as exc:
            if not op.future.done():
                op.future.set_exception(exc)
            return
        if not op.future.done():
            op.future.set_result(None)

    def _apply_sync(self, op: _SessionOp) -> None:
        if op.kind == "open":
            self._sessions[op.key] = self.model.prefill(op.payload)
            self.stats.prefills += 1
        elif op.kind == "append":
            cache = self._sessions[op.key]
            before = cache.length
            self.model.extend_state(cache, op.payload)
            self.stats.appended_tokens += cache.length - before
        elif op.kind == "close":
            self._sessions.pop(op.key, None)

    async def _answer(self, items: list[_Pending]) -> None:
        live = [p for p in items if not p.future.done()]
        if not live:
            return
        try:
            answers = await asyncio.to_thread(self._compute, live)
        except Exception as exc:  # surface model errors to every waiter
            self._cache.pop(live[0].state_key, None)
            for p in live:
                if not p.future.done():
                    p.future.set_exception(exc)
            return
        now = time.perf_counter()
        for p, a in zip(live, answers):
            self.stats.latencies.append(now - p.submitted)
            if not p.future.done():
                p.future.set_result((p.question.name, a.to_json()))
        self.stats.passes += 1
        self.stats.questions += len(live)
        self.stats.batch_sizes.append(len(live))

    def _state(self, key: str, state: Any) -> StateCache:
        if key in self._sessions:
            return self._sessions[key]
        if key.startswith("session:"):
            raise KeyError(f"unknown session {key[len('session:'):]!r}")
        cache = self._cache.get(key)
        if cache is not None:
            self._cache.move_to_end(key)
            self.stats.cache_hits += 1
            return cache
        cache = self.model.prefill(state)
        self.stats.prefills += 1
        self._cache[key] = cache
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cache

    def _compute(self, items: list[_Pending]):
        cache = self._state(items[0].state_key, items[0].state)
        return self.model.answer_cached(cache, [p.question for p in items])
