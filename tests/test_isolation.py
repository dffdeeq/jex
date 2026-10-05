"""The core claim: questions packed into one pass are answered exactly as if
each were asked alone, in any order, with or without a cached state."""

import torch

from jex.packing import pack_branches, pack_requests
from jex.schema import parse_request

ATOL = 1e-4


def _sub(payload, names):
    return {"state": payload["state"], "questions": {n: payload["questions"][n] for n in names}}


def _priors(feats):
    return [f.prior for f in feats]


def test_tree_mask_blocks_cross_branch_attention(tiny_backbone, payload):
    enc = tiny_backbone.encode(parse_request(payload))
    packed = pack_requests([enc], tiny_backbone.pad_id)
    allowed = packed.attention_mask[0, 0] == 0
    S = len(enc.prefix_ids)
    offsets = packed.branch_offsets[0]
    for i, (a, b) in enumerate(offsets):
        assert allowed[a:b, :S].all(), "branch must see the whole state"
        assert torch.equal(allowed[a:b, a:b], torch.ones(b - a, b - a, dtype=torch.bool).tril())
        for j, (c, d) in enumerate(offsets):
            if i != j:
                assert not allowed[a:b, c:d].any(), "branches must not see each other"
        # positions restart after the prefix for every branch
        assert packed.position_ids[0, a].item() == S
    assert not allowed[:S, S:].any(), "state must not see questions"


def test_packed_equals_separate_calls(any_backbone, payload):
    req = parse_request(payload)
    packed = any_backbone.run([any_backbone.encode(req)])[0]
    for q, f in zip(req.questions, packed):
        alone = any_backbone.run([any_backbone.encode(parse_request(_sub(payload, [q.name])))])[0][0]
        assert torch.allclose(alone.prior, f.prior, atol=ATOL)
        assert torch.allclose(alone.answer_hidden, f.answer_hidden, atol=ATOL)


def test_adding_and_reordering_questions_changes_nothing(any_backbone, payload):
    names = list(payload["questions"])
    base = any_backbone.run([any_backbone.encode(parse_request(payload))])[0]
    rev = any_backbone.run([any_backbone.encode(parse_request(_sub(payload, names[::-1])))])[0]
    for f, g in zip(base, rev[::-1]):
        assert torch.allclose(f.prior, g.prior, atol=ATOL)
    more = dict(payload, questions={**payload["questions"], "extra": {"type": "noul", "instructions": "Is it spam?"}})
    extended = any_backbone.run([any_backbone.encode(parse_request(more))])[0]
    for f, g in zip(base, extended):
        assert torch.allclose(f.prior, g.prior, atol=ATOL)


def test_batched_requests_match_single(any_backbone, payload):
    other = {"state": "short state", "questions": {"q": {"type": "noul", "instructions": "Is it short?"}}}
    reqs = [parse_request(payload), parse_request(other)]
    batched = any_backbone.run([any_backbone.encode(r) for r in reqs])
    for r, feats in zip(reqs, batched):
        single = any_backbone.run([any_backbone.encode(r)])[0]
        for f, g in zip(feats, single):
            assert torch.allclose(f.prior, g.prior, atol=ATOL)


def test_cached_state_matches_and_is_restored(any_backbone, payload):
    req = parse_request(payload)
    enc = any_backbone.encode(req)
    full = any_backbone.run([enc], keep_memory=True)[0]
    cache = any_backbone.prefill(enc.prefix_ids, keep_memory=True)
    for _ in range(2):  # reuse the same cache twice
        cached = any_backbone.run_branches(cache, enc.questions, keep_memory=True)
        assert cache.kv.get_seq_length() == cache.length  # the shared state cache is never modified
        for f, g in zip(full, cached):
            assert torch.allclose(f.prior, g.prior, atol=ATOL)
            assert torch.allclose(f.branch_hidden, g.branch_hidden, atol=ATOL)
        assert torch.allclose(full[0].state_hidden, cached[0].state_hidden, atol=ATOL)
    one = any_backbone.run_branches(cache, enc.questions[1:2])
    assert torch.allclose(one[0].prior, full[1].prior, atol=ATOL)


def test_branch_packing_positions(tiny_backbone, payload):
    enc = tiny_backbone.encode(parse_request(payload))
    S = len(enc.prefix_ids)
    p = pack_branches(enc.questions, S)
    assert p.attention_mask.shape[-1] == S + p.input_ids.shape[1]
    for a, b in p.branch_offsets[0]:
        assert p.position_ids[0, a:b].tolist() == list(range(S, S + b - a))


def test_option_spans_cover_option_lines(tiny_backbone, payload):
    enc = tiny_backbone.encode(parse_request(payload))
    tok = tiny_backbone.tokenizer
    dept = enc.questions[0]
    lines = [tok.decode(dept.ids[a:b]) for a, b in dept.option_spans]
    assert lines == ["A. billing: invoices, refunds\n", "B. technical: bugs, outages\n", "C. sales: upgrades\n"]
    assert tok.decode(dept.ids[-3:]).endswith("assistant\n")


def test_large_answer_space_gets_single_token_codes(any_backbone):
    crit = {f"intent_{i}": "" for i in range(120)}
    req = parse_request({"state": "s", "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": crit}}})
    q = any_backbone.encode(req).questions[0]
    assert q.has_prior and len(q.label_token_ids) == 120
    assert all(len(ids) >= 1 for ids in q.label_token_ids)
    feats = any_backbone.run([any_backbone.encode(req)])[0][0]
    assert feats.prior.shape == (120,)


def test_hybrid_uses_forked_branches(tiny_backbone, tiny_hybrid_backbone):
    assert tiny_backbone.tree_packing and not tiny_hybrid_backbone.tree_packing
