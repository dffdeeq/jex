import torch

from jex.calibration import ece, fit_temperature, summarize
from jex.head import DecisionHead, HeadConfig, collate
from jex.rlcd import log_score, proper_reward, rlcd_loss, spherical_score
from jex.schema import parse_request


def _feats(tiny_backbone, payload):
    req = parse_request(payload)
    feats = tiny_backbone.run([tiny_backbone.encode(req)], keep_memory=True)[0]
    return req, feats


def test_head_shapes_and_padding(tiny_backbone, payload):
    req, feats = _feats(tiny_backbone, payload)
    head = DecisionHead(HeadConfig(d_backbone=tiny_backbone.hidden_size, d_model=32, n_heads=2)).eval()
    logits = head(collate(feats, [q.type for q in req.questions]))
    assert logits.shape == (3, 4)
    for i, q in enumerate(req.questions):
        assert torch.isfinite(logits[i, : q.num_options]).all()
        assert torch.isinf(logits[i, q.num_options :]).all()


def test_untrained_head_stays_close_to_prior(tiny_backbone, payload):
    req, feats = _feats(tiny_backbone, payload)
    torch.manual_seed(0)
    head = DecisionHead(HeadConfig(d_backbone=tiny_backbone.hidden_size, d_model=32, n_heads=2)).eval()
    logits = head(collate(feats, [q.type for q in req.questions]))
    for i, (q, f) in enumerate(zip(req.questions, feats)):
        p_head = torch.softmax(logits[i, : q.num_options], -1)
        p_prior = torch.softmax(f.prior, -1)
        assert (p_head - p_prior).abs().max() < 0.15


def test_head_is_permutation_equivariant_for_choice(tiny_backbone, payload):
    req, feats = _feats(tiny_backbone, payload)
    head = DecisionHead(HeadConfig(d_backbone=tiny_backbone.hidden_size, d_model=32, n_heads=2)).eval()
    f = feats[0]
    perm = torch.tensor([2, 0, 1])
    g = type(f)(f.answer_hidden, f.option_hidden[perm], f.prior[perm], f.state_hidden, f.branch_hidden)
    a = head(collate([f], ["choice"]))[0]
    b = head(collate([g], ["choice"]))[0]
    assert torch.allclose(a[perm], b, atol=1e-5)


def test_scoring_rules_are_proper():
    q = torch.tensor([0.6, 0.3, 0.1])
    best = proper_reward(q, q, ordinal=True)
    for _ in range(50):
        p = torch.softmax(torch.log(q) + 0.5 * torch.randn(3), -1)
        assert proper_reward(p, q, ordinal=True) <= best + 1e-6
        assert log_score(p, q) <= log_score(q, q) + 1e-6
        assert spherical_score(p, q) <= spherical_score(q, q) + 1e-6


def test_rlcd_recovers_true_probabilities():
    torch.manual_seed(0)
    target = torch.tensor([[0.7, 0.2, 0.1]])
    logits = torch.zeros(1, 3, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.05)
    mask = torch.ones(1, 3, dtype=torch.bool)
    for _ in range(600):
        opt.zero_grad()
        rlcd_loss(logits, target, mask, torch.tensor([False]), group=32, sigma=0.3).backward()
        opt.step()
    p = torch.softmax(logits.detach(), -1)[0]
    assert (p - target[0]).abs().max() < 0.08, p


def test_calibration_metrics():
    probs = [[0.8, 0.2]] * 10
    labels = [0] * 8 + [1] * 2
    assert ece([0.8] * 10, [True] * 8 + [False] * 2) < 1e-9
    s = summarize(probs, labels)
    assert abs(s["accuracy"] - 0.8) < 1e-9 and s["ece"] < 1e-9
    # logits scaled by 3 are over-confident: the fitted temperature undoes it
    torch.manual_seed(0)
    true_logits = [torch.randn(4) for _ in range(400)]
    labels = [int(torch.multinomial(torch.softmax(l, -1), 1)) for l in true_logits]
    t = fit_temperature([3 * l for l in true_logits], labels)
    assert 2.2 < t < 4.0
