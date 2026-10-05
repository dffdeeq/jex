"""RLCD - reinforcement learning for calibrated decisions.

Reconstructed from the public descriptions (TypeSafe's Jev blog, Convai's Laya
model card): the policy *reports a distribution*; exploration adds zero-mean
Gaussian noise to the logits; the reward is a strictly proper scoring rule
(log + spherical, plus ranked probability score for ordinal questions); updates
are REINFORCE with a group-mean baseline (GRPO-style). Because the scoring rule
is strictly proper, expected reward is maximized only by reporting the true
outcome probabilities - accuracy cannot be bought by becoming overconfident.

Targets may be soft (a teacher distribution): the reward is then the expected
score under that distribution, which is maximized by matching it.
"""

from __future__ import annotations

import torch


def log_score(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (q * torch.log(p.clamp_min(1e-9))).sum(-1)


def spherical_score(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (q * p).sum(-1) / p.norm(dim=-1).clamp_min(1e-9)


def ranked_probability_score(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Negative RPS (higher is better); rewards mass close to the true level."""
    return -((p.cumsum(-1) - q.cumsum(-1)) ** 2).sum(-1) / max(p.shape[-1] - 1, 1)


def proper_reward(p: torch.Tensor, q: torch.Tensor, ordinal: bool) -> torch.Tensor:
    r = log_score(p, q) + spherical_score(p, q)
    if ordinal:
        r = r + ranked_probability_score(p, q)
    return r


def rlcd_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    option_mask: torch.Tensor,
    ordinal: torch.Tensor,
    group: int = 8,
    sigma: float = 0.5,
) -> torch.Tensor:
    """REINFORCE over a Gaussian policy on logits.

    logits (N, O) with -inf on padded options; target (N, O) one-hot or soft;
    ordinal (N,) bool marks score questions.
    """
    N, O = logits.shape
    safe = logits.masked_fill(~option_mask, 0.0)
    noise = torch.randn(group, N, O, device=logits.device) * sigma
    actions = (safe.detach()[None] + noise).masked_fill(~option_mask[None], float("-inf"))
    p = torch.softmax(actions, dim=-1)
    with torch.no_grad():
        r = proper_reward(p, target[None].expand_as(p), ordinal=False)
        if ordinal.any():
            rps = ranked_probability_score(p, target[None].expand_as(p))
            r = r + rps * ordinal.float()[None]
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (r.std(0, keepdim=True) + 1e-6)
    # log N(action; logits, sigma) up to a constant, summed over real options.
    logp = -(((actions.masked_fill(~option_mask[None], 0.0) - safe[None]) ** 2) * option_mask[None]).sum(-1) / (
        2 * sigma**2
    )
    return -(adv * logp).mean()
