"""Calibration metrics and per-type temperature scaling."""

from __future__ import annotations

import math

import torch


def ece(confidences: list[float], correct: list[bool], bins: int = 15) -> float:
    """Expected calibration error of top-1 confidence, equal-width bins."""
    if not confidences:
        return float("nan")
    total, err = len(confidences), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confidences[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(acc - conf)
    return err


def brier(probs: list[list[float]], labels: list[int]) -> float:
    return sum(
        sum((p - (1.0 if k == y else 0.0)) ** 2 for k, p in enumerate(ps)) for ps, y in zip(probs, labels)
    ) / max(len(labels), 1)


def nll(probs: list[list[float]], labels: list[int]) -> float:
    return -sum(math.log(max(ps[y], 1e-12)) for ps, y in zip(probs, labels)) / max(len(labels), 1)


def summarize(probs: list[list[float]], labels: list[int]) -> dict[str, float]:
    top = [max(range(len(p)), key=p.__getitem__) for p in probs]
    correct = [t == y for t, y in zip(top, labels)]
    return {
        "n": len(labels),
        "accuracy": sum(correct) / max(len(correct), 1),
        "ece": ece([max(p) for p in probs], correct),
        "brier": brier(probs, labels),
        "nll": nll(probs, labels),
    }


def fit_temperature(logits: list[torch.Tensor], labels: list[int], steps: int = 200) -> float:
    """Single temperature minimizing NLL on held-out logits (variable option counts)."""
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps)
    ys = torch.tensor(labels)

    def closure():
        opt.zero_grad()
        t = log_t.exp()
        loss = torch.stack(
            [-torch.log_softmax(lg / t, dim=-1)[y] for lg, y in zip(logits, ys)]
        ).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))
