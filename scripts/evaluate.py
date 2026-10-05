"""Accuracy / calibration of every variant on in-domain and held-out tasks.

Variants (all prefill-only, same request format):
  zero-shot <student>        frozen student, verbalizer readout, no training
  zero-shot <student> +T     same, with per-type temperatures fitted on dev
  jex <student> + head       frozen student + trained head (+ dev temperatures)
  teacher zero-shot <t> +T   frozen teacher, verbalizer readout + dev temperatures

    python scripts/evaluate.py --feats artifacts/feats --heads artifacts/jex-head artifacts/ablation-*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jex.calibration import summarize
from jex.data import HELDOUT_TASKS
from jex.model import JexModel
from jex.training import Item, fit_type_temperatures, head_logits, load_split


class _NoBackbone:
    name = "cached"


def _metrics(items: list[Item], logits: list[torch.Tensor], temps: dict[str, float]) -> dict[str, dict]:
    by_task: dict[str, tuple[list, list]] = {}
    for it, lg in zip(items, logits):
        if it.gold is None:
            continue
        p = torch.softmax(lg.float() / temps.get(it.type, 1.0), -1).tolist()
        by_task.setdefault(it.task, ([], []))
        by_task[it.task][0].append(p)
        by_task[it.task][1].append(it.gold)
    out = {task: summarize(*v) for task, v in sorted(by_task.items())}
    for group, pred in (("in-domain", lambda t: t not in HELDOUT_TASKS), ("held-out", lambda t: t in HELDOUT_TASKS)):
        rows = [m for t, m in out.items() if pred(t) and not t.startswith("avg")]
        if rows:
            out[f"avg {group}"] = {k: sum(r[k] for r in rows) / len(rows) for k in ("accuracy", "ece", "brier", "nll")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="artifacts/feats")
    ap.add_argument("--heads", nargs="*", default=["artifacts/jex-head"])
    ap.add_argument("--out", default="artifacts/eval.json")
    args = ap.parse_args()

    dev = load_split(args.feats, "dev")
    ev = load_split(args.feats, "eval")
    meta_path = Path(args.feats, "meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    student = meta.get("student", "student").split("/")[-1] + (" + LoRA" if meta.get("student_adapter") else "")
    teacher = (meta.get("teacher") or "teacher").split("/")[-1]
    results = {}

    def _or_flat(x, n):  # no single-token prior (>26 options): uniform
        return x.float() if x is not None else torch.zeros(n)

    zs = lambda items: [_or_flat(it.feats.prior, it.n) for it in items]  # noqa: E731
    results[f"zero-shot {student}"] = _metrics(ev, zs(ev), {})
    results[f"zero-shot {student} +T"] = _metrics(ev, zs(ev), fit_type_temperatures(zs(dev), dev))
    tz = lambda items: [_or_flat(it.teacher_logits, it.n) for it in items]  # noqa: E731
    if any(it.teacher_logits is not None for it in ev):
        results[f"teacher zero-shot {teacher} +T"] = _metrics(ev, tz(ev), fit_type_temperatures(tz(dev), dev))
    for path in args.heads:
        model = JexModel.load(path, backbone=_NoBackbone())
        label = f"jex {student} + head [{Path(path).name}]"
        results[label] = _metrics(ev, head_logits(model.head, ev), model.temperatures)

    Path(args.out).write_text(json.dumps(results, indent=2))
    tasks = list(next(iter(results.values())))
    print("| variant | " + " | ".join(tasks) + " |")
    print("|---" * (len(tasks) + 1) + "|")
    for name, res in results.items():
        cells = [f"{res[t]['accuracy']:.3f} / {res[t]['ece']:.3f}" for t in tasks]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\ncells: accuracy / ECE")


if __name__ == "__main__":
    main()
