"""Build typed-decision datasets, run the frozen student backbone once per
record (features for head training / evaluation) and the bigger teacher once
per record (soft labels for distillation).

    python scripts/extract.py --out artifacts/feats --student Qwen/Qwen2.5-0.5B-Instruct \
        --teacher Qwen/Qwen2.5-1.5B-Instruct --train-per-task 250

Both models answer through the same prefill-only, branch-packed path, so the
teacher is itself a (bigger, slower) System One model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from jex.backbone import Backbone
from jex.data import HELDOUT_TASKS, TRAIN_TASKS, build
from jex.schema import parse_request


def splits(args) -> dict[str, list]:
    out = {"train": [], "dev": [], "eval": []}
    for task in TRAIN_TASKS:
        out["train"] += build(task, "train", args.train_per_task, seed=0, aux_max=2)
        # dev is carved from train right after the training records: no overlap.
        out["dev"] += build(task, "train", args.dev_per_task, seed=0, aux_max=1, offset=args.train_per_task)
        out["eval"] += build(task, "test", args.eval_per_task, seed=2)
    for task in HELDOUT_TASKS:
        out["eval"] += build(task, "test", args.heldout_per_task, seed=2)
    return out


def featurize(bb: Backbone, records, keep_memory: bool, log_every: int = 100):
    rows, t0 = [], time.time()
    for i, rec in enumerate(records):
        req = parse_request(rec.payload())
        enc = bb.encode(req)
        feats = bb.run([enc], keep_memory=keep_memory)[0]
        row = {"questions": [], "num_tokens": enc.num_tokens}
        if keep_memory:
            row["state_hidden"] = feats[0].state_hidden.half()
        for q, f in zip(req.questions, feats):
            item = {
                "name": q.name,
                "type": q.type,
                "n": q.num_options,
                "prior": f.prior,
                "gold": rec.labels.get(q.name),
            }
            if keep_memory:
                item.update(
                    answer_hidden=f.answer_hidden.half(),
                    option_hidden=f.option_hidden.half(),
                    branch_hidden=f.branch_hidden.half(),
                )
            row["questions"].append(item)
        rows.append(row)
        if (i + 1) % log_every == 0:
            print(f"  {bb.name}: {i + 1}/{len(records)} ({(time.time() - t0) / (i + 1):.2f}s/record)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/feats")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train-per-task", type=int, default=250)
    ap.add_argument("--dev-per-task", type=int, default=40)
    ap.add_argument("--eval-per-task", type=int, default=100)
    ap.add_argument("--heldout-per-task", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = splits(args)
    for name, recs in data.items():
        print(f"{name}: {len(recs)} records, {sum(len(r.questions) for r in recs)} questions", flush=True)
    torch.save({k: [r.__dict__ for r in v] for k, v in data.items()}, out / "records.pt")

    student = Backbone(args.student)
    for name, recs in data.items():
        path = out / f"student_{name}.pt"
        if not path.exists():
            torch.save(featurize(student, recs, keep_memory=True), path)
    del student

    teacher = Backbone(args.teacher)
    for name in ("train", "dev", "eval"):
        path = out / f"teacher_{name}.pt"
        if not path.exists():
            rows = featurize(teacher, data[name], keep_memory=False)
            torch.save([[q["prior"] for q in r["questions"]] for r in rows], path)
    print("done", flush=True)


if __name__ == "__main__":
    main()
