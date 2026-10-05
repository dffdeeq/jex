"""LoRA-tune the student so its own answer-slot distribution matches gold
labels mixed with the calibrated teacher (see jex/lora.py).

    python scripts/train_lora.py --feats artifacts/feats --out artifacts/lora
    python scripts/extract.py --student-adapter artifacts/lora --records-from artifacts/feats/records.pt \
        --splits dev,eval --skip-teacher --out artifacts/feats-lora
    python scripts/evaluate.py --feats artifacts/feats-lora --heads
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from jex.backbone import Backbone
from jex.lora import train_lora
from jex.training import build_targets, fit_type_temperatures, load_split, teacher_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="artifacts/feats")
    ap.add_argument("--out", default="artifacts/lora")
    ap.add_argument("--student", default=None, help="defaults to the student recorded by extract.py")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-records", type=int, default=4)
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    meta_path = Path(args.feats, "meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    student = args.student or meta.get("student", "Qwen/Qwen2.5-0.5B-Instruct")

    records = torch.load(Path(args.feats, "records.pt"), weights_only=False)["train"]
    items = load_split(args.feats, "train")
    use_teacher = not args.no_teacher
    t_temps = fit_type_temperatures([it.teacher_logits for it in items], items) if use_teacher else {}
    flat = build_targets(items, teacher_probs(items, t_temps), 0.3, 0.05, use_teacher)
    targets, k = [], 0
    for rec in records:  # regroup the flat per-question targets by record
        n = len(rec["questions"])
        targets.append(flat[k : k + n])
        k += n
    assert k == len(flat)
    del items

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    bb = Backbone(student, dtype=dtype, device=args.device)
    t0 = time.time()
    model = train_lora(bb, records, targets, epochs=args.epochs, lr=args.lr, batch_records=args.batch_records)
    model.save_pretrained(args.out)
    info = {"student": student, "teacher_temperatures": t_temps, "seconds": time.time() - t0, **vars(args)}
    Path(args.out, "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"saved LoRA adapter to {args.out} in {info['seconds']:.0f}s")


if __name__ == "__main__":
    main()
