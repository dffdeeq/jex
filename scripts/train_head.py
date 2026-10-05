"""Train the decision head on cached features (see scripts/extract.py).

    python scripts/train_head.py --feats artifacts/feats --out artifacts/jex-head
    python scripts/train_head.py --no-teacher --out artifacts/ablation-gold-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from jex.head import HeadConfig
from jex.model import save_checkpoint
from jex.training import TrainConfig, load_split, train_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="artifacts/feats")
    ap.add_argument("--out", default="artifacts/jex-head")
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--rl-epochs", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    train = load_split(args.feats, "train")
    dev = load_split(args.feats, "dev")
    d_backbone = train[0].feats.answer_hidden.shape[-1]
    head_cfg = HeadConfig(d_backbone=d_backbone, d_model=args.d_model, n_layers=args.layers, use_memory=not args.no_memory)
    cfg = TrainConfig(epochs=args.epochs, rl_epochs=args.rl_epochs, use_teacher=not args.no_teacher)

    t0 = time.time()
    head, info = train_head(train, dev, head_cfg, cfg)
    info["train_seconds"] = time.time() - t0
    info["head_parameters"] = sum(p.numel() for p in head.parameters())
    info["config"] = {**cfg.__dict__, **head_cfg.__dict__}
    save_checkpoint(args.out, "jex-0.1", args.backbone, head, info["temperatures"])
    Path(args.out, "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"saved to {args.out}: {info['head_parameters'] / 1e6:.1f}M head params, "
          f"{info['train_seconds']:.0f}s, temperatures {info['temperatures']}")


if __name__ == "__main__":
    main()
