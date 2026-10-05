"""Side-by-side table of primary metrics from the jev-benchmarking harness results.

    python scripts/compare_jev.py --harness ../jev-benchmarking \
        --model Jev=jev-1.13.0 --model "Qwen3.8-27B=hf:Qwen/Qwen3.8-27B" --model "jex 0.5B=hf:jex/jex-0.5b-zs"

Each model's results/<split>/ must come from evaluate.py with the same --limit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Datasets whose training split was used to train jex (v1 data): in-domain for the LoRA/head variants.
JEX_TRAINED_ON = {"sst2", "ag_news", "emotion", "sst5"}


def results_dir(harness: Path, model: str, split: str) -> Path:
    if model.startswith("jev-"):
        return harness / "results" / split
    return harness / "results" / split / "open_models" / model.replace("hf:", "").replace("/", "__")


def load(harness: Path, model: str, split: str) -> dict[str, tuple[str, float]]:
    out = {}
    for f in sorted(results_dir(harness, model, split).glob("*.json")):
        res = json.loads(f.read_text())
        prim = res.get("primary") or {}
        if prim.get("value") is not None:
            out[f.stem] = (prim["metric"], float(prim["value"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="label=model_tag")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    labels, tables = [], []
    for spec in args.model:
        label, tag = spec.split("=", 1)
        labels.append(label)
        tables.append(load(args.harness, tag, args.split))
    tasks = [t for t in tables[0] if all(t in tb for tb in tables)]

    lines = ["| dataset | metric | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for t in tasks:
        vals = [tb[t][1] for tb in tables]
        best = max(vals)
        cells = [f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in vals]
        mark = " †" if t in JEX_TRAINED_ON else ""
        lines.append(f"| {t}{mark} | {tables[0][t][0]} | " + " | ".join(cells) + " |")
    for name, subset in (("mean, all", tasks), ("mean, w/o jex-trained †", [t for t in tasks if t not in JEX_TRAINED_ON])):
        means = [sum(tb[t][1] for t in subset) / len(subset) for tb in tables]
        lines.append(f"| **{name}** ({len(subset)}) | | " + " | ".join(f"**{m:.3f}**" for m in means) + " |")
    ref = tables[0]
    for label, tb in zip(labels[1:], tables[1:]):
        wins = sum(tb[t][1] > ref[t][1] for t in tasks)
        ties = sum(tb[t][1] == ref[t][1] for t in tasks)
        lines.append(f"\n{label} vs {labels[0]}: better on {wins}, equal on {ties}, worse on {len(tasks) - wins - ties} of {len(tasks)}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
