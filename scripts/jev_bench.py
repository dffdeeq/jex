"""Run jex on the public Jev benchmark (arXiv 2609.37647) for a head-to-head comparison.

The harness (github.com/AppliedMachineLearning-Lab/jev-benchmarking) builds the exact Jev requests for 37
datasets and ships Jev's own responses on Zenodo. This script answers the same requests with a jex model
(our template, our readout, one prefill-only pass per request) and writes JSONL in the harness format;
the harness then scores jex and Jev on identical requests with the same metrics.

    git clone https://github.com/AppliedMachineLearning-Lab/jev-benchmarking && pip install -e jev-benchmarking
    # Jev's responses: https://doi.org/10.5281/zenodo.23039006 -> jev-benchmarking/cache/responses.db
    python scripts/jev_bench.py --checkpoint artifacts/lora --tag jex-0.5b-lora --limit 30 --out runs/jex-0.5b-lora
    cd jev-benchmarking
    python scripts/import_responses.py ../runs/jex-0.5b-lora/*.jsonl
    python scripts/evaluate.py all --split eval --limit 30 --model hf:jex/jex-0.5b-lora --no-ci
    python scripts/evaluate.py all --split eval --limit 30 --no-ci        # Jev on the same sample
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from jex.backbone import Backbone
from jex.model import JexModel
from jex.schema import SchemaError


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*", default=["all"])
    ap.add_argument("--checkpoint", default=None, help="jex checkpoint (LoRA and/or head); default: zero-shot")
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B-Instruct", help="used without --checkpoint")
    ap.add_argument("--tag", required=True, help="model name; responses are stored as hf:jex/<tag>")
    ap.add_argument("--split", choices=["eval", "dev"], default="eval")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-state-tokens", type=int, default=3072)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    torch.set_num_threads(args.threads)

    from jev_benchmarking.cache import request_key
    from jev_benchmarking.tasks import get_tasks

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    kw = dict(device=args.device, dtype=dtype, max_state_tokens=args.max_state_tokens)
    model = JexModel.load(args.checkpoint, **kw) if args.checkpoint else JexModel(Backbone(args.backbone, **kw))
    tag = f"hf:jex/{args.tag}"

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "responses.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["key"] for line in out_path.read_text().splitlines() if line.strip()}

    with out_path.open("a") as out:
        for task in get_tasks(args.tasks):
            if task.splits.get(args.split) is None:
                continue
            try:
                examples = task.examples(args.split, limit=args.limit)
            except Exception as e:  # e.g. gated datasets
                logging.warning("%s: could not load (%s)", task.name, type(e).__name__)
                continue
            t_task, n = time.time(), 0
            for e in examples:
                key = request_key(tag, e.state, e.questions)
                if key in done:
                    continue
                t0 = time.perf_counter()
                try:
                    resp = model.predict({"state": e.state, "questions": e.questions})
                except SchemaError as err:
                    logging.warning("%s/%s: rejected (%s)", task.name, e.uid, err)
                    continue
                resp["model"] = tag
                out.write(json.dumps({"key": key, "task": task.name, "uid": e.uid,
                                      "latency_s": time.perf_counter() - t0, "response": resp},
                                     ensure_ascii=False) + "\n")
                out.flush()
                done.add(key)
                n += 1
            logging.info("%s: %d requests in %.0fs", task.name, n, time.time() - t_task)


if __name__ == "__main__":
    main()
