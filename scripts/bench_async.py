"""Open-loop load test: single questions about a handful of states arrive at
random (Poisson) times; every question is awaited on its own.

  jex engine   state KV cache + continuous micro-batching of isolated branches
  naive        same model, one question per pass, state re-encoded every time

    python scripts/bench_async.py --rates 2 5 10 --n 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import torch

from jex.backbone import Backbone
from jex.engine import AsyncEngine
from jex.model import JexModel

from bench_latency import QUESTIONS, STATE  # noqa: E402  (scripts/ is on sys.path)

STATES = [
    STATE,
    {"subject": "Love the new release", "body": "Just wanted to say the new export feature saves us hours. Thanks!"},
    {"subject": "Upgrade to enterprise?", "body": "We are 40 people now. What would the enterprise plan cost per seat?"},
    {"subject": "Password reset loop", "body": "Every time I reset my password the login page sends me back to reset."},
]


async def run_load(engine: AsyncEngine, rate: float, n: int, seed: int) -> dict:
    rng = random.Random(seed)
    names = list(QUESTIONS)
    latencies: list[float] = []

    async def one(state, name):
        t = time.perf_counter()
        await engine.ask(state, name, QUESTIONS[name])
        latencies.append(time.perf_counter() - t)

    tasks = []
    start = time.perf_counter()
    for _ in range(n):
        await asyncio.sleep(rng.expovariate(rate))
        tasks.append(asyncio.create_task(one(rng.choice(STATES), rng.choice(names))))
    await asyncio.gather(*tasks)
    makespan = time.perf_counter() - start
    lat = sorted(latencies)
    return {
        "mean_ms": 1000 * statistics.mean(lat),
        "p50_ms": 1000 * lat[len(lat) // 2],
        "p95_ms": 1000 * lat[int(0.95 * (len(lat) - 1))],
        "throughput_qps": n / makespan,
        "passes": engine.stats.passes,
        "mean_batch": statistics.mean(engine.stats.batch_sizes),
        "prefills": engine.stats.prefills,
    }


async def main_async(args):
    model = JexModel(Backbone(args.backbone))
    model.predict({"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
    rows = []
    for rate in args.rates:
        for label, kwargs in (
            ("jex engine", dict(max_batch_questions=64, max_wait_ms=2.0, cache_size=64)),
            ("naive", dict(max_batch_questions=1, max_wait_ms=0.0, cache_size=0)),
        ):
            async with AsyncEngine(model, **kwargs) as engine:
                res = await run_load(engine, rate, args.n, seed=int(rate * 100))
            row = {"rate_qps": rate, "server": label, **res}
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print("\n| arrival rate | server | mean | p50 | p95 | throughput | mean batch | prefills |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['rate_qps']} q/s | {r['server']} | {r['mean_ms']:.0f} ms | {r['p50_ms']:.0f} ms | "
              f"{r['p95_ms']:.0f} ms | {r['throughput_qps']:.1f} q/s | {r['mean_batch']:.1f} | {r['prefills']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rates", type=float, nargs="*", default=[2, 5, 10])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="artifacts/bench_async.json")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
