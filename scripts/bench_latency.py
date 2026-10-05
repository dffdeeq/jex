"""Latency of answering K questions about one state, same 0.5B backbone:

  jex: one pass         prefix + K isolated branches in a single forward pass
  jex: cached state     state already prefilled, only question tokens computed
  jex: K separate       K prefill-only calls, nothing shared
  LLM: JSON generation  autoregressive JSON with all K answers (one call)
  LLM: K generations    K autoregressive calls, one short answer each

    python scripts/bench_latency.py --ks 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import torch

from jex.backbone import Backbone
from jex.model import JexModel
from jex.render import option_labels
from jex.schema import parse_request

STATE = {
    "from": "maria@northwind.io",
    "subject": "Invoice 4411 charged twice + dashboard down",
    "body": (
        "Hello, our card was charged twice for the March invoice (#4411). On top of that the analytics "
        "dashboard has been returning 502 errors since this morning and my team cannot work. We are "
        "evaluating other vendors and will cancel if this is not fixed by Friday. Please call me at "
        "+1 415 555 0199."
    ),
}

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages", "sales": "upgrades"}},
    "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "somewhat urgent", "urgent", "critical"]},
    "has_phone": {"type": "noul", "instructions": "Does the message contain a phone number?"},
    "sentiment": {"type": "choice", "instructions": "What is the customer's sentiment?",
                  "criteria": {"positive": "", "neutral": "", "negative": ""}},
    "refund": {"type": "noul", "instructions": "Is the customer asking for money back?"},
    "outage": {"type": "noul", "instructions": "Does the message report a service outage?"},
    "language": {"type": "choice", "instructions": "Which language is the message written in?",
                 "criteria": {"english": "", "german": "", "spanish": "", "french": ""}},
    "politeness": {"type": "score", "instructions": "How polite is the message?",
                   "criteria": ["rude", "neutral", "polite"]},
    "deadline": {"type": "noul", "instructions": "Does the customer mention a deadline?"},
    "channel": {"type": "choice", "instructions": "How does the customer want to be contacted?",
                "criteria": {"email": "", "phone": "", "chat": "", "unspecified": ""}},
    "spam": {"type": "noul", "instructions": "Is this message spam?"},
    "multiple_issues": {"type": "noul", "instructions": "Does the message describe more than one problem?"},
    "priority": {"type": "choice", "instructions": "Which ticket priority fits best?",
                 "criteria": {"p1": "service down for many users", "p2": "major issue", "p3": "minor issue"}},
    "competitor": {"type": "noul", "instructions": "Does the customer mention evaluating other vendors?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["calm", "a bit annoyed", "frustrated", "furious"]},
}


def timed(fn, repeats: int) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def ar_json_prompt(bb: Backbone, request) -> list[int]:
    lines = []
    for q in request.questions:
        labels = option_labels(q)
        opts = ", ".join(
            f"{lab}" + (f" ({oid})" if q.type == "choice" else f" ({d})" if q.type == "score" else "")
            for lab, oid, d in zip(labels, q.option_ids, q.option_descriptions)
        )
        lines.append(f'- "{q.name}": {q.instructions} Allowed labels: {opts}')
    user = (
        "STATE:\n" + request.state_text() + "\n\nAnswer every question. Reply with only a JSON object that maps "
        "each question id to one allowed label.\n" + "\n".join(lines)
    )
    msgs = [{"role": "user", "content": user}]
    text = bb.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return bb.tokenizer(text, add_special_tokens=False)["input_ids"]


def ar_json_valid(text: str, request) -> bool:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    for q in request.questions:
        if str(obj.get(q.name)) not in option_labels(q):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="artifacts/bench_latency.json")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    bb = Backbone(args.backbone)
    model = JexModel(bb)
    rows = []
    for k in args.ks:
        names = list(QUESTIONS)[:k]
        payload = {"state": STATE, "questions": {n: QUESTIONS[n] for n in names}}
        req = parse_request(payload)
        enc = bb.encode(req)
        cache = model.prefill(STATE)
        row = {"k": k, "tokens": enc.num_tokens}
        row["jex_one_pass"] = timed(lambda: model.predict(payload), args.repeats)
        row["jex_cached_state"] = timed(lambda: model.answer_cached(cache, list(req.questions)), args.repeats)
        singles = [{"state": STATE, "questions": {n: QUESTIONS[n]}} for n in names]
        row["jex_k_separate"] = timed(lambda: [model.predict(p) for p in singles], 1)

        prompt = ar_json_prompt(bb, req)
        out = {}

        def gen():
            out["ids"] = bb.generate_text(prompt, max_new_tokens=14 * k + 16)

        row["llm_json_generation"] = timed(gen, 1)
        text = bb.tokenizer.decode(out["ids"], skip_special_tokens=True)
        row["llm_json_new_tokens"] = len(out["ids"])
        row["llm_json_valid"] = ar_json_valid(text, req)

        def gen_each():
            for q in enc.questions:
                bb.generate_text(enc.prefix_ids + q.ids, max_new_tokens=4)

        row["llm_k_generations"] = timed(gen_each, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print("\n| K | tokens | jex one pass | jex cached state | jex K separate | LLM JSON gen | LLM K gens | JSON valid |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['k']} | {r['tokens']} | {r['jex_one_pass'] * 1000:.0f} ms | {r['jex_cached_state'] * 1000:.0f} ms | "
              f"{r['jex_k_separate'] * 1000:.0f} ms | {r['llm_json_generation'] * 1000:.0f} ms | "
              f"{r['llm_k_generations'] * 1000:.0f} ms | {r['llm_json_valid']} |")


if __name__ == "__main__":
    main()
