"""LoRA fine-tuning of the backbone itself (the Clef recipe, minus the head).

The readout stays the backbone's own next-token distribution over option
labels at each answer slot, so the model keeps reading the question with its
own weights instead of an external head learning task shortcuts. Training uses
the same packed tree as inference: one forward pass per record covers all of
its questions, and the loss is soft cross-entropy (+ Brier) against gold
labels mixed with the calibrated teacher.
"""

from __future__ import annotations

import random
import time

import torch

from .backbone import Backbone
from .packing import pack_requests
from .schema import parse_request

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def add_lora(bb: Backbone, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS, bias="none")
    # get_peft_model swaps the Linear layers in place, so bb.base / bb.lm_head now run through LoRA.
    peft_model = get_peft_model(bb.lm, cfg)
    return peft_model


def answer_label_logits(bb: Backbone, encoded) -> list[list[torch.Tensor | None]]:
    """Differentiable version of the verbalizer readout for a batch of records."""
    packed = pack_requests(encoded, bb.pad_id, bb.dtype)
    hidden = bb.base(
        input_ids=packed.input_ids.to(bb.device),
        position_ids=packed.position_ids.to(bb.device),
        attention_mask=packed.attention_mask.to(bb.device, bb.dtype),
    ).last_hidden_state
    out = []
    for b, enc in enumerate(encoded):
        row = []
        for q, (start, _) in zip(enc.questions, packed.branch_offsets[b]):
            if not q.has_prior:
                row.append(None)
                continue
            logits = bb.lm_head(hidden[b, start + q.answer_index][None]).float()[0]
            row.append(torch.stack([torch.logsumexp(logits[ids], 0) for ids in q.label_token_ids]))
        out.append(row)
    return out


def train_lora(
    bb: Backbone,
    records: list[dict],
    targets: list[list[torch.Tensor | None]],
    epochs: int = 1,
    lr: float = 2e-4,
    batch_records: int = 4,
    brier_weight: float = 1.0,
    seed: int = 0,
    log=print,
):
    """records: Record dicts (state/questions); targets[i][j]: distribution for question j or None."""
    peft_model = add_lora(bb)
    peft_model.train()
    params = [p for p in peft_model.parameters() if p.requires_grad]
    log(f"LoRA trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    order = list(range(len(records)))
    steps = epochs * ((len(order) + batch_records - 1) // batch_records)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    rng = random.Random(seed)
    t0, step = time.time(), 0
    for epoch in range(epochs):
        rng.shuffle(order)
        for i in range(0, len(order), batch_records):
            idx = order[i : i + batch_records]
            encoded = [bb.encode(parse_request({"state": records[k]["state"], "questions": records[k]["questions"]}))
                       for k in idx]
            logits = answer_label_logits(bb, encoded)
            losses = []
            for k, row in zip(idx, logits):
                for lg, tg in zip(row, targets[k]):
                    if lg is None or tg is None:
                        continue
                    logp = torch.log_softmax(lg, -1)
                    tg = tg.to(lg.device)
                    losses.append(-(tg * logp).sum() + brier_weight * ((logp.exp() - tg) ** 2).sum())
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0:
                log(f"  step {step}/{steps} loss {loss.item():.4f} ({(time.time() - t0) / step:.2f}s/step)")
    peft_model.eval()
    return peft_model


@torch.no_grad()
def calibrate(bb: Backbone, records: list[dict], batch_records: int = 4) -> dict[str, float]:
    """Per-type temperatures of the tuned readout, fitted on gold questions of ``records``."""
    from types import SimpleNamespace

    from .training import fit_type_temperatures

    logits, items = [], []
    for i in range(0, len(records), batch_records):
        chunk = records[i : i + batch_records]
        reqs = [parse_request({"state": r["state"], "questions": r["questions"]}) for r in chunk]
        for r, req, row in zip(chunk, reqs, answer_label_logits(bb, [bb.encode(q) for q in reqs])):
            for q, lg in zip(req.questions, row):
                logits.append(lg.cpu() if lg is not None else None)
                items.append(SimpleNamespace(type=q.type, gold=r["labels"].get(q.name)))
    return fit_type_temperatures(logits, items)
