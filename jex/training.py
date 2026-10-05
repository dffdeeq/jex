"""Head training on cached backbone features: distillation from a bigger
teacher + gold labels + Brier calibration loss + RLCD fine-tuning."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .backbone import QuestionFeatures
from .calibration import fit_temperature, summarize
from .head import DecisionHead, HeadConfig, collate
from .rlcd import rlcd_loss


@dataclass
class Item:
    task: str
    type: str
    n: int
    feats: QuestionFeatures
    gold: int | None
    teacher_logits: torch.Tensor | None


def load_split(feats_dir: str | Path, split: str, with_teacher: bool = True) -> list[Item]:
    feats_dir = Path(feats_dir)
    records = torch.load(feats_dir / "records.pt", weights_only=False)[split]
    rows = torch.load(feats_dir / f"student_{split}.pt", weights_only=False)
    teacher_path = feats_dir / f"teacher_{split}.pt"
    teacher = torch.load(teacher_path, weights_only=False) if with_teacher and teacher_path.exists() else None
    items = []
    for r_idx, (rec, row) in enumerate(zip(records, rows)):
        for q_idx, q in enumerate(row["questions"]):
            f = QuestionFeatures(
                answer_hidden=q["answer_hidden"],
                option_hidden=q["option_hidden"],
                prior=q["prior"],
                state_hidden=row["state_hidden"],
                branch_hidden=q["branch_hidden"],
            )
            t = teacher[r_idx][q_idx] if teacher is not None else None
            items.append(Item(rec["task"], q["type"], q["n"], f, q["gold"], t))
    return items


def fit_type_temperatures(logits: list[torch.Tensor | None], items: list[Item]) -> dict[str, float]:
    """Per-type temperature on gold questions; entries without logits (e.g. no
    single-token prior for >26 options) are skipped."""
    temps = {}
    for t in ("noul", "choice", "score"):
        idx = [i for i, it in enumerate(items) if it.type == t and it.gold is not None and logits[i] is not None]
        if len(idx) >= 20:
            temps[t] = fit_temperature([logits[i] for i in idx], [items[i].gold for i in idx])
    return temps


def teacher_probs(items: list[Item], temps: dict[str, float]) -> list[torch.Tensor | None]:
    return [
        torch.softmax(it.teacher_logits.float() / temps.get(it.type, 1.0), -1) if it.teacher_logits is not None else None
        for it in items
    ]


def build_targets(
    items: list[Item], t_probs: list[torch.Tensor | None], mix: float, smoothing: float, use_teacher: bool
) -> list[torch.Tensor | None]:
    targets = []
    for it, tp in zip(items, t_probs):
        tp = tp if use_teacher else None
        if it.gold is not None:
            onehot = torch.full((it.n,), smoothing / it.n)
            onehot[it.gold] += 1 - smoothing
            targets.append(onehot if tp is None else (1 - mix) * onehot + mix * tp)
        else:
            targets.append(tp)
    return targets


@dataclass
class TrainConfig:
    epochs: int = 8
    rl_epochs: int = 2
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    brier_weight: float = 1.0
    rl_weight: float = 0.5
    teacher_mix: float = 0.3
    label_smoothing: float = 0.05
    use_teacher: bool = True
    # KL(calibrated zero-shot prior || head): keeps the head close to the backbone's
    # own reading, which matters on schemas unlike the training ones.
    prior_kl: float = 0.0
    seed: int = 0
    device: str = "cpu"


def head_logits(head: DecisionHead, items: list[Item], batch_size: int = 64) -> list[torch.Tensor]:
    head.eval()
    device = next(head.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            b = collate([it.feats for it in chunk], [it.type for it in chunk], head.cfg.use_memory, head.cfg.max_memory)
            lg = head(b.to(device)).cpu()
            out += [lg[j, : it.n] for j, it in enumerate(chunk)]
    return out


def gold_metrics(logits: list[torch.Tensor], items: list[Item], temps: dict[str, float] | None = None):
    temps = temps or {}
    idx = [i for i, it in enumerate(items) if it.gold is not None]
    probs = [torch.softmax(logits[i] / temps.get(items[i].type, 1.0), -1).tolist() for i in idx]
    return summarize(probs, [items[i].gold for i in idx])


def train_head(
    train: list[Item], dev: list[Item], head_cfg: HeadConfig, cfg: TrainConfig, log=print
) -> tuple[DecisionHead, dict]:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    # The teacher is itself calibrated (temperature on gold labels) before it teaches.
    t_temps = fit_type_temperatures([it.teacher_logits for it in train], train) if cfg.use_teacher else {}
    targets = build_targets(train, teacher_probs(train, t_temps), cfg.teacher_mix, cfg.label_smoothing, cfg.use_teacher)
    pool = [(it, tg) for it, tg in zip(train, targets) if tg is not None]
    log(f"training on {len(pool)} questions; teacher temperatures {t_temps}")
    zs_temps = fit_type_temperatures([it.feats.prior for it in train], train) if cfg.prior_kl else {}

    device = torch.device(cfg.device)
    head = DecisionHead(head_cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * ((len(pool) + cfg.batch_size - 1) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.1)
    best, best_state, history = float("inf"), None, []

    for epoch in range(cfg.epochs):
        head.train()
        rng.shuffle(pool)
        use_rl = epoch >= cfg.epochs - cfg.rl_epochs
        total = 0.0
        for i in range(0, len(pool), cfg.batch_size):
            chunk = pool[i : i + cfg.batch_size]
            b = collate([it.feats for it, _ in chunk], [it.type for it, _ in chunk], head_cfg.use_memory, head_cfg.max_memory)
            b = b.to(device)
            logits = head(b)
            q = torch.zeros_like(logits)
            for j, (_, tg) in enumerate(chunk):
                q[j, : tg.shape[0]] = tg.to(device)
            mask = b.option_mask
            logp = torch.log_softmax(logits, -1).masked_fill(~mask, 0.0)
            p = logp.exp() * mask
            loss = -(q * logp).sum(-1).mean() + cfg.brier_weight * ((p - q) ** 2).sum(-1).mean()
            if cfg.prior_kl:
                t = torch.tensor([zs_temps.get(it.type, 1.0) for it, _ in chunk], device=device)[:, None]
                ref = torch.softmax((b.prior / t).masked_fill(~mask, float("-inf")), -1)
                kl = (ref * (torch.log(ref.clamp_min(1e-9)) - logp)).masked_fill(~mask, 0.0).sum(-1)
                loss = loss + cfg.prior_kl * (kl * b.has_prior).mean()
            if use_rl:
                is_score = torch.tensor([it.type == "score" for it, _ in chunk], device=device)
                loss = loss + cfg.rl_weight * rlcd_loss(logits, q, mask, is_score)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(chunk)
        dev_m = gold_metrics(head_logits(head, dev), dev)
        history.append({"epoch": epoch, "train_loss": total / len(pool), **{f"dev_{k}": v for k, v in dev_m.items()}})
        log(f"epoch {epoch}{' (rlcd)' if use_rl else ''}: loss {total / len(pool):.4f} "
            f"dev acc {dev_m['accuracy']:.3f} nll {dev_m['nll']:.3f} ece {dev_m['ece']:.3f}")
        if dev_m["nll"] < best:
            best, best_state = dev_m["nll"], {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    head = head.cpu().eval()
    temps = fit_type_temperatures(head_logits(head, dev), dev)
    return head, {"history": history, "temperatures": temps, "teacher_temperatures": t_temps, "zero_shot_temperatures": zs_temps}
