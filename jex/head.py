"""Trainable decision head on top of the frozen backbone (Clef / Laya style).

For every question it builds one *field* token (from the answer-slot hidden
state) and one token per option (mean hidden state of the option line), then

1. routes evidence: field/option tokens cross-attend to the state + question
   hidden states ("memory");
2. lets options of the same question compete through self-attention (no
   positional embeddings, so the head is permutation-equivariant and works for
   any answer space given at request time; score levels get a level embedding);
3. scores each option with an MLP over [f, o, f*o, |f-o|] and adds it, through a
   learned gate, to the backbone's verbalizer logits (lexical prior).

The gate starts almost closed, so an untrained head reproduces the zero-shot
verbalizer and training can only move away from it when it helps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .backbone import QuestionFeatures
from .schema import MAX_SCORE_LEVELS, QUESTION_TYPES

TYPE_IDS = {t: i for i, t in enumerate(QUESTION_TYPES)}


@dataclass
class HeadConfig:
    d_backbone: int
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    use_memory: bool = True
    max_memory: int = 1024  # most recent memory tokens kept per question


@dataclass
class HeadBatch:
    answer: torch.Tensor  # (N, D)
    options: torch.Tensor  # (N, O, D)
    option_mask: torch.Tensor  # (N, O) bool, True = real option
    prior: torch.Tensor  # (N, O)
    has_prior: torch.Tensor  # (N,) bool
    type_ids: torch.Tensor  # (N,)
    memory: torch.Tensor | None  # (N, M, D)
    memory_mask: torch.Tensor | None  # (N, M) bool, True = real token

    def to(self, device) -> "HeadBatch":
        return HeadBatch(**{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in self.__dict__.items()})


def collate(
    feats: list[QuestionFeatures], types: list[str], use_memory: bool = True, max_memory: int = 1024
) -> HeadBatch:
    N = len(feats)
    D = feats[0].answer_hidden.shape[-1]
    O = max(f.option_hidden.shape[0] for f in feats)
    dev = feats[0].answer_hidden.device
    options = torch.zeros(N, O, D, device=dev)
    option_mask = torch.zeros(N, O, dtype=torch.bool, device=dev)
    prior = torch.zeros(N, O, device=dev)
    has_prior = torch.zeros(N, dtype=torch.bool, device=dev)
    for i, f in enumerate(feats):
        n = f.option_hidden.shape[0]
        options[i, :n] = f.option_hidden.float()
        option_mask[i, :n] = True
        if f.prior is not None:
            prior[i, :n] = f.prior.float()
            has_prior[i] = True
    memory = memory_mask = None
    if use_memory:
        mems = [torch.cat([f.state_hidden, f.branch_hidden])[-max_memory:] for f in feats]
        M = max(m.shape[0] for m in mems)
        memory = torch.zeros(N, M, D, device=dev)
        memory_mask = torch.zeros(N, M, dtype=torch.bool, device=dev)
        for i, m in enumerate(mems):
            memory[i, : m.shape[0]] = m.float()
            memory_mask[i, : m.shape[0]] = True
    return HeadBatch(
        answer=torch.stack([f.answer_hidden.float() for f in feats]),
        options=options,
        option_mask=option_mask,
        prior=prior,
        has_prior=has_prior,
        type_ids=torch.tensor([TYPE_IDS[t] for t in types], device=dev),
        memory=memory,
        memory_mask=memory_mask,
    )


class _Block(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float, use_memory: bool):
        super().__init__()
        self.use_memory = use_memory
        if use_memory:
            self.ln_x = nn.LayerNorm(d)
            self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_s = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_f = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, x, x_pad, mem, mem_pad):
        if self.use_memory:
            h = self.ln_x(x)
            x = x + self.cross(h, mem, mem, key_padding_mask=mem_pad, need_weights=False)[0]
        h = self.ln_s(x)
        x = x + self.self_attn(h, h, h, key_padding_mask=x_pad, need_weights=False)[0]
        return x + self.ffn(self.ln_f(x))


class DecisionHead(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.proj = nn.Sequential(nn.Linear(cfg.d_backbone, d), nn.LayerNorm(d))
        self.mem_proj = nn.Sequential(nn.Linear(cfg.d_backbone, d), nn.LayerNorm(d)) if cfg.use_memory else None
        self.type_emb = nn.Embedding(len(QUESTION_TYPES), d)
        self.level_emb = nn.Embedding(MAX_SCORE_LEVELS, d)
        self.field_emb = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(
            _Block(d, cfg.n_heads, cfg.dropout, cfg.use_memory) for _ in range(cfg.n_layers)
        )
        self.out_ln = nn.LayerNorm(d)
        self.scorer = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.prior_scale = nn.Parameter(torch.ones(len(QUESTION_TYPES)))
        self.gate = nn.Parameter(torch.full((len(QUESTION_TYPES),), -3.0))

    def forward(self, b: HeadBatch) -> torch.Tensor:
        N, O, _ = b.options.shape
        t = self.type_emb(b.type_ids)  # (N, d)
        field = self.proj(b.answer) + t + self.field_emb
        opts = self.proj(b.options) + t[:, None]
        is_score = (b.type_ids == TYPE_IDS["score"]).float()[:, None, None]
        levels = self.level_emb(torch.arange(O, device=opts.device).clamp(max=MAX_SCORE_LEVELS - 1))
        opts = opts + is_score * levels[None]

        x = torch.cat([field[:, None], opts], dim=1)
        x_pad = torch.cat([torch.zeros(N, 1, dtype=torch.bool, device=x.device), ~b.option_mask], dim=1)
        mem = mem_pad = None
        if self.cfg.use_memory:
            mem, mem_pad = self.mem_proj(b.memory), ~b.memory_mask
        for blk in self.blocks:
            x = blk(x, x_pad, mem, mem_pad)
        x = self.out_ln(x)
        f, o = x[:, :1].expand(-1, O, -1), x[:, 1:]
        residual = self.scorer(torch.cat([f, o, f * o, (f - o).abs()], dim=-1)).squeeze(-1)

        scale = self.prior_scale[b.type_ids][:, None] * b.has_prior.float()[:, None]
        gate = torch.sigmoid(self.gate[b.type_ids])[:, None]
        # Questions without a single-token prior rely on the head alone.
        gate = torch.where(b.has_prior[:, None], gate, torch.ones_like(gate))
        logits = scale * b.prior + gate * residual
        return logits.masked_fill(~b.option_mask, float("-inf"))

    def config_dict(self) -> dict:
        return asdict(self.cfg)
