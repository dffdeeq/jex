"""JexModel: backbone + (optional) decision head + calibration -> typed answers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .backbone import Backbone, QuestionFeatures, StateCache
from .head import DecisionHead, HeadConfig, collate
from .render import EncodedQuestion
from .schema import Answer, Question, Request, build_response, parse_request


class JexModel:
    def __init__(
        self,
        backbone: Backbone,
        head: DecisionHead | None = None,
        temperatures: dict[str, float] | None = None,
        name: str = "jex-0.1",
    ):
        self.backbone = backbone
        device = getattr(backbone, "device", torch.device("cpu"))
        self.head = head.to(device).eval() if head is not None else None
        self.temperatures = temperatures or {}
        self.name = name

    @property
    def keep_memory(self) -> bool:
        return self.head is not None and self.head.cfg.use_memory

    # ----------------------------------------------------------------- scoring

    @torch.inference_mode()
    def logits(self, feats: list[QuestionFeatures], questions: list[Question]) -> list[torch.Tensor]:
        if self.head is not None:
            batch = collate(feats, [q.type for q in questions], self.head.cfg.use_memory, self.head.cfg.max_memory)
            out = self.head(batch)
            raw = [out[i, : q.num_options] for i, q in enumerate(questions)]
        else:
            raw = [
                f.prior if f.prior is not None else torch.zeros(q.num_options, device=f.answer_hidden.device)
                for f, q in zip(feats, questions)
            ]
        return [lg.float() / self.temperatures.get(q.type, 1.0) for lg, q in zip(raw, questions)]

    def answers(self, feats: list[QuestionFeatures], questions: list[Question]) -> list[Answer]:
        return [
            Answer(q, torch.softmax(lg, dim=-1).tolist())
            for lg, q in zip(self.logits(feats, questions), questions)
        ]

    # ------------------------------------------------------------------ public

    def predict(self, payload: dict[str, Any] | Request) -> dict[str, Any]:
        return self.predict_batch([payload])[0]

    def predict_batch(self, payloads: list[dict[str, Any] | Request]) -> list[dict[str, Any]]:
        requests = [p if isinstance(p, Request) else parse_request(p) for p in payloads]
        encoded = [self.backbone.encode(r) for r in requests]
        all_feats = self.backbone.run(encoded, keep_memory=self.keep_memory)
        return [
            build_response(self.name, self.answers(feats, list(r.questions)), enc.num_tokens)
            for r, enc, feats in zip(requests, encoded, all_feats)
        ]

    def prefill(self, state: Any) -> StateCache:
        text = Request(state=state, questions=()).state_text()
        return self.backbone.prefill(self.backbone.renderer.encode_prefix(text), keep_memory=self.keep_memory)

    def extend_state(self, cache: StateCache, text: str) -> None:
        self.backbone.extend(cache, self.backbone.tokenizer(text, add_special_tokens=False)["input_ids"])

    def answer_cached(self, cache: StateCache, questions: list[Question]) -> list[Answer]:
        encoded: list[EncodedQuestion] = [self.backbone.renderer.encode_question(q) for q in questions]
        feats = self.backbone.run_branches(cache, encoded, keep_memory=self.keep_memory)
        return self.answers(feats, questions)

    # ------------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        save_checkpoint(path, self.name, self.backbone.name, self.head, self.temperatures)

    @classmethod
    def load(cls, path: str | Path, backbone: Backbone | None = None, **backbone_kwargs) -> "JexModel":
        path = Path(path)
        meta = json.loads((path / "jex.json").read_text())
        if backbone is None:
            adapter = meta.get("adapter")
            if adapter is not None:
                backbone_kwargs["adapter"] = str((path / adapter).resolve())
            backbone = Backbone(meta["backbone"], **backbone_kwargs)
        head = None
        if "head" in meta:
            head = DecisionHead(HeadConfig(**meta["head"]))
            head.load_state_dict(torch.load(path / "head.pt", map_location="cpu"))
        return cls(backbone, head, meta.get("temperatures"), meta.get("name", "jex"))


def save_checkpoint(
    path: str | Path,
    name: str,
    backbone_name: str,
    head: DecisionHead | None,
    temperatures: dict[str, float],
    adapter: str | None = None,
) -> None:
    """A checkpoint is tiny: the backbone is referenced by name; only the head and/or
    a LoRA adapter (``adapter`` is relative to the checkpoint directory) are stored."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "backbone": backbone_name, "temperatures": temperatures}
    if adapter is not None:
        meta["adapter"] = adapter
    if head is not None:
        meta["head"] = head.config_dict()
        torch.save(head.state_dict(), path / "head.pt")
    (path / "jex.json").write_text(json.dumps(meta, indent=2))
