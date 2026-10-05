"""Frozen causal-LM backbone run in prefill-only mode.

Nothing is ever generated. One forward pass over the packed tree (prefix +
isolated question branches) yields, per question:

* ``prior``         - next-token logits of the option labels at the answer slot
                      (the zero-shot "verbalizer" readout);
* ``answer_hidden`` - final hidden state at the answer slot;
* ``option_hidden`` - mean hidden state over each option line;
* ``memory``        - hidden states of state + branch tokens (for the head's
                      evidence routing).

Two execution modes, chosen from the architecture:

* *tree packing* (pure attention models): prefix and branches share one
  sequence; a block-sparse mask and restarted positions isolate the branches;
* *forked branches* (hybrid models with linear-attention / recurrent layers,
  e.g. Qwen3.5/3.6/3.8): recurrent layers ignore attention masks, so packed
  branches would leak into each other. The state is prefilled once and its
  cache (key/values and recurrent states) is copied per branch; branches run
  as rows of one batch.

Both give every question exactly the answer of a separate ``state + question``
call.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .packing import Packed, pack_branches, pack_requests
from .render import EncodedQuestion, EncodedRequest, Renderer
from .schema import Request


@dataclass
class QuestionFeatures:
    answer_hidden: torch.Tensor  # (D,)
    option_hidden: torch.Tensor  # (n_opt, D)
    prior: torch.Tensor | None  # (n_opt,) verbalizer logits, None if labels are not single tokens
    state_hidden: torch.Tensor | None  # (S, D), shared between questions of a request
    branch_hidden: torch.Tensor | None  # (L, D)


@dataclass
class StateCache:
    """Key/values (and hidden states) of an encoded state prefix."""

    prefix_ids: list[int]
    kv: DynamicCache
    hidden: torch.Tensor | None

    @property
    def length(self) -> int:
        return len(self.prefix_ids)


class Backbone:
    def __init__(
        self,
        name_or_path: str,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        attn_implementation: str = "sdpa",
        max_state_tokens: int | None = 4096,
        quantization: str | None = None,
        adapter: str | None = None,
    ):
        """``quantization`` = "4bit" / "8bit" loads the weights with bitsandbytes
        (CUDA only), e.g. a 7B-14B teacher on a single 16 GB T4. ``adapter`` is a
        LoRA directory (see jex/lora.py), merged into the weights at load time."""
        self.name = name_or_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        kwargs = {"dtype": dtype, "attn_implementation": attn_implementation}
        if quantization:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit",
                load_in_8bit=quantization == "8bit",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = {"": self.device.index or 0}
            self.lm = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs).to(self.device)
        self.adapter = adapter
        if adapter:
            from peft import PeftModel

            self.lm = PeftModel.from_pretrained(self.lm, adapter).merge_and_unload()
        self.lm.eval().requires_grad_(False)
        cfg = self.lm.config
        self.text_config = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        layer_types = getattr(self.text_config, "layer_types", None) or []
        self.tree_packing = not any("linear" in t or "mamba" in t for t in layer_types)
        self.base = self.lm.base_model
        self.lm_head = self.lm.get_output_embeddings()
        self.renderer = Renderer(self.tokenizer, max_state_tokens=max_state_tokens)
        pad = self.tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else self.tokenizer.eos_token_id

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    def encode(self, request: Request) -> EncodedRequest:
        return self.renderer.encode(request)

    # ------------------------------------------------------------------ forward

    def _forward(self, packed: Packed, past: DynamicCache | None = None) -> torch.Tensor:
        out = self.base(
            input_ids=packed.input_ids.to(self.device),
            position_ids=packed.position_ids.to(self.device),
            attention_mask=packed.attention_mask.to(self.device, self.dtype),
            past_key_values=past,
            use_cache=past is not None,
        )
        return out.last_hidden_state

    def _prior(self, q: EncodedQuestion, answer_hidden: torch.Tensor) -> torch.Tensor | None:
        if not q.has_prior:
            return None
        logits = self.lm_head(answer_hidden[None]).float()[0]
        return torch.stack([torch.logsumexp(logits[ids], dim=0) for ids in q.label_token_ids])

    def _features(
        self,
        questions: list[EncodedQuestion],
        hidden: torch.Tensor,
        offsets: list[tuple[int, int]],
        state_hidden: torch.Tensor | None,
        keep_memory: bool,
    ) -> list[QuestionFeatures]:
        feats = []
        for q, (start, end) in zip(questions, offsets):
            branch = hidden[start:end]
            answer = branch[q.answer_index]
            options = torch.stack([branch[a:b].mean(0) for a, b in q.option_spans])
            feats.append(
                QuestionFeatures(
                    answer_hidden=answer.float(),
                    option_hidden=options.float(),
                    prior=self._prior(q, answer),
                    state_hidden=state_hidden if keep_memory else None,
                    branch_hidden=branch.float() if keep_memory else None,
                )
            )
        return feats

    @torch.inference_mode()
    def run(self, encoded: list[EncodedRequest], keep_memory: bool = False) -> list[list[QuestionFeatures]]:
        """One forward pass for a batch of requests, all questions in parallel."""
        if not self.tree_packing:
            return [self.run_branches(self.prefill(enc.prefix_ids, keep_memory), enc.questions, keep_memory)
                    for enc in encoded]
        packed = pack_requests(encoded, self.pad_id, self.dtype)
        hidden = self._forward(packed)
        results = []
        for b, enc in enumerate(encoded):
            S = packed.prefix_lengths[b]
            state_hidden = hidden[b, :S].float() if keep_memory else None
            results.append(
                self._features(enc.questions, hidden[b], packed.branch_offsets[b], state_hidden, keep_memory)
            )
        return results

    # ------------------------------------------------------- cached-state path

    @torch.inference_mode()
    def prefill(self, prefix_ids: list[int], keep_memory: bool = False) -> StateCache:
        kv = DynamicCache(config=self.lm.config)
        out = self.base(
            input_ids=torch.tensor([prefix_ids], device=self.device),
            past_key_values=kv,
            use_cache=True,
        )
        hidden = out.last_hidden_state[0].float() if keep_memory else None
        return StateCache(list(prefix_ids), out.past_key_values, hidden)

    @torch.inference_mode()
    def extend(self, state: StateCache, new_ids: list[int]) -> None:
        """Append tokens to a cached state in place (a live state: chat, logs,
        game events). Only the new tokens are computed."""
        if not new_ids:
            return
        out = self.base(
            input_ids=torch.tensor([new_ids], device=self.device),
            position_ids=torch.arange(state.length, state.length + len(new_ids), device=self.device)[None],
            past_key_values=state.kv,
            use_cache=True,
        )
        state.prefix_ids.extend(new_ids)
        if state.hidden is not None:
            state.hidden = torch.cat([state.hidden, out.last_hidden_state[0].float()])

    @torch.inference_mode()
    def run_branches(
        self, state: StateCache, questions: list[EncodedQuestion], keep_memory: bool = False
    ) -> list[QuestionFeatures]:
        """Answer questions against a cached state: only question tokens are computed."""
        if not self.tree_packing:
            return self._run_branches_forked(state, questions, keep_memory)
        packed = pack_branches(questions, state.length, self.dtype)
        try:
            hidden = self._forward(packed, past=state.kv)[0]
        finally:
            # Drop the branch key/values again, layer by layer, so the cache holds
            # only the state even if the forward pass failed half way.
            for layer in state.kv.layers:
                extra = layer.get_seq_length() - state.length
                if extra > 0:
                    layer.crop(-extra)
        return self._features(questions, hidden, packed.branch_offsets[0], state.hidden, keep_memory)

    def _run_branches_forked(
        self, state: StateCache, questions: list[EncodedQuestion], keep_memory: bool
    ) -> list[QuestionFeatures]:
        S, lens = state.length, [len(q.ids) for q in questions]
        B, L = len(questions), max(lens)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((B, S + L), dtype=torch.long)
        mask[:, :S] = 1
        for i, q in enumerate(questions):
            input_ids[i, : lens[i]] = torch.tensor(q.ids)
            mask[i, S : S + lens[i]] = 1  # right padding: it never reaches an answer slot
        out = self.base(
            input_ids=input_ids.to(self.device),
            attention_mask=mask.to(self.device),
            position_ids=torch.arange(S, S + L, device=self.device)[None].expand(B, L),
            past_key_values=fork_cache(state.kv, B),
            use_cache=True,
        )
        hidden = torch.cat([out.last_hidden_state[i, : lens[i]] for i in range(B)])
        offsets, start = [], 0
        for n in lens:
            offsets.append((start, start + n))
            start += n
        return self._features(questions, hidden, offsets, state.hidden, keep_memory)

    # ------------------------------------------------------ autoregressive ref

    @torch.inference_mode()
    def generate_text(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        """Plain greedy decoding, used only as the autoregressive baseline."""
        out = self.lm.generate(
            torch.tensor([prompt_ids], device=self.device),
            attention_mask=torch.ones(1, len(prompt_ids), dtype=torch.long, device=self.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.pad_id,
        )
        return out[0, len(prompt_ids):].tolist()


def fork_cache(cache: DynamicCache, n: int) -> DynamicCache:
    """A copy of ``cache`` repeated ``n`` times along the batch dimension. Works for
    attention layers (keys/values) and linear-attention layers (conv and recurrent
    states) alike; the original cache is left untouched."""

    def rep(v):
        if torch.is_tensor(v) and v.dim() > 0:
            return v.repeat_interleave(n, dim=0)
        if isinstance(v, list):
            return [rep(x) for x in v]
        if isinstance(v, dict):
            return {k: rep(x) for k, x in v.items()}
        return v

    forked = copy.deepcopy(cache)
    for layer in forked.layers:
        for name, value in list(vars(layer).items()):
            setattr(layer, name, rep(value))
    return forked
