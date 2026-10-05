"""Pack a shared prefix and N question branches into one sequence.

The trick that makes all answers come out of a single forward pass while
staying exactly as good as N separate calls:

* every branch attends to the full prefix and causally to itself, never to
  another branch (block-sparse "tree" mask);
* every branch's position ids restart right after the prefix, so from the
  model's point of view branch i sits at the same positions as it would in a
  standalone ``prefix + question_i`` prompt.

Hence the hidden states of branch i are (up to float noise) identical to a
separate call - questions are independent by construction, adding a question
never changes another answer, and they can be computed in any order or
grouping. With a cached prefix the same mask is applied against the cached
key/values, so new questions arriving later only pay for their own tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .render import EncodedQuestion, EncodedRequest

PAD_SEGMENT = -1


@dataclass
class Packed:
    input_ids: torch.Tensor  # (B, T)
    position_ids: torch.Tensor  # (B, T)
    attention_mask: torch.Tensor  # (B, 1, T, K) additive float mask
    # For every row, the packed [start, end) of each branch.
    branch_offsets: list[list[tuple[int, int]]]
    prefix_lengths: list[int]


def _segments(prefix_len: int, branch_lens: list[int]) -> tuple[list[int], list[int]]:
    seg, pos = [0] * prefix_len, list(range(prefix_len))
    for i, n in enumerate(branch_lens, start=1):
        seg += [i] * n
        pos += list(range(prefix_len, prefix_len + n))
    return seg, pos


def _to_additive(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros(allowed.shape, dtype=dtype)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)


def tree_mask(q_seg: torch.Tensor, k_seg: torch.Tensor, q_idx: torch.Tensor, k_idx: torch.Tensor):
    """Boolean (Tq, Tk) mask: causal in packed order, and a key is visible when it
    is in the shared prefix (segment 0) or in the query's own branch."""
    causal = k_idx[None, :] <= q_idx[:, None]
    same = (k_seg[None, :] == 0) | (k_seg[None, :] == q_seg[:, None])
    allowed = causal & same & (k_seg[None, :] != PAD_SEGMENT)
    # Padding queries attend to themselves only, so softmax never sees an empty row.
    pad_q = q_seg == PAD_SEGMENT
    if pad_q.any():
        allowed[pad_q] = False
        allowed[pad_q, q_idx[pad_q]] = True
    return allowed


def pack_requests(
    encoded: list[EncodedRequest], pad_id: int, dtype: torch.dtype = torch.float32
) -> Packed:
    """Full (uncached) packing; several requests are stacked along the batch dim."""
    rows, offsets = [], []
    for enc in encoded:
        lens = [len(q.ids) for q in enc.questions]
        seg, pos = _segments(len(enc.prefix_ids), lens)
        ids = list(enc.prefix_ids)
        offs, start = [], len(enc.prefix_ids)
        for q in enc.questions:
            ids += q.ids
            offs.append((start, start + len(q.ids)))
            start += len(q.ids)
        rows.append((ids, pos, seg))
        offsets.append(offs)

    T = max(len(r[0]) for r in rows)
    B = len(rows)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    position_ids = torch.zeros((B, T), dtype=torch.long)
    mask = torch.empty((B, 1, T, T), dtype=dtype)
    idx = torch.arange(T)
    for b, (ids, pos, seg) in enumerate(rows):
        n = len(ids)
        input_ids[b, :n] = torch.tensor(ids)
        position_ids[b, :n] = torch.tensor(pos)
        segs = torch.full((T,), PAD_SEGMENT, dtype=torch.long)
        segs[:n] = torch.tensor(seg)
        mask[b, 0] = _to_additive(tree_mask(segs, segs, idx, idx), dtype)
    return Packed(input_ids, position_ids, mask, offsets, [len(e.prefix_ids) for e in encoded])


def pack_branches(
    questions: list[EncodedQuestion], prefix_len: int, dtype: torch.dtype = torch.float32
) -> Packed:
    """Packing of branches only, to run against a cached prefix of ``prefix_len`` tokens."""
    lens = [len(q.ids) for q in questions]
    seg, pos = _segments(prefix_len, lens)
    q_seg = torch.tensor(seg[prefix_len:], dtype=torch.long)
    k_seg = torch.tensor(seg, dtype=torch.long)
    Q = len(q_seg)
    q_idx = torch.arange(prefix_len, prefix_len + Q)
    k_idx = torch.arange(prefix_len + Q)
    allowed = tree_mask(q_seg, k_seg, q_idx, k_idx)
    ids = [t for q in questions for t in q.ids]
    offs, start = [], 0
    for n in lens:
        offs.append((start, start + n))
        start += n
    return Packed(
        input_ids=torch.tensor([ids], dtype=torch.long),
        position_ids=torch.tensor([pos[prefix_len:]], dtype=torch.long),
        attention_mask=_to_additive(allowed, dtype)[None, None],
        branch_offsets=[offs],
        prefix_lengths=[prefix_len],
    )
