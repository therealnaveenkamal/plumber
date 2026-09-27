"""Readout heads. The trunk's hidden states are read at option spans and at DECIDE; nothing here
ever sees a vocabulary."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerHead(nn.Module):
    """z_k = <W_q q, W_o o_k> / sqrt(d_proj) + u . o_k   with o_k = LN(Linear([h_last_k ; mean_k]))

    d      : trunk hidden size (2688 for Nemotron 3.5 Lightning)
    d_proj : bilinear rank (512 default; ablate 256/512/1024 and identity)
    """

    def __init__(self, d: int, d_proj: int = 512):
        super().__init__()
        self.pool = nn.Linear(2 * d, d)
        self.ln = nn.LayerNorm(d)
        self.wq = nn.Linear(d, d_proj, bias=False)
        self.wo = nn.Linear(d, d_proj, bias=False)
        self.u = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.u.weight)
        self.scale = 1.0 / math.sqrt(d_proj)

    def forward(self, h: torch.Tensor, opt_spans, decide_pos) -> torch.Tensor:
        """h: [B, L, d]. opt_spans: list (len B) of lists of (start, end). decide_pos: list (len B).
        Returns logits [B, K_max] with -inf padding."""
        B, K_max = h.shape[0], max(len(s) for s in opt_spans)
        logits = h.new_full((B, K_max), float("-inf"), dtype=torch.float32)
        for b in range(B):
            q = h[b, decide_pos[b]]  # [d]
            last = torch.stack([h[b, e - 1] for (s, e) in opt_spans[b]])  # [K, d]
            mean = torch.stack([h[b, s:e].mean(0) for (s, e) in opt_spans[b]])  # [K, d]
            o = self.ln(self.pool(torch.cat([last, mean], -1)))  # [K, d]
            z = (self.wo(o) @ self.wq(q)) * self.scale + self.u(o).squeeze(-1)
            logits[b, : z.shape[0]] = z.float()
        return logits


def decision_loss(logits: torch.Tensor, gold, teacher, qtype, lam: float = 0.5) -> torch.Tensor:
    """CE on gold when present; + lam * KL(teacher || p) when a teacher distribution is present.
    logits [B,K] (-inf padded). gold: list[int|None]. teacher: list[list[float]|None]."""
    logp = F.log_softmax(logits, -1)
    total, n = logits.new_zeros(()), 0
    for b in range(logits.shape[0]):
        if gold[b] is not None:
            total = total - logp[b, gold[b]]
            n += 1
        if teacher[b] is not None:
            t = logits.new_tensor(teacher[b])
            t = t / t.sum().clamp_min(1e-9)
            k = t.shape[0]
            total = total + lam * (t * (t.clamp_min(1e-9).log() - logp[b, :k])).sum()
            n += 1
    return total / max(n, 1)
