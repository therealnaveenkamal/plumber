"""DecisionHead: a small transformer that reads a FROZEN trunk and returns a distribution over the supplied options.

Input per question: the trunk's hidden states at the decision suffix only (question, option and DECIDE tokens), taken
from several layers ("taps"). The state itself is never re-read here: the trunk already attended over it, and not
touching state positions is what lets a decision reuse a generator's KV cache.

    taps [S, T, d] --per-tap LayerNorm, concat, Linear--> x [S, d_h] (+ segment embedding: question/option/decide)
      --token blocks (bidirectional attention over the suffix)-->
      --pool each option: Linear([last ; mean])--> o_k,   q = x[DECIDE]
      --set blocks over {q, o_1..o_K} (no positions: order-invariant)-->
      z_k = <W_q q, W_o o_k> / sqrt(d_proj) + u . o_k

Dense on purpose: the head sees a few hundred tokens and costs ~1% of a trunk forward, so capacity is cheap and exact
attention beats linear-attention or MoE variants at this length.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SEG_QUESTION, SEG_OPTION, SEG_DECIDE, SEG_OTHER = 0, 1, 2, 3


@dataclass
class HeadConfig:
    d: int  # trunk hidden size
    n_taps: int  # number of trunk layers read
    d_h: int = 1024
    n_heads: int = 16
    token_blocks: int = 1
    set_blocks: int = 2
    d_proj: int = 512
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    """Pre-norm transformer block with a key-padding mask (bidirectional)."""

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h, self.p = n_heads, dropout

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """x [B, S, d]; keep [B, S] bool (False = padding)."""
        B, S, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, S, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        mask = keep[:, None, None, :]  # attend only to real positions
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0
        )
        x = x + self.out(a.transpose(1, 2).reshape(B, S, d))
        return x + self.mlp(self.n2(x))


@dataclass
class SuffixBatch:
    """Padded decision suffixes. feats [B, S, T, d]; seg [B, S] segment ids; keep [B, S];
    opt_spans per row [(start, end)] within the suffix; decide per row (index within the suffix)."""

    feats: torch.Tensor
    seg: torch.Tensor
    keep: torch.Tensor
    opt_spans: list[list[tuple[int, int]]]
    decide: list[int]

    @staticmethod
    def collate(items: list[dict], device=None) -> SuffixBatch:
        """items: {"feats": [S, T, d] tensor, "opt_spans": [...], "decide": int} with spans relative to the suffix."""
        B, S = len(items), max(it["feats"].shape[0] for it in items)
        T, d = items[0]["feats"].shape[1:]
        dev = device or items[0]["feats"].device
        feats = torch.zeros(B, S, T, d, dtype=items[0]["feats"].dtype, device=dev)
        seg = torch.full((B, S), SEG_OTHER, dtype=torch.long, device=dev)
        keep = torch.zeros(B, S, dtype=torch.bool, device=dev)
        for b, it in enumerate(items):
            n = it["feats"].shape[0]
            feats[b, :n] = it["feats"].to(dev)
            keep[b, :n] = True
            seg[b, :n] = SEG_QUESTION
            for s, e in it["opt_spans"]:
                seg[b, s:e] = SEG_OPTION
            seg[b, it["decide"]] = SEG_DECIDE
        return SuffixBatch(
            feats, seg, keep, [it["opt_spans"] for it in items], [it["decide"] for it in items]
        )


class DecisionHead(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        self.tap_norm = nn.ModuleList(nn.LayerNorm(cfg.d) for _ in range(cfg.n_taps))
        self.inp = nn.Linear(cfg.n_taps * cfg.d, cfg.d_h)
        self.seg = nn.Embedding(4, cfg.d_h)
        self.tok_blocks = nn.ModuleList(
            Block(cfg.d_h, cfg.n_heads, cfg.dropout) for _ in range(cfg.token_blocks)
        )
        self.pool = nn.Linear(2 * cfg.d_h, cfg.d_h)
        self.set_type = nn.Embedding(2, cfg.d_h)  # 0 = decide, 1 = option
        self.set_blocks = nn.ModuleList(
            Block(cfg.d_h, cfg.n_heads, cfg.dropout) for _ in range(cfg.set_blocks)
        )
        self.norm = nn.LayerNorm(cfg.d_h)
        self.wq = nn.Linear(cfg.d_h, cfg.d_proj, bias=False)
        self.wo = nn.Linear(cfg.d_h, cfg.d_proj, bias=False)
        self.u = nn.Linear(cfg.d_h, 1, bias=False)
        nn.init.zeros_(self.u.weight)
        self.scale = 1.0 / math.sqrt(cfg.d_proj)
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, b: SuffixBatch) -> torch.Tensor:
        """Returns logits [B, K_max] in fp32, -inf where a row has fewer options."""
        with torch.autocast(b.feats.device.type, enabled=False):
            f = b.feats.float()
            x = self.inp(
                torch.cat([n(f[:, :, t]) for t, n in enumerate(self.tap_norm)], -1)
            ) + self.seg(b.seg)
            for blk in self.tok_blocks:
                x = blk(x, b.keep)
            B, K = x.shape[0], max(len(s) for s in b.opt_spans)
            # option pooling: [last ; mean] over each span, gathered into a [B, 1 + K] set (decide first)
            last = x.new_zeros(B, K, x.shape[-1])
            mean = x.new_zeros(B, K, x.shape[-1])
            okeep = torch.zeros(B, 1 + K, dtype=torch.bool, device=x.device)
            okeep[:, 0] = True
            for i, spans in enumerate(b.opt_spans):  # K is small; spans are ragged
                for k, (s, e) in enumerate(spans):
                    last[i, k] = x[i, e - 1]
                    mean[i, k] = x[i, s:e].mean(0)
                    okeep[i, 1 + k] = True
            o = self.pool(torch.cat([last, mean], -1)) + self.set_type.weight[1]
            q = (
                x[torch.arange(B, device=x.device), torch.tensor(b.decide, device=x.device)]
                + self.set_type.weight[0]
            )
            s = torch.cat([q[:, None], o], 1)
            for blk in self.set_blocks:
                s = blk(s, okeep)
            s = self.norm(s)
            q, o = s[:, 0], s[:, 1:]
            z = (self.wo(o) * self.wq(q)[:, None]).sum(-1) * self.scale + self.u(o).squeeze(-1)
            return (z / self.temperature).masked_fill(~okeep[:, 1:], float("-inf"))


def decision_loss(logits: torch.Tensor, gold, teacher, lam: float = 0.5) -> torch.Tensor:
    """CE on gold when present; + lam * KL(teacher || p) when a teacher distribution is present.
    logits [B,K] (-inf padded). gold: list[int|None]. teacher: list[list[float]|None]."""
    logp = F.log_softmax(logits, -1)
    B, K = logits.shape
    terms = []
    has_gold = [b for b in range(B) if gold[b] is not None]
    if has_gold:
        idx = torch.tensor(has_gold, device=logits.device)
        g = torch.tensor([gold[b] for b in has_gold], device=logits.device)
        terms.append(-logp[idx, g])
    has_t = [b for b in range(B) if teacher[b] is not None]
    if has_t:
        t = logits.new_zeros((len(has_t), K))
        for i, b in enumerate(has_t):
            t[i, : len(teacher[b])] = torch.tensor(teacher[b], dtype=t.dtype)
        t = t / t.sum(-1, keepdim=True).clamp_min(1e-9)
        lp = logp[torch.tensor(has_t, device=logits.device)].masked_fill(t == 0, 0.0)
        terms.append(lam * (t * (t.clamp_min(1e-9).log() - lp)).sum(-1))
    if not terms:
        return logits.sum() * 0.0
    return torch.cat(terms).sum() / len(torch.cat(terms))
