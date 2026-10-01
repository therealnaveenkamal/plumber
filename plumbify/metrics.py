"""Decision metrics: accuracy, NLL, Brier, ECE, coverage at an error budget. Pure Python; used by train-head and
eval-head."""

from __future__ import annotations

import math
from collections.abc import Sequence


def accuracy(correct: Sequence[int]) -> float:
    return sum(correct) / len(correct) if correct else 0.0


def nll(p_gold: Sequence[float]) -> float:
    return -sum(math.log(max(p, 1e-9)) for p in p_gold) / len(p_gold) if p_gold else 0.0


def brier(probs: Sequence[Sequence[float]], gold: Sequence[int]) -> float:
    """Multi-class Brier score, mean over rows of sum_k (p_k - 1[k == gold])^2."""
    tot = 0.0
    for p, g in zip(probs, gold, strict=True):
        tot += sum((pk - (1.0 if k == g else 0.0)) ** 2 for k, pk in enumerate(p))
    return tot / len(gold) if gold else 0.0


def ece(conf: Sequence[float], correct: Sequence[int], bins: int = 15) -> float:
    """Expected calibration error on the max-probability, equal-width bins."""
    n = len(conf)
    if n == 0:
        return 0.0
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi]
        if idx:
            tot += (
                len(idx)
                / n
                * abs(
                    sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx)
                )
            )
    return tot


def coverage_at_error(
    conf: Sequence[float], correct: Sequence[int], max_error: float = 0.05
) -> float:
    """Largest fraction of rows that can be answered automatically (highest confidence first) while keeping error <= max_error."""
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    best, wrong = 0.0, 0
    for k, i in enumerate(order, 1):
        wrong += 1 - correct[i]
        if wrong / k <= max_error:
            best = k / len(order)
    return best


def summarize(rows: list[dict]) -> dict:
    """rows: dicts with keys correct (0/1), p_max, p_gold, probs (list), gold (int)."""
    scored = [r for r in rows if r.get("correct") is not None]
    if not scored:
        return {"n": 0}
    c = [r["correct"] for r in scored]
    conf = [r["p_max"] for r in scored]
    return {
        "n": len(scored),
        "accuracy": accuracy(c),
        "nll": nll([r["p_gold"] for r in scored]),
        "brier": brier([r["probs"] for r in scored], [r["gold"] for r in scored]),
        "ece": ece(conf, c),
        "coverage_at_5pct_error": coverage_at_error(conf, c, 0.05),
    }
