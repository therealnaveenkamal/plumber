"""Calibration: one temperature, a split-conformal threshold, and the System 1 -> System 2 escalation it implies.

Everything works on prediction rows ``{"probs": [...], "gold": int}`` as written by ``plumbify eval-head`` (probs at T = 1).

- ``fit_temperature``: NLL-optimal T; moves confidence, never the winning option.
- ``fit_conformal``:   split conformal with the LAC score s = 1 - p_gold. ``prediction_set`` then contains the gold
  option with probability >= 1 - alpha on exchangeable data, distribution-free.
- ``escalation_curve``: for each confidence threshold, the share of traffic System 1 keeps and its accuracy there.
  A request whose conformal set is not a singleton is exactly one System 1 should hand to System 2.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def rescale(probs: Sequence[float], temperature: float) -> list[float]:
    z = [math.log(max(p, 1e-12)) / temperature for p in probs]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def fit_temperature(rows: list[dict], lo: float = 0.25, hi: float = 8.0, iters: int = 60) -> float:
    """Golden-section search of the mean NLL over log T (the NLL is unimodal in log T)."""
    rows = [r for r in rows if r.get("gold") is not None]

    def nll(log_t: float) -> float:
        t = math.exp(log_t)
        return -sum(math.log(max(rescale(r["probs"], t)[r["gold"]], 1e-12)) for r in rows) / max(
            len(rows), 1
        )

    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(d)
    return math.exp((a + b) / 2)


def fit_conformal(rows: list[dict], alpha: float = 0.1, temperature: float = 1.0) -> dict:
    """qhat = the ceil((n+1)(1-alpha))/n empirical quantile of s = 1 - p_gold on held-out rows."""
    scores = sorted(
        1.0 - rescale(r["probs"], temperature)[r["gold"]] for r in rows if r.get("gold") is not None
    )
    n = len(scores)
    if n == 0:
        raise ValueError("conformal calibration needs labelled rows")
    k = math.ceil((n + 1) * (1 - alpha))
    qhat = 1.0 if k > n else scores[k - 1]
    return {"alpha": alpha, "qhat": qhat, "n": n}


def prediction_set(probs: Sequence[float], qhat: float) -> list[int]:
    """Options whose probability clears 1 - qhat; never empty (falls back to the argmax)."""
    s = [k for k, p in enumerate(probs) if p >= 1.0 - qhat]
    return s or [max(range(len(probs)), key=probs.__getitem__)]


def coverage(rows: list[dict], qhat: float, temperature: float = 1.0) -> dict:
    """Empirical coverage and mean set size of the conformal sets on another labelled set."""
    hit = size = singles = 0
    rows = [r for r in rows if r.get("gold") is not None]
    for r in rows:
        s = prediction_set(rescale(r["probs"], temperature), qhat)
        hit += r["gold"] in s
        size += len(s)
        singles += len(s) == 1
    n = max(len(rows), 1)
    return {
        "coverage": hit / n,
        "mean_set_size": size / n,
        "singleton_rate": singles / n,
        "n": len(rows),
    }


def escalation_curve(
    rows: list[dict],
    temperature: float = 1.0,
    thresholds: Sequence[float] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
) -> list[dict]:
    """For each p_max threshold: share kept by System 1, System 1 accuracy on what it keeps."""
    rows = [r for r in rows if r.get("gold") is not None]
    out = []
    for t in thresholds:
        kept = [r for r in rows if max(rescale(r["probs"], temperature)) >= t]
        acc = sum(
            max(range(len(r["probs"])), key=r["probs"].__getitem__) == r["gold"] for r in kept
        )
        out.append(
            {
                "threshold": t,
                "system1_share": len(kept) / max(len(rows), 1),
                "system1_accuracy": acc / max(len(kept), 1),
            }
        )
    return out
