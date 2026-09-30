"""Option probabilities -> typed answers (choice / noul / score) with a confidence and, optionally, a conformal set."""

from __future__ import annotations

import json

from .row import Row


def state_text(state) -> str:
    """Objects and arrays become labeled text; strings pass through."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(
            f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
            for k, v in state.items()
        )
    return json.dumps(state, ensure_ascii=False)


def choice_confidence(probs: list[float]) -> float:
    """How far the leading option stands above uniform: (p_max - 1/K) / (1 - 1/K); 1 for a single option."""
    if len(probs) <= 1:
        return 1.0
    u = 1.0 / len(probs)
    return (max(probs) - u) / (1.0 - u)


def score_confidence(probs: list[float]) -> float:
    """Concentration around the modal level: max(0, 1 - E|level - mode| / D), D = uniform mean absolute deviation."""
    n = len(probs)
    if n <= 1:
        return 1.0
    mode = max(range(n), key=probs.__getitem__)
    spread = sum(p * abs(i - mode) for i, p in enumerate(probs))
    center = (n - 1) / 2
    d_uniform = sum(abs(i - center) for i in range(n)) / n
    return max(0.0, 1.0 - spread / d_uniform)


def answer(row: Row, probs: list[float], qhat: float | None = None) -> dict:
    """Probabilities over ``row.options`` (in order) -> the answer object for ``row.qtype``. With a conformal
    threshold ``qhat``, the answer also carries ``set``: the options that cannot be ruled out."""
    out = _typed(row, probs)
    if qhat is not None:
        from ..calibration import prediction_set

        out["set"] = [row.options[k].name for k in prediction_set(probs, qhat)]
    return out


def _typed(row: Row, probs: list[float]) -> dict:
    named = {o.name: round(p, 6) for o, p in zip(row.options, probs, strict=True)}
    if row.qtype == "noul":
        return {
            "type": "noul",
            "noul": named["yes"],
            "probabilities": named,
            "confidence": round(choice_confidence(probs), 6),
        }
    if row.qtype == "choice":
        return {
            "type": "choice",
            "choice": max(named, key=named.get),
            "probabilities": named,
            "confidence": round(choice_confidence(probs), 6),
        }
    return {
        "type": "score",
        "score": round(sum(i * p for i, p in enumerate(probs)), 4),
        "legend": {o.name: o.desc for o in row.options},
        "probabilities": named,
        "confidence": round(score_confidence(probs), 6),
    }
