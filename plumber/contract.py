"""The System One contract: request questions -> Rows, option probabilities -> typed answers.

Field names, confidence formulas and the training label format follow TypeSafe's reference adapter
(system-one-adapter 0.2.1), so the official SDK and Jev clients work against a Plumb server unchanged.
"""

from __future__ import annotations

import json

from .core.rendering import Option, Row

NOUL_OPTIONS = ("no", "yes")  # noul is the two-option case of choice; index 1 is "yes"


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


def question_to_row(qid, q: dict, state, label=None) -> Row:
    """One API question -> one Row.

    choice: criteria {name: description | null} (or a list of names); noul: optional {"true": …, "false": …};
    score: ordered list of level descriptions. ``label`` (training only) is the option name for choice,
    true/false for noul, and the 0-based level index for score.
    """
    t = q.get("type")
    crit = q.get("criteria")
    instr = q.get("instructions", "")
    if t == "choice":
        if isinstance(crit, dict):
            opts = [Option(str(k), str(v or "")) for k, v in crit.items()]
        elif isinstance(crit, list):
            opts = [Option(str(k), "") for k in crit]
        else:
            raise ValueError(f"{qid}: choice needs criteria")
        found = described_by_state(
            state, [o.name for o in opts], require_exact=any(o.desc for o in opts)
        )
        if found:
            opts = [
                Option(o.name, f"{o.desc}: {found[o.name]}" if o.desc else found[o.name])
                for o in opts
            ]
    elif t == "noul":
        c = crit if isinstance(crit, dict) else {}
        opts = [Option("no", str(c.get("false") or "")), Option("yes", str(c.get("true") or ""))]
    elif t == "score":
        if not isinstance(crit, list) or not crit:
            raise ValueError(f"{qid}: score needs a list of level descriptions")
        opts = [Option(str(i), str(d)) for i, d in enumerate(crit)]
    else:
        raise ValueError(f"{qid}: unknown type {t!r}")
    gold = None if label is None else _gold_index(qid, t, opts, label)
    return Row(
        id=str(qid),
        state=state_text(state),
        question=instr if isinstance(instr, str) else state_text(instr),
        qtype=t,
        options=opts,
        gold=gold,
    )


def described_by_state(
    state, names: list[str], require_exact: bool = False
) -> dict[str, str] | None:
    """Option labels that key a mapping inside the state (``candidate_summaries: {a: …, b: …}``) take that content as
    their description, so the readout sees the candidate text under its marker rather than a lone letter.

    Bare labels match any mapping containing all the names; labels that already carry a description only match a
    nested mapping whose keys are exactly the option set (``require_exact``), and the content is appended.
    """
    if not isinstance(state, dict) or not names:
        return None
    nested = [v for v in state.values() if isinstance(v, dict)]
    for m in nested if require_exact else [state, *nested]:
        if require_exact and set(m) != set(names):
            continue
        if all(n in m and isinstance(m[n], str | int | float) for n in names):
            return {n: str(m[n]) for n in names}
    return None


def _gold_index(qid, t, opts, label) -> int:
    if t == "noul":
        if isinstance(label, str):
            label = label.strip().lower() in ("true", "yes", "1")
        return int(bool(label))
    if t == "score":
        i = int(label)
        if not 0 <= i < len(opts):
            raise ValueError(f"{qid}: score label {label} outside 0..{len(opts) - 1}")
        return i
    names = [o.name for o in opts]
    if str(label) not in names:
        raise ValueError(f"{qid}: choice label {label!r} not in {names}")
    return names.index(str(label))


def rows_from_request(req: dict, prefix: str = "") -> list[Row]:
    """A labelled API request (every question carries ``label``) -> one Row per question."""
    state = req.get("state", "")
    rows = []
    for qid, q in (req.get("questions") or {}).items():
        if "label" not in q:
            raise ValueError(f"{qid}: training rows need a label on every question")
        r = question_to_row(qid, q, state, label=q["label"])
        r.id = f"{prefix}{qid}" if prefix else r.id
        rows.append(r)
    return rows


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


def answer(row: Row, probs: list[float]) -> dict:
    """Probabilities over ``row.options`` (in order) -> the answer object for ``row.qtype``."""
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
