"""The two frozen interfaces.

1. Row  — one decision: state, question, typed options, gold index, optional teacher distribution.
2. render(tok, row, rng) — deterministic tokenisation into ONE sequence with the token span of every
   option and the DECIDE position, so any head can read the trunk at those positions.

Options are tokenised piecewise (never as one big string) so token spans are exact and the state
prefix is tokenised alone — which is what makes Option B (cache the state, fork per question) clean.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json, random

QTYPES = ("choice", "noul", "score")

@dataclass
class Option:
    name: str
    desc: str = ""

@dataclass
class Row:
    id: str
    state: str
    question: str
    qtype: str                       # choice | noul | score
    options: List[Option]            # noul: exactly [yes, no]; score: ordered levels low -> high
    gold: Optional[int] = None       # index into options, or None (teacher-only row)
    teacher: Optional[List[float]] = None   # distribution over options in the ORIGINAL order, or None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.qtype in QTYPES, self.qtype
        if self.qtype == "noul": assert len(self.options) == 2
        if self.gold is not None: assert 0 <= self.gold < len(self.options)
        if self.teacher is not None: assert len(self.teacher) == len(self.options)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "Row":
        d = json.loads(s); d["options"] = [Option(**o) for o in d["options"]]; return Row(**d)


# text markers: plain strings, no special tokens, no embedding resize.
S_OPEN, S_CLOSE = "<|state|>\n", "\n<|/state|>\n"
Q_OPEN, Q_CLOSE = "<|q|> ", " <|/q|>\n"
O_OPEN, O_CLOSE = "<|opt|> ", " <|/opt|>\n"
DECIDE = "[DECIDE]"

def _ids(tok, text: str) -> List[int]:
    return tok(text, add_special_tokens=False).input_ids

def render(tok, row: Row, rng: Optional[random.Random] = None, shuffle: bool = True, bos: bool = True) -> Dict[str, Any]:
    """Return token ids for ONE sequence plus the positions every head needs.

    keys: input_ids, prefix_len (state only — the Option-B cache boundary), q_start,
          opt_spans [(start, end_exclusive)] in the PRESENTED order, perm (presented -> original index),
          decide_pos, gold (remapped to presented order or None), teacher (permuted or None)
    """
    K = len(row.options)
    perm = list(range(K))
    if shuffle and row.qtype != "score" and rng is not None:       # never shuffle ordinal levels
        rng.shuffle(perm)
    ids: List[int] = ([tok.bos_token_id] if (bos and tok.bos_token_id is not None) else [])
    ids += _ids(tok, S_OPEN + row.state + S_CLOSE)
    prefix_len = len(ids)
    q_start = len(ids)
    ids += _ids(tok, Q_OPEN + row.question + Q_CLOSE)
    spans = []
    for j in perm:
        o = row.options[j]
        text = O_OPEN + (f"{o.name}: {o.desc}" if o.desc else o.name) + O_CLOSE
        s = len(ids); ids += _ids(tok, text); spans.append((s, len(ids)))
    ids += _ids(tok, DECIDE)
    inv = {orig: pos for pos, orig in enumerate(perm)}
    return {
        "input_ids": ids, "prefix_len": prefix_len, "q_start": q_start, "opt_spans": spans, "perm": perm,
        "decide_pos": len(ids) - 1,
        "gold": None if row.gold is None else inv[row.gold],
        "teacher": None if row.teacher is None else [row.teacher[j] for j in perm],
        "qtype": row.qtype, "id": row.id,
    }


def score_signature(state: str, question: str, options: List[Option]):
    """The callable contract every model must satisfy: score(state, question, options) -> logits[K].
    Implemented by PlumbModel.score; documented here so workstreams can build against it."""
    raise NotImplementedError
