"""One decision: a state, a question, typed options, and (for training) a gold index or a teacher distribution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

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
    qtype: str  # choice | noul | score
    options: list[Option]  # noul: exactly [yes, no]; score: ordered levels low -> high
    gold: int | None = None  # index into options, or None (teacher-only row)
    teacher: list[float] | None = None  # distribution over options in the ORIGINAL order, or None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.qtype in QTYPES, self.qtype
        if self.qtype == "noul":
            assert len(self.options) == 2
        if self.gold is not None:
            assert 0 <= self.gold < len(self.options)
        if self.teacher is not None:
            assert len(self.teacher) == len(self.options)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> Row:
        d = json.loads(s)
        d["options"] = [Option(**o) for o in d["options"]]
        return Row(**d)
