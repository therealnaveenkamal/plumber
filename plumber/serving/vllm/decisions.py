"""Find decisions that a conversation already poses, and map a model's tool-call options back onto them.

When a user message states a decision (a question followed by a bulleted option list), System 1 should see it exactly
as written: a model restating it in a tool call shortens descriptions, drops or renames options, or skips the call
altogether, and each of those costs accuracy. ``find_decision`` extracts the stated decision; ``resolve`` maps a tool
call's options onto the stated list (exact, then case-insensitive name, then "name: description" prefix).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$")
_LEAD = re.compile(r"^\s*(options|choices|levels[^:]*)\s*:\s*$", re.I)


@dataclass
class Stated:
    question: str
    options: list[tuple[str, str]]  # (name, description)
    qtype: str
    before: str = ""  # the message text before the question: the state the decision is about


def _split(item: str) -> tuple[str, str]:
    name, sep, desc = item.partition(": ")
    return (name.strip(), desc.strip()) if sep else (item.strip(), "")


def _qtype(names: list[str]) -> str:
    if [n.lower() for n in names] == ["no", "yes"]:
        return "noul"
    if names == [str(i) for i in range(len(names))]:
        return "score"
    return "choice"


def find_decision(text: str) -> Stated | None:
    """The last question + option list in ``text``: a ``Decision: ...`` line (or a line ending in '?') followed,
    possibly after an ``Options:`` line, by at least two bullet items."""
    lines = text.splitlines()
    best = None
    i = 0
    while i < len(lines):
        m = re.match(r"^\s*Decision:\s*(.+)$", lines[i]) or (
            re.match(r"^\s*(.+\?)\s*$", lines[i]) if not _BULLET.match(lines[i]) else None
        )
        if not m:
            i += 1
            continue
        j = i + 1
        if j < len(lines) and _LEAD.match(lines[j]):
            j += 1
        items = []
        while j < len(lines) and _BULLET.match(lines[j]):
            items.append(_split(_BULLET.match(lines[j]).group(1)))
            j += 1
        names = [n for n, _ in items]
        if len(items) >= 2 and len(set(names)) == len(names):
            best = Stated(m.group(1).strip(), items, _qtype(names), "\n".join(lines[:i]).strip())
        i = max(j, i + 1)
    return best


def _key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip("*`'\"").lower())


def _matches(name: str, stated_name: str, stated_desc: str) -> bool:
    k, n = _key(name), _key(stated_name)
    full = _key(f"{stated_name}: {stated_desc}") if stated_desc else n
    return (
        k == n
        or k == full
        # "Low" for "Low: Reversible edit to local draft"
        or (len(k) >= 3 and full.startswith(k))
        # "contractor: independent contractor" for "contractor"
        or (len(n) >= 3 and k.startswith(n + ":"))
    )


def resolve(names: list[str], stated: Stated | None) -> list[tuple[str, str]] | None:
    """The stated option list when the tool call's option names refer to it (at least half of them, and at least two,
    match a stated option); None when the call is about something else."""
    if stated is None or not names:
        return None
    hit = sum(any(_matches(nm, n, d) for n, d in stated.options) for nm in names)
    return stated.options if hit >= max(2, (len(names) + 1) // 2) else None


def pick(text: str, names: list[str]) -> str | None:
    """The option a reply commits to: its last 'Answer: X' line, matched to an option name; None if it names none."""
    text = text.split("</think>")[-1]
    norm = {_key(n): n for n in names}
    for m in reversed(re.findall(r"Answer:\s*\**\s*([^\n*]+)", text)):
        cand = _key(m.strip().strip(".`'\" "))
        if cand in norm:
            return norm[cand]
        hits = [n for k, n in norm.items() if re.search(rf"(^|\W){re.escape(k)}(\W|$)", cand)]
        if len(hits) == 1:
            return hits[0]
    return None
