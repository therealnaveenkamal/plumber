"""The comparison bar at the bottom of the side-by-side demo (tmux status line, refreshed every second).

Reads the per-turn timings both demo_chat.py panes write with --report and prints the current turn: each side's
time (ticking while it runs) and the speedup: a lower bound while the normal model is still running, the final figure
once both are done. The speedup is shown only for turns where the plumb made a decision; on other turns the
difference comes from thinking being on or off, not from the plumb. Two optional files in the same directory:
`times_only` hides the speedup, and `answer` holds the correct answer, shown once both sides are done.
"""

import json
import os
import sys
import time

NORMAL = "#[fg=colour203,bold]NORMAL MODEL#[default]"
PLUMB = "#[fg=colour114,bold]WITH PLUMB#[default]"


def load(path: str) -> dict[int, dict]:
    turns: dict[int, dict] = {}
    try:
        for line in open(path):
            row = json.loads(line)
            turns.setdefault(row["turn"], {}).update(row)
    except (OSError, ValueError):
        pass
    return turns


def side(t: dict | None) -> str:
    if not t:
        return "waiting"
    if "seconds" in t:
        return f"{t['seconds']:.1f} s"
    return f"{time.time() - t['start']:.0f} s…"


def main() -> None:
    d = sys.argv[1]
    normal, plumb = load(f"{d}/normal.jsonl"), load(f"{d}/plumb.jsonl")
    if not normal and not plumb:
        print("#[fg=colour245]same model, same message, same server#[default]")
        return
    n = max([*normal, *plumb])
    a, b = normal.get(n), plumb.get(n)
    line = f"turn {n}     {NORMAL}  {side(a)}     {PLUMB}  {side(b)}"
    if a and b and "seconds" in a and "seconds" in b and os.path.exists(f"{d}/answer"):
        line += f"     #[fg=colour221,bold]correct answer: {open(f'{d}/answer').read()}#[default]"
    if os.path.exists(f"{d}/times_only"):
        pass
    elif b and "seconds" in b and b.get("decisions") and a:
        if "seconds" in a:
            line += f"     #[fg=colour221,bold]{verdict(a['seconds'] / b['seconds'])}#[default]"
        else:  # the normal model is still running: the speedup so far is a lower bound
            so_far = (time.time() - a["start"]) / b["seconds"]
            if so_far >= SAME:
                line += f"     #[fg=colour221]≥ {so_far:.1f}× faster so far#[default]"
    print(line)


SAME = 1.15  # within this ratio either way, the two took about the same time


def verdict(ratio: float) -> str:
    if ratio >= SAME:
        return f"{ratio:.1f}× faster with the plumb"
    if ratio > 1 / SAME:
        return "about the same time"
    return f"{1 / ratio:.1f}× slower with the plumb"


if __name__ == "__main__":
    main()
