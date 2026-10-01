"""The comparison bar at the bottom of the side-by-side demo (tmux status line, refreshed every second).

Reads the per-turn timings both demo_chat.py panes write with --report and prints the current turn: each side's
time (ticking while it runs) and the speedup: a lower bound while the normal model is still running, the final figure
once both are done. The speedup is shown only for turns where the plumb made a decision; on other turns the
difference comes from thinking being on or off, not from the plumb.
"""

import json
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
    if b and "seconds" in b and b.get("decisions") and a:
        if "seconds" in a:
            line += f"     #[fg=colour221,bold]{a['seconds'] / b['seconds']:.0f}× faster with the plumb#[default]"
        else:  # the normal model is still running: the speedup so far is a lower bound
            so_far = (time.time() - a["start"]) / b["seconds"]
            line += f"     #[fg=colour221]≥ {so_far:.0f}× faster so far#[default]"
    print(line)


if __name__ == "__main__":
    main()
