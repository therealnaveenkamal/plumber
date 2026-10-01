"""One table across plumbed models: each model alone and with its plumb, without and with thinking.

  python scripts/bench_report.py qwen3.5-9b=runs/bench_9b gemma-4-12b=runs/bench_gemma ...

Each directory holds a bench_s1_vs_s2.py summary.json. With thinking, the model alone is scored in its better setup
(prompt with or without a tool, whichever is more accurate), so the plumb is compared with the model at its best.
"""

from __future__ import annotations

import json
import os
import sys


def best_thinking(s: dict) -> dict | None:
    runs = [s[k] for k in ("s2_think", "s2_think_tool") if k in s]
    return max(runs, key=lambda r: r["accuracy"]) if runs else None


def pair(model: float, plumbed: float) -> str:
    """'model → plumbed' with the higher of the two in bold."""
    m, p = f"{model:.3f}", f"{plumbed:.3f}"
    return (
        f"{m} → **{p}**"
        if plumbed > model
        else f"**{m}** → {p}"
        if model > plumbed
        else f"{m} → {p}"
    )


def secs(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def main():
    print(
        "| Model | No thinking: model → plumbed | Thinking: model → plumbed | Thinking latency p50: model → plumbed |"
    )
    print("|---|---|---|---|")
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        s = json.load(open(os.path.join(path, "summary.json")))
        off_m, off_p = s["s2_fast"], s["s2_with_plumb_tool"]
        on_m, on_p = best_thinking(s), s.get("s2_with_plumb_think")
        row = f"| {name} | {pair(off_m['accuracy'], off_p['accuracy'])} |"
        if on_m and on_p:
            speed = on_m["latency_p50_ms"] / on_p["latency_p50_ms"]
            row += (
                f" {pair(on_m['accuracy'], on_p['accuracy'])} |"
                f" {secs(on_m['latency_p50_ms'])} → {secs(on_p['latency_p50_ms'])} ({speed:.1f}× faster) |"
            )
        else:
            row += " – | – |"
        print(row)


if __name__ == "__main__":
    main()
