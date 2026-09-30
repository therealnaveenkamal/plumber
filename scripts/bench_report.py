"""One table across plumbed models: each model with and without its plumb (bench_s1_vs_s2.py summaries).

python scripts/bench_report.py qwen3.5-4b=runs/bench_native qwen3-1.7b=runs/bench_q3 ...
"""

from __future__ import annotations

import json
import os
import sys


def main():
    rows = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        s = json.load(open(os.path.join(path, "summary.json")))
        fast, think, s1, plumbed = (
            s.get(k) for k in ("s2_fast", "s2_think", "s1", "s2_with_plumb_tool")
        )
        best_base = max((m for m in (fast, think) if m), key=lambda m: m["accuracy"])
        rows.append(
            (
                name,
                fast["accuracy"],
                think["accuracy"] if think else None,
                s1["accuracy"],
                plumbed["accuracy"],
                plumbed["accuracy"] - best_base["accuracy"],
                think["latency_p50_ms"] / plumbed["latency_p50_ms"] if think else None,
                fast["latency_p50_ms"],
                think["latency_p50_ms"] if think else None,
                s1["latency_p50_ms"],
                plumbed["latency_p50_ms"],
            )
        )

    def f(x, fmt):
        return "–" if x is None else format(x, fmt)

    print(
        "| model | base, no thinking | base, thinking | System 1 only | **plumbed** | Δ vs best base | "
        "faster than thinking | p50 ms: base / thinking / S1 / plumbed |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r[0]} | {r[1]:.3f} | {f(r[2], '.3f')} | {r[3]:.3f} | **{r[4]:.3f}** | {r[5]:+.3f} | "
            f"{f(r[6], '.0f')}× | {r[7]:.0f} / {f(r[8], '.0f')} / {r[9]:.0f} / {r[10]:.0f} |"
        )


if __name__ == "__main__":
    main()
