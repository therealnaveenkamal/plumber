"""Reproduce the README benchmark row for a Plumb model: DecisionBench never-seen families, hard-skill templates, JevBench.

  python evals/bench.py --model totum-labs/plumb-nemotron-3.5-lightning-30b-a3b --out out/bench \\
      [--decisionbench data/decisionbench/test_ood.jsonl] [--hard data/kev_hard/test.jsonl] [--jevbench /path/to/jevbench]

Data:
  python -m plumber.data.decisionbench && python -m plumber.data.split     -> data/decisionbench/test_ood.jsonl (six held-out families)
  python -m plumber.data.kev --out data/kev_hard                             -> data/kev_hard/test.jsonl (held-out generator templates)
  git clone https://github.com/jevbench/jevbench                             -> --jevbench (public tiers, `typesafe` adapter)
Each part is skipped when its input is absent. Writes <out>/{decisionbench,hard,jevbench}/ and prints the table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decisionbench", default="data/decisionbench/test_ood.jsonl")
    ap.add_argument("--hard", default="data/kev_hard/test.jsonl")
    ap.add_argument("--jevbench", default=None, help="JevBench checkout; skipped when not given")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows: list[tuple[str, str, str]] = []

    if os.path.exists(a.decisionbench):
        out = os.path.join(a.out, "decisionbench")
        run(
            [
                sys.executable,
                os.path.join(HERE, "decisionbench.py"),
                "--model",
                a.model,
                "--rows",
                a.decisionbench,
                "--out",
                out,
            ]
        )
        r = json.load(open(os.path.join(out, "report.json")))
        macro = sum(v["accuracy"] for v in r["per_family"].values()) / len(r["per_family"])
        rows.append(
            (
                "DecisionBench, never-seen families",
                "accuracy, micro / macro",
                f"{r['overall']['accuracy']:.3f} / {macro:.3f}",
            )
        )
        rows.append(("", "ECE", f"{r['overall']['ece']:.3f}"))

    if os.path.exists(a.hard):
        out = os.path.join(a.out, "hard")
        run(
            [
                sys.executable,
                os.path.join(HERE, "decisionbench.py"),
                "--model",
                a.model,
                "--rows",
                a.hard,
                "--out",
                out,
            ]
        )
        r = json.load(open(os.path.join(out, "report.json")))
        rows.append(
            ("Hard skills, held-out templates", "accuracy", f"{r['overall']['accuracy']:.3f}")
        )

    if a.jevbench:
        out = os.path.join(a.out, "jevbench")
        run(["bash", os.path.join(HERE, "jevbench.sh"), a.model, a.jevbench, out])
        tiers = {}
        for t in ("easy", "original", "hard"):
            ids = {
                json.loads(line).get("id") or json.loads(line).get("task_id")
                for line in open(os.path.join(a.jevbench, "datasets", "public", f"{t}.jsonl"))
            }
            res = [json.loads(line) for line in open(os.path.join(out, "results.jsonl"))]
            got = [x for x in res if x.get("task_id") in ids]
            tiers[t] = sum(bool(x.get("correct")) for x in got) / max(len(got), 1)
        s = json.load(open(os.path.join(out, "summary.json")))
        p50 = next((v.get("p50_s") for v in _walk(s) if isinstance(v, dict) and "p50_s" in v), None)
        rows.append(("JevBench public, hard tier", "accuracy", f"{tiers['hard']:.3f}"))
        rows.append(
            (
                "JevBench public, easy / standard",
                "accuracy",
                f"{tiers['easy']:.3f} / {tiers['original']:.3f}",
            )
        )
        if p50:
            rows.append(("JevBench, server-side latency", "p50", f"{p50:.3f} s"))

    print("\n| Benchmark | Metric | " + a.model + " |\n|---|---|---:|")
    for b, m, v in rows:
        print(f"| {b} | {m} | {v} |")
    json.dump(
        {"model": a.model, "rows": rows}, open(os.path.join(a.out, "table.json"), "w"), indent=1
    )


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


if __name__ == "__main__":
    main()
