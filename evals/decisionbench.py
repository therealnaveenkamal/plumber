"""Evaluate a Plumb model on DecisionBench rows through the engine (no server), grouped by family.

python evals/decisionbench.py --model totum-labs/plumb-nemotron-3.5-lightning-30b-a3b --rows data/ood_v0/test_ood.jsonl --out out/decisionbench
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time

from plumber import Plumber
from plumber.core import Row
from plumber.metrics import summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    engine = Plumber(a.model)
    preds: list[dict] = []
    t0 = time.time()
    with open(a.rows) as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            r = Row.from_json(line)
            crit = (
                {o.name: o.desc for o in r.options}
                if r.qtype == "choice"
                else ([o.desc or o.name for o in r.options] if r.qtype == "score" else None)
            )
            ans = engine.decide(
                r.state, {"q": {"type": r.qtype, "instructions": r.question, "criteria": crit}}
            )["answers"]["q"]
            probs = [ans["probabilities"][o.name] for o in r.options]
            pred = max(range(len(probs)), key=lambda k: probs[k])
            preds.append(
                {
                    "id": r.id,
                    "family": r.meta.get("family"),
                    "gold": r.gold,
                    "pred": pred,
                    "correct": int(pred == r.gold) if r.gold is not None else None,
                    "p_max": max(probs),
                    "p_gold": probs[r.gold] if r.gold is not None else None,
                    "probs": probs,
                }
            )
    by_family = collections.defaultdict(list)
    for p in preds:
        by_family[p["family"]].append(p)
    report = {
        "overall": summarize(preds),
        "per_family": {k: summarize(v) for k, v in by_family.items()},
        "seconds": time.time() - t0,
    }
    with open(os.path.join(a.out, "preds.jsonl"), "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(json.dumps(report["overall"], indent=1))
    for k, v in sorted(report["per_family"].items(), key=lambda kv: -kv[1]["n"]):
        print(
            f"  {k:32} n={v['n']:5d} acc={v['accuracy']:.4f} nll={v['nll']:.3f} ece={v['ece']:.3f}"
        )


if __name__ == "__main__":
    main()
