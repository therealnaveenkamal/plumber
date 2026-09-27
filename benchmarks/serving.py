"""Serving benchmark: latency per request as the number of questions per state grows (prefix caching on/off).

python benchmarks/serving.py --model totum-labs/plumb-nemotron-3.5-lightning-30b-a3b --questions 1 4 16 64
"""

from __future__ import annotations

import argparse
import statistics
import time

from plumber import Plumber

STATE = (
    "Invoice #4411 was billed twice for March. The customer asks for a refund today or they cancel. "
    "Account opened 2021, enterprise tier, previous tickets: 3 (all billing). Contract renews in 40 days."
)


def question(i: int) -> dict:
    return {
        "type": "choice",
        "instructions": f"Question {i}: which team should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages",
            "sales": "pricing",
            "legal": "contracts",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    engine = Plumber(a.model)
    print(f"{'questions':>9} {'shared prefix':>14} {'full':>10}  (p50 ms per request)")
    for n in a.questions:
        qs = {f"q{i}": question(i) for i in range(n)}
        engine.decide(STATE, qs)  # warm-up
        t_shared, t_full = [], []
        for _ in range(a.repeats):
            t = time.perf_counter()
            engine.decide(STATE, qs, share_prefix=True)
            t_shared.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            engine.decide(STATE, qs, share_prefix=False)
            t_full.append((time.perf_counter() - t) * 1000)
        print(f"{n:9d} {statistics.median(t_shared):14.1f} {statistics.median(t_full):10.1f}")


if __name__ == "__main__":
    main()
