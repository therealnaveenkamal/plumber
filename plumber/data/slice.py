"""Slice row files by metadata: build domain sets (finance, medicine, ...) from any converted rows.

  python -m plumber.data.slice --rows a.jsonl b.jsonl --match "finan|econom" --out finance.jsonl --limit 2000 --holdout 0.1

``--match`` is a regex tested against the row's JSON-encoded ``meta`` (task_id, family, dataset, ...); ``--exclude`` drops
matches; ``--holdout`` writes that fraction to ``<out>.holdout.jsonl``. Rows are shuffled with ``--seed`` before capping.
"""

from __future__ import annotations

import argparse
import json
import random
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--match", required=True, help="regex on json.dumps(meta)")
    ap.add_argument("--exclude", default=None, help="regex on json.dumps(meta); drop matches")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--holdout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260927)
    a = ap.parse_args()
    keep_re = re.compile(a.match, re.I)
    drop_re = re.compile(a.exclude, re.I) if a.exclude else None
    lines = []
    for path in a.rows:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                meta = json.dumps(json.loads(line).get("meta", {}), ensure_ascii=False)
                if keep_re.search(meta) and not (drop_re and drop_re.search(meta)):
                    lines.append(line)
    random.Random(a.seed).shuffle(lines)
    n_hold = int(len(lines) * a.holdout)
    hold, train = lines[:n_hold], lines[n_hold:]
    if a.limit:
        train = train[: a.limit]
    with open(a.out, "w") as f:
        f.writelines(train)
    if hold:
        with open(a.out.replace(".jsonl", "") + ".holdout.jsonl", "w") as f:
            f.writelines(hold)
    print(f"matched={len(lines)} train={len(train)} holdout={len(hold)} -> {a.out}")


if __name__ == "__main__":
    main()
