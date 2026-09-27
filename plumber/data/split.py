"""Leave-families-out split for the DecisionBench-OOD experiment.

  python -m plumber.data.split --rows data/decisionbench/rows.jsonl --out data/ood_v0 \
      --holdout routing_triage guardrails_moderation document_workflows risk_scoring triage content_moderation

Writes: train.jsonl (90% of every non-held-out family, shuffled), dev_in.jsonl (the other 10%: IN-FAMILY number),
        test_ood.jsonl (all rows of the held-out families: OOD number), split.json (manifest with counts + seed).
Rows longer than --max_len tokens are NOT dropped here (the trainer skips them); lengths are reported.
"""

import argparse
import collections
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", nargs="+", required=True)
    ap.add_argument("--dev_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20260926)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    by_fam = collections.defaultdict(list)
    for line in open(a.rows):
        by_fam[json.loads(line)["meta"]["family"]].append(line)
    missing = [f for f in a.holdout if f not in by_fam]
    assert not missing, f"unknown families: {missing}"
    tr, dv, te = [], [], []
    man = {"seed": a.seed, "holdout": a.holdout, "families": {}}
    for fam, lines in sorted(by_fam.items()):
        lines = list(lines)
        rng.shuffle(lines)
        if fam in a.holdout:
            te += lines
            man["families"][fam] = {"role": "test_ood", "n": len(lines)}
        else:
            k = max(1, int(len(lines) * a.dev_frac))
            dv += lines[:k]
            tr += lines[k:]
            man["families"][fam] = {"role": "train", "n_train": len(lines) - k, "n_dev": k}
    rng.shuffle(tr)
    rng.shuffle(dv)
    rng.shuffle(te)
    for name, rows in (("train", tr), ("dev_in", dv), ("test_ood", te)):
        with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
            f.writelines(rows)
    golds = collections.Counter(json.loads(line)["gold"] for line in tr)
    man["counts"] = {"train": len(tr), "dev_in": len(dv), "test_ood": len(te)}
    man["train_gold_index_hist"] = dict(sorted(golds.items()))
    json.dump(man, open(os.path.join(a.out, "split.json"), "w"), indent=1)
    print(
        json.dumps(man["counts"]),
        " train families:",
        sum(1 for f in man["families"].values() if f["role"] == "train"),
        " gold-index hist (top):",
        golds.most_common(5),
    )


if __name__ == "__main__":
    main()
