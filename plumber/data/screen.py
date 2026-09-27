"""Contamination screen: drop candidate rows whose state matches any held-out eval row (exact normalized text, or
8-gram Jaccard >= 0.2 via MinHash-free banded shingles). Cheap and conservative.

  python -m plumber.data.screen --candidates data/ext_v2/rows.jsonl --heldout data/ood_v0/test_ood.jsonl --out data/ext_v2/rows_clean.jsonl
"""

import argparse
import collections
import hashlib
import json
import re


def _txt(s):
    if isinstance(s, str):
        return s
    if isinstance(s, dict):
        return "\n".join(
            f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
            for k, v in s.items()
        )
    return json.dumps(s, ensure_ascii=False)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", _txt(s).lower())).strip()


def shingles(s, n=8):
    w = norm(s).split()
    return {
        hashlib.md5(" ".join(w[i : i + n]).encode()).hexdigest()[:12]
        for i in range(max(0, len(w) - n + 1))
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jaccard", type=float, default=0.2)
    a = ap.parse_args()
    exact = set()
    index = collections.defaultdict(set)
    held_sh = {}
    for i, line in enumerate(open(a.heldout)):
        s = json.loads(line)["state"]
        exact.add(norm(s))
        sh = shingles(s)
        held_sh[i] = sh
        for g in sh:
            index[g].add(i)
    kept = dropped_exact = dropped_ngram = 0
    dropped_ids = []
    with open(a.out, "w") as f:
        for line in open(a.candidates):
            r = json.loads(line)
            s = r["state"]
            if norm(s) in exact:
                dropped_exact += 1
                dropped_ids.append(r["id"])
                continue
            sh = shingles(s)
            hit = False
            if sh:
                cand = collections.Counter(j for g in sh for j in index.get(g, ()))
                for j, c in cand.most_common(3):
                    if c / len(sh | held_sh[j]) >= a.jaccard:
                        hit = True
                        break
            if hit:
                dropped_ngram += 1
                dropped_ids.append(r["id"])
                continue
            f.write(line)
            kept += 1
    json.dump(
        {
            "kept": kept,
            "dropped_exact": dropped_exact,
            "dropped_ngram": dropped_ngram,
            "dropped_ids": dropped_ids[:2000],
        },
        open(a.out + ".screen.json", "w"),
        indent=1,
    )
    print(f"kept={kept} dropped_exact={dropped_exact} dropped_ngram={dropped_ngram}")


if __name__ == "__main__":
    main()
