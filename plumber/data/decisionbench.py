"""DecisionBench (Hanno-Labs/decision-bench, canonical `eval` parquet, 23,900 rows) -> plumb Rows,
plus the leave-family-out OOD folds and a source-kind summary for the contamination denylist.

  python -m plumber.data.decisionbench --parquet decisionbench_eval.parquet --out data/decisionbench
    -> data/decisionbench/rows.jsonl          one Row per line (all 23,900)
       data/decisionbench/ood_folds.json      {family: [row_id]} + grouping + protocol
       data/decisionbench/sources.json        source_json kinds / names for the denylist
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from ..core.rendering import Option, Row


def render_state(state_json: str) -> str:
    """Deterministic 'key: value' rendering of the state dict (values JSON-dumped when not str)."""
    try:
        st = json.loads(state_json)
    except Exception:
        return state_json
    if isinstance(st, dict):
        return "\n".join(
            f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
            for k, v in st.items()
        )
    return json.dumps(st, ensure_ascii=False)


def parse_probs(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in x]
    s = str(x).strip("[] \n")
    return [float(v) for v in re.split(r"[\s,]+", s) if v]


def to_row(r: dict) -> Row:
    cands = json.loads(r["candidates_json"])
    if r["primitive"] == "ordinal_scoring":
        cands = sorted(
            cands, key=lambda c: c.get("ordinal_value") if c.get("ordinal_value") is not None else 0
        )
    options = [Option(str(c["label"]), str(c.get("description") or "")) for c in cands]
    ids = [c["id"] for c in cands]
    gold = ids.index(r["gold_candidate_id"]) if r["gold_candidate_id"] in ids else None
    qtype = {
        "binary_classification": "noul" if len(cands) == 2 else "choice",
        "candidate_selection": "choice",
        "ordinal_scoring": "score",
    }[r["primitive"]]
    probs = parse_probs(r.get("gold_probabilities"))
    if (
        probs is not None and r["primitive"] == "ordinal_scoring"
    ):  # reorder to the sorted level order
        orig = json.loads(r["candidates_json"])
        order = [orig.index(c) for c in cands]
        probs = [probs[i] for i in order] if len(probs) == len(orig) else probs
    return Row(
        id=r["row_id"],
        state=render_state(r["state_json"]),
        question=r["instruction"],
        qtype=qtype,
        options=options,
        gold=gold,
        teacher=None,
        meta={
            "task_id": r["task_id"],
            "family": r["family"],
            "domain": r["domain"],
            "primitive": r["primitive"],
            "candidate_count": int(r["candidate_count"]),
            "reasoning_required": bool(r["reasoning_required"]),
            "reasoning_type": None if pd.isna(r.get("reasoning_type")) else r.get("reasoning_type"),
            "gold_probs": probs,
            "candidate_ids": ids,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_parquet(a.parquet)
    fam_rows, kinds, names, n_bad = defaultdict(list), Counter(), Counter(), 0
    with open(os.path.join(a.out, "rows.jsonl"), "w") as f:
        for r in df.to_dict("records"):
            row = to_row(r)
            if row.gold is None:
                n_bad += 1
            f.write(row.to_json() + "\n")
            fam_rows[r["family"]].append(r["row_id"])
            try:
                sj = json.loads(r["source_json"])
                kinds[sj.get("kind")] += 1
                for k in (
                    "source",
                    "dataset",
                    "source_dataset",
                    "name",
                    "origin",
                    "upstream",
                    "hf_id",
                ):
                    if k in sj and isinstance(sj[k], str):
                        names[f"{k}={sj[k]}"] += 1
            except Exception:
                pass
    big = sorted([f for f, v in fam_rows.items() if len(v) >= 2000])
    small = sorted([f for f in fam_rows if f not in big])
    folds = {
        "protocol": {
            "leave_one_family_out": "for each family F: train on all rows NOT in F (and never on F's source datasets), test on F",
            "train_set_declared": "for a published checkpoint, score only the families its card/data admit it did not train on",
            "report": "macro over families AND micro over rows; in-family number reported beside OOD, never merged",
        },
        "families": {f: sorted(v) for f, v in fam_rows.items()},
        "family_sizes": {f: len(v) for f, v in fam_rows.items()},
        "groups": {"core9_2k_each": big, "small18": small},
        "primitive_by_family": {f: dict(Counter(df[df.family == f].primitive)) for f in fam_rows},
    }
    json.dump(folds, open(os.path.join(a.out, "ood_folds.json"), "w"), indent=1)
    json.dump(
        {"source_kinds": dict(kinds), "source_names_top": dict(names.most_common(60))},
        open(os.path.join(a.out, "sources.json"), "w"),
        indent=1,
    )
    print(f"rows={len(df)}  families={len(fam_rows)}  rows_without_gold={n_bad}  core9={big}")
    print("source kinds:", dict(kinds))
    print("source names (top 15):", names.most_common(15))


if __name__ == "__main__":
    main()
