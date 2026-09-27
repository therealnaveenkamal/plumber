"""Build the plumb recipe: a balanced, size-capped training set for plumbifying any base LM.

  python -m plumber.data.recipe --size large --rows data/decisionbench/train.jsonl data/tasksource/rows.jsonl \\
      --hard data/kev_hard/train.jsonl --out data/recipe/large

Rows are grouped by task family (``meta.family``, else ``meta.dataset``, else the first segment of ``meta.task_id``),
deduplicated, shuffled with a fixed seed and capped per family, so no source dominates and the set stays small enough
to train in minutes. Hard-skill rows (``--hard``) are capped per skill separately. Structured states are rendered to
text. Writes ``train.jsonl``, ``dev.jsonl`` (a few rows per family, disjoint from train) and ``manifest.json``.

| size  | rows / family | hard rows / skill | ~rows  | Qwen3.5-4B, 1× A100 |
|-------|---------------|-------------------|--------|---------------------|
| small | 20            | 100               | ~5k    | ~5 min              |
| large | 60            | 300               | ~13k   | ~15 min             |
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random

from ..contract import state_text

SIZES = {
    "small": (20, 100, 3),
    "large": (60, 300, 5),
}  # rows/family, hard rows/skill, dev rows/family


def family_of(meta: dict) -> str:
    return meta.get("family") or meta.get("dataset") or str(meta.get("task_id", "?")).split("/")[0]


def _normalized(line: str) -> str:
    d = json.loads(line)
    d["state"] = state_text(d["state"])
    d["question"] = state_text(d["question"])
    return json.dumps(d, ensure_ascii=False) + "\n"


def group(paths: list[str]) -> dict[str, list[str]]:
    by: dict[str, list[str]] = collections.defaultdict(list)
    seen: set[str] = set()
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                line = _normalized(line)
                if line in seen:
                    continue
                seen.add(line)
                by[family_of(json.loads(line).get("meta", {}))].append(line)
    return by


def build(rows: list[str], hard: list[str], size: str, out: str, seed: int = 20260927) -> dict:
    cap, hard_cap, dev_n = SIZES[size]
    rng = random.Random(seed)
    train, dev, manifest = [], [], {"size": size, "seed": seed, "families": {}, "skills": {}}
    for label, paths, k in (("families", rows, cap), ("skills", hard, hard_cap)):
        for fam, lines in sorted(group(paths).items()):
            rng.shuffle(lines)
            take, rest = lines[:k], lines[k:]
            d = (
                rest[:dev_n] if rest else take[-1:]
            )  # tiny families: dev shares a row rather than none
            train += take
            dev += d
            manifest[label][fam] = {"train": len(take), "dev": len(d), "available": len(lines)}
    rng.shuffle(train)
    rng.shuffle(dev)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "train.jsonl"), "w") as f:
        f.writelines(train)
    with open(os.path.join(out, "dev.jsonl"), "w") as f:
        f.writelines(dev)
    manifest["train_rows"], manifest["dev_rows"] = len(train), len(dev)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(SIZES), default="large")
    ap.add_argument(
        "--rows",
        nargs="+",
        required=True,
        help="converted row files (DecisionBench, tasksource, ...)",
    )
    ap.add_argument("--hard", nargs="*", default=[], help="hard-skill row files (plumber.data.kev)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260927)
    a = ap.parse_args()
    m = build(a.rows, a.hard, a.size, a.out, a.seed)
    print(
        f"recipe {a.size}: train={m['train_rows']} dev={m['dev_rows']} "
        f"families={len(m['families'])} skills={len(m['skills'])} -> {a.out}"
    )


if __name__ == "__main__":
    main()
