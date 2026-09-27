"""Build the v2 training mix: external rows (cap per task) + DecisionBench train rows (upsampled), shuffled with a fixed seed.

  python -m plumber.data.mix --ext data/ext_v2/rows_clean.jsonl --cap_per_task 400 --db data/ood_v0/train.jsonl --db_repeat 2 --out data/v2/train.jsonl
"""
import argparse, json, random, collections
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ext", required=True); ap.add_argument("--cap_per_task", type=int, default=400)
    ap.add_argument("--cap_per_dataset", type=int, default=1500, help="cap across all tasks of one dataset (task_id prefix), e.g. babi_nli/*")
    ap.add_argument("--db", required=True); ap.add_argument("--db_repeat", type=int, default=2); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=20260927)
    a = ap.parse_args(); rng = random.Random(a.seed); import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    by = collections.defaultdict(list)
    for l in open(a.ext): by[json.loads(l)["meta"]["task_id"]].append(l)
    ext = []; per_ds = collections.Counter()
    for t in sorted(by, key=lambda t: rng.random()):
        ls = by[t]; rng.shuffle(ls); ds = t.split("/")[0]
        take = min(a.cap_per_task, max(0, a.cap_per_dataset - per_ds[ds])); ext += ls[:take]; per_ds[ds] += take
    db = [l for l in open(a.db)]; rows = ext + db * a.db_repeat; rng.shuffle(rows)
    open(a.out, "w").writelines(rows)
    fam = collections.Counter(json.loads(l)["meta"].get("family") for l in rows)
    print(f"ext rows={len(ext)} from {len(by)} tasks / {len(per_ds)} datasets (cap {a.cap_per_task}/task, {a.cap_per_dataset}/dataset); db rows={len(db)}x{a.db_repeat}; total={len(rows)}; families={len(fam)}")
    print("top datasets:", per_ds.most_common(8))
if __name__ == "__main__": main()
