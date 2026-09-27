"""Breadth data: every tasksource classification task -> plumb Rows, denylist-filtered, capped per task.

  python -m plumber.data.tasksource --out data/ext_v2 --cap 1500 --workers 16
Writes data/ext_v2/rows.jsonl, data/ext_v2/manifest.json (per task: n, labels, license note, skipped reason).
Label names are read as text by the model (label-as-text), so no per-task templates are needed; the question is generic.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

DENY = re.compile(
    r"banking77|clinc|civil_comments|jigsaw|toxigen|hatexplain|olid|hatecheck|legalbench|cuad|contractnli|ledgar|casehold|"
    r"esci|wands|msmarco|ms_marco|beir|toolace|glaive|xlam|bfcl|big_patent|nemotron.?pii|entity.?resolution|musique|"
    r"finqa|tat.?qa|financebench|fiqa|folio|logiqa|reclor|prontoqa|strategyqa|qasc|bamboogle|massive|slurp|hwu|"
    r"nlu_evaluation|snips|ag_news|dair|emotion|banking|hint3|bitext|20_?newsgroups|bbc|huffpost|reuters|mind",
    re.I,
)


def gen_question(task_id, task_name, label_names):
    base = task_name or task_id
    base = re.sub(r"[_/]+", " ", base).strip()
    return f"Task: {base}. Read the input and pick the label that applies."


def _load_with_retry(task_id, cap, attempts=3):
    import time

    from tasksource import load_task

    last = None
    for k in range(attempts):
        try:
            return (
                load_task(task_id, max_rows=cap * 3, max_rows_eval=0)
                if "max_rows" in load_task.__code__.co_varnames
                else load_task(task_id)
            )
        except Exception as e:
            last = e
            msg = str(e)
            if "429" in msg or "HfHubHTTPError" in type(e).__name__ or "rate" in msg.lower():
                time.sleep(25 * (2**k))
            else:
                raise
    raise last


def convert_one(task_id, cap, seed):
    try:
        ds = _load_with_retry(task_id, cap)
        if hasattr(ds, "keys"):
            ds = ds["train"] if "train" in ds else ds[next(iter(ds.keys()))]
        feats = ds.features
        if "labels" not in feats:
            return task_id, None, "no labels column"
        lab = feats["labels"]
        names = getattr(lab, "names", None)
        if not names or len(names) < 2 or len(names) > 255:
            return task_id, None, f"labels not classlabel or K={len(names) if names else 0}"
        text_cols = [
            c for c in ds.column_names if c != "labels" and str(feats[c].dtype) == "string"
        ]
        if not text_cols:
            return task_id, None, "no text columns"
        rng = random.Random(seed)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        idx = idx[:cap]
        rows = []
        q = gen_question(task_id, task_id, names)
        for i in idx:
            ex = ds[i]
            y = ex["labels"]
            if not isinstance(y, int) or y < 0 or y >= len(names):
                continue
            state = (
                "\n".join(f"{c}: {ex[c]}" for c in text_cols if ex[c])
                if len(text_cols) > 1
                else str(ex[text_cols[0]])
            )
            if not state.strip() or len(state) > 20000:
                continue
            rows.append(
                {
                    "id": f"ts:{task_id}:{i}",
                    "state": state,
                    "question": q,
                    "qtype": "noul"
                    if len(names) == 2
                    and set(n.lower() for n in names)
                    <= {"yes", "no", "true", "false", "0", "1", "entailment", "not_entailment"}
                    else "choice",
                    "options": [{"name": str(n), "desc": ""} for n in names],
                    "gold": y,
                    "teacher": None,
                    "meta": {
                        "source": "tasksource",
                        "task_id": task_id,
                        "family": f"ts/{task_id.split('/')[0]}",
                        "K": len(names),
                    },
                }
            )
        return task_id, rows, None
    except Exception as e:
        return task_id, None, f"{type(e).__name__}: {str(e)[:160]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit_tasks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument(
        "--retry_manifest",
        default=None,
        help="only re-run tasks this manifest marks as skipped with an HTTP error; append to --out",
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from tasksource import list_tasks

    df = list_tasks()
    df = df[df["task_type"].astype(str).str.contains("Classification", case=False)]
    ids = [
        t
        for t in df["id"].tolist()
        if not DENY.search(t) and not DENY.search(str(df[df.id == t].dataset_name.iloc[0]))
    ]
    denied = [t for t in df["id"].tolist() if t not in ids]
    if a.retry_manifest:
        prev = json.load(open(a.retry_manifest))["tasks"]
        ids = [
            t
            for t in ids
            if t in prev and not prev[t].get("n") and "HTTP" in str(prev[t].get("skipped", ""))
        ]
        print(f"retrying {len(ids)} HTTP-failed tasks", flush=True)
    if a.limit_tasks:
        ids = ids[: a.limit_tasks]
    print(
        f"classification tasks: {len(df)}  after denylist: {len(ids)}  denied: {len(denied)}",
        flush=True,
    )
    man = {"denied": denied, "tasks": {}}
    n_rows = 0
    n_ok = 0
    mode = "a" if a.retry_manifest else "w"
    with open(os.path.join(a.out, "rows.jsonl"), mode) as f, ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(convert_one, t, a.cap, a.seed): t for t in ids}
        for k, fu in enumerate(as_completed(futs), 1):
            tid, rows, err = fu.result()
            if rows:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_rows += len(rows)
                n_ok += 1
                man["tasks"][tid] = {
                    "n": len(rows),
                    "K": rows[0]["meta"]["K"],
                    "labels": [o["name"] for o in rows[0]["options"]][:12],
                }
            else:
                man["tasks"][tid] = {"n": 0, "skipped": err}
            if k % 25 == 0:
                print(f"[{k}/{len(ids)}] tasks ok={n_ok} rows={n_rows}", flush=True)
    if a.retry_manifest:
        prev = json.load(open(a.retry_manifest))
        prev["tasks"].update(man["tasks"])
        man = prev
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"DONE tasks ok={n_ok}/{len(ids)} rows={n_rows}", flush=True)


if __name__ == "__main__":
    main()
