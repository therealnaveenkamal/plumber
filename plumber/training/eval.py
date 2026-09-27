"""Evaluate a trained plumb checkpoint (LoRA adapter dir + head.pt) on a rows.jsonl.

  python -m plumber.training.eval --ckpt runs/v0/step800 --rows data/ood_v0/test_ood.jsonl --out runs/v0/eval_ood
Writes preds.jsonl (one line per row) and metrics.json (accuracy, NLL, Brier, ECE, coverage at 5% error; per-family, per-primitive).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time

import torch

from ..core.rendering import render
from ..core.trunk import PlumbModel, collate
from ..metrics import summarize
from .train import load_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--d_proj", type=int, default=512)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    model = PlumbModel.load(
        a.ckpt, base=a.model, d_proj=a.d_proj, dtype=torch.bfloat16, device_map="auto"
    )  # adapter dir OR merged release
    rows = load_rows(a.rows, a.limit)
    in_dev = next(model.trunk.parameters()).device
    preds, skipped, t0 = [], 0, time.time()
    with torch.no_grad():
        for i in range(0, len(rows), a.bsz):
            chunk = rows[i : i + a.bsz]
            rend = [render(tok, r, rng=None, shuffle=False) for r in chunk]
            keep = [
                (r, x)
                for r, x in zip(chunk, rend, strict=False)
                if len(x["input_ids"]) <= a.max_len
            ]
            skipped += len(rend) - len(keep)
            if not keep:
                continue
            b = collate(tok, [x for _, x in keep], pad_id, in_dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b["input_ids"], b["attention_mask"], b["opt_spans"], b["decide_pos"])
            p = torch.softmax(logits.float(), -1)
            for j, (r, _x) in enumerate(keep):
                K = len(r.options)
                pj = p[j, :K]
                pred = int(pj.argmax())
                m = r.meta
                preds.append(
                    {
                        "id": r.id,
                        "family": m.get("family"),
                        "task_id": m.get("task_id"),
                        "primitive": m.get("primitive"),
                        "K": K,
                        "gold": r.gold,
                        "pred": pred,
                        "correct": int(pred == r.gold) if r.gold is not None else None,
                        "p_max": float(pj.max()),
                        "p_gold": float(pj[r.gold]) if r.gold is not None else None,
                        "probs": [round(float(v), 6) for v in pj.tolist()],
                    }
                )
            if (i // a.bsz) % 50 == 0:
                print(f"[eval] {len(preds)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out, "preds.jsonl"), "w") as f:
        for q in preds:
            f.write(json.dumps(q) + "\n")
    scored = [q for q in preds if q["correct"] is not None]

    def group(key):
        d = collections.defaultdict(list)
        for q in scored:
            d[q[key]].append(q)
        return {k: _stats(v) for k, v in d.items()}

    fam = group("family")
    prim = group("primitive")
    metrics = {
        **_stats(scored),
        "skipped_too_long": skipped,
        "macro_family_acc": sum(v["acc"] for v in fam.values()) / max(len(fam), 1),
        "per_family": fam,
        "per_primitive": prim,
        "seconds": time.time() - t0,
    }
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(
        json.dumps(
            {k: v for k, v in metrics.items() if k not in ("per_family", "per_primitive")}, indent=1
        )
    )
    for k, v in sorted(fam.items(), key=lambda kv: -kv[1]["n"]):
        print(
            f"  {k:32} n={v['n']:5d} acc={v['acc']:.4f} brier={v['brier']:.3f} nll={v['nll']:.3f} ece={v['ece']:.3f} cov@5%={v['coverage_at_5pct_error']:.3f}"
        )


def _stats(rows):
    m = summarize(rows)
    m["acc"] = m.pop("accuracy", 0.0)
    m["micro_acc"] = m["acc"]
    return m


if __name__ == "__main__":
    main()
