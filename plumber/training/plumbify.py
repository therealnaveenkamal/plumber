"""Plumbify an open model: train its plumb and write a plumbed model directory that vLLM serves natively.

  plumber plumbify --base Qwen/Qwen3.5-9B --rows train.jsonl --dev dev.jsonl --out plumbed-qwen3.5-9b
  vllm serve plumbed-qwen3.5-9b

Steps: (1) train the decision head and the suffix-only LoRA on the frozen base (``plumber train-head
--suffix_lora``), (2) calibrate on --dev (temperature and a conformal threshold), (3) package the plumbed model
directory (plumber/plumbed.py). The base weights are never modified.
"""

from __future__ import annotations

import argparse
import json
import os

from ..plumbed import package
from . import train_head


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="any Hugging Face causal / image-text LM")
    ap.add_argument(
        "--rows", default=os.environ.get("PLUMBER_ROWS"), help="training rows (plumb Row JSONL)"
    )
    ap.add_argument(
        "--dev", default=os.environ.get("PLUMBER_DEV"), help="held-out rows for calibration"
    )
    ap.add_argument("--out", required=True, help="the plumbed model directory to write")
    ap.add_argument("--work", default=None, help="training directory (default: <out>.train)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument(
        "--no_lora", action="store_true", help="head only (no suffix LoRA): cheaper, weaker"
    )
    ap.add_argument(
        "--copy", action="store_true", help="copy the base weights instead of linking them"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="train on the first N rows (smoke runs)"
    )
    ap.add_argument("--tokens_per_batch", type=int, default=32768)
    a, extra = ap.parse_known_args(argv)
    if not a.rows:
        ap.error("--rows is required (or set PLUMBER_ROWS)")
    work = a.work or f"{a.out.rstrip('/')}.train"
    args = [
        "--base",
        a.base,
        "--rows",
        a.rows,
        "--out",
        work,
        "--epochs",
        str(a.epochs),
        "--tokens_per_batch",
        str(a.tokens_per_batch),
        "--max_rows",
        "32",
        *extra,
    ]
    if a.dev:
        args += ["--dev", a.dev]
    if not a.no_lora:
        args += ["--suffix_lora", "--lora_r", str(a.lora_r), "--lora_alpha", str(2 * a.lora_r)]
    if a.limit:
        args += ["--limit", str(a.limit)]
    print(f"[plumbify] 1/2 train + calibrate: {' '.join(args)}", flush=True)
    train_head.main(args)
    metrics_path = os.path.join(work, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else None
    print(f"[plumbify] 2/2 package -> {a.out}", flush=True)
    package(
        os.path.join(work, "final"),
        a.out,
        a.base,
        copy=a.copy,
        metrics={
            k: metrics[k] for k in ("dev_calibrated", "temperature", "conformal") if k in metrics
        }
        if metrics
        else None,
    )
    print(f"[plumbify] done. Serve it with:\n  vllm serve {a.out}", flush=True)


if __name__ == "__main__":
    main()
