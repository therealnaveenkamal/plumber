#!/usr/bin/env bash
# Qwen3.5-4B: hybrid Gated DeltaNet + attention. 1x A100 80 GB, 20k rows, ~54 min of training.
# Measured: plumbed 0.801 | base 0.667 (thinking off), 0.720 (thinking on, 29 s p50).
# Needs DATA: a directory with train_20000.jsonl and dev.jsonl (plumb Row JSONL, see recipes/README.md).
: "${DATA:?set DATA to the directory holding train_20000.jsonl and dev.jsonl}"
ROWS=${ROWS:-$DATA/train_20000.jsonl} DEV=${DEV:-$DATA/dev.jsonl} TPB=32768 GPU_UTIL=0.7 \
    exec "$(dirname "$0")/plumbify_and_bench.sh" Qwen/Qwen3.5-4B "${OUT:-plumbed-qwen3.5-4b}" "$@"
