#!/usr/bin/env bash
# Qwen3-1.7B: dense attention. 1x A100 80 GB, 20k rows, ~23 min of training.
# Measured, model alone → plumbed: no thinking 0.570 → 0.658; thinking 0.642 → 0.707 (3.9 s → 3.1 s p50).
# Needs DATA: a directory with train_20000.jsonl and dev.jsonl (plumb Row JSONL, see recipes/README.md).
: "${DATA:?set DATA to the directory holding train_20000.jsonl and dev.jsonl}"
ROWS=${ROWS:-$DATA/train_20000.jsonl} DEV=${DEV:-$DATA/dev.jsonl} TPB=32768 GPU_UTIL=0.7 \
    exec "$(dirname "$0")/plumbify_and_bench.sh" Qwen/Qwen3-1.7B "${OUT:-plumbed-qwen3-1.7b}" "$@"
