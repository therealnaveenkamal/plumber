#!/usr/bin/env bash
# Qwen3.5-9B: hybrid Gated DeltaNet + attention. 1x A100 80 GB, 10k rows, ~37 min of training.
# Measured: plumbed 0.779 | base 0.696 (thinking off), 0.772 (thinking on, 33 s p50).
# Needs DATA: a directory with train_10000.jsonl and dev.jsonl (plumb Row JSONL, see recipes/README.md).
: "${DATA:?set DATA to the directory holding train_10000.jsonl and dev.jsonl}"
ROWS=${ROWS:-$DATA/train_10000.jsonl} DEV=${DEV:-$DATA/dev.jsonl} TPB=32768 GPU_UTIL=0.7 \
    exec "$(dirname "$0")/plumbify_and_bench.sh" Qwen/Qwen3.5-9B "${OUT:-plumbed-qwen3.5-9b}" "$@"
