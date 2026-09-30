#!/usr/bin/env bash
# Qwen3.5-27B: hybrid Gated DeltaNet + attention, 56 GB of weights. 1x A100 80 GB, 10k rows.
# Smaller training batches and a high serving memory fraction to fit the trunk on one card.
# Needs DATA: a directory with train_10000.jsonl and dev.jsonl (plumb Row JSONL, see recipes/README.md).
: "${DATA:?set DATA to the directory holding train_10000.jsonl and dev.jsonl}"
ROWS=${ROWS:-$DATA/train_10000.jsonl} DEV=${DEV:-$DATA/dev.jsonl} TPB=16384 GPU_UTIL=0.92 \
    exec "$(dirname "$0")/plumbify_and_bench.sh" Qwen/Qwen3.5-27B "${OUT:-plumbed-qwen3.5-27b}" "$@"
