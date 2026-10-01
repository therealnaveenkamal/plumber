#!/usr/bin/env bash
# Gemma 4 12B: sliding/global attention, K=V global layers. 1x A100 80 GB, 10k rows, ~76 min of training.
# Measured, model alone → plumbed: no thinking 0.736 → 0.826; thinking 0.770 → 0.872 (24.1 s → 5.0 s p50).
# Needs DATA: a directory with train_10000.jsonl and dev.jsonl (plumb Row JSONL, see recipes/README.md).
: "${DATA:?set DATA to the directory holding train_10000.jsonl and dev.jsonl}"
ROWS=${ROWS:-$DATA/train_10000.jsonl} DEV=${DEV:-$DATA/dev.jsonl} TPB=32768 GPU_UTIL=0.7 \
    exec "$(dirname "$0")/plumbify_and_bench.sh" google/gemma-4-12B-it "${OUT:-plumbed-gemma-4-12b}" "$@"
