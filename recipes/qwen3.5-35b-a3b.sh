#!/usr/bin/env bash
# Qwen3.5-35B-A3B: MoE (256 experts, 3B active) + Gated DeltaNet, 72 GB of weights. 1x A100 80 GB, 10k rows,
# ~37 min of training. The adapter covers DeltaNet + attention; experts stay untouched.
# Measured: plumbed 0.810 | base 0.720 (thinking off), 0.787 (thinking on, 20 s p50).
# Needs DATA: a directory with train_10000.jsonl and dev.jsonl (plumb Row JSONL, see recipes/README.md).
: "${DATA:?set DATA to the directory holding train_10000.jsonl and dev.jsonl}"
ROWS=${ROWS:-$DATA/train_10000.jsonl} DEV=${DEV:-$DATA/dev.jsonl} TPB=8192 GPU_UTIL=0.95 MAX_LEN=16384 MAX_NUM_SEQS=32 \
    exec "$(dirname "$0")/plumbify_and_bench.sh" Qwen/Qwen3.5-35B-A3B "${OUT:-plumbed-qwen3.5-35b-a3b}" "$@"
