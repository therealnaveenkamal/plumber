#!/usr/bin/env bash
# Plumbify an open model, check the vLLM plugin against transformers, then serve it with a plain `vllm serve` and
# benchmark it with and without its plumb.
#
#   ROWS=train.jsonl DEV=dev.jsonl [EVAL_DATA=decision-data] recipes/plumbify_and_bench.sh <base model> <out dir> \
#       [extra `vllm serve` args]
#
# Settings (environment):
#   ROWS, DEV        training rows and held-out calibration rows (plumb Row JSONL), required
#   EVAL_DATA        directory for scripts/bench_s1_vs_s2.py --data; the benchmark is skipped when unset
#   TPB              tokens per training batch (default 32768; lower it for big trunks)
#   GPU_UTIL         vLLM --gpu-memory-utilization (default 0.7)
#   MAX_LEN          vLLM --max-model-len (default 32768)
#   MAX_NUM_SEQS     vLLM --max-num-seqs (unset: vLLM's default)
#   COPY=1           copy the base weights into the plumbed directory instead of linking them (to ship it)
#   PLUMBER, VLLM, PYTHON   the training env's `plumber`, the serving env's `vllm` and `python` (default: on PATH)
#   PORT             server port (default 8000)
set -euo pipefail

BASE=${1:?usage: plumbify_and_bench.sh <base model> <out dir> [vllm serve args]}
OUT=${2:?usage: plumbify_and_bench.sh <base model> <out dir> [vllm serve args]}
shift 2
: "${ROWS:?set ROWS to the training rows (plumb Row JSONL)}"
: "${DEV:?set DEV to held-out rows for calibration}"
TPB=${TPB:-32768}
GPU_UTIL=${GPU_UTIL:-0.7}
MAX_LEN=${MAX_LEN:-32768}
PLUMBER=${PLUMBER:-plumber}
VLLM=${VLLM:-vllm}
PYTHON=${PYTHON:-python}
PORT=${PORT:-8000}
NAME=$(basename "$OUT")
REPO=$(cd "$(dirname "$0")/.." && pwd)
SEQS=()
[ -n "${MAX_NUM_SEQS:-}" ] && SEQS=(--max-num-seqs "$MAX_NUM_SEQS")

echo "== 1/4 plumbify $BASE -> $OUT"
"$PLUMBER" plumbify --base "$BASE" --rows "$ROWS" --dev "$DEV" --out "$OUT" --tokens_per_batch "$TPB" \
    ${COPY:+--copy}

echo "== 2/4 parity: vLLM vs transformers on 200 dev rows"
"$PYTHON" "$REPO/scripts/vllm_check.py" --model "$OUT" --rows "$DEV" --ref "$OUT.train/dev_preds.jsonl" \
    --n 200 --gpu_mem "$GPU_UTIL" --max_model_len "$MAX_LEN" ${MAX_NUM_SEQS:+--max_num_seqs "$MAX_NUM_SEQS"}

if [ -z "${EVAL_DATA:-}" ]; then
    echo "done. Serve it with: vllm serve $OUT"
    exit 0
fi

echo "== 3/4 serve"
"$VLLM" serve "$OUT" --served-model-name "$NAME" --port "$PORT" --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" "${SEQS[@]}" "$@" > "$OUT.serve.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "localhost:$PORT/v1/models" > /dev/null; do
    kill -0 "$SERVER" 2> /dev/null || { tail -30 "$OUT.serve.log"; exit 1; }
    sleep 5
done

echo "== 4/4 benchmark: base model (thinking off / on), System 1 alone, plumbed model"
"$PYTHON" "$REPO/scripts/bench_s1_vs_s2.py" --url "http://localhost:$PORT" --data "$EVAL_DATA" --out "$OUT.bench"
echo "results: $OUT.bench/summary.json"
