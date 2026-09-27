#!/bin/bash
# Run the JevBench public tiers against a Plumb model through JevBench's unmodified `typesafe` adapter.
#   tools/run_jevbench.sh <model repo or dir> <jevbench checkout> <output dir outside the checkout> [port]
set -euo pipefail
MODEL=${1:?model}; JB=${2:?jevbench dir}; OUT=${3:?output dir}; PORT=${4:-$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1])")}
mkdir -p "$OUT"
plumber serve --model "$MODEL" --port "$PORT" > "$OUT/server.log" 2>&1 & SPID=$!
for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/" >/dev/null && break; sleep 5; done
T="$JB/datasets/public/easy.jsonl,$JB/datasets/public/original.jsonl,$JB/datasets/public/hard.jsonl"
(cd "$JB" && python -m jevbench.cli run --tasks "$T" --adapter typesafe --endpoint "http://127.0.0.1:$PORT" --model plumb --key-env "" \
   --results "$OUT/results.jsonl" --raw-dir "$OUT/raw" --ledger "$OUT/ledger.jsonl" --cost-basis local_gpu_no_provider_tariff --price-in-per-m 0 --price-out-per-m 0 \
 && python -m jevbench.cli summarize --tasks "$T" --results "$OUT/results.jsonl" --public-export "$OUT/summary.json")
kill $SPID
