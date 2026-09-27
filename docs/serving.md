# Serving

- **Memory.** The trunk is 30B parameters in bf16 (~62 GB). One 80 GB GPU, or two 40 GB+ GPUs with `device_map="auto"`.
- **Kernels.** Install `mamba_ssm` and `causal_conv1d` built against your torch for the fast path. Without them the transformers reference implementation runs; with them, CPU inference is unavailable.
- **Latency.** One prefill per request. With prefix caching the state is encoded once; each additional question costs its own
  ~20–200 tokens of continuation against the cached state. `benchmarks/serving.py` measures p50 per request against question count.
- **Requests.** Put every question about a state in one request. Keep option descriptions short; they are read as text.
- **Compatibility.** The server speaks the TypeSafe `/v1/systemone` contract, so existing Jev clients and benchmark harnesses
  (JevBench's `typesafe` adapter, DecisionBench's runtime) work unmodified.
