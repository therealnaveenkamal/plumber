# Scripts

Tools that need a GPU or a running server, and so live outside `plumbify/` and `tests/`. Run them from the repository
root. Each script's docstring has the full usage, and each takes `--help`.

| Script | Needs | Does |
|---|---|---|
| [`vllm_check.py`](vllm_check.py) | GPU, vLLM | Checks a plumbed model in vLLM against the transformers reference: decision parity, prefix-cache reuse, unchanged generation and the `plumb_decide` loop. Run it after training and after any vLLM upgrade. |
| [`bench_s1_vs_s2.py`](bench_s1_vs_s2.py) | a running server | The benchmark behind the README results: the model alone (thinking off and on, with and without a tool in the prompt), the plumb alone, and the plumbed model, on the same server. Writes `summary.json`. |
| [`bench_report.py`](bench_report.py) | `summary.json` files | Builds one results table across models from several benchmark runs. |

## Typical order

```bash
# 1. after `plumbify train`: check the plugin against the reference
python scripts/vllm_check.py --model plumbed-qwen3.5-9b --rows dev.jsonl \
    --ref plumbed-qwen3.5-9b.train/dev_preds.jsonl

# 2. with `vllm serve plumbed-qwen3.5-9b` running: benchmark it
python scripts/bench_s1_vs_s2.py --url http://localhost:8000 \
    --data path/to/decision-data --out runs/bench-9b

# 3. one table across models
python scripts/bench_report.py qwen3.5-9b=runs/bench-9b gemma-4-12b=runs/bench-gemma
```

The [recipes](../recipes/README.md) run steps 1 and 2 for you.

The live chat that used to live here is now part of the package: `plumbify chat`.
