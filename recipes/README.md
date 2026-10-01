# Recipes

One script per model we have plumbed, with the settings that worked on a single A100 80 GB. Each script runs the
same four steps through `plumbify_and_bench.sh`:

1. `plumber plumbify`: train the plumb on the frozen model, calibrate it, and write the plumbed model directory.
2. `scripts/vllm_check.py`: load the plumbed model in vLLM and compare its decisions with the transformers
   implementation on 200 held-out rows. It also checks cache reuse, unchanged generation and the tool-call loop.
3. `vllm serve` the plumbed model with no extra flags beyond memory settings.
4. `scripts/bench_s1_vs_s2.py`: measure the base model with thinking off and on, the plumb alone, and the plumbed
   model, all on the same server.

```bash
DATA=path/to/rows EVAL_DATA=path/to/evals recipes/qwen3.5-9b.sh
```

Leave `EVAL_DATA` unset to stop after step 2. Set `COPY=1` to copy the base weights into the plumbed directory, which
you need before uploading it anywhere.

## Models

| Recipe | Model | Rows | Tokens per batch | Training | Serving memory | No thinking: alone → plumbed | Thinking: alone → plumbed |
|---|---|---:|---:|---|---|---|---|
| `qwen3.5-27b.sh` | Qwen3.5-27B | 10k | 16k | 104 min | 0.92 | 0.732 → 0.839 | 0.866 → 0.875 |
| `qwen3.5-35b-a3b.sh` | Qwen3.5-35B-A3B (MoE) | 10k | 8k | 37 min | 0.95, 16k context, 32 sequences | 0.720 → 0.810 | 0.852 → 0.875 |
| `gemma-4-12b.sh` | Gemma 4 12B | 10k | 32k | 76 min | 0.70 | 0.736 → 0.826 | 0.770 → 0.872 |
| `qwen3.5-9b.sh` | Qwen3.5-9B | 10k | 32k | 37 min | 0.70 | 0.696 → 0.779 | 0.808 → 0.846 |
| `qwen3.5-4b.sh` | Qwen3.5-4B | 20k | 32k | 54 min | 0.70 | 0.667 → 0.801 | 0.841 → 0.826 |
| `qwen3-1.7b.sh` | Qwen3-1.7B | 20k | 32k | 23 min | 0.70 | 0.570 → 0.658 | 0.642 → 0.707 |

"Serving memory" is the value passed to `--gpu-memory-utilization`. The big models need it high: Qwen3.5-35B-A3B's
72 GB of weights leave about 5 GB on an 80 GB card, which is why it also runs with a shorter context and at most 32
concurrent sequences.

For a model without a recipe, start from the one closest in size and architecture. Lower `TPB` if training runs out
of memory (the batch size in tokens), and raise `GPU_UTIL` or lower `MAX_LEN` and `MAX_NUM_SEQS` if serving does.

## Data

`DATA` is a directory holding the training rows (`train_10000.jsonl` or `train_20000.jsonl`, as each recipe names)
and `dev.jsonl`, the held-out rows used for calibration and the parity check. Rows are one decision per line, in the
format described in the [README](../README.md#training-data). Point `ROWS` and `DEV` at other files to use your own.

`EVAL_DATA` is the benchmark's data directory. `scripts/bench_s1_vs_s2.py` reads three held-out sets from it:
`evalsets/test_ood.jsonl` (DecisionBench families held out of training), `evalsets/test.jsonl` (held-out hard-skill
templates) and `out/general/test_new.jsonl` (unseen templates of the general set). It samples a fixed number of rows
per decision family from each, with a fixed seed, so every model sees the same 447 decisions.

## Environments

Training and serving can live in separate environments; that is how these runs were made. Set `PLUMBER` to the
training environment's `plumber` and `VLLM` and `PYTHON` to the serving environment's `vllm` and `python`:

```bash
PLUMBER=~/train-env/bin/plumber VLLM=~/serve-env/bin/vllm PYTHON=~/serve-env/bin/python \
    DATA=path/to/rows recipes/gemma-4-12b.sh
```

Qwen3.5 and Qwen3-Next models train about ten times faster with `flash-linear-attention` installed in the training
environment.
