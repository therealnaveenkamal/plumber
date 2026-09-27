# Recipes: plumbify any base in minutes

A plumb is a LoRA and a pointer head on an unchanged base LM. The recipe is a balanced training set — a few dozen rows from each of ~200 task families plus a sample of solver-labelled hard-skill rows — small enough to fit in one coffee break and broad enough to teach the format and the decision skills. The same recipe on different bases gives comparable plumbs.

## 1. Sources

| Source | Converter | Rows | License |
|---|---|---|---|
| DecisionBench (Hanno-Labs), 27 task families | `python -m plumber.data.decisionbench` then `python -m plumber.data.split` (leave-families-out) | 15k train / 7k held-out | see dataset card |
| tasksource classification tasks | `python -m plumber.data.tasksource --out data/tasksource/rows.jsonl` (denylists DecisionBench's source datasets) | ~100k, ~200 families | per dataset |
| Kev hard-skill generators (judge, tradeoff, probability, long policy, multi-hop, ambiguous, temporal/numeric) | `python -m plumber.data.kev --out data/kev_hard` (templates 4–5 held out as `test.jsonl`) | 46k train / 1.5k test | Apache-2.0 |

Run `python -m plumber.data.screen --candidates <rows> --heldout data/decisionbench/test_ood.jsonl` before training on anything you also evaluate on.

## 2. Build

```bash
python -m plumber.data.recipe --size small --rows data/decisionbench/train.jsonl data/tasksource/rows.jsonl --hard data/kev_hard/train.jsonl --out data/recipe/small
python -m plumber.data.recipe --size large --rows data/decisionbench/train.jsonl data/tasksource/rows.jsonl --hard data/kev_hard/train.jsonl --out data/recipe/large
```

| size | rows / family | hard rows / skill | rows | Qwen3.5-4B on 1× A100 |
|---|---|---|---|---|
| small | 20 | 100 | ~5k | ~5 min |
| large | 60 | 300 | ~13k | ~15 min |

`manifest.json` records every family's count. Structured states are rendered to `key: value` text; dev rows are disjoint from train.

## 3. Plumbify

```bash
plumber plumbify --base Qwen/Qwen3.5-4B          --rows data/recipe/large/train.jsonl --dev data/recipe/large/dev.jsonl --out runs/qwen3.5-4b   --epochs 1 --tokens_per_batch 16384 --max_len 8192 --lr 5e-5 --head_lr 5e-4
plumber plumbify --base Qwen/Qwen3.5-0.8B        --rows data/recipe/large/train.jsonl --dev data/recipe/large/dev.jsonl --out runs/qwen3.5-0.8b --epochs 1 --tokens_per_batch 16384 --max_len 8192 --lr 1e-4 --head_lr 5e-4
plumber plumbify --base google/gemma-4-26B-A4B   --rows data/recipe/large/train.jsonl --dev data/recipe/large/dev.jsonl --out runs/gemma4-26b-a4b --epochs 1 --tokens_per_batch 16384 --max_len 8192 --lr 5e-5 --head_lr 5e-4
plumber plumbify --base nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 --rows data/recipe/large/train.jsonl --dev data/recipe/large/dev.jsonl --out runs/nemotron --epochs 1 --tokens_per_batch 16384 --max_len 8192 --lr 5e-5 --head_lr 5e-4
```

LoRA targets are chosen per architecture (`plumber/core/targets.py`) and training refuses to start if fewer than 90% of the trunk's layers received one. Install `flash-linear-attention` for Qwen3.5 / Qwen3.8 bases: without it transformers runs the Gated DeltaNet layers in plain PyTorch, about 3× slower. Sub-1B bases need more than one pass over the recipe (`--epochs 2`); a 0.8B model at one epoch stays at chance on never-seen families. Image-text checkpoints (Qwen3.5, Gemma 4) load their text trunk only. Dense models above ~10B and MoE models above ~25B want two GPUs (`device_map="auto"` is the default).

## 4. Evaluate, calibrate, serve

```bash
plumber eval      --base Qwen/Qwen3.5-4B --ckpt runs/qwen3.5-4b/final --rows data/decisionbench/test_ood.jsonl --out runs/qwen3.5-4b/final/eval_new
plumber eval      --base Qwen/Qwen3.5-4B --ckpt runs/qwen3.5-4b/final --rows data/kev_hard/test.jsonl         --out runs/qwen3.5-4b/final/eval_hard
plumber eval      --base Qwen/Qwen3.5-4B --ckpt runs/qwen3.5-4b/final --rows data/recipe/large/dev.jsonl       --out runs/qwen3.5-4b/final/eval_dev
plumber calibrate --ckpt runs/qwen3.5-4b/final --preds runs/qwen3.5-4b/final/eval_dev/preds.jsonl --apply runs/qwen3.5-4b/final/eval_new/preds.jsonl
plumber serve     --base Qwen/Qwen3.5-4B --model runs/qwen3.5-4b/final
```

`eval_new` is the six DecisionBench families no recipe row comes from; `eval_hard` is the hard-skill templates no recipe row comes from. Both are the numbers in the README's recipe table.
