# Training

`--rows` takes JSONL in either of two shapes, one object per line:

- an API request with a `label` on every question (`choice`: option name; `noul`: `true`/`false`; `score`: level index from 0) — each question becomes one row;
- the internal `Row` schema (`plumber/core/rendering.py`): `state`, `question`, `qtype`, `options[{name, desc}]`, `gold`, optional `teacher`, `meta`.

```bash
plumber train  --rows data/train.jsonl --dev data/dev.jsonl --out runs/x --epochs 2 --tokens_per_batch 32768 --max_len 32768
plumber eval   --ckpt runs/x/final --rows data/test.jsonl --out runs/x/final/eval     # accuracy, NLL, Brier, ECE, coverage@5%
plumber calibrate --ckpt runs/x/final --preds runs/x/final/eval/preds.jsonl [--apply other/preds.jsonl]
plumber export --ckpt runs/x/final --out ./release --card README.md [--repo org/name]
```

- LoRA targets are selected per architecture (`plumber/core/targets.py`): Mamba/DeltaNet `in_proj`, attention q/k/v/o, dense and shared-expert MLP projections; routed experts and embeddings frozen. Training aborts if fewer than 90% of layers received LoRA modules. The readout head is trained fully.
- `plumber plumbify --base <hf id> ...` is `plumber train` on another causal LM: the LM head is dropped at load and the same LoRA + head — a plumb — are fitted.
- Micro-batches are built to a token budget (`--tokens_per_batch`), so 32k-token rows train at batch 1 while short rows batch wide.
- `calibrate` fits a single temperature by NLL on held-out predictions (golden-section over log T) and writes it into `head.pt`; the head divides its logits by it at inference. `--apply` reports before/after metrics on a second prediction set without refitting.
- Checkpoints are written every `--save_every` steps as `step<N>/` and at the end as `final/` (a plumb: `adapter_model.safetensors` + `head.pt`). `--init_from <dir>` warm-starts from one.
- Data tooling under `plumber/data/`: DecisionBench conversion and leave-families-out splits, tasksource and Kev converters,
  a contamination screen, and the mix builder. Released rows and split definitions: [totum-labs/plumb-data](https://huggingface.co/datasets/totum-labs/plumb-data).
