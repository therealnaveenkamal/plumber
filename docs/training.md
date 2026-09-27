# Training

Rows are in the `Row` schema (`plumber/core/rendering.py`): `state`, `question`, `qtype`, `options[{name, desc}]`, `gold`, optional `teacher`, `meta`.

```bash
plumber train  --rows data/train.jsonl --dev data/dev.jsonl --out runs/x --epochs 2 --tokens_per_batch 32768 --max_len 32768
plumber eval   --ckpt runs/x/step2205 --rows data/test.jsonl --out runs/x/step2205/eval
plumber export --ckpt runs/x/step2205 --out ./release --card README.md [--repo org/name]
```

- LoRA on Mamba `in_proj`, attention q/k/v/o and the shared-expert projections; routed experts and embeddings frozen. The readout head is trained fully.
- Micro-batches are built to a token budget (`--tokens_per_batch`), so 32k-token rows train at batch 1 while short rows batch wide.
- `--init_from` warm-starts from a previous run's adapter and head.
- Data tooling under `plumber/data/`: DecisionBench conversion and leave-families-out splits, tasksource and Kev converters,
  a contamination screen, and the mix builder. Released rows and split definitions: [totum-labs/plumb-data](https://huggingface.co/datasets/totum-labs/plumb-data).
