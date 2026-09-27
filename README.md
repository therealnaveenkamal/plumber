<div align="center">

# Plumber

**Turn an open-weight LLM into a plumb — a typed decision model — and serve it.**

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-totum--labs%2Fplumb-blue)](https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/therealnaveenkamal/plumber/actions/workflows/ci.yml/badge.svg)](https://github.com/therealnaveenkamal/plumber/actions/workflows/ci.yml)

[Model](https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b) · [Data](https://huggingface.co/datasets/totum-labs/plumb-data) · [Docs](docs/) · [Examples](examples/)

</div>

A plumb answers typed questions about a state — `choice`, `noul` (yes/no),
`score` — with a probability over the options you supply, in one forward pass.
Nothing is generated, so the answer is always one of your options and the
probabilities are calibrated. The server speaks TypeSafe's System One
contract, so the official SDK and existing Jev clients work unchanged.

Plumber is the tooling: `plumbify` fits a LoRA and a pointer head on any
Hugging Face causal LM, `serve` runs it with the state prefilled once per
request and shared by every question, and `eval`, `calibrate` and `export`
cover the rest.

## Install

```bash
pip install git+https://github.com/therealnaveenkamal/plumber
```

The released 30B model needs one 80 GB GPU or two 40 GB; transformers ≥ 5.17,
plus `mamba_ssm` and `causal_conv1d` for the fast path.

## Use

```python
from plumber import Plumber

engine = Plumber("totum-labs/plumb-nemotron-3.5-lightning-30b-a3b")
state = "Invoice #4411 was billed twice. Refund it today or we cancel."

engine.choice(state, "Which team should handle this?",
              {"billing": "invoices, refunds", "technical": "bugs, outages"})
# {'choice': 'billing', 'probabilities': {'billing': 0.94, ...}, ...}
engine.noul(state, "Does the customer threaten to leave?")
engine.score(state, "How urgent is this?", ["not urgent", "soon", "blocking"])
```

Serve it, then point any System One client at it:

```bash
plumber serve --model totum-labs/plumb-nemotron-3.5-lightning-30b-a3b
```

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(api_key="local",
                        base_url="http://127.0.0.1:8123", model="plumb")
r = client.system_one(
    state="I was charged twice. Fix this today or I cancel.",
    questions={
        "billing": Noul(instructions="Is this about billing?"),
        "tone": Choice(instructions="What is the customer's tone?",
                       criteria={"calm": None, "angry": None}),
    },
)
r.nouls["billing"].noul, r.choices["tone"].choice
```

`POST /v1/systemone` takes `{state, model, questions}` and returns the answers
with probabilities; [docs/api.md](docs/api.md) has the contract, the confidence
formulas, `GET /v1/models` and bearer auth.

## Benchmarks

DecisionBench here is [Hanno-Labs'](https://huggingface.co/datasets/Hanno-Labs/decision-bench)
(27 task families; six were never trained on), not Atlan's
[Decision Bench](https://decisionbench.ai/), which is the row below it.
Comparator values are each benchmark's published records; protocols and
per-family results are on the [model card](https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b).

| Benchmark | Metric | Plumb | Jev 1.13 | Kev-4B | Laya |
|---|---|---:|---:|---:|---:|
| DecisionBench, never-seen families | accuracy, micro / macro | **0.730 / 0.786** | 0.657 / 0.668 | 0.568 / 0.612 | 0.516 / 0.585 |
| | ECE | 0.074 | 0.128 | 0.255 | **0.044** |
| Decision Bench (Atlan) | accuracy | 0.816 | **0.924** | – | 0.528 |
| Decision Bench (Atlan) | latency p50 ¹ | **0.17 s** | 0.44 s | – | 1.52 s |
| JevBench, hard tier | accuracy | 0.622 | **0.741** | 0.423 | 0.341 |
| JevBench, easy / standard | accuracy | 1.000 / 0.972 | 1.000 / **0.990** | 1.000 / 0.917 | 0.944 / 0.729 |
| Hard-skill templates, held out | accuracy | 0.750 | – | – | – |

¹ Atlan's harness, per decision: Plumb on a local server (one A100, one request
in flight, p95 0.22 s); the others through their hosted APIs, network included.
TypeSafe publishes rate limits, not latency.

Reproduce Plumb's column:

```bash
python -m plumber.data.decisionbench && python -m plumber.data.split
python -m plumber.data.kev --out data/kev_hard
python evals/bench.py --model totum-labs/plumb-nemotron-3.5-lightning-30b-a3b \
  --out out/bench --jevbench /path/to/jevbench
```

## Plumbify

```bash
plumber plumbify --base Qwen/Qwen3.5-4B \
  --rows data/recipe/large/train.jsonl --dev data/recipe/large/dev.jsonl \
  --out runs/qwen3.5-4b --epochs 1 --tokens_per_batch 16384 --max_len 8192
plumber eval --base Qwen/Qwen3.5-4B --ckpt runs/qwen3.5-4b/final \
  --rows data/decisionbench/test_ood.jsonl --out runs/qwen3.5-4b/eval
plumber serve --base Qwen/Qwen3.5-4B --model runs/qwen3.5-4b/final
```

The LM head is dropped at load. LoRA targets come from the architecture —
Mamba and DeltaNet projections, attention, MLP; routed experts stay frozen —
and training refuses to start if fewer than 90% of the layers received one.
Image-text checkpoints (Qwen3.5, Gemma 4) contribute their text trunk.

`plumber.data.recipe` builds the training set: a capped number of rows from
each of ~200 task families plus solver-labelled hard-skill rows. Thirty
thousand rows under 1k tokens take the Nemotron trunk from base to 0.711 on
the never-seen DecisionBench families in 33 minutes on one A100; the released
model adds a long-context stage. [docs/recipes.md](docs/recipes.md) has the
sources, commands and numbers. Plumbs so far: Nemotron 3.5 Lightning
30B-A3B (released); Qwen3.8-27B and Gemma 4 26B-A4B next.

## Fine-tune on your own data

Training rows are API requests with a `label` on each question — the JSON
your code already sends, one request per line:

```json
{"state": "Two charges for order #4411. Please refund one.",
 "questions": {
   "team": {"type": "choice", "instructions": "Which team?",
            "criteria": {"billing": "refunds", "shipping": "delivery"},
            "label": "billing"},
   "angry": {"type": "noul", "instructions": "Is the customer angry?",
             "label": false},
   "priority": {"type": "score", "instructions": "How urgent?",
                "criteria": ["low", "normal", "high"], "label": 1}}}
```

```bash
plumber train --rows train.jsonl --dev dev.jsonl --out runs/mine \
  --init_from <plumb dir> --epochs 2 --lr 2e-5
plumber eval --ckpt runs/mine/final --rows dev.jsonl --out runs/mine/eval
plumber calibrate --ckpt runs/mine/final --preds runs/mine/eval/preds.jsonl
plumber export --ckpt runs/mine/final --out release/mine
```

`--init_from` keeps what the released plumb knows. `calibrate` fits one
temperature on held-out predictions and stores it in `head.pt`; it moves the
confidence, never the winning option. [docs/training.md](docs/training.md).

## How it works

```
<|state|> ... <|/state|>
<|q|> instructions <|/q|>
<|opt|> name: description <|/opt|>     one per option
[DECIDE]
```

One sequence per question. The trunk is the unchanged LLM; a 17M-parameter
pointer head scores each option's hidden state against the decision position,
and a softmax over those scores is the answer. `noul` is a two-option
`choice`; `score` is a `choice` over the levels and returns the expected
level. LoRA adapts the trunk during training. When a request carries several
questions, the state is prefilled once and its cache — attention KV and
recurrent state — is forked to each question.

## Citation

```bibtex
@misc{plumb2026,
  title={Plumb: a typed decision model on Nemotron 3.5 Lightning},
  author={Kamalakannan, Naveenraj}, year={2026},
  url={https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b}
}
```

## License

Apache-2.0. Built on Nemotron 3.5 Lightning (NVIDIA, OpenMDW-1.1), DecisionBench
(Hanno-Labs), JevBench, Kev's hard-skill generators and tasksource.
