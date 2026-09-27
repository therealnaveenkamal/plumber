<div align="center">

# Plumber

**Turn an open-weight LLM into a plumb — a typed decision model — and serve it.**

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-totum--labs%2Fplumb-blue)](https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/therealnaveenkamal/plumber/actions/workflows/ci.yml/badge.svg)](https://github.com/therealnaveenkamal/plumber/actions/workflows/ci.yml)

[Model](https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b) · [Data](https://huggingface.co/datasets/totum-labs/plumb-data) · [Docs](docs/) · [Examples](examples/)

</div>

A plumb takes a piece of state and a typed question — `choice`, `noul`
(yes/no) or `score` (an ordered scale) — and returns a probability over the
options you pass in. One forward pass, no decoding. We delete the LM head and
read the trunk's hidden states at each option and at a final decision token
instead, so the model can't answer anything outside your options, and the
probabilities are trained directly rather than parsed out of generated text.
The server implements TypeSafe's System One API, so the official SDK and
anything written for Jev works as-is.

We built this to see how close a decision model on an open trunk gets to Jev.
The released plumb sits on Nemotron 3.5 Lightning 30B-A3B; the same code
plumbifies Qwen3.5 and Gemma 4.

## Install

```bash
pip install git+https://github.com/therealnaveenkamal/plumber
```

The released model wants one 80 GB GPU or two 40 GB ones, and transformers
≥ 5.17. Install `mamba_ssm` and `causal_conv1d` for the fast path.

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

Serve it and talk to it with TypeSafe's SDK:

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

The contract — request shape, confidence formulas, `GET /v1/models`, bearer
auth — is in [docs/api.md](docs/api.md).

## Results

We hold six of DecisionBench's 27 task families out of training and report
those. Atlan's Decision Bench and JevBench are separate benchmarks we never
trained on. The other columns are the numbers each benchmark publishes.

| Benchmark | Metric | Plumb | Jev 1.13 | Kev-4B | Laya |
|---|---|---:|---:|---:|---:|
| DecisionBench (Hanno-Labs), never-seen families | accuracy, micro / macro | **0.730 / 0.786** | 0.657 / 0.668 | 0.568 / 0.612 | 0.516 / 0.585 |
| | ECE | 0.074 | 0.128 | 0.255 | **0.044** |
| Decision Bench (Atlan) | accuracy | 0.816 | **0.924** | – | 0.528 |
| Decision Bench (Atlan) | latency p50 ¹ | **0.17 s** | 0.44 s | – | 1.52 s |
| JevBench, hard tier | accuracy | 0.622 | **0.741** | 0.423 | 0.341 |
| JevBench, easy / standard | accuracy | 1.000 / 0.972 | 1.000 / **0.990** | 1.000 / 0.917 | 0.944 / 0.729 |
| Hard-skill templates, held out | accuracy | 0.750 | – | – | – |

¹ Measured by Atlan's harness. Ours is a local server on one A100 with one
request in flight (p95 0.22 s); the others go through hosted APIs, so their
numbers include the network. TypeSafe doesn't publish latency.

Where we lose: Jev is 12 points ahead on JevBench's hard tier and 11 on
Atlan's bench, almost all of it on long documents (contracts, transcripts)
and judge-style questions. `evals/bench.py` reproduces our column; its
docstring says where the data comes from.

## Plumbify another model

```bash
plumber plumbify --base Qwen/Qwen3.5-4B --rows train.jsonl --out runs/qwen
plumber eval  --ckpt runs/qwen/final --rows test.jsonl --out runs/qwen/eval
plumber serve --model runs/qwen/final
```

`plumbify` puts LoRA on the trunk (Mamba/DeltaNet projections, attention,
MLP; routed experts are left alone) and trains it together with the head. If
fewer than 90% of the layers got an adapter it refuses to start — that's our
guard against a new architecture quietly training only the head. Image-text
checkpoints like Qwen3.5 and Gemma 4 load their text trunk. A plumb remembers
its base, so `eval` and `serve` don't need to be told.

For training data, `plumber.data.recipe` samples a fixed number of rows per
task family from the converters in `plumber/data/`, plus solver-labelled
hard-skill rows. The short version — 30k rows under 1k tokens — gets Nemotron
from base to 0.711 on the held-out families in 33 minutes on one A100; the
released model adds a long-context stage on top. Everything is in
[docs/recipes.md](docs/recipes.md).

## Fine-tune on your own data

Write your examples as the requests you would send the server, with a
`label` on each question. One request per line:

```json
{
  "state": "Two charges for order #4411. Please refund one.",
  "questions": {
    "team":  {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": null, "shipping": null},
              "label": "billing"},
    "angry": {"type": "noul", "instructions": "Is the customer angry?",
              "label": false}
  }
}
```

`choice` labels are the option name, `noul` labels are `true`/`false`, and
`score` labels are the level's index from 0. Keep some rows aside to evaluate
on.

```bash
plumber train --base totum-labs/plumb-nemotron-3.5-lightning-30b-a3b \
  --rows train.jsonl --out runs/mine
plumber eval  --ckpt runs/mine/final --rows heldout.jsonl --out runs/mine/eval
plumber calibrate --ckpt runs/mine/final --preds runs/mine/eval/preds.jsonl
plumber export --ckpt runs/mine/final --out release/mine
```

Using the released model as `--base` continues from its head; starting from
the raw Nemotron trunk instead throws away what it already knows. `calibrate`
fits a temperature on the held-out predictions and stores it in `head.pt` —
it moves the confidence, never the winning option. Details in
[docs/training.md](docs/training.md).

## How it works

```
<|state|> ... <|/state|>
<|q|> instructions <|/q|>
<|opt|> name: description <|/opt|>     one per option
[DECIDE]
```

One sequence per question. The trunk is the unchanged LLM; a 17M-parameter
pointer head scores each option's hidden state against the state at
`[DECIDE]`, and a softmax over those scores is the answer. `noul` is a
two-option `choice`; `score` is a `choice` over the levels, reported as the
expected level. When a request carries several questions we prefill the state
once and fork its cache — attention KV and recurrent state — to each question.

## Citation

```bibtex
@misc{plumb2026,
  title={Plumb: a typed decision model on Nemotron 3.5 Lightning},
  author={Kamalakannan, Naveenraj}, year={2026},
  url={https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b}
}
```

## License

Apache-2.0. Built on Nemotron 3.5 Lightning (NVIDIA, OpenMDW-1.1),
DecisionBench (Hanno-Labs), JevBench, Kev's hard-skill generators and
tasksource.
