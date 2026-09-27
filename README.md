<div align="center">

# Plumber

**An inference engine for typed decision models. State in, calibrated decision out, one forward pass.**

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-totum--labs%2Fplumb-blue)](https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![CI](https://github.com/therealnaveenkamal/plumber/actions/workflows/ci.yml/badge.svg)](https://github.com/therealnaveenkamal/plumber/actions/workflows/ci.yml)

[Model](https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b) · [Data](https://huggingface.co/datasets/totum-labs/plumb-data) · [Docs](docs/) · [Examples](examples/)

</div>

---

## News

- **2026-09-27** — Plumb v1 released: Nemotron 3.5 Lightning 30B-A3B trunk, LM head removed, pointer readout. Evaluated on six held-out DecisionBench task families and the JevBench public tiers. Weights on the Hub.

## About

Plumber serves **Plumb** models: decision models that answer typed questions about a state with a probability distribution over caller-supplied options, in a single forward pass and without generating text. The interface is the System One contract — `choice`, `score` and `noul` questions — so existing Jev clients and benchmark harnesses run unmodified.

Plumber is built for decisions, not chat:

- **No decode loop.** The language-model head is deleted at load; decisions are read from the trunk at each option's position. Latency is one prefill.
- **Type safety by construction.** There is no vocabulary in the graph, so the answer is always one of the supplied options.
- **Calibrated probabilities** over options, with a chance-corrected confidence, for every question type.
- **Prefix caching.** A request carries any number of questions; the state is encoded once and its cache — attention KV and Mamba recurrent states — is forked to every question.
- **Long states.** 32k-token training window on a 256k-context trunk.
- **Jev-compatible server.** `POST /v1/systemone` with the TypeSafe request and response schema.

## Getting Started

Install:

```bash
pip install git+https://github.com/therealnaveenkamal/plumber
```

Run a model:

```python
from plumber import Plumber

engine = Plumber("totum-labs/plumb-nemotron-3.5-lightning-30b-a3b")

state = "Invoice #4411 was billed twice for March. The customer asks for a refund today or they cancel."

engine.choice(state, "Which team should handle this?",
              {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "sales": "pricing"})
# {'type': 'choice', 'choice': 'billing', 'probabilities': {'billing': 0.94, 'technical': 0.04, 'sales': 0.02}, 'confidence': 0.91}

engine.noul(state, "Does the customer threaten to leave?")
engine.score(state, "How urgent is this?", ["not urgent", "soon", "blocking"])
engine.decide(state, questions)          # any mix of questions in one request
```

Serve it:

```bash
plumber serve --model totum-labs/plumb-nemotron-3.5-lightning-30b-a3b --port 8123
```

```bash
curl localhost:8123/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"subject": "Duplicate charge on invoice #4411", "body": "We were billed twice for March."},
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages"}},
    "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to leave?"}
  }}'
```

Requirements: a CUDA GPU with ~64 GB of memory in bf16 (one 80 GB card, or two 40 GB+ cards with `device_map="auto"`); `mamba_ssm` and `causal_conv1d` for the fast path. Python 3.10+, transformers ≥ 5.17.

## Supported Models

| Model | Trunk | Context | Weights |
|---|---|---|---|
| Plumb v1 | Nemotron 3.5 Lightning 30B-A3B (Mamba-2 / MoE / attention hybrid) | 32k trained, 256k native | [totum-labs/plumb-nemotron-3.5-lightning-30b-a3b](https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b) |

`Plumber(model)` accepts a merged release (Hub id or directory) or an un-merged adapter directory, which it stacks on the base trunk.

## Benchmarks

Plumb v1, evaluated on data excluded from training. Comparator values are the benchmarks' published records; protocols and per-family results are on the [model card](https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b).

| Benchmark | Metric | Plumb v1 | Jev 1.13 |
|---|---|---:|---:|
| DecisionBench, 6 held-out task families (6,966 rows) | accuracy, micro / macro | 0.703 / 0.767 | 0.657 / 0.668 |
| | ECE | 0.067 | 0.128 |
| JevBench public, hard tier (111) | accuracy | 0.514 | 0.741 |
| JevBench public, easy / standard | accuracy | 1.000 / 0.967 | 1.000 / 0.990 |
| JevBench, server-side latency | p50 | 0.173 s | 0.652 s |

## How It Works

```
[BOS] <|state|> … <|/state|>  <|q|> instructions <|/q|>  <|opt|> name: description <|/opt|> …  [DECIDE]
                                                           ▲ o_1                       ▲ o_K     ▲ q
z_k = ⟨W_q q, W_o o_k⟩ / √d′ + u·o_k        p = softmax(z)
```

One sequence per question. The trunk is Nemotron 3.5 Lightning with the LM head removed; a 17M-parameter pointer head scores each option's hidden state against the decision position. `noul` is the two-option case of `choice`, so `P(yes) + P(no) = 1` exactly. `score` reads the level descriptions as options and returns the expected level. Training adapts the trunk with LoRA on the Mamba, attention and shared-expert projections; routed experts stay frozen.

## Training

`plumber train` fits the LoRA adapter and readout head on rows in the `Row` schema (`plumber/core/rendering.py`) with length-bucketed token-budget batches up to 32k tokens; `plumber eval` reports accuracy, NLL and ECE per family with per-row predictions; `plumber export` merges an adapter into the trunk and writes a release. Converters for DecisionBench, tasksource and Kev hard-skill data, the leave-families-out split and a contamination screen live in `plumber/data/`; the released training data and split definitions are on the Hub at [totum-labs/plumb-data](https://huggingface.co/datasets/totum-labs/plumb-data).

## Repository Layout

```
plumber/
  engine.py      Plumber — decide, choice, noul, score
  server.py      POST /v1/systemone
  cli.py         plumber serve | decide | eval | train | export
  core/          rendering, heads, trunk
  training/      train, eval, export
  data/          converters, splits, contamination screen
  client.py      HTTP client for a served model
  metrics.py     accuracy, NLL, Brier, ECE, coverage
tests/
examples/      quickstart.py, client.py
evals/         DecisionBench and JevBench runners
benchmarks/    serving latency vs. questions per request
docs/          api.md, serving.md, training.md
```

## Citation

```bibtex
@misc{plumb2026,
  title  = {Plumb: a typed decision model on Nemotron 3.5 Lightning},
  author = {Kamalakannan, Naveenraj},
  year   = {2026},
  url    = {https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b}
}
```

## Acknowledgments

Nemotron 3.5 Lightning (NVIDIA, OpenMDW-1.1) · DecisionBench (Hanno-Labs) · JevBench · Kev hard-skill generators (Apache-2.0) · tasksource.

## License

Apache-2.0.
