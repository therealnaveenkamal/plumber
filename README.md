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
- **Drop-in for Jev.** `POST /v1/systemone` with the System One request and response schema; the official `typesafe_sdk` client works against a Plumb server unchanged.
- **Plumbify.** A Plumb model is an adapter and a 17M-parameter head on an unchanged open-weight LLM. `plumber plumbify --base <hf id>` builds one on any causal LM.

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

If you already call Jev, point the TypeSafe client at Plumber and keep the rest of your code:

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8123", model="plumb")
response = client.system_one(
    state="I was charged twice for my subscription. Fix this today or I cancel.",
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(instructions="What is the customer's tone?",
                       criteria={"calm": None, "frustrated": None, "angry": None}),
        "urgency": Score(instructions="How urgent is this ticket?",
                         criteria=["can wait", "this week", "today"]),
    },
)
print(response.nouls["billing"].noul, response.choices["tone"].choice, response.scores["urgency"].score)
```

Requirements: a CUDA GPU with ~64 GB of memory in bf16 (one 80 GB card, or two 40 GB+ cards with `device_map="auto"`); `mamba_ssm` and `causal_conv1d` for the fast path. Python 3.10+, transformers ≥ 5.17.

## Models

| Model | Base | Accuracy new / trained | Brier new / trained | Coverage @ 5% error | Runs on | |
|---|---|:---:|:---:|:---:|---|---|
| **Plumb v1** | Nemotron 3.5 Lightning 30B-A3B | 0.703 / 0.846 | 0.412 / 0.217 | 0.216 / 0.753 | 1× 80 GB, or 2× 40 GB | [Model card](https://huggingface.co/totum-labs/plumb-nemotron-3.5-lightning-30b-a3b) |

DecisionBench rows. **New** is six task families the model never saw (routing_triage, document_workflows, guardrails_moderation, risk_scoring, triage, content_moderation; 6,966 rows) — the closest thing here to your own questions. **Trained** is held-out rows from the 21 families it trained on. Brier scores the whole distribution, lower is better; coverage is the share of decisions that can be automated at a 5% error budget. Split definitions ship with [totum-labs/plumb-data](https://huggingface.co/datasets/totum-labs/plumb-data).

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

## API

`POST /v1/systemone` — `state` (string, object or array), `model`, and `questions` keyed by an id the model never sees.

| Type | `criteria` | Answer |
|---|---|---|
| `noul` | optional descriptions for `true` and `false` | `noul`: P(yes); `probabilities` over `{no, yes}`, `confidence` |
| `choice` | `{option: description \| null}` | `choice`, `probabilities` over the options, `confidence` |
| `score` | ordered list of level descriptions | `score`: expected level index from 0; `legend`, `probabilities`, `confidence` |

Confidence is the reference adapter's: `(p_max − 1/K) / (1 − 1/K)` for `choice` and `noul`; `max(0, 1 − E|level − mode| / D)` for `score`, with `D` the mean absolute deviation of a uniform distribution over the levels. Neither is an accuracy estimate — the calibrated quantity is `probabilities`. Invalid requests return `422`; `usage.output_tokens` is 0 because nothing is generated; `latency_ms` is model time. `GET /v1/models` lists the loaded model. `PLUMBER_API_KEY` requires `Authorization: Bearer <key>` on `/v1/*`, which the TypeSafe clients always send; the server binds to `127.0.0.1` unless `--host 0.0.0.0`.

## How It Works

```
[BOS] <|state|> … <|/state|>  <|q|> instructions <|/q|>  <|opt|> name: description <|/opt|> …  [DECIDE]
                                                           ▲ o_1                       ▲ o_K     ▲ q
z_k = ⟨W_q q, W_o o_k⟩ / √d′ + u·o_k        p = softmax(z)
```

One sequence per question. The trunk is Nemotron 3.5 Lightning with the LM head removed; a 17M-parameter pointer head scores each option's hidden state against the decision position. `noul` is the two-option case of `choice`, so `P(yes) + P(no) = 1` exactly. `score` reads the level descriptions as options and returns the expected level. Training adapts the trunk with LoRA on the Mamba, attention and shared-expert projections; routed experts stay frozen.

## Fine-Tune on Your Own Data

Training rows are API requests with a `label` on every question — the same JSON your code already sends, one request per line:

```jsonl
{"state": {"subject": "Charged twice", "body": "I see two charges for order #4411. Please refund one."},
 "questions": {
   "team":     {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments and refunds", "shipping": "Delivery problems"}, "label": "billing"},
   "angry":    {"type": "noul",   "instructions": "Is the customer angry?", "label": false},
   "priority": {"type": "score",  "instructions": "How urgent is this?", "criteria": ["low", "normal", "high"], "label": 1}}}
```

`choice` labels are the option name, `noul` labels `true`/`false`, `score` labels the level's position from 0. Keep 10–20% aside for evaluation. Continue from the released adapter (`adapter/` plus `head.pt` from the model repo) so the model keeps what it knows, then evaluate, merge and serve:

```bash
plumber train  --rows train.jsonl --dev heldout.jsonl --out runs/mine --init_from <adapter dir> \
               --epochs 2 --lr 2e-5 --tokens_per_batch 32768 --max_len 32768
plumber eval      --ckpt runs/mine/final --rows heldout.jsonl --out runs/mine/eval   # accuracy, NLL, Brier, ECE, coverage
plumber calibrate --ckpt runs/mine/final --preds runs/mine/eval/preds.jsonl          # fit a temperature on your labels
plumber export    --ckpt runs/mine/final --out release/mine                          # merged bf16 trunk + head.pt
plumber serve  --model release/mine
```

`calibrate` fits one temperature on held-out predictions and stores it in `head.pt`; it never changes which option wins, only how far the probabilities sit from uniform, so thresholds you set on `probabilities` are measured on your own labels. Option order is shuffled every pass, so the head learns the options rather than their positions. Batches are built to a token budget, so 32k-token states train alongside short ones. Rows in the internal `Row` schema (`plumber/core/rendering.py`) are accepted too; the converters, leave-families-out split and contamination screen behind the released data live in `plumber/data/`.

## Plumbify Any Model

The recipe above is not specific to Nemotron. `plumber plumbify` fits the adapter and head on any Hugging Face causal LM:

```bash
plumber plumbify --base Qwen/Qwen3.8-27B --rows train.jsonl --dev dev.jsonl --out runs/qwen3.8-27b
```

The LM head is dropped at load. LoRA targets are chosen from the architecture (`plumber/core/targets.py`): Mamba and DeltaNet `in_proj`, attention `q/k/v/o`, dense and shared-expert MLP projections; routed experts stay frozen. Training refuses to start unless at least 90% of the trunk's layers received an adapter, so an unrecognised architecture fails before it burns GPU time rather than training a head alone. The result runs through the same engine, server and SDK.

| Base | Adapter |
|---|---|
| [Nemotron 3.5 Lightning 30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) | Plumb v1 — released |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | planned |
| [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B) | planned |

## Repository Layout

```
plumber/
  engine.py      Plumber — decide, choice, noul, score
  contract.py    System One contract: questions -> rows, probabilities -> answers
  server.py      POST /v1/systemone
  cli.py         plumber serve | decide | plumbify | train | eval | calibrate | export
  core/          rendering, heads, trunk, LoRA targets
  training/      train, eval, calibrate, export
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
