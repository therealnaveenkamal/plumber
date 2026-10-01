# Guide

How to train a plumb, serve it and call it. For how the pieces work inside, see [Architecture](architecture.md).

[Install](#install) · [Train](#train) · [Commands](#commands) · [Serve](#serve) · [API](#api) ·
[Check a model](#check-a-model) · [Troubleshooting](#troubleshooting) · [Limits](#limits)

## Install

Plumbify needs Linux, an NVIDIA GPU and Python 3.10 or later.

```bash
pip install "plumbify[vllm]"
```

This installs vLLM 0.30.0, the plugin and the `plumbify` command. If vLLM 0.30.0 is already installed, add plumbify
alone with `pip install --no-deps plumbify`. For the latest code on `main`, install from GitHub instead:
`pip install "plumbify[vllm] @ git+https://github.com/therealnaveenkamal/plumbify"`.

## Train

```bash
plumbify train --base Qwen/Qwen3.5-9B \
    --rows train.jsonl --dev dev.jsonl --out plumbed-qwen3.5-9b
```

`--rows` is the training set and `--dev` a held-out set used for calibration. Each line is one decision:

```json
{"id": "support/0001", "state": "Ticket: I was charged twice for invoice 4411.",
 "question": "Which team should handle this?", "qtype": "choice",
 "options": [{"name": "billing", "desc": "invoices, refunds"}, {"name": "technical", "desc": "bugs, outages"}],
 "gold": 0, "meta": {"family": "support/routing"}}
```

| Field | Meaning |
|---|---|
| `qtype` | `choice`, `noul` (options must be `no`, `yes`) or `score` (options are levels, low to high) |
| `gold` | index of the right option |
| `meta.family` | optional; used to report accuracy per family |

Useful flags: `--epochs` (default 1), `--lora_r` (default 32), `--tokens_per_batch` (lower it if training runs out of
memory), `--copy` (copy the base weights into the output instead of linking them, needed before uploading it),
`--limit N` (train on the first N rows, for a quick test).

The output is an ordinary model directory: the base weights, a `config.json` whose architecture is
`Plumb<BaseArchitecture>`, and the plumb (`plumb.json`, `head.safetensors`, `suffix_adapter.*`). The base weights are
never changed. [recipes/](../recipes/) has tested settings for each model we have run.

## Commands

`plumbify train` is the only command most people need. The others are its steps, for when you want to run them
apart. Every command takes `--help`.

| Command | Does |
|---|---|
| `plumbify train` | Train a plumb, calibrate it on `--dev` and write the plumbed model directory |
| `plumbify train-head` | Train and calibrate a plumb only; writes `<out>/final`, `dev_preds.jsonl` and `metrics.json` |
| `plumbify package <plumb> <out>` | Combine a trained plumb with its base model into a plumbed model directory |
| `plumbify eval-head` | Evaluate a plumb on labelled rows, or the base model zero-shot with `--zeroshot` |

## Serve

```bash
vllm serve plumbed-qwen3.5-9b
```

No flags or environment variables are needed.

### Share a plumbed model

Every plumbed model directory includes a model card (`README.md`) with the plumb's settings and its results on
`--dev`. To publish one, train it with `--copy` so the base weights are real files rather than links into your
Hugging Face cache, then upload the directory:

```bash
hf upload your-org/Qwen3.5-9B-plumb plumbed-qwen3.5-9b
```

Anyone with plumbify installed can then `vllm serve your-org/Qwen3.5-9B-plumb`.

## API

The server speaks the standard OpenAI API. `/v1/chat/completions` sends decisions to the plumb in two ways:

- **Stated decisions.** If the last user message contains a question followed by at least two bulleted options, the
  plumb answers it before the model writes anything.
- **Model hand-offs.** The model is told about the plumb in its system prompt and sees a `plumb_decide` tool. When it
  calls the tool mid-generation, the server answers the call from the conversation's KV cache and the model continues.
  The call never reaches the client.

Client tools work as usual and come back in `tool_calls`.

### Request

Add an optional `plumb` field to a normal chat request:

```json
{"model": "plumbed-qwen3.5-9b", "messages": [...], "plumb": {"trust": 0.7}}
```

| Option | Default | Effect |
|---|---|---|
| `enabled` | `true` | `false` (or `"plumb": false`) serves the request with the base model alone |
| `mode` | `"auto"` | `"system1"` answers the stated decision with the plumb only; nothing is generated |
| `trust` | `0.7` | the plumb's answer wins when its top probability is at least this; otherwise the model's does |
| `announce` | `true` | add the note about the plumb to the system prompt |
| `auto_decide` | `true` | answer a decision stated in the user message before generating |

Requests with `n > 1`, `response_format`, a forced `tool_choice`, `logprobs` or `stop` are served by vLLM's own
handler without the plumb.

### Response

A normal chat completion, plus a `plumb` field:

```json
"plumb": {
  "decisions": [{
    "event": "decision", "source": "stated",
    "arguments": {"question": "Which team should handle this?", "type": "choice", "options": ["..."]},
    "result": {"type": "choice", "choice": "billing", "confidence": 0.96,
               "probabilities": {"billing": 0.973, "technical": 0.015, "sales": 0.012},
               "set": ["billing"]},
    "latency_ms": 58.1, "context_tokens": 0, "cached_tokens": 0, "computed_tokens": 212
  }],
  "answer": {"choice": "billing", "by": "system1", "p": 0.973}
}
```

| Field | Meaning |
|---|---|
| `decisions[].source` | `stated` (from the user message) or `model` (a `plumb_decide` call) |
| `decisions[].arguments` | the question and options that were decided |
| `cached_tokens` | context tokens read from the KV cache instead of recomputed |
| `result.probabilities` | probability of each option |
| `result.confidence` | how far the top option stands above a uniform guess, from 0 to 1 |
| `result.set` | options that can't be ruled out at 90% coverage; one option means the plumb is sure |
| `result.noul` / `result.score` | for yes/no questions, P(yes); for scales, the expected level |
| `answer` | the final choice for a stated decision, and `by`: `system1` (the plumb) or `model` |

When streaming, each decision arrives as a chunk with a `plumb` field as soon as it is made, and the last chunk
carries the final `answer`.

### From Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="plumbed-qwen3.5-9b",
    messages=[{"role": "user", "content": "Ticket: charged twice.\n\nWhich team?\n- billing\n- technical\n- sales"}],
    extra_body={"plumb": {"trust": 0.8}},
)
print(r.model_extra["plumb"]["answer"])
```

`python scripts/demo_chat.py` opens a live chat that shows each decision as it happens.

## Check a model

```bash
python scripts/vllm_check.py --model plumbed-qwen3.5-9b --rows dev.jsonl \
    --ref plumbed-qwen3.5-9b.train/dev_preds.jsonl
```

This compares the served decisions with the training implementation (expect 194 or more of 200 to agree), and
checks that decisions reuse the KV cache and don't change what the model generates.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Training runs out of GPU memory | Lower `--tokens_per_batch`. The [recipes](../recipes/README.md) list values that fit on 80 GB. |
| Training Qwen3.5 or Qwen3-Next is slow | Install `flash-linear-attention` in the training environment (about ten times faster). |
| `vllm serve` runs out of memory | Raise `--gpu-memory-utilization`, or lower `--max-model-len` and `--max-num-seqs`. |
| The server logs "plumbify's vLLM plugin is tested with vLLM ..." | You are on an untested vLLM version. Run the parity check above before trusting decisions. |
| A response has no `plumb` field | The request used `n > 1`, `response_format`, a forced `tool_choice`, `logprobs` or `stop`, or `plumb` was set to `false`, so vLLM's own handler served it. |
| Decisions are slow on long conversations | Prefix caching is off. Leave it on (the vLLM default). |

## Limits

- Tested with vLLM 0.30.0 only; other versions log a warning.
- One GPU per model (no tensor or pipeline parallelism).
- The base architecture must expose EAGLE-3 hidden states in vLLM (most recent Llama, Qwen, Gemma, DeepSeek, GPT-OSS
  and Kimi models do).
- Keep prefix caching on (the default); without it every decision recomputes the whole conversation.
