

# Plumber

**Give an open LLM a fast decision path, then serve it with vLLM.**

[CI](https://github.com/therealnaveenkamal/plumber/actions/workflows/ci.yml)
[License](LICENSE)
Python
vLLM

[Guide](docs/guide.md) · [Recipes](recipes/) · [Results](#results)



Language models are slow and inconsistent at the small judgements that fill real conversations: which team gets
this ticket, is this refund allowed, how urgent is it. Thinking helps, at the cost of seconds per decision.

Plumber trains a small decision head, a **plumb**, onto an open model without changing its weights. With the
plugin installed, `vllm serve` loads the result like any other model. The model generates as usual, and when a
conversation reaches a decision, the plumb answers it in one forward pass, reading the KV cache the conversation
already filled, with a calibrated probability for every option.

![The normal model next to the plumbed model, on the same server](docs/demo.gif)

*Qwen3.5-35B-A3B on one A100, real time. The same messages go to both sides. Left: the normal model with thinking on.
Right: the plumbed model, which hands the routing decision to its plumb mid-reply.*

**A plumb is a Jev model built into the LLM.** It answers the same kind of typed decision as a Jev-style decision
model such as TypeSafe's Jev: pick one of these options, yes or no, or a score on a scale. The difference is that it
is not a second model next to the LLM. It is a Jev decision head inside the model, sharing its weights and its KV
cache, so plumbifying a model is jevifying it.

## Quickstart

```bash
pip install "plumber[vllm] @ git+https://github.com/therealnaveenkamal/plumber"

plumber plumbify --base Qwen/Qwen3.5-9B --rows train.jsonl --dev dev.jsonl --out plumbed-qwen3.5-9b
vllm serve plumbed-qwen3.5-9b
```

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "plumbed-qwen3.5-9b",
  "messages": [{"role": "user", "content": "Ticket: I was charged twice.\n\nWhich team should handle this?\n- billing\n- technical\n- sales"}]
}' | jq .plumb.answer
```

The [guide](docs/guide.md) covers the training data format, the request options and the response fields.
`python scripts/demo_chat.py` opens a live chat that shows each decision as the model hands it off.

## Results

Each model against itself on the same vLLM server, with the plumb off and on, over 447 held-out decisions (task
families and templates no model trained on). Each prompt states a decision: context, question, bulleted options.


| Model                 | Base, thinking off | Base, thinking on | **Plumbed** | Latency p50, thinking → plumbed |
| --------------------- | ------------------ | ----------------- | ----------- | ------------------------------- |
| Gemma 4 12B           | 0.736              | 0.718             | **0.826**   | 34.0 s → 2.2 s                  |
| Qwen3.5-35B-A3B (MoE) | 0.720              | 0.787             | **0.810**   | 19.8 s → 0.78 s                 |
| Qwen3.5-4B            | 0.667              | 0.720             | **0.801**   | 29.2 s → 0.24 s                 |
| Qwen3.5-9B            | 0.696              | 0.772             | **0.779**   | 32.6 s → 1.6 s                  |
| Qwen3-1.7B            | 0.570              | 0.642             | **0.658**   | 3.9 s → 0.05 s                  |


The plumbed model beats the base model without thinking by 8 to 13 points on every model. It beats thinking by 8 to
11 points on Gemma 4 12B and Qwen3.5-4B, and matches it on the other three (the gap is within one standard error),
at 15 to 120 times lower latency. The decision itself takes 20 to 100 ms; the rest of the latency is the model
writing its reply. Settings for each run are in [recipes/](recipes/).

## How it works

Take a support chat that reaches a decision: *which team should handle this ticket: billing, technical or sales?*

1. **The question is added to the conversation.** The question and the three options are appended, written like a
   normal message to the model.
2. **The model reads only the new part.** It has already read the conversation, and that work is cached, so only the
   few words of the question are processed.
3. **The plumb scores the options.** It looks at what the model computed while reading the question and turns that
   into a probability per option: billing 0.97, technical 0.02, sales 0.01.
4. **The model carries on.** No text was generated for the decision. The model uses the answer and continues its
   reply.

```
conversation so far (cached)  +  "Which team? billing / technical / sales"  →  plumb  →  billing 0.97
```

Training adds two small pieces and nothing else: the plumb itself, and an adapter that switches on only while the
model reads a decision question. The model's own weights never change, so everything else it does (chat, reasoning,
tools) works exactly as before.

## Development

```bash
pip install -e ".[dev]"
pytest -q tests
ruff check . && ruff format --check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@misc{plumber2026,
  title  = {Plumber: a System 1 decision branch for open language models},
  author = {Kamalakannan, Naveenraj},
  year   = {2026},
  url    = {https://github.com/therealnaveenkamal/plumber}
}
```



## License

Apache-2.0.