# Changelog

Notable changes to Plumbify (named Plumber before 0.2.0). The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [semantic versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-10-02

### Added

- `plumbify chat`: a live terminal chat with a running server that shows each decision as the model hands it to its
  plumb. It replaces `scripts/demo_chat.py`, so trying a model no longer needs a clone. Thinking is on by default
  (`--no-think` turns it off) and replies get up to 8,192 tokens.

### Changed

- Model cards are shorter: no latency footnote and no development-set paragraph. They now show `plumbify chat`.

## [0.2.1] - 2026-10-02

### Changed

- A shorter system prompt. The note about the plumb is one sentence, and the `plumb_decide` description and schema
  no longer repeat it: the plumb adds 198 tokens to the prompt instead of 314 (on Qwen3.5, plus the 232 tokens of
  tool-calling instructions its chat template adds for any tool).

## [0.2.0] - 2026-10-01

A rewrite. The plumb now lives inside the LLM, sharing its weights and KV cache, and vLLM serves it. In 0.1 it was a
standalone decision model with its own server. The project is renamed from Plumber to Plumbify.

### Added

- A vLLM plugin, registered as a `vllm.general_plugins` entry point. `vllm serve <plumbed model>` needs no flags,
  for any architecture vLLM supports that exposes EAGLE-3 hidden states.
- Decisions from the conversation's KV cache: a decision stated in the last user message is answered before
  generation, and the model can hand decisions to its plumb mid-generation through the `plumb_decide` tool.
- `/v1/chat/completions` accepts an optional `plumb` request field (`enabled`, `mode`, `trust`, `announce`,
  `auto_decide`) and returns a `plumb` response field with every decision, streaming included.
- A suffix-only LoRA that acts on decision tokens alone, so the base weights, the context's KV cache and generation
  are exactly the base model's.
- The plumbed model directory (`Plumb<BaseArchitecture>`) and `plumbify package` to build one.
- Calibration with a fitted temperature and split-conformal prediction sets (`result.set`).
- `System1`, the reference implementation on transformers.
- Recipes and measured results for Qwen3.5-27B, Qwen3.5-35B-A3B, Gemma 4 12B, Qwen3.5-9B, Qwen3.5-4B and
  Qwen3-1.7B.
- A model card (`README.md`) in every plumbed model directory, ready for `hf upload`.
- Scripts: a vLLM parity check, the benchmark, a results table and a live chat.

### Changed

- Renamed to Plumbify: `pip install plumbify`, `import plumbify`, and the `plumbify` command.
- `plumber plumbify` is now `plumbify train`, and `plumber eval` is now `plumbify eval-head`.
- The base model keeps its LM head: the same loaded model generates and decides.

### Removed

- The `Plumber` Python engine and `plumber serve`, the TypeSafe System One-compatible server. Serve with
  `vllm serve` and the OpenAI chat completions API instead.
- The dataset converters in `plumber.data`.

## [0.1.0] - 2026-09-27

First release: a plumb as a standalone typed decision model on Nemotron 3.5 Lightning 30B-A3B, served with a
TypeSafe System One-compatible API.
