# Contributing

Thanks for helping. This page covers setup, the checks a change must pass, and the parts of the code that need care.
[Architecture](docs/architecture.md) explains how the pieces fit together and maps the code.

## Setup

```bash
git clone git@github.com:therealnaveenkamal/plumbify.git && cd plumbify
pip install -e ".[dev]" && pre-commit install
```

## Checks

```bash
pytest -q tests
ruff check . && ruff format --check .
```

CI runs both on every pull request. The tests run on CPU in seconds, using a tiny random hybrid model (Mamba,
attention and MoE blocks; see `tests/tiny.py`), and don't need vLLM. Anything that needs a GPU or a real model
belongs in `scripts/vllm_check.py` or a recipe, not in `tests/`.

If your change touches `plumbify/serving/`, `plumbify/core/` or `plumbify/branch.py`, also run `scripts/vllm_check.py`
on a real model and put the result in the pull request.

## Parts that need care

- **Training/serving parity.** The decision suffix is rendered in two places: `plumbify/core/render.py` (training)
  and `plumbify.branch.decision_suffix` (serving, after a live turn). `tests/test_branch.py` checks they agree; keep
  it passing.
- **The base model stays untouched.** Context tokens must be computed by the base weights alone, and generation
  must be bit-identical to the base model. `tests/test_suffix_lora.py` and `tests/test_system1.py` check this.
- **vLLM internals.** Code that depends on vLLM internals is marked `vLLM-internal`, with the vLLM file it reaches
  into. Keep those marks accurate, and list new ones in [Architecture](docs/architecture.md#coupling-to-vllm).
- **New architectures.** Add LoRA targets to `plumbify/core/targets.py` if the generic choice is wrong, add chat
  markup to `plumbify.branch.Markup` if the model's tool-call format is new, and add a recipe with measured numbers.

## Upgrading vLLM

1. Bump the pin in `pyproject.toml` (`vllm` extra) and add the version to `TESTED_VLLM` in
   `plumbify/serving/vllm/__init__.py`.
2. Revisit each `vLLM-internal` mark against the new vLLM source.
3. Run `scripts/vllm_check.py` on at least one hybrid model (Qwen3.5) and one attention-only model, and
   include the output in the pull request.

## Recipes and results

A new or changed recipe needs measured numbers: training time, serving memory, and the
`scripts/bench_s1_vs_s2.py` results, in both the script header and the table in `recipes/README.md`. If a result in
the README changes, update it in the same pull request.

## Commits and pull requests

- Start the commit subject with the area: `core:`, `training:`, `plugin:`, `recipes:`, `scripts:`, `demo:`,
  `results:`, `docs:`, `ci:`.
- Keep one logical change per pull request, and say how you tested it.
- Add a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) for anything a user would notice.
