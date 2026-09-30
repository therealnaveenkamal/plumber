# Contributing

```bash
git clone git@github.com:therealnaveenkamal/plumber.git && cd plumber
pip install -e ".[dev]" && pre-commit install
pytest -q tests
ruff check . && ruff format --check .
```

The tests run on CPU with a tiny random model and don't need vLLM. Anything that needs a GPU or a real model
belongs in `scripts/vllm_check.py` or a recipe, not in `tests/`.

- **vLLM internals.** Code that depends on vLLM internals is marked `vLLM-internal` with the file it reaches into.
  Keep those marks up to date. When moving to a new vLLM version, run `scripts/vllm_check.py` on at least one hybrid
  model (Qwen3.5) and one attention-only model before bumping the pin in `pyproject.toml`.
- **Training/serving parity.** The decision suffix is rendered in two places: `plumber/core/render.py` (training) and
  `plumber.branch.decision_suffix` (serving, after a live turn). `tests/test_branch.py` checks they agree; keep it
  passing.
- **New architectures.** Add LoRA targets to `plumber/core/targets.py` if the generic choice is wrong, add chat markup
  to `plumber.branch.Markup` if the model's tool-call format is new, and add a recipe with measured numbers.
- **Commit messages** start with the area: `plugin:`, `training:`, `core:`, `recipes:`, `docs:`.
