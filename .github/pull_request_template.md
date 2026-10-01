## What and why

<!-- What this changes and why. Link the issue if there is one. -->

## How it was tested

<!-- Commands you ran and what they showed. For serving changes, paste the scripts/vllm_check.py summary. -->

## Checklist

- [ ] `pytest -q tests` and `ruff check . && ruff format --check .` pass
- [ ] `scripts/vllm_check.py` run on a real model, if `plumbify/serving/`, `plumbify/core/` or `plumbify/branch.py` changed
- [ ] Docs, recipes and README results updated, if behaviour or numbers changed
- [ ] A line under "Unreleased" in `CHANGELOG.md`, if users would notice
