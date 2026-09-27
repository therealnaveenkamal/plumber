# Contributing

```bash
git clone git@github.com:therealnaveenkamal/plumber.git && cd plumber
pip install -e ".[dev]" && pre-commit install
pytest -q tests          # tiny random NemotronH; CPU or one GPU
ruff check . && ruff format --check .
```

- Keep `Plumber`'s interface small; new capability goes behind `decide`, not beside it.
- Anything that needs the real 30B trunk is not a unit test — put it under `examples/` or document it in the model card.
- Commit messages: `engine:`, `server:`, `training:`, `data:`, `docs:` prefixes.
