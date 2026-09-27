# API

## `Plumber`

```python
Plumber(model, base=DEFAULT_BASE, dtype="bfloat16", device_map="auto", max_len=32768)
```

`model` is a merged release (Hub id or directory) or an adapter directory, which is stacked on `base`.

| method | returns |
|---|---|
| `decide(state, questions, share_prefix=None)` | `{"answers": {qid: answer}, "usage": {...}, "model", "latency_ms"}` |
| `choice(state, instructions, criteria)` | `{"type": "choice", "choice", "probabilities", "confidence"}` |
| `noul(state, instructions, criteria=None)` | `{"type": "noul", "noul": P(yes), "probabilities", "confidence"}` |
| `score(state, instructions, levels)` | `{"type": "score", "score": E[level], "probabilities", "confidence"}` |

`state` is a string or a JSON object (rendered as `key: value` lines). `questions` follow the TypeSafe schema:
`{"type": "choice" | "noul" | "score", "instructions": str, "criteria": ...}` — a `{name: description}` map for `choice`,
an optional `{"true": ..., "false": ...}` for `noul`, an ordered list of level descriptions for `score`.

`share_prefix` (default: on for more than one question) encodes the state once and forks its cache to every question.
`confidence = (p_max − 1/K) / (1 − 1/K)`.

## Server

`plumber serve --model REPO --port 8123` exposes `POST /v1/systemone` with the same request and response shape, and
`GET /` for health. `plumber.client.Client(base_url)` mirrors the engine methods over HTTP.
