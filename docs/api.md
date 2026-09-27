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
| `score(state, instructions, levels)` | `{"type": "score", "score": E[level], "legend", "probabilities", "confidence"}` |

`state` is a string or a JSON object (rendered as `key: value` lines). `questions` follow the TypeSafe schema:
`{"type": "choice" | "noul" | "score", "instructions": str, "criteria": ...}` — a `{name: description}` map for `choice`,
an optional `{"true": ..., "false": ...}` for `noul`, an ordered list of level descriptions for `score`.

`share_prefix` (default: on for more than one question) encodes the state once and forks its cache to every question.
Confidence follows TypeSafe's reference adapter (`plumber/contract.py`): `(p_max − 1/K) / (1 − 1/K)` for `choice` and
`noul`; `max(0, 1 − E|level − mode| / D)` for `score`, where `D` is the mean absolute deviation of a uniform distribution over
the levels. A single option has confidence 1.

## Server

`plumber serve --model REPO [--host 127.0.0.1] [--port 8123]`

| Method | Path | |
|---|---|---|
| `POST` | `/v1/systemone` | the request and response above; `model` in the request is echoed back |
| `GET` | `/v1/models` | loaded model, base and engine version |
| `GET` | `/` | health |

Bad JSON returns `400`; a request without a non-empty `questions` object, or a question that fails validation, returns `422`.
Every response carries an `x-typesafe-request-id` header. Set `PLUMBER_API_KEY` to require `Authorization: Bearer <key>` on `/v1/*`.
The official `typesafe_sdk` client works unchanged (`tests/test_sdk_compat.py`); `plumber.client.Client(base_url, api_key=None)`
mirrors the engine methods over HTTP.
