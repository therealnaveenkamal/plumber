"""``/v1/chat/completions`` for plumbed models: the OpenAI protocol, with System 1 decisions from the same KV cache.

For a plumbed model every chat request runs the plumb loop (client.py): a decision stated in the user's message goes
to System 1 first; routine judgements the model meets mid-generation go to it through the ``plumb_decide`` tool, which
the server answers itself; client tools come back as ordinary ``tool_calls``. The model is told (system prompt) that
its routine judgements are served by a dedicated decision module. Responses are standard chat completions (or SSE
chunks) with thinking in ``reasoning_content`` and a ``plumb`` extension field listing the decisions made.

Per-request options, all optional: ``"plumb": {"enabled": true, "mode": "auto" | "system1", "trust": 0.7,
"announce": true, "auto_decide": true}`` (``false`` or ``{"enabled": false}`` = vLLM's own handler, unchanged).
Models without a plumb, and requests that use fields the plumb loop does not implement (``n > 1``,
``response_format``, a forced ``tool_choice``, ``logprobs``, ``stop``), get vLLM's own handler.

vLLM-internal: replaces the ``POST /v1/chat/completions`` route on vLLM's chat APIRouter
(vllm/entrypoints/openai/chat_completion/api_router.py, v0.30.0) before the app includes it.
"""

# No `from __future__ import annotations`: FastAPI resolves the route's `Request` annotation at runtime.
import json
import logging
import re
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...plumbed import plumb_dir

logger = logging.getLogger("vllm.plumbify")  # inherits vLLM's log handlers and format

# The policy only; how to read the answer is in the tool's own description (core/tool.py)
PLUMB_NOTE = (
    "Hand routine judgements (routing, classification, priority, yes/no checks, ratings) to plumb_decide instead of "
    "deciding them yourself: it is faster and more accurate. Then continue with its answer."
)
_SPECIAL = re.compile(r"<\|[^|>]+\|>")


def install_route() -> None:
    from fastapi.routing import APIRoute
    from vllm.entrypoints.openai.chat_completion import api_router as chat_api

    if getattr(chat_api, "_plumb_installed", False):
        return
    original = None
    for r in list(chat_api.router.routes):
        if isinstance(r, APIRoute) and r.path == "/v1/chat/completions" and "POST" in r.methods:
            original = r.endpoint
            chat_api.router.routes.remove(r)

    async def create_chat_completion(raw_request: Request):
        return await handle(raw_request, original)

    chat_api.router.add_api_route("/v1/chat/completions", create_chat_completion, methods=["POST"])
    chat_api._plumb_installed = True


def find_plumb(model: str) -> str | None:
    """The served model's plumb directory, or None (the model is then served without a plumb)."""
    try:
        return plumb_dir(model)
    except Exception as e:  # offline, gated or missing on the Hub
        logger.warning(
            "plumbify: could not look for a plumb in %s (%s); serving without it", model, e
        )
        return None


def get_decider(state):
    """The app's Decider, or None when the served model has no plumb (cached on the app state)."""
    if hasattr(state, "_plumb_decider"):
        return state._plumb_decider
    from .client import Decider

    engine = getattr(state, "engine_client", None)
    path = find_plumb(engine.model_config.model) if engine is not None else None
    state._plumb_decider = Decider(engine, path) if path else None
    return state._plumb_decider


def announce(messages: list[dict]) -> list[dict]:
    """Tell the model its routine judgements are served by System 1: appended to the system prompt."""
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system" and isinstance(msgs[0].get("content"), str):
        msgs[0]["content"] = f"{msgs[0]['content']}\n\n{PLUMB_NOTE}"
    else:
        msgs.insert(0, {"role": "system", "content": PLUMB_NOTE})
    return msgs


class Split:
    """Raw generated text -> (content, reasoning) deltas: thinking blocks become reasoning, tool-call markup and
    special tokens are dropped (the plumb extension reports decisions; client tool calls come back as tool_calls).
    Markers are the model's own (``Markup``): ``<think>``/``<tool_call>`` for Qwen, ``<|channel>thought``/
    ``<|tool_call>`` for Gemma 4."""

    def __init__(self, markup=None, specials=()):
        from ...branch import Markup

        self.markup = m = markup or Markup("<tool_call>", "</tool_call>", "<think>", "</think>")
        self.tags = tuple(t for t in (m.call_open, m.think_open, m.think_close) if t)
        self.specials = tuple(sorted(set(specials), key=len, reverse=True))
        # a tag or special token may straddle deltas: hold back this many characters until more text arrives
        self.hold = max(len(t) for t in (*self.tags, m.call_close, *self.specials, "<|x|>"))
        self.buf, self.hiding, self.thinking = "", False, False

    def start(self, in_think: bool) -> None:
        self.buf, self.hiding, self.thinking = "", False, in_think

    def feed(self, delta: str, final: bool = False) -> list[tuple[str, str]]:
        self.buf += delta
        out = []

        def emit(text):
            for s in self.specials:
                text = text.replace(s, "")
            text = _SPECIAL.sub("", text)
            if text:
                out.append(("reasoning" if self.thinking else "content", text))

        while self.buf:
            if self.hiding:
                end = self.buf.find(self.markup.call_close)
                if end < 0:
                    self.buf = "" if final else self.buf[-self.hold :]
                    break
                self.buf, self.hiding = self.buf[end + len(self.markup.call_close) :], False
                continue
            hits = [(self.buf.find(t), t) for t in self.tags if self.buf.find(t) >= 0]
            if not hits:
                # a tag may straddle deltas
                keep = len(self.buf) if final else max(0, len(self.buf) - self.hold)
                emit(self.buf[:keep])
                self.buf = self.buf[keep:]
                break
            pos, tag = min(hits)
            emit(self.buf[:pos])
            self.buf = self.buf[pos + len(tag) :]
            if tag == self.markup.call_open:
                self.hiding = True
            elif tag == self.markup.think_open:
                self.thinking = True
                self.buf = self.buf.removeprefix("\n") if self.buf or final else self.buf
            else:
                self.thinking = False
        return out


def _defaults(engine) -> dict:
    try:  # the model's generation_config sampling defaults, as vLLM's own handler uses them
        return dict(engine.model_config.get_diff_sampling_param() or {})
    except Exception:
        return {}


UNSUPPORTED = ("response_format", "logprobs", "top_logprobs", "stop")


def unsupported(body: dict) -> str | None:
    """Why the plumb loop can't serve this request as asked, if it can't (vLLM's own handler serves it instead)."""
    if int(body.get("n") or 1) != 1:
        return "n > 1"
    for k in UNSUPPORTED:
        if body.get(k):
            return k
    choice = body.get("tool_choice")
    if choice not in (None, "auto", "none"):
        return "a forced tool_choice"
    return None


def build_request(body: dict, opts: dict, engine) -> dict:
    from .client import SAMPLING_KEYS

    d = _defaults(engine)
    tools = body.get("tools") if body.get("tool_choice") != "none" else None
    return {
        "messages": announce(body["messages"]) if opts.get("announce", True) else body["messages"],
        "tools": [
            t for t in tools or [] if (t.get("function") or {}).get("name") != "plumb_decide"
        ],
        "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens") or 2048,
        "temperature": body.get("temperature", d.get("temperature", 0.7)),
        "top_p": body.get("top_p", d.get("top_p", 1.0)),
        "chat_template_kwargs": body.get("chat_template_kwargs"),
        "plumb_trust": opts.get("trust", 0.7),
        "auto_decide": opts.get("auto_decide", True),
        "mode": opts.get("mode", "auto"),
        **{k: body[k] for k in SAMPLING_KEYS if body.get(k) is not None},
    }


def _tool_call(ev: dict) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": ev["name"], "arguments": json.dumps(ev["arguments"])},
    }


def _plumb_event(ev: dict) -> dict:
    keep = (
        "source",
        "question",
        "qtype",
        "options",
        "arguments",
        "result",
        "latency_ms",
        "context_tokens",
        "cached_tokens",
        "computed_tokens",
    )
    return {"event": ev["type"], **{k: ev[k] for k in keep if k in ev}}


async def handle(raw: Request, original):
    try:
        body = await raw.json()
    except ValueError as e:
        return JSONResponse(
            {"error": {"message": f"bad json: {e}", "type": "BadRequestError"}}, status_code=400
        )
    opts = body.pop("plumb", None)
    decider = get_decider(raw.app.state)
    enabled = (
        decider is not None
        and opts is not False
        and not (isinstance(opts, dict) and opts.get("enabled") is False)
    )
    if not enabled or not body.get("messages"):
        return await passthrough(raw, original, body)
    reason = unsupported(body)
    if reason is not None:
        logger.info("plumbify: request uses %s; served by vLLM's handler without the plumb", reason)
        return await passthrough(raw, original, body)
    opts = opts if isinstance(opts, dict) else {}
    try:
        req = build_request(body, opts, decider.engine)
    except (KeyError, TypeError, ValueError) as e:
        return JSONResponse(
            {"error": {"message": str(e)[:300], "type": "BadRequestError"}}, status_code=400
        )
    cid, created, model = (
        f"chatcmpl-{uuid.uuid4().hex}",
        int(time.time()),
        body.get("model") or decider.name,
    )
    if body.get("stream"):
        return StreamingResponse(
            stream(decider, req, cid, created, model, body), media_type="text/event-stream"
        )
    try:
        return JSONResponse(await complete(decider, req, cid, created, model))
    except ValueError as e:
        return JSONResponse(
            {"error": {"message": str(e)[:300], "type": "BadRequestError"}}, status_code=400
        )


async def passthrough(raw: Request, original, body: dict):
    """vLLM's own handler, untouched (models without a plumb, `"plumb": false`, unsupported fields)."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    if original is None:
        return JSONResponse({"error": {"message": "chat completions unavailable"}}, status_code=501)
    return await original(request=ChatCompletionRequest.model_validate(body), raw_request=raw)


async def complete(decider, req: dict, cid: str, created: int, model: str) -> dict:
    split, content, reasoning, events, calls, done = (
        Split(decider.markup, decider.specials),
        [],
        [],
        [],
        [],
        {},
    )
    async for ev in decider.chat_events(req):
        t = ev["type"]
        if t == "segment":
            split.start(ev["in_think"])
        elif t == "token":
            for kind, text in split.feed(ev["text"]):
                (reasoning if kind == "reasoning" else content).append(text)
        elif t == "tool_response":
            for kind, text in split.feed("", final=True):
                (reasoning if kind == "reasoning" else content).append(text)
        elif t == "decision":
            events.append(_plumb_event(ev))
        elif t == "tool_call":
            calls.append(_tool_call(ev))
        elif t == "done":
            done = ev
            for kind, text in split.feed("", final=True):
                (reasoning if kind == "reasoning" else content).append(text)
    message = {"role": "assistant", "content": done.get("text") or "".join(content).strip() or None}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning).strip()
    if calls:
        message["tool_calls"] = calls
    usage = {
        "prompt_tokens": done.get("prompt_tokens", 0),
        "completion_tokens": done.get("generated_tokens", 0),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": done.get("finish_reason", "stop"),
                "logprobs": None,
            }
        ],
        "usage": usage,
        "plumb": {"decisions": events, "answer": done.get("answer")},
    }


async def stream(decider, req: dict, cid: str, created: int, model: str, body: dict):
    def chunk(delta: dict, finish=None, **extra) -> str:
        c = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
            **extra,
        }
        return f"data: {json.dumps(c)}\n\n"

    def deltas(pairs):
        for kind, text in pairs:
            yield chunk({"reasoning_content": text} if kind == "reasoning" else {"content": text})

    split = Split(decider.markup, decider.specials)
    yield chunk({"role": "assistant", "content": ""})
    n_calls = 0
    try:
        async for ev in decider.chat_events(req):
            t = ev["type"]
            if t == "segment":
                split.start(ev["in_think"])
            elif t == "token":
                for s in deltas(split.feed(ev["text"])):
                    yield s
            elif t == "tool_response":
                for s in deltas(split.feed("", final=True)):
                    yield s
            elif t in ("decision_request", "decision"):
                yield chunk({}, plumb=_plumb_event(ev))
            elif t == "tool_call":
                yield chunk({"tool_calls": [{"index": n_calls, **_tool_call(ev)}]})
                n_calls += 1
            elif t == "done":
                for s in deltas(split.feed("", final=True)):
                    yield s
                if ev.get("text"):
                    yield chunk({"content": ev["text"]})
                extra = {"plumb": {"event": "done", "answer": ev.get("answer")}}
                if (body.get("stream_options") or {}).get("include_usage"):
                    p, g = ev.get("prompt_tokens", 0), ev.get("generated_tokens", 0)
                    extra["usage"] = {
                        "prompt_tokens": p,
                        "completion_tokens": g,
                        "total_tokens": p + g,
                    }
                yield chunk({}, finish=ev.get("finish_reason", "stop"), **extra)
    except ValueError as e:
        yield f"data: {json.dumps({'error': {'message': str(e)[:300], 'type': 'BadRequestError'}})}\n\n"
    except Exception as e:
        # the stream has started: report the failure in-band instead of cutting it off
        logger.exception("plumbify: streaming request failed")
        yield f"data: {json.dumps({'error': {'message': str(e)[:300], 'type': 'InternalServerError'}})}\n\n"
    yield "data: [DONE]\n\n"
