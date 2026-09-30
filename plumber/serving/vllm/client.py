"""The plumb loop on a vLLM engine: decision requests, reading answers back, and generation with decisions.

A decision request is an ordinary generation request (one token, greedy, logprobs over the options) whose
``extra_args["plumb"]`` marks the decision suffix. ``Decider`` runs them on the server's ``EngineClient`` and drives
the chat loop: generate until the model closes a tool call (``</tool_call>``, Gemma's ``<tool_call|>``, ...). A
``plumb_decide`` call becomes a decision request over the tokens generated so far, mostly served from the prefix
cache, and its answer goes back as the tool response. A client tool call ends the turn.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
import uuid
from collections import OrderedDict

from ...branch import Frames, Markup, decision_suffix, parse_tool_call
from ...core.answers import answer
from ...core.render import render_decision
from ...core.row import Option, Row
from ...core.tool import PLUMB_TOOL, row_from_tool_args, tool_schema
from .decisions import find_decision, pick, resolve

MEMO_SIZE = 4096  # exact-repeat decisions answered from memory
MAX_DECISIONS = 8  # plumb_decide calls per request
# OpenAI sampling fields forwarded to vLLM's SamplingParams as-is
SAMPLING_KEYS = (
    "seed",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)


def plumb_extra(suffix_start: int, spans: list[tuple[int, int]], decide: int, k: int) -> dict:
    return {
        "plumb": {
            "suffix_start": suffix_start,
            "spans": [list(s) for s in spans],
            "decide": decide,
            "k": k,
        }
    }


def read_probs(logprobs: dict, k: int) -> list[float] | None:
    """Option probabilities from the one sampled position's logprobs (token id j = option j); None when the engine
    signalled missing suffix rows (token id k)."""
    lp = {int(t): (v.logprob if hasattr(v, "logprob") else float(v)) for t, v in logprobs.items()}
    if k in lp and lp[k] > -1e-3:
        return None
    p = [math.exp(lp.get(j, float("-inf"))) for j in range(k)]
    s = sum(p)
    return [x / s for x in p] if s > 0 else None


def mid_generation_request(tok, frames: Frames, context_ids: list[int], row):
    """(prompt ids, extra_args) for a decision right after ``context_ids`` (e.g. a generated tool call)."""
    suf, spans, decide = decision_suffix(tok, frames, row)
    P = len(context_ids)
    return context_ids + suf, plumb_extra(
        P, [(P + a, P + b) for a, b in spans], P + decide, len(row.options)
    )


def standalone_request(tok, row, context=None, template_kwargs=None):
    """(prompt ids, extra_args) for a decision about a state (and/or a conversation), rendered as in training."""
    rd = render_decision(tok, row, context, None, template_kwargs)
    return rd.input_ids, plumb_extra(rd.suffix_start, rd.opt_spans, rd.decide_pos, len(row.options))


class Decider:
    def __init__(self, engine, plumb_path: str):
        from transformers import AutoTokenizer

        from ...artifact import read_spec

        # vLLM-internal: EngineClient.model_config (vllm/engine/protocol.py)
        mc = engine.model_config
        self.engine, self.name = engine, mc.served_model_name
        self.spec = read_spec(plumb_path)
        self.template_kwargs = self.spec.extra.get("template_kwargs")
        conf = self.spec.calibration.conformal
        self.qhat = None if conf is None else conf.qhat
        self.tok = AutoTokenizer.from_pretrained(
            mc.tokenizer, revision=mc.tokenizer_revision, trust_remote_code=mc.trust_remote_code
        )
        self.markup = Markup.of(self.tok)
        self.frames = Frames.of(self.tok, self.template_kwargs, self.markup)
        # stripped from replies
        self.specials = sorted({str(t) for t in self.tok.all_special_tokens}, key=len, reverse=True)
        self.memo: OrderedDict[tuple, list[float]] = OrderedDict()
        # vLLM caps logprobs per request (--max-logprobs); options past the cap read as probability 0
        self.max_logprobs = int(getattr(mc, "max_logprobs", 20) or 20)

    async def _final(self, prompt: dict, params):
        out = None
        async for item in self.engine.generate(prompt, params, f"plumb-{uuid.uuid4().hex}"):
            out = item
        return out

    async def _decision(self, ids: list[int], extra: dict) -> tuple[list[float], dict]:
        """(probabilities, info) for one decision request; info: cached / computed tokens, memo, retry."""
        k = extra["plumb"]["k"]
        key = (tuple(ids), json.dumps(extra, sort_keys=True))
        if key in self.memo:  # an exact repeat: the answer is a function of the prompt
            self.memo.move_to_end(key)
            return self.memo[key], {"cached": len(ids), "computed": 0, "memo": True}
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=1, temperature=0.0, logprobs=min(k, self.max_logprobs), extra_args=extra
        )
        out = await self._final({"prompt_token_ids": ids}, params)
        probs, retry = read_probs(out.outputs[0].logprobs[0], k), False
        if probs is None:
            # part of the suffix came from the prefix cache, so the head never saw it: recompute without the cache
            retry = True
            out = await self._final(
                {"prompt_token_ids": ids, "cache_salt": uuid.uuid4().hex}, params
            )
            probs = read_probs(out.outputs[0].logprobs[0], k)
            if probs is None:
                raise RuntimeError("decision rows missing even without prefix-cache reads")
        cached = out.num_cached_tokens or 0
        self.memo[key] = probs
        if len(self.memo) > MEMO_SIZE:
            self.memo.popitem(last=False)
        return probs, {
            "cached": cached,
            "computed": len(ids) - cached,
            "memo": False,
            "retry": retry,
        }

    def _answer(self, row: Row, probs: list[float]) -> dict:
        if row.qtype == "noul" and [o.name for o in row.options] != ["no", "yes"]:
            # a yes/no question with other option names
            row = dataclasses.replace(row, qtype="choice")
        return answer(row, probs, self.qhat)

    @staticmethod
    def normalize(messages: list[dict]) -> list[dict]:
        """OpenAI-format history -> what chat templates accept: tool-call arguments as dicts, no null content."""
        out = []
        for m in messages:
            m = dict(m)
            if m.get("content") is None:
                m["content"] = ""
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    c = json.loads(json.dumps(c))
                    fn = c.get("function") or {}
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"] or "{}")
                        except ValueError:
                            fn["arguments"] = {}
                    calls.append(c)
                m["tool_calls"] = calls
            out.append(m)
        return out

    def _sampling(self, req: dict, max_tokens: int):
        from vllm import SamplingParams

        extra = {k: req[k] for k in SAMPLING_KEYS if req.get(k) is not None}
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=float(req.get("temperature", 0.0)),
            top_p=float(req.get("top_p", 1.0)),
            stop=[self.markup.call_close],
            include_stop_str_in_output=True,
            skip_special_tokens=False,
            **extra,
        )

    async def _generate(self, ids: list[int], params, stream: bool):
        """Yield token events while generating (when ``stream``); the last item is the finished output."""
        seen, out = 0, None
        async for item in self.engine.generate(
            {"prompt_token_ids": ids}, params, f"plumb-{uuid.uuid4().hex}"
        ):
            out = item.outputs[0]
            if stream and len(out.text) > seen:
                yield {"type": "token", "text": out.text[seen:]}
                seen = len(out.text)
        yield out

    def _call_body(self, text: str) -> str | None:
        """The body of the tool call that ``text`` ends with, if it ends with one."""
        m = self.markup
        opened = text.rfind(m.call_open)
        if not text.endswith(m.call_close) or opened < 0:
            return None
        return text[opened + len(m.call_open) : -len(m.call_close)]

    async def chat_events(self, req: dict):
        """Generation with System 1 decisions on the same engine and KV cache, as events:
        {"type": "segment", "in_think"}   a generation segment starts (inside a thinking block or not)
        {"type": "token", "text"}         generated text (raw: may contain thinking / tool-call markup)
        {"type": "decision_request", ...} / {"type": "decision", ...}   System 1 answering a decision
        {"type": "tool_response", "text"} the decision fed back to the model
        {"type": "tool_call", "name", "arguments"}   the model called a client tool
        {"type": "done", "finish_reason", "prompt_tokens", "generated_tokens", "answer"}
        ``mode="system1"`` answers the decision stated in the last user message and generates nothing."""
        messages, tools = self.normalize(req["messages"]), req.get("tools") or []
        kw = {**(self.template_kwargs or {}), **(req.get("chat_template_kwargs") or {})}
        # continuation frames follow the client's template settings; decision frames follow training
        conv = Frames.of(self.tok, kw, self.markup)
        last = messages[-1] if messages else {}
        stated = (
            find_decision(last["content"])
            if last.get("role") == "user" and isinstance(last.get("content"), str)
            else None
        )
        stated_ans = None
        if req.get("mode") == "system1" and stated is None:
            raise ValueError(
                "mode=system1 needs a question followed by a bulleted option list in the last user message"
            )
        if stated is not None and req.get("auto_decide", True):
            # The message poses a decision: System 1 answers it as written, rendered as in training (the text
            # before the question is the state; earlier turns are context; operator system prompts are left out).
            row = Row(
                "stated",
                stated.before,
                stated.question,
                stated.qtype,
                [Option(n, d) for n, d in stated.options],
            )
            earlier = [m for m in messages[:-1] if m.get("role") != "system"] or None
            args = {
                "question": row.question,
                "type": row.qtype,
                "options": [{"name": o.name, "description": o.desc} for o in row.options],
            }
            yield {
                "type": "decision_request",
                "source": "stated",
                "question": row.question,
                "qtype": row.qtype,
                "options": args["options"],
            }
            t0 = time.perf_counter()
            probs, info = await self._decision(
                *standalone_request(self.tok, row, earlier, self.template_kwargs)
            )
            stated_ans = self._answer(row, probs)
            yield {
                "type": "decision",
                "source": "stated",
                "arguments": args,
                "result": stated_ans,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "context_tokens": 0,
                "cached_tokens": info["cached"],
                "computed_tokens": info["computed"],
            }
            if req.get("mode") == "system1":
                top = max(stated_ans["probabilities"], key=stated_ans["probabilities"].get)
                yield {
                    "type": "done",
                    "finish_reason": "stop",
                    "prompt_tokens": 0,
                    "generated_tokens": 0,
                    "answer": {
                        "choice": top,
                        "by": "system1",
                        "p": round(stated_ans["probabilities"][top], 6),
                    },
                    "text": top,
                }
                return
            call = {
                "id": "plumb_stated",
                "type": "function",
                "function": {"name": PLUMB_TOOL, "arguments": args},
            }
            messages = [
                *messages,
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {
                    "role": "tool",
                    "tool_call_id": "plumb_stated",
                    "name": PLUMB_TOOL,
                    "content": json.dumps(stated_ans),
                },
            ]
        text = self.tok.apply_chat_template(
            messages,
            tools=[tool_schema(), *tools],
            tokenize=False,
            add_generation_prompt=True,
            **kw,
        )
        ids = self.tok(text, add_special_tokens=False).input_ids
        prompt_tokens, in_think = len(ids), self.markup.opens_thinking(text)
        budget, finish, generated, out_text = int(req.get("max_tokens") or 1024), "length", 0, ""
        decisions, client_calls = 0, 0
        while budget > 0:
            # After a client tool call, the continuation is only kept if it is another tool call (parallel calls);
            # it is generated silently so nothing else the model writes after its last call reaches the client.
            probing = client_calls > 0
            if not probing:
                yield {"type": "segment", "in_think": in_think}
            out = None
            async for item in self._generate(ids, self._sampling(req, budget), stream=not probing):
                if isinstance(item, dict):
                    yield item
                else:
                    out = item
            body = self._call_body(out.text)
            if probing and (
                body is None
                or out.text.lstrip()[: len(self.markup.call_open)] != self.markup.call_open
            ):
                break
            if probing:
                yield {"type": "segment", "in_think": False}
                yield {"type": "token", "text": out.text}
            generated += len(out.token_ids)
            ids, budget, finish = (
                ids + list(out.token_ids),
                budget - len(out.token_ids),
                out.finish_reason,
            )
            out_text += out.text
            parsed = parse_tool_call(body) if body is not None else None
            if parsed is None:
                break
            name, call_args = parsed
            if name != PLUMB_TOOL:  # the client's own tool: handed back in OpenAI form
                yield {"type": "tool_call", "name": name, "arguments": call_args}
                client_calls += 1
                finish = "tool_calls"
                continue
            if client_calls or decisions >= MAX_DECISIONS:
                break
            decisions += 1
            names = [
                str(o.get("name")) for o in call_args.get("options") or [] if isinstance(o, dict)
            ]
            original = resolve(names, stated)
            try:
                if original is not None:  # the call restates the stated decision: use it as written
                    row = Row(
                        "tool",
                        "",
                        stated.question,
                        stated.qtype,
                        [Option(n, d) for n, d in original],
                    )
                else:
                    row = row_from_tool_args(call_args)
            except (KeyError, TypeError, ValueError, AssertionError, AttributeError):
                row = None
            if row is None or len(row.options) < 2:
                ans = {"error": "plumb_decide needs a question and at least two options"}
            else:
                yield {
                    "type": "decision_request",
                    "source": "model",
                    "question": row.question,
                    "qtype": row.qtype,
                    "options": [{"name": o.name, "description": o.desc} for o in row.options],
                }
                t0 = time.perf_counter()
                if original is not None and stated_ans is not None:
                    # asking again about a decision System 1 already answered gets the same answer
                    ans, info = stated_ans, {"cached": 0, "computed": 0}
                else:
                    probs, info = await self._decision(
                        *mid_generation_request(self.tok, self.frames, ids, row)
                    )
                    ans = self._answer(row, probs)
                yield {
                    "type": "decision",
                    "source": "model",
                    "arguments": call_args,
                    "result": ans,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "context_tokens": len(ids),
                    "cached_tokens": info["cached"],
                    "computed_tokens": info["computed"],
                }
            resp = conv.to_tool + json.dumps(ans) + conv.after_tool
            ids = ids + self.tok(resp, add_special_tokens=False).input_ids
            in_think = self.markup.opens_thinking(resp)
            yield {"type": "tool_response", "text": resp}
        final_answer = None
        if stated_ans is not None:
            # System 1 decides when it is confident enough; below `trust`, the model's own stated choice wins
            probs = stated_ans["probabilities"]
            top, p_top = max(probs.items(), key=lambda kv: kv[1])
            own = pick(self.final_text(out_text), list(probs))
            if p_top >= float(req.get("plumb_trust", 0.7)) or own is None:
                final_answer = {"choice": top, "by": "system1", "p": round(p_top, 6)}
            else:
                final_answer = {"choice": own, "by": "model", "system1": top, "p": round(p_top, 6)}
        yield {
            "type": "done",
            "finish_reason": finish,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated,
            "answer": final_answer,
        }

    def final_text(self, out_text: str) -> str:
        """The model's final reply: after the last tool call, without thinking, markup or special tokens."""
        final = self.markup.final(out_text)
        for special in self.specials:
            final = final.replace(special, "")
        return final.strip()
