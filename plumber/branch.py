"""System 1 inside System 2's generation: a decision reads the live KV cache, then the cache is rolled back.

    Qwen generating ... <tool_call>{"name": "plumb_decide", "arguments": {question, options}}</tool_call>
        -> checkpoint the cache (clone the small recurrent / conv states; remember the attention length)
        -> append the decision suffix (close the turn, "Decision: ... Options: ...", assistant-open), adapter ON
        -> the head reads the suffix at the tapped layers -> probabilities
        -> restore the cache exactly, append <tool_response>{answer}</tool_response>, keep generating

The tool call is only the trigger; nothing external is called. The decision costs its ~200 suffix tokens, never a
re-read of the conversation. That is valid because the plumb's adapter is suffix-only: the context in the live cache
was computed by the base weights, which is exactly what the adapter was trained on top of. Without a cache (a fresh
request) the same decision is ``System1.decide`` over the rendered text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .calibration import prediction_set
from .core.answers import answer
from .core.decision_head import SuffixBatch
from .core.render import LEADS
from .core.row import Row
from .core.tool import PLUMB_TOOL, row_from_tool_args, tool_schema

_A, _D, _T = "\x00ASSISTANT\x00", "\x00DECISION\x00", "\x00TOOL\x00"
_XML_FN = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


_GEMMA_CALL = re.compile(r"^call:([^\s{]+)\s*(\{.*\})$", re.S)
_GEMMA_KEY = re.compile(r"([{,]\s*)([A-Za-z_][\w\-]*)\s*:")
_GEMMA_BARE = re.compile(r"(:\s*)([A-Za-z_][\w\-]*)(\s*[,}\]])")
_GEMMA_STR = '<|"|>'


@dataclass(frozen=True)
class Markup:
    """The model's own tool-call and thinking markup, read from its chat template."""

    call_open: str
    call_close: str
    think_open: str | None
    think_close: str | None

    # Gemma 4; Hermes / Qwen
    CALLS = (
        ("<|tool_call>", "<tool_call|>"),
        ("<tool_call>", "</tool_call>"),
    )
    # Gemma 4; Qwen, DeepSeek, GLM
    THINKS = (
        ("<|channel>thought", "<channel|>"),
        ("<think>", "</think>"),
    )

    @staticmethod
    def of(tok) -> Markup:
        tmpl = getattr(tok, "chat_template", None) or ""
        if isinstance(tmpl, dict):
            tmpl = " ".join(tmpl.values())
        call = next((c for c in Markup.CALLS if c[0] in tmpl and c[1] in tmpl), Markup.CALLS[1])
        think = next((t for t in Markup.THINKS if t[0] in tmpl), (None, None))
        return Markup(*call, *think)

    def calls(self, text: str) -> list[str]:
        """Bodies of the complete tool calls in ``text``."""
        pat = re.escape(self.call_open) + "(.*?)" + re.escape(self.call_close)
        return re.findall(pat, text, re.S)

    def opens_thinking(self, prompt: str) -> bool:
        """The prompt ends inside an open thinking block (generation starts as reasoning)."""
        return bool(self.think_open) and prompt.rstrip().endswith(self.think_open.rstrip())

    def final(self, text: str) -> str:
        """The reply after the last tool call and the last thinking block, without later call markup."""
        text = text.split(self.call_close)[-1]
        if self.think_close:
            text = text.split(self.think_close)[-1]
        return text.split(self.call_open)[0]


def _gemma_args(s: str) -> dict | None:
    """Gemma 4's argument syntax -> dict: bare keys, strings between ``<|"|>`` marks, JSON-like nesting."""
    parts = s.split(_GEMMA_STR)
    if len(parts) % 2 == 0:
        return None
    code = []
    for i, p in enumerate(parts):
        if i % 2:
            code.append(json.dumps(p))
        else:
            p = _GEMMA_KEY.sub(r'\1"\2":', p)
            p = _GEMMA_BARE.sub(
                lambda m: m[0] if m[2] in ("true", "false", "null") else f'{m[1]}"{m[2]}"{m[3]}', p
            )
            code.append(p)
    try:
        v = json.loads("".join(code))
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def parse_tool_call(body: str) -> tuple[str, dict] | None:
    """One tool-call body (between the model's call markers) -> (name, arguments). Formats open models emit:
    Hermes JSON ``{"name": ..., "arguments": {...}}`` (Qwen3, most templates), XML
    ``<function=NAME><parameter=KEY>VALUE</parameter>...</function>`` (Qwen3.5, Qwen3-Coder) and Gemma 4's
    ``call:NAME{key:<|"|>value<|"|>,...}``. XML values that parse as JSON (lists, numbers) are decoded; others stay
    strings."""
    body = body.strip()
    g = _GEMMA_CALL.match(body)
    if g:
        args = _gemma_args(g[2])
        return (g[1], args) if args is not None else None
    if body.startswith("{"):
        try:
            c = json.loads(body)
        except ValueError:
            return None
        args = c.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return None
        return c.get("name"), args
    m = _XML_FN.search(body)
    if not m:
        return None
    args = {}
    for k, v in _XML_PARAM.findall(m.group(2)):
        try:
            args[k] = json.loads(v)
        except ValueError:
            args[k] = v.strip()
    return m.group(1), args


class CacheCheckpoint:
    """Everything needed to put a transformers cache back as it was: the attention length (keys/values grow by
    concatenation, so a slice undoes them) and clones of linear-attention / Mamba states (updated in place)."""

    def __init__(self, cache):
        self.length = cache.get_seq_length()
        self.states = []
        for layer in cache.layers:
            for attr in ("conv_states", "recurrent_states"):
                d = getattr(layer, attr, None)
                if isinstance(d, dict):
                    self.states += [(d, i, t.clone()) for i, t in d.items() if t is not None]

    def restore(self, cache) -> None:
        for d, i, t in self.states:
            if d[i].shape == t.shape:
                d[i].copy_(t)  # keep the tensor's address (static for cudagraphs)
            else:
                d[i] = t
        for layer in cache.layers:
            keys = getattr(layer, "keys", None)
            if isinstance(keys, torch.Tensor) and keys.dim() == 4 and keys.shape[-2] > self.length:
                layer.keys = keys[..., : self.length, :]
                layer.values = layer.values[..., : self.length, :]


@dataclass
class Frames:
    """Chat-template text that follows an assistant turn in progress, derived from the tokenizer's own template."""

    to_decision: str  # close the assistant turn, open a user turn
    after_decision: str  # close the user turn, open the assistant (non-thinking, as in training)
    to_tool: str  # close the assistant turn, open the tool response
    after_tool: str  # close the tool response, open the assistant

    @staticmethod
    def of(tok, template_kwargs: dict | None = None, markup: Markup | None = None) -> Frames:
        kw = {"enable_thinking": False, **(template_kwargs or {})}
        markup = markup or Markup.of(tok)
        u = {"role": "user", "content": "hi"}
        a = {"role": "assistant", "content": _A}
        t1 = tok.apply_chat_template(
            [u, a, {"role": "user", "content": _D}],
            tokenize=False,
            add_generation_prompt=True,
            **kw,
        )
        # the tool response follows a real tool call: some templates (Gemma 4) render tool messages only after one,
        # inside the same model turn
        call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": PLUMB_TOOL, "arguments": {"question": "q"}},
                }
            ],
        }
        tool = {"role": "tool", "tool_call_id": "call_0", "name": PLUMB_TOOL, "content": _T}
        t2 = tok.apply_chat_template(
            [u, call, tool], tokenize=False, add_generation_prompt=True, **kw
        )
        close = t2.rfind(markup.call_close, 0, t2.index(_T)) if _T in t2 else -1
        if close >= 0:
            start = close + len(markup.call_close)
        else:  # a template that doesn't render tool calls: the tool turn follows plain assistant text
            t2 = tok.apply_chat_template(
                [u, a, {"role": "tool", "content": _T}],
                tokenize=False,
                add_generation_prompt=True,
                **kw,
            )
            start = t2.index(_A) + len(_A)
        return Frames(
            t1[t1.index(_A) + len(_A) : t1.index(_D)],
            t1[t1.index(_D) + len(_D) :],
            t2[start : t2.index(_T)],
            t2[t2.index(_T) + len(_T) :],
        )


def decision_suffix(tok, frames: Frames, row: Row) -> tuple[list[int], list[tuple[int, int]], int]:
    """Token ids of the decision suffix after a live assistant turn, with option spans and DECIDE relative to it.
    Same pieces, same order and same instruction as ``render_decision`` (training)."""
    lead, instruction = LEADS[row.qtype]
    ids = tok(
        frames.to_decision + f"Decision: {row.question}\n{lead}\n", add_special_tokens=False
    ).input_ids
    spans = []
    for o in row.options:
        s = len(ids)
        ids += tok(
            f"- {o.name}: {o.desc}\n" if o.desc else f"- {o.name}\n", add_special_tokens=False
        ).input_ids
        spans.append((s, len(ids)))
    ids += tok(instruction + frames.after_decision, add_special_tokens=False).input_ids
    return ids, spans, len(ids) - 1


@torch.no_grad()
def decide_on_cache(s1, cache, row: Row, frames: Frames) -> list[float]:
    """Probabilities over ``row.options`` from the live cache; the cache is left exactly as it was."""
    ids, spans, decide = decision_suffix(s1.tok, frames, row)
    ck = CacheCheckpoint(cache)
    dev = next(s1.trunk.parameters()).device
    x = torch.tensor([ids], device=dev)
    try:
        s1._set_mask(torch.ones(1, len(ids), device=dev))  # every token here is suffix: adapter on
        feats = s1.reader.run(
            x,
            None,
            [0],
            [len(ids)],
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(ck.length, ck.length + len(ids), device=dev),
        )
    finally:
        s1._set_mask(None)
        ck.restore(cache)
    item = {"feats": feats[0], "opt_spans": spans, "decide": decide}
    logits = s1.head(SuffixBatch.collate([item], device=next(s1.head.parameters()).device))
    return torch.softmax(logits[0, : len(row.options)].float(), -1).tolist()


@dataclass
class Turn:
    text: str
    decisions: list[dict] = field(default_factory=list)
    tokens: int = 0


class Assistant:
    """Generation with the untouched base model; ``plumb_decide`` tool calls are answered by System 1 on the live
    cache. ``sample(logits) -> token id`` defaults to greedy (a scripted sampler drives the tests)."""

    def __init__(
        self, s1, conformal_qhat: float | None = None, template_kwargs: dict | None = None
    ):
        self.s1, self.tok, self.lm = s1, s1.tok, s1.lm
        self.qhat = conformal_qhat
        self.template_kwargs = template_kwargs or {}
        self.markup = Markup.of(self.tok)
        self.frames = Frames.of(self.tok, s1.template_kwargs, self.markup)

    def _feed(self, ids: list[int], cache):
        dev = next(self.lm.parameters()).device
        start = cache.get_seq_length() if cache is not None else 0
        out = self.lm(
            input_ids=torch.tensor([ids], device=dev),
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(start, start + len(ids), device=dev),
            logits_to_keep=1,
        )
        return out.logits[0, -1], out.past_key_values

    @torch.no_grad()
    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        tools: list | None = None,
        sample: Callable[[torch.Tensor], int] | None = None,
        max_decisions: int = 8,
    ) -> Turn:
        sample = sample or (lambda z: int(z.argmax()))
        gen_eos = getattr(getattr(self.lm, "generation_config", None), "eos_token_id", None)
        eos = (
            {
                self.tok.convert_tokens_to_ids(t)
                for t in ("<|im_end|>", "<|endoftext|>")
                if hasattr(self.tok, "convert_tokens_to_ids")
            }
            | {getattr(self.tok, "eos_token_id", None)}
            | set(gen_eos if isinstance(gen_eos, list) else [gen_eos])
        )
        prompt = self.tok.apply_chat_template(
            messages,
            tools=[tool_schema(), *(tools or [])],
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs,
        )
        logits, cache = self._feed(self.tok(prompt, add_special_tokens=False).input_ids, None)
        turn, gen, handled, sampled = Turn(""), [], 0, 0
        while sampled < max_new_tokens:  # injected tool responses don't count against the budget
            t = sample(logits)
            if t in eos:
                break
            gen.append(t)
            sampled += 1
            logits, cache = self._feed([t], cache)
            text = self.tok.decode(gen, skip_special_tokens=False)
            calls = self.markup.calls(text)
            if len(calls) > handled and len(turn.decisions) < max_decisions:
                handled = len(calls)
                parsed = parse_tool_call(calls[-1])
                if parsed is None:
                    continue
                name, args = parsed
                if name != PLUMB_TOOL:
                    break  # a client tool: hand the turn back
                try:
                    row = row_from_tool_args(args)
                except (KeyError, TypeError, ValueError):
                    continue  # malformed arguments: let generation carry on
                probs = decide_on_cache(self.s1, cache, row, self.frames)
                ans = answer(row, probs)
                if self.qhat is not None:
                    ans["set"] = [row.options[k].name for k in prediction_set(probs, self.qhat)]
                turn.decisions.append(
                    {
                        "arguments": args,
                        "result": ans,
                        "context_tokens": cache.get_seq_length(),
                    }
                )
                resp = self.frames.to_tool + json.dumps(ans) + self.frames.after_tool
                resp_ids = self.tok(resp, add_special_tokens=False).input_ids
                gen += resp_ids
                logits, cache = self._feed(resp_ids, cache)
        turn.text, turn.tokens = self.tok.decode(gen, skip_special_tokens=True), len(gen)
        return turn
