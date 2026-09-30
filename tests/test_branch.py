"""System 1 inside generation: a decision on the live cache equals the same decision over the full text, the cache is
restored exactly, and a scripted generation that emits plumb_decide gets its answer and continues on the same cache."""

import json

import pytest
import torch

from plumber.branch import Assistant, CacheCheckpoint, Frames, decide_on_cache, decision_suffix
from plumber.core.decision_head import SuffixBatch
from plumber.core.row import Option, Row
from plumber.system1 import System1
from tests.test_system1 import ChatTok

qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")


class QwenTok(ChatTok):
    """Char-level tokenizer with a Qwen3-style template: tool turns, tools kwarg, empty think block."""

    eos_token_id = 1

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_tensors=None,
        enable_thinking=False,
        tools=None,
    ):
        out = ""
        for m in messages:
            if m["role"] == "tool":
                out += f"<|im_start|>user\n<tool_response>\n{m['content']}\n</tool_response><|im_end|>\n"
            else:
                out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n" + (
                "" if enable_thinking else "<think>\n\n</think>\n\n"
            )
        return out


def setup(seed=0):
    from transformers import Qwen3_5TextConfig

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        vocab_size=512,
        layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
    )
    lm = qwen.Qwen3_5ForCausalLM(cfg).float().eval()
    s1 = System1.new(
        lm, QwenTok(), taps=[1, 2, -1], d_h=32, n_heads=4, d_proj=16, dropout=0.0
    ).eval()
    s1.attach_suffix_lora(r=4, alpha=8, dropout=0.0)
    with torch.no_grad():
        for n, p in s1.trunk.named_parameters():
            if ".lora_B" in n:
                p.normal_(0, 0.5)
    return s1


ROW = Row(
    "q",
    "",
    "Which team handles this?",
    "choice",
    [Option("billing", "invoices"), Option("technical", "bugs"), Option("sales", "deals")],
)
CONTEXT = "<|im_start|>user\nI was charged twice for invoice 4411, fix it today.<|im_end|>\n<|im_start|>assistant\nLet me check."


def prefill(s1, ids):
    out = s1.lm(input_ids=torch.tensor([ids]), use_cache=True)
    return out.past_key_values, out.logits[0, -1]


def test_decision_on_cache_equals_full_text():
    s1 = setup()
    frames = Frames.of(s1.tok, s1.template_kwargs)
    ctx = s1.tok(CONTEXT).input_ids
    cache, _ = prefill(s1, ctx)
    live = decide_on_cache(s1, cache, ROW, frames)
    # the same tokens in one masked forward: context on base weights, suffix with the adapter
    suf, spans, decide = decision_suffix(s1.tok, frames, ROW)
    ids = torch.tensor([ctx + suf])
    s1._set_mask(s1.suffix_lora.suffix_mask([len(ctx)], ids.shape[1], ids.device))
    with torch.no_grad():
        feats = s1.reader.run(ids, None, [len(ctx)], [ids.shape[1]], use_cache=False)
        logits = s1.head(
            SuffixBatch.collate([{"feats": feats[0], "opt_spans": spans, "decide": decide}])
        )
    s1._set_mask(None)
    full = torch.softmax(logits[0], -1).tolist()
    assert max(abs(a - b) for a, b in zip(live, full, strict=True)) < 1e-4


def test_cache_restored_exactly():
    s1 = setup(1)
    frames = Frames.of(s1.tok, s1.template_kwargs)
    ctx = s1.tok(CONTEXT).input_ids
    nxt = torch.tensor([[s1.tok(" ").input_ids[0]]])
    cache_a, _ = prefill(s1, ctx)
    cache_b, _ = prefill(s1, ctx)
    before = CacheCheckpoint(cache_b)
    decide_on_cache(s1, cache_b, ROW, frames)
    assert cache_b.get_seq_length() == before.length
    with torch.no_grad():
        la = s1.lm(input_ids=nxt, past_key_values=cache_a, use_cache=True).logits
        lb = s1.lm(input_ids=nxt, past_key_values=cache_b, use_cache=True).logits
    torch.testing.assert_close(la, lb, atol=0, rtol=0)


def test_generation_with_a_mid_stream_decision():
    s1 = setup(2)
    tok = s1.tok
    call = {
        "name": "plumb_decide",
        "arguments": {
            "question": "Which team handles this?",
            "options": [
                {"name": "billing", "description": "invoices"},
                {"name": "technical", "description": "bugs"},
                {"name": "sales", "description": "deals"},
            ],
        },
    }
    script = tok(f"Let me route this. <tool_call>\n{json.dumps(call)}\n</tool_call>").input_ids
    tail = tok("Routed.").input_ids
    state = {"i": 0, "after": False}

    # teacher-forced: the tool call, then (after the tool response) a short reply, then EOS
    def sample(
        _logits,
    ):
        if state["i"] < len(script):
            state["i"] += 1
            return script[state["i"] - 1]
        seq = [*tail, tok.eos_token_id]
        k = state["i"] - len(script)
        state["i"] += 1
        return seq[min(k, len(seq) - 1)]

    msgs = [{"role": "user", "content": "I was charged twice for invoice 4411, fix it today."}]
    turn = Assistant(s1, conformal_qhat=0.5).chat(msgs, max_new_tokens=400, sample=sample)
    assert len(turn.decisions) == 1
    res = turn.decisions[0]["result"]
    assert (
        res["type"] == "choice"
        and abs(sum(res["probabilities"].values()) - 1) < 1e-4
        and res["set"]
    )
    assert "<tool_response>" in turn.text and turn.text.rstrip().endswith("Routed.")
    # the in-generation decision equals a decision made from scratch over the same tokens
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
    cache, _ = prefill(s1, tok(prompt).input_ids + script)
    fresh = decide_on_cache(
        s1,
        cache,
        Row(
            "q",
            "",
            call["arguments"]["question"],
            "choice",
            [Option(o["name"], o["description"]) for o in call["arguments"]["options"]],
        ),
        Frames.of(tok, s1.template_kwargs),
    )
    assert max(abs(a - b) for a, b in zip(fresh, res["probabilities"].values(), strict=True)) < 1e-4


def test_parse_both_tool_call_formats():
    from plumber.branch import parse_tool_call

    j = parse_tool_call(
        '{"name": "plumb_decide", "arguments": {"question": "Q?", "options": [{"name": "a"}]}}'
    )
    x = parse_tool_call(
        "<function=plumb_decide>\n<parameter=question>\nQ?\n</parameter>\n<parameter=type>\nchoice\n</parameter>\n"
        '<parameter=options>\n[{"name": "a"}]\n</parameter>\n</function>'
    )
    assert j == ("plumb_decide", {"question": "Q?", "options": [{"name": "a"}]})
    assert x == ("plumb_decide", {"question": "Q?", "type": "choice", "options": [{"name": "a"}]})
    assert parse_tool_call("not a call") is None
