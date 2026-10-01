"""Plumbed-model packaging and the OpenAI-facing pieces (no vLLM needed): package -> config/weights/plumb files and a
System1 roundtrip; the stream splitter (thinking, hidden tool markup, split tags); the System 1 announcement; message
normalisation; stated-decision parsing and option resolution."""

import json
import os

import torch

from plumbify.plumbed import package
from plumbify.serving.vllm.decisions import find_decision, pick, resolve
from plumbify.system1 import System1
from tests.test_system1 import rows, system1


def test_package_roundtrip(tmp_path):
    s1 = system1(0).eval()
    s1.attach_suffix_lora(r=4, alpha=8, dropout=0.0)
    base = tmp_path / "base"
    s1.lm.save_pretrained(base)  # a local "base model"
    s1.save(str(tmp_path / "plumb"), base_model=str(base))
    out = package(str(tmp_path / "plumb"), str(tmp_path / "plumbed"))
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["architectures"] == ["PlumbNemotronHForCausalLM"]
    assert (
        cfg["plumb"]["base_architecture"] == "NemotronHForCausalLM"
        and cfg["plumb"]["suffix_adapter"]
    )
    weights = [
        f
        for f in os.listdir(out)
        if f.endswith(".safetensors") and not f.startswith(("head", "suffix"))
    ]
    # linked, not copied
    assert weights and all(os.path.islink(os.path.join(out, f)) for f in weights)
    # a single-file base gets an index, so loaders don't read the plumb's own .safetensors as base weights
    index = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    assert set(index.values()) == set(weights)
    for f in (
        "plumb.json",
        "head.safetensors",
        "suffix_adapter.json",
        "suffix_adapter.safetensors",
        "README.md",
    ):
        assert os.path.exists(os.path.join(out, f)), f
    fresh = system1(0).eval()  # the same random base, without the adapter
    s2 = System1.load(out, lm=fresh.lm, tok=fresh.tok).float()
    for a, b in zip(s1.decide(rows()), s2.decide(rows()), strict=True):
        assert max(abs(x - y) for x, y in zip(a, b, strict=True)) < 1e-5
    assert torch.is_tensor(next(s2.head.parameters()))


def test_stream_split():
    from plumbify.serving.vllm.openai import Split

    sp = Split()
    sp.start(in_think=True)  # Qwen-style: the prompt already opened <think>
    stream = [
        "I should",
        " check.</th",
        "ink>\n\nLet me decide <tool_c",
        'all>{"name": "x"}</tool_call>',
        " Done <|im_end|>",
    ]
    out = [p for d in stream for p in sp.feed(d)] + sp.feed("", final=True)
    reasoning = "".join(t for k, t in out if k == "reasoning")
    content = "".join(t for k, t in out if k == "content")
    assert reasoning == "I should check."
    assert content == "\n\nLet me decide  Done "
    assert "tool_call" not in content and "im_end" not in content


def test_announce_and_normalize():
    from plumbify.serving.vllm.client import Decider
    from plumbify.serving.vllm.openai import PLUMB_NOTE, announce

    msgs = announce([{"role": "user", "content": "hi"}])
    assert msgs[0] == {"role": "system", "content": PLUMB_NOTE}
    msgs = announce([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}])
    assert msgs[0]["content"].startswith("Be brief.") and PLUMB_NOTE in msgs[0]["content"]
    hist = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
            ],
        }
    ]
    norm = Decider.normalize(hist)
    assert norm[0]["content"] == "" and norm[0]["tool_calls"][0]["function"]["arguments"] == {
        "a": 1
    }
    # the caller's copy is untouched
    assert hist[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_stated_decisions():
    text = (
        "Ticket: charged twice.\n\nWhich team should handle this?\n- billing: invoices, refunds\n"
        "- technical: bugs\n- sales: deals"
    )
    d = find_decision(text)
    assert d.question == "Which team should handle this?" and d.before == "Ticket: charged twice."
    assert [n for n, _ in d.options] == ["billing", "technical", "sales"] and d.qtype == "choice"
    assert resolve(["Billing", "technical", "sales"], d) == d.options
    assert resolve(["apple", "banana"], d) is None
    assert pick("reasons...\nAnswer: billing", ["billing", "technical", "sales"]) == "billing"
    assert find_decision("just chatting, no options here") is None


def test_gemma_markup():
    from plumbify.branch import Markup, parse_tool_call
    from plumbify.serving.vllm.openai import Split

    m = Markup("<|tool_call>", "<tool_call|>", "<|channel>thought", "<channel|>")
    body = (
        'call:plumb_decide{options:[{description:<|"|>refunds, invoices: all<|"|>,name:<|"|>billing<|"|>},'
        '{description:<|"|><|"|>,name:<|"|>sales<|"|>}],question:<|"|>Which team {now}?<|"|>,k:2,x:true}'
    )
    name, args = parse_tool_call(body)
    assert name == "plumb_decide" and args["question"] == "Which team {now}?"
    assert args["options"][0] == {"description": "refunds, invoices: all", "name": "billing"}
    assert args["k"] == 2 and args["x"] is True
    assert parse_tool_call('call:f{a:<|"|>unclosed}') is None
    text = f"<|channel>thought\nhmm<channel|>Checking.<|tool_call>{body}<tool_call|>"
    assert m.calls(text) == [body]
    assert m.final(text + "Answer: billing") == "Answer: billing"
    assert m.opens_thinking("<tool_response|><|channel>thought\n")
    assert not m.opens_thinking("<|turn>model\n<|channel>thought\n<channel|>")

    sp = Split(m, specials=["<turn|>", "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>"])
    sp.start(in_think=False)
    stream = [
        "<|chan",
        "nel>thought\nweigh",
        " it<chan",
        "nel|>Asking.<|tool_",
        "call>" + body[:20],
        body[20:] + "<tool_call|>",
        " Done<turn|>",
    ]
    out = [p for d in stream for p in sp.feed(d)] + sp.feed("", final=True)
    assert "".join(t for k, t in out if k == "reasoning") == "weigh it"
    assert "".join(t for k, t in out if k == "content") == "Asking. Done"
