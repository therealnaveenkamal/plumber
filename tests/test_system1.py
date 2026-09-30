"""Head-only System 1 on a tiny random hybrid trunk (Mamba + attention + MoE): chat-framed render, tapped features,
shared-context cache fork == full forward, training moves only the head, artifact roundtrip, base weights untouched."""

import hashlib
import random

import torch

from plumber.core.row import Option, Row
from plumber.system1 import System1
from tests.tiny import tiny


class ChatTok:
    """Deterministic char-level tokenizer with a ChatML template."""

    bos_token_id, pad_token_id = None, 0

    def __call__(self, text, add_special_tokens=False):
        return type("R", (), {"input_ids": [2 + (ord(c) % 500) for c in text]})()

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 2) for i in ids)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_tensors=None,
        enable_thinking=False,
    ):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        if not tokenize and return_tensors is None:
            return text
        ids = self(text).input_ids
        return torch.tensor([ids]) if return_tensors == "pt" else ids


STATE = (
    "Invoice 4411 was billed twice. The customer asks for a refund today or they cancel the account. "
    * 2
)


def rows():
    return [
        Row(
            "team",
            STATE,
            "Which team handles this?",
            "choice",
            [
                Option("billing", "invoices, refunds"),
                Option("technical", "bugs"),
                Option("sales", "new deals"),
            ],
            0,
        ),
        Row(
            "angry",
            STATE,
            "Is the customer threatening to leave?",
            "noul",
            [Option("no"), Option("yes")],
            1,
        ),
        Row(
            "urgency",
            STATE,
            "How urgent?",
            "score",
            [Option("0", "later"), Option("1", "soon"), Option("2", "now")],
            2,
        ),
    ]


def system1(seed=0):
    torch.manual_seed(seed)
    return System1.new(
        tiny(), ChatTok(), taps=[1, 3, -1], d_h=32, n_heads=4, d_proj=16, dropout=0.0
    ).float()


def test_render_spans():
    s1 = system1()
    tok = s1.tok
    r = rows()[0]
    rd = s1.render(r, rng=random.Random(3))
    texts = [tok.decode(rd.input_ids[a:b]) for a, b in rd.opt_spans]
    assert texts == [f"- {r.options[j].name}: {r.options[j].desc}\n" for j in rd.perm]
    assert rd.perm[rd.gold] == r.gold
    assert tok.decode(rd.input_ids[: rd.suffix_start]).endswith(STATE + "\n\n")
    assert tok.decode(rd.input_ids[rd.decide_pos :]) == "\n"  # last token of the assistant-open


def test_shared_context_matches_full_forward():
    s1 = system1().eval()
    rend = [s1.render(r) for r in rows()]
    # same state -> one prefill, forked cache (verified against a full forward)
    shared = s1.features(rend)
    assert s1._fork_ok is True
    full = s1._full(rend)
    for a, b in zip(shared, full, strict=True):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=1e-3)


def test_conversation_context():
    s1 = system1().eval()
    ctx = [
        {"role": "user", "content": STATE},
        {"role": "assistant", "content": "I can help with that invoice."},
    ]
    q = Row(
        "team", "", "Which team handles this?", "choice", [Option("billing"), Option("technical")]
    )
    rd = s1.render(q, ctx)
    text = s1.tok.decode(rd.input_ids)
    assert text.index("I can help") < rd.suffix_start <= text.index("Decision:")
    assert len(s1.decide([q, q], context=ctx)[0]) == 2


def test_training_moves_only_the_head():
    s1 = system1()
    base = hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in s1.lm.parameters())
    ).hexdigest()
    opt = torch.optim.AdamW(s1.head.parameters(), lr=3e-3)
    rng = random.Random(0)
    losses = []
    for _ in range(25):
        rend = [s1.render(r, rng=rng) for r in rows()]
        logits = s1.logits(rend)
        loss = torch.nn.functional.cross_entropy(
            logits, torch.tensor([r.gold for r in rend], device=logits.device)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5
    assert all(p.grad is None for p in s1.lm.parameters())
    after = hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in s1.lm.parameters())
    ).hexdigest()
    assert base == after


def test_artifact_roundtrip(tmp_path):
    s1 = system1().eval()
    s1.head.temperature.fill_(1.7)
    s1.save(str(tmp_path), base_model="tiny-nemotron-h")
    s2 = System1.load(str(tmp_path), lm=s1.lm, tok=s1.tok).float()
    assert s2.reader.taps == [1, 3, -1]
    for a, b in zip(s1.decide(rows()), s2.decide(rows()), strict=True):
        assert max(abs(x - y) for x, y in zip(a, b, strict=True)) < 1e-6
