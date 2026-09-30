"""Suffix-only LoRA: context tokens keep the base model's exact states, the suffix adapts, a base-weights context cache
plus an adapted suffix continuation equals the masked full forward, training moves only adapter + head, roundtrip."""

import hashlib
import random

import torch

from tests.test_system1 import rows, system1


def adapted(seed=0):
    s1 = system1(seed).eval()
    lora = s1.attach_suffix_lora(r=4, alpha=8, dropout=0.0)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in s1.trunk.named_parameters():
            if ".lora_B" in n:  # LoRA B starts at zero; make the adapter do something
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    return s1, lora


def base_hash(s1):
    ps = [p for n, p in s1.lm.named_parameters() if ".lora_" not in n]
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in ps)).hexdigest()


def test_context_untouched_suffix_adapted():
    s1, lora = adapted()
    rd = s1.render(rows()[0])
    ids = torch.tensor([rd.input_ids])
    with torch.no_grad():
        lora.mask = torch.zeros(1, ids.shape[1])
        base = s1.trunk(input_ids=ids).last_hidden_state
        lora.mask = lora.suffix_mask([rd.suffix_start], ids.shape[1], ids.device)
        mixed = s1.trunk(input_ids=ids).last_hidden_state
    P = rd.suffix_start
    torch.testing.assert_close(mixed[:, :P], base[:, :P], atol=0, rtol=0)
    assert (mixed[:, P:] - base[:, P:]).abs().max() > 1e-3


def test_cached_base_context_matches_masked_full_forward():
    s1, _ = adapted(1)
    rend = [s1.render(r) for r in rows()]  # one state, three questions
    # context prefilled on base weights, suffixes continued with the adapter
    forked = s1.features(rend)
    assert s1._fork_ok is True
    full = s1._full(rend)  # one masked forward per row
    for a, b in zip(forked, full, strict=True):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=1e-3)


def test_training_moves_only_adapter_and_head():
    s1 = system1(2)
    s1.attach_suffix_lora(r=4, alpha=8, dropout=0.0)
    before = base_hash(s1)
    params = list(s1.head.parameters()) + s1.suffix_lora.parameters()
    opt = torch.optim.AdamW(params, lr=3e-3)
    rng, losses = random.Random(0), []
    for _ in range(20):
        rend = [s1.render(r, rng=rng) for r in rows()]
        logits = s1.logits(rend, s1.features(rend, grad=True))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([r.gold for r in rend]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5
    assert all(p.grad is not None for p in s1.suffix_lora.parameters())
    assert base_hash(s1) == before


def test_roundtrip(tmp_path):
    s1, _ = adapted(3)
    s1.save(str(tmp_path), base_model="tiny")
    fresh = system1(3).eval()  # same random base, no adapter
    from plumber.system1 import System1

    s2 = System1.load(str(tmp_path), lm=fresh.lm, tok=fresh.tok).float()
    assert s2.suffix_lora is not None
    for a, b in zip(s1.decide(rows()), s2.decide(rows()), strict=True):
        assert max(abs(x - y) for x, y in zip(a, b, strict=True)) < 1e-5


def test_generation_path_is_the_base_model():
    """After a decision, the loaded model generates with base weights: no stale mask, no adapter."""
    s1, lora = adapted(4)
    x = torch.tensor([s1.render(rows()[0]).input_ids])
    with torch.no_grad():
        lora.mask = torch.zeros(1, x.shape[1])  # adapter off: reference base logits
        ref = s1.lm(input_ids=x).logits
        s1.decide(rows())
        assert lora.mask is None
        torch.testing.assert_close(s1.lm(input_ids=x).logits, ref, atol=0, rtol=0)
