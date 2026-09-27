"""Plumbify a tiny Qwen3.5 text trunk (Gated DeltaNet + attention): trunk extraction, LoRA targets, forward, shared prefix."""

import random

import pytest
import torch
from torch import nn

from plumber.core import Option, PlumbModel, Row, render
from plumber.core.targets import coverage, select_targets
from plumber.core.trunk import collate
from plumber.engine import Plumber
from tests.test_core import DEV, Tok

qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")


def tiny_qwen():
    from transformers import Qwen3_5TextConfig

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
    return qwen.Qwen3_5TextModel(cfg).float().to(DEV)


class FakeVL(
    nn.Module
):  # the shape of Qwen3_5ForConditionalGeneration / Gemma4ForConditionalGeneration
    def __init__(self, trunk):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = trunk
        self.model.visual = nn.Linear(4, 4)
        self.lm_head = nn.Linear(64, 512, bias=False)


def test_from_lm_keeps_only_the_text_trunk():
    trunk = tiny_qwen()
    lm = FakeVL(trunk)
    m = PlumbModel.from_lm(lm, d_proj=16)
    assert m.trunk is trunk and not hasattr(lm, "lm_head") and not hasattr(lm.model, "visual")
    assert m.deleted_lm_head_params == 64 * 512


def test_targets_cover_every_layer():
    trunk = tiny_qwen()
    targets = select_targets(trunk, trunk.config.model_type)
    assert "in_proj_qkv" in targets and "q_proj" in targets and "down_proj" in targets
    cov = coverage(trunk, targets)
    assert cov["_layers_adapted"] == cov["_layers_total"] == 4


def test_forward_and_shared_prefix():
    m = PlumbModel(tiny_qwen(), 64, d_proj=16).eval()
    tok = Tok()
    rows = [
        Row(
            "a",
            "billed twice for march",
            "which team?",
            "choice",
            [Option("billing", "refunds"), Option("tech", "bugs")],
            gold=0,
        ),
        Row(
            "b", "billed twice for march", "leaving?", "noul", [Option("no"), Option("yes")], gold=0
        ),
    ]
    rend = [render(tok, r, random.Random(0), shuffle=False) for r in rows]
    b = collate(tok, rend, tok.pad_token_id, DEV)
    with torch.no_grad():
        full = m(b["input_ids"], b["attention_mask"], b["opt_spans"], b["decide_pos"])
    assert full.shape == (2, 2) and torch.isfinite(full).all()
    eng = Plumber.__new__(Plumber)
    eng.tok, eng.pad, eng.max_len, eng.model, eng.device = tok, 0, 4096, m, torch.device(DEV)
    with torch.no_grad():
        shared = eng._forward_shared_prefix(rend)
    pf, ps = torch.softmax(full[:, :2], -1), torch.softmax(shared[:, :2], -1)
    assert (pf - ps).abs().max() < 0.02, (pf, ps)
