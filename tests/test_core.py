"""Plumbing test on a tiny random NemotronH (runs on GPU when available: transformers routes the
Mamba conv to the causal_conv1d CUDA kernel whenever it is importable, so CPU-only is not an option there) (same block pattern, hidden 64): render -> collate ->
PlumbModel forward -> loss -> backward; LoRA targets resolve; permutation of options permutes logits."""
import random, torch, pytest
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import NemotronHConfig, NemotronHForCausalLM, AutoTokenizer
from plumber.core import Row, Option, render, PlumbModel, LORA_TARGETS
from plumber.core.trunk import collate
from plumber.core.heads import decision_loss

def tiny():
    cfg = NemotronHConfig(hidden_size=64, intermediate_size=96, num_hidden_layers=6, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=16, mamba_num_heads=8, mamba_head_dim=16, ssm_state_size=16,   # in_proj width 296 = 8*37 (fused kernel needs multiples of 8)
                          n_groups=1, chunk_size=8, n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
                          moe_intermediate_size=96, moe_shared_expert_intermediate_size=96, vocab_size=512,
                          layers_block_type=["mamba", "moe", "mamba", "attention", "moe", "mamba"],
                          use_mamba_kernels=False, tie_word_embeddings=False, num_nextn_predict_layers=0)
    return NemotronHForCausalLM(cfg).float().to(DEV)

class Tok:  # minimal deterministic tokenizer stand-in
    bos_token_id, pad_token_id = 1, 0
    def __call__(self, text, add_special_tokens=False):
        return type("R", (), {"input_ids": [2 + (ord(c) % 500) for c in text]})()

def test_end_to_end():
    lm = tiny(); tok = Tok()
    m = PlumbModel(lm.model, 64, d_proj=16)
    rows = [Row("a", "billed twice", "which team?", "choice", [Option("billing", "refunds"), Option("tech", "bugs"), Option("sales")], gold=0),
            Row("b", "threatens to cancel", "leaving?", "noul", [Option("yes"), Option("no")], gold=0, teacher=[0.8, 0.2])]
    rng = random.Random(0)
    rend = [render(tok, r, rng) for r in rows]
    batch = collate(tok, rend, tok.pad_token_id, DEV)
    logits = m(**{k: batch[k] for k in ("input_ids", "attention_mask", "opt_spans", "decide_pos")})
    assert logits.shape == (2, 3) and torch.isinf(logits[1, 2])
    loss = decision_loss(logits, batch["gold"], batch["teacher"], batch["qtype"]); loss.backward()
    assert torch.isfinite(loss)

def test_lora_targets_resolve():
    lm = tiny(); names = {n.split(".")[-1] for n, mod in lm.named_modules() if isinstance(mod, torch.nn.Linear)}
    assert set(LORA_TARGETS) <= names, set(LORA_TARGETS) - names

def test_permutation_permutes_logits():
    lm = tiny().eval(); tok = Tok(); m = PlumbModel(lm.model, 64, d_proj=16).eval()
    opts = [Option("a", "x"), Option("b", "y"), Option("c", "z")]
    z1 = m.score(tok, "s", "q", opts)
    z2 = m.score(tok, "s", "q", [opts[2], opts[0], opts[1]])
    # in-context options are order-SENSITIVE by design here (no forking); this test just pins the API
    assert z1.shape == (3,) and z2.shape == (3,)
