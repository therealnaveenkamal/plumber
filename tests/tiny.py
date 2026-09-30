"""A tiny random hybrid model for CPU tests: NemotronH with Mamba, attention and MoE blocks, hidden size 64. Its layer
mix covers every cache type a plumb has to handle (recurrent state, KV, experts)."""

import torch
from transformers import NemotronHConfig, NemotronHForCausalLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tiny():
    cfg = NemotronHConfig(
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        mamba_num_heads=8,
        mamba_head_dim=16,
        ssm_state_size=16,  # in_proj width 296 = 8*37 (fused kernel needs multiples of 8)
        n_groups=1,
        chunk_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=96,
        moe_shared_expert_intermediate_size=96,
        vocab_size=512,
        layers_block_type=["mamba", "moe", "mamba", "attention", "moe", "mamba"],
        use_mamba_kernels=False,
        tie_word_embeddings=False,
        num_nextn_predict_layers=0,
    )
    return NemotronHForCausalLM(cfg).float().to(DEV)
