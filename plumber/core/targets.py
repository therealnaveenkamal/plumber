"""LoRA target selection for any Hugging Face causal LM: which nn.Linear leaves to adapt, by architecture.

Known families get an explicit list; anything else falls back to every attention/MLP projection found on the trunk,
minus modules that fused kernels consume as raw weights (e.g. Mamba ``out_proj`` on NemotronH). ``coverage()`` reports
how many layers the chosen targets actually touch so a wrong list fails loudly before training.
"""

from __future__ import annotations

from torch import nn

KNOWN: dict[str, list[str]] = {
    # 23 Mamba-2 (in_proj only: out_proj feeds the fused kernel as a raw tensor), 6 attention, 23 MoE shared experts
    "nemotron_h": ["in_proj", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    # Qwen3.5 / Qwen3.8 hybrids: gated DeltaNet in_proj_* + attention + MLP
    "qwen3_5": [
        "in_proj_qkvz",
        "in_proj_ba",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "qwen3_next": [
        "in_proj_qkvz",
        "in_proj_ba",
        "out_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma4": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen3_moe": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma3_text": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "granitemoehybrid": ["in_proj", "q_proj", "k_proj", "v_proj", "o_proj"],
    "falcon_h1": [
        "in_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}
# never adapt: fused-kernel inputs, embeddings, heads, routers
NEVER = {"lm_head", "embed_tokens", "conv1d", "router", "gate", "score", "classifier"}
ATTN_MLP = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "qkv_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj",
    "out_proj",
    "in_proj_qkvz",
    "in_proj_ba",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "fc1",
    "fc2",
    "dense",
}


def linear_leaves(trunk: nn.Module) -> dict[str, int]:
    """Leaf module names of every nn.Linear in the trunk, with counts."""
    counts: dict[str, int] = {}
    for name, mod in trunk.named_modules():
        if isinstance(mod, nn.Linear):
            leaf = name.split(".")[-1]
            counts[leaf] = counts.get(leaf, 0) + 1
    return counts


def select_targets(trunk: nn.Module, model_type: str | None = None) -> list[str]:
    """Targets present on this trunk: the known list for its model_type if any, else the generic attention/MLP set."""
    present = linear_leaves(trunk)
    model_type = (model_type or "").removesuffix(
        "_text"
    )  # image-text checkpoints expose <family>_text trunks
    wanted = KNOWN.get(model_type)
    if wanted is None:
        wanted = [n for n in ATTN_MLP if n in present]
        if model_type == "nemotron_h" or "conv1d" in {
            n.split(".")[-1] for n, _ in trunk.named_modules()
        }:
            wanted = [n for n in wanted if n != "out_proj"]
    chosen = [n for n in wanted if n in present and n not in NEVER]
    if not chosen:
        raise ValueError(
            f"no LoRA targets found for model_type={model_type!r}; linear leaves: {sorted(present)}"
        )
    return chosen


def coverage(trunk: nn.Module, targets: list[str]) -> dict[str, int]:
    """Number of layers each target touches; and how many trunk layers have no adapted module at all."""
    per = {t: 0 for t in targets}
    layers_hit: set[str] = set()
    n_layers = 0
    for name, mod in trunk.named_modules():
        parts = name.split(".")
        if (
            len(parts) >= 3
            and parts[-3] == "layers"
            and parts[-2].isdigit()
            and parts[-1] == "mixer"
        ):
            pass
        if isinstance(mod, nn.Linear) and parts[-1] in per:
            per[parts[-1]] += 1
            idx = next(
                (parts[i + 1] for i, p in enumerate(parts) if p == "layers" and i + 1 < len(parts)),
                None,
            )
            if idx is not None:
                layers_hit.add(idx)
    for name, _ in trunk.named_modules():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit():
            n_layers += 1
    per["_layers_adapted"] = len(layers_hit)
    per["_layers_total"] = n_layers
    return per
