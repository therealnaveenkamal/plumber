"""PlumbModel: a language-model trunk with the LM head deleted and a pointer head on top.

- from_pretrained loads any causal or image-text LM, keeps the text trunk, deletes lm_head and vision towers.
- forward(batch) -> logits over options, one forward pass, no vocabulary anywhere.
- score(state, question, options) -> logits[K]  (the frozen interface)
- LORA_TARGETS is the module list PEFT must be given; the defaults would touch 6 of 52 layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .heads import PointerHead
from .rendering import Option, Row, render

# transformers>=5.18 names (verified on the meta-device enumeration, lora_targets_enum.json):
#   linear_attention x23: in_proj, out_proj | full_attention x6: q/k/v/o_proj | moe x23: up_proj, down_proj (shared expert)
# routed experts (mixer.experts.up_proj/down_proj) are fused 3-D parameters -> not adaptable, stay frozen.
# Mamba `out_proj` is EXCLUDED on purpose: the fused training kernel (mamba_split_conv1d_scan_combined) consumes
# out_proj.weight as a raw tensor, bypassing the module, so a LoRA there is silently ignored; PEFT>=0.21 refuses it.
# `in_proj` is a real module call and is adapted. conv1d/A_log/dt_bias/D are raw parameters (tiny, frozen).
LORA_TARGETS = ["in_proj", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]


class PlumbModel(nn.Module):
    def __init__(self, trunk: nn.Module, hidden_size: int, d_proj: int = 512):
        super().__init__()
        self.trunk = trunk
        # the head lives where the trunk's LAST parameter lives (final norm) - correct for device_map="auto" sharding too
        last_dev = list(trunk.parameters())[-1].device
        self.head = PointerHead(hidden_size, d_proj).to(last_dev)

    @classmethod
    def from_lm(cls, lm: nn.Module, d_proj: int = 512) -> PlumbModel:
        """Any causal or image-text LM -> its text trunk with the LM head (and any vision tower) deleted."""
        inner = (
            getattr(lm, "model", None)
            or getattr(lm, "transformer", None)
            or getattr(lm, "base_model", None)
        )
        if inner is None:
            raise ValueError(f"cannot find the trunk on {type(lm).__name__}")
        trunk = getattr(inner, "language_model", inner)
        n_head = sum(p.numel() for p in lm.lm_head.parameters()) if hasattr(lm, "lm_head") else 0
        for name in (
            "lm_head",
            "visual",
            "vision_tower",
            "embed_vision",
            "audio_tower",
            "multi_modal_projector",
        ):
            for owner in (lm, inner):
                if hasattr(owner, name):
                    delattr(owner, name)
        m = cls(trunk, trunk.config.hidden_size, d_proj)
        m.deleted_lm_head_params = n_head
        return m

    @classmethod
    def from_pretrained(cls, name_or_path: str, d_proj: int = 512, **hf_kwargs) -> PlumbModel:
        """Load a base LM from the Hub or disk and keep only its text trunk."""
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

        cfg = AutoConfig.from_pretrained(name_or_path)
        multimodal = any(hasattr(cfg, k) for k in ("vision_config", "text_config"))
        loader = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
        lm = loader.from_pretrained(name_or_path, **hf_kwargs)
        return cls.from_lm(lm, d_proj=d_proj)

    @classmethod
    def from_merged(cls, repo_or_dir: str, d_proj: int = 512, **hf_kwargs) -> PlumbModel:
        """Load a MERGED release (trunk shards with the LM head removed + head.pt), e.g. totum-labs/plumb-nemotron-3.5-lightning-30b-a3b."""
        import os

        from huggingface_hub import hf_hub_download
        from transformers import AutoModel

        trunk = AutoModel.from_pretrained(
            repo_or_dir, **hf_kwargs
        )  # NemotronHModel (no lm_head in the checkpoint)
        m = cls(trunk, trunk.config.hidden_size, d_proj)
        head_path = (
            os.path.join(repo_or_dir, "head.pt")
            if os.path.isdir(repo_or_dir)
            else hf_hub_download(repo_or_dir, "head.pt")
        )
        m.head.load_state_dict(torch.load(head_path, map_location=next(m.head.parameters()).device))
        m.deleted_lm_head_params = 0
        return m

    @staticmethod
    def load(
        ckpt: str,
        base: str = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        d_proj: int = 512,
        **hf_kwargs,
    ) -> PlumbModel:
        """One entry point for both layouts: an adapter dir (adapter_config.json + head.pt, stacked on `base`) or a merged
        release (config.json + model shards + head.pt, local dir or HF repo id)."""
        import os

        from huggingface_hub import list_repo_files

        files = set(os.listdir(ckpt)) if os.path.isdir(ckpt) else set(list_repo_files(ckpt))
        if "adapter_config.json" in files:
            from peft import PeftModel

            m = PlumbModel.from_pretrained(base, d_proj=d_proj, **hf_kwargs)
            pin_mamba_devices(m.trunk)
            m.trunk = PeftModel.from_pretrained(m.trunk, ckpt)
            m.head.load_state_dict(
                torch.load(
                    os.path.join(ckpt, "head.pt"), map_location=next(m.head.parameters()).device
                )
            )
            return m.eval()
        m = PlumbModel.from_merged(ckpt, d_proj=d_proj, **hf_kwargs)
        pin_mamba_devices(m.trunk)
        return m.eval()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, opt_spans, decide_pos, **kw
    ):
        h = self.trunk(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kw
        ).last_hidden_state
        return self.head(h, opt_spans, decide_pos)

    @torch.no_grad()
    def score(
        self, tok, state: str, question: str, options: list[Option], qtype: str = "choice"
    ) -> torch.Tensor:
        """The frozen interface. Presented order == given order (no shuffle at inference)."""
        r = render(tok, Row("q", state, question, qtype, options), rng=None, shuffle=False)
        dev = next(self.trunk.parameters()).device  # inputs go where the FIRST trunk parameter is
        ids = torch.tensor([r["input_ids"]], device=dev)
        return self.forward(ids, torch.ones_like(ids), [r["opt_spans"]], [r["decide_pos"]])[
            0, : len(options)
        ]


def pin_mamba_devices(trunk: nn.Module) -> int:
    """mamba_ssm's Triton kernels launch on the CURRENT cuda device. Under device_map="auto" sharding a Mamba
    layer may live on cuda:1..3 while the current device is cuda:0 -> 'Pointer argument cannot be accessed from
    Triton'. Pin the current device to each mixer's own device right before it runs. Returns #hooks."""
    n = 0
    for _, mod in trunk.named_modules():
        if hasattr(mod, "conv1d") and hasattr(mod, "in_proj"):
            d = mod.in_proj.weight.device
            if d.type == "cuda":
                mod.register_forward_pre_hook(lambda m, args, _d=d: torch.cuda.set_device(_d))
                n += 1
    return n


def collate(tok, rendered: list, pad_id: int, device):
    """Right-pad a list of render() outputs into one batch."""
    L = max(len(r["input_ids"]) for r in rendered)
    ids = torch.full((len(rendered), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(rendered), L), dtype=torch.long)
    for i, r in enumerate(rendered):
        n = len(r["input_ids"])
        ids[i, :n] = torch.tensor(r["input_ids"])
        am[i, :n] = 1
    return {
        "input_ids": ids.to(device),
        "attention_mask": am.to(device),
        "opt_spans": [r["opt_spans"] for r in rendered],
        "decide_pos": [r["decide_pos"] for r in rendered],
        "gold": [r["gold"] for r in rendered],
        "teacher": [r["teacher"] for r in rendered],
        "qtype": [r["qtype"] for r in rendered],
    }
