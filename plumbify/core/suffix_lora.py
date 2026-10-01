"""Suffix-only LoRA: the adapter acts on the decision suffix tokens and nothing else.

Every adapted linear layer computes ``base(x) + mask * B(A(x))`` with ``mask`` = 1 on decision-suffix tokens (question,
options, DECIDE) and 0 on context tokens (conversation / state). So:

- context tokens are processed by the untouched base weights: their KV and recurrent state are exactly the base model's,
  shareable with a generator that has already read the same conversation;
- generation (no suffix) is bit-identical to the base model; the adapter is never merged, and with no mask set it
  contributes nothing (System1 clears the mask after every inference forward);
- decision tokens get a trained trunk, reading the base model's context through adapted projections: the computation a
  frozen trunk plus head cannot learn.

The mask is applied by a forward hook on every ``lora_B`` module, so it holds through gradient checkpointing (the hook
reruns on recompute with the same mask). Adapters are injected in place (no PeftModel wrapper), so layer paths and tap
hooks are unchanged. Files: ``suffix_adapter.json`` (LoRA config) + ``suffix_adapter.safetensors`` (weights).
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn

from .targets import coverage, select_targets

CONFIG, WEIGHTS = "suffix_adapter.json", "suffix_adapter.safetensors"


class SuffixLoRA:
    def __init__(
        self,
        trunk: nn.Module,
        r: int = 32,
        alpha: int = 64,
        dropout: float = 0.05,
        targets: list[str] | None = None,
    ):
        from peft import LoraConfig, inject_adapter_in_model

        mtype = getattr(trunk.config, "model_type", None)
        self.targets = list(targets or select_targets(trunk, mtype))
        cov = coverage(trunk, self.targets)
        if cov["_layers_total"] and cov["_layers_adapted"] < 0.9 * cov["_layers_total"]:
            raise ValueError(
                f"LoRA targets touch {cov['_layers_adapted']}/{cov['_layers_total']} layers"
            )
        self.cfg = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
            "target_modules": self.targets,
        }
        inject_adapter_in_model(LoraConfig(bias="none", **self.cfg), trunk)
        self.trunk = trunk
        self.mask: torch.Tensor | None = (
            None  # [B, L] for a decision forward; None = adapter off (base model)
        )
        self.lora_B = [m for n, m in trunk.named_modules() if n.endswith("lora_B.default")]
        if not self.lora_B:
            raise ValueError(f"no LoRA modules were injected (targets={self.targets})")
        self._hooks = [m.register_forward_hook(self._gate) for m in self.lora_B]
        dev = next(trunk.parameters()).device
        for p in self.parameters():  # injected in fp32 on CPU by default; live next to the trunk
            p.data = p.data.to(dev)

    def _gate(self, _mod, _args, out):
        m = self.mask
        if m is None:  # no decision in flight: the base model, e.g. for generation
            return out * 0
        if out.dim() == 3 and out.shape[:2] == m.shape:
            return out * m.unsqueeze(-1).to(out.dtype)
        if out.dim() == 2 and out.shape[0] == m.numel():
            return out * m.reshape(-1, 1).to(out.dtype)
        raise RuntimeError(
            f"suffix mask {tuple(m.shape)} does not fit adapter output {tuple(out.shape)}"
        )

    def parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.trunk.named_parameters() if ".lora_" in n]

    def trainable(self, on: bool = True) -> SuffixLoRA:
        for p in self.parameters():
            p.requires_grad_(on)
        return self

    @staticmethod
    def suffix_mask(suffix_starts: list[int], length: int, device) -> torch.Tensor:
        pos = torch.arange(length, device=device)
        return (pos[None, :] >= torch.tensor(suffix_starts, device=device)[:, None]).float()

    def save(self, out: str) -> None:
        from safetensors.torch import save_file

        sd = {
            n: p.detach().contiguous().cpu()
            for n, p in self.trunk.named_parameters()
            if ".lora_" in n
        }
        save_file(sd, os.path.join(out, WEIGHTS))
        with open(os.path.join(out, CONFIG), "w") as f:
            json.dump({**self.cfg, "scope": "suffix"}, f, indent=1)

    @classmethod
    def load(cls, trunk: nn.Module, plumb: str) -> SuffixLoRA:
        from safetensors.torch import load_file

        from ..artifact import fetch

        cfg = json.load(open(fetch(plumb, CONFIG)))
        lora = cls(trunk, cfg["r"], cfg["lora_alpha"], cfg["lora_dropout"], cfg["target_modules"])
        sd = load_file(fetch(plumb, WEIGHTS), device=str(next(trunk.parameters()).device))
        params = dict(trunk.named_parameters())
        missing = [n for n in params if ".lora_" in n and n not in sd]
        if missing or any(n not in params for n in sd):
            raise RuntimeError(
                f"suffix adapter mismatch: {len(missing)} missing, "
                f"{sum(n not in params for n in sd)} unexpected"
            )
        with torch.no_grad():
            for n, t in sd.items():
                params[n].copy_(t)
        return lora.trainable(False)
