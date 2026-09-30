"""Plumbed models in vLLM: any supported base architecture, generating normally and deciding from its own KV cache.

A plumbed model directory (``plumber plumbify`` / ``plumber export``) is an ordinary model directory whose
``config.json`` names ``Plumb<BaseArchitecture>`` and which also holds the plumb (``plumb.json``, ``head.safetensors``,
``suffix_adapter.*``). ``Plumb<Arch>`` is created on demand here as a subclass of vLLM's own class for ``<Arch>``; the
base weights load through the base class unchanged, then the plumb is attached:

- the suffix adapter is stacked onto vLLM's fused linear layers (the fusion is read from the base class's
  ``packed_modules_mapping``) as ``base(x) + mask * B(A(x))``; the per-token mask (hooks.py) is 1 only on decision
  suffix tokens, so generation and every context token run on the base weights exactly;
- the tapped layers (vLLM's EAGLE-3 aux hidden states + the final output) are copied to a buffer each forward;
- a decision request (one-token generation with ``extra_args["plumb"]``) gets its answer through ``compute_logits``:
  option k is token id k, everything else -inf; the client reads the probabilities from ``logprobs``.

vLLM-internal pieces are marked; written against vLLM v0.30.0.
"""

from __future__ import annotations

import json
import logging
import os
import re

import torch
from torch import nn

from ...plumbed import PLUMB_PREFIX

logger = logging.getLogger("vllm.plumber")  # inherits vLLM's log handlers and format

_KEY = re.compile(r"^layers\.(\d+)\.(.+)\.lora_([AB])\.default\.weight$")
MISSING = (
    "missing"  # sentinel: some suffix rows came from the prefix cache and never reached the head
)


def find_plumb_dir(vllm_config) -> str:
    """A plumbed model carries its plumb in its own directory (or Hub repo)."""
    from ...plumbed import plumb_dir

    model = vllm_config.model_config.model
    path = plumb_dir(model)
    if path is None:
        raise ValueError(
            f"{model} is not a plumbed model (no plumb.json); run `plumber plumbify` first"
        )
    return path


class PlumbState:
    """Per-worker decision state: request metadata, buffers, the head. Held as a plain attribute (not a submodule),
    so vLLM's weight loader and torch.compile never see it."""

    def __init__(self, path: str, max_tokens: int, hidden: int, device, dtype):
        from safetensors.torch import load_file

        from ...artifact import read_spec
        from ...core.decision_head import DecisionHead, HeadConfig

        self.path, self.spec = path, read_spec(path)
        self.taps = list(self.spec.taps)
        # vLLM builds models under a bf16 default dtype; the readout runs in fp32
        self.head = DecisionHead(HeadConfig(**self.spec.head_config)).to(
            device=device, dtype=torch.float32
        )
        self.head.load_state_dict(
            load_file(os.path.join(path, "head.safetensors"), device=str(device))
        )
        self.head.temperature.fill_(self.spec.calibration.temperature)
        self.head.eval()
        self.max_tokens = max_tokens
        self.mask = torch.zeros(max_tokens, 1, dtype=dtype, device=device)
        self.tap_buf = torch.zeros(max_tokens, len(self.taps), hidden, dtype=dtype, device=device)
        self.meta: dict[str, dict] = {}  # req_id -> {"suffix_start", "spans", "decide", "k"}
        self.rows: dict[
            str, list[tuple[int, torch.Tensor]]
        ] = {}  # req_id -> [(first position, [m, T, d])]
        self.step: list[
            tuple
        ] = []  # this step's decision requests: (batch idx, req_id, start, n, pos0, done)
        self.override: dict[
            int, tuple[int, torch.Tensor | str]
        ] = {}  # logits row -> (k, logits | MISSING)
        self._mask_dirty = 0

    def forget(self, req_id: str) -> None:
        self.meta.pop(req_id, None)
        self.rows.pop(req_id, None)

    # called by the runner hooks -------------------------------------------------------------------------------

    def on_batch(self, ib) -> None:
        """Right before the forward: the per-token adapter mask for this batch (1 on decision-suffix tokens)."""
        import numpy as np

        self.step = []
        n = int(ib.num_tokens)
        mask = None
        for i, rid in enumerate(ib.req_ids):
            m = self.meta.get(rid)
            if m is None:
                continue
            s, cnt = int(ib.query_start_loc_np[i]), int(ib.num_scheduled_tokens[i])
            pos0 = int(ib.num_computed_tokens_np[i])
            lo = max(m["suffix_start"] - pos0, 0)
            if lo < cnt:
                if mask is None:
                    mask = np.zeros(n, dtype=np.float32)
                mask[s + lo : s + cnt] = 1.0
            self.step.append((i, rid, s, cnt, pos0, pos0 + cnt >= int(ib.prefill_len_np[i])))
        if mask is not None:
            self.mask[:n, 0].copy_(torch.from_numpy(mask), non_blocking=True)
            if self._mask_dirty > n:
                self.mask[n : self._mask_dirty].zero_()
            self._mask_dirty = n
        elif self._mask_dirty:
            self.mask[: self._mask_dirty].zero_()
            self._mask_dirty = 0

    @torch.no_grad()
    def before_sample(self, ib) -> None:
        """After the forward, before logits: keep this step's suffix rows; run the head for finished prompts."""
        from ...core.decision_head import SuffixBatch

        for i, rid, s, cnt, pos0, done in self.step:
            m = self.meta.get(rid)
            if m is None:
                continue
            ss = m["suffix_start"]
            lo = max(ss - pos0, 0)
            if lo < cnt:
                self.rows.setdefault(rid, []).append(
                    (pos0 + lo, self.tap_buf[s + lo : s + cnt].clone())
                )
            if not done:
                continue
            row = int(ib.cu_num_logits_np[i])
            got = sorted(self.rows.pop(rid, []), key=lambda t: t[0])
            want = m["decide"] - ss + 1
            if not got or got[0][0] != ss or sum(t.shape[0] for _, t in got) < want:
                # part of the suffix was served from the prefix cache
                self.override[row] = (
                    m["k"],
                    MISSING,
                )
                continue
            feats = torch.cat([t for _, t in got])[:want]
            spans = [(a - ss, b - ss) for a, b in m["spans"]]
            batch = SuffixBatch.collate(
                [{"feats": feats, "opt_spans": spans, "decide": m["decide"] - ss}]
            )
            self.override[row] = (m["k"], self.head(batch)[0, : m["k"]].float())

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Option k -> token id k; MISSING -> token id K (one past the options), which the client retries."""
        for row, (k, z) in self.override.items():
            logits[row].fill_(float("-inf"))
            if isinstance(z, str):
                logits[row, k] = 0.0
            else:
                logits[row, :k] = z.to(logits.dtype)
        return logits


def fused_map(packed_modules_mapping: dict) -> dict[str, tuple[str, int]]:
    """HF projection name -> (vLLM fused module, slot), from a vLLM class's packed_modules_mapping."""
    out = {}
    for fused, parts in (packed_modules_mapping or {}).items():
        for k, p in enumerate(parts):
            out[p] = (fused, k)
    return out


def attach_suffix_lora(
    root: nn.Module, path: str, mask: torch.Tensor, prefix: str, fused: dict, k_eq_v: bool = False
) -> int:
    """Stack the HF-named suffix adapter onto vLLM's (possibly fused) linear layers; returns layers adapted.

    ``k_eq_v``: layers without a V projection use the K projection as V (Gemma 4 full-attention layers,
    ``attention_k_eq_v``). vLLM loads K's weights into both slots of the fused qkv, so K's adapter goes into both too:
    in the HF model the adapted K output is the V."""
    from safetensors.torch import load_file

    cfg = json.load(open(os.path.join(path, "suffix_adapter.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    groups: dict[tuple[int, str, str], dict[int, dict[str, torch.Tensor]]] = {}
    for key, t in load_file(os.path.join(path, "suffix_adapter.safetensors")).items():
        m = _KEY.match(key)
        if not m:
            raise ValueError(f"unexpected suffix-adapter key {key}")
        layer, sub, ab = int(m[1]), m[2], m[3]
        parent, _, proj = sub.rpartition(".")
        target, slot = fused.get(proj, (proj, 0))
        groups.setdefault((layer, parent, target), {}).setdefault(slot, {})[ab] = t
    if k_eq_v and "k_proj" in fused and "v_proj" in fused:
        (qkv, k_slot), (_, v_slot) = fused["k_proj"], fused["v_proj"]
        for (_, _, target), parts in groups.items():
            if target == qkv and k_slot in parts and v_slot not in parts:
                parts[v_slot] = parts[k_slot]
    modules = dict(root.named_modules())
    for (layer, parent, target), parts in groups.items():
        name = f"{prefix}layers.{layer}.{parent + '.' if parent else ''}{target}"
        mod = modules.get(name)
        if mod is None or not hasattr(mod, "weight"):
            raise ValueError(f"vLLM module {name} not found for the suffix adapter")
        slots = sorted(parts)
        A = torch.cat([parts[s]["A"] for s in slots])  # [k*r, in]
        outs = [parts[s]["B"].shape[0] for s in slots]
        if sum(outs) != mod.weight.shape[0]:
            raise ValueError(
                f"{name}: adapter outputs {outs} don't tile the fused output {mod.weight.shape[0]}"
            )
        B = torch.block_diag(*[parts[s]["B"] for s in slots]) * scaling  # [sum out, k*r]
        dev, dt = mod.weight.device, mod.weight.dtype
        mod.register_buffer("plumb_A", A.to(dev, dt), persistent=False)
        mod.register_buffer("plumb_B", B.to(dev, dt), persistent=False)
        mod.register_buffer("plumb_mask", mask, persistent=False)
        base_forward = mod.forward

        def forward(x, *args, _m=mod, _f=base_forward, **kw):
            out = _f(x, *args, **kw)
            delta = ((x @ _m.plumb_A.t()) @ _m.plumb_B.t()) * _m.plumb_mask[: x.shape[0]]
            if isinstance(out, tuple):
                return (out[0] + delta.to(out[0].dtype), *out[1:])
            return out + delta.to(out.dtype)

        mod.forward = forward
    return len(groups)


def find_text_model(model: nn.Module) -> tuple[nn.Module, str]:
    """The decoder stack that supports EAGLE-3 aux hidden states, and its module-path prefix."""
    for name, mod in model.named_modules():
        if hasattr(mod, "aux_hidden_state_layers") and isinstance(
            getattr(mod, "layers", None), nn.ModuleList
        ):
            return mod, (name + "." if name else "")
    raise NotImplementedError(
        f"{type(model).__name__}: vLLM exposes no intermediate hidden states (EAGLE-3 aux) for this architecture, "
        "which the decision head reads"
    )


class PlumbMixin:
    """Mixed into vLLM's class for the base architecture by ``make_plumb_class``."""

    def _plumb_init(self, vllm_config) -> None:
        from .hooks import install

        pc = vllm_config.parallel_config
        if pc.tensor_parallel_size != 1 or pc.pipeline_parallel_size != 1:
            raise NotImplementedError("plumbed models run with tensor/pipeline parallel size 1")
        path = find_plumb_dir(vllm_config)
        if not vllm_config.cache_config.enable_prefix_caching:
            logger.warning(
                "plumber: prefix caching is off, so every decision recomputes its whole context; "
                "leave it on (the vLLM default) for decisions that reuse the generator's KV cache"
            )
        text_model, prefix = find_text_model(self)
        cap = getattr(vllm_config.compilation_config, "max_cudagraph_capture_size", None) or 0
        max_tokens = max(vllm_config.scheduler_config.max_num_batched_tokens, cap) + 16
        hidden = vllm_config.model_config.hf_text_config.hidden_size
        dev = next(text_model.parameters()).device
        self.plumb = PlumbState(path, max_tokens, hidden, dev, vllm_config.model_config.dtype)
        # vLLM-internal: EAGLE-3 aux states (vllm/model_executor/models/interfaces.py EagleModelMixin): aux index
        # i + 1 is hidden + residual after decoder layer i, the residual stream plumber taps as layer i.
        aux = sorted({t + 1 for t in self.plumb.taps if t != -1})
        text_model.aux_hidden_state_layers = tuple(aux)
        self._plumb_aux_pos = {a - 1: k for k, a in enumerate(aux)}
        self._plumb_prefix = prefix
        self._plumb_k_eq_v = bool(
            getattr(vllm_config.model_config.hf_text_config, "attention_k_eq_v", False)
        )
        # vLLM-internal: compiled graphs are cached by vllm_config.compute_hash(), which covers additional_config;
        # a graph built for other taps or another adapter must not be reused (vllm/config/vllm.py).
        extra = (
            vllm_config.additional_config if isinstance(vllm_config.additional_config, dict) else {}
        )
        extra["plumb"] = {"path": os.path.abspath(path), "taps": self.plumb.taps}
        vllm_config.additional_config = extra
        install()

    def _plumb_after_load(self) -> None:
        if self.plumb.spec.has_adapter:
            n = attach_suffix_lora(
                self,
                self.plumb.path,
                self.plumb.mask,
                self._plumb_prefix,
                fused_map(getattr(type(self), "packed_modules_mapping", {})),
                self._plumb_k_eq_v,
            )
            logger.info(
                "plumber: %s with a suffix adapter on %d layers, taps %s, head %.1fM params",
                type(self).__name__,
                n,
                self.plumb.taps,
                sum(p.numel() for p in self.plumb.head.parameters()) / 1e6,
            )

    def _plumb_forward(self, out):
        hidden, aux = out if isinstance(out, tuple) else (out, [])
        if isinstance(hidden, torch.Tensor):
            parts = [hidden if t == -1 else aux[self._plumb_aux_pos[t]] for t in self.plumb.taps]
            self.plumb.tap_buf[: hidden.shape[0]].copy_(torch.stack(parts, 1))
        return hidden


_CLASSES: dict[str, type] = {}


def make_plumb_class(arch: str) -> type:
    """``Plumb<arch>``: vLLM's class for ``arch`` plus the plumb."""
    if arch in _CLASSES:
        return _CLASSES[arch]
    from vllm.model_executor.models.registry import ModelRegistry

    # vLLM-internal: vllm/model_executor/models/registry.py
    base = ModelRegistry._try_load_model_cls(arch)
    if base is None:
        raise ValueError(f"vLLM has no model class for {arch}")

    def __init__(self, *, vllm_config, prefix: str = ""):
        base.__init__(self, vllm_config=vllm_config, prefix=prefix)
        self._plumb_init(vllm_config)

    def forward(self, *args, **kwargs):
        return self._plumb_forward(base.forward(self, *args, **kwargs))

    def compute_logits(self, hidden_states, *args, **kwargs):
        logits = base.compute_logits(self, hidden_states, *args, **kwargs)
        return self.plumb.apply(logits) if (logits is not None and self.plumb.override) else logits

    def load_weights(self, weights):
        loaded = base.load_weights(self, weights)
        self._plumb_after_load()
        return loaded

    cls = type(
        PLUMB_PREFIX + arch,
        (PlumbMixin, base),
        {
            "__init__": __init__,
            "forward": forward,
            "compute_logits": compute_logits,
            "load_weights": load_weights,
            "__module__": __name__,
            "__qualname__": PLUMB_PREFIX + arch,
        },
    )
    _CLASSES[arch] = cls
    return cls


# "plumber.serving.vllm.model:Plumb<Arch>", resolved lazily by vLLM's registry
def __getattr__(
    name,
):
    if name.startswith(PLUMB_PREFIX) and len(name) > len(PLUMB_PREFIX):
        cls = make_plumb_class(name[len(PLUMB_PREFIX) :])
        globals()[name] = cls
        return cls
    raise AttributeError(name)
