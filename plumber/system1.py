"""System 1 on transformers: a frozen base LM, read at several layers, plus a DecisionHead. The base keeps its LM head,
so the same loaded model also generates (System 2) with the original, untouched weights.

    s1 = System1.load("runs/qwen4b-head")                    # base model id comes from plumb.json
    s1.decide([row])                                          # -> [[p_option_0, p_option_1, ...]]
    s1.decide([row], context=[{"role": "user", "content": "..."}, ...])   # decide over a conversation

Rows that share their context (several questions about one state, or decisions over one conversation) prefill that
context once and fork its cache (attention KV and recurrent state), so each extra question costs only its suffix.
"""

from __future__ import annotations

import collections
import logging
import random
from collections.abc import Sequence

import torch
from torch import nn

from .core.decision_head import DecisionHead, HeadConfig, SuffixBatch
from .core.render import Rendered, render_decision
from .core.row import Row
from .core.taps import TapReader, decoder_layers, default_taps

logger = logging.getLogger(__name__)


def text_trunk(lm: nn.Module) -> nn.Module:
    inner = getattr(lm, "model", None) or getattr(lm, "transformer", None)
    if inner is None:
        raise ValueError(f"cannot find the trunk on {type(lm).__name__}")
    return getattr(inner, "language_model", inner)


def load_lm(name_or_path: str, **hf_kwargs):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(name_or_path)
    multimodal = any(hasattr(cfg, k) for k in ("vision_config", "text_config"))
    return (AutoModelForImageTextToText if multimodal else AutoModelForCausalLM).from_pretrained(
        name_or_path, **hf_kwargs
    )


class System1(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        tok,
        head: DecisionHead,
        taps: Sequence[int],
        template_kwargs=None,
        min_shared_prefix: int = 64,
    ):
        super().__init__()
        self.lm, self.tok, self.head = lm, tok, head
        self.trunk = text_trunk(lm)
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.reader = TapReader(self.trunk, list(taps))
        self.template_kwargs = template_kwargs or {}
        self.min_shared_prefix = min_shared_prefix
        # None = not yet verified. Some transformers hybrids (Qwen3.5's gated DeltaNet) drop the cached recurrent
        # state on multi-token continuations, so a fork is only trusted after matching a full forward once.
        self._fork_ok: bool | None = None
        self.pad_multiple = 64
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.suffix_lora = (
            None  # optional: adapter on the decision suffix only (core/suffix_lora.py)
        )

    def attach_suffix_lora(self, **kw):
        from .core.suffix_lora import SuffixLoRA

        self.suffix_lora = SuffixLoRA(self.trunk, **kw).trainable(True)
        return self.suffix_lora

    # ------------------------------------------------------------------ construction

    @classmethod
    def new(cls, lm, tok, taps: Sequence[int] | None = None, **head_kw) -> System1:
        trunk = text_trunk(lm)
        taps = list(taps) if taps else default_taps(len(decoder_layers(trunk)))
        cfg = HeadConfig(d=trunk.config.hidden_size, n_taps=len(taps), **head_kw)
        head = DecisionHead(cfg).to(next(trunk.parameters()).device)
        return cls(lm, tok, head, taps)

    @classmethod
    def load(cls, plumb: str, base: str | None = None, lm=None, tok=None, **hf_kwargs) -> System1:
        from .artifact import fetch, read_spec

        spec = read_spec(plumb)
        if lm is None:
            lm = load_lm(base or spec.base_model, **hf_kwargs)
        if tok is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(base or spec.base_model)
        head = DecisionHead(HeadConfig(**spec.head_config)).to(
            next(text_trunk(lm).parameters()).device
        )
        from safetensors.torch import load_file

        head.load_state_dict(
            load_file(fetch(plumb, "head.safetensors"), device=str(next(head.parameters()).device))
        )
        head.temperature.fill_(spec.calibration.temperature)
        s1 = cls(lm, tok, head.eval(), spec.taps, spec.extra.get("template_kwargs"))
        if spec.has_adapter:
            if spec.extra.get("adapter_scope") != "suffix":
                raise ValueError(f"{plumb}: only suffix-scoped adapters load on System1")
            from .core.suffix_lora import SuffixLoRA

            s1.suffix_lora = SuffixLoRA.load(s1.trunk, plumb)
        return s1

    # ------------------------------------------------------------------ decision path

    def render(self, row: Row, context=None, rng: random.Random | None = None) -> Rendered:
        perm = None
        if rng is not None and row.qtype != "score":
            perm = list(range(len(row.options)))
            rng.shuffle(perm)
        return render_decision(self.tok, row, context, perm, self.template_kwargs)

    def features(self, rend: list[Rendered], grad: bool = False) -> list[torch.Tensor]:
        """Suffix features [S, T, d] per rendered decision, sharing prefill across rows with the same context.
        ``grad`` keeps the graph through the trunk (training a suffix adapter); it always uses full forwards."""
        if grad:  # training: the mask must survive until backward recomputes checkpointed layers
            return self._full(rend, grad=True)
        try:
            with torch.no_grad():
                return self._features(rend)
        finally:
            self._set_mask(None)  # back to the base model for anything that runs next (generation)

    def _features(self, rend: list[Rendered]) -> list[torch.Tensor]:
        groups = collections.defaultdict(list)
        for i, r in enumerate(rend):
            groups[tuple(r.input_ids[: r.suffix_start])].append(i)
        out: list[torch.Tensor | None] = [None] * len(rend)
        solo = []
        for prefix, idx in groups.items():
            if (
                self._fork_ok is not False
                and len(idx) >= 2
                and len(prefix) >= self.min_shared_prefix
            ):
                sub = [rend[i] for i in idx]
                try:
                    feats = self._shared(list(prefix), sub)
                except (AttributeError, NotImplementedError, RuntimeError) as e:
                    self._fork_ok = False  # a cache type that can't be forked
                    logger.info("prefix fork unavailable (%s: %s)", type(e).__name__, e)
                    solo += idx
                    continue
                if self._fork_ok is None and not self._verify_fork(feats, self._full(sub)):
                    solo += idx
                    continue
                for i, f in zip(idx, feats, strict=True):
                    out[i] = f
            else:
                solo += idx
        if solo:
            for i, f in zip(solo, self._full([rend[i] for i in solo]), strict=True):
                out[i] = f
        return out  # type: ignore[return-value]

    def _verify_fork(
        self, fork: list[torch.Tensor], full: list[torch.Tensor], tol: float = 0.02
    ) -> bool:
        num = sum(
            (a.float() - b.float()).abs().sum().item() for a, b in zip(fork, full, strict=True)
        )
        den = sum(b.float().abs().sum().item() for b in full)
        rel = num / max(den, 1e-12)
        self._fork_ok = rel < tol
        verdict = (
            "enabled" if self._fork_ok else "DISABLED (continuation over a cached prefix is wrong)"
        )
        logger.info("prefix-fork check: mean relative difference %.4f, fork %s", rel, verdict)
        return self._fork_ok

    def _device(self):
        return next(self.trunk.parameters()).device

    def _set_mask(self, mask: torch.Tensor | None) -> None:
        if self.suffix_lora is not None:
            self.suffix_lora.mask = mask

    def _full(self, rend: list[Rendered], grad: bool = False) -> list[torch.Tensor]:
        dev, L = self._device(), max(len(r.input_ids) for r in rend)
        # few distinct shapes -> few kernel compiles
        L = -(-L // self.pad_multiple) * self.pad_multiple
        ids = torch.full((len(rend), L), self.pad_id, dtype=torch.long, device=dev)
        am = torch.zeros_like(ids)
        for b, r in enumerate(rend):
            ids[b, : len(r.input_ids)] = torch.tensor(r.input_ids, device=dev)
            am[b, : len(r.input_ids)] = 1
        starts = [r.suffix_start for r in rend]
        if self.suffix_lora is not None:
            self._set_mask(self.suffix_lora.suffix_mask(starts, L, dev))
        return self.reader.run(
            ids, am, starts, [len(r.input_ids) for r in rend], detach=not grad, use_cache=False
        )

    def _shared(self, prefix: list[int], rend: list[Rendered]) -> list[torch.Tensor]:
        """Prefill the context once, fork the cache N ways, run the N suffixes as one batched continuation."""
        dev, N, P = self._device(), len(rend), len(prefix)
        self._set_mask(torch.zeros(1, P, device=dev))  # the context runs on base weights
        cache = self.trunk(
            input_ids=torch.tensor([prefix], device=dev), use_cache=True
        ).past_key_values
        cache.reorder_cache(torch.zeros(N, dtype=torch.long, device=dev))
        sufs = [r.input_ids[P:] for r in rend]
        S = max(len(s) for s in sufs)
        ids = torch.full((N, S), self.pad_id, dtype=torch.long, device=dev)
        am = torch.zeros((N, P + S), dtype=torch.long, device=dev)
        am[:, :P] = 1
        for b, s in enumerate(sufs):
            ids[b, : len(s)] = torch.tensor(s, device=dev)
            am[b, P : P + len(s)] = 1
        self._set_mask(torch.ones(N, S, device=dev))  # every continuation token is suffix
        return self.reader.run(
            ids,
            am,
            [0] * N,
            [len(s) for s in sufs],
            past_key_values=cache,
            cache_position=torch.arange(P, P + S, device=dev),
            use_cache=True,
        )

    def logits(self, rend: list[Rendered], feats: list[torch.Tensor] | None = None) -> torch.Tensor:
        feats = self.features(rend) if feats is None else feats
        items = [{"feats": f, **r.suffix()} for f, r in zip(feats, rend, strict=True)]
        return self.head(SuffixBatch.collate(items, device=next(self.head.parameters()).device))

    @torch.no_grad()
    def decide(self, rows: list[Row], context=None) -> list[list[float]]:
        """Probabilities over each row's options, in the row's own option order."""
        rend = [self.render(r, context) for r in rows]
        p = torch.softmax(self.logits(rend), -1).cpu()
        return [p[b, : len(r.options)].tolist() for b, r in enumerate(rows)]

    # ------------------------------------------------------------------ generation path (the untouched base model)

    @torch.no_grad()
    def generate(self, messages: list[dict], max_new_tokens: int = 512, **gen_kw) -> str:
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", **self.template_kwargs
        ).to(self._device())
        out = self.lm.generate(ids, max_new_tokens=max_new_tokens, **gen_kw)
        return self.tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)

    def save(self, out: str, base_model: str, calibration=None) -> None:
        from .artifact import Calibration, PlumbSpec, save

        spec = PlumbSpec(
            base_model=base_model,
            d=self.head.cfg.d,
            has_adapter=self.suffix_lora is not None,
            render_version=2,
            head_type="decision",
            head_config=self.head.cfg.to_dict(),
            taps=self.reader.taps,
            calibration=calibration or Calibration(float(self.head.temperature)),
            extra={
                **({"template_kwargs": self.template_kwargs} if self.template_kwargs else {}),
                **({"adapter_scope": "suffix"} if self.suffix_lora is not None else {}),
            },
        )
        save(out, spec, self.head)
        if self.suffix_lora is not None:
            self.suffix_lora.save(out)
