"""Read a frozen trunk at several layers, at the decision suffix only.

A tap is the residual stream after decoder layer ``i`` (0-based); tap ``-1`` is the trunk's final, normed output. That
is the same quantity vLLM exposes as auxiliary hidden states (EAGLE3) plus the pooler input, so the head sees the same
features in training (transformers) and in serving (vLLM).

Hooks slice positions >= ``suffix_start`` inside the forward, so a 32k-token context never materialises 32k x taps
hidden states.
"""

from __future__ import annotations

import torch
from torch import nn


def decoder_layers(trunk: nn.Module) -> nn.ModuleList:
    for name in ("layers", "h", "blocks"):
        layers = getattr(trunk, name, None)
        if isinstance(layers, nn.ModuleList):
            return layers
    raise ValueError(f"cannot find the decoder layer list on {type(trunk).__name__}")


def default_taps(n_layers: int, fractions=(0.5, 0.625, 0.75, 0.875)) -> list[int]:
    """Four mid-to-upper layers plus the final output: decision-relevant features peak mid-depth in frozen LMs,
    and the final layer is specialised for next-token prediction."""
    return [*sorted({min(n_layers - 2, round(f * n_layers) - 1) for f in fractions}), -1]


class TapReader:
    """``reader.run(input_ids, mask, suffix_starts, lengths)`` -> per-row suffix features ``[S_b, T, d]``.

    ``suffix_starts[b]`` is the first suffix position of row b *in this forward's input* (0 when the input is the
    suffix itself, e.g. a continuation over a cached prefix); ``lengths[b]`` excludes right padding.
    """

    def __init__(self, trunk: nn.Module, taps: list[int]):
        self.trunk, self.taps = trunk, list(taps)
        self.layers = decoder_layers(trunk)
        n = len(self.layers)
        bad = [t for t in self.taps if t != -1 and not 0 <= t < n]
        if bad:
            raise ValueError(f"taps {bad} out of range for {n} layers")

    def run(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        suffix_starts: list[int],
        lengths: list[int],
        detach: bool = True,
        **trunk_kw,
    ) -> list[torch.Tensor]:
        lo = min(suffix_starts)  # keep only positions any row needs, inside the hook
        grabbed: dict[int, torch.Tensor] = {}

        def hook(t):
            def f(_m, _args, out):
                h = (out[0] if isinstance(out, tuple) else out)[:, lo:]
                grabbed[t] = h.detach() if detach else h

            return f

        handles = [self.layers[t].register_forward_hook(hook(t)) for t in self.taps if t != -1]
        try:
            final = self.trunk(
                input_ids=input_ids, attention_mask=attention_mask, **trunk_kw
            ).last_hidden_state
        finally:
            for h in handles:
                h.remove()
        missing = [t for t in self.taps if t != -1 and t not in grabbed]
        if missing:
            raise RuntimeError(f"taps {missing} were not reached in the forward")
        dev = final.device
        stack = [
            grabbed[t].to(dev) if t != -1 else final[:, lo:] for t in self.taps
        ]  # T x [B, L - lo, d]
        return [
            torch.stack([x[b, suffix_starts[b] - lo : lengths[b] - lo] for x in stack], 1)
            for b in range(input_ids.shape[0])
        ]
