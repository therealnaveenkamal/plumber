"""Plumber: the inference engine for Plumb checkpoints. Loads tokenizer, trunk and head; answers typed questions in one forward pass.

from plumber import Plumber
engine = Plumber("totum-labs/plumb-nemotron-3.5-lightning-30b-a3b")
out = engine.decide(state, {"department": {"type": "choice", "instructions": "...", "criteria": {"billing": "...", "sales": "..."}}})
out["answers"]["department"]  ->  {"type": "choice", "choice": "billing", "probabilities": {...}, "confidence": 0.91}
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch

from .core.rendering import Option, Row, render
from .core.trunk import PlumbModel, collate

DEFAULT_BASE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
}


def state_text(state) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return "\n".join(
            f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
            for k, v in state.items()
        )
    return json.dumps(state, ensure_ascii=False)


def question_to_row(qid, q: dict, state) -> Row:
    """Jev-shaped question -> Row. choice: criteria {name: description}; noul: optional {"true","false"}; score: [level descriptions]."""
    t = q.get("type")
    crit = q.get("criteria")
    instr = q.get("instructions", "")
    if t == "choice":
        if isinstance(crit, dict):
            opts = [Option(str(k), str(v or "")) for k, v in crit.items()]
        elif isinstance(crit, list):
            opts = [Option(str(k), "") for k in crit]
        else:
            raise ValueError(f"{qid}: choice needs criteria")
    elif t == "noul":
        c = crit if isinstance(crit, dict) else {}
        opts = [Option("no", str(c.get("false") or "")), Option("yes", str(c.get("true") or ""))]
    elif t == "score":
        if not isinstance(crit, list) or not crit:
            raise ValueError(f"{qid}: score needs a list of level descriptions")
        opts = [Option(str(i), str(d)) for i, d in enumerate(crit)]
    else:
        raise ValueError(f"{qid}: unknown type {t!r}")
    return Row(id=str(qid), state=state_text(state), question=instr, qtype=t, options=opts)


class Plumber:
    """Inference engine. `model` is a merged release (HF repo id or local dir) or an adapter dir (stacked on `base`)."""

    def __init__(
        self,
        model: str,
        base: str = DEFAULT_BASE,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_len: int = 32768,
        d_proj: int = 512,
    ):
        from transformers import AutoTokenizer

        t0 = time.time()
        self.model_id = model
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model if _has_tokenizer(model) else base)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.model = PlumbModel.load(
            model, base=base, d_proj=d_proj, dtype=_DTYPES[dtype], device_map=device_map
        )
        self.device = next(self.model.trunk.parameters()).device
        self.load_seconds = time.time() - t0

    @classmethod
    def from_pretrained(cls, model: str, **kw) -> Plumber:
        return cls(model, **kw)

    @torch.no_grad()
    def decide(
        self,
        state,
        questions: dict[str, dict],
        max_len: int | None = None,
        share_prefix: bool | None = None,
    ) -> dict[str, Any]:
        """Answer every question about one state and return the Jev response shape.

        With ``share_prefix`` (default: on when there is more than one question) the state is encoded once, its
        cache — attention KV and Mamba recurrent states — is forked to every question, and the questions run as
        one batched continuation. Otherwise each question is one full sequence in a single batched forward.
        """
        t0 = time.perf_counter()
        max_len = max_len or self.max_len
        rows = [question_to_row(qid, q, state) for qid, q in questions.items()]
        rend = [render(self.tok, r, rng=None, shuffle=False) for r in rows]
        too_long = [r.id for r, x in zip(rows, rend, strict=False) if len(x["input_ids"]) > max_len]
        if too_long:
            raise ValueError(f"questions exceed max_len={max_len}: {too_long}")
        if share_prefix is None:
            share_prefix = len(rows) > 1
        logits = self._forward_shared_prefix(rend) if share_prefix else self._forward_full(rend)
        P = torch.softmax(logits.float(), -1).cpu()
        answers = {}
        for r, p in zip(rows, P, strict=False):
            K = len(r.options)
            pk = p[:K].tolist()
            Kc = max(K, 2)
            conf = (max(pk) - 1 / Kc) / (1 - 1 / Kc)
            probs = {o.name: round(v, 6) for o, v in zip(r.options, pk, strict=False)}
            if r.qtype == "noul":
                answers[r.id] = {
                    "type": "noul",
                    "noul": probs["yes"],
                    "probabilities": probs,
                    "confidence": round(conf, 6),
                }
            elif r.qtype == "choice":
                answers[r.id] = {
                    "type": "choice",
                    "choice": max(probs, key=probs.get),
                    "probabilities": probs,
                    "confidence": round(conf, 6),
                }
            else:
                answers[r.id] = {
                    "type": "score",
                    "score": round(sum(i * v for i, v in enumerate(pk)), 4),
                    "probabilities": probs,
                    "confidence": round(conf, 6),
                }
        return {
            "answers": answers,
            "usage": {"input_tokens": sum(len(x["input_ids"]) for x in rend), "output_tokens": 0},
            "model": self.model_id,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def _forward_full(self, rend: list[dict]) -> torch.Tensor:
        b = collate(self.tok, rend, self.pad, self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return self.model(b["input_ids"], b["attention_mask"], b["opt_spans"], b["decide_pos"])

    def _forward_shared_prefix(self, rend: list[dict]) -> torch.Tensor:
        """Prefill the common state prefix once, fork the cache N ways, run the N question continuations batched."""
        L_p = rend[0]["prefix_len"]
        prefix = rend[0]["input_ids"][:L_p]
        if any(x["prefix_len"] != L_p or x["input_ids"][:L_p] != prefix for x in rend):
            return self._forward_full(rend)  # different states (should not happen for one request)
        N = len(rend)
        dev = self.device
        conts = [x["input_ids"][L_p:] for x in rend]
        L_max = max(len(c) for c in conts)
        ids = torch.full((N, L_max), self.pad, dtype=torch.long, device=dev)
        am = torch.zeros((N, L_p + L_max), dtype=torch.long, device=dev)
        am[:, :L_p] = 1
        for i, c in enumerate(conts):
            ids[i, : len(c)] = torch.tensor(c, device=dev)
            am[i, L_p : L_p + len(c)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            cache = self.model.trunk(
                input_ids=torch.tensor([prefix], device=dev), use_cache=True
            ).past_key_values
            cache.reorder_cache(torch.zeros(N, dtype=torch.long, device=dev))
            out = self.model.trunk(
                input_ids=ids,
                attention_mask=am,
                past_key_values=cache,
                cache_position=torch.arange(L_p, L_p + L_max, device=dev),
                use_cache=True,
            )
            h = out.last_hidden_state
            spans = [[(s - L_p, e - L_p) for (s, e) in x["opt_spans"]] for x in rend]
            decide = [x["decide_pos"] - L_p for x in rend]
            return self.model.head(h, spans, decide)

    def choice(self, state, instructions: str, criteria) -> dict:
        return self.decide(
            state, {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}
        )["answers"]["q"]

    def noul(self, state, instructions: str, criteria: dict | None = None) -> dict:
        return self.decide(
            state, {"q": {"type": "noul", "instructions": instructions, "criteria": criteria}}
        )["answers"]["q"]

    def score(self, state, instructions: str, levels: list[str]) -> dict:
        return self.decide(
            state, {"q": {"type": "score", "instructions": instructions, "criteria": list(levels)}}
        )["answers"]["q"]

    def batch(self, requests: list[tuple]) -> list[dict]:
        return [self.decide(s, q) for s, q in requests]


def _has_tokenizer(model: str) -> bool:
    if os.path.isdir(model):
        return os.path.exists(os.path.join(model, "tokenizer.json")) or os.path.exists(
            os.path.join(model, "tokenizer_config.json")
        )
    try:
        from huggingface_hub import list_repo_files

        return "tokenizer.json" in set(list_repo_files(model))
    except Exception:
        return False
