"""Plumber: the inference engine for Plumb checkpoints. Loads tokenizer, trunk and head; answers typed questions in one forward pass.

    from plumber import Plumber
    engine = Plumber("totum-labs/plumb-nemotron-3.5-lightning-30b-a3b")
    out = engine.decide(state, {"department": {"type": "choice", "instructions": "...", "criteria": {"billing": "...", "sales": "..."}}})
    out["answers"]["department"]  ->  {"type": "choice", "choice": "billing", "probabilities": {...}, "confidence": 0.91}
"""
from __future__ import annotations
import json, os, time
from typing import Any, Dict, List, Optional
import torch
from .core.rendering import Row, Option, render
from .core.trunk import PlumbModel, collate

DEFAULT_BASE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
_DTYPES = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16, "float32": torch.float32}

def state_text(state) -> str:
    if isinstance(state, str): return state
    if isinstance(state, dict): return "\n".join(f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}" for k, v in state.items())
    return json.dumps(state, ensure_ascii=False)

def question_to_row(qid, q: dict, state) -> Row:
    """Jev-shaped question -> Row. choice: criteria {name: description}; noul: optional {"true","false"}; score: [level descriptions]."""
    t = q.get("type"); crit = q.get("criteria"); instr = q.get("instructions", "")
    if t == "choice":
        if isinstance(crit, dict): opts = [Option(str(k), str(v or "")) for k, v in crit.items()]
        elif isinstance(crit, list): opts = [Option(str(k), "") for k in crit]
        else: raise ValueError(f"{qid}: choice needs criteria")
    elif t == "noul":
        c = crit if isinstance(crit, dict) else {}
        opts = [Option("no", str(c.get("false") or "")), Option("yes", str(c.get("true") or ""))]
    elif t == "score":
        if not isinstance(crit, list) or not crit: raise ValueError(f"{qid}: score needs a list of level descriptions")
        opts = [Option(str(i), str(d)) for i, d in enumerate(crit)]
    else: raise ValueError(f"{qid}: unknown type {t!r}")
    return Row(id=str(qid), state=state_text(state), question=instr, qtype=t, options=opts)

class Plumber:
    """Inference engine. `model` is a merged release (HF repo id or local dir) or an adapter dir (stacked on `base`)."""

    def __init__(self, model: str, base: str = DEFAULT_BASE, dtype: str = "bfloat16", device_map: str = "auto", max_len: int = 32768, d_proj: int = 512):
        from transformers import AutoTokenizer
        t0 = time.time()
        self.model_id = model; self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model if _has_tokenizer(model) else base)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.model = PlumbModel.load(model, base=base, d_proj=d_proj, dtype=_DTYPES[dtype], device_map=device_map)
        self.device = next(self.model.trunk.parameters()).device
        self.load_seconds = time.time() - t0

    @classmethod
    def from_pretrained(cls, model: str, **kw) -> "Plumber": return cls(model, **kw)

    @torch.no_grad()
    def decide(self, state, questions: Dict[str, dict], max_len: Optional[int] = None) -> Dict[str, Any]:
        """Answer every question about one state. One batched forward; returns the Jev response shape."""
        t0 = time.perf_counter(); max_len = max_len or self.max_len
        rows = [question_to_row(qid, q, state) for qid, q in questions.items()]
        rend = [render(self.tok, r, rng=None, shuffle=False) for r in rows]
        too_long = [r.id for r, x in zip(rows, rend) if len(x["input_ids"]) > max_len]
        if too_long: raise ValueError(f"questions exceed max_len={max_len}: {too_long}")
        b = collate(self.tok, rend, self.pad, self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            logits = self.model(b["input_ids"], b["attention_mask"], b["opt_spans"], b["decide_pos"])
        P = torch.softmax(logits.float(), -1).cpu()
        answers = {}
        for r, p in zip(rows, P):
            K = len(r.options); pk = p[:K].tolist(); Kc = max(K, 2); conf = (max(pk) - 1 / Kc) / (1 - 1 / Kc)
            probs = {o.name: round(v, 6) for o, v in zip(r.options, pk)}
            if r.qtype == "noul": answers[r.id] = {"type": "noul", "noul": probs["yes"], "probabilities": probs, "confidence": round(conf, 6)}
            elif r.qtype == "choice": answers[r.id] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs, "confidence": round(conf, 6)}
            else: answers[r.id] = {"type": "score", "score": round(sum(i * v for i, v in enumerate(pk)), 4), "probabilities": probs, "confidence": round(conf, 6)}
        return {"answers": answers, "usage": {"input_tokens": sum(len(x["input_ids"]) for x in rend), "output_tokens": 0},
                "model": self.model_id, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def choice(self, state, instructions: str, criteria) -> dict:
        return self.decide(state, {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}})["answers"]["q"]
    def noul(self, state, instructions: str, criteria: Optional[dict] = None) -> dict:
        return self.decide(state, {"q": {"type": "noul", "instructions": instructions, "criteria": criteria}})["answers"]["q"]
    def score(self, state, instructions: str, levels: List[str]) -> dict:
        return self.decide(state, {"q": {"type": "score", "instructions": instructions, "criteria": list(levels)}})["answers"]["q"]
    def batch(self, requests: List[tuple]) -> List[dict]:
        return [self.decide(s, q) for s, q in requests]

def _has_tokenizer(model: str) -> bool:
    if os.path.isdir(model): return os.path.exists(os.path.join(model, "tokenizer.json")) or os.path.exists(os.path.join(model, "tokenizer_config.json"))
    try:
        from huggingface_hub import list_repo_files
        return "tokenizer.json" in set(list_repo_files(model))
    except Exception: return False
