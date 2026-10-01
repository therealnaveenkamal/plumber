"""Evaluate a plumb, or the base model zero-shot, on labelled rows.

  plumbify eval-head --plumb runs/qwen4b-s1/final --rows data/decisionbench/test_ood.jsonl --out runs/qwen4b-s1/eval_ood
  plumbify eval-head --zeroshot --base Qwen/Qwen3.5-4B --rows ... --out runs/qwen4b-zeroshot/eval_ood

Zero-shot renders the same decision turn and scores each option name as the
model's answer: the mean log-probability of the name's tokens after the assistant-open. No training, no head.

Writes preds.jsonl and metrics.json: overall and per-family accuracy / NLL / Brier / ECE, conformal coverage and set
sizes at the plumb's alpha, and the escalation curve.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import torch

from ..calibration import coverage, escalation_curve
from ..core.render import render_decision
from ..metrics import summarize
from .data import load_rows


@torch.no_grad()
def zeroshot_probs(lm, tok, row, template_kwargs=None, chunk: int = 4) -> list[float]:
    """Mean log-probability of each option name after the assistant-open, options scored ``chunk`` at a time (a long
    state times many options overflows a single batch)."""
    rd = render_decision(tok, row, template_kwargs=template_kwargs)
    dev = next(lm.parameters()).device
    prefix = rd.input_ids
    conts = [tok(o.name, add_special_tokens=False).input_ids for o in row.options]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    scores = []
    for c0 in range(0, len(conts), chunk):
        part = conts[c0 : c0 + chunk]
        L = max(len(c) for c in part)
        ids = torch.full((len(part), len(prefix) + L), pad, dtype=torch.long, device=dev)
        for k, c in enumerate(part):
            ids[k, : len(prefix) + len(c)] = torch.tensor(prefix + c, device=dev)
        # only the positions that predict option tokens: full-vocab logits over a long state don't fit
        logp = torch.log_softmax(lm(input_ids=ids, logits_to_keep=L + 1).logits[:, :-1].float(), -1)
        for k, c in enumerate(part):
            tgt = torch.tensor(c, device=dev)
            scores.append(logp[k, torch.arange(len(c), device=dev), tgt].mean().item())
    return torch.softmax(torch.tensor(scores), -1).tolist()


def pred_row(r, probs):
    top = max(range(len(probs)), key=probs.__getitem__)
    return {
        "id": r.id,
        "family": r.meta.get("family"),
        "qtype": r.qtype,
        "gold": r.gold,
        "probs": [round(p, 6) for p in probs],
        "p_max": max(probs),
        "p_gold": None if r.gold is None else probs[r.gold],
        "correct": None if r.gold is None else int(top == r.gold),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plumb", default=None, help="plumb or plumbed model directory")
    ap.add_argument(
        "--zeroshot", action="store_true", help="no head: score option names with the LM"
    )
    ap.add_argument("--base", default=None, help="base model (default: the plumb's)")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bsz", type=int, default=32, help="max rows per forward")
    ap.add_argument(
        "--tokens_per_batch", type=int, default=65536, help="max padded tokens per forward"
    )
    ap.add_argument("--max_len", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=1, help="evaluate every k-th row (an even sample)")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args(argv)
    if not (a.plumb or (a.zeroshot and a.base)):
        ap.error("give --plumb, or --zeroshot with --base")
    os.makedirs(a.out, exist_ok=True)
    rows = load_rows(a.rows, a.limit)[:: a.every]
    kw = {
        "dtype": getattr(torch, a.dtype),
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    qhat = None
    if a.zeroshot:
        from transformers import AutoTokenizer

        from ..system1 import load_lm

        lm, tok = load_lm(a.base, **kw).eval(), AutoTokenizer.from_pretrained(a.base)
        preds = [pred_row(r, zeroshot_probs(lm, tok, r)) for r in rows]
    else:
        from ..artifact import read_spec
        from ..system1 import System1
        from .train_head import predict

        spec = read_spec(a.plumb)
        s1 = System1.load(a.plumb, base=a.base, **kw).eval()
        # the loaded head already applies the calibrated temperature
        preds = predict(s1, rows, a.bsz, a.max_len, a.tokens_per_batch)
        if spec.calibration.conformal:
            qhat = spec.calibration.conformal.qhat
    with open(os.path.join(a.out, "preds.jsonl"), "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in preds)
    fam = collections.defaultdict(list)
    for p in preds:
        fam[p["family"]].append(p)
    per_family = {k: summarize(v) for k, v in sorted(fam.items(), key=lambda kv: str(kv[0]))}
    metrics = {
        "overall": summarize(preds),
        "macro_family_accuracy": sum(v["accuracy"] for v in per_family.values())
        / max(len(per_family), 1),
        "escalation_curve": escalation_curve(preds),
        "per_family": per_family,
    }
    if qhat is not None:
        metrics["conformal"] = coverage(preds, qhat)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    o = metrics["overall"]
    print(
        f"n={o['n']} acc={o['accuracy']:.4f} macro={metrics['macro_family_accuracy']:.4f} "
        f"ece={o['ece']:.4f}"
        + (
            f" conformal cov={metrics['conformal']['coverage']:.3f} singletons="
            f"{metrics['conformal']['singleton_rate']:.2f}"
            if qhat is not None
            else ""
        )
    )


if __name__ == "__main__":
    main()
