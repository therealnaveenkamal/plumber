"""Fit one temperature on held-out predictions and store it in the checkpoint's head.pt.

  plumber calibrate --ckpt runs/x/final --preds runs/x/final/eval_dev/preds.jsonl [--apply runs/x/final/eval_test/preds.jsonl]

The temperature rescales the option logits before the softmax. It never changes which option wins; it moves the
probabilities toward (or away from) uniform so that stated confidence matches observed accuracy. Fitted by
minimising NLL on the held-out rows; ``--apply`` reports the effect on another prediction set without refitting.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

from ..metrics import summarize


def _rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rescale(rows: list[dict], temperature: float) -> list[dict]:
    """Recompute p_max / p_gold / probs of prediction rows under a temperature (probs were written at T = 1)."""
    out = []
    for r in rows:
        if r.get("gold") is None:
            continue
        z = torch.tensor([math.log(max(p, 1e-9)) for p in r["probs"]]) / temperature
        p = torch.softmax(z, -1).tolist()
        out.append(
            {
                **r,
                "probs": p,
                "p_max": max(p),
                "p_gold": p[r["gold"]],
                "correct": int(max(range(len(p)), key=p.__getitem__) == r["gold"]),
            }
        )
    return out


def fit_temperature(rows: list[dict], lo: float = 0.25, hi: float = 8.0, iters: int = 60) -> float:
    """Golden-section search of the NLL over log T (unimodal)."""
    rows = [r for r in rows if r.get("gold") is not None]

    def nll(log_t):
        t = math.exp(log_t)
        return sum(-math.log(max(r["p_gold"], 1e-12)) for r in rescale(rows, t)) / max(len(rows), 1)

    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(d)
    return math.exp((a + b) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir containing head.pt")
    ap.add_argument(
        "--preds", required=True, help="preds.jsonl from `plumber eval` on held-out rows"
    )
    ap.add_argument("--apply", default=None, help="another preds.jsonl to report before/after on")
    a = ap.parse_args()
    fit = _rows(a.preds)
    t = fit_temperature(fit)
    report = {
        "temperature": round(t, 4),
        "fit": {"before": summarize(rescale(fit, 1.0)), "after": summarize(rescale(fit, t))},
    }
    if a.apply:
        other = _rows(a.apply)
        report["apply"] = {
            "before": summarize(rescale(other, 1.0)),
            "after": summarize(rescale(other, t)),
        }
    path = os.path.join(a.ckpt, "head.pt")
    sd = torch.load(path, map_location="cpu")
    sd["temperature"] = torch.tensor(t)
    torch.save(sd, path)
    with open(os.path.join(a.ckpt, "calibration.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
