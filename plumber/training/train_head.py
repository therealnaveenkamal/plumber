"""Train a plumb: a decision head that reads the frozen base model at several layers, and (with ``--suffix_lora``)
a LoRA that acts on decision tokens only. The base weights get no gradient. ``plumber plumbify`` runs this step and
then packages the result.

  plumber train-head --base Qwen/Qwen3.5-9B --rows train.jsonl --dev dev.jsonl --out runs/qwen3.5-9b --suffix_lora

At the end it evaluates on --dev, fits temperature and a split-conformal threshold there (alpha from --alpha), and
writes ``<out>/final`` (plumb.json + head.safetensors), ``dev_preds.jsonl`` and ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from ..calibration import coverage, escalation_curve, fit_conformal, fit_temperature, rescale
from ..core.decision_head import decision_loss
from ..metrics import summarize
from ..system1 import System1, load_lm
from .data import bucketed_batches, load_rows


def lengths_of(s1: System1, rows) -> list[int]:
    return [len(s1.render(r).input_ids) for r in rows]


@torch.no_grad()
def predict(
    s1: System1, rows, max_rows: int, max_len: int, tokens_per_batch: int = 65536
) -> list[dict]:
    """Predictions in input order; batches are bounded by padded tokens as well as rows, so long states don't
    blow up memory (a row-count cap alone let 32 long rows land in one forward)."""
    s1.head.eval()
    rend = [s1.render(r) for r in rows]
    lens = [len(x.input_ids) for x in rend]
    order = sorted(range(len(rows)), key=lens.__getitem__)
    by_id: dict[int, dict] = {}
    batch: list[int] = []

    def flush():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = s1.logits([rend[i] for i in batch])
        p = torch.softmax(logits.float(), -1).cpu()
        for i, pj in zip(batch, p, strict=True):
            r = rows[i]
            probs = pj[: len(r.options)].tolist()
            top = max(range(len(probs)), key=probs.__getitem__)
            by_id[i] = {
                "id": r.id,
                "family": r.meta.get("family"),
                "qtype": r.qtype,
                "gold": r.gold,
                "probs": [round(v, 6) for v in probs],
                "p_max": max(probs),
                "p_gold": None if r.gold is None else probs[r.gold],
                "correct": None if r.gold is None else int(top == r.gold),
            }

    for i in order:  # ascending length: the padded size of a batch is its last row's length
        if lens[i] > max_len:
            continue
        if batch and (len(batch) >= max_rows or lens[i] * (len(batch) + 1) > tokens_per_batch):
            flush()
            batch = []
        batch.append(i)
    if batch:
        flush()
    s1.head.train()
    return [by_id[i] for i in range(len(rows)) if i in by_id]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="Hugging Face causal LM to read (stays frozen)")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--taps",
        default="auto",
        help="comma-separated layer indices, -1 = final output; auto = 4 + final",
    )
    ap.add_argument("--d_h", type=int, default=1024)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--token_blocks", type=int, default=1)
    ap.add_argument("--set_blocks", type=int, default=2)
    ap.add_argument("--d_proj", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument(
        "--tokens_per_batch", type=int, default=65536, help="padded tokens per trunk forward"
    )
    ap.add_argument("--max_rows", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument(
        "--alpha", type=float, default=0.1, help="conformal error rate stored in the plumb"
    )
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--suffix_lora",
        action="store_true",
        help="also train a LoRA that acts on the decision suffix only (context stays on base weights)",
    )
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_lr", type=float, default=1e-4)
    ap.add_argument(
        "--init_from",
        default=None,
        help="warm start from another plumb on the same base (its taps and head shape are reused)",
    )
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--log_every", type=int, default=20)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    dtype = getattr(torch, a.dtype)
    lm = load_lm(
        a.base, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
    ).eval()
    tok = AutoTokenizer.from_pretrained(a.base)
    if a.init_from:
        from safetensors.torch import load_file

        from ..artifact import fetch, read_spec

        init = read_spec(a.init_from)
        hc = {k: v for k, v in init.head_config.items() if k not in ("d", "n_taps")}
        s1 = System1.new(lm, tok, init.taps, **hc)
        dev = next(s1.head.parameters()).device
        s1.head.load_state_dict(load_file(fetch(a.init_from, "head.safetensors"), device=str(dev)))
        s1.head.temperature.fill_(1.0)  # train at T = 1; recalibrated on --dev at the end
        print(f"[s1] warm start from {a.init_from}", flush=True)
    else:
        taps = None if a.taps == "auto" else [int(t) for t in a.taps.split(",")]
        s1 = System1.new(
            lm,
            tok,
            taps,
            d_h=a.d_h,
            n_heads=a.n_heads,
            token_blocks=a.token_blocks,
            set_blocks=a.set_blocks,
            d_proj=a.d_proj,
            dropout=a.dropout,
        )
    lora_params = []
    if a.suffix_lora:
        lora_params = s1.attach_suffix_lora(r=a.lora_r, alpha=a.lora_alpha).parameters()
        s1.trunk.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        s1.trunk.train()  # transformers only checkpoints in train mode; the trunk has no dropout to speak of
        print(
            f"[s1] suffix LoRA r={a.lora_r} on {s1.suffix_lora.targets}: "
            f"{sum(p.numel() for p in lora_params) / 1e6:.1f}M params",
            flush=True,
        )
    n_head = sum(p.numel() for p in s1.head.parameters())
    print(
        f"[s1] base={a.base} taps={s1.reader.taps} head={n_head / 1e6:.1f}M params "
        f"({'suffix LoRA' if lora_params else 'trunk frozen'})",
        flush=True,
    )

    rows = load_rows(a.rows, a.limit)
    dev = load_rows(a.dev) if a.dev else []
    lens = lengths_of(s1, rows)
    groups = [{"params": list(s1.head.parameters()), "lr": a.lr}]
    if lora_params:
        groups.append({"params": lora_params, "lr": a.lora_lr})
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    order = list(range(len(rows)))
    per_epoch = len(
        bucketed_batches(order, lens, a.tokens_per_batch, a.max_rows, a.max_len, random.Random(0))
    )
    total = max(1, int(per_epoch * a.epochs))
    warm = max(1, int(total * a.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))),
    )
    print(f"[s1] rows={len(rows)} dev={len(dev)} steps={total} ({per_epoch}/epoch)", flush=True)

    step, t0, ema = 0, time.time(), None
    s1.head.train()
    while step < total:
        rng.shuffle(order)
        for batch in bucketed_batches(order, lens, a.tokens_per_batch, a.max_rows, a.max_len, rng):
            if step >= total:
                break
            rend = [s1.render(rows[i], rng=rng) for i in batch]  # fresh option order every time
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                # frozen trunk: no graph; suffix LoRA: graph through the trunk (checkpointed)
                feats = s1.features(rend, grad=bool(lora_params))
            logits = s1.logits(rend, feats)
            loss = decision_loss(logits, [r.gold for r in rend], [r.teacher for r in rend])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(s1.head.parameters()) + lora_params, 1.0)
            opt.step()
            sched.step()
            step += 1
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % a.log_every == 0 or step == total:
                print(
                    f"[s1] step {step}/{total} loss {ema:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{(time.time() - t0) / step:.2f}s/step",
                    flush=True,
                )

    s1.trunk.eval()
    from ..artifact import Calibration, Conformal

    cal, metrics = Calibration(), {}
    if dev:
        preds = predict(s1, dev, a.max_rows, a.max_len, a.tokens_per_batch)
        with open(os.path.join(a.out, "dev_preds.jsonl"), "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in preds)
        t = fit_temperature(preds)
        conf = fit_conformal(preds, a.alpha, t)
        cal = Calibration(t, Conformal(**conf))
        scaled = [{**p, "probs": rescale(p["probs"], t)} for p in preds if p["gold"] is not None]
        for p in scaled:
            p["p_max"], p["p_gold"] = max(p["probs"]), p["probs"][p["gold"]]
        metrics = {
            "dev_raw": summarize(preds),
            "dev_calibrated": summarize(scaled),
            "temperature": t,
            "conformal": conf,
            "conformal_dev": coverage(preds, conf["qhat"], t),
            "escalation_curve": escalation_curve(preds, t),
        }
        print(
            f"[s1] dev acc {metrics['dev_raw']['accuracy']:.4f} ece {metrics['dev_raw']['ece']:.4f} -> "
            f"{metrics['dev_calibrated']['ece']:.4f} (T={t:.3f}); conformal a={a.alpha} qhat={conf['qhat']:.3f} "
            f"singletons {metrics['conformal_dev']['singleton_rate']:.2f}",
            flush=True,
        )
    s1.head.temperature.fill_(cal.temperature)
    s1.save(os.path.join(a.out, "final"), a.base, cal)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(
            {**metrics, "steps": total, "head_params": n_head, "taps": s1.reader.taps}, f, indent=1
        )
    print(f"[s1] saved {a.out}/final", flush=True)


if __name__ == "__main__":
    main()
