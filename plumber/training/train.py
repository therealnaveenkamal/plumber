"""LoRA + pointer-head training for PlumbModel on Nemotron 3.5 Lightning.

  python -m plumber.training.train --rows data/x/rows.jsonl --out runs/smoke --limit 256 --epochs 1      # smoke
  python -m plumber.training.train --rows data/mix/train.jsonl --dev data/mix/dev.jsonl --out runs/v0     # real

Trunk: bf16, device_map="auto" (sharded), gradient checkpointing, LoRA on LORA_TARGETS (in_proj/out_proj on the
23 Mamba layers, q/k/v/o on the 6 attention layers, shared-expert up/down on the 23 MoE layers). Routed experts
(93% of params) and the embeddings stay frozen. LM head is deleted at load. Head trained fully at its own LR.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch

from ..core.heads import decision_loss
from ..core.rendering import Row, render
from ..core.trunk import LORA_TARGETS, PlumbModel, collate, pin_mamba_devices


def load_rows(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(Row.from_json(line))
            if limit and len(rows) >= limit:
                break
    return rows


@torch.no_grad()
def evaluate(model, tok, rows, pad_id, bsz, max_len):
    model.eval()
    n = correct = 0
    nll = 0.0
    skipped = 0
    for i in range(0, len(rows), bsz):
        chunk = rows[i : i + bsz]
        rend = [render(tok, r, rng=None, shuffle=False) for r in chunk]
        keep = [r for r in rend if len(r["input_ids"]) <= max_len]
        skipped += len(rend) - len(keep)
        rend = keep
        if not rend:
            continue
        b = collate(tok, rend, pad_id, next(model.trunk.parameters()).device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(b["input_ids"], b["attention_mask"], b["opt_spans"], b["decide_pos"])
        lp = torch.log_softmax(logits.float(), -1)
        for j, r in enumerate(rend):
            if r["gold"] is None:
                continue
            n += 1
            correct += int(lp[j].argmax().item() == r["gold"])
            nll -= lp[j, r["gold"]].item()
    model.train()
    return {
        "n": n,
        "acc": round(correct / max(n, 1), 4),
        "nll": round(nll / max(n, 1), 4),
        "skipped_too_long": skipped,
    }


def rendered_lengths(tok, rows, cache_path):
    """Token length of every row rendered once (unshuffled); cached to disk keyed by row id."""
    if os.path.exists(cache_path):
        c = json.load(open(cache_path))
        if all(r.id in c for r in rows):
            return [c[r.id] for r in rows]
    c = {}
    for r in rows:
        c[r.id] = len(render(tok, r, rng=None, shuffle=False)["input_ids"])
    json.dump(c, open(cache_path, "w"))
    return [c[r.id] for r in rows]


def bucketed_batches(order, lengths, tokens_per_batch, max_rows, max_len, rng, chunk=1024):
    """Shuffle -> local sort by length in chunks -> greedy fill up to tokens_per_batch (padded) or max_rows -> shuffle batches."""
    batches = []
    for c0 in range(0, len(order), chunk):
        part = sorted(order[c0 : c0 + chunk], key=lambda i: lengths[i])
        cur = []
        for i in part:
            if lengths[i] > max_len:
                continue
            L = max([lengths[j] for j in cur] + [lengths[i]])
            if cur and (L * (len(cur) + 1) > tokens_per_batch or len(cur) >= max_rows):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dev_limit", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument(
        "--tokens_per_batch",
        type=int,
        default=0,
        help=">0: length-bucketed micro-batches capped at this many tokens (bsz becomes the max rows per micro-batch)",
    )
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--d_proj", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--no_ckpt", action="store_true", help="disable gradient checkpointing")
    ap.add_argument(
        "--init_from",
        default=None,
        help="warm start: a previous run's step dir (LoRA adapter + head.pt)",
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    t0 = time.time()
    model = PlumbModel.from_pretrained(
        a.model, d_proj=a.d_proj, dtype=torch.bfloat16, device_map="auto"
    )
    n_pin = pin_mamba_devices(model.trunk)
    if not a.no_ckpt:
        model.trunk.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.trunk.enable_input_require_grads()
    if a.init_from:
        from peft import PeftModel

        model.trunk = PeftModel.from_pretrained(model.trunk, a.init_from, is_trainable=True)
        model.head.load_state_dict(
            torch.load(
                os.path.join(a.init_from, "head.pt"),
                map_location=next(model.head.parameters()).device,
            )
        )
        print(f"[init] warm-started LoRA + head from {a.init_from}", flush=True)
    else:
        lcfg = LoraConfig(
            r=a.lora_r,
            lora_alpha=a.lora_alpha,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            bias="none",
        )
        model.trunk = get_peft_model(model.trunk, lcfg)
    # coverage report: the run does not start if LoRA missed the Mamba or MoE layers
    hit = {}
    for n_, m_ in model.trunk.named_modules():
        if hasattr(m_, "lora_A"):
            hit[n_.split(".")[-1]] = hit.get(n_.split(".")[-1], 0) + 1
    print(f"[lora] adapted modules: {hit}", flush=True)
    assert (
        hit.get("in_proj", 0) == 23 and hit.get("q_proj", 0) == 6 and hit.get("up_proj", 0) == 23
    ), f"LoRA coverage short - check LORA_TARGETS: {hit}"
    lora_params = [p for n_, p in model.trunk.named_parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    n_tr = sum(p.numel() for p in lora_params) + sum(p.numel() for p in head_params)
    print(
        f"[model] loaded {time.time() - t0:.0f}s  lm_head deleted={model.deleted_lm_head_params / 1e6:.0f}M  trainable={n_tr / 1e6:.1f}M "
        f"(lora {sum(p.numel() for p in lora_params) / 1e6:.1f}M + head {sum(p.numel() for p in head_params) / 1e6:.2f}M)  mamba hooks={n_pin}",
        flush=True,
    )
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": a.lr, "weight_decay": 0.01},
            {"params": head_params, "lr": a.head_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    rows = load_rows(a.rows, a.limit)
    dev_rows = load_rows(a.dev, a.dev_limit) if a.dev else None
    lengths = (
        rendered_lengths(tok, rows, os.path.join(a.out, "lengths.json"))
        if a.tokens_per_batch
        else None
    )
    if lengths:
        kept = sum(1 for L in lengths if a.max_len >= L)
        print(
            f"[data] lengths: p50={sorted(lengths)[len(lengths) // 2]} max={max(lengths)} kept<= {a.max_len}: {kept}/{len(lengths)}",
            flush=True,
        )
        _probe = bucketed_batches(
            list(range(len(rows))), lengths, a.tokens_per_batch, a.bsz, a.max_len, random.Random(0)
        )
        steps_per_epoch = math.ceil(len(_probe) / a.accum)
    else:
        steps_per_epoch = math.ceil(len(rows) / (a.bsz * a.accum))
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / max(1, int(0.03 * total)))
            * 0.5
            * (1 + math.cos(math.pi * min(s, total) / max(total, 1)))
            * 0.9
            + 0.1
        ),
    )
    print(
        f"[data] train rows={len(rows)}  dev rows={len(dev_rows) if dev_rows else 0}  optimizer steps={total}",
        flush=True,
    )
    in_dev = next(model.trunk.parameters()).device
    rng = random.Random(a.seed)
    step = 0
    model.train()
    t_last = time.time()
    tok_seen = 0
    run_loss = 0.0
    run_n = 0
    skipped = 0
    for ep in range(a.epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        if lengths:
            micro = bucketed_batches(order, lengths, a.tokens_per_batch, a.bsz, a.max_len, rng)
        else:
            micro = [order[i : i + a.bsz] for i in range(0, len(order), a.bsz)]
        for idxs in micro:
            chunk = [rows[j] for j in idxs]
            rend = [render(tok, r, rng, shuffle=True) for r in chunk]
            keep = [r for r in rend if len(r["input_ids"]) <= a.max_len]
            skipped += len(rend) - len(keep)
            if not keep:
                continue
            b = collate(tok, keep, pad_id, in_dev)
            tok_seen += int(b["attention_mask"].sum())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b["input_ids"], b["attention_mask"], b["opt_spans"], b["decide_pos"])
            loss = decision_loss(logits, b["gold"], b["teacher"], b["qtype"]) / a.accum
            loss.backward()
            run_loss += loss.item() * a.accum
            run_n += 1
            if run_n % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0:
                    dt = time.time() - t_last
                    print(
                        f"[step {step}/{total}] loss={run_loss / run_n:.4f}  lr={sched.get_last_lr()[0]:.2e}  tok/s={tok_seen / dt:.0f}  skipped>{a.max_len}={skipped}",
                        flush=True,
                    )
                    run_loss = 0.0
                    run_n = 0
                    tok_seen = 0
                    t_last = time.time()
                if step % a.save_every == 0:
                    save(model, a.out, step)
        if dev_rows:
            print(
                f"[epoch {ep}] dev {evaluate(model, tok, dev_rows, pad_id, a.bsz, a.max_len)}",
                flush=True,
            )
    save(model, a.out, step)
    print("[done]", flush=True)


def save(model, out, step):
    d = os.path.join(out, f"step{step}")
    os.makedirs(d, exist_ok=True)
    model.trunk.save_pretrained(d)  # LoRA adapter only
    torch.save(model.head.state_dict(), os.path.join(d, "head.pt"))
    print(f"[save] {d}", flush=True)


if __name__ == "__main__":
    main()
