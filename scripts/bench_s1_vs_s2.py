"""System 1 (plumb) vs System 2 (the base model answering itself) on labelled decisions, through one live server.

  python scripts/bench_s1_vs_s2.py --url http://localhost:8000 --data path/to/decision-data --out runs/bench

  s1        plumb mode "system1": one forward pass, calibrated probabilities
  s2_fast   "plumb": false, thinking off, output constrained to the option names (the model alone)
  s2_think  "plumb": false, thinking on, final "Answer: <option>" parsed (the model alone)
  s2_with_plumb_tool  the plumbed model as served: System 1 when confident, the model's judgement otherwise
  cascade   s1 when its conformal set is one option, else s2_think (computed from the same rows)

Reports accuracy, single-request latency (sequential subset), throughput under concurrency, generated tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import re
import statistics
import time

import httpx

SOURCES = {  # name -> (file under --data, rows per family)
    "decisionbench_new": ("evalsets/test_ood.jsonl", 25),
    "kev_hard": ("evalsets/test.jsonl", 15),
    "general_new": ("out/general/test_new.jsonl", 4),
}


def load(data: str, seed: int) -> list[dict]:
    rng, rows = random.Random(seed), []
    for src, (path, per_family) in SOURCES.items():
        by = collections.defaultdict(list)
        for line in open(os.path.join(data, path)):
            r = json.loads(line)
            if (
                isinstance(r["state"], str)
                and isinstance(r["question"], str)
                and r.get("gold") is not None
            ):
                by[r["meta"].get("family", "?")].append(r)
        for fam in sorted(by):
            rng.shuffle(by[fam])
            rows += [{**r, "_source": src} for r in by[fam][:per_family]]
    rng.shuffle(rows)
    return rows


def prompt(r: dict, think: bool) -> str:
    opts = "\n".join(
        f"- {o['name']}: {o['desc']}" if o["desc"] else f"- {o['name']}" for o in r["options"]
    )
    tail = (
        "Think it through, then end your reply with a final line 'Answer: <option name>'."
        if think
        else "Answer with exactly one option name."
    )
    return f"{r['state']}\n\nDecision: {r['question']}\nOptions:\n{opts}\n{tail}"


def parse_answer(text: str, names: list[str]) -> str | None:
    text = text.split("</think>")[-1]
    norm = {n.lower().strip(): n for n in names}
    for m in reversed(re.findall(r"Answer:\s*\**\s*([^\n*]+)", text)):
        cand = m.strip().strip(".`'\" ").lower()
        if cand in norm:
            return norm[cand]
        hits = [n for k, n in norm.items() if re.search(rf"(^|\W){re.escape(k)}(\W|$)", cand)]
        if len(hits) == 1:
            return hits[0]
    return None


MODEL = {"name": None}  # the served model id, read from /v1/models


def _resolve(choice, names):
    """The option a system answer names ("Low" names "Low: Reversible edit ..." when it is the only match)."""
    from plumber.serving.vllm.decisions import _matches

    if choice is None:
        return None
    hits = [n for n in names if n == choice] or [n for n in names if _matches(choice, n, "")]
    return hits[0] if len(hits) == 1 else None


async def s1(c: httpx.AsyncClient, url: str, r: dict) -> dict:
    """System 1 alone: the standard endpoint in plumb mode "system1" (the stated decision, no generation)."""
    names = [o["name"] for o in r["options"]]
    body = {
        "model": MODEL["name"],
        "messages": [{"role": "user", "content": prompt(r, False)}],
        "plumb": {"mode": "system1"},
    }
    t = time.perf_counter()
    resp = (await c.post(f"{url}/v1/chat/completions", json=body)).json()
    lat = time.perf_counter() - t
    decisions = (resp.get("plumb") or {}).get("decisions") or []
    if not decisions:  # record, don't crash the run
        print(f"  s1 error on {r['id']}: {str(resp)[:200]}", flush=True)
        return {"pred": None, "p_max": 0.0, "set_size": 99, "gen_tokens": 0, "latency": lat}
    res = decisions[0]["result"]
    probs = res["probabilities"]
    top = _resolve(max(probs, key=probs.get), names)
    return {
        "pred": names.index(top) if top else None,
        "p_max": max(probs.values()),
        "set_size": len(res.get("set") or [0, 0]),
        "gen_tokens": 0,
        "latency": lat,
    }


async def s2(c: httpx.AsyncClient, url: str, r: dict, think: bool) -> dict:
    """The model alone: vLLM's own handler ("plumb": false)."""
    names = [o["name"] for o in r["options"]]
    body = {
        "model": MODEL["name"],
        "messages": [{"role": "user", "content": prompt(r, think)}],
        "chat_template_kwargs": {"enable_thinking": think},
        "plumb": False,
    }
    if think:
        body |= {"max_tokens": 4096, "temperature": 0.6, "top_p": 0.95}
    else:
        body |= {"max_tokens": 48, "temperature": 0.0, "structured_outputs": {"choice": names}}
    t = time.perf_counter()
    resp = (await c.post(f"{url}/v1/chat/completions", json=body)).json()
    lat = time.perf_counter() - t
    ch = resp["choices"][0]
    text = (
        (ch["message"].get("content") or "") + "\n" + (ch["message"].get("reasoning_content") or "")
    )
    ans = (
        (ch["message"].get("content") or "").strip()
        if not think
        else parse_answer(ch["message"].get("content") or text, names)
    )
    return {
        "pred": names.index(ans) if ans in names else None,
        "gen_tokens": resp["usage"]["completion_tokens"],
        "prompt_tokens": resp["usage"]["prompt_tokens"],
        "truncated": ch.get("finish_reason") == "length",
        "latency": lat,
    }


DELEGATE = "For the decision below, give a final line 'Answer: <option name>'."


async def s2_delegate(c: httpx.AsyncClient, url: str, r: dict) -> dict:
    """The plumbed model: the standard endpoint with the plumb on (System 1 when confident, the model otherwise)."""
    names = [o["name"] for o in r["options"]]
    body = {
        "model": MODEL["name"],
        "messages": [
            {"role": "system", "content": DELEGATE},
            {"role": "user", "content": prompt(r, False)},
        ],
        "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t = time.perf_counter()
    resp = (await c.post(f"{url}/v1/chat/completions", json=body)).json()
    lat = time.perf_counter() - t
    plumb = resp.get("plumb") or {}
    text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    ans = parse_answer(text, names)
    strict, fallback = (names.index(ans) if ans in names else None), False
    sys_choice = _resolve((plumb.get("answer") or {}).get("choice"), names)
    if sys_choice is not None:
        ans = sys_choice
    # the reply names no option: use the structured decision
    if ans is None and plumb.get("decisions"):
        probs = plumb["decisions"][-1]["result"].get("probabilities") or {}
        best = _resolve(max(probs, key=probs.get), names) if probs else None
        if best:
            ans, fallback = best, True
    return {
        "pred": names.index(ans) if ans in names else None,
        "strict_pred": strict,
        "fallback": fallback,
        "latency": lat,
        "gen_tokens": (resp.get("usage") or {}).get("completion_tokens") or 0,
        "delegated": bool(plumb.get("decisions")),
    }


async def run_all(url, rows, method, concurrency):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=900) as c:

        async def one(r):
            async with sem:
                return await method(c, url, r)

        t = time.perf_counter()
        out = await asyncio.gather(*(one(r) for r in rows))
        return out, time.perf_counter() - t


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument(
        "--data",
        required=True,
        help="directory with evalsets/test_ood.jsonl, evalsets/test.jsonl and out/general/test_new.jsonl",
    )
    ap.add_argument("--out", default="runs/bench")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--latency_n", type=int, default=30, help="rows timed one at a time")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--methods", default="all", help="comma list; others are kept from --out/rows.jsonl"
    )
    a = ap.parse_args()
    rows = load(a.data, a.seed)
    MODEL["name"] = httpx.get(f"{a.url}/v1/models").json()["data"][0]["id"]
    os.makedirs(a.out, exist_ok=True)
    print(
        f"{len(rows)} rows: "
        + ", ".join(f"{s}={sum(r['_source'] == s for r in rows)}" for s in SOURCES)
    )
    methods = {
        "s1": s1,
        "s2_fast": lambda c, u, r: s2(c, u, r, False),
        "s2_think": lambda c, u, r: s2(c, u, r, True),
        "s2_with_plumb_tool": s2_delegate,
    }
    results, summary = {}, {}
    wanted = list(methods) if a.methods == "all" else a.methods.split(",")
    prev_rows, prev_sum = os.path.join(a.out, "rows.jsonl"), os.path.join(a.out, "summary.json")
    # same seed -> same rows; keep the other methods' results
    if a.methods != "all" and os.path.exists(prev_rows):
        old = [json.loads(line) for line in open(prev_rows)]
        assert [o["id"] for o in old] == [r["id"] for r in rows], (
            "row sample changed; rerun all methods"
        )
        prev = json.load(open(prev_sum))
        for m in methods:
            if m not in wanted and m in old[0]:
                results[m], summary[m] = [o[m] for o in old], prev[m]
    for name, fn in methods.items():
        if name not in wanted:
            continue
        seq, _ = asyncio.run(run_all(a.url, rows[: a.latency_n], fn, 1))  # single-request latency
        res, wall = asyncio.run(run_all(a.url, rows, fn, a.concurrency))  # accuracy + throughput
        results[name] = res
        ok = [x["pred"] == r["gold"] for x, r in zip(res, rows, strict=True)]
        summary[name] = {
            "accuracy": sum(ok) / len(rows),
            "invalid": sum(x["pred"] is None for x in res) / len(rows),
            "truncated": sum(bool(x.get("truncated")) for x in res) / len(rows),
            "by_source": {
                s: sum(o for o, r in zip(ok, rows, strict=True) if r["_source"] == s)
                / max(1, sum(r["_source"] == s for r in rows))
                for s in SOURCES
            },
            "latency_p50_ms": 1000 * statistics.median(x["latency"] for x in seq),
            "latency_p95_ms": 1000 * pct([x["latency"] for x in seq], 0.95),
            "throughput_per_s": len(rows) / wall,
            "gen_tokens_mean": statistics.mean(x["gen_tokens"] for x in res),
            **(
                {
                    "delegated": sum(x["delegated"] for x in res) / len(rows),
                    "accuracy_text_only": sum(
                        x["strict_pred"] == r["gold"] for x, r in zip(res, rows, strict=True)
                    )
                    / len(rows),
                    "used_decision_fallback": sum(x["fallback"] for x in res) / len(rows),
                }
                if "delegated" in res[0]
                else {}
            ),
        }
        print(
            f"[{name}] acc {summary[name]['accuracy']:.3f}  p50 {summary[name]['latency_p50_ms']:.0f} ms  "
            f"{summary[name]['throughput_per_s']:.1f}/s  gen tokens {summary[name]['gen_tokens_mean']:.0f}",
            flush=True,
        )
    s1r, thr = results["s1"], results["s2_think"]
    for label, keep in (
        ("cascade (conformal singleton)", lambda x: x["set_size"] == 1),
        ("cascade (p_max >= 0.9)", lambda x: x["p_max"] >= 0.9),
    ):
        kept = [keep(x) for x in s1r]
        pred = [x["pred"] if k else t["pred"] for x, t, k in zip(s1r, thr, kept, strict=True)]
        acc = sum(p == r["gold"] for p, r in zip(pred, rows, strict=True)) / len(rows)
        share = sum(kept) / len(rows)
        lat = summary["s1"]["latency_p50_ms"] + (1 - share) * summary["s2_think"]["latency_p50_ms"]
        tok = (1 - share) * summary["s2_think"]["gen_tokens_mean"]
        summary[label] = {
            "accuracy": acc,
            "system1_share": share,
            "expected_latency_ms": lat,
            "gen_tokens_mean": tok,
        }
        s1_acc = sum(
            x["pred"] == r["gold"] for x, r, k in zip(s1r, rows, kept, strict=True) if k
        ) / max(1, sum(kept))
        summary[label]["system1_accuracy_on_kept"] = s1_acc
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(a.out, "rows.jsonl"), "w") as f:
        for i, r in enumerate(rows):
            f.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "source": r["_source"],
                        "family": r["meta"].get("family"),
                        "gold": r["gold"],
                        **{m: results[m][i] for m in results},
                    }
                )
                + "\n"
            )
    print("\n" + json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
