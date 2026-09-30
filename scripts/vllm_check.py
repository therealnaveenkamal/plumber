"""GPU check of the plumb vLLM plugin (one engine: generation + decisions from the same KV cache).

  python scripts/vllm_check.py --model plumbed-qwen3.5-4b \\
      --rows data/general/dev.jsonl --ref plumbed-qwen3.5-4b.train/dev_preds.jsonl

1. parity: decision requests vs the transformers System1 probabilities for the same rows (``--ref`` preds.jsonl)
2. cache: a decision after a long context reads it from the prefix cache (num_cached_tokens) and stays fast
3. generation: a greedy generation is unaffected by the plugin (same text with and without a decision in between)
4. loop: generate -> plumb_decide -> decision on the cached context -> tool response -> continue
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a plumbed model directory (plumber plumbify)")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--ref", default=None, help="System1 preds.jsonl for the same rows")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gpu_mem", type=float, default=0.7)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--max_num_seqs", type=int, default=None, help="lower it when memory is tight")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from plumber.artifact import read_spec
    from plumber.branch import Frames, Markup, parse_tool_call
    from plumber.calibration import rescale
    from plumber.core.tool import row_from_tool_args, tool_schema
    from plumber.serving.vllm.client import mid_generation_request, read_probs, standalone_request
    from plumber.training.data import load_rows

    spec = read_spec(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    markup = Markup.of(tok)
    frames = Frames.of(tok, spec.extra.get("template_kwargs"), markup)
    llm = LLM(
        model=a.model,
        enable_prefix_caching=True,
        gpu_memory_utilization=a.gpu_mem,
        max_model_len=a.max_model_len,
        **({"max_num_seqs": a.max_num_seqs} if a.max_num_seqs else {}),
    )

    def decide(ids, extra, salt=None):
        k = extra["plumb"]["k"]
        prompt = {"prompt_token_ids": ids, **({"cache_salt": salt} if salt else {})}
        out = llm.generate(
            [prompt],
            SamplingParams(max_tokens=1, temperature=0.0, logprobs=k, extra_args=extra),
            use_tqdm=False,
        )[0]
        return read_probs(out.outputs[0].logprobs[0], k), out.num_cached_tokens or 0

    # 1. parity with transformers ------------------------------------------------------------------------------
    rows = load_rows(a.rows, a.n)
    ref = {}
    # train-head's dev_preds.jsonl is written at temperature 1, before calibration; the served plumb applies the
    # fitted temperature, so the reference is rescaled to match (eval-head's preds.jsonl is already calibrated)
    t_ref = (
        spec.calibration.temperature if os.path.basename(a.ref or "") == "dev_preds.jsonl" else 1.0
    )
    if a.ref:
        for line in open(a.ref):
            p = json.loads(line)
            ref[p["id"]] = rescale(p["probs"], t_ref)
    reqs = [standalone_request(tok, r, None, spec.extra.get("template_kwargs")) for r in rows]
    params = [
        SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(r.options), extra_args=e)
        for r, (_, e) in zip(rows, reqs, strict=True)
    ]
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": i} for i, _ in reqs], params, use_tqdm=False)
    dt = time.time() - t0
    agree = n_ref = correct = 0
    diffs = []
    for r, o in zip(rows, outs, strict=True):
        p = read_probs(o.outputs[0].logprobs[0], len(r.options))
        top = max(range(len(p)), key=p.__getitem__)
        correct += top == r.gold
        if r.id in ref:
            q = ref[r.id]
            n_ref += 1
            agree += top == max(range(len(q)), key=q.__getitem__)
            diffs.append(max(abs(x - y) for x, y in zip(p, q, strict=True)))
    print(
        f"[parity] n={len(rows)} acc={correct / len(rows):.3f} ({dt:.1f}s batched, {len(rows) / dt:.0f} decisions/s)"
    )
    if n_ref:
        diffs.sort()
        print(
            f"[parity] vs transformers System1: argmax agree {agree}/{n_ref}, max|dp| median "
            f"{diffs[len(diffs) // 2]:.4f} p95 {diffs[int(0.95 * len(diffs))]:.4f}"
        )

    # 2. prefix-cache reuse and latency as the context grows -------------------------------------------------------
    base_row = next(r for r in rows if r.state)
    q = type(base_row)("q", "", base_row.question, base_row.qtype, base_row.options)
    for target in (1000, 4000, 16000):
        state = (base_row.state + "\n\n") * max(
            1, target // max(1, len(tok(base_row.state).input_ids))
        )
        text = tok.apply_chat_template(
            [{"role": "user", "content": state}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ctx = tok(text + "Let me look into this.", add_special_tokens=False).input_ids
        # the "generator" read it
        llm.generate([{"prompt_token_ids": ctx}], SamplingParams(max_tokens=1), use_tqdm=False)
        ids, extra = mid_generation_request(tok, frames, ctx, q)
        t = time.time()
        p_hit, cached = decide(ids, extra)
        t_hit = time.time() - t
        t = time.time()
        p_cold, _ = decide(ids, extra, salt=os.urandom(8).hex())
        t_cold = time.time() - t
        d = max(abs(x - y) for x, y in zip(p_hit, p_cold, strict=True))
        print(
            f"[cache] context {len(ctx):6d}: cached {cached:6d} tokens, decision {t_hit * 1000:6.1f} ms vs "
            f"{t_cold * 1000:6.1f} ms without the cache; max|dp| hit vs cold {d:.4f}"
        )

    # 3 + 4. generation unaffected; the full loop ------------------------------------------------------------------
    system = {
        "role": "system",
        "content": "You are a support agent. For routine routing, classification or yes/no "
        "checks, call the plumb_decide tool instead of deciding yourself, then act on its answer.",
    }
    user = {
        "role": "user",
        "content": "Customer email: 'I was charged twice for invoice 4411 this month. Refund one "
        "of them today or I cancel.' Which team should handle this ticket?",
    }
    prompt = tok.apply_chat_template(
        [system, user],
        tools=[tool_schema()],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok(prompt, add_special_tokens=False).input_ids
    gp = SamplingParams(
        max_tokens=400,
        temperature=0.0,
        stop=[markup.call_close],
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )
    first = llm.generate([{"prompt_token_ids": ids}], gp, use_tqdm=False)[0].outputs[0]
    again = llm.generate([{"prompt_token_ids": ids}], gp, use_tqdm=False)[0].outputs[0]
    print(
        f"[generation] greedy repeat identical after decisions ran: {first.token_ids == again.token_ids}"
    )
    ids = ids + list(first.token_ids)
    t0 = time.time()
    if first.text.endswith(markup.call_close):
        name, args = parse_tool_call(markup.calls(first.text)[-1])
        row = row_from_tool_args(args)
        t = time.time()
        probs, cached = decide(*mid_generation_request(tok, frames, ids, row))
        ans = {o.name: round(p, 3) for o, p in zip(row.options, probs, strict=True)}
        print(
            f"[loop] the model asked {name}: {args.get('question')!r} -> {ans} "
            f"(read {cached} cached tokens of {len(ids)}, {1000 * (time.time() - t):.0f} ms)"
        )
        resp = (
            frames.to_tool
            + json.dumps({"choice": max(ans, key=ans.get), "probabilities": ans})
            + frames.after_tool
        )
        ids = ids + tok(resp, add_special_tokens=False).input_ids
        cont = llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(max_tokens=120, temperature=0.0, skip_special_tokens=False),
            use_tqdm=False,
        )[0]
        print(
            f"[loop] continued (read {cont.num_cached_tokens} cached tokens): "
            f"{markup.final(cont.outputs[0].text).strip()[:200]!r}"
        )
    else:
        print(f"[loop] no tool call this time: {first.text[-200:]!r}")
    print(f"[loop] total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
