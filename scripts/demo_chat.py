"""Live chat with a plumbed model over the standard /v1/chat/completions: the model generates (System 2) and hands
routine decisions to its plumb (System 1), which answers from the same KV cache. Decisions appear inline.

  python scripts/demo_chat.py                          # server on localhost:8000
  python scripts/demo_chat.py --think --url http://host:8000
  python scripts/demo_chat.py --no-plumb --think       # the normal model, for comparison

Type \\n in a message for a line break (e.g. a question followed by bulleted options).

Commands: /think (toggle reasoning), /system <prompt>, /trust <0-1>, /reset, /history, /help, /quit
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time

import httpx

with contextlib.suppress(ImportError):
    import readline  # noqa: F401  (line editing and history for input())

# A plain system prompt: with the plumb on, the server tells the model about its decision module itself
DEFAULT_SYSTEM = "You are a helpful assistant."


class Colors:
    def __init__(self, on: bool):
        c = (lambda code: f"\033[{code}m") if on else (lambda code: "")
        self.dim, self.bold, self.cyan, self.green, self.yellow, self.red, self.reset = (
            c("2"),
            c("1"),
            c("36"),
            c("32"),
            c("33"),
            c("31"),
            c("0"),
        )


class Bold:
    """Streams text with markdown **bold** shown as terminal bold; a marker split across chunks is held back."""

    def __init__(self, c: Colors):
        self.c, self.on, self.held = c, False, ""

    def __call__(self, text: str) -> str:
        text, self.held = self.held + text, ""
        if text.endswith("*") and not text.endswith("**"):
            text, self.held = text[:-1], "*"
        parts = text.split("**")
        out = parts[0]
        for part in parts[1:]:
            self.on = not self.on
            out += (self.c.bold if self.on else self.c.reset) + part
        return out


def bar(p: float, width: int = 24) -> str:
    return "█" * max(1, round(p * width)) if p >= 0.005 else "▏"


def decision_box(c: Colors, ev: dict) -> None:
    a, args = ev["result"], ev.get("arguments") or {"question": ev.get("question")}
    probs = a["probabilities"]
    top = max(probs, key=probs.get)
    w = min(40, max(len(k) for k in probs))
    src = "stated in your message" if ev.get("source") == "stated" else "the model decided to ask"
    print(f"\n\n  {c.cyan}┌─ System 1 decision ({src}) " + "─" * 30)
    print(f"  │ {c.bold}Q:{c.reset}{c.cyan} {args.get('question', '')}")
    sure = a.get("set")
    sure = (
        ""
        if sure is None
        else "   conformal set: 1 option (confident)"
        if len(sure) == 1
        else f"   conformal set: {len(sure)} options (unsure)"
    )
    print(f"  │ {c.green}{c.bold}→ {top[:60]}{c.reset}{c.cyan}{sure}")
    for k, p in sorted(probs.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  │   {k[:w]:<{w}}  {c.green}{bar(p)}{c.cyan} {p:.3f}")
    cached = (
        f"read {ev['cached_tokens']:,} tokens from the KV cache, "
        if ev.get("cached_tokens")
        else ""
    )
    print(
        f"  │ {c.yellow}{ev['latency_ms']:.0f} ms · {cached}computed {ev['computed_tokens']:,}{c.cyan}"
    )
    print("  └" + "─" * 64 + c.reset)


# Listed (never called) when --tool-prompt is on: Qwen3.5 thinks much less whenever its prompt lists a tool, and the
# plumbed model's prompt always lists plumb_decide, so this keeps the two sides' prompts alike.
UNRELATED_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def report(path: str | None, **row) -> None:
    """Append one JSON line (a turn's start or end) for scripts/demo/status.py."""
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


def turn(
    client: httpx.Client, url: str, model: str, messages: list, opts: dict, c: Colors
) -> list[dict]:
    """One assistant turn over the standard /v1/chat/completions (SSE). Returns the messages for the history."""
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": opts["max_tokens"],
        "chat_template_kwargs": {"enable_thinking": opts["think"]},
        "plumb": {"trust": opts["trust"]} if opts["plumb"] else False,
    }
    if opts.get("tool_prompt") and not opts["plumb"]:
        body |= {"tools": [UNRELATED_TOOL], "tool_choice": "none"}
    new, content, n_dec, t0, thinking, finish = [], "", 0, time.time(), False, None
    opts["turn"] = opts.get("turn", 0) + 1
    report(opts.get("report"), turn=opts["turn"], start=t0)
    # Without the plumb, vLLM's own handler returns the model's thinking inline, ending at </think>
    inline_think, pending = opts["think"] and not opts["plumb"], ""
    bold = Bold(c)
    sys.stdout.write(f"{c.bold}model:{c.reset} ")
    sys.stdout.flush()
    with client.stream("POST", f"{url}/v1/chat/completions", json=body) as resp:
        if resp.status_code != 200:
            print(f"{c.red}server error {resp.status_code}: {resp.read().decode()[:300]}{c.reset}")
            return []
        for line in resp.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[len("data: ") :])
            if "error" in ch:
                print(f"{c.red}{ch['error'].get('message')}{c.reset}")
                continue
            ev = ch.get("plumb") or {}
            if ev.get("event") == "decision":
                decision_box(c, ev)
                n_dec += 1
                new.append(
                    {
                        "role": "assistant",
                        "content": content.strip(),
                        "tool_calls": [
                            {
                                "id": f"plumb{n_dec}",
                                "type": "function",
                                "function": {
                                    "name": "plumb_decide",
                                    "arguments": json.dumps(ev.get("arguments") or {}),
                                },
                            }
                        ],
                    }
                )
                new.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"plumb{n_dec}",
                        "content": json.dumps(ev["result"]),
                    }
                )
                content = ""
                sys.stdout.write(f"{c.bold}model:{c.reset} ")
                continue
            choice = (ch.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                if not thinking:
                    sys.stdout.write(f"{c.dim}(thinking) ")
                    thinking = True
                sys.stdout.write(f"{c.dim}{delta['reasoning_content']}{c.reset}")
            text = delta.get("content") or ""
            if text and inline_think:
                pending += text
                if "</think>" not in pending:
                    if not thinking:
                        sys.stdout.write(f"{c.dim}(thinking) ")
                        thinking = True
                    keep = len(pending) - len("</think>")
                    if keep > 0:
                        sys.stdout.write(f"{c.dim}{pending[:keep].replace('<think>', '')}{c.reset}")
                        pending = pending[keep:]
                    text = ""
                else:
                    before, _, text = pending.partition("</think>")
                    sys.stdout.write(f"{c.dim}{before}{c.reset}")
                    inline_think, text = False, text.lstrip("\n")
            if text:
                if thinking:
                    sys.stdout.write(f"{c.reset}\n")
                    thinking = False
                content += text
                sys.stdout.write(bold(text))
            sys.stdout.flush()
            finish = choice.get("finish_reason") or finish
    new.append({"role": "assistant", "content": content.strip()})
    took = time.time() - t0
    report(opts.get("report"), turn=opts["turn"], seconds=round(took, 2), decisions=n_dec)
    how = (
        f"{n_dec} plumb decision{'s' * (n_dec != 1)}"
        if opts["plumb"]
        else "normal model" + (", thinking" if opts["think"] else "")
    )
    print(f"\n{c.yellow}{c.bold}⏱ {took:.1f} s{c.reset}{c.dim}  ·  {how}{c.reset}")
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--think", action="store_true", help="start with reasoning on")
    ap.add_argument(
        "--trust", type=float, default=0.7, help="System 1 decides when at least this confident"
    )
    ap.add_argument("--max_tokens", type=int, default=1500)
    ap.add_argument("--plain", action="store_true", help="no colours")
    ap.add_argument(
        "--no-plumb",
        action="store_true",
        help='the normal model ("plumb": false), for comparison',
    )
    ap.add_argument(
        "--report", default=None, help="append per-turn timings to this file (JSON lines)"
    )
    ap.add_argument(
        "--tool-prompt",
        action="store_true",
        help="with --no-plumb: list an unrelated tool, so the prompt is shaped like the plumbed model's",
    )
    a = ap.parse_args()
    c = Colors(sys.stdout.isatty() and not a.plain)
    opts = {
        "think": a.think,
        "trust": a.trust,
        "max_tokens": a.max_tokens,
        "plumb": not a.no_plumb,
        "report": a.report,
        "tool_prompt": a.tool_prompt,
    }
    system, history = a.system, []
    client = httpx.Client(timeout=600)
    try:
        client.get(f"{a.url}/health").raise_for_status()
        model = client.get(f"{a.url}/v1/models").json()["data"][0]["id"]
    except Exception as e:
        sys.exit(
            f"cannot reach {a.url} ({e}). Start one with `vllm serve <plumbed model dir>`, or forward a remote "
            "server's port: ssh -L 8000:localhost:8000 <gpu host>"
        )
    label = "WITH PLUMB" if opts["plumb"] else "NORMAL MODEL"
    colour = "\033[1;30;48;5;114m" if opts["plumb"] else "\033[1;97;48;5;124m"
    banner = f"{colour}  {label}  {c.reset}" if c.bold else label
    print(f"{banner}  {c.dim}{model} · thinking {'on' if opts['think'] else 'off'}{c.reset}\n")
    while True:
        try:
            user = input(f"{c.bold}you:{c.reset} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        user = user.replace("\\n", "\n")
        if user.startswith("/"):
            cmd, _, arg = user.partition(" ")
            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/think":
                opts["think"] = not opts["think"]
                print(f"{c.dim}thinking {'on' if opts['think'] else 'off'}{c.reset}")
            elif cmd == "/system":
                system = arg or DEFAULT_SYSTEM
                print(f"{c.dim}system prompt set{c.reset}")
            elif cmd == "/trust":
                try:
                    opts["trust"] = float(arg)
                    print(f"{c.dim}System 1 decides when p >= {opts['trust']}{c.reset}")
                except ValueError:
                    print(f"{c.red}usage: /trust 0.7{c.reset}")
            elif cmd == "/reset":
                history = []
                print(f"{c.dim}conversation cleared{c.reset}")
            elif cmd == "/history":
                print(json.dumps(history, indent=1)[-3000:])
            else:
                print(
                    "/think  toggle reasoning   /system <prompt>   /trust <0-1>   /reset   /history   /quit"
                )
            continue
        history.append({"role": "user", "content": user})
        messages = ([{"role": "system", "content": system}] if system else []) + history
        try:
            history += turn(client, a.url, model, messages, opts, c)
        except httpx.HTTPError as e:
            print(f"\n{c.red}request failed: {e}{c.reset}")
            history.pop()
        print()


if __name__ == "__main__":
    main()
