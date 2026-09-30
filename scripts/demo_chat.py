"""Live chat with a plumbed model over the standard /v1/chat/completions: the model generates (System 2) and hands
routine decisions to its plumb (System 1), which answers from the same KV cache. Decisions appear inline.

  python scripts/demo_chat.py                          # server on localhost:8000
  python scripts/demo_chat.py --think --url http://host:8000

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

DEFAULT_SYSTEM = (
    "You are a helpful assistant for ACME's support team. Whenever you need a routine judgement — routing a "
    "ticket, setting a priority, a yes/no check, classifying something — call the plumb_decide tool with the "
    "question and the options instead of deciding yourself, then continue using its answer."
)


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
    print(f"  │ {c.green}{c.bold}→ {top}{c.reset}{c.cyan}   conformal set: {a.get('set')}")
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
        "plumb": {"trust": opts["trust"]},
    }
    new, content, n_dec, t0, thinking, finish = [], "", 0, time.time(), False, None
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
            if delta.get("content"):
                if thinking:
                    sys.stdout.write(f"{c.reset}\n")
                    thinking = False
                content += delta["content"]
                sys.stdout.write(delta["content"])
            sys.stdout.flush()
            finish = choice.get("finish_reason") or finish
    new.append({"role": "assistant", "content": content.strip()})
    print(
        f"\n{c.dim}[{time.time() - t0:.1f}s · {n_dec} decision{'s' * (n_dec != 1)} · finish={finish}]{c.reset}"
    )
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
    a = ap.parse_args()
    c = Colors(sys.stdout.isatty() and not a.plain)
    opts = {"think": a.think, "trust": a.trust, "max_tokens": a.max_tokens}
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
    print(
        f"{c.bold}Chat with {model} + plumb{c.reset} at {a.url}  (thinking {'on' if opts['think'] else 'off'}, "
        f"trust {opts['trust']})  — /help for commands\n"
    )
    while True:
        try:
            user = input(f"{c.bold}you:{c.reset} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
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
