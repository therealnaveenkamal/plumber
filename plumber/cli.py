"""``plumber <command>``: each command is a module with ``main()``."""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "plumbify": (
        "plumber.training.plumbify",
        "train a plumb on an open model and write a plumbed model directory for vLLM",
    ),
    "package": (None, "turn a trained plumb into a plumbed model directory"),
    "train-head": ("plumber.training.train_head", "train a plumb (the step plumbify runs)"),
    "eval-head": ("plumber.training.eval_head", "evaluate a plumb, or the base model zero-shot"),
}


def _usage() -> str:
    w = max(map(len, COMMANDS))
    return "usage: plumber <command> [args]\n\n" + "\n".join(
        f"  {k:<{w}}  {v[1]}" for k, v in COMMANDS.items()
    )


def _package(rest: list[str]) -> int:
    from .plumbed import package

    ap = argparse.ArgumentParser("plumber package")
    ap.add_argument("plumb", help="trained plumb directory (plumb.json, head, suffix adapter)")
    ap.add_argument("out", help="plumbed model directory to write")
    ap.add_argument("--base", default=None, help="override the base model recorded in plumb.json")
    ap.add_argument(
        "--copy", action="store_true", help="copy the base weights instead of linking them"
    )
    a = ap.parse_args(rest)
    print(package(a.plumb, a.out, a.base, a.copy))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}\n\n{_usage()}")
        return 2
    if cmd == "package":
        return _package(rest)
    sys.argv = [f"plumber {cmd}", *rest]  # command modules parse sys.argv
    return importlib.import_module(COMMANDS[cmd][0]).main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
