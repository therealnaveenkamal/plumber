"""plumber CLI: serve | decide | eval | train."""

import argparse
import json
import sys

from .engine import DEFAULT_BASE, Plumber


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: plumber {serve,decide,plumbify,train,eval,calibrate,export} ...")
        print("  serve   --model REPO --port 8123")
        print("  decide  --model REPO --state TEXT --question TEXT --options a,b,c [--desc ...]")
        print("  eval    --ckpt DIR --rows ROWS.jsonl --out DIR")
        print("  calibrate --ckpt DIR --preds DIR/preds.jsonl [--apply OTHER/preds.jsonl]")
        print(
            "  plumbify --base HF_ID --rows ROWS.jsonl --out DIR   (fit a plumb on any causal LM)"
        )
        print("  train   --rows ROWS.jsonl --out DIR ...             (same, Nemotron default)")
        print("  export  --ckpt DIR --out DIR --card README.md [--repo org/name]")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "serve":
        from .server import main as m

        sys.argv = ["plumber serve", *rest]
        return m()
    if cmd == "eval":
        from .training.eval import main as m

        sys.argv = ["plumber eval", *rest]
        return m()
    if cmd == "train":
        from .training.train import main as m

        sys.argv = ["plumber train", *rest]
        return m()
    if cmd == "plumbify":  # fit a plumb (LoRA + head) on any base LM = turn it into a decision model
        from .training.train import main as m

        sys.argv = ["plumber plumbify", *rest]
        return m()
    if cmd == "calibrate":
        from .training.calibrate import main as m

        sys.argv = ["plumber calibrate", *rest]
        return m()
    if cmd == "export":
        from .training.export import main as m

        sys.argv = ["plumber export", *rest]
        return m()
    if cmd == "decide":
        ap = argparse.ArgumentParser("plumber decide")
        ap.add_argument("--model", required=True)
        ap.add_argument("--base", default=DEFAULT_BASE)
        ap.add_argument("--state", required=True)
        ap.add_argument("--question", required=True)
        ap.add_argument("--options", required=True, help="comma-separated option names")
        ap.add_argument("--desc", default=None, help="comma-separated descriptions, same order")
        ap.add_argument("--type", default="choice", choices=["choice", "noul", "score"])
        a = ap.parse_args(rest)
        eng = Plumber(a.model, base=a.base)
        names = [x.strip() for x in a.options.split(",")]
        descs = [x.strip() for x in a.desc.split(",")] if a.desc else [""] * len(names)
        crit = (
            dict(zip(names, descs, strict=True))
            if a.type == "choice"
            else (names if a.type == "score" else None)
        )
        print(
            json.dumps(
                eng.decide(
                    a.state, {"q": {"type": a.type, "instructions": a.question, "criteria": crit}}
                )["answers"]["q"],
                indent=1,
            )
        )
        return 0
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
