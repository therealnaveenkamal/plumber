"""Publish a plumbed model to the Hugging Face Hub with a full model card.

  python scripts/push_to_hub.py plumbed-qwen3.5-9b totum-labs/Qwen3.5-9B-plumb \
      --bench runs/bench/summary.json --rows 10k --minutes 37

The card (README.md) is rebuilt from the directory itself, the training metrics (``<dir>.train/metrics.json`` unless
``--metrics`` says otherwise) and a scripts/bench_s1_vs_s2.py summary. Base weights that are symlinks into the local
Hugging Face cache are uploaded as the files they point to. ``--dry-run`` writes the card and uploads nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

from plumber.plumbed import card_inputs, write_card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="plumbed model directory")
    ap.add_argument("repo", help="Hub repo id, e.g. totum-labs/Qwen3.5-9B-plumb")
    ap.add_argument("--bench", default=None, help="bench_s1_vs_s2.py summary.json")
    ap.add_argument(
        "--metrics",
        default=None,
        help="training metrics.json (default: <model>.train/metrics.json)",
    )
    ap.add_argument("--rows", default=None, help="training rows, as shown on the card (e.g. 10k)")
    ap.add_argument("--minutes", type=int, default=None, help="training time")
    ap.add_argument("--hardware", default="one A100 80 GB")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the card only")
    a = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi()
    base = card_inputs(a.model)["base"]
    info = api.model_info(base)
    card_data = info.card_data.to_dict() if info.card_data else {}
    metrics_path = a.metrics or os.path.join(a.model.rstrip("/") + ".train", "metrics.json")
    write_card(
        a.model,
        repo=a.repo,
        dev=json.load(open(metrics_path)) if os.path.exists(metrics_path) else None,
        bench=json.load(open(a.bench)) if a.bench else None,
        train={"rows": a.rows, "minutes": a.minutes, "hardware": a.hardware},
        license_name=card_data.get("license") or "other",
        license_link=card_data.get("license_link")
        if "huggingface.co" not in str(card_data.get("license_link"))
        else None,
    )
    if "LICENSE" in {s.rfilename for s in info.siblings or []} and not os.path.exists(
        os.path.join(a.model, "LICENSE")
    ):
        shutil.copyfile(hf_hub_download(base, "LICENSE"), os.path.join(a.model, "LICENSE"))
    print(f"card written: {os.path.join(a.model, 'README.md')}")
    if a.dry_run:
        return

    files = sorted(f for f in os.listdir(a.model) if os.path.isfile(os.path.join(a.model, f)))
    ops = [CommitOperationAdd(f, os.path.realpath(os.path.join(a.model, f))) for f in files]
    api.create_repo(a.repo, private=a.private, exist_ok=True)
    total = sum(os.path.getsize(os.path.realpath(os.path.join(a.model, f))) for f in files) / 1e9
    print(f"uploading {len(files)} files ({total:.1f} GB) to {a.repo}")
    api.create_commit(a.repo, ops, commit_message=f"Plumbed {base}")
    print(f"done: https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
