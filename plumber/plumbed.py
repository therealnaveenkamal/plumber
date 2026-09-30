"""A plumbed model: an ordinary model directory that vLLM (with the plumber plugin) serves natively.

    plumbed-qwen3.5-4b/
      config.json                 the base config, architectures = ["Plumb<BaseArchitecture>"], plus a "plumb" block
      *.safetensors (+ index)     the base weights, untouched (linked from the Hugging Face cache, or copied)
      tokenizer files, chat template, generation_config.json
      plumb.json                  the plumb spec: base model, taps, head shape, calibration
      head.safetensors            the decision head
      suffix_adapter.json/.safetensors   the suffix-only LoRA (acts on decision tokens only; never merged)
      README.md                   model card

``vllm serve plumbed-qwen3.5-4b`` loads the base weights through vLLM's own class for the base architecture and
attaches the plumb (plumber.serving.vllm). ``System1.load(dir)`` loads the same directory with transformers.
"""

from __future__ import annotations

import json
import os
import shutil

PLUMB_FILES = (
    "plumb.json",
    "head.safetensors",
    "suffix_adapter.json",
    "suffix_adapter.safetensors",
)
PLUMB_PREFIX = "Plumb"  # a plumbed model's architecture is Plumb<BaseArchitecture>
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
BASE_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.model",
    "*.jinja",
    "*.tiktoken",
    "*.py",
    "tokenizer*",
]


def plumb_dir(model: str) -> str | None:
    """The local directory holding the plumb of ``model`` (a directory or a Hub repo), or None if it has none."""
    if os.path.isdir(model):
        return model if os.path.exists(os.path.join(model, "plumb.json")) else None
    from huggingface_hub import snapshot_download

    local = snapshot_download(model, allow_patterns=list(PLUMB_FILES))
    return local if os.path.exists(os.path.join(local, "plumb.json")) else None


def base_snapshot(base: str) -> str:
    if os.path.isdir(base):
        return base
    from huggingface_hub import snapshot_download

    return snapshot_download(base, allow_patterns=BASE_PATTERNS)


def write_weight_index(out: str) -> None:
    """Give a single-file checkpoint a ``model.safetensors.index.json``. Loaders (vLLM, transformers) read every
    ``*.safetensors`` in a directory that has no index, which would include the plumb's own files."""
    if any(f.endswith(".safetensors.index.json") for f in os.listdir(out)):
        return
    from safetensors import safe_open

    weights = sorted(
        f for f in os.listdir(out) if f.endswith(".safetensors") and f not in PLUMB_FILES
    )
    weight_map = {}
    for f in weights:
        with safe_open(os.path.join(out, f), "pt") as st:
            weight_map.update(dict.fromkeys(st.keys(), f))
    if weight_map:
        with open(os.path.join(out, "model.safetensors.index.json"), "w") as fh:
            json.dump({"metadata": {}, "weight_map": weight_map}, fh, indent=1)


def package(
    plumb_dir: str,
    out: str,
    base: str | None = None,
    copy: bool = False,
    metrics: dict | None = None,
) -> str:
    """Write the plumbed model directory for a trained plumb (``plumber train-head`` output) on its base."""
    from .artifact import read_spec

    spec = read_spec(plumb_dir)
    base = base or spec.base_model
    src = base_snapshot(base)
    os.makedirs(out, exist_ok=True)
    for name in sorted(os.listdir(src)):
        p = os.path.join(src, name)
        if not os.path.isfile(p) or name in PLUMB_FILES or name in ("config.json", "README.md"):
            continue
        dst = os.path.join(out, name)
        if os.path.lexists(dst):
            os.remove(dst)
        if name.endswith(WEIGHT_SUFFIXES) and not copy:
            # the weights stay where they are: no second copy on disk
            os.symlink(os.path.realpath(p), dst)
        else:
            shutil.copyfile(os.path.realpath(p), dst)
    write_weight_index(out)
    cfg = json.load(open(os.path.join(src, "config.json")))
    arch = cfg["architectures"][0]
    if arch.startswith(PLUMB_PREFIX):
        raise ValueError(f"{base} is already a plumbed model")
    cfg["architectures"] = [PLUMB_PREFIX + arch]
    cfg["plumb"] = {
        "format": spec.format,
        "base_model": base,
        "base_architecture": arch,
        "taps": spec.taps,
        "suffix_adapter": spec.has_adapter,
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for name in PLUMB_FILES:
        p = os.path.join(plumb_dir, name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(out, name))
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(card(base, arch, spec, metrics))
    return out


def card(base: str, arch: str, spec, metrics: dict | None) -> str:
    conf = spec.calibration.conformal
    lines = [
        f"# Plumbed {base}",
        "",
        f"`{base}` with a **plumb**: a System 1 decision module that answers routine judgements (routing, "
        "classification, priority, yes/no, ratings) in one forward pass, with calibrated probabilities and a "
        "conformal set, reading the model's own KV cache. The base weights are untouched: generation is exactly the "
        "base model's.",
        "",
        "## Serve",
        "",
        "```bash",
        'pip install "plumber[vllm] @ git+https://github.com/therealnaveenkamal/plumber"',
        "vllm serve <this directory>",
        "```",
        "",
        "`/v1/chat/completions` then routes routine decisions to System 1 (see the `plumb` field of each response).",
        "",
        "## What is inside",
        "",
        f"- base architecture `{arch}`, served as `Plumb{arch}`",
        f"- decision head reading layers {spec.taps} (−1 = final output)",
        f"- suffix-only LoRA: {'yes' if spec.has_adapter else 'no (head only)'}",
        f"- temperature {spec.calibration.temperature:.3f}"
        + (f"; conformal alpha {conf.alpha} (qhat {conf.qhat:.3f}, n={conf.n})" if conf else ""),
    ]
    if metrics:
        lines += ["", "## Dev metrics", "", "```json", json.dumps(metrics, indent=1)[:2000], "```"]
    return "\n".join(lines) + "\n"
