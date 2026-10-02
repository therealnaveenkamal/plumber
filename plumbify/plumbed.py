"""A plumbed model: an ordinary model directory that vLLM (with the plumbify plugin) serves natively.

    plumbed-qwen3.5-4b/
      config.json                 the base config, architectures = ["Plumb<BaseArchitecture>"], plus a "plumb" block
      *.safetensors (+ index)     the base weights, untouched (linked from the Hugging Face cache, or copied)
      tokenizer files, chat template, generation_config.json
      plumb.json                  the plumb spec: base model, taps, head shape, calibration
      head.safetensors            the decision head
      suffix_adapter.json/.safetensors   the suffix-only LoRA (acts on decision tokens only; never merged)
      README.md                   model card

``vllm serve plumbed-qwen3.5-4b`` loads the base weights through vLLM's own class for the base architecture and
attaches the plumb (plumbify.serving.vllm). ``System1.load(dir)`` loads the same directory with transformers.
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
) -> str:
    """Write the plumbed model directory for a trained plumb (``plumbify train-head`` output) on its base."""
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
    write_card(out)
    return out


def card_inputs(out: str) -> dict:
    """What a plumbed directory says about itself: base, architecture, spec, adapter config, head size."""
    from safetensors import safe_open

    from .artifact import read_spec

    cfg = json.load(open(os.path.join(out, "config.json")))
    spec = read_spec(out)
    adapter_cfg = os.path.join(out, "suffix_adapter.json")
    with safe_open(os.path.join(out, "head.safetensors"), "pt") as f:
        names = f.keys()
        head_params = sum(_numel(f.get_slice(k).get_shape()) for k in names if k != "temperature")
    return {
        "base": cfg["plumb"]["base_model"],
        "arch": cfg["plumb"]["base_architecture"],
        "spec": spec,
        "adapter": json.load(open(adapter_cfg)) if os.path.exists(adapter_cfg) else None,
        "head_params": head_params,
    }


def _numel(shape) -> int:
    n = 1
    for x in shape:
        n *= x
    return n


def write_card(out: str, **kw) -> str:
    """Write README.md (the model card) into a plumbed model directory; ``kw`` goes to ``card``."""
    i = card_inputs(out)
    text = card(i.pop("base"), i.pop("arch"), i.pop("spec"), **{**i, **kw})
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(text)
    return text


REPO_URL = "https://github.com/therealnaveenkamal/plumbify"
EVAL_SETS = {
    "decisionbench_new": "DecisionBench, held-out task families (150)",
    "kev_hard": "Hard-skill templates, held out (105)",
    "general_new": "General decisions, unseen templates (192)",
}


LICENSE_NAMES = {"apache-2.0": "Apache 2.0", "mit": "MIT"}


def _cells(bench: dict) -> dict:
    """The 2x2 of a benchmark summary. With thinking, the model alone counts in its better prompt setup."""
    thinking = [bench[k] for k in ("s2_think", "s2_think_tool") if k in bench]
    return {
        "model alone, thinking off": bench.get("s2_fast"),
        "plumbed, thinking off": bench.get("s2_with_plumb_tool"),
        "model alone, thinking on": max(thinking, key=lambda r: r["accuracy"])
        if thinking
        else None,
        "plumbed, thinking on": bench.get("s2_with_plumb_think"),
    }


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def _secs(ms: float | None) -> str:
    if ms is None:
        return "–"
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def card(
    base: str,
    arch: str,
    spec,
    *,
    repo: str | None = None,
    adapter: dict | None = None,
    head_params: int | None = None,
    bench: dict | None = None,
    train: dict | None = None,
    license_name: str = "apache-2.0",
    license_link: str | None = None,
) -> str:
    """The model card (README.md) of a plumbed model. ``bench`` is a scripts/bench_s1_vs_s2.py summary, ``train``
    free-form facts about the run (rows, minutes, hardware)."""
    name = (repo or base).split("/")[-1]
    ref = repo or "path/to/" + name
    conf = spec.calibration.conformal
    train = train or {}
    meta = [
        "---",
        f"license: {license_name}",
        *([f"license_link: {license_link}"] if license_link else []),
        f"base_model: {base}",
        "base_model_relation: adapter",
        "library_name: vllm",
        "pipeline_tag: text-generation",
        "tags: [plumbify, plumb, jev, decision-making, calibration, vllm-plugin, lora]",
    ]
    if bench:
        cells = _cells(bench)
        meta += [
            "model-index:",
            f"- name: {name}",
            "  results:",
            "  - task: {type: text-classification, name: Typed decisions}",
            "    dataset: {type: plumbify-heldout-decisions, name: Plumbify held-out decisions (447)}",
            "    metrics:",
            *[
                f"    - {{type: accuracy, name: Accuracy ({label}), value: {run['accuracy']:.3f}}}"
                for label, run in cells.items()
                if run
            ],
        ]
    meta.append("---")

    out = [
        *meta,
        "",
        f"# {name}",
        "",
        f"[{base}](https://huggingface.co/{base}) with a **plumb**: a decision head built into the model. It answers",
        "typed decisions (pick one of these options, yes or no, a score on a scale) in one forward pass, with a",
        "calibrated probability for every option, reading the KV cache the conversation already filled. It is a",
        "Jev-style decision model inside the LLM instead of next to it. The base weights are unchanged, so the model",
        f"chats, reasons and calls tools exactly like `{base}`.",
        "",
        f"Made with [Plumbify]({REPO_URL}) and served by vLLM through the Plumbify plugin.",
    ]

    if bench:
        c = _cells(bench)
        off_m, off_p = c["model alone, thinking off"], c["plumbed, thinking off"]
        on_m, on_p = c["model alone, thinking on"], c["plumbed, thinking on"]

        def cell(run):
            return (
                f"{_pct(run.get('accuracy'))} ({_secs(run.get('latency_p50_ms'))})" if run else "–"
            )

        def row(alone, plumbed):
            """Both cells, the more accurate one in bold."""
            a, p = cell(alone), cell(plumbed)
            if alone and plumbed and plumbed["accuracy"] > alone["accuracy"]:
                p = f"**{p}**"
            elif alone and plumbed and alone["accuracy"] > plumbed["accuracy"]:
                a = f"**{a}**"
            return f"{a} | {p}"

        def by(run, k):
            return _pct(((run or {}).get("by_source") or {}).get(k))

        out += [
            "",
            "## Results",
            "",
            "447 held-out decisions (task families and templates the plumb never trained on), on one vLLM server. Each",
            "cell is accuracy (latency p50 for one request). With thinking on, the model alone is scored in its better",
            "prompt setup, so the plumb is compared with the model at its best.",
            "",
            f"| | {base} alone | {name} |",
            "|---|---:|---:|",
            f"| Thinking off | {row(off_m, off_p)} |",
            f"| Thinking on | {row(on_m, on_p)} |",
            "",
            "| Eval set | Off: alone | Off: plumbed | On: alone | On: plumbed |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {label} | {by(off_m, k)} | {by(off_p, k)} | {by(on_m, k)} | {by(on_p, k)} |"
                for k, label in EVAL_SETS.items()
            ],
        ]

    out += [
        "",
        "## Use",
        "",
        "```bash",
        'pip install "plumbify[vllm]"',
        f"vllm serve {ref}",
        "```",
        "",
        "```bash",
        "curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{",
        f'  "model": "{ref}",',
        '  "messages": [{"role": "user", "content": "Ticket: I was charged twice.\\n\\nWhich team should handle this?\\n- billing\\n- technical\\n- sales"}]',
        "}' | jq .plumb.answer",
        "```",
        "",
        "Decisions stated in a message go to the plumb before the model replies; the model can also hand decisions to",
        "it mid-conversation through a `plumb_decide` tool, answered from the KV cache. Each response carries a `plumb`",
        'field with the probabilities, a conformal set and the final answer. `"plumb": false` in a request serves it',
        f"with the base model alone. Request options and response fields are in the [guide]({REPO_URL}/blob/main/docs/guide.md).",
        "",
        "## What is in this repository",
        "",
        f"- The weights of [{base}](https://huggingface.co/{base}), unchanged.",
        f"- `config.json` with the architecture `{PLUMB_PREFIX}{arch}`, which tells vLLM and the plugin to attach the plumb.",
        f"- `head.safetensors`: the decision head{f' ({head_params / 1e6:.1f}M parameters)' if head_params else ''}. It reads "
        f"the model's hidden states at layers {', '.join(str(t) for t in spec.taps if t != -1)} and the final output.",
    ]
    if spec.has_adapter:
        r = (adapter or {}).get("r", "")
        targets = ", ".join((adapter or {}).get("target_modules", []))
        out.append(
            f"- `suffix_adapter.*`: a LoRA (rank {r}) on {targets or 'the attention and MLP projections'}, active only "
            "while the model reads a decision. It is never merged."
        )
    out += [
        f"- `plumb.json`: taps, head shape and calibration (temperature {spec.calibration.temperature:.3f}"
        + (f", conformal threshold {conf.qhat:.3f} at alpha {conf.alpha}" if conf else "")
        + ").",
        "",
        "## Training",
        "",
        f"`plumbify train` on {train.get('rows', 'about 10k')} decision rows drawn from about 200 decision families",
        "(support, finance, coding, safety, medicine, law, engineering) plus solver-labelled hard-skill rows, one epoch,",
        f"{train.get('hardware', 'one A100 80 GB')}"
        + (f", {train['minutes']} minutes" if train.get("minutes") else "")
        + ". The base model is frozen: only the decision head and the suffix adapter are trained, then the head's",
        "temperature and a conformal threshold are fitted on a held-out development set.",
        "",
        "## Limitations",
        "",
        "- Needs vLLM 0.30.0 with the Plumbify plugin, on one GPU (no tensor or pipeline parallelism).",
        "- The results above are for decisions stated in the prompt. Decisions the model hands off mid-generation use the",
        "  same head but see the whole conversation as context, which training did not include.",
        "- The training decisions are in English. The plumb chooses among the options it is given; it is not a",
        "  safety classifier and does not check that the options make sense.",
        "",
        "## License",
        "",
        f"{LICENSE_NAMES.get(license_name, license_name)}, inherited from [{base}](https://huggingface.co/{base})"
        + (f" ([terms]({license_link}))." if license_link else "."),
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{plumbify2026,",
        "  title  = {Plumbify: a System 1 decision branch for open language models},",
        "  author = {Kamalakannan, Naveenraj},",
        "  year   = {2026},",
        f"  url    = {{{REPO_URL}}}",
        "}",
        "```",
    ]
    return "\n".join(out) + "\n"
