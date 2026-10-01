"""The plumb: the trained decision branch of a model, stored next to (never inside) the base weights.

    plumb.json                    spec: base model, taps, head shape, calibration
    head.safetensors              the decision head
    suffix_adapter.json/.safetensors   the suffix-only LoRA (absent when trained with --no_lora)

``plumbify train-head`` writes this layout; ``plumbify package`` (plumbify/plumbed.py) combines it with the base model
into a plumbed model directory. Paths can be local directories or Hugging Face Hub repos.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field

FORMAT = "plumb/2"
RENDER_VERSION = 2  # plumbify.core.render: bump when the decision text layout changes


@dataclass
class Conformal:
    """Split-conformal threshold fitted on held-out rows: Set = {k : p_k >= 1 - qhat}, P(gold in Set) >= 1 - alpha."""

    alpha: float
    qhat: float
    n: int


@dataclass
class Calibration:
    temperature: float = 1.0
    conformal: Conformal | None = None


@dataclass
class PlumbSpec:
    base_model: str
    d: int
    d_proj: int = 512
    has_adapter: bool = True
    render_version: int = RENDER_VERSION
    head_type: str = "decision"
    head_config: dict = field(default_factory=dict)
    taps: list = field(default_factory=list)  # layers the head reads; -1 is the final normed output
    calibration: Calibration = field(default_factory=Calibration)
    name: str = ""
    format: str = FORMAT
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=1)

    @staticmethod
    def from_dict(d: dict) -> PlumbSpec:
        cal = d.get("calibration") or {}
        conf = cal.get("conformal")
        d = {
            **d,
            "calibration": Calibration(
                cal.get("temperature", 1.0), Conformal(**conf) if conf else None
            ),
        }
        known = {f.name for f in dataclasses.fields(PlumbSpec)}
        return PlumbSpec(**{k: v for k, v in d.items() if k in known})


def fetch(path: str, name: str) -> str:
    """Local path of a file in a plumb directory or a Hub repo."""
    if os.path.isdir(path):
        return os.path.join(path, name)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(path, name)


def read_spec(path: str) -> PlumbSpec:
    try:
        spec_file = fetch(path, "plumb.json")
    except Exception as e:  # Hub errors come in several types; the message says which
        raise FileNotFoundError(f"{path}: cannot read plumb.json ({e})") from e
    if not os.path.exists(spec_file):
        raise FileNotFoundError(
            f"{path}: no plumb.json; is this a plumb or plumbed model directory?"
        )
    spec = PlumbSpec.from_dict(json.load(open(spec_file)))
    if spec.head_type != "decision":
        raise ValueError(f"{path}: unsupported plumb head type {spec.head_type!r}")
    return spec


def save(out: str, spec: PlumbSpec, head) -> None:
    """Write the spec and the head (with its calibrated temperature)."""
    from safetensors.torch import save_file

    os.makedirs(out, exist_ok=True)
    sd = {k: v.detach().contiguous().cpu() for k, v in head.state_dict().items()}
    sd["temperature"] = sd["temperature"].new_tensor(spec.calibration.temperature)
    save_file(sd, os.path.join(out, "head.safetensors"))
    with open(os.path.join(out, "plumb.json"), "w") as f:
        f.write(dataclasses.replace(spec, format=FORMAT).to_json())
