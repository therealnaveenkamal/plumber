"""vLLM plugin: native serving of plumbed models. ``pip install plumber`` and::

    vllm serve ./plumbed-qwen3.5-4b          # a directory made by `plumber plumbify`

The model generates exactly like its base; ``/v1/chat/completions`` gains a System 1 decision path that answers
routine judgements from the same KV cache (see model.py, hooks.py, openai.py). No extra flags or environment.

``register`` is the ``vllm.general_plugins`` entry point, which vLLM loads in every process:

- every architecture vLLM supports gets a lazily-resolved ``Plumb<Arch>`` (the base class plus the plumb), with the
  base's per-architecture config hook and room for decisions with many options (``max_logprobs``);
- in the API server process, the standard ``/v1/chat/completions`` route is replaced by the plumb-aware one, which
  falls back to vLLM's own handler for models without a plumb.
"""

from __future__ import annotations

import sys

from ...plumbed import PLUMB_PREFIX

PLUMB_MAX_LOGPROBS = 256  # decision answers come back as logprobs: one per option
# the vLLM-internal pieces (hooks.py, model.py, openai.py) are written against these
TESTED_VLLM = ("0.30.0",)
_warned = False


def _config_hook(base_hook):
    from vllm.model_executor.models.config import VerifyAndUpdateConfig

    def verify_and_update_config(vllm_config) -> None:
        if base_hook is not None:
            base_hook.verify_and_update_config(vllm_config)
        mc = vllm_config.model_config
        if mc.max_logprobs != -1 and mc.max_logprobs < PLUMB_MAX_LOGPROBS:
            mc.max_logprobs = PLUMB_MAX_LOGPROBS

    return type(
        "PlumbConfig",
        (base_hook or VerifyAndUpdateConfig,),
        {"verify_and_update_config": staticmethod(verify_and_update_config)},
    )


def _check_version() -> None:
    global _warned
    import vllm

    if _warned or vllm.__version__ in TESTED_VLLM:
        return
    _warned = True
    import logging

    logging.getLogger("vllm.plumber").warning(
        "plumber's vLLM plugin is tested with vLLM %s; this is vLLM %s. It hooks vLLM internals, so check a plumbed "
        "model with scripts/vllm_check.py (parity with transformers) before trusting its decisions.",
        " / ".join(TESTED_VLLM),
        vllm.__version__,
    )


def register() -> None:
    """Runs in every vLLM process, possibly more than once: names only, no CUDA, no model import."""
    from vllm import ModelRegistry

    _check_version()

    # vLLM-internal: per-architecture config hooks are keyed by architecture name
    # (vllm/model_executor/models/config.py MODELS_CONFIG_MAP), so Plumb<Arch> inherits <Arch>'s hook.
    from vllm.model_executor.models import config as vllm_model_configs

    known = set(ModelRegistry.get_supported_archs())
    for arch in sorted(known):
        if arch.startswith(PLUMB_PREFIX):
            continue
        name = PLUMB_PREFIX + arch
        if name not in known:
            ModelRegistry.register_model(name, f"plumber.serving.vllm.model:{name}")
        if name not in vllm_model_configs.MODELS_CONFIG_MAP:
            vllm_model_configs.MODELS_CONFIG_MAP[name] = _config_hook(
                vllm_model_configs.MODELS_CONFIG_MAP.get(arch)
            )
    if any(
        m in sys.modules
        for m in ("vllm.entrypoints.cli.main", "vllm.entrypoints.openai.api_server")
    ):
        from .openai import install_route

        install_route()
