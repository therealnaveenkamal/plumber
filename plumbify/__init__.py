"""Plumbify: a fast decision path (System 1) for open language models, served natively by vLLM.

    plumbify train --base Qwen/Qwen3.5-9B --rows train.jsonl --dev dev.jsonl --out plumbed-qwen3.5-9b
    vllm serve plumbed-qwen3.5-9b

In Python, ``System1`` runs a plumb with transformers (training, evaluation, the reference implementation).
"""

__version__ = "0.3.0"


def __getattr__(name):  # System1 pulls in torch and transformers: load it on first use
    if name == "System1":
        from .system1 import System1

        return System1
    raise AttributeError(name)


__all__ = ["System1", "__version__"]
