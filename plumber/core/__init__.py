"""Implementation behind the Plumber engine: rendering, heads, trunk. Internal seam; training and tests use it directly."""

from .heads import PointerHead, decision_loss
from .rendering import QTYPES, Option, Row, render
from .trunk import LORA_TARGETS, PlumbModel, collate, pin_mamba_devices

__all__ = [
    "LORA_TARGETS",
    "QTYPES",
    "Option",
    "PlumbModel",
    "PointerHead",
    "Row",
    "collate",
    "decision_loss",
    "pin_mamba_devices",
    "render",
]
