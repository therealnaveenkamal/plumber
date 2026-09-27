"""Implementation behind the Plumber engine: rendering, heads, trunk. Internal seam; training and tests use it directly."""
from .rendering import Row, Option, render, QTYPES
from .heads import PointerHead, decision_loss
from .trunk import PlumbModel, LORA_TARGETS, pin_mamba_devices, collate
