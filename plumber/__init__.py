"""Plumber: inference engine for Plumb typed-decision models.

    from plumber import Plumber
    engine = Plumber("totum-labs/plumb-nemotron-3.5-lightning-30b-a3b")
    engine.decide(state, questions)
"""
from .engine import Plumber, DEFAULT_BASE, question_to_row, state_text

__all__ = ["Plumber", "DEFAULT_BASE", "question_to_row", "state_text"]
__version__ = "0.1.0"
