"""Plumber: inference engine for Plumb typed-decision models.

from plumber import Plumber
engine = Plumber("totum-labs/plumb-nemotron-3.5-lightning-30b-a3b")
engine.decide(state, questions)
"""

from .contract import question_to_row, rows_from_request, state_text
from .engine import DEFAULT_BASE, Plumber

__all__ = ["DEFAULT_BASE", "Plumber", "question_to_row", "rows_from_request", "state_text"]
__version__ = "0.1.0"
