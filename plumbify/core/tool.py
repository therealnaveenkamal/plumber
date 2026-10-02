"""``plumb_decide``: the tool a generating model calls to hand a decision to its plumb. Nothing external runs: the
call is the trigger for a decision on the model's own KV cache."""

from __future__ import annotations

from .row import Option, Row

PLUMB_TOOL = "plumb_decide"


def tool_schema() -> dict:
    """The System 1 tool a generative model can call instead of reasoning a routine decision out in tokens."""
    return {
        "type": "function",
        "function": {
            "name": PLUMB_TOOL,
            "description": (
                "Your decision module for routine judgements about this conversation. Returns the choice, "
                "probabilities, and `set` (the options it can't rule out): trust it when `set` has one "
                "option, else reason it through. Copy a question and options already in the conversation "
                "exactly."
            ),
            "parameters": {
                "type": "object",
                "required": ["question", "options"],
                "properties": {
                    "question": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["choice", "noul", "score"],
                    },  # default: choice
                    "options": {
                        "type": "array",
                        "minItems": 2,
                        "items": {
                            "type": "object",
                            "required": ["name"],
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    }


def row_from_tool_args(args: dict) -> Row:
    qtype = args.get("type") or "choice"
    opts = [Option(str(o["name"]), str(o.get("description") or "")) for o in args["options"]]
    if qtype == "noul":
        by = {o.name.lower(): o for o in opts}
        opts = [
            Option("no", getattr(by.get("no"), "desc", "")),
            Option("yes", getattr(by.get("yes"), "desc", "")),
        ]
    return Row("tool", "", str(args["question"]), qtype, opts)
