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
                "Fast, calibrated decision about the conversation so far (routing, classification, yes/no "
                "checks, ratings). Returns the chosen option, probabilities, and the set of options that "
                "cannot be ruled out. Use it for routine judgements; reason yourself when the set has "
                "more than one option. When the conversation already lists the question and options, copy "
                "them exactly: the same question wording and every option name, unchanged."
            ),
            "parameters": {
                "type": "object",
                "required": ["question", "options"],
                "properties": {
                    "question": {"type": "string", "description": "the decision to make"},
                    "type": {
                        "type": "string",
                        "enum": ["choice", "noul", "score"],
                        "default": "choice",
                    },
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
