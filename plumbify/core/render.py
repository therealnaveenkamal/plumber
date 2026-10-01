"""Render v2: a decision is a user turn in the model's own chat format, appended after the context.

    <context: a conversation, or a state>  <user turn: question, options, instruction>  <assistant-open>
    '------------ prefix (cacheable) -----''---------------- decision suffix ---------------------------'

The frozen trunk reads the question in the format it was trained on, and the prefix is exactly what a generator has
already cached for the same conversation, so a decision only prefills its suffix. The head reads the suffix only:
option spans, and DECIDE = the last token of the assistant-open (the position where the model would start answering).

Pieces are tokenised separately (prefix, question, each option, closing) so option spans are exact; training and
serving both go through ``render_decision``, so the token ids always agree.
"""

from __future__ import annotations

from dataclasses import dataclass

from .row import Row

SENTINEL = "\x00PLUMB_DECISION\x00"
LEADS = {
    "choice": ("Options:", "Answer with exactly one option."),
    "noul": ("Options:", "Answer yes or no."),
    "score": ("Levels, lowest to highest:", "Answer with exactly one level."),
}


@dataclass
class Rendered:
    input_ids: list[int]
    suffix_start: int  # first token of the decision suffix; everything before is shared context
    opt_spans: list[tuple[int, int]]  # absolute positions, presented order
    decide_pos: int
    perm: list[int]  # presented position -> original option index
    gold: int | None
    teacher: list[float] | None
    qtype: str
    id: str

    def suffix(self) -> dict:
        """Spans relative to the suffix, as the head consumes them."""
        s = self.suffix_start
        return {
            "opt_spans": [(a - s, b - s) for a, b in self.opt_spans],
            "decide": self.decide_pos - s,
        }


def _ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False).input_ids


def _template(tok, messages: list[dict], **kw) -> tuple[str, str]:
    """Chat-template text before and after the decision turn's content."""
    text = tok.apply_chat_template(
        [*messages, {"role": "user", "content": SENTINEL}],
        tokenize=False,
        add_generation_prompt=True,
        **kw,
    )
    head, tail = text.split(SENTINEL)
    return head, tail


def render_decision(
    tok,
    row: Row,
    context: list[dict] | None = None,
    perm: list[int] | None = None,
    template_kwargs: dict | None = None,
) -> Rendered:
    """Token ids + positions for one decision.

    ``context``: prior chat messages to decide over (the state is then ``row.state`` if non-empty, placed at the top
    of the decision turn). Without ``context`` the state opens the decision turn. ``perm`` presents options in a
    given order (training shuffles; never for score).
    """
    kw = {"enable_thinking": False, **(template_kwargs or {})}
    head, tail = _template(tok, list(context or []), **kw)
    K = len(row.options)
    perm = list(range(K)) if perm is None or row.qtype == "score" else perm
    lead, instruction = LEADS[row.qtype]
    from ..core.answers import state_text

    state = f"{state_text(row.state)}\n\n" if row.state else ""
    ids = _ids(tok, head + state)  # prefix: template head (+ state), tokenised as one run
    suffix_start = len(ids)
    ids += _ids(tok, f"Decision: {state_text(row.question)}\n{lead}\n")
    spans = []
    for j in perm:
        o = row.options[j]
        s = len(ids)
        ids += _ids(tok, f"- {o.name}: {o.desc}\n" if o.desc else f"- {o.name}\n")
        spans.append((s, len(ids)))
    ids += _ids(tok, instruction + tail)
    inv = {orig: pos for pos, orig in enumerate(perm)}
    return Rendered(
        ids,
        suffix_start,
        spans,
        len(ids) - 1,
        perm,
        None if row.gold is None else inv[row.gold],
        None if row.teacher is None else [row.teacher[j] for j in perm],
        row.qtype,
        row.id,
    )
