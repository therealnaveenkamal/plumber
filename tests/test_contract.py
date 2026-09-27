"""The System One contract without weights: question parsing, training labels, confidence formulas, answer shapes."""

import pytest

from plumber.contract import (
    answer,
    choice_confidence,
    question_to_row,
    rows_from_request,
    score_confidence,
)

REQUEST = {
    "state": {"subject": "Charged twice", "body": "I see two charges for order #4411."},
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "Payments", "shipping": None},
            "label": "billing",
        },
        "angry": {"type": "noul", "instructions": "Is the customer angry?", "label": False},
        "priority": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["low", "normal", "high"],
            "label": 1,
        },
    },
}


def test_labelled_request_becomes_rows():
    rows = rows_from_request(REQUEST, prefix="7:")
    assert [r.id for r in rows] == ["7:team", "7:angry", "7:priority"]
    assert rows[0].state == "subject: Charged twice\nbody: I see two charges for order #4411."
    assert [o.name for o in rows[0].options] == ["billing", "shipping"] and rows[0].gold == 0
    assert [o.name for o in rows[1].options] == ["no", "yes"] and rows[1].gold == 0
    assert [o.desc for o in rows[2].options] == ["low", "normal", "high"] and rows[2].gold == 1


def test_bad_labels_fail_loudly():
    with pytest.raises(ValueError):
        question_to_row("q", {"type": "choice", "criteria": {"a": None}}, "s", label="b")
    with pytest.raises(ValueError):
        question_to_row("q", {"type": "score", "criteria": ["x", "y"]}, "s", label=2)
    with pytest.raises(ValueError):
        rows_from_request({"state": "s", "questions": {"q": {"type": "noul"}}})


def test_confidence_matches_reference_adapter():
    assert choice_confidence([0.8, 0.1, 0.1]) == pytest.approx(0.7)
    assert choice_confidence([1.0]) == 1.0
    assert score_confidence([0.0, 1.0, 0.0]) == pytest.approx(1.0)
    assert score_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert score_confidence([0.0, 0.56, 0.44]) == pytest.approx(1 - 0.44 / (2 / 3))


def test_answer_shapes():
    rows = rows_from_request(REQUEST)
    a = answer(rows[0], [0.9, 0.1])
    assert a["choice"] == "billing" and a["probabilities"] == {"billing": 0.9, "shipping": 0.1}
    a = answer(rows[1], [0.3, 0.7])
    assert a["noul"] == 0.7 and a["probabilities"] == {"no": 0.3, "yes": 0.7}
    a = answer(rows[2], [0.0, 0.56, 0.44])
    assert a["score"] == pytest.approx(1.44)
    assert a["legend"] == {"0": "low", "1": "normal", "2": "high"}
    assert a["confidence"] == pytest.approx(1 - 0.44 / (2 / 3), abs=1e-6)
