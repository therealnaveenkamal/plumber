"""Metric definitions and the eval report shape, on synthetic predictions."""

import json

import pytest

from plumber.metrics import brier, coverage_at_error, ece, summarize
from plumber.training.eval import _stats
from plumber.training.train import load_rows

PREDS = [
    {"correct": 1, "p_max": 0.9, "p_gold": 0.9, "probs": [0.9, 0.1], "gold": 0},
    {"correct": 0, "p_max": 0.6, "p_gold": 0.4, "probs": [0.6, 0.4], "gold": 1},
    {"correct": 1, "p_max": 0.7, "p_gold": 0.7, "probs": [0.7, 0.2, 0.1], "gold": 0},
    {"correct": None, "p_max": 0.5, "p_gold": None, "probs": [0.5, 0.5], "gold": None},
]


def test_brier_and_coverage():
    assert brier([[0.9, 0.1]], [0]) == pytest.approx(0.01 + 0.01)
    assert brier([[1.0, 0.0]], [1]) == pytest.approx(2.0)
    # sorted by confidence: 0.9 (right), 0.7 (right), 0.6 (wrong) -> 2 of 3 automatable at <=5% error
    assert coverage_at_error([0.9, 0.6, 0.7], [1, 0, 1], 0.05) == pytest.approx(2 / 3)
    assert 0.0 <= ece([0.9, 0.6, 0.7], [1, 0, 1]) <= 1.0


def test_eval_report_keys():
    m = _stats(PREDS)
    assert m["n"] == 3 and m["acc"] == m["micro_acc"] == pytest.approx(2 / 3)
    assert set(m) >= {"nll", "brier", "ece", "coverage_at_5pct_error"}


def test_load_rows_accepts_labelled_requests(tmp_path):
    req = {
        "state": "Two charges for one order.",
        "questions": {
            "team": {
                "type": "choice",
                "criteria": {"billing": None, "shipping": None},
                "label": "billing",
            },
            "angry": {"type": "noul", "label": True},
        },
    }
    f = tmp_path / "rows.jsonl"
    f.write_text(json.dumps(req) + "\n" + json.dumps(req) + "\n")
    rows = load_rows(f)
    assert [r.id for r in rows] == ["0:team", "0:angry", "1:team", "1:angry"]
    assert rows[1].gold == 1 and rows[0].gold == 0
    assert len(load_rows(f, limit=3)) == 3


def test_temperature_fit_fixes_overconfidence():
    import random

    from plumber.training.calibrate import fit_temperature, rescale

    rng = random.Random(0)
    rows = []
    for _ in range(400):
        gold = rng.randrange(3)
        # a head that is right 70% of the time but says 0.95: overconfident -> T > 1 should help
        pred = gold if rng.random() < 0.7 else (gold + 1) % 3
        probs = [0.025] * 3
        probs[pred] = 0.95
        rows.append(
            {
                "gold": gold,
                "probs": probs,
                "p_max": 0.95,
                "p_gold": probs[gold],
                "correct": int(pred == gold),
            }
        )
    t = fit_temperature(rows)
    assert t > 1.5
    before, after = summarize(rescale(rows, 1.0)), summarize(rescale(rows, t))
    assert after["ece"] < before["ece"] and after["nll"] < before["nll"]
    assert after["accuracy"] == before["accuracy"]
