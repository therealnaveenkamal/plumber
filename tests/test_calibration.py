"""Metrics, temperature and conformal calibration, the training loss, and loading rows."""

import random

import pytest
import torch
import torch.nn.functional as F

from plumber.calibration import coverage, fit_conformal, fit_temperature, rescale
from plumber.core.decision_head import decision_loss
from plumber.core.row import Option, Row
from plumber.metrics import brier, coverage_at_error, ece, summarize
from plumber.training.data import load_rows


def overconfident(n=400, seed=0):
    """A head that is right 70% of the time but always says 0.95."""
    rng, rows = random.Random(seed), []
    for _ in range(n):
        gold = rng.randrange(3)
        pred = gold if rng.random() < 0.7 else (gold + 1) % 3
        probs = [0.025] * 3
        probs[pred] = 0.95
        rows.append({"gold": gold, "probs": probs})
    return rows


def scored(rows, t=1.0):
    out = []
    for r in rows:
        p = rescale(r["probs"], t)
        top = max(range(len(p)), key=p.__getitem__)
        out.append(
            {
                **r,
                "probs": p,
                "p_max": p[top],
                "p_gold": p[r["gold"]],
                "correct": int(top == r["gold"]),
            }
        )
    return out


def test_metric_definitions():
    assert brier([[0.9, 0.1]], [0]) == pytest.approx(0.02)
    assert brier([[1.0, 0.0]], [1]) == pytest.approx(2.0)
    # by confidence: 0.9 right, 0.7 right, 0.6 wrong -> 2 of 3 automatable at <= 5% error
    assert coverage_at_error([0.9, 0.6, 0.7], [1, 0, 1], 0.05) == pytest.approx(2 / 3)
    assert 0.0 <= ece([0.9, 0.6, 0.7], [1, 0, 1]) <= 1.0


def test_temperature_fixes_overconfidence_without_changing_answers():
    rows = overconfident()
    t = fit_temperature(rows)
    assert t > 1.5
    before, after = summarize(scored(rows)), summarize(scored(rows, t))
    assert after["ece"] < before["ece"] and after["nll"] < before["nll"]
    assert after["accuracy"] == before["accuracy"]


def noisy(n=1000, seed=0):
    """Varied, tie-free probabilities: right about 70% of the time."""
    rng, rows = random.Random(seed), []
    for _ in range(n):
        gold = rng.randrange(4)
        pred = gold if rng.random() < 0.7 else rng.randrange(4)
        w = [rng.random() for _ in range(4)]
        w[pred] += 2.0
        rows.append({"gold": gold, "probs": [x / sum(w) for x in w]})
    return rows


def test_conformal_sets_cover_held_out_rows():
    conf = fit_conformal(noisy(seed=1), alpha=0.1)
    cov = coverage(noisy(seed=2), conf["qhat"])
    assert cov["coverage"] >= 0.87 and cov["mean_set_size"] < 4


def reference_loss(logits, gold, teacher, lam=0.5):
    """Per-row loop: CE on gold, plus lam * KL(teacher || p)."""
    logp = F.log_softmax(logits, -1)
    total, n = logits.new_zeros(()), 0
    for b in range(logits.shape[0]):
        if gold[b] is not None:
            total, n = total - logp[b, gold[b]], n + 1
        if teacher[b] is not None:
            t = logits.new_tensor(teacher[b])
            t = t / t.sum().clamp_min(1e-9)
            k = t.shape[0]
            total, n = total + lam * (t * (t.clamp_min(1e-9).log() - logp[b, :k])).sum(), n + 1
    return total / max(n, 1)


def test_decision_loss_matches_reference():
    torch.manual_seed(0)
    K = [2, 3, 4, 3]
    logits = torch.full((4, 4), float("-inf"))
    for b, k in enumerate(K):
        logits[b, :k] = torch.randn(k)
    logits.requires_grad_(True)
    gold = [0, None, 3, 1]
    teacher = [None, [0.2, 0.2, 0.6], [0.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.0]]
    loss = decision_loss(logits, gold, teacher)
    torch.testing.assert_close(loss, reference_loss(logits, gold, teacher))
    loss.backward()
    assert torch.isfinite(logits.grad[logits.grad != 0]).all()


def test_load_rows(tmp_path):
    rows = [
        Row(
            f"r{i}",
            "state",
            "Which team?",
            "choice",
            [Option("billing"), Option("sales")],
            gold=i % 2,
        )
        for i in range(3)
    ]
    f = tmp_path / "rows.jsonl"
    f.write_text("\n".join(r.to_json() for r in rows) + "\n\n")
    assert [r.id for r in load_rows(str(f))] == ["r0", "r1", "r2"]
    assert len(load_rows(str(f), limit=2)) == 2
