"""The recipe builder: balanced per-family caps, disjoint dev rows, structured states rendered to text."""

import json

from plumber.data.recipe import build, family_of


def _rows(path, fam, n, state=None):
    with open(path, "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "id": f"{fam}-{i}",
                        "state": state or f"text {fam} {i}",
                        "question": "q?",
                        "qtype": "choice",
                        "options": [{"name": "a", "desc": ""}, {"name": "b", "desc": ""}],
                        "gold": i % 2,
                        "meta": {"family": fam},
                    }
                )
                + "\n"
            )


def test_family_key_fallbacks():
    assert family_of({"family": "f"}) == "f"
    assert family_of({"dataset": "glue/sst2"}) == "glue/sst2"
    assert family_of({"task_id": "finance/x/y"}) == "finance"


def test_build_caps_and_splits(tmp_path):
    a, b, h = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "hard.jsonl"
    _rows(a, "big", 100)
    _rows(b, "tiny", 2, state={"subject": "s", "body": "b"})
    _rows(h, "kev/judge", 500)
    m = build([str(a), str(b)], [str(h)], "small", str(tmp_path / "out"))
    assert m["families"]["big"] == {"train": 20, "dev": 3, "available": 100}
    assert m["families"]["tiny"]["train"] == 2
    assert m["skills"]["kev/judge"]["train"] == 100
    train = [json.loads(line) for line in open(tmp_path / "out" / "train.jsonl")]
    dev = [json.loads(line) for line in open(tmp_path / "out" / "dev.jsonl")]
    assert len(train) == m["train_rows"] == 122
    assert not ({r["id"] for r in train if r["meta"]["family"] == "big"} & {r["id"] for r in dev})
    assert next(r["state"] for r in train if r["meta"]["family"] == "tiny") == "subject: s\nbody: b"
