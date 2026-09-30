"""Training rows on disk and length-bucketed batches."""

from __future__ import annotations

from ..core.row import Row


def load_rows(path: str, limit: int | None = None) -> list[Row]:
    """Rows from a JSONL file, one decision per line (see ``plumber.core.row.Row``)."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(Row.from_json(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def bucketed_batches(order, lengths, tokens_per_batch, max_rows, max_len, rng, chunk=1024):
    """Shuffle -> local sort by length in chunks -> greedy fill up to tokens_per_batch (padded) or max_rows -> shuffle batches."""
    batches = []
    for c0 in range(0, len(order), chunk):
        part = sorted(order[c0 : c0 + chunk], key=lambda i: lengths[i])
        cur = []
        for i in part:
            if lengths[i] > max_len:
                continue
            L = max([lengths[j] for j in cur] + [lengths[i]])
            if cur and (L * (len(cur) + 1) > tokens_per_batch or len(cur) >= max_rows):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches
