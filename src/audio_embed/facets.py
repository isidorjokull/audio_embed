"""Filter groups for the page.

A file can carry labels of several kinds (its sound category, and whatever
`Matrix.labels` holds: bpm, key, loop or one-shot, and so on). This turns the
per-file labels into something that can be counted and filtered on.
"""

import numpy as np

# Kinds with more distinct values than this are free text, not something to pick from.
MAX_DISTINCT = 200
# At most this many values are offered per kind, most common first.
MAX_OFFERED = 30
# The order filter groups are offered in; any other kind follows alphabetically.
KIND_ORDER = ["sound", "type", "category", "source", "project", "key", "bpm", "genre", "year", "artist"]
# Kinds whose values read better in their natural order than most-common-first.
IN_VALUE_ORDER = {"bpm", "year", "key"}
TEMPO_STEP = 20


def tempo_range(value: str) -> str | None:
    """The range a tempo falls in, e.g. "165.5" -> "160 to 179", so tempos can be picked as a few ranges."""
    try:
        low = int(float(value) // TEMPO_STEP) * TEMPO_STEP
    except ValueError:
        return None
    return f"{low} to {low + TEMPO_STEP - 1}"


def _natural(value: str):
    head = value.split()[0] if value.split() else ""
    return (0, float(head), value) if head.replace(".", "", 1).isdigit() else (1, 0.0, value)


def build(labels_per_file: list[dict[str, list[str]]]) -> dict[str, dict[str, np.ndarray]]:
    """kind -> value -> positions of the files that have that value."""
    found: dict[str, dict[str, list[int]]] = {}
    for position, labels in enumerate(labels_per_file):
        for kind, values in labels.items():
            for value in values:
                found.setdefault(kind, {}).setdefault(value, []).append(position)
    return {
        kind: {value: np.array(positions) for value, positions in values.items()}
        for kind, values in found.items()
    }


def mask(index: dict[str, dict[str, np.ndarray]], n_files: int, picks: dict[str, set[str]]) -> np.ndarray:
    """True for each file that has, for every picked kind, at least one of the picked values."""
    keep = np.ones(n_files, dtype=bool)
    for kind, values in picks.items():
        has_one = np.zeros(n_files, dtype=bool)
        for value in values:
            has_one[index.get(kind, {}).get(value, [])] = True
        keep &= has_one
    return keep


def summary(index: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    """What to offer as filters: each kind with its most common values and their file counts."""
    out = []
    order = {kind: n for n, kind in enumerate(KIND_ORDER)}
    for kind in sorted(index, key=lambda k: (order.get(k, len(order)), k)):
        values = index[kind]
        if len(values) > MAX_DISTINCT:
            continue
        common = sorted(values.items(), key=lambda item: (-len(item[1]), item[0]))[:MAX_OFFERED]
        if kind in IN_VALUE_ORDER:
            common.sort(key=lambda item: _natural(item[0]))
        out.append({"kind": kind, "values": [{"value": v, "count": len(p)} for v, p in common]})
    return out
