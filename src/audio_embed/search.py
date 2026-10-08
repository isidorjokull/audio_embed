"""Ranking files against a query vector."""

from dataclasses import dataclass

import numpy as np

from .store import Matrix


@dataclass
class Hit:
    path: str
    score: float
    start_s: float  # where in the file the best-matching window begins
    index: int  # position of the file in the matrix's `paths`
    tags: list[str]


def rank(
    matrix: Matrix,
    query: np.ndarray,
    k: int,
    exclude: str | None = None,
    keep: np.ndarray | None = None,
) -> list[Hit]:
    """Top `k` files by their best-matching window.

    Scoring a file by its best window, not its average, lets a long file match
    on one relevant passage and tells us where that passage is. `keep` is an
    optional true/false array over the files: only files marked true are returned.
    """
    if len(matrix.paths) == 0:
        return []
    scores = matrix.vecs @ query
    best = np.full(len(matrix.paths), -np.inf)
    np.maximum.at(best, matrix.file_idx, scores)
    # The first row (in window order) that reaches each file's best score.
    is_best = scores == best[matrix.file_idx]
    best_row = np.full(len(matrix.paths), -1)
    rows = np.flatnonzero(is_best)[::-1]
    best_row[matrix.file_idx[rows]] = rows

    if keep is not None:
        best = np.where(keep, best, -np.inf)
    hits = []
    for f in np.argsort(-best):
        if len(hits) == k or best[f] == -np.inf:
            break
        if matrix.paths[f] != exclude:
            hits.append(
                Hit(matrix.paths[f], float(best[f]), float(matrix.starts[best_row[f]]), int(f), matrix.tags[f])
            )
    return hits


def file_vector(matrix: Matrix, path: str) -> np.ndarray | None:
    """The unit-length mean of a file's window vectors, or None if it is not indexed."""
    if path not in matrix.paths:
        return None
    return unit_mean(matrix.vecs[matrix.file_idx == matrix.paths.index(path)])


def closeness(query: np.ndarray, others: np.ndarray, floor: float) -> np.ndarray:
    """How much the votes on other searches count toward this one, from 0 to 1.

    `others` are the query vectors of those searches. One that is no more like
    this search than `floor` counts for nothing; one worded the same counts fully.
    """
    return np.clip((others @ query - floor) / (1 - floor), 0.0, 1.0)


def steer(
    query: np.ndarray,
    liked: np.ndarray,
    rejected: np.ndarray,
    weight: float,
    liked_counts: np.ndarray | None = None,
    rejected_counts: np.ndarray | None = None,
) -> np.ndarray:
    """Pull a query toward the files the user liked and away from the ones they rejected.

    `liked` and `rejected` are the vectors of those files (either may be empty).
    `weight` is how hard the votes pull compared with the original query.
    The counts say how much each vote counts, 1 when not given: a vote borrowed
    from a similar search counts for less than one on this search, and the pull
    as a whole is only as hard as the vote that counts most. Counts must be above 0.
    """
    shift = np.zeros_like(query)
    strongest = 0.0
    for sign, vecs, counts in ((1, liked, liked_counts), (-1, rejected, rejected_counts)):
        if len(vecs):
            counts = np.ones(len(vecs)) if counts is None else counts
            shift += sign * np.average(vecs, axis=0, weights=counts)
            strongest = max(strongest, float(counts.max()))
    steered = query + weight * strongest * shift
    return steered / np.linalg.norm(steered)


def file_means(matrix: Matrix) -> np.ndarray:
    """One unit-length vector per file: the mean of its window vectors.

    Relies on each file's windows being stored together, which is how `Store.load` returns them.
    """
    first_rows = np.flatnonzero(np.diff(matrix.file_idx, prepend=-1))
    sums = np.add.reduceat(matrix.vecs, first_rows, axis=0)
    return sums / np.linalg.norm(sums, axis=1, keepdims=True)


def unit_mean(vecs: np.ndarray) -> np.ndarray:
    mean = vecs.mean(axis=0)
    return mean / np.linalg.norm(mean)
