"""Categorical MDAV clustering for MASSC's micro-agglomeration step.

Same structural loop as the continuous MDAV in
``core/perturbative/microaggregation.py`` (with the M2 second-seed fix) but
operating on integer-coded QI tuples under Hamming distance, with the cluster
"centroid" replaced by the per-column mode (Torra 2004 / Domingo-Ferrer &
Torra 2005). Used by ``MASSC`` to k-anonymise records by QI tuple before the
substitution step.
"""

from __future__ import annotations

import numpy as np


def assign(codes: np.ndarray, k: int) -> np.ndarray:
    """Assign every row in ``codes`` to a cluster index under Hamming MDAV.

    ``codes`` is an ``(n, p)`` integer matrix of factorised QI values. Each
    resulting cluster has size ``>= k`` (the final residual may pick up the
    remainder).
    """
    n = codes.shape[0]
    assignment = np.full(n, -1, dtype=np.int64)
    remaining = np.arange(n)
    cluster_id = 0

    while remaining.size >= 3 * k:
        centroid = _mode_per_column(codes[remaining])
        r_idx = _arg_max_distance(codes, remaining, centroid)
        remaining, cluster_id = _form_cluster_at(codes, remaining, k, r_idx, assignment, cluster_id)

        s_idx = _arg_max_distance(codes, remaining, codes[r_idx])
        remaining, cluster_id = _form_cluster_at(codes, remaining, k, s_idx, assignment, cluster_id)

    while remaining.size >= 2 * k:
        centroid = _mode_per_column(codes[remaining])
        r_idx = _arg_max_distance(codes, remaining, centroid)
        remaining, cluster_id = _form_cluster_at(codes, remaining, k, r_idx, assignment, cluster_id)

    if remaining.size:
        assignment[remaining] = cluster_id
    return assignment


def _arg_max_distance(codes: np.ndarray, remaining: np.ndarray, anchor: np.ndarray) -> int:
    """Return the index (in the *full* codes matrix) of the remaining row farthest from ``anchor``."""
    return int(remaining[int(np.argmax(_hamming(codes[remaining], anchor)))])


def _form_cluster_at(
    codes: np.ndarray,
    remaining: np.ndarray,
    k: int,
    seed_idx: int,
    assignment: np.ndarray,
    cluster_id: int,
) -> tuple[np.ndarray, int]:
    """Form a cluster from the ``k`` remaining records nearest to ``codes[seed_idx]``."""
    d_seed = _hamming(codes[remaining], codes[seed_idx])
    nearest = remaining[np.argsort(d_seed)[:k]]
    assignment[nearest] = cluster_id
    return _drop(remaining, nearest), cluster_id + 1


def _hamming(block: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (block != target).sum(axis=1)


def _mode_per_column(block: np.ndarray) -> np.ndarray:
    p = block.shape[1]
    out = np.empty(p, dtype=block.dtype)
    for j in range(p):
        vals, counts = np.unique(block[:, j], return_counts=True)
        out[j] = vals[int(np.argmax(counts))]
    return out


def _drop(remaining: np.ndarray, removed: np.ndarray) -> np.ndarray:
    removed_set = set(removed.tolist())
    return np.array([r for r in remaining if r not in removed_set], dtype=np.int64)
