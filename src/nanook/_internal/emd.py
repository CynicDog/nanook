from __future__ import annotations

import numpy as np


def emd_ordinal(p: np.ndarray, q: np.ndarray, support: np.ndarray) -> float:
    """Wasserstein-1 distance between two discrete distributions on an ordinal support.

    Uses the closed-form CDF difference: ``Σ |F_p(v) - F_q(v)| · Δv``. ``p`` and ``q``
    must be aligned with ``support`` (ascending) and sum to 1.
    """
    if support.size <= 1:
        return 0.0
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    deltas = np.diff(support)
    return float(np.sum(np.abs(cdf_p[:-1] - cdf_q[:-1]) * deltas))


def emd_nominal(p: np.ndarray, q: np.ndarray, cost: np.ndarray) -> float:
    """Wasserstein-1 distance between two discrete distributions on a nominal support.

    Solves the transportation LP with the supplied symmetric non-negative ``cost``
    matrix. Falls back to scipy.optimize.linprog; raises ImportError if scipy is
    not installed and the caller cannot satisfy the nominal-emd extra.
    """
    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "nominal EMD requires scipy; install nanook[nominal-emd] or use ordinal support"
        ) from exc

    n = p.size
    c = cost.reshape(-1)
    # row sums = p, column sums = q
    a_eq = np.zeros((2 * n, n * n))
    for i in range(n):
        a_eq[i, i * n : (i + 1) * n] = 1.0
    for j in range(n):
        a_eq[n + j, j::n] = 1.0
    b_eq = np.concatenate([p, q])
    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"nominal EMD LP failed: {res.message}")
    return float(res.fun)
