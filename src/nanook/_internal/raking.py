"""Iterative proportional fitting (raking) for MASSC's calibration step.

Adjusts a vector of design weights so that, for every calibration variable,
the weighted within-cell totals match the population totals. One outer pass
visits each calibration variable in turn; convergence is declared when no
within-cell scaling factor deviates from 1.0 by more than ``tol``.

Standard references: Deming & Stephan (1940); Deville & Särndal (1992) for
the calibration framework MASSC §4 invokes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl


def rake(
    frame: pl.DataFrame,
    weights: np.ndarray,
    targets: Sequence[Mapping[Any, float]],
    calibration_cols: Sequence[str],
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """Return weights calibrated against ``targets`` on ``calibration_cols``.

    ``frame`` is the (substituted, subsampled) frame the weights apply to.
    ``weights[i]`` is the initial design weight of row ``i``. ``targets[j]``
    maps each observed value of ``calibration_cols[j]`` to its required
    weighted total.

    Iterates until the maximum relative cell scaling factor sits within
    ``tol`` of 1.0 or ``max_iter`` outer passes elapse.
    """
    if len(targets) != len(calibration_cols):
        raise ValueError("raking: len(targets) must equal len(calibration_cols)")
    w = weights.astype(np.float64).copy()
    column_arrays = {c: frame.get_column(c).to_numpy() for c in calibration_cols}

    for _ in range(max_iter):
        max_change = 0.0
        for col, target in zip(calibration_cols, targets, strict=True):
            values = column_arrays[col]
            for v, t in target.items():
                mask = values == v
                cell_sum = float(w[mask].sum())
                if cell_sum == 0.0:
                    continue
                factor = float(t) / cell_sum
                max_change = max(max_change, abs(factor - 1.0))
                w[mask] *= factor
        if max_change < tol:
            break
    return w
