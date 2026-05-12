from __future__ import annotations

import numpy as np


def generator(seed: int | None) -> np.random.Generator:
    """Return a numpy Generator seeded by ``seed`` (None → entropy from the OS)."""
    return np.random.default_rng(seed)
