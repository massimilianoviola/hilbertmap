"""Vision Banana power transform: f(d, lam, c) = 1 - (1 - d / (lam * c)) ** (lam + 1).
Strictly increasing and invertible on d in [0, inf), maps to [0, 1).
"""

import numpy as np
from numpy.typing import NDArray


def depth_to_normalized(
    depth: NDArray[np.floating],
    lam: float = -3.0,
    c: float = 10.0 / 3.0,
) -> NDArray[np.floating]:
    """Apply the forward power transform: metric depth in [0, inf) to scalar in [0, 1)."""
    return 1.0 - (1.0 - depth / (lam * c)) ** (lam + 1.0)


def normalized_to_depth(
    f: NDArray[np.floating],
    lam: float = -3.0,
    c: float = 10.0 / 3.0,
) -> NDArray[np.floating]:
    """Apply the inverse power transform: scalar in [0, 1) back to metric depth in [0, inf)."""
    return lam * c * (1.0 - (1.0 - f) ** (1.0 / (lam + 1.0)))
