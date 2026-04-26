"""Hamiltonian walk on the RGB cube (first-iter 3D Hilbert curve).

Eight unit-cube vertices are visited in a Gray-code order from black (0, 0, 0)
to white (1, 1, 1), traversing seven unit edges. A scalar in [0, 1) indexes
monotonically along the polyline; the inverse projects an RGB point onto the
closest segment to recover the scalar.
"""

import numpy as np
from numpy.typing import NDArray

from hilbertmap.transform import depth_to_normalized, normalized_to_depth

VERTICES: NDArray[np.float64] = np.array(
    [
        [0.0, 0.0, 0.0],  # black
        [1.0, 0.0, 0.0],  # red
        [1.0, 1.0, 0.0],  # yellow
        [0.0, 1.0, 0.0],  # green
        [0.0, 1.0, 1.0],  # cyan
        [0.0, 0.0, 1.0],  # blue
        [1.0, 0.0, 1.0],  # magenta
        [1.0, 1.0, 1.0],  # white
    ],
    dtype=np.float64,
)
N_SEGMENTS: int = 7


def walk(f: NDArray[np.floating]) -> NDArray[np.floating]:
    """Sample RGB along the cube walk at scalar positions f in [0, 1]."""
    s = np.clip(f, 0.0, 1.0) * N_SEGMENTS
    # Split s=f*7 into segment index k and local position t
    k = np.minimum(np.floor(s).astype(np.int64), N_SEGMENTS - 1)
    t = s - k
    v0 = VERTICES[k]
    v1 = VERTICES[k + 1]
    return v0 + t[..., None] * (v1 - v0)


def project(rgb: NDArray[np.floating]) -> NDArray[np.floating]:
    """Project RGB samples onto the cube walk and return scalar positions in [0, 1]."""
    starts = VERTICES[:-1]
    ends = VERTICES[1:]
    edges = ends - starts
    edge_sq = np.sum(edges * edges, axis=-1)

    # Project each point against all 7 segments at once via a broadcast over a new axis
    p = rgb[..., None, :]
    t = np.clip(np.sum((p - starts) * edges, axis=-1) / edge_sq, 0.0, 1.0)
    proj = starts + t[..., None] * edges

    # Winning segment index k and its local position t recombine into a global scalar in [0, 1]
    k = np.argmin(np.sum((p - proj) ** 2, axis=-1), axis=-1)
    t_best = np.take_along_axis(t, k[..., None], axis=-1)[..., 0]
    return (k + t_best) / N_SEGMENTS


def depth_to_rgb(
    depth: NDArray[np.floating],
    lam: float = -3.0,
    c: float = 10.0 / 3.0,
) -> NDArray[np.floating]:
    """Encode metric depth as an RGB image along the cube walk."""
    f = depth_to_normalized(depth, lam=lam, c=c)
    return walk(f)


def rgb_to_depth(
    rgb: NDArray[np.floating],
    lam: float = -3.0,
    c: float = 10.0 / 3.0,
) -> NDArray[np.floating]:
    """Decode an RGB image back to metric depth by inverting the cube walk."""
    f = project(rgb)
    return normalized_to_depth(f, lam=lam, c=c)
