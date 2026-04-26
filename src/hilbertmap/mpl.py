"""Matplotlib integration: cmap, Normalize subclass, and a colorbar helper that paints only the data subset."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from numpy.typing import NDArray

from hilbertmap.cubewalk import walk
from hilbertmap.transform import depth_to_normalized, normalized_to_depth


def cmap(n: int = 1024, name: str = "hilbertmap") -> Colormap:
    """Build a matplotlib Colormap by sampling the cube walk at n positions in [0, 1]."""
    f = np.linspace(0.0, 1.0, n)
    rgb = walk(f)
    return LinearSegmentedColormap.from_list(name, rgb, N=n)


class Norm(Normalize):
    """Normalize subclass that maps metric depth to [0, 1] via the fixed power transform.

    vmin and vmax are stored only so the colorbar knows the data range to label;
    they do not rescale the mapping. As a consequence, the colorbar shows the
    cmap subset that the data actually covers, not the full black-to-white path.
    """

    def __init__(
        self,
        lam: float = -3.0,
        c: float = 10.0 / 3.0,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        super().__init__(vmin=vmin, vmax=vmax)
        self.lam = lam
        self.c = c

    def __call__(
        self,
        value: NDArray[np.floating],
        clip: bool | None = None,
    ) -> NDArray[np.floating]:
        self.autoscale_None(value)
        return depth_to_normalized(np.asarray(value), lam=self.lam, c=self.c)

    def inverse(self, value: NDArray[np.floating]) -> NDArray[np.floating]:
        # Clamp the inverse to [vmin, vmax] so the colorbar y-axis auto-restricts to the data range.
        fmin = (
            depth_to_normalized(np.asarray(self.vmin), lam=self.lam, c=self.c)
            if self.vmin is not None
            else 0.0
        )
        fmax = (
            depth_to_normalized(np.asarray(self.vmax), lam=self.lam, c=self.c)
            if self.vmax is not None
            else 1.0 - 1e-9
        )
        v = np.clip(np.asarray(value), float(fmin), float(fmax))
        return normalized_to_depth(v, lam=self.lam, c=self.c)


def colorbar(mappable: ScalarMappable, n: int = 1024, **kwargs) -> Colorbar:
    """Build a colorbar that paints only the cmap subset the data covers."""
    norm = mappable.norm
    if norm.vmin is None or norm.vmax is None:
        raise ValueError("mappable.norm must have vmin and vmax set before colorbar")
    boundaries = np.linspace(norm.vmin, norm.vmax, n)
    fig = mappable.axes.figure
    return fig.colorbar(mappable, boundaries=boundaries, **kwargs)


__all__ = ["cmap", "Norm", "colorbar", "mcolors"]
