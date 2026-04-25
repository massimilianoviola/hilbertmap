"""hilbertmap: bijective metric-depth <-> RGB colormap along a 3D Hilbert cube walk."""

from hilbertmap.cubewalk import depth_to_rgb, rgb_to_depth
from hilbertmap.mpl import Norm, cmap, colorbar
from hilbertmap.transform import depth_to_normalized, normalized_to_depth

__version__ = "0.1.0"

__all__ = [
    "depth_to_rgb",
    "rgb_to_depth",
    "depth_to_normalized",
    "normalized_to_depth",
    "cmap",
    "Norm",
    "colorbar",
    "__version__",
]
