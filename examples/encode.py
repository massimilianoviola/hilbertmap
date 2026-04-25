"""Encode every depth .npy in examples/data into an RGB PNG via depth_to_rgb.

Run from the repo root:
    python examples/encode.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hilbertmap import depth_to_rgb

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "depth_npy_meters"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main() -> None:
    for npy in sorted(DATA.glob("*.npy")):
        depth = np.load(npy)
        rgb = depth_to_rgb(depth)
        plt.imsave(OUT / f"{npy.stem}_hilbertmap.png", rgb)
        print(
            f"{npy.name} -> {npy.stem}_hilbertmap.png  ({depth.min():.2f}-{depth.max():.2f} m)"
        )


if __name__ == "__main__":
    main()
