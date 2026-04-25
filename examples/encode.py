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
        if npy.stem != "city":
            rgb = depth_to_rgb(depth)
            plt.imsave(OUT / f"{npy.stem}_hilbertmap.png", rgb)
            print(
                f"{npy.name} -> {npy.stem}_hilbertmap.png  ({depth.min():.2f}-{depth.max():.2f} m)"
            )
        else:
            rgb = depth_to_rgb(depth, lam=-4.0, c=120.0)
            plt.imsave(OUT / "city_lam-4_c120.png", rgb)
            print(f"{npy.name} -> city_lam-4_c120.png")


if __name__ == "__main__":
    main()
