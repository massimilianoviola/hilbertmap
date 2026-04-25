"""Visualize a depth map with matplotlib using the hilbertmap cmap and Norm.

Run from the repo root:
    python examples/plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import hilbertmap as hm

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "depth_npy_meters"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main() -> None:
    for npy in sorted(DATA.glob("*.npy")):
        depth = np.load(npy)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(depth, cmap=hm.cmap(), norm=hm.Norm())
        hm.colorbar(im, ax=ax, label="depth (m)")
        ax.set_title(npy.stem)
        plt.axis("off")

        out_path = OUT / f"{npy.stem}_plot.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"{npy.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
