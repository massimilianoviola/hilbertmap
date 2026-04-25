"""Plot saturation curves of the power transform for different (lambda, c) parameters.

Run from the repo root:
    python examples/curves.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hilbertmap import cmap, depth_to_normalized

OUT = Path(__file__).parent / "curves.png"

PARAMS = [
    (-3.0, 1.0),
    (-3.0, 10.0 / 3.0),  # default (Vision Banana)
    (-3.0, 7.0),
    (-3.0, 15.0),
    (-3.0, 30.0),
]


def main() -> None:
    d = np.linspace(0.0, 200.0, 1000)
    fig, ax = plt.subplots(figsize=(7, 5))
    others = [p for p in PARAMS if p != (-3.0, 10.0 / 3.0)]
    grays = iter(np.linspace(0.0, 0.6, len(others)))
    styles = iter(["-", "--", "-.", ":", (0, (3, 1, 1, 1))])
    for lam, c in PARAMS:
        f = depth_to_normalized(d, lam=lam, c=c)
        label = f"λ={lam:g}, c={c:g}"
        if (lam, c) == (-3.0, 10.0 / 3.0):
            ax.plot(d, 100.0 * f, color="red", label=label + "  (default)", linewidth=2)
        else:
            ax.plot(
                d,
                100.0 * f,
                color=str(next(grays)),
                linestyle=next(styles),
                label=label,
            )
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("% of cube walk")
    ax.set_xlim(0, d.max())
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Slim cube-walk gradient strip pinned to the right of the y-axis as a visual aid.
    strip = ax.inset_axes([1.005, 0.0, 0.025, 1.0], transform=ax.transAxes)
    gradient = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    strip.imshow(
        gradient, aspect="auto", cmap=cmap(), origin="lower", extent=(0, 1, 0, 100)
    )
    strip.set_xticks([])
    strip.set_yticks([])
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
