# 🎨 3D Hilbert Depth Colormap

Give your depth estimation a fancy new colormap! Here you'll find an implementation of a bijective metric depth $\leftrightarrow$ RGB mapping along a 3D Hilbert cube walk, as used in the Vision Banana 🍌 paper [1].

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/data/rgb/studio.png" width="240"/></td>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/data/rgb/living_room.png" width="240"/></td>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/data/rgb/street.png" width="240"/></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/outputs/studio_hilbertmap.png" width="240"/></td>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/outputs/living_room_hilbertmap.png" width="240"/></td>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/outputs/street_hilbertmap.png" width="240"/></td>
  </tr>
</table>

## 📦 Installation

From [PyPI](https://pypi.org/project/hilbertmap/):

```bash
pip install hilbertmap
```

From source:

```bash
git clone https://github.com/massimilianoviola/hilbertmap
cd hilbertmap
pip install -e .
```

## 🛠️ Usage

### Direct encoding/decoding

```python
import numpy as np
from hilbertmap import depth_to_rgb, rgb_to_depth

depth = np.load("depth.npy")          # (H, W) float meters
rgb   = depth_to_rgb(depth)           # (H, W, 3) float in [0, 1]
back  = rgb_to_depth(rgb)             # (H, W) recovered meters
```

Because the utility of accurate metric depth for nearby image content is generally higher than that of distant content, the default parameters $\lambda = -3$, $c = 10/3$ make the cube walk most sensitive in the first few meters and saturate beyond ~40 m. This behavior can be tuned by changing the parameters to get more meaningful color variation on deep outdoor scenes.

```python
rgb  = depth_to_rgb(depth, lam=-4.0, c=120.0)  # tuned for long-range outdoor scene
```

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/data/rgb/city.png" width="240"/></td>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/outputs/city_lam-4_c120.png" width="240"/></td>
  </tr>
</table>

### Visualization with matplotlib

```python
import matplotlib.pyplot as plt
import hilbertmap as hm

im = plt.imshow(depth, cmap=hm.cmap(), norm=hm.Norm())
hm.colorbar(im, label="depth (m)")
plt.show()
```

`hm.Norm` applies the fixed power transform (same depth $\to$ same color across images). With this, `hm.colorbar` spans only the cmap subset the data actually covers.

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/outputs/studio_plot.png" width="240"/></td>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/outputs/living_room_plot.png" width="240"/></td>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/outputs/street_plot.png" width="240"/></td>
  </tr>
</table>

In addition, transform params can be tuned as in direct encoding:

```python
im = plt.imshow(depth, cmap=hm.cmap(), norm=hm.Norm(lam=-4.0, c=120.0))  # global, long-range outdoor
hm.colorbar(im, label="depth (m)")
plt.show()
```

Note that passing `vmin` / `vmax` to `hm.Norm` does not rescale the mapping, only the displayed colorbar range:

```python
im = plt.imshow(depth, cmap=hm.cmap(), norm=hm.Norm(vmin=2.0, vmax=10.0))  # same global mapping, colorbar rescaled
hm.colorbar(im, label="depth (m)")  # <- this now shows [2, 10]
plt.show()
```

For per-image rescaling *without* the power transform, pair `hm.cmap()` with a standard matplotlib normalizer or simply omit it. This is the default behavior of other matplotlib colormaps.

Omit the normalizer to autoscale linearly to the data's min and max:

```python
im = plt.imshow(depth, cmap=hm.cmap())  # linear, autoscaled to min/max, covering full cmap from black to white
hm.colorbar(im, label="depth (m)")
plt.show()
```

Or pass `vmin` and `vmax` for a fixed range:

```python
im = plt.imshow(depth, cmap=hm.cmap(), vmin=0.0, vmax=80.0)               # linear, fixed range
# im = plt.imshow(depth, cmap=hm.cmap(), norm=plt.Normalize(0.0, 80.0))   # equivalent
hm.colorbar(im, label="depth (m)")
plt.show()
```

## 🧭 How it works

The seven-edge Hamiltonian path on the RGB cube (left) carries depth values from black at zero to white at infinity. The shape parameters $\lambda$ and $c$ produce different saturation curves (right) that decide how much depth lives on each segment of the walk.

<table>
  <tr>
    <td align="center"><b>Cube walk</b></td>
    <td align="center"><b>Saturation curves</b></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/cube_walk.png" alt="cube walk" height="268"/></td>
    <td><img src="https://raw.githubusercontent.com/massimilianoviola/hilbertmap/main/examples/curves.png" alt="saturation curves" height="268"/></td>
  </tr>
</table>

Unbounded metric depth $d \in [0, \infty)$ is squashed into $[0, 1)$ by a power transform from Barron (2025) [2], with $\lambda < -1$:

$$f(d, \lambda, c) = 1 - \left(1 - \frac{d}{\lambda c}\right)^{\lambda + 1}$$

With defaults $\lambda = -3$, $c = 10/3$ this simplifies to $f(d) = 1 - (1 + d/10)^{-2}$, mapping $d \in [0, \infty)$ to $f \in [0, 1)$, which is then read as the fractional position along the edge walk to land on $\mathrm{RGB} \in [0, 1]^3$. The mapping is a strict bijection, so any RGB encoding can be decoded back to metric depth by projecting onto the nearest edge.

## 📚 References

- V. Gabeur et al. *Vision Banana: Image generators are generalist vision learners*. arXiv preprint [arXiv:2604.20329](https://arxiv.org/abs/2604.20329), 2026. Project page: <https://vision-banana.github.io/>.
- J. T. Barron. *A power transform*. arXiv preprint [arXiv:2502.10647](https://arxiv.org/abs/2502.10647), 2025.
