"""Tests for the matplotlib integration: cmap, Norm, colorbar helper."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, Normalize

from hilbertmap.mpl import Norm, cmap, colorbar


def test_cmap_is_colormap():
    assert isinstance(cmap(), Colormap)


def test_cmap_size():
    assert cmap(n=64).N == 64


def test_cmap_endpoints_are_black_and_white():
    cm = cmap()
    np.testing.assert_allclose(cm(0.0)[:3], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(cm(1.0)[:3], [1.0, 1.0, 1.0], atol=1e-6)


def test_norm_is_normalize():
    assert isinstance(Norm(), Normalize)


def test_norm_zero_to_zero():
    assert Norm()(np.array(0.0)) == 0.0


def test_norm_inverse_roundtrip():
    n = Norm()
    d = np.linspace(0.0, 100.0, 100)
    np.testing.assert_allclose(n.inverse(n(d)), d, rtol=1e-9, atol=1e-9)


def test_norm_custom_params():
    n = Norm(lam=-2.0, c=1.0)
    d = np.array([1.0, 2.0, 5.0])
    np.testing.assert_allclose(n.inverse(n(d)), d, rtol=1e-9, atol=1e-9)


def test_norm_autoscales_from_data():
    n = Norm()
    n(np.array([10.0, 20.0, 30.0]))  # triggers autoscale
    assert n.vmin == 10.0
    assert n.vmax == 30.0


def test_norm_explicit_vmin_vmax_kept():
    n = Norm(vmin=5.0, vmax=50.0)
    n(np.array([10.0, 20.0]))
    assert n.vmin == 5.0
    assert n.vmax == 50.0


def test_norm_inverse_clipped_to_data_range():
    n = Norm(vmin=2.0, vmax=20.0)
    # Even at the [0, 1] extremes the inverse must stay within [vmin, vmax].
    assert float(n.inverse(np.array(0.0))) == pytest.approx(2.0)
    assert float(n.inverse(np.array(1.0))) == pytest.approx(20.0)


def test_norm_fixed_mapping_independent_of_vmin_vmax():
    d = np.array([5.0, 10.0])
    a = Norm(vmin=0.0, vmax=100.0)(d)
    b = Norm(vmin=2.0, vmax=20.0)(d)
    np.testing.assert_allclose(a, b)


def test_colorbar_returns_colorbar():
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([[1.0, 5.0], [10.0, 20.0]]), cmap=cmap(), norm=Norm())
    cb = colorbar(im, ax=ax)
    assert isinstance(cb, Colorbar)
    plt.close(fig)


def test_colorbar_requires_vmin_vmax():
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((2, 2)), cmap=cmap(), norm=Norm())
    im.norm.vmin = None
    im.norm.vmax = None
    with pytest.raises(ValueError):
        colorbar(im, ax=ax)
    plt.close(fig)


def test_colorbar_ylim_matches_data_range():
    fig, ax = plt.subplots()
    d = np.array([[10.0, 20.0], [30.0, 40.0]])
    im = ax.imshow(d, cmap=cmap(), norm=Norm())
    cb = colorbar(im, ax=ax)
    lo, hi = cb.ax.get_ylim()
    assert lo == pytest.approx(10.0)
    assert hi == pytest.approx(40.0)
    plt.close(fig)
