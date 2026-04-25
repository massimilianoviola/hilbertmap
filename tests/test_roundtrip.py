"""Headline test: depth_to_rgb -> rgb_to_depth must roundtrip with tight error."""

import numpy as np
import pytest

from hilbertmap.cubewalk import depth_to_rgb, rgb_to_depth


def test_roundtrip_dense_range():
    d = np.linspace(0.0, 100.0, 10_000)
    rec = rgb_to_depth(depth_to_rgb(d))
    np.testing.assert_allclose(rec, d, rtol=1e-6, atol=1e-6)


def test_roundtrip_image_shape():
    rng = np.random.default_rng(0)
    d = rng.uniform(0.0, 50.0, size=(64, 64))
    rec = rgb_to_depth(depth_to_rgb(d))
    np.testing.assert_allclose(rec, d, rtol=1e-6, atol=1e-6)


def test_roundtrip_far_distances():
    d = np.array([100.0, 500.0, 1000.0, 1e4])
    rec = rgb_to_depth(depth_to_rgb(d))
    np.testing.assert_allclose(rec, d, rtol=1e-3)


def test_rgb_in_unit_cube():
    d = np.linspace(0.0, 1000.0, 10_000)
    rgb = depth_to_rgb(d)
    assert np.all(rgb >= 0.0)
    assert np.all(rgb <= 1.0)


@pytest.mark.parametrize("lam, c", [(-3.0, 10.0 / 3.0), (-2.0, 1.0), (-5.0, 2.0)])
def test_roundtrip_custom_params(lam, c):
    d = np.linspace(0.0, 50.0, 1_000)
    rec = rgb_to_depth(depth_to_rgb(d, lam=lam, c=c), lam=lam, c=c)
    np.testing.assert_allclose(rec, d, rtol=1e-6, atol=1e-6)
