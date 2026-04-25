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


def test_roundtrip_with_8bit_quantization():
    # PNG-style 8-bit quantization moves pixels off the cube edges by up to 1/255.
    d = np.linspace(0.5, 30.0, 1_000)
    rgb = depth_to_rgb(d)
    rgb_q = np.round(rgb * 255.0) / 255.0
    rec = rgb_to_depth(rgb_q)
    np.testing.assert_allclose(rec, d, atol=0.1)


def test_decoder_handles_arbitrary_rgb():
    # Any point in the unit cube should map to a finite, non-negative depth.
    rng = np.random.default_rng(1)
    rgb = rng.uniform(0.0, 1.0, size=(500, 3))
    rec = rgb_to_depth(rgb)
    assert np.all(np.isfinite(rec))
    assert np.all(rec >= 0.0)
