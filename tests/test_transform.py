"""Tests for the power transform."""

import numpy as np
import pytest

from hilbertmap.transform import depth_to_normalized, normalized_to_depth


def test_zero_maps_to_zero():
    assert depth_to_normalized(np.array([0.0]))[0] == 0.0


def test_infinity_approaches_one():
    out = depth_to_normalized(np.array([1e9]))
    assert 0.999 < out[0] < 1.0


def test_output_in_unit_interval():
    d = np.linspace(0.0, 1000.0, 10_000)
    out = depth_to_normalized(d)
    assert np.all(out >= 0.0)
    assert np.all(out < 1.0)


def test_strictly_increasing():
    d = np.linspace(0.0, 1000.0, 10_000)
    out = depth_to_normalized(d)
    assert np.all(np.diff(out) > 0)


def test_roundtrip():
    d = np.linspace(0.0, 1000.0, 10_000)
    rec = normalized_to_depth(depth_to_normalized(d))
    np.testing.assert_allclose(rec, d, rtol=1e-9, atol=1e-9)


def test_known_values():
    # f(d) = 1 - (1 + d/10)**(-2) with lam=-3, c=10/3 -> lam*c = -10, lam+1 = -2
    d = np.array([0.0, 10.0, 30.0])
    expected = 1.0 - (1.0 + d / 10.0) ** (-2.0)
    np.testing.assert_allclose(depth_to_normalized(d), expected, rtol=1e-12)


def test_custom_parameters():
    d = np.array([1.0, 2.0, 5.0])
    rec = normalized_to_depth(depth_to_normalized(d, lam=-2.0, c=1.0), lam=-2.0, c=1.0)
    np.testing.assert_allclose(rec, d, rtol=1e-9, atol=1e-9)


def test_preserves_shape():
    d = np.zeros((4, 5, 6))
    assert depth_to_normalized(d).shape == d.shape


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype(dtype):
    d = np.array([0.0, 1.0, 5.0], dtype=dtype)
    out = depth_to_normalized(d)
    assert out.dtype == dtype
