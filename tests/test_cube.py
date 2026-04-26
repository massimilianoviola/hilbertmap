"""Tests for the RGB cube walk: vertex layout, segments, and corner mapping."""

import numpy as np

import hilbertmap as hm
from hilbertmap.cubewalk import N_SEGMENTS, VERTICES, project, walk


def test_walk_project_public_api():
    assert hm.walk is walk
    assert hm.project is project
    f = np.linspace(0.0, 1.0, 50)
    np.testing.assert_allclose(hm.project(hm.walk(f)), f, atol=1e-12)


def test_walk_supports_custom_normalization():
    # Linear normalization with vmin/vmax, no Barron transform
    depth = np.array([0.0, 5.0, 10.0])
    vmin, vmax = 0.0, 10.0
    f = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = hm.walk(f)
    back = vmin + (vmax - vmin) * hm.project(rgb)
    np.testing.assert_allclose(back, depth, atol=1e-12)


def test_vertex_count():
    assert VERTICES.shape == (N_SEGMENTS + 1, 3)


def test_seven_segments():
    assert VERTICES.shape[0] - 1 == N_SEGMENTS


def test_path_starts_black_ends_white():
    np.testing.assert_array_equal(VERTICES[0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(VERTICES[-1], [1.0, 1.0, 1.0])


def test_unit_edges():
    edges = np.diff(VERTICES, axis=0)
    lengths = np.linalg.norm(edges, axis=-1)
    np.testing.assert_allclose(lengths, 1.0)


def test_gray_code_one_change_per_step():
    diffs = np.abs(np.diff(VERTICES, axis=0))
    assert np.all(diffs.sum(axis=-1) == 1)


def test_walk_endpoints():
    np.testing.assert_array_equal(walk(np.array([0.0])), [[0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(walk(np.array([1.0])), [[1.0, 1.0, 1.0]])


def test_walk_hits_each_vertex():
    f = np.arange(VERTICES.shape[0], dtype=np.float64) / N_SEGMENTS
    out = walk(f)
    np.testing.assert_allclose(out, VERTICES)


def test_project_inverts_walk():
    f = np.linspace(0.0, 1.0, 1000)
    rgb = walk(f)
    rec = project(rgb)
    np.testing.assert_allclose(rec, f, atol=1e-12)


def test_project_off_edge_snaps_to_nearest():
    # Point near edge 0 (black -> red), slightly above the x-axis
    p = np.array([[0.5, 0.01, 0.0]])
    rec = project(p)
    assert 0.0 <= rec[0] <= 1.0 / N_SEGMENTS
