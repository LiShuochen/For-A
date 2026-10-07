import numpy as np
import pytest

from figurine import sdf
from figurine.check import mesh_report, thin_voxels
from figurine.mesher import to_mesh


def _at(node, pts):
    """Evaluate a Func node at explicit points (N,3)."""
    p = np.asarray(pts, dtype=np.float32)
    return np.asarray(node.f(p[:, 0], p[:, 1], p[:, 2]), dtype=np.float64)


def test_sphere_distance():
    s = sdf.sphere((1, 2, 3), 2.0)
    assert _at(s, [[1, 2, 3], [5, 2, 3], [1, 2, 4]]) == pytest.approx([-2, 2, -1], abs=1e-5)


def test_round_cone_matches_brute_force():
    a, b, ra, rb = np.array([0, 0, 0.0]), np.array([0, 0, 10.0]), 3.0, 1.0
    rc = sdf.round_cone(a, b, ra, rb)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-8, 18, size=(400, 3))
    got = _at(rc, pts)
    # brute force: distance to union of many spheres along the axis (lower-bounds exact SDF closely)
    t = np.linspace(0, 1, 4001)
    centres = a[None] + t[:, None] * (b - a)[None]
    radii = ra + t * (rb - ra)
    d = np.sqrt(((pts[:, None, :] - centres[None]) ** 2).sum(-1)) - radii[None]
    want = d.min(1)
    assert np.abs(got - want).max() < 0.02


def test_capsule_constant_radius():
    c = sdf.capsule((0, 0, 0), (10, 0, 0), 1.0)
    assert _at(c, [[5, 0, 0], [5, 2, 0], [-2, 0, 0], [12, 0, 0]]) == pytest.approx([-1, 1, 1, 1], abs=1e-4)


def test_union_region_eval_matches_direct():
    u = sdf.Union([sdf.sphere((0, 0, 0), 3), sdf.sphere((5, 0, 0), 3)], k=1.0)
    field, grid = sdf.evaluate(u, 0.25)
    x, y, z = grid.axes(grid.full())
    a = np.sqrt(x ** 2 + y ** 2 + z ** 2) - 3
    b = np.sqrt((x - 5) ** 2 + y ** 2 + z ** 2) - 3
    direct = sdf.smin(np.broadcast_to(a, field.shape), np.broadcast_to(b, field.shape), 1.0)
    near = np.abs(direct) < 1.0
    assert np.abs(field[near] - direct[near]).max() < 1e-4


def test_subtract_and_intersect():
    a = sdf.box((0, 0, 0), (5, 5, 5))
    b = sdf.sphere((5, 0, 0), 3)
    field, grid = sdf.evaluate(a - b, 0.25)
    mesh, _ = to_mesh(field, grid)
    assert mesh.is_watertight
    expected = 1000 - 0.5 * 4 / 3 * np.pi * 27
    assert mesh.volume == pytest.approx(expected, rel=0.02)
    field2, grid2 = sdf.evaluate(a & b, 0.25)
    mesh2, _ = to_mesh(field2, grid2)
    assert mesh2.volume == pytest.approx(0.5 * 4 / 3 * np.pi * 27, rel=0.03)


def test_mesher_sphere_watertight_volume_and_islands():
    u = sdf.Union([sdf.sphere((0, 0, 0), 5), sdf.sphere((20, 0, 0), 1)])
    field, grid = sdf.evaluate(u, 0.2)
    mesh, info = to_mesh(field, grid)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume == pytest.approx(4 / 3 * np.pi * 125, rel=0.01)
    assert len(info["dropped_islands_mm3"]) == 1  # the small sphere is dropped
    rep = mesh_report(mesh)
    assert rep["bodies"] == 1 and rep["watertight"]


def test_mesher_keeps_islands_above_threshold():
    # a colour part like "both hands": two separate pieces plus a crumb
    u = sdf.Union([sdf.sphere((0, 0, 0), 3), sdf.sphere((10, 0, 0), 2), sdf.sphere((20, 0, 0), 0.4)])
    field, grid = sdf.evaluate(u, 0.1)
    mesh, info = to_mesh(field, grid, min_island_mm3=1.0)
    assert mesh.is_watertight
    assert mesh_report(mesh)["bodies"] == 2
    assert len(info["dropped_islands_mm3"]) == 1  # only the crumb


def test_decimation_keeps_watertight():
    field, grid = sdf.evaluate(sdf.sphere((0, 0, 0), 5), 0.1)
    mesh, info = to_mesh(field, grid, target_faces=5000)
    assert len(mesh.faces) <= 5200 and mesh.is_watertight
    assert mesh.volume == pytest.approx(4 / 3 * np.pi * 125, rel=0.01)


def test_thin_voxels_flags_thin_plate_only():
    plate = sdf.box((0, 0, 0), (5, 5, 0.2))  # 0.4 mm thick
    block = sdf.box((15, 0, 0), (3, 3, 3))
    field, grid = sdf.evaluate(sdf.Union([plate, block]), 0.1)
    thin, inside = thin_voxels(field, grid.h, 0.8)
    x, _, _ = grid.axes(grid.full())
    xs = np.broadcast_to(x, field.shape)
    assert thin[xs < 6].sum() > 0.9 * inside[xs < 6].sum()
    assert thin[xs > 10].sum() < 0.02 * inside[xs > 10].sum()


def test_slim_keeps_watertight_and_shape():
    from figurine.slim import slim

    field, grid = sdf.evaluate(sdf.sphere((0, 0, 0), 5), 0.1)
    mesh, _ = to_mesh(field, grid)
    out, info = slim(mesh, 4000)
    assert out.is_watertight and len(out.faces) < 0.2 * len(mesh.faces)
    assert info["dev_max_mm"] < 0.05
