import numpy as np
import pytest
import trimesh

from figurine import sdf
from figurine.meshsdf import GridSDF, fit_height, mesh_to_grid
from figurine.mesher import to_mesh


def test_closed_sphere_volume():
    m = trimesh.creation.icosphere(subdivisions=4, radius=5.0)
    field, grid = mesh_to_grid(m, 0.2)
    mesh, _ = to_mesh(field, grid)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(m.volume, rel=0.03)


def test_open_shell_gets_thickened_and_closed():
    # open-ended tube (a "skirt" made of a single surface) touching a closed sphere
    tube = trimesh.creation.cylinder(radius=4.0, height=6.0, sections=64)
    keep = np.abs(tube.face_normals[:, 2]) < 0.5  # drop the caps -> open single-sided shell
    tube = trimesh.Trimesh(tube.vertices, tube.faces[keep], process=False)
    tube.apply_translation([0, 0, -3])
    ball = trimesh.creation.icosphere(subdivisions=3, radius=4.5)
    m = trimesh.util.concatenate([tube, ball])
    field, grid = mesh_to_grid(m, 0.1, min_thickness=0.8)
    mesh, info = to_mesh(field, grid)
    assert mesh.is_watertight
    # the shell survives as a ~0.8 mm wall: its volume ~ 2*pi*r*h*t
    shell_vol = 2 * np.pi * 4.0 * 3.0 * 0.8  # lower half (below the ball) roughly
    assert mesh.volume > ball.volume + 0.5 * shell_vol


def test_gridsdf_node_matches_field():
    m = fit_height(trimesh.creation.icosphere(subdivisions=3, radius=1.0), 10.0)
    field, grid = mesh_to_grid(m, 0.25)
    node = GridSDF(field, grid)
    f2, g2 = sdf.evaluate(node, 0.25, pad=0.0)
    mesh, _ = to_mesh(f2, g2)
    assert mesh.is_watertight
    assert mesh.extents[2] == pytest.approx(10.0, abs=0.3)
