"""Any triangle mesh (STL/OBJ/GLB/VRM, even non-watertight game meshes) -> printable SDF grid.

Sign comes from the generalized winding number (robust to holes, self-intersections and
overlapping shells); magnitude from exact point-triangle distance. Thin open surfaces such as
hair cards or single-sided skirts are thickened to `min_thickness` so they print.
Distance queries run only in a narrow band around the surface (coarse grid elsewhere).
"""
from __future__ import annotations

import os
import shutil
import tempfile

import igl
import numpy as np
import open3d as o3d
import trimesh
from scipy import ndimage

from .sdf import BIG, F32, Grid, Node, _shape


def load_mesh(path):
    """Load any mesh file into one Trimesh (scene graphs flattened with their transforms)."""
    if path.lower().endswith(".vrm"):
        tmp = os.path.join(tempfile.mkdtemp(), "model.glb")
        shutil.copy(path, tmp)
        path = tmp
    m = trimesh.load(path, force="mesh", process=False)
    m.remove_unreferenced_vertices()
    return m


def y_up_to_z_up(m):
    """glTF/VRM/OBJ convention (Y up, facing +Z) -> ours (Z up, facing -Y)."""
    R = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    m = m.copy()
    m.apply_transform(R)
    return m


def fit_height(m, height, z0=0.0):
    """Uniformly scale to `height`, centre x/y on the bounding box, put the lowest point at z0."""
    m = m.copy()
    ext = m.bounds[1] - m.bounds[0]
    m.apply_scale(height / ext[2])
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, z0 - lo[2]])
    return m


class _Query:
    def __init__(self, mesh):
        self.V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        self.F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(self.V.astype(np.float32)),
                                 o3d.core.Tensor(self.F.astype(np.uint32)))

    def udist(self, P):
        out = np.empty(len(P), dtype=np.float32)
        for i in range(0, len(P), 2_000_000):
            q = o3d.core.Tensor(np.ascontiguousarray(P[i:i + 2_000_000], dtype=np.float32))
            out[i:i + 2_000_000] = self.scene.compute_distance(q).numpy()
        return out

    def winding(self, P):
        out = np.empty(len(P), dtype=np.float32)
        for i in range(0, len(P), 4_000_000):
            out[i:i + 4_000_000] = igl.fast_winding_number(self.V, self.F, np.ascontiguousarray(P[i:i + 4_000_000], dtype=np.float64))
        return out


def sheet_faces(mesh, q, probe=0.3):
    """Faces of open, two-sided sheets (hair cards, single-layer cloth).

    For a face on a closed surface the winding number is ~0 on one side and ~1 on the other;
    for a sheet it is ~0.5 in magnitude on both sides.
    """
    c, n = mesh.triangles_center, mesh.face_normals
    wa = np.abs(q.winding(c + probe * n))
    wb = np.abs(q.winding(c - probe * n))
    open_a = (wa > 0.2) & (wa < 0.8)
    open_b = (wb > 0.2) & (wb < 0.8)
    return open_a & open_b


def _signed(d, w, d_sheet, half_t):
    s = np.where(np.abs(w) > 0.75, -d, d)
    if d_sheet is not None:
        s = np.minimum(s, d_sheet - half_t)
    return s.astype(np.float32)


def mesh_to_grid(mesh, h, min_thickness=0.8, pad=2.0, coarse=4):
    """Signed distance grid of `mesh` (mm). Returns (field float32, Grid)."""
    lo = mesh.bounds[0] - pad
    hi = mesh.bounds[1] + pad
    grid = Grid(lo, hi, h)
    q = _Query(mesh)
    half_t = 0.5 * min_thickness
    sheets = sheet_faces(mesh, q)
    qs = None
    if sheets.any():
        sm = trimesh.Trimesh(mesh.vertices, mesh.faces[sheets], process=False)
        qs = _Query(sm)

    # coarse pass over everything
    hc = h * coarse
    cg = Grid(lo, hi + hc, hc)
    cx, cy, cz = (np.asarray(a, dtype=np.float64).ravel() for a in cg.axes(cg.full()))
    CP = np.stack(np.meshgrid(cx, cy, cz, indexing="ij"), -1).reshape(-1, 3)
    cd = q.udist(CP)
    csdf = _signed(cd, q.winding(CP), None if qs is None else qs.udist(CP), half_t).reshape(cg.shape)

    # upsample coarse field to the fine grid (trilinear)
    fy = np.arange(grid.shape[1]) / coarse
    fz = np.arange(grid.shape[2]) / coarse
    field = np.empty(grid.shape, dtype=F32)
    for i in range(grid.shape[0]):
        X, Y, Z = np.meshgrid([i / coarse], fy, fz, indexing="ij")
        field[i] = ndimage.map_coordinates(csdf, [X.ravel(), Y.ravel(), Z.ravel()], order=1,
                                           mode="nearest").reshape(grid.shape[1:])

    # exact values in the narrow band around the surface
    band = np.abs(field) < (1.8 * hc + half_t)
    idx = np.argwhere(band)
    P = grid.lo + idx * h
    field[band] = _signed(q.udist(P), q.winding(P), None if qs is None else qs.udist(P), half_t)
    return field, grid


class GridSDF(Node):
    """Node wrapping a precomputed SDF grid (trilinear interpolation)."""

    def __init__(self, field, grid, name="mesh"):
        self.field, self.grid, self.name = field, grid, name
        lo = grid.lo
        hi = grid.lo + (np.array(grid.shape) - 1) * grid.h
        self.box = (lo, hi)

    def eval(self, grid, reg):
        x, y, z = grid.axes(reg)
        g = self.grid
        out = np.empty(_shape(reg), dtype=F32)
        fy = (np.ravel(y) - g.lo[1]) / g.h
        fz = (np.ravel(z) - g.lo[2]) / g.h
        for k, xv in enumerate(np.ravel(x)):
            fx = (xv - g.lo[0]) / g.h
            X, Y, Z = np.meshgrid([fx], fy, fz, indexing="ij")
            out[k] = ndimage.map_coordinates(self.field, [X.ravel(), Y.ravel(), Z.ravel()], order=1,
                                             mode="constant", cval=float(BIG)).reshape(out.shape[1:])
        return out
