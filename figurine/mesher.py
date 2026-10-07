"""SDF field -> clean, watertight, decimated triangle mesh -> STL."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

from .sdf import BIG, Grid


def solid_mask(field):
    return field < 0


def keep_largest(field, h):
    """Drop floating islands (keeps the largest connected solid). Returns (field, dropped volumes mm^3)."""
    inside = field < 0
    lab, n = ndimage.label(inside)
    if n <= 1:
        return field, []
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes.argmax()
    dropped = sorted((float(s) * h ** 3 for i, s in enumerate(sizes) if i not in (0, keep)), reverse=True)
    field = field.copy()
    field[(lab != keep) & inside] = np.float32(h)
    return field, dropped


def to_mesh(field, grid: Grid, target_faces=None):
    """Marching cubes on the zero level set. Returns (trimesh, info dict)."""
    h = grid.h
    field, dropped = keep_largest(field, h)
    # crop to the solid's bounding box (+2 voxels) and pad with "outside" so the surface is closed
    idx = np.argwhere(field < 0)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = np.minimum(idx.max(0) + 3, field.shape)
    sub = field[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sub = np.pad(sub, 1, constant_values=BIG)
    # a tiny iso offset avoids exact zeros, whose degenerate triangles would otherwise tear holes
    verts, faces, _, _ = marching_cubes(sub, 1e-5, spacing=(h, h, h), allow_degenerate=False)
    verts += grid.lo + (lo - 1) * h
    mesh = trimesh.Trimesh(verts, faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    raw_faces = len(mesh.faces)
    if target_faces and len(mesh.faces) > target_faces:
        mesh = decimate(mesh, target_faces)
    return mesh, {"dropped_islands_mm3": dropped, "raw_faces": raw_faces}


def decimate(mesh, target_faces):
    """Quadric decimation; falls back to the input if the result is not watertight."""
    import fast_simplification

    red = 1.0 - target_faces / len(mesh.faces)
    v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int64),
                                        target_reduction=red, agg=5)
    out = trimesh.Trimesh(v, f, process=True)
    trimesh.repair.fix_normals(out)
    if out.is_watertight and out.is_winding_consistent and out.volume > 0:
        return out
    print("[mesher] decimated mesh not watertight; keeping full-resolution mesh")
    return mesh


def save_stl(mesh, path):
    mesh.export(path, file_type="stl")
