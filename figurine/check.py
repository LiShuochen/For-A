"""Printability checks for FDM (Bambu Lab A1 mini, 0.4 mm nozzle, 0.08 mm layers)."""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

A1_MINI = (180.0, 180.0, 180.0)
PLA_DENSITY = 1.24  # g/cm^3


def thin_voxels(field, h, min_thickness):
    """Solid voxels not covered by any ball of diameter `min_thickness` that fits inside the solid.

    Returns (thin mask, inside mask). Uses two Euclidean distance transforms (morphological opening).
    """
    inside = field < 0
    r = 0.5 * min_thickness
    depth = ndimage.distance_transform_edt(inside).astype(np.float32) * h
    centres = depth >= r
    if not centres.any():
        return inside, inside
    reach = ndimage.distance_transform_edt(~centres).astype(np.float32) * h
    return inside & (reach > r), inside


def mesh_report(mesh: trimesh.Trimesh, overhang_deg=45.0, bed_z=None):
    ext = mesh.extents
    bed_z = mesh.bounds[0, 2] if bed_z is None else bed_z
    n = mesh.face_normals
    c = mesh.triangles_center
    a = mesh.area_faces
    down = -n[:, 2] > np.cos(np.radians(overhang_deg))
    on_bed = c[:, 2] < bed_z + 0.05
    over = down & ~on_bed
    comps = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))
    vol_cm3 = mesh.volume / 1000.0
    return {
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "bodies": len(comps),
        "faces": int(len(mesh.faces)),
        "size_mm": [round(float(e), 2) for e in ext],
        "fits_a1_mini": bool(np.all(ext <= np.array(A1_MINI) - 2.0)),
        "volume_cm3": round(vol_cm3, 2),
        "solid_mass_g": round(vol_cm3 * PLA_DENSITY, 1),
        "overhang_area_mm2": round(float(a[over].sum()), 1),
        "overhang_pct": round(100.0 * float(a[over].sum() / a.sum()), 2),
        "bed_contact_mm2": round(float(a[on_bed & (n[:, 2] < -0.99)].sum()), 1),
    }
