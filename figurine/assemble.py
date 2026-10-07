"""Put his head (Head2) on a pretty VRoid body: cut the body at the neck, align, mesh, union."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from . import face as F
from . import sdf
from .head import Placement
from .head2 import Head2, Head2Params
from .mesher import to_mesh
from .meshsdf import GridSDF, mesh_to_grid
from .sdf import rot
from .vrm import VRM
from .vrmfig import figure_parts, relaxed_pose

HEAD_DROP = ("Face", "FaceMouth", "EyeWhite")


@dataclass
class Attach:
    z_cut: float  # world z where the body's neck is cut
    neck_xy: np.ndarray  # neck axis (x, y) at the cut
    neck_r: float  # neck radius at the cut
    chin: np.ndarray  # world position of the original chin
    head_h: float  # original chin-to-crown height (mm)


def body(path, k_mm_per_m, base_h=3.0, pose=None, subdivide=1):
    """Posed body parts (no head, no hair) at k mm per model metre, and where the head goes."""
    v = VRM(path)
    pose = relaxed_pose(v) if pose is None else pose
    groups, info = figure_parts(path, 1000.0, z0=base_h, pose=pose, keep_hair=False, subdivide=subdivide,
                                extra_drop=HEAD_DROP)
    # figure_parts scales to height 1000 mm; rescale to k mm per model metre
    full_h = info["scale"]  # mm per model unit at height 1000
    f = k_mm_per_m / full_h
    for g in groups.values():
        g.vertices[:, 2] = base_h + (g.vertices[:, 2] - base_h) * f
        g.vertices[:, :2] *= f
    tf = lambda V: np.c_[info["tf"](V)[:, :2] * f, base_h + (info["tf"](V)[:, 2] - base_h) * f]
    neck = tf(v.bone_position("neck", pose)[None])[0]
    head = tf(v.bone_position("head", pose)[None])[0]
    z_cut = neck[2] + 0.55 * (head[2] - neck[2])
    skin = groups["skin"].vertices
    ring = skin[np.abs(skin[:, 2] - z_cut) < 0.6]
    neck_xy = ring[:, :2].mean(0)
    neck_r = float(np.median(np.linalg.norm(ring[:, :2] - neck_xy, axis=1)))
    faces = [tf(p.vertices) for p in v.parts(pose) if "Face_00_SKIN" in p.name]
    fv = np.concatenate(faces)
    front = fv[np.abs(fv[:, 0]) < 0.02 * k_mm_per_m]
    chin = front[front[:, 2].argmin()]
    head_h = fv[:, 2].max() - chin[2]
    return groups, Attach(z_cut, neck_xy, neck_r, chin, head_h)


def his_head(s, params=None):
    return Head2(F.build(), s, params or Head2Params())


def place_head(hd: Head2, at: Attach, roll_deg=0.0, pitch_deg=0.0, chin_drop=0.0):
    """Placement putting his chin where the original chin was and his neck on the body's neck."""
    s = hd.s
    R = rot("y", roll_deg) @ rot("x", pitch_deg)
    chin_l = hd.pts[F.CHIN]
    neck_l = np.array([0.0, hd.C[1] - 12.0, chin_l[2]])
    want_chin_z = at.chin[2] - chin_drop
    # x/y from the neck axis, z from the chin
    pos = np.array([at.neck_xy[0], at.neck_xy[1], want_chin_z]) - s * (R @ np.array([0.0, neck_l[1], chin_l[2]]))
    hd.neck_r = 0.97 * at.neck_r / s
    hd.neck_bottom = (at.z_cut - 6.0 - pos[2]) / s
    return Placement(pos, s, R)


def body_nodes(groups, at: Attach, h, combined=False, collar_h=4.0, smooth_mm=0.2, zmin=None):
    """SDF nodes of the body; combined=True solidifies skin+clothes together (game meshes that
    have no skin under the clothes need the clothes to close the volume)."""
    nodes = {}
    if combined:
        m = trimesh.util.concatenate(list(groups.values()))
        f, gr = mesh_to_grid(m, h, min_thickness=0.9, zmin=zmin)
        if smooth_mm:
            from scipy import ndimage
            f = ndimage.gaussian_filter(f, smooth_mm / h)
        node = GridSDF(f, gr, name="body")
        zc = float(at.z_cut)
        top = sdf.Func(lambda x, y, z: z - zc, (node.box[0] - 1, node.box[1] + 1))
        # keep clothing above the cut (collar) but drop the old head: cut only inside the neck column
        nx, ny, r = float(at.neck_xy[0]), float(at.neck_xy[1]), float(at.neck_r) * 3.2
        column = sdf.Func(lambda x, y, z: np.sqrt((x - nx) ** 2 + (y - ny) ** 2) - r, (node.box[0] - 1, node.box[1] + 1))
        nodes["body"] = sdf.Intersect(node, top, name="body")
        # the collar: clothing only, in a short band above the cut, outside the neck column
        if "cloth" in groups and collar_h > 0:
            fc, gc = mesh_to_grid(groups["cloth"], h, min_thickness=0.9)
            cloth = GridSDF(fc, gc, name="collar")
            band_h = 0.9 * collar_h
            band = sdf.Func(lambda x, y, z: np.maximum(zc - z, z - (zc + band_h)), (cloth.box[0] - 1, cloth.box[1] + 1))
            rr = float(at.neck_r) * 1.18
            outside = sdf.Func(lambda x, y, z: rr - np.sqrt((x - nx) ** 2 + (y - ny) ** 2), (cloth.box[0] - 1, cloth.box[1] + 1))
            front_ok = sdf.Func(lambda x, y, z: (ny - 1.6 * rr) - y, (cloth.box[0] - 1, cloth.box[1] + 1))
            # drop collar bits in front of the neck that sit where the old chin was
            keep = sdf.Intersect(sdf.Intersect(cloth, band), outside)
            nodes["collar"] = sdf.Subtract(keep, sdf.Intersect(front_ok, band) & sdf.Func(lambda x, y, z: z - (zc + 0.35 * band_h), (cloth.box[0] - 1, cloth.box[1] + 1)), name="collar")
        return nodes
    for g, m in groups.items():
        f, gr = mesh_to_grid(m, h, min_thickness=0.0 if g == "skin" else 0.9)
        node = GridSDF(f, gr, name=g)
        if g == "skin":  # remove everything above the neck cut
            zc = float(at.z_cut)
            cut = sdf.Func(lambda x, y, z: z - zc, (node.box[0] - 1, node.box[1] + 1))
            node = sdf.Intersect(node, cut, name="skin")
        nodes[g] = node
    return nodes


def head_nodes(hd: Head2, pl: Placement, hair=True):
    out = {"skin_head": hd.node(pl, "skin"), "glasses": hd.node(pl, "glasses")}
    if hair:
        out["hair"] = hd.node(pl, "hair")
    return out


def mesh_nodes(nodes, h, box=None, target_faces=None):
    u = sdf.Union(list(nodes.values()))
    if box is not None:
        lo, hi = box
        u = sdf.Intersect(u, sdf.box(0.5 * (lo + hi), 0.5 * (hi - lo)))
    f, g = sdf.evaluate(u, h)
    m, info = to_mesh(f, g, target_faces=target_faces)
    return m, info


def union_meshes(meshes):
    return trimesh.boolean.union(meshes, engine="manifold")


def gltf_body(path, body_objects, face_object, k_mm_per_unit, base_h=3.0, smooth_levels=1, neck_search=(0.78, 0.86)):
    """Static (already posed) glTF figure body -> ({group: mesh}, Attach).

    body_objects: {group: [geometry names]} to keep (e.g. skin / cloth / shoes); face_object: the
    face-skin geometry name used to find the chin. The head is cut at the narrowest neck section
    found between the given fractions of the model height.
    """
    from .vrmfig import smooth

    sc = trimesh.load(path)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])  # Y-up, +Z front -> Z-up, -Y front

    def get(name):
        g = sc.geometry[name]
        T = sc.graph.get(sc.graph.geometry_nodes[name][0])[0]
        return trimesh.transform_points(g.vertices, T) @ R.T, g.faces

    allv = np.concatenate([get(n)[0] for ns in body_objects.values() for n in ns])
    lo, hi = allv.min(0), allv.max(0)
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])

    def tf(V):
        return np.c_[(V[:, 0] - cx) * k_mm_per_unit, (V[:, 1] - cy) * k_mm_per_unit, base_h + (V[:, 2] - lo[2]) * k_mm_per_unit]

    groups = {}
    for g, names in body_objects.items():
        ms = [smooth(trimesh.Trimesh(tf(get(n)[0]), get(n)[1], process=False), smooth_levels) for n in names]
        groups[g] = trimesh.util.concatenate(ms)
    skin = groups["skin"].vertices
    fv = tf(get(face_object)[0])
    fx = 0.5 * (fv[:, 0].min() + fv[:, 0].max())
    front = fv[np.abs(fv[:, 0] - fx) < 0.08 * (fv[:, 0].max() - fv[:, 0].min())]
    chin = front[front[:, 2].argmin()]
    head_h = fv[:, 2].max() - chin[2]
    best = None
    # the neck: narrowest skin section a little below the chin
    for z in np.linspace(chin[2] - 0.40 * head_h, chin[2] - 0.04 * head_h, 40):
        ring = skin[np.abs(skin[:, 2] - z) < 0.5]
        if len(ring) < 12:
            continue
        c = ring[:, :2].mean(0)
        r = np.median(np.linalg.norm(ring[:, :2] - c, axis=1))
        if best is None or r < best[1]:
            best = (z, r, c)
    z_cut, neck_r, neck_xy = best
    return groups, Attach(float(z_cut), neck_xy, float(neck_r), chin, float(head_h))
