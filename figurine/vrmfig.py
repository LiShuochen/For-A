"""VRoid/VRM avatar -> posed, printable figurine parts (mm, Z up, facing -Y)."""
from __future__ import annotations

import numpy as np
import trimesh

from .vrm import VRM, axis_angle

# material-name keywords -> colour group; parts not matched are dropped
GROUPS = {
    "skin": ["_SKIN", "EyeWhite"],
    "cloth": ["Tops", "Bottoms", "AccessoryNeck", "Shoes", "CLOTH"],
    "hair": ["_HAIR"],
}
DROP = ["EyeIris", "EyeHighlight", "EyeExtra", "FaceEyeline", "FaceEyelash", "FaceBrow"]


def expression_morphs(v: VRM, name, weight=1.0):
    """{mesh index: {target index: weight}} for a VRM 0.x blendshape group or VRM 1.0 expression."""
    out = {}
    ext = v.g.extensions or {}
    if "VRM" in ext:
        for g in ext["VRM"]["blendShapeMaster"]["blendShapeGroups"]:
            if g["name"].lower() == name.lower():
                for b in g.get("binds", []):
                    out.setdefault(b["mesh"], {})[b["index"]] = weight * b["weight"] / 100.0
    elif "VRMC_vrm" in ext:
        e = ext["VRMC_vrm"]["expressions"]
        for group in ("preset", "custom"):
            for k, val in e.get(group, {}).items():
                if k.lower() == name.lower():
                    for b in val.get("morphTargetBinds", []):
                        mesh = v.g.nodes[b["node"]].mesh
                        out.setdefault(mesh, {})[b["index"]] = weight * b["weight"]
    return out


def relaxed_pose(v: VRM, arm_down=72.0, elbow=14.0, wrist_in=8.0, legs_apart=1.5):
    """Arms down from the T-pose with a slight elbow bend; works for VRM 0.x and 1.0."""
    rest = v.globals({})
    fwd = -1.0 if v.version == 0 else 1.0  # +Z forward for VRM 1.0, -Z for 0.x
    rots = {}
    for side, sx in (("left", None), ("right", None)):
        up = v.bone_position(f"{side}UpperArm")
        lo = v.bone_position(f"{side}LowerArm")
        sgn = np.sign(lo[0] - up[0])  # arm points to +x or -x
        rots[f"{side}UpperArm"] = axis_angle([0, 0, 1], -sgn * arm_down)
        # bend the elbow forward: rotate the forearm about the world x axis
        rots[f"{side}LowerArm"] = axis_angle([1, 0, 0], -fwd * elbow) @ axis_angle([0, 0, 1], -sgn * 4.0)
        rots[f"{side}Hand"] = axis_angle([0, 1, 0], sgn * wrist_in)
        ul = v.bone_position(f"{side}UpperLeg")
        rots[f"{side}UpperLeg"] = axis_angle([0, 0, 1], np.sign(ul[0]) * legs_apart)
    return v.world_rotation_pose(rest, rots)


def taubin_pinned(v, f, iterations=8, lamb=0.5, mu=-0.53):
    """Taubin smoothing that keeps open-boundary vertices fixed (so parts stay attached)."""
    import scipy.sparse as sp

    n = len(v)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    W = sp.diags(1.0 / np.maximum(deg, 1)) @ A
    # boundary vertices: on an edge used by exactly one face
    es = np.sort(e, axis=1)
    uniq, cnt = np.unique(es, axis=0, return_counts=True)
    free = np.ones(n, bool)
    free[np.unique(uniq[cnt == 1])] = False
    v = v.copy()
    for _ in range(iterations):
        for k in (lamb, mu):
            d = W @ v - v
            v[free] += k * d[free]
    return v


def smooth(m, levels):
    """Remove game-mesh faceting: weld UV seams, midpoint-subdivide, then Taubin-smooth.

    Loop subdivision throws long spikes on VRoid's non-manifold clothing, so it is not used;
    open boundaries (e.g. the top of the neck) are pinned so parts stay connected.
    """
    if levels <= 0:
        return m
    m = m.copy()
    m.merge_vertices(merge_tex=True, merge_norm=True)
    v, f = m.vertices, m.faces
    for _ in range(levels):
        v, f = trimesh.remesh.subdivide(v, f)
    v = taubin_pinned(np.asarray(v, float), np.asarray(f), iterations=8 * levels)
    return trimesh.Trimesh(v, f, process=False)


def figure_parts(path, height_mm, z0=0.0, expression=None, expression_weight=1.0, pose=None,
                 keep_hair=True, extra_drop=(), subdivide=0, vertex_hook=None):
    """Return ({group: trimesh}, info) posed and scaled so the whole figure is height_mm tall."""
    v = VRM(path)
    pose = relaxed_pose(v) if pose is None else pose
    morphs = {}
    exprs = expression if isinstance(expression, dict) else ({expression: expression_weight} if expression else {})
    for name, wgt in exprs.items():
        for mesh, tw in expression_morphs(v, name, wgt).items():
            for ti, w in tw.items():
                morphs.setdefault(mesh, {})[ti] = morphs.get(mesh, {}).get(ti, 0.0) + w
    parts = v.parts(pose, morphs)
    # scale from the full model (hair included) so every variant shares one scale
    scale_parts = [p for p in parts if not any(d in p.name for d in DROP)
                   and any(k in p.name for keys in GROUPS.values() for k in keys)]
    kept = []
    decals = []  # not printed, but handed to vertex_hook (e.g. for landmark measurement)
    for p in parts:
        if any(d in p.name for d in DROP) or any(d in p.name for d in extra_drop) or "FaceMouth" in p.name:
            decals.append(p)
            continue
        for g, keys in GROUPS.items():
            if any(k in p.name for k in keys):
                if g == "hair" and not keep_hair:
                    break
                kept.append((g, p))
                break
    # glTF Y-up -> Z-up, and turn the model to face -Y
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
    if v.version == 0:
        R = np.diag([-1, -1, 1.0]) @ R
    allv = np.concatenate([p.vertices for p in scale_parts]) @ R.T
    lo, hi = allv.min(0), allv.max(0)
    s = height_mm / (hi[2] - lo[2])
    off = np.array([-(lo[0] + hi[0]) / 2 * s, -(lo[1] + hi[1]) / 2 * s, z0 - lo[2] * s])

    def tf(V):
        return (V @ R.T) * s + off

    verts = {p.name: tf(p.vertices) for _, p in kept}
    verts.update({p.name: tf(p.vertices) for p in decals})
    if vertex_hook is not None:
        verts = vertex_hook(verts)
    out = {}
    for g in GROUPS:
        ms = []
        for gg, p in kept:
            if gg != g:
                continue
            smooth_it = ("_SKIN" in p.name or "CLOTH" in p.name) and "HAIR" not in p.name
            ms.append(smooth(trimesh.Trimesh(verts[p.name], p.faces, process=False), subdivide if smooth_it else 0))
        if ms:
            out[g] = trimesh.util.concatenate(ms)
    info = {"scale": s, "R": R, "offset": off, "vrm": v, "pose": pose, "tf": tf, "parts": verts}
    return out, info


CHIBI_SCALES = {
    "spine": [0.78, 0.62, 0.76], "chest": [0.78, 0.62, 0.74], "upperChest": [0.78, 0.62, 0.74],
    "neck": [0.75, 0.5, 0.75],
    "leftShoulder": [0.85, 0.85, 0.85], "rightShoulder": [0.85, 0.85, 0.85],
    "leftUpperArm": [0.62] * 3, "rightUpperArm": [0.62] * 3,
    "leftHand": [0.8] * 3, "rightHand": [0.8] * 3,
    "leftUpperLeg": [0.82, 0.40, 0.82], "rightUpperLeg": [0.82, 0.40, 0.82],
    "leftFoot": [1.12] * 3, "rightFoot": [1.12] * 3,
}


def chibi_pose(v, scales=None, arm_down=66.0, elbow=12.0):
    """Q-version body: short torso and legs, small arms, big shoes, arms relaxed down."""
    rots = {}
    for side in ("left", "right"):
        up, lo = v.bone_position(f"{side}UpperArm"), v.bone_position(f"{side}LowerArm")
        sgn = np.sign(lo[0] - up[0])
        rots[f"{side}UpperArm"] = axis_angle([0, 0, 1], -sgn * arm_down)
        rots[f"{side}LowerArm"] = axis_angle([1, 0, 0], elbow) @ axis_angle([0, 0, 1], -sgn * 6.0)
    return v.scaled_pose(scales or CHIBI_SCALES, rots)
