"""Build the final figurines.

    python -m figurine.build full      # full-body JK (Luntima body) + his head, ~170 mm
    python -m figurine.build bust      # bust down to the skirt hem
    python -m figurine.build chibi_bald | chibi_long

Each writes out/<name>/<name>.stl (single-colour, one watertight body) and out/<name>/parts/*.stl
(non-overlapping colour parts for Bambu Studio / AMS: skin, hair, glasses, body, base).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import trimesh

from . import assemble as A
from . import sdf
from .check import mesh_report
from .head2 import crown_tufts

LUNTIMA = "assets/ref_models/beauty/luntima_student/scene.gltf"
OUT = "out"


def _mesh(node, h, faces=None):
    f, g = sdf.evaluate(node, h)
    m, info = A.to_mesh(f, g, target_faces=faces)
    del f, g
    return m, info


def _minus(a, *others):
    n = a
    for o in others:
        n = sdf.Subtract(n, o)
    return n


def whole_from_sdf(parts, h_parts, head_box, overlap=2.0, max_voxels=110e6):
    """Single-colour print: SDF union of the parts, meshed as three overlapping pieces (fine head
    core, coarser rest of the figure, the base disc) joined by a robust boolean union. The pieces
    overlap instead of touching, so the union is clean."""
    head_keys = [k for k in ("skin_head", "glasses") if k in parts]
    others = {k: v for k, v in parts.items() if k != "base"}
    u = sdf.Union(list(others.values()))
    # head core box: the head skin/glasses extents, from the neck cut up
    hl = np.min([parts[k].box[0] for k in head_keys], axis=0) - 1.0
    hh = np.max([parts[k].box[1] for k in head_keys], axis=0) + 1.0
    hl[2] = max(hl[2], head_box[0][2])
    head_core = sdf.Intersect(u, sdf.box(0.5 * (hl + hh), 0.5 * (hh - hl)))
    inner = (hl + overlap, hh - overlap)
    not_head = sdf.Func(lambda x, y, z, c=0.5 * (inner[0] + inner[1]), r=0.5 * (inner[1] - inner[0]): -(
        np.maximum(np.maximum(np.abs(x - c[0]) - r[0], np.abs(y - c[1]) - r[1]), np.abs(z - c[2]) - r[2])),
        u.box)
    rest = sdf.Intersect(u, not_head)
    lo, hi = (np.asarray(v, float) for v in u.box)
    vol = np.prod(hi - lo + 2.0)
    h_head = max(min(h_parts.values()), (np.prod(hh - hl) / (0.8 * max_voxels)) ** (1 / 3))
    h_rest = max(max(v for k, v in h_parts.items() if k != "base"), (vol / max_voxels) ** (1 / 3))
    pieces = []
    for node, h in ((head_core, h_head), (rest, h_rest), (parts.get("base"), h_parts.get("base", 0.3))):
        if node is None:
            continue
        m, _ = _mesh(node, h)
        pieces.append(m)
        print(f"[whole] piece {len(pieces)}: {len(m.faces)} faces at h={h:.3f}", flush=True)
    return drop_slivers(robust_union(pieces))


def drop_slivers(m, min_vol=0.05, min_faces=30):
    """Remove the zero-volume sliver shells a boolean union leaves where parts touch exactly."""
    comps = m.split(only_watertight=False)
    keep = [c for c in comps if len(c.faces) >= min_faces and abs(c.volume) > min_vol]
    out = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    out.merge_vertices()
    return fix_pinches(out)


def _bad_edges(m):
    u, c = np.unique(m.edges_sorted, axis=0, return_counts=True)
    return u[c != 2]


def fix_pinches(m):
    """Split edges shared by four faces (two sheets touching along a line): the second sheet gets
    its own copies of the two vertices, so every edge has exactly two faces. Nothing is deleted."""
    m = trimesh.Trimesh(m.vertices, m.faces, process=True)
    faces = m.faces.copy()
    verts = list(m.vertices)
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    bad = u[c == 4]
    if len(bad) == 0:
        return m
    vf = {}
    for fi, f in enumerate(faces):
        for v in f:
            vf.setdefault(int(v), []).append(fi)
    for a, b in bad:
        a, b = int(a), int(b)
        for v, other in ((a, b), (b, a)):
            fan = vf.get(v, [])
            parent = {f: f for f in fan}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            by_vertex = {}
            for f in fan:
                for x in faces[f]:
                    x = int(x)
                    if x != v and x != other:
                        by_vertex.setdefault(x, []).append(f)
            for fs in by_vertex.values():
                for f2 in fs[1:]:
                    parent[find(f2)] = find(fs[0])
            groups = {}
            for f in fan:
                groups.setdefault(find(f), []).append(f)
            for g in list(groups.values())[1:]:
                nv = len(verts)
                verts.append(verts[v])
                for f in g:
                    faces[f][faces[f] == v] = nv
                vf[nv] = g
                vf[v] = [f for f in vf[v] if f not in g]
    return trimesh.Trimesh(np.asarray(verts), faces, process=False)


def robust_union(meshes):
    """Boolean union; repairs and retries, and as a last resort concatenates the closed shells
    (slicers such as Bambu Studio merge overlapping shells of one STL anyway)."""
    try:
        return trimesh.boolean.union(meshes, engine="manifold")
    except Exception as ex:  # noqa: BLE001
        print("[union] manifold failed:", ex, "- repairing", flush=True)
    fixed = []
    for m in meshes:
        m = m.copy()
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
        fixed.append(m)
    try:
        return trimesh.boolean.union(fixed, engine="manifold")
    except Exception as ex:  # noqa: BLE001
        print("[union] still failing:", ex, "- exporting concatenated shells", flush=True)
        return trimesh.util.concatenate(fixed)


def export(name, parts, h_parts, faces_per_part=600_000, meta=None, head_box=None):
    """parts: ordered {part: node}; later parts lose their overlap with earlier ones.

    With PREVIEW=<voxel mm> in the environment, only a quick single mesh is made (for iteration).
    """
    if os.environ.get("PREVIEW"):
        hprev = float(os.environ["PREVIEW"])
        m, _ = _mesh(sdf.Union(list(parts.values())), hprev)
        from .preview import render, sheet
        os.makedirs("scratch/previews", exist_ok=True)
        m.export(f"scratch/previews/{name}.stl")
        top = m.bounds[1, 2]
        ims = [render(m, az, 4, px=max(hprev, 0.18)) for az in (0, -30, 90)]
        ims += [render(m, az, 4, px=0.08, crop=(top - 70, top + 1)) for az in (0, -35)]
        sheet(ims, None, height=800).save(f"scratch/previews/{name}.png")
        print(name, "preview", m.bounds.round(1).tolist(), flush=True)
        return m, {}
    d = os.path.join(OUT, name, "parts")
    os.makedirs(d, exist_ok=True)
    meshes, report = {}, {}
    earlier = []
    for pname, node in parts.items():
        if os.environ.get("WHOLE_ONLY"):
            break
        t = time.time()
        m, info = _mesh(_minus(node, *earlier), h_parts[pname], faces_per_part)
        m.export(os.path.join(d, f"{name}_{pname}.stl"))
        meshes[pname] = m
        report[pname] = {"faces": len(m.faces), "watertight": bool(m.is_watertight),
                         "dropped_islands_mm3": [round(x, 2) for x in info["dropped_islands_mm3"][:5]],
                         "seconds": round(time.time() - t)}
        print(f"[{name}] part {pname}: {len(m.faces)} faces, {report[pname]['seconds']}s", flush=True)
        earlier.append(node)
    whole = whole_from_sdf(parts, h_parts, head_box)
    whole.export(os.path.join(OUT, name, f"{name}.stl"))
    rep = mesh_report(whole)
    old_rep = os.path.join(OUT, name, "report.json")
    if not report and os.path.exists(old_rep):  # WHOLE_ONLY: keep the earlier per-part info
        report = json.load(open(old_rep)).get("parts", {})
    rep["parts"] = report
    rep["meta"] = meta or {}
    with open(os.path.join(OUT, name, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(f"[{name}] whole: {json.dumps({k: v for k, v in rep.items() if k != 'parts'})}", flush=True)
    return whole, meshes


def full(height=170.0, base_h=3.0, h_body=0.14, h_head=0.08):
    k = (height - base_h) / 17.2  # Luntima is ~17.2 units from soles to crown
    groups, at = A.gltf_body(LUNTIMA, {"skin": ["Object_0"], "cloth": ["Object_9", "Object_3"]}, "Object_5", k,
                             base_h=base_h)
    s = 1.04 * at.head_h / 232.0
    hd = A.his_head(s)
    pl = A.place_head(hd, at, roll_deg=-4.0, pitch_deg=4.0)
    body = A.body_nodes(groups, at, h_body, combined=True, collar_h=0.0, smooth_mm=0.3)["body"]
    feet = np.concatenate([g.vertices for g in groups.values()])
    feet = feet[feet[:, 2] < base_h + 10]
    r_base = float(np.linalg.norm(feet[:, :2], axis=1).max() * 1.2 + 4.0)
    base = sdf.cylinder_z((0, 0), r_base, 0.0, base_h + 0.4, round_=0.8, name="base")
    head = A.head_nodes(hd, pl)
    parts = {"skin_head": head["skin_head"], "body": body, "hair": head["hair"], "glasses": head["glasses"], "base": base}
    hp = {"skin_head": h_head, "body": h_body, "hair": h_head, "glasses": h_head, "base": 0.3}
    return export("his_jk_full", parts, hp, meta={"k": k, "base_h": base_h}, head_box=_head_box(at))




def _head_box(at, long_hair=False):
    """Region meshed at head resolution: from just below the neck cut upward, full width."""
    z0 = at.z_cut - 4.0
    return (np.array([-80.0, -80.0, z0]), np.array([80.0, 80.0, 400.0]))


def _luntima(height_body_to_neck=None, k=None, base_h=3.0):
    groups, at = A.gltf_body(LUNTIMA, {"skin": ["Object_0"], "cloth": ["Object_9", "Object_3"]}, "Object_5",
                             k, base_h=base_h)
    return groups, at


def long_hair(at: A.Attach, hd, pl, k, base_h=3.0):
    """Luntima's long hair without the bangs, moved from her head onto his (similarity fit)."""
    from .vrmfig import smooth  # noqa: F401  (hair cards stay as is)

    sc = trimesh.load(LUNTIMA)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])

    def get(n):
        g = sc.geometry[n]
        T = sc.graph.get(sc.graph.geometry_nodes[n][0])[0]
        return trimesh.transform_points(g.vertices, T) @ R.T, g.faces

    allv = np.concatenate([get(n)[0] for n in ("Object_0", "Object_9", "Object_3")])
    lo, hi = allv.min(0), allv.max(0)
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    tf = lambda V: np.c_[(V[:, 0] - cx) * k, (V[:, 1] - cy) * k, base_h + (V[:, 2] - lo[2]) * k]
    hv, hf = get("Object_8")
    hv = tf(hv)
    cap = tf(get("Object_2")[0])
    face = tf(get("Object_5")[0])
    # remove the bangs: faces hanging in front of the face, between the temples
    c = hv[hf].mean(1)
    fx0, fx1 = face[:, 0].min(), face[:, 0].max()
    front_y = face[:, 1].min()
    hw = 0.5 * (fx1 - fx0)
    xc = 0.5 * (fx0 + fx1)
    bangs = (np.abs(c[:, 0] - xc) < 0.80 * hw) & (c[:, 1] < front_y + 0.55 * hw) & (c[:, 2] > at.chin[2] - 2)
    hf = hf[~bangs]
    # similarity: her cranium (hair cap) -> his cranium
    her_c = 0.5 * (cap.min(0) + cap.max(0))
    her_w = 0.5 * (cap[:, 0].max() - cap[:, 0].min())
    his_c = pl.to_world(hd.C)
    his_w = (hd.P.half_width + 0.5 * hd.P.hair_top) * hd.s
    sc_f = his_w / her_w
    # sit the wig on his skull: a touch lower and further back so it hugs the crown
    V = (hv - her_c) * sc_f + his_c + np.array([0, 0.06 * his_w, -0.05 * his_w])
    m = trimesh.Trimesh(V, hf, process=False)
    m.remove_unreferenced_vertices()
    return m


def big_head(name="his_jk_q_bald", hair="bald", total=150.0, head_ratio=0.30, base_h=3.0, h_body=0.13, h_head=0.09):
    """Q-version: the pretty body with his head enlarged (bobble-head), bald+tufts or long hair."""
    from .meshsdf import GridSDF, mesh_to_grid

    # body: soles -> chin height so that the head (chin to crown ~232*s) is head_ratio of the total
    head_mm = head_ratio * (total - base_h)
    chin_frac = 14.45 / 17.2
    k = (total - base_h - head_mm) / (chin_frac * 17.2)
    groups, at = A.gltf_body(LUNTIMA, {"skin": ["Object_0"], "cloth": ["Object_9", "Object_3"]}, "Object_5", k,
                             base_h=base_h)
    s = head_mm / 232.0
    hd = A.his_head(s)
    pl = A.place_head(hd, at, roll_deg=-6.0, pitch_deg=5.0, chin_drop=-0.06 * head_mm)
    hd.neck_r = 1.25 * at.neck_r / s  # a bobble head needs a sturdier neck
    body = A.body_nodes(groups, at, h_body, combined=True, collar_h=0.0, smooth_mm=0.3)["body"]
    feet = np.concatenate([g.vertices for g in groups.values()])
    feet = feet[feet[:, 2] < base_h + 10]
    r_base = float(np.linalg.norm(feet[:, :2], axis=1).max() * 1.35 + 5.0)
    base = sdf.cylinder_z((0, 0), r_base, 0.0, base_h + 0.4, round_=0.8, name="base")
    head = A.head_nodes(hd, pl, hair=(hair == "long"))
    parts = {"skin_head": head["skin_head"], "body": body}
    hp = {"skin_head": h_head, "body": h_body}
    if hair == "bald":
        parts["hair"] = crown_tufts(hd, pl, n=56, length=(14.0, 24.0), r_base_p=0.36, r_tip_p=0.2)
        hp["hair"] = 0.06
    else:
        hm = long_hair(at, hd, pl, k, base_h)
        f, g = mesh_to_grid(hm, h_body, min_thickness=1.0)
        parts["hair"] = sdf.Union([GridSDF(f, g, name="long_hair"), head["hair"]], name="hair")
        hp["hair"] = h_body
    parts["glasses"] = head["glasses"]
    hp["glasses"] = h_head
    parts["base"] = base
    hp["base"] = 0.3
    return export(name, parts, hp, meta={"k": k, "base_h": base_h}, head_box=_head_box(at, hair == "long"))


def chibi_bald():
    return big_head("his_jk_q_bald", "bald")


def chibi_long():
    return big_head("his_jk_q_longhair", "long")


def bust(height=172.0, base_h=3.0, head_mm=50.0, h_body=0.16, h_head=0.08):
    """Bust: from the skirt hem up, standing on a disc; bigger head scale for a detailed face."""
    s = head_mm / 232.0
    # Luntima units: head (chin->crown) ~2.58, skirt hem at ~8.9 units, chin at ~14.45
    k = (height - base_h - head_mm) / (14.45 - 8.9)
    groups, at = A.gltf_body(LUNTIMA, {"skin": ["Object_0"], "cloth": ["Object_9", "Object_3"]}, "Object_5", k,
                             base_h=base_h - 8.9 * k)
    hd = A.his_head(s)
    pl = A.place_head(hd, at, roll_deg=-4.0, pitch_deg=4.0)
    body = A.body_nodes(groups, at, h_body, combined=True, collar_h=0.0, smooth_mm=0.3, zmin=base_h - 2.0)["body"]
    zcut = base_h + 0.2
    body = sdf.Intersect(body, sdf.Func(lambda x, y, z: zcut - z, (body.box[0] - 1, body.box[1] + 1)))
    skirt = groups["cloth"].vertices
    ring = skirt[np.abs(skirt[:, 2] - (base_h + 2)) < 3]
    r_base = float(np.linalg.norm(ring[:, :2], axis=1).max() + 6.0) if len(ring) else 40.0
    base = sdf.cylinder_z((0, 0), r_base, 0.0, base_h + 0.4, round_=0.8, name="base")
    head = A.head_nodes(hd, pl)
    parts = {"skin_head": head["skin_head"], "body": body, "hair": head["hair"], "glasses": head["glasses"], "base": base}
    hp = {"skin_head": h_head, "body": h_body, "hair": h_head, "glasses": h_head, "base": 0.3}
    return export("his_jk_bust", parts, hp, meta={"k": k, "base_h": base_h - 8.9 * k}, head_box=_head_box(at))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "full"
    globals()[what]()
