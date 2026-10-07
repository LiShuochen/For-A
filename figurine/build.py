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


def export(name, parts, h_parts, faces_per_part=600_000):
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
        t = time.time()
        m, info = _mesh(_minus(node, *earlier), h_parts[pname], faces_per_part)
        m.export(os.path.join(d, f"{name}_{pname}.stl"))
        meshes[pname] = m
        report[pname] = {"faces": len(m.faces), "watertight": bool(m.is_watertight),
                         "dropped_islands_mm3": [round(x, 2) for x in info["dropped_islands_mm3"][:5]],
                         "seconds": round(time.time() - t)}
        print(f"[{name}] part {pname}: {len(m.faces)} faces, {report[pname]['seconds']}s", flush=True)
        earlier.append(node)
    whole = robust_union(list(meshes.values()))
    whole.export(os.path.join(OUT, name, f"{name}.stl"))
    rep = mesh_report(whole)
    rep["parts"] = report
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
    return export("his_jk_full", parts, hp)




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
    return export(name, parts, hp)


def chibi_bald():
    return big_head("his_jk_q_bald", "bald")


def chibi_long():
    return big_head("his_jk_q_longhair", "long")


def bust(height=172.0, base_h=3.0, head_mm=50.0, h_body=0.13, h_head=0.08):
    """Bust: from the skirt hem up, standing on a disc; bigger head scale for a detailed face."""
    s = head_mm / 232.0
    # Luntima units: head (chin->crown) ~2.58, skirt hem at ~8.9 units, chin at ~14.45
    k = (height - base_h - head_mm) / (14.45 - 8.9)
    groups, at = A.gltf_body(LUNTIMA, {"skin": ["Object_0"], "cloth": ["Object_9", "Object_3"]}, "Object_5", k,
                             base_h=base_h - 8.9 * k)
    hd = A.his_head(s)
    pl = A.place_head(hd, at, roll_deg=-4.0, pitch_deg=4.0)
    body = A.body_nodes(groups, at, h_body, combined=True, collar_h=0.0, smooth_mm=0.3)["body"]
    zcut = base_h + 0.2
    body = sdf.Intersect(body, sdf.Func(lambda x, y, z: zcut - z, (body.box[0] - 1, body.box[1] + 1)))
    skirt = groups["cloth"].vertices
    ring = skirt[np.abs(skirt[:, 2] - (base_h + 2)) < 3]
    r_base = float(np.linalg.norm(ring[:, :2], axis=1).max() + 6.0) if len(ring) else 40.0
    base = sdf.cylinder_z((0, 0), r_base, 0.0, base_h + 0.4, round_=0.8, name="base")
    head = A.head_nodes(hd, pl)
    parts = {"skin_head": head["skin_head"], "body": body, "hair": head["hair"], "glasses": head["glasses"], "base": base}
    hp = {"skin_head": h_head, "body": h_body, "hair": h_head, "glasses": h_head, "base": 0.3}
    return export("his_jk_bust", parts, hp)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "full"
    globals()[what]()
