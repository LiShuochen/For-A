"""Colour previews of finished figures: body colours sampled from the source model's textures,
head / hair / glasses / base in fixed colours (what a multi-colour AMS print would look like)."""
from __future__ import annotations

import os

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

from .preview import render, sheet

SKIN = np.array([0.96, 0.84, 0.74])
HAIR = np.array([0.17, 0.15, 0.14])
BLACK = np.array([0.08, 0.08, 0.09])
BASE = np.array([0.82, 0.82, 0.84])


def _texture(geom):
    mat = geom.visual.material
    img = getattr(mat, "baseColorTexture", None)
    if img is None:
        c = getattr(mat, "baseColorFactor", None)
        c = np.array([0.15, 0.15, 0.16]) if c is None else np.asarray(c, float).ravel()[:3]
        return None, (c / 255.0 if c.max() > 1 else c)
    return np.asarray(img.convert("RGB"), float) / 255.0, None


def body_colors(mesh, gltf_path, body_objects, k, base_h, prefer_less=("Object_0",)):
    """Per-vertex colours for `mesh` sampled from the textured source objects (same transform as
    assemble.gltf_body)."""
    sc = trimesh.load(gltf_path)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
    srcs = []
    for n in body_objects:
        g = sc.geometry[n]
        T = sc.graph.get(sc.graph.geometry_nodes[n][0])[0]
        srcs.append((n, trimesh.transform_points(g.vertices, T) @ R.T, g))
    allv = np.concatenate([v for _, v, _ in srcs])
    lo, hi = allv.min(0), allv.max(0)
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    tf = lambda V: np.c_[(V[:, 0] - cx) * k, (V[:, 1] - cy) * k, base_h + (V[:, 2] - lo[2]) * k]
    P = np.asarray(mesh.vertices, np.float32)
    best_d = np.full(len(P), np.inf)
    out = np.zeros((len(P), 3))
    # clothing wins over skin when both are about equally close (skin often sits just under it)
    bias = {n: (-0.6 if n not in prefer_less else 0.0) for n, _, _ in srcs}
    for n, v, g in srcs:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(tf(v).astype(np.float32)), o3d.core.Tensor(g.faces.astype(np.uint32)))
        q = scene.compute_closest_points(o3d.core.Tensor(P))
        d = np.linalg.norm(q["points"].numpy() - P, axis=1) + bias[n]
        sel = d < best_d
        if not sel.any():
            continue
        best_d[sel] = d[sel]
        tex, flat = _texture(g)
        if tex is None or getattr(g.visual, "uv", None) is None:
            out[sel] = flat
            continue
        tri = g.faces[q["primitive_ids"].numpy()[sel]]
        bary = q["primitive_uvs"].numpy()[sel]
        uv = g.visual.uv
        w1, w2 = bary[:, 0:1], bary[:, 1:2]
        p = (1 - w1 - w2) * uv[tri[:, 0]] + w1 * uv[tri[:, 1]] + w2 * uv[tri[:, 2]]
        H, W = tex.shape[:2]
        px = np.clip((p[:, 0] % 1.0) * (W - 1), 0, W - 1).astype(int)
        py = np.clip((1 - (p[:, 1] % 1.0)) * (H - 1), 0, H - 1).astype(int)
        out[sel] = tex[py, px]
    return out


def present(name, out_dir, parts, body_cols=None, face_crop_mm=60.0):
    """parts: {part: mesh}. Writes preview_colour.png, preview_face.png, preview_single.png."""
    cols = {"skin_head": SKIN, "skin": SKIN, "hair": HAIR, "glasses": BLACK, "base": BASE,
            "top": np.array([0.97, 0.97, 0.96]), "collar": np.array([0.13, 0.17, 0.32]),
            "bow": np.array([0.80, 0.12, 0.14]), "skirt": np.array([0.13, 0.17, 0.32]),
            "socks": np.array([0.12, 0.13, 0.20]), "shoes": np.array([0.24, 0.15, 0.10])}
    meshes, colors = [], []
    for p, m in parts.items():
        meshes.append(m)
        if p == "body" and body_cols is not None:
            colors.append(body_cols)
        else:
            colors.append(np.tile(cols.get(p, SKIN), (len(m.vertices), 1)))
    whole = trimesh.util.concatenate(meshes)
    col = np.concatenate(colors)
    top = whole.bounds[1, 2]
    px = max(whole.extents.max() / 900.0, 0.12)
    ims = [render(whole, az, 5, px=px, colors=col) for az in (0, -30, 30, 90, 180)]
    sheet(ims, None, height=1000).save(os.path.join(out_dir, "preview_colour.png"))
    face = [render(whole, az, 3, px=0.08, colors=col, crop=(top - face_crop_mm, top + 1)) for az in (0, -35, 35)]
    sheet(face, None, height=800).save(os.path.join(out_dir, "preview_face.png"))
    single = [render(whole, az, 5, px=px) for az in (0, -30)]
    sheet(single, None, height=1000).save(os.path.join(out_dir, "preview_single_colour.png"))
