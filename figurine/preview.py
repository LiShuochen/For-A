"""Software preview renderer: dense vertex splatting with a z-buffer + clay shading.

Works headless (no OpenGL). Needs a dense mesh (the un-decimated marching-cubes output),
where vertex spacing is at most about the pixel size.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .sdf import rot

CLAY = np.array([0.93, 0.89, 0.84])


def _view_matrix(az, el):
    # camera looks along +y from -y; az rotates the model about z, el tilts the camera down
    return rot("x", el) @ rot("z", -az)


def render(mesh, az=0.0, el=0.0, px=0.15, colors=None, crop=None, bg=255):
    """Render mesh. az/el in degrees (az=0 -> front view, face looks at the camera).

    colors: optional (n_vertices, 3) floats 0..1. crop: world-space (zmin, zmax) to frame.
    """
    if crop is not None:
        import trimesh
        c = mesh.triangles_center[:, 2]
        sel = (c >= crop[0] - 1) & (c <= crop[1] + 1)
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[sel], process=False)
        mesh.remove_unreferenced_vertices()
        colors = None
    if mesh.edges_unique_length.max() > 1.5 * px:
        # coarse triangles relative to the pixel size: split them so splats leave no holes
        import trimesh
        v2, f2 = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.9 * px, max_iter=12)
        mesh = trimesh.Trimesh(v2, f2, process=False)
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    M = _view_matrix(az, el)
    P = V @ M.T
    Nn = N @ M.T
    if crop is not None:
        keep = (V[:, 2] >= crop[0]) & (V[:, 2] <= crop[1])
        P, Nn = P[keep], Nn[keep]
        colors = None if colors is None else colors[keep]
    u, v, d = P[:, 0], P[:, 2], P[:, 1]
    umin, vmax = u.min() - 2 * px, v.max() + 2 * px
    W = int(np.ceil((u.max() - umin) / px)) + 3
    H = int(np.ceil((vmax - v.min()) / px)) + 3
    j = ((u - umin) / px).astype(np.int64)
    i = ((vmax - v) / px).astype(np.int64)
    # 2x2 splat footprint; sort all splats far -> near so nearer ones are written last and win
    ii = np.concatenate([i + di for di in (0, 1) for dj in (0, 1)])
    jj = np.concatenate([j + dj for di in (0, 1) for dj in (0, 1)])
    dd4 = np.tile(d, 4)
    src = np.tile(np.arange(len(d)), 4)
    order = np.argsort(-dd4, kind="stable")
    ii, jj, src = ii[order], jj[order], src[order]
    depth = np.full((H, W), np.inf)
    nimg = np.zeros((H, W, 3))
    cimg = np.tile(CLAY, (H, W, 1))
    depth[ii, jj] = d[src]
    nimg[ii, jj] = Nn[src]
    if colors is not None:
        cimg[ii, jj] = colors[src]
    hit = np.isfinite(depth)
    # fill pin-holes
    holes = ~hit & ndimage.binary_closing(hit, iterations=2)
    if holes.any():
        _, (ii, jj) = ndimage.distance_transform_edt(~hit, return_indices=True)
        nimg[holes] = nimg[ii[holes], jj[holes]]
        cimg[holes] = cimg[ii[holes], jj[holes]]
        depth[holes] = depth[ii[holes], jj[holes]]
        hit = hit | holes
    nrm = nimg / np.maximum(np.linalg.norm(nimg, axis=2, keepdims=True), 1e-9)
    # lights in camera space: camera at -y, so "toward camera" is -y
    key = np.array([-0.45, -0.75, 0.5]); key /= np.linalg.norm(key)
    fill = np.array([0.6, -0.6, 0.1]); fill /= np.linalg.norm(fill)
    rim = np.array([0.0, 0.8, 0.6]); rim /= np.linalg.norm(rim)
    lam = 0.62 * np.clip(nrm @ key, 0, 1) + 0.25 * np.clip(nrm @ fill, 0, 1) + 0.18 * np.clip(nrm @ rim, 0, 1)
    # cheap ambient occlusion from the depth buffer
    far = depth[hit].max() if hit.any() else 0.0
    dz = np.where(hit, depth, far)
    blur = ndimage.gaussian_filter(dz, 5)
    ao = np.clip(1.0 - 0.12 * np.maximum(dz - blur, 0), 0.55, 1.0)
    shade = (0.22 + lam)[..., None] * ao[..., None] * cimg
    img = np.full((H, W, 3), bg / 255.0)
    img[hit] = np.clip(shade[hit], 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def sheet(images, labels=None, height=900, pad=16):
    ims = []
    for im in images:
        r = height / im.height
        ims.append(im.resize((max(1, int(im.width * r)), height), Image.LANCZOS))
    W = sum(i.width for i in ims) + pad * (len(ims) + 1)
    out = Image.new("RGB", (W, height + 2 * pad + (30 if labels else 0)), "white")
    x = pad
    d = ImageDraw.Draw(out)
    for k, im in enumerate(ims):
        out.paste(im, (x, pad))
        if labels:
            d.text((x + 4, height + pad + 6), labels[k], fill=(60, 60, 60))
        x += im.width + pad
    return out
