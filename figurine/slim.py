"""Print-resolution copies small enough to send: quadric decimation, pinch repair, verification.

    python -m figurine.slim      # out/<name>/... -> out/deliver/*.zip (each under ~30 MiB)

Marching-cubes meshes are heavily over-tessellated; decimating to ~1M triangles moves the surface
by only a few microns, far below what a 0.4 mm nozzle can print.
"""
from __future__ import annotations

import os
import shutil
import zipfile

import numpy as np
import open3d as o3d
import trimesh

from .build import drop_slivers
from .mesher import nonmanifold_edges

# triangle budgets: whole models and colour parts
WHOLE = 1_000_000
PARTS = {
    "his_jk_full": {"skin_head": 400_000, "body": 450_000, "hair": 200_000, "glasses": 60_000, "base": 30_000},
    "his_jk_q_doll": {"skin_head": 350_000, "hair": 180_000, "glasses": 60_000, "top": 150_000, "collar": 60_000,
                      "bow": 40_000, "skirt": 150_000, "socks": 60_000, "shoes": 60_000, "skin": 100_000,
                      "base": 30_000},
}
LABEL = {"his_jk_full": "全身版", "his_jk_q_doll": "Q版"}


def deviation(a, b, n=300_000):
    """Distances (mm) from points sampled on b to the surface of a."""
    pts, _ = trimesh.sample.sample_surface(b, n, seed=0)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(a.vertices.astype(np.float32)), o3d.core.Tensor(a.faces.astype(np.uint32)))
    return np.abs(sc.compute_signed_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy())


def slim(m, target):
    """Decimate to `target` triangles; returns (mesh, info). Falls back to the input if no attempt
    stays watertight and manifold."""
    import fast_simplification

    if len(m.faces) <= target:
        return m, {"faces": len(m.faces), "kept": True}
    for agg in (3, 5, 2, 7):
        v, f = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int64),
                                            target_count=target, agg=agg)
        out = trimesh.Trimesh(v, f, process=True)
        if not (out.is_watertight and nonmanifold_edges(out) == 0):
            out = drop_slivers(out)
        trimesh.repair.fix_normals(out)
        if out.is_watertight and nonmanifold_edges(out) == 0 and out.volume > 0:
            d = np.r_[deviation(m, out), deviation(out, m)]
            return out, {"faces": len(out.faces), "agg": agg, "dvol_pct": 100 * (out.volume / m.volume - 1),
                         "dev_p99_mm": float(np.percentile(d, 99)), "dev_max_mm": float(d.max())}
    return m, {"faces": len(m.faces), "failed": True}


def main(out="out"):
    dst = os.path.join(out, "deliver")
    tmp = os.path.join(dst, "tmp")
    shutil.rmtree(dst, ignore_errors=True)
    for name, parts in PARTS.items():
        src = os.path.join(out, name)
        if not os.path.exists(os.path.join(src, f"{name}.stl")):
            continue
        whole, info = slim(trimesh.load(os.path.join(src, f"{name}.stl")), WHOLE)
        print(name, info, flush=True)
        os.makedirs(os.path.join(tmp, name), exist_ok=True)
        files = []
        for p, target in parts.items():
            f = os.path.join(src, "parts", f"{name}_{p}.stl")
            if os.path.exists(f):
                m, info = slim(trimesh.load(f), target)
                print(f"  {p}", info, flush=True)
                files.append(os.path.join(tmp, name, f"{name}_{p}.stl"))
                m.export(files[-1])
        wf = os.path.join(tmp, f"{name}.stl")
        whole.export(wf)
        for kind, fs in (("单色整体", [wf]), ("多色分件", files)):
            z = os.path.join(dst, f"{LABEL[name]}_{kind}.zip")
            with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
                for f in fs:
                    zf.write(f, os.path.basename(f))
            print(f"{z}: {os.path.getsize(z) / 2 ** 20:.1f} MiB", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
