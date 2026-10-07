"""Colour previews + summary for every built figure in out/ (run after figurine.build)."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import trimesh

from .colorize import body_colors, present

LUNTIMA = "assets/ref_models/beauty/luntima_student/scene.gltf"
BODY_OBJECTS = ["Object_0", "Object_9", "Object_3"]
DEFAULT_META = {"his_jk_full": {"k": (170.0 - 3.0) / 17.2, "base_h": 3.0}}


def _light(m, target=250_000):
    """Previews don't need print resolution: simplify big parts to keep memory low."""
    if len(m.faces) <= target:
        return m
    import fast_simplification

    v, fc = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int64),
                                         target_reduction=1 - target / len(m.faces), agg=5)
    return trimesh.Trimesh(v, fc, process=True)


def main(only=None):
    for d in sorted(glob.glob("out/*/")):
        name = os.path.basename(d.rstrip("/"))
        if only and name != only:
            continue
        rep_path = os.path.join(d, "report.json")
        if not os.path.exists(rep_path):
            continue
        rep = json.load(open(rep_path))
        meta = rep.get("meta") or DEFAULT_META.get(name)
        parts = {}
        for p in ("skin_head", "body", "hair", "glasses", "top", "collar", "bow", "skirt", "socks", "shoes",
                  "skin", "base"):
            f = os.path.join(d, "parts", f"{name}_{p}.stl")
            if os.path.exists(f):
                parts[p] = _light(trimesh.load(f))
        cols = None
        if "body" in parts and meta and "k" in meta:
            cols = body_colors(parts["body"], LUNTIMA, BODY_OBJECTS, meta["k"], meta["base_h"])
        present(name, d, parts, cols)
        print(name, "previews written", flush=True)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
