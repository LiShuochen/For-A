"""Full-body figurine: VRoid JK girl body + his likeness (face proportions, hair, glasses, brows)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import face as F
from . import sdf
from .features import FaceFrame, brows, glasses, short_hair
from .likeness import anime_landmarks, caricature_targets, his_ratios, warp
from .meshsdf import GridSDF, mesh_to_grid
from .vrmfig import figure_parts

SHINO = "assets/ref_models/full/sendagaya-shino/Sendagaya_Shino.vrm"
EXPRESSION = {"Fun": 0.8, "Blink": 0.45}  # gentle smile, narrow eyes (his look)
GLASSES = dict(rim_w=0.7, height=0.45, lens_w=0.78, lens_h=0.50, drop=0.06, corner=0.18)


def likeness_hook(state, gain=1.6):
    ratios = his_ratios(F.build())

    def hook(verts):
        an = anime_landmarks(verts)
        keys, src, dst = caricature_targets(an, ratios, gain=gain)
        skin = np.concatenate([v for k, v in verts.items() if "_SKIN" in k])
        mid, ipd = an[1], an[2]
        cr = skin[skin[:, 2] > mid[2] + 0.3 * ipd]
        r = 0.5 * (np.ptp(cr[:, 0]) / 2 + np.ptp(cr[:, 1]) / 2)
        hc = np.array([0.0, 0.5 * (cr[:, 1].min() + cr[:, 1].max()), cr[:, 2].max() - r])
        state.update(head_c=hc, head_r=r, chin=dict(zip(keys, dst))["chin"])
        return warp(verts, src, dst, hc, r, neck_z=an[0]["chin"][2] - 0.9 * ipd)

    return hook


def base_disc(radius, height=3.0):
    return sdf.cylinder_z((0, 0), radius, 0.0, height, round_=0.8, name="base")


def build(variant="his", height=164.0, h=0.15, base_h=3.0, path=SHINO, gain=1.6, hair_front_deg=32.0):
    """variant: 'original' (her, untouched) or 'his'. Returns ({group: node}, info)."""
    state = {}
    his = variant == "his"
    groups, info = figure_parts(path, height, z0=base_h, expression=EXPRESSION, keep_hair=not his,
                                subdivide=1, vertex_hook=likeness_hook(state, gain) if his else None)
    # per-group SDFs: skin is a closed body (no sheet thickening, which would leave a ruff under
    # the jaw); clothing and hair cards are thickened so they print
    nodes = {}
    for g, m in groups.items():
        f, gr = mesh_to_grid(m, h, min_thickness=0.0 if g == "skin" else 0.9)
        if g == "cloth":  # soften the jagged inner pleat edges at the skirt hem
            f = ndimage.gaussian_filter(f, 0.22 / h)
        nodes[g] = GridSDF(f, gr, name=g)
    feet_r = max(np.abs(groups["cloth"].vertices[groups["cloth"].vertices[:, 2] < base_h + 12][:, :2]).max(), 20)
    nodes["base"] = base_disc(feet_r * 1.25 + 4, base_h + 0.4)
    if his:
        ew = next(v for k, v in info["parts"].items() if "EyeWhite" in k)
        fr = FaceFrame(eye_l=ew[ew[:, 0] < 0].mean(0), eye_r=ew[ew[:, 0] > 0].mean(0),
                       head_c=state["head_c"], head_r=state["head_r"], chin=state["chin"])
        skin = nodes["skin"]
        nodes["glasses"] = glasses(skin, fr, **GLASSES)
        nodes["brows"] = brows(skin, fr, length=0.60, thick=0.11, lift=0.40, tilt=0.03, height=0.3)
        nodes["hair"] = short_hair(skin, fr, front_deg=hair_front_deg)
        info["frame"] = fr
    return nodes, info


def union(nodes):
    return sdf.Union(list(nodes.values()), k=0.0)
