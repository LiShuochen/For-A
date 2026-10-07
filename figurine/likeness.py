"""Make an anime face look like *him*: caricature-blend its proportions toward his landmarks.

Landmarks are measured on both faces in the same normalised frame (origin = midpoint of the
eyes, unit = eye distance). The anime face is then warped with a smooth 3D thin-plate-spline
so that its landmarks move part of the way (alpha) toward his: longer lower face, narrower
chin, stronger nose, wider mouth. The surfaces stay smooth anime surfaces, only the proportions
change, so it reads as "him, drawn as an anime character".
"""
from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import RBFInterpolator

from . import face as F

LEVELS = (0.5, 0.7, 0.85)  # jaw-outline sample heights, fraction of eye->chin distance
CANON = os.path.join(F.ROOT, "assets", "models", "canonical_face_model.obj")


def canonical_points():
    """MediaPipe's average face (468 landmarks) in our head frame (x right, -y front, z up), mm."""
    v = np.array([[float(t) for t in l.split()[1:4]] for l in open(CANON) if l.startswith("v ")])
    p = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1) * 10.0
    if p[33, 0] > p[263, 0]:  # make the subject's right eye (33) appear on the viewer's left
        p[:, 0] *= -1
    return p


def _oval_halfwidth(p, z):
    oval = p[F.FACE_OVAL]
    xs = []
    for i in range(len(oval)):
        a, b = oval[i], oval[(i + 1) % len(oval)]
        if (a[2] - z) * (b[2] - z) <= 0 and a[2] != b[2]:
            xs.append(a[0] + (z - a[2]) / (b[2] - a[2]) * (b[0] - a[0]))
    xs = np.array(xs)
    return 0.5 * (xs.max() - xs.min())


def measure(p):
    """Proportions of a 468-landmark face in units of eye distance."""
    el, er = p[F.RIGHT_EYE].mean(0), p[F.LEFT_EYE].mean(0)
    mid, ipd = 0.5 * (el + er), np.linalg.norm(er - el)
    face_len = mid[2] - p[F.CHIN][2]
    m = {
        "face_len": face_len / ipd,
        "nose_drop": (mid[2] - p[1][2]) / ipd,
        "nose_proj": (mid[1] - p[1][1]) / ipd,
        "mouth_drop": (mid[2] - 0.5 * (p[13][2] + p[14][2])) / ipd,
        "mouth_w": abs(p[291][0] - p[61][0]) / ipd,
        "eye_w": 0.5 * (abs(p[33][0] - p[133][0]) + abs(p[263][0] - p[362][0])) / ipd,
        "eye_h": 0.5 * ((p[159][2] - p[145][2]) + (p[386][2] - p[374][2])) / ipd,
        "cheek_w": abs(p[454][0] - p[234][0]) / ipd,
        "brow_h": (0.5 * (p[105][2] + p[334][2]) - mid[2]) / ipd,
    }
    for f in LEVELS:
        m[f"jaw_{f}"] = _oval_halfwidth(p, mid[2] - f * face_len) / ipd
    return m


def his_ratios(fm: F.FaceModel):
    """How his face differs from the average face (ratios), e.g. face_len 1.08 = 8% longer."""
    h, c = measure(fm.points), measure(canonical_points())
    return {k: h[k] / c[k] for k in h}


def _outline_halfwidth(pts, z, band):
    sel = np.abs(pts[:, 2] - z) < band
    if not sel.any():
        return None
    q = pts[sel]
    i = np.abs(q[:, 0]).argmax()
    return abs(q[i, 0]), q[i, 1]


def anime_landmarks(parts):
    """parts: {part name: (n,3) print-space vertices} of a posed VRoid head."""
    ew = next(v for k, v in parts.items() if "EyeWhite" in k)
    face = next(v for k, v in parts.items() if "Face.baked/7" in k or ("Face_00_SKIN" in k and len(v) > 400))
    L = {}
    for side, sgn in (("l", -1), ("r", 1)):
        e = ew[np.sign(ew[:, 0]) == sgn]
        L[f"eye_{side}"] = e.mean(0)
        L[f"eye_in_{side}"] = e[np.abs(e[:, 0]).argmin()]
        L[f"eye_out_{side}"] = e[np.abs(e[:, 0]).argmax()]
    mid = 0.5 * (L["eye_l"] + L["eye_r"])
    ipd = np.linalg.norm(L["eye_r"] - L["eye_l"])
    centre = face[np.abs(face[:, 0]) < 0.06 * ipd]
    front = centre[centre[:, 1] < mid[1] + 0.2 * ipd]
    L["nose"] = front[front[:, 1].argmin()]
    L["chin"] = front[front[:, 2].argmin()]
    # mouth: the smile line is the most recessed centre-line point between nose and chin
    m = front[(front[:, 2] < L["nose"][2] - 0.1 * ipd) & (front[:, 2] > L["chin"][2] + 0.15 * ipd)]
    L["mouth"] = m[m[:, 1].argmax()]
    mouth = next((v for k, v in parts.items() if "FaceMouth" in k), None)
    if mouth is not None:
        L["mouth_l"] = mouth[mouth[:, 0].argmin()]
        L["mouth_r"] = mouth[mouth[:, 0].argmax()]
    span = mid[2] - L["chin"][2]
    fr = face[face[:, 1] < mid[1] + 0.45 * ipd]
    for f in LEVELS:
        z = mid[2] - f * span
        hw, y = _outline_halfwidth(fr, z, 0.04 * ipd)
        L[f"jaw_l_{f}"] = np.array([-hw, y, z])
        L[f"jaw_r_{f}"] = np.array([hw, y, z])
    return L, mid, ipd


def caricature_targets(anime, ratios, gain=1.6, lo=0.8, hi=1.3):
    """Move anime landmarks by his ratios (amplified by `gain`). Returns (keys, src, dst)."""
    A, mid, ipd = anime

    def r(k):
        return float(np.clip(ratios[k] ** gain, lo, hi))

    # MediaPipe's chin is unreliable and his face reads long (high forehead): never shorten it
    ratios = dict(ratios, face_len=max(ratios["face_len"], 1.0))

    T = {k: v.copy() for k, v in A.items()}
    flen_a = mid[2] - A["chin"][2]
    flen = flen_a * r("face_len")
    T["chin"][2] = mid[2] - flen
    T["nose"][2] = mid[2] - (mid[2] - A["nose"][2]) * r("nose_drop")
    T["nose"][1] = mid[1] - (mid[1] - A["nose"][1]) * r("nose_proj")
    T["mouth"][2] = mid[2] - (mid[2] - A["mouth"][2]) * r("mouth_drop")
    for side in ("l", "r"):
        k = f"mouth_{side}"
        if k in A:
            T[k][0] = A[k][0] * r("mouth_w")
            T[k][2] = A[k][2] + (T["mouth"][2] - A["mouth"][2])
    for f in LEVELS:
        for side in ("l", "r"):
            k = f"jaw_{side}_{f}"
            T[k][0] = A[k][0] * r(f"jaw_{f}")
            T[k][2] = mid[2] - f * flen
    keys = list(A)
    return keys, np.array([A[k] for k in keys]), np.array([T[k] for k in keys])


def warp(verts, src, dst, head_c, head_r, neck_z):
    """Thin-plate-spline warp of every vertex near the head; anchors keep cranium and body fixed."""
    # anchors: back/top of the cranium and a ring around the neck base stay put
    u = np.random.default_rng(0).normal(size=(80, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    cran = head_c + 1.05 * head_r * u[u[:, 1] > -0.1]
    t = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    neck = np.stack([0.55 * head_r * np.cos(t), 0.55 * head_r * np.sin(t) + head_c[1], np.full_like(t, neck_z)], 1)
    far = np.stack([1.8 * head_r * np.cos(t), 1.8 * head_r * np.sin(t) + head_c[1], np.full_like(t, head_c[2])], 1)
    S = np.concatenate([src, cran, neck, far])
    D = np.concatenate([dst - src, np.zeros((len(cran) + len(neck) + len(far), 3))])
    rbf = RBFInterpolator(S, D, kernel="thin_plate_spline", smoothing=1e-3)
    out = {}
    for k, V in verts.items():
        r = np.linalg.norm(V - head_c, axis=1)
        w = np.clip((2.0 * head_r - r) / (0.7 * head_r), 0, 1)  # fade out away from the head
        w *= np.clip((V[:, 2] - (neck_z - 0.2 * head_r)) / (0.3 * head_r), 0, 1)
        if not (w > 0).any():
            out[k] = V
            continue
        Vn = V.copy()
        sel = w > 0
        Vn[sel] += w[sel, None] * rbf(V[sel])
        out[k] = Vn
    return out
