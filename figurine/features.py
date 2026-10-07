"""His recognisable features, applied on top of any base model given as an SDF grid.

From the photos: thin dark rectangular metal glasses, very short hair with a high, receding
M-shaped hairline, straight fairly thick eyebrows, narrow single-lid eyes, a slight smirk,
and a long face with a narrow chin.

All features are thin shells that follow the base model's surface (so they print as raised
detail, never floating), each a separate colour group for AMS multi-colour printing.
Model frame: Z up, face looking toward -Y.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .meshsdf import GridSDF
from .sdf import BIG, Func


@dataclass
class FaceFrame:
    eye_l: np.ndarray  # viewer's-left eye centre (world, on the face surface)
    eye_r: np.ndarray
    head_c: np.ndarray  # centre of the cranium
    head_r: float  # approx cranium radius
    chin: np.ndarray

    @property
    def ipd(self):
        return float(np.linalg.norm(self.eye_r - self.eye_l))

    @property
    def mid(self):
        return 0.5 * (self.eye_l + self.eye_r)


def sample(g: GridSDF, x, y, z):
    """Trilinear sample of a GridSDF at broadcastable world coordinates."""
    X, Y, Z = np.broadcast_arrays(x, y, z)
    G = g.grid
    c = [(X.ravel() - G.lo[0]) / G.h, (Y.ravel() - G.lo[1]) / G.h, (Z.ravel() - G.lo[2]) / G.h]
    return ndimage.map_coordinates(g.field, c, order=1, mode="constant", cval=float(BIG)).reshape(X.shape)


def _rrect(px, pz, cx, cz, hw, hh, r):
    qx = np.abs(px - cx) - (hw - r)
    qz = np.abs(pz - cz) - (hh - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0) - r


def _seg_dist(px, pz, a, b):
    ab = b - a
    t = np.clip(((px - a[0]) * ab[0] + (pz - a[1]) * ab[1]) / (ab @ ab), 0, 1)
    return np.hypot(px - a[0] - t * ab[0], pz - a[1] - t * ab[1]), t


def surface_shell(base: GridSDF, mask_fn, box, height, embed=0.35, name="feature"):
    """Raised shell of `height` mm on the base surface wherever mask_fn(x,y,z) < 0 (mm-ish SDF)."""

    def f(x, y, z):
        d = sample(base, x, y, z)
        m = mask_fn(*np.broadcast_arrays(x, y, z))
        return np.maximum(np.maximum(d - height, -d - embed), m)

    return Func(f, box, name)


def glasses(base: GridSDF, fr: FaceFrame, rim_w=0.8, height=0.45, lens_w=0.92, lens_h=0.62,
            drop=0.10, corner=0.22, temples=True, name="glasses"):
    """His rectangular glasses. Sizes are fractions of the model's eye distance (ipd)."""
    ipd = fr.ipd
    hw, hh = 0.5 * lens_w * ipd, 0.5 * lens_h * ipd
    r = corner * min(hw, hh) * 2
    cz = fr.mid[2] - drop * ipd
    xl, xr = fr.eye_l[0], fr.eye_r[0]
    y_face = fr.mid[1]
    w2 = 0.5 * rim_w
    inner = min(abs(xl), abs(xr)) - hw if xl * xr < 0 else 0.0
    outer = max(abs(xl), abs(xr)) + hw
    bz = cz + 0.42 * hh
    tz = cz + 0.55 * hh
    y_back = fr.head_c[1] + 0.15 * fr.head_r

    def mask(x, y, z):
        g = np.minimum(np.abs(_rrect(x, z, xl, cz, hw, hh, r)), np.abs(_rrect(x, z, xr, cz, hw, hh, r))) - w2
        bridge = np.maximum(np.abs(z - bz) - w2, np.abs(x) - inner - w2)
        g = np.minimum(g, bridge)
        front = y < y_face + 0.55 * ipd  # only the front of the face for rims
        g = np.where(front, g, BIG)
        if temples:
            t = np.maximum(np.abs(z - tz) - w2 * 0.85, outer - w2 - np.abs(x))
            t = np.maximum(t, y - y_back)
            g = np.minimum(g, t)
        return g

    pad = 3.0
    lo = np.array([-outer - fr.head_r * 0.6, y_face - fr.head_r, cz - hh - pad])
    hi = np.array([outer + fr.head_r * 0.6, y_back + pad, cz + hh + pad])
    return surface_shell(base, mask, (lo, hi), height, name=name)


def brows(base: GridSDF, fr: FaceFrame, length=0.62, thick=0.11, lift=0.42, tilt=0.06, height=0.3,
          name="brows"):
    """Straight, fairly thick eyebrows (his), above each eye. Fractions of ipd."""
    ipd = fr.ipd
    segs = []
    for e, sgn in ((fr.eye_l, -1), (fr.eye_r, 1)):
        c = np.array([e[0], e[2] + lift * ipd])
        a = c + np.array([-sgn * 0.5 * length * ipd * 0.85, -tilt * ipd * 0.2])  # inner end, a bit lower
        b = c + np.array([sgn * 0.5 * length * ipd, -tilt * ipd])
        segs.append((a, b))
    y_face = fr.mid[1]

    def mask(x, y, z):
        m = BIG
        for a, b in segs:
            d, t = _seg_dist(x, z, a, b)
            w = thick * ipd * (1.0 - 0.35 * t)  # fuller toward the nose
            m = np.minimum(m, d - 0.5 * w)
        return np.where(y < y_face + 0.4 * ipd, m, BIG)

    lo = np.array([min(fr.eye_l[0], fr.eye_r[0]) - ipd, y_face - fr.head_r, fr.mid[2]])
    hi = np.array([max(fr.eye_l[0], fr.eye_r[0]) + ipd, y_face + fr.head_r, fr.mid[2] + ipd])
    return surface_shell(base, mask, (lo, hi), height, name=name)


def smile_eyes(base: GridSDF, fr: FaceFrame, width=0.42, arch=0.07, line=0.06, depth=0.3, name="eyes"):
    """Narrow, gently smiling eyes (his look) as engraved arcs; returned as a cutting node."""
    ipd = fr.ipd
    arcs = []
    for e in (fr.eye_l, fr.eye_r):
        xs = np.linspace(-0.5, 0.5, 9) * width * ipd
        zs = e[2] + arch * ipd * (1 - (2 * xs / (width * ipd)) ** 2) - 0.5 * arch * ipd
        arcs.append(np.stack([e[0] + xs, zs], 1))
    y_face = fr.mid[1]

    def f(x, y, z):
        X, Y, Z = np.broadcast_arrays(x, y, z)
        m = np.full(X.shape, BIG, dtype=np.float64)
        for arc in arcs:
            for k in range(len(arc) - 1):
                d, _ = _seg_dist(X, Z, arc[k], arc[k + 1])
                m = np.minimum(m, d - 0.5 * line * ipd)
        d = sample(base, X, Y, Z)
        cut = np.maximum(m, np.abs(d) - depth)  # groove of `depth` around the surface
        return np.where(Y < y_face + 0.3 * ipd, cut, BIG)

    lo = np.array([min(fr.eye_l[0], fr.eye_r[0]) - ipd, y_face - fr.head_r, fr.mid[2] - 0.4 * ipd])
    hi = np.array([max(fr.eye_l[0], fr.eye_r[0]) + ipd, y_face + 0.5 * fr.head_r, fr.mid[2] + 0.4 * ipd])
    return Func(f, (lo, hi), name)


def short_hair(scalp: GridSDF, fr: FaceFrame, thickness=1.3, front_deg=32.0, temple_deg=22.0,
               temple_phi=40.0, nape_deg=-40.0, texture=0.22, strands=110, edge_deg=5.0, name="hair"):
    """His very short hair with a high, receding M-shaped hairline, as a shell over `scalp`.

    Angles: elevation (deg) of the hairline seen from the cranium centre; front_deg at the
    centre of the forehead, temple corners recede by 6 deg at phi = +-temple_phi.
    Thickness is fullest on the crown and tapers to ~35% at the hairline; fine grooves running
    front-to-back read as a short crop.
    """
    c = fr.head_c

    def hairline(phi):
        a = np.abs(phi)
        knots = [0, 12, temple_phi, 60, 90, 120, 180]
        vals = [front_deg, front_deg + 2.0, front_deg + 6.0, temple_deg, 12.0, -5.0, nape_deg]
        return np.interp(a, knots, vals)

    def f(x, y, z):
        X, Y, Z = np.broadcast_arrays(x - c[0], y - c[1], z - c[2])
        rho = np.sqrt(X * X + Y * Y + Z * Z) + 1e-6
        phi = np.degrees(np.arctan2(X, -Y))
        th = np.degrees(np.arcsin(np.clip(Z / rho, -1, 1)))
        d = sample(scalp, x, y, z)
        hl = hairline(phi)
        above = th - hl
        crown = 0.75 + 0.25 * np.clip(th / 60.0, 0, 1)
        edge = 0.35 + 0.65 * np.clip(above / edge_deg, 0, 1)
        grooves = texture * (0.5 + 0.5 * np.cos(np.radians(phi) * strands)) ** 2
        t = thickness * crown * edge - grooves * edge
        shell = np.maximum(d - t, -d - 0.4)
        line = np.radians(-above) * rho
        return np.maximum(shell, line)

    R = fr.head_r * 1.35
    return Func(f, (c - R, c + R), name)
