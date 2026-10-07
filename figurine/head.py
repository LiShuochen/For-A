"""The head: a star-shaped radial surface R(phi, theta) around a skull centre.

* skull: smooth two-part ellipsoid (anthropometric adult male proportions)
* face: fused MediaPipe landmarks, interpolated in angle space and blended into the skull
* detail: eyelid / mouth grooves and raised eyebrows, sized in *printed* millimetres
* glasses (separate colour group): rounded-rectangle rims + bridge + temples as a thin raised shell
* hair (separate colour group): girl's hair cap with curtain bangs, hairline and strand grooves

All internal geometry is in real-world millimetres in the head-local frame
(x = viewer's right, -y = toward the viewer, z = up, origin = between the pupils).
`Placement` maps head-local to printed world coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator

from . import face as F
from .sdf import Func, ellipsoid, rot

STEP = 0.25  # angular grid resolution, degrees
PHI = np.arange(-180.0, 180.0, STEP)
THETA = np.arange(-90.0, 90.0 + STEP / 2, STEP)


def unit(phi_deg, theta_deg):
    p, t = np.radians(phi_deg), np.radians(theta_deg)
    return np.stack([np.cos(t) * np.sin(p), -np.cos(t) * np.cos(p), np.sin(t)], axis=-1)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def polyline_dist(px, pz, line):
    """Distance from points (px, pz) to a 2D polyline (k,2); also returns the parameter along it (0..1)."""
    best = np.full(px.shape, np.inf)
    tpar = np.zeros(px.shape)
    seg_len = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    for k in range(len(line) - 1):
        a, b = line[k], line[k + 1]
        ab = b - a
        t = np.clip(((px - a[0]) * ab[0] + (pz - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
        d = np.hypot(px - a[0] - t * ab[0], pz - a[1] - t * ab[1])
        m = d < best
        best = np.where(m, d, best)
        tpar = np.where(m, (cum[k] + t * seg_len[k]) / cum[-1], tpar)
    return best, tpar


def rrect(px, pz, cx, cz, hw, hh, r):
    """Signed distance to a rounded rectangle in the x-z plane."""
    qx = np.abs(px - cx) - (hw - r)
    qz = np.abs(pz - cz) - (hh - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + np.minimum(np.maximum(qx, qz), 0) - r


@dataclass
class HeadParams:
    # skull (real mm). Centre is placed relative to the glabella of the fused face.
    half_width: float = 74.0
    half_length: float = 96.0
    up: float = 100.0
    down: float = 112.0
    low_width: float = 46.0  # half width / length at the bottom of the lower half
    low_length: float = 70.0
    centre_z: float = 14.0
    blend_deg: float = 6.0
    corr_deg: float = 24.0
    # printed-mm detail sizes
    groove_w: float = 0.38
    groove_d: float = 0.20
    brow_h: float = 0.28
    glasses_t: float = 0.45
    glasses_w: float = 0.75
    hair_min_print: float = 0.7
    # glasses (real mm, frontal plane)
    lens_cx: float = 33.0
    lens_cz: float = -4.0
    lens_hw: float = 25.0
    lens_hh: float = 17.0
    lens_r: float = 7.0
    temple_z: float = 6.0


@dataclass
class Placement:
    pos: np.ndarray  # world position of head-local origin
    s: float  # print mm per real mm
    R: np.ndarray = field(default_factory=lambda: np.eye(3))  # world rotation of local axes

    def to_world(self, p):
        return self.pos + self.s * (self.R @ np.asarray(p, float))

    def dir_world(self, d):
        return self.R @ np.asarray(d, float)


class Head:
    def __init__(self, fm: F.FaceModel, s: float, P: HeadParams = HeadParams()):
        self.P, self.s, self.fm = P, s, fm
        pts = fm.points
        glab = pts[9]
        self.C = np.array([0.0, glab[1] + P.half_length, P.centre_z])
        self._build_maps()

    # ---------------------------------------------------------------- angle-space maps
    def _skull_R(self, phi, theta):
        P = self.P
        d = unit(phi, theta)
        lowt = smoothstep(0, -70, theta)
        a = P.half_width + (P.low_width - P.half_width) * lowt
        b = P.half_length + (P.low_length - P.half_length) * lowt
        c = np.where(theta >= 0, P.up, P.down)
        return 1.0 / np.sqrt((d[..., 0] / a) ** 2 + (d[..., 1] / b) ** 2 + (d[..., 2] / c) ** 2)

    def _to_angles(self, pts):
        q = pts - self.C
        r = np.linalg.norm(q, axis=-1)
        phi = np.degrees(np.arctan2(q[..., 0], -q[..., 1]))
        theta = np.degrees(np.arcsin(np.clip(q[..., 2] / r, -1, 1)))
        return phi, theta, r

    def _build_maps(self):
        P, s = self.P, self.s
        T, Ph = np.meshgrid(THETA, PHI, indexing="ij")  # (nt, np)
        Rsk = self._skull_R(Ph, T)

        # --- face radius from landmarks (exclude iris points: they float in front of the eyeball)
        pts = self.fm.points[:468]
        fphi, fth, fr = self._to_angles(pts)
        interp = LinearNDInterpolator(np.stack([fphi, fth], 1), fr)  # linear: no overshoot spikes
        front = (np.abs(Ph) < 100) & (T > -80) & (T < 70)
        Rf = np.full(Ph.shape, np.nan)
        Rf[front] = interp(Ph[front], T[front])
        oval = self.fm.idx(F.FACE_OVAL)
        ophi, oth, _ = self._to_angles(oval)
        poly = Path(np.stack([ophi, oth], 1))
        inside = poly.contains_points(np.stack([Ph.ravel(), T.ravel()], 1)).reshape(Ph.shape)
        inside &= np.isfinite(Rf)
        # nan-aware smoothing removes the facets of the landmark triangulation (~1.5 mm real)
        sig = 0.8 / STEP
        num = ndimage.gaussian_filter(np.where(inside, Rf, 0.0), sig)
        den = ndimage.gaussian_filter(inside.astype(float), sig)
        Rf = np.where(inside, num / np.maximum(den, 1e-6), np.nan)
        # skull correction: carry the face/skull mismatch at the oval edge outward and let it decay,
        # so the skull rises to meet the face instead of leaving a mask-like step
        dist_in = ndimage.distance_transform_edt(inside) * STEP
        rim = inside & (dist_in <= 2.0)
        diff = np.where(rim, Rf - Rsk, 0.0)
        dist_out, (ii, jj) = ndimage.distance_transform_edt(~rim, return_indices=True)
        dist_out = dist_out * STEP
        corr = diff[ii, jj] * smoothstep(P.corr_deg, 0.0, dist_out)
        corr = ndimage.gaussian_filter(corr, 3.0 / STEP, mode=["nearest", "wrap"])
        Rsk = Rsk + corr
        w = smoothstep(0.0, P.blend_deg, dist_in)
        R = np.where(inside, w * np.nan_to_num(Rf) + (1 - w) * Rsk, Rsk)

        # --- surface points in head-local frame, for frontal-plane feature masks
        S = self.C + R[..., None] * unit(Ph, T)
        sx, sy, sz = S[..., 0], S[..., 1], S[..., 2]
        facing = unit(Ph, T)[..., 1] < -0.2
        disp = np.zeros_like(R)

        def groove(ids, depth_p, width_p, taper=True):
            line = self.fm.idx(ids)[:, [0, 2]]
            d, t = polyline_dist(sx, sz, line)
            wr = width_p / s
            amp = depth_p / s
            if taper:
                amp = amp * (0.35 + 0.65 * np.sin(np.pi * t))
            return -amp * np.exp(-(d / (0.5 * wr)) ** 2) * facing

        upper_r = [33, 246, 161, 160, 159, 158, 157, 173, 133]
        upper_l = [263, 466, 388, 387, 386, 385, 384, 398, 362]
        lower_r = [33, 7, 163, 144, 145, 153, 154, 155, 133]
        lower_l = [263, 249, 390, 373, 374, 380, 381, 382, 362]
        for ids in (upper_r, upper_l):
            disp += groove(ids, P.groove_d, P.groove_w)
        for ids in (lower_r, lower_l):
            disp += groove(ids, 0.5 * P.groove_d, 0.8 * P.groove_w)
        # slight recess inside the eye opening so the eyes read as open
        for ids in (F.RIGHT_EYE, F.LEFT_EYE):
            eye = Path(self.fm.idx(ids)[:, [0, 2]])
            m = eye.contains_points(np.stack([sx.ravel(), sz.ravel()], 1)).reshape(sx.shape) & facing
            m = ndimage.gaussian_filter(m.astype(float), 0.25 / STEP)
            disp -= (0.6 * P.groove_d / s) * m
        # mouth line: mid-line between upper and lower inner lip contours
        up_in = self.fm.idx([78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308])
        lo_in = self.fm.idx([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308])
        mouth = 0.5 * (up_in + lo_in)
        d, t = polyline_dist(sx, sz, mouth[:, [0, 2]])
        disp += -(P.groove_d * 1.1 / s) * (0.4 + 0.6 * np.sin(np.pi * t)) * np.exp(-(d / (0.5 * P.groove_w / s)) ** 2) * facing
        # nasolabial folds (faint) and alar creases
        for ids in ([129, 203, 206, 216], [358, 423, 426, 436]):
            disp += groove(ids, 0.35 * P.groove_d, 1.4 * P.groove_w)
        # eyebrows: raised band between upper and lower brow landmark rows
        for upper, lower in (([70, 63, 105, 66, 107], [46, 53, 52, 65, 55]),
                             ([300, 293, 334, 296, 336], [276, 283, 282, 295, 285])):
            mid = 0.5 * (self.fm.idx(upper) + self.fm.idx(lower))[:, [0, 2]]
            half = 0.5 * np.linalg.norm(self.fm.idx(upper) - self.fm.idx(lower), axis=1).mean()
            d, t = polyline_dist(sx, sz, mid)
            # landmark rows run outer -> inner; brows are fuller toward the nose
            hgt = (P.brow_h / s) * smoothstep(half * 1.15, half * 0.55, d) * (0.6 + 0.4 * t)
            disp += hgt * facing
        self.R = (R + disp).astype(np.float32)
        self.R_nodetail = R.astype(np.float32)

        # --- glasses mask (signed distance, real mm, negative on the frame)
        wr = max(P.glasses_w / s, 3.0) * 0.5
        g = np.full(R.shape, np.inf)
        for sgn in (-1, 1):
            g = np.minimum(g, np.abs(rrect(sx, sz, sgn * P.lens_cx, P.lens_cz, P.lens_hw, P.lens_hh, P.lens_r)) - wr)
        bridge_z = P.lens_cz + 0.45 * P.lens_hh
        inner = P.lens_cx - P.lens_hw
        bridge = np.maximum(np.abs(sz - bridge_z) - wr, np.abs(sx) - inner - wr)
        g = np.minimum(g, bridge)
        outer = P.lens_cx + P.lens_hw
        temple = np.maximum(np.abs(sz - (P.lens_cz + P.temple_z)) - wr * 0.9, outer - 2.0 - np.abs(sx))
        temple = np.maximum(temple, sy - (self.C[1] - 10.0))  # stop near the ears
        g = np.minimum(g, temple)
        g = np.where(unit(Ph, T)[..., 1] < 0.35, g, np.inf)
        self.G = np.minimum(g, 1e3).astype(np.float32)

        # --- hair: hairline elevation (deg) as a function of |phi|, curtain bangs at the front
        aphi = np.abs(Ph)
        knots = [0, 6, 14, 24, 36, 46, 60, 80, 105, 130, 155, 180]
        vals = [33, 27, 19.5, 17.0, 14.0, 6.0, -12, -26, -30, -52, -62, -64]
        hl = np.interp(aphi, knots, vals)
        # pointed locks along the bang edge: a sharpened triangle wave, one lock every 10 deg
        tri = 1.0 - np.abs(2.0 * np.mod(Ph / 10.0 + 0.5, 1.0) - 1.0)
        lock = tri ** 2.2
        bang = smoothstep(3, 9, aphi) * smoothstep(54, 38, aphi)
        hl = hl - 6.0 * lock * bang
        self.HL = hl.astype(np.float32)
        thick = 5.0 + 9.0 * smoothstep(35, 75, aphi) + 4.0 * smoothstep(10, 60, T)
        # keep side hair close to the cheeks/jaw so the face outline stays his
        thick = thick * (0.45 + 0.55 * np.maximum(smoothstep(-40, 5, T), smoothstep(100, 130, aphi)))
        edge = smoothstep(0.0, 10.0, T - hl)
        tmin = P.hair_min_print / s
        th = np.maximum(thick * (0.18 + 0.82 * edge), tmin)
        # soft strand clumps following meridians, phase-shifted per lock, plus a centre part
        clump = (0.5 + 0.5 * np.cos(np.radians(Ph) * 32.0 + 0.6 * np.sin(np.radians(T) * 3))) ** 2
        strands = -1.6 * clump * smoothstep(-70, -20, T)
        part = -2.2 * np.exp(-(Ph / 1.5) ** 2) * smoothstep(14, 30, T)
        self.RH = (np.maximum(self.R_nodetail, Rsk) + th + strands + part).astype(np.float32)

    # ---------------------------------------------------------------- field lookups
    def _lookup(self, M, phi, theta):
        fi = (phi + 180.0) / STEP
        ti = (theta + 90.0) / STEP
        Mp = np.concatenate([M[:, -2:], M, M[:, :2]], axis=1)
        return ndimage.map_coordinates(Mp, [ti.ravel(), fi.ravel() + 2], order=1, mode="nearest").reshape(phi.shape)

    def local_fields(self, X, Y, Z, which):
        q = np.stack([X - self.C[0], Y - self.C[1], Z - self.C[2]])
        rho = np.sqrt((q ** 2).sum(0)) + 1e-6
        phi = np.degrees(np.arctan2(q[0], -q[1]))
        theta = np.degrees(np.arcsin(np.clip(q[2] / rho, -1, 1)))
        if which == "skin":
            return rho - self._lookup(self.R, phi, theta)
        if which == "glasses":
            Rr = self._lookup(self.R, phi, theta)
            t = self.P.glasses_t / self.s
            radial = np.maximum(rho - (Rr + t), (Rr - 2 * t) - rho)
            mask = self._lookup(self.G, phi, theta)
            return np.maximum(radial, mask)
        if which == "hair":
            Rh = self._lookup(self.RH, phi, theta)
            Rr = self._lookup(self.R, phi, theta)
            hl = self._lookup(self.HL, phi, theta)
            line = np.radians(hl - theta) * rho
            return np.maximum(np.maximum(rho - Rh, (Rr - 3.0) - rho), line)
        raise ValueError(which)

    # ---------------------------------------------------------------- world nodes
    def bbox_local(self, which):
        R = self.RH if which == "hair" else self.R
        T, Ph = np.meshgrid(THETA, PHI, indexing="ij")
        S = self.C + R[..., None] * unit(Ph, T)
        return S.reshape(-1, 3).min(0) - 2, S.reshape(-1, 3).max(0) + 2

    def node(self, pl: Placement, which, name=None):
        lo, hi = self.bbox_local(which)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        wc = np.array([pl.to_world(c) for c in corners])
        Rt = pl.R.T
        px, py, pz = (float(v) for v in pl.pos)
        s = pl.s

        def f(x, y, z):
            X, Y, Zw = np.broadcast_arrays(x - px, y - py, z - pz)
            lx = (Rt[0, 0] * X + Rt[0, 1] * Y + Rt[0, 2] * Zw) / s
            ly = (Rt[1, 0] * X + Rt[1, 1] * Y + Rt[1, 2] * Zw) / s
            lz = (Rt[2, 0] * X + Rt[2, 1] * Y + Rt[2, 2] * Zw) / s
            return (s * self.local_fields(lx, ly, lz, which)).astype(np.float32)

        return Func(f, (wc.min(0), wc.max(0)), name or f"head_{which}")

    def ears(self, pl: Placement):
        out = []
        for sgn in (-1, 1):
            c_local = np.array([sgn * 71.0, self.C[1] + 2.0, -14.0])
            Rl = rot("x", -12) @ rot("z", sgn * 12)
            out.append(ellipsoid(pl.to_world(c_local), np.array([8.0, 17.0, 29.0]) * pl.s, R=pl.R @ Rl, name="ear"))
        return out

    # convenient anchor points (head-local, real mm)
    def anchor(self, name):
        C = self.C
        P = self.P
        table = {
            "vertex": C + np.array([0, -5.0, P.up + 8.0]),
            "neck_top": C + np.array([0, -12.0, -P.down + 36.0]),
            "tail_base_l": C + np.array([-62.0, 18.0, 58.0]),
            "tail_base_r": C + np.array([62.0, 18.0, 58.0]),
            "chin": self.fm.points[F.CHIN],
        }
        return table[name]
