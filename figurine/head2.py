"""Head v2: his real face proportions on a clean, smooth, figure-style head.

What makes it him (from studying the photos): big domed forehead, M-shaped receding hairline
with a ~1 cm crew cut (shorter on the sides/back), thin black rectangular glasses, narrow
single-lid eyes that close into a smile, low straight brows, straight fairly long nose with a
rounded tip, wide thin-lipped mouth with a closed one-sided smile, rounded (not pointed) chin,
large ears that stand out from the head.

"Beautify a little": full symmetry, 15% toward the average face, slightly slimmer lower face
(chin stays round), smooth skin, eyes opened a touch.

Face surface: the fused 468 landmarks *with MediaPipe's own triangle topology*, Loop-subdivided,
then ray-cast from the skull centre into a radial map R(phi, theta) and blended into the skull.
All geometry is real-world mm in the head-local frame (x = viewer's right, -y = toward the
viewer, z = up, origin = between the pupils); detail sizes are given in printed mm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import trimesh
from matplotlib.path import Path
from scipy import ndimage

from . import face as F
from .head import PHI, STEP, THETA, Placement, polyline_dist, rrect, smoothstep, unit
from .likeness import CANON, canonical_points
from .sdf import Func


def canonical_faces():
    out = []
    for line in open(CANON):
        if line.startswith("f "):
            out.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(out)


def _umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum()
    return (s * (R @ a.T)).T + md


def _mirror_partner(p):
    m = p * np.array([-1, 1, 1])
    return ((p[:, None, :] - m[None, :, :]) ** 2).sum(-1).argmin(1)


@dataclass
class Head2Params:
    # beautify
    toward_avg: float = 0.15
    symmetry: float = 1.0
    jaw_scale: float = 1.03  # >1 widens the lower face (checked against photo overlays)
    eye_open: float = 1.85  # open the narrow eyes so the irises read (his look when facing the camera)
    eye_scale: float = 1.12  # slightly larger eyes (beautify, more expressive)
    smile_lift_r: float = 3.2  # mm: his right mouth corner (viewer's left) rises more
    smile_lift_l: float = 1.8
    # skull (real mm)
    half_width: float = 80.5  # fitted to silhouettes, trimmed from photo overlays
    half_length: float = 97.0
    up: float = 90.0  # round, broad crown
    down: float = 132.0
    low_width: float = 48.0
    low_length: float = 70.0
    centre_z: float = 16.0
    crown_n: float = 1.9  # superellipsoid exponent of the crown (2 = ellipsoid, higher = rounder/fuller)
    blend_deg: float = 5.0
    corr_deg: float = 24.0
    # printed-mm details
    groove_w: float = 0.40
    groove_d: float = 0.22
    brow_h: float = 0.26
    glasses_t: float = 0.5
    glasses_w: float = 0.5
    # glasses (real mm, frontal plane): thin black rectangular frames
    lens_cx: float = 34.0
    lens_cz: float = -5.0
    lens_hw: float = 28.0
    lens_hh: float = 19.5
    lens_r: float = 5.5
    # hair (real mm)
    hair_top: float = 4.5
    hair_side: float = 1.8
    hair_stubble: float = 0.22


class Head2:
    def __init__(self, fm: F.FaceModel, s: float, P: Head2Params = Head2Params()):
        self.P, self.s = P, s
        self.pts = self._beautify(fm.points[:468].copy())
        glab = self.pts[9]
        self.C = np.array([0.0, glab[1] + P.half_length, P.centre_z])
        self.mesh = self._face_mesh()
        self.neck_r, self.neck_bottom = 50.0, -235.0
        self._build_maps()

    # ---------------------------------------------------------------- landmarks
    def _beautify(self, p):
        P = self.P
        can = _umeyama(canonical_points(), p)
        p = p + P.toward_avg * (can - p)
        part = _mirror_partner(p)
        p = (1 - P.symmetry) * p + P.symmetry * 0.5 * (p + p[part] * np.array([-1, 1, 1]))
        mouth_z = 0.5 * (p[13, 2] + p[14, 2])
        chin_z = p[F.CHIN, 2]
        lower = smoothstep(p[1, 2], chin_z, p[:, 2])  # slim from the nose down; chin stays round
        p[:, 0] *= 1.0 + (P.jaw_scale - 1.0) * lower
        # open the eyes a little around each eye's centre line
        for ids in (F.RIGHT_EYE, F.LEFT_EYE):
            ids = np.array(ids)
            c = p[ids].mean(0)
            # enlarge the whole eye region smoothly, then open the lids
            d = np.linalg.norm((p - c)[:, [0, 2]] / np.array([24.0, 14.0]), axis=1)
            w = np.exp(-d ** 2)
            p[:, 0] = c[0] + (p[:, 0] - c[0]) * (1 + (P.eye_scale - 1) * w)
            p[:, 2] = c[2] + (p[:, 2] - c[2]) * (1 + (P.eye_scale - 1) * w)
            zc = p[ids, 2].mean()
            p[ids, 2] = zc + (p[ids, 2] - zc) * P.eye_open
        # his nose: rounder tip, wider wings
        nc = p[1]
        d = np.linalg.norm((p - nc)[:, [0, 2]] / np.array([22.0, 16.0]), axis=1)
        p[:, 0] = nc[0] + (p[:, 0] - nc[0]) * (1.0 + 0.12 * np.exp(-d ** 2))
        # his mouth sits a little lower than the landmark fit
        mc0 = 0.5 * (p[13] + p[14])
        d = np.linalg.norm((p - mc0)[:, [0, 2]] / np.array([30.0, 12.0]), axis=1)
        p[:, 2] -= 1.5 * np.exp(-d ** 2)
        # his mouth is wide: stretch the mouth region sideways about its centre
        mc0 = 0.5 * (p[13] + p[14])
        d = np.linalg.norm((p - mc0)[:, [0, 2]] / np.array([34.0, 14.0]), axis=1)
        p[:, 0] = mc0[0] + (p[:, 0] - mc0[0]) * (1.0 + 0.09 * np.exp(-d ** 2))
        # thin lips: MediaPipe over-estimates lip depth, so push the mouth area back
        mc = 0.5 * (p[13] + p[14])
        d = np.linalg.norm((p - mc)[:, [0, 2]] / np.array([30.0, 16.0]), axis=1)
        p[:, 1] += 5.5 * np.exp(-d ** 2)
        # closed-mouth smile: lift the corners (more on his right), fading out over ~14 mm
        for idx, lift in ((61, P.smile_lift_r), (291, P.smile_lift_l)):
            d = np.linalg.norm(p - p[idx], axis=1)
            w = np.exp(-(d / 11.0) ** 2)
            p[:, 2] += lift * w
            p[:, 1] += 0.6 * lift * w  # corners tuck in slightly
        return p

    def _face_mesh(self):
        v, f = self.pts, canonical_faces()
        for _ in range(3):
            v, f = trimesh.remesh.subdivide_loop(v, f)
        return trimesh.Trimesh(v, f, process=False)

    # ---------------------------------------------------------------- radial maps
    def _skull_R(self, phi, theta):
        P = self.P
        d = unit(phi, theta)
        lowt = smoothstep(0, -70, theta)
        a = P.half_width + (P.low_width - P.half_width) * lowt
        b = P.half_length + (P.low_length - P.half_length) * lowt
        c = np.where(theta >= 0, P.up, P.down)
        n = 2.0 + (P.crown_n - 2.0) * smoothstep(0, 40, theta)  # fuller, rounder crown
        return 1.0 / (np.abs(d[..., 0] / a) ** n + np.abs(d[..., 1] / b) ** n + np.abs(d[..., 2] / c) ** n) ** (1 / n)

    def _to_angles(self, pts):
        q = pts - self.C
        r = np.linalg.norm(q, axis=-1)
        return (np.degrees(np.arctan2(q[..., 0], -q[..., 1])),
                np.degrees(np.arcsin(np.clip(q[..., 2] / r, -1, 1))), r)

    def _raycast_face(self, Ph, T):
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(self.mesh.vertices.astype(np.float32)),
                         o3d.core.Tensor(self.mesh.faces.astype(np.uint32)))
        d = unit(Ph, T).reshape(-1, 3)
        rays = np.concatenate([np.broadcast_to(self.C, d.shape), d], 1).astype(np.float32)
        t = sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(Ph.shape)
        return np.where(np.isfinite(t), t, np.nan)

    def _build_maps(self):
        P, s = self.P, self.s
        T, Ph = np.meshgrid(THETA, PHI, indexing="ij")
        Rsk = self._skull_R(Ph, T)
        Rf = self._raycast_face(Ph, T)
        inside = np.isfinite(Rf) & (np.abs(Ph) < 110)
        dist_in = ndimage.distance_transform_edt(inside) * STEP
        rim = inside & (dist_in <= 2.0)
        diff = np.where(rim, np.nan_to_num(Rf) - Rsk, 0.0)
        dist_out, (ii, jj) = ndimage.distance_transform_edt(~rim, return_indices=True)
        corr = diff[ii, jj] * smoothstep(P.corr_deg, 0.0, dist_out * STEP)
        corr = ndimage.gaussian_filter(corr, 3.0 / STEP, mode=["nearest", "wrap"])
        Rsk = Rsk + corr
        w = smoothstep(0.0, P.blend_deg, dist_in)
        R = np.where(inside, w * np.nan_to_num(Rf) + (1 - w) * Rsk, Rsk)
        R = ndimage.gaussian_filter(R, 0.35 / STEP, mode=["nearest", "wrap"])  # figure-smooth skin
        self.Rskull = Rsk

        # frontal-plane coordinates of the surface, for feature masks
        S = self.C + R[..., None] * unit(Ph, T)
        sx, sy, sz = S[..., 0], S[..., 1], S[..., 2]
        facing = smoothstep(-0.15, -0.35, unit(Ph, T)[..., 1])
        p = self.pts
        disp = np.zeros_like(R)

        def line_feature(points, depth_p, width_p, taper=0.6):
            d, t = polyline_dist(sx, sz, points[:, [0, 2]])
            amp = (depth_p / s) * ((1 - taper) + taper * np.sin(np.pi * t))
            return amp * np.exp(-(d / (0.5 * width_p / s)) ** 2) * facing

        def soft_mask(poly, blur_p=0.12):
            m = Path(poly).contains_points(np.stack([sx.ravel(), sz.ravel()], 1)).reshape(sx.shape)
            return ndimage.gaussian_filter(m.astype(float), (blur_p / s) / (R.mean() * np.radians(STEP)))

        # eyes: narrow single-lid eyes with a sculpted eyeball, iris, pupil and catch-light so the
        # print has a gaze even without paint; lid margins have a little thickness
        up_r = p[[33, 246, 161, 160, 159, 158, 157, 173, 133]]
        up_l = p[[263, 466, 388, 387, 386, 385, 384, 398, 362]]
        lo_r = p[[33, 7, 163, 144, 145, 153, 154, 155, 133]]
        lo_l = p[[263, 249, 390, 373, 374, 380, 381, 382, 362]]
        for up, lo in ((up_r, lo_r), (up_l, lo_l)):
            poly = np.concatenate([up[:, [0, 2]], lo[::-1, [0, 2]]])
            m = soft_mask(poly) * facing
            ex = 0.5 * (up[0, 0] + up[-1, 0])
            ew = abs(up[0, 0] - up[-1, 0])
            z_up, z_lo = up[len(up) // 2, 2], lo[len(lo) // 2, 2]
            ez = 0.5 * (z_up + z_lo)
            # eyeball: set back behind the lids, gently domed
            r_e = np.hypot((sx - ex) / (0.5 * ew), (sz - ez) / (0.5 * ew))
            disp += m * (-0.36 + 0.26 * np.clip(1 - r_e ** 2, 0, 1)) / s  # a real ball sitting behind the lids
            # iris (partly hidden by the narrow lids), pupil, and a catch-light
            ri = 0.27 * ew
            iz = z_lo + 0.58 * (z_up - z_lo)
            r_i = np.hypot(sx - ex, sz - iz)
            iris = smoothstep(ri + 0.05 / s, ri - 0.05 / s, r_i)
            pupil = smoothstep(0.45 * ri + 0.04 / s, 0.45 * ri - 0.04 / s, r_i)
            hl = np.exp(-(np.hypot(sx - (ex - 0.32 * ri), sz - (iz + 0.30 * ri)) / (0.17 * ri)) ** 2)
            disp += m * (-0.12 * iris - 0.14 * pupil + 0.16 * hl) / s
            # lid margins: a crisp line where lid meets eyeball, and a little lid thickness above it
            disp -= line_feature(up, 0.15, 0.22, taper=0.3)
            disp -= line_feature(lo, 0.05, 0.18, taper=0.5)
            lid = up.copy()
            lid[:, 2] += 0.22 / s
            disp += line_feature(lid, 0.14, 0.50, taper=0.6)  # upper-lid thickness overhangs the ball
            # smile: the cheek just under the lower lid puffs up a little
            under = lo.copy()
            under[:, 2] -= 0.18 * ew
            disp += line_feature(under, 0.08, 1.4, taper=0.8)
        # mouth: wide mouth, thin upper lip with a crisp border, slightly fuller lower lip,
        # closed smile (corners already lifted in the landmarks)
        up_out = p[[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]]
        lo_out = p[[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]]
        up_in = p[[78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]]
        lo_in = p[[78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]]
        def smooth_line(pts, n=40):
            t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            u = np.linspace(0, t[-1], n)
            out = np.stack([np.interp(u, t, pts[:, k]) for k in range(3)], 1)
            for _ in range(3):  # light Laplacian smoothing keeps the ends fixed
                out[1:-1] = 0.5 * out[1:-1] + 0.25 * (out[:-2] + out[2:])
            return out

        up_out, lo_out = smooth_line(up_out), smooth_line(lo_out)
        mouth = smooth_line(np.concatenate([p[[61]], 0.5 * (up_in + lo_in), p[[291]]]))
        upper_lip = soft_mask(np.concatenate([up_out[:, [0, 2]], mouth[::-1, [0, 2]]]), 0.08) * facing
        lower_lip = soft_mask(np.concatenate([mouth[:, [0, 2]], lo_out[::-1, [0, 2]]]), 0.10) * facing
        lip_mid = np.exp(-((sx - p[0, 0]) / (0.55 * abs(p[291, 0] - p[61, 0]))) ** 2)
        disp += (0.05 * upper_lip + 0.12 * lower_lip * (0.5 + 0.5 * lip_mid)) / s
        disp += line_feature(up_out, 0.05, 0.20, taper=0.7)  # vermilion border / cupid's bow
        disp -= line_feature(mouth, 0.24, 0.30, taper=0.5)  # where the lips meet
        chin_fold = lo_out.copy()
        chin_fold[:, 2] -= 0.35 * abs(lo_out[len(lo_out) // 2, 2] - mouth[len(mouth) // 2, 2])
        disp -= line_feature(chin_fold[6:-6], 0.06, 0.8, taper=0.9)  # soft shadow under the lower lip
        for c in (61, 291):  # little dimples at the corners
            d = np.hypot(sx - p[c, 0], sz - p[c, 2])
            disp -= (0.10 / s) * np.exp(-(d / (0.30 / s)) ** 2) * facing
        # faint smile lines from the nose wings
        for ids in ([129, 203, 206, 216], [358, 423, 426, 436]):
            disp -= line_feature(p[ids], 0.03, 0.6)
        # nostrils: small soft dents beside the tip
        for c in (98, 327):
            d = np.hypot(sx - p[c, 0], sz - p[c, 2])
            disp -= (0.10 / s) * np.exp(-(d / (0.4 / s)) ** 2) * facing
        # brows: straight, low, medium thick, fuller toward the nose
        for upper, lower in (([70, 63, 105, 66, 107], [46, 53, 52, 65, 55]),
                             ([300, 293, 334, 296, 336], [276, 283, 282, 295, 285])):
            mid = 0.5 * (p[upper] + p[lower])
            mid[:, 2] -= 1.0  # sit a touch lower, like his
            half = 0.5 * np.linalg.norm(p[upper] - p[lower], axis=1).mean()
            d, t = polyline_dist(sx, sz, mid[:, [0, 2]])
            disp += (P.brow_h / s) * smoothstep(half * 1.2, half * 0.5, d) * (0.55 + 0.45 * t) * facing
        self.R = (R + disp).astype(np.float32)
        self.R_smooth = R.astype(np.float32)

        # glasses: frame rests on a heavily smoothed face so it bridges the eye sockets
        Rg = np.maximum(ndimage.gaussian_filter(R, 2.5 / STEP, mode=["nearest", "wrap"]), R)
        self.RG = Rg.astype(np.float32)
        wr = max(P.glasses_w / s, 2.0) * 0.5
        g = np.full(R.shape, np.inf)
        for sg in (-1, 1):
            g = np.minimum(g, np.abs(rrect(sx, sz, sg * P.lens_cx, P.lens_cz, P.lens_hw, P.lens_hh, P.lens_r)) - wr)
        bz = P.lens_cz + 0.55 * P.lens_hh
        inner = P.lens_cx - P.lens_hw
        g = np.minimum(g, np.maximum(np.abs(sz - bz) - wr * 0.9, np.abs(sx) - inner - wr))
        outer = P.lens_cx + P.lens_hw
        tz = P.lens_cz + 0.7 * P.lens_hh
        temple = np.maximum(np.abs(sz - tz) - wr * 0.85, outer - 1.5 - np.abs(sx))
        temple = np.maximum(temple, sy - (self.C[1] - 4.0))
        g = np.minimum(g, temple)
        g = np.where(unit(Ph, T)[..., 1] < 0.45, g, np.inf)
        self.G = np.minimum(g, 1e3).astype(np.float32)

        # hair: M-shaped receding hairline, crew cut, shorter on the sides and back
        aphi = np.abs(Ph)
        knots = [0, 10, 20, 32, 42, 55, 68, 82, 100, 125, 150, 180]
        vals = [37, 38, 40.5, 44, 41, 29, 14, 2, -14, -38, -50, -54]
        self.HL = np.interp(aphi, knots, vals).astype(np.float32)
        side = smoothstep(30, 75, aphi) * smoothstep(40, 10, T)
        thick = P.hair_top + (P.hair_side - P.hair_top) * np.maximum(side, smoothstep(110, 160, aphi) * 0.5)
        front_sparse = 1.0 - 0.45 * smoothstep(14, 0, aphi) * smoothstep(48, 38, T)
        edge = smoothstep(0.0, 6.0, T - self.HL)
        th = thick * front_sparse * (0.3 + 0.7 * edge)
        th = np.maximum(th, 0.6 / s)
        # crew-cut stubble: fine isotropic bumps (about 0.6 mm printed), not strand grooves
        rng = np.random.default_rng(7)
        noise = ndimage.gaussian_filter(rng.standard_normal(R.shape), 0.45 / STEP, mode=["nearest", "wrap"])
        noise /= noise.std() + 1e-9
        stub = P.hair_stubble * np.clip(noise, -2, 2) * edge
        self.RH = (np.maximum(self.R_smooth, Rsk) + th + stub).astype(np.float32)

    # ---------------------------------------------------------------- fields
    def _lookup(self, M, phi, theta):
        fi = (phi + 180.0) / STEP
        ti = (theta + 90.0) / STEP
        Mp = np.concatenate([M[:, -2:], M, M[:, :2]], axis=1)
        return ndimage.map_coordinates(Mp, [ti.ravel(), fi.ravel() + 2], order=1, mode="nearest").reshape(phi.shape)

    def ear(self, X, Y, Z, sgn):
        """Large ear standing out from the head: cupped auricle, rolled helix rim along the top
        and back, a deep concha hollow and a rounded lobe."""
        alpha = np.radians(40.0)  # how far the ear stands out from the head
        ec = np.array([sgn * 79.0, self.C[1] + 2.0, -8.0])
        b = np.array([sgn * np.sin(alpha), np.cos(alpha), 0.0])  # along the ear, toward the back
        n = np.array([sgn * np.cos(alpha), -np.sin(alpha), 0.0])  # outward normal
        qx, qy, qz = X - ec[0], Y - ec[1], Z - ec[2]
        a = qx * b[0] + qy * b[1]
        c = qz
        w = qx * n[0] + qy * n[1]
        # outline: egg shape, broad on top, narrowing to the lobe
        ra = 15.0 + 3.5 * np.clip(c / 30.0, -1, 1)
        rc = 31.0
        e = np.sqrt(((a - 4.0) / ra) ** 2 + (c / rc) ** 2)
        d2 = (e - 1.0) * np.minimum(ra, rc)  # <0 inside the outline
        inner = np.clip(-d2, 0, None)
        # outer-surface height above the attachment plane
        top_back = smoothstep(-6.0, 6.0, a - 2.0 + 0.6 * c)  # rim only on the top/back edge
        rim = 2.4 * smoothstep(5.0, 1.0, inner) * smoothstep(0.0, 1.2, inner) * top_back
        concha = 3.2 * np.exp(-((a - 1.0) / 7.5) ** 2 - ((c + 3.0) / 11.0) ** 2)
        lobe = 1.2 * smoothstep(-16.0, -26.0, c)
        cup = 2.0 * ((a - 4.0) / 15.0) ** 2  # the ear curls outward toward its back edge
        hgt = 3.8 + rim - concha + lobe + cup
        slab = np.maximum(w - hgt, -w - 3.0)
        k = 1.2  # soften the edges
        hm = np.clip(0.5 - 0.5 * (slab - d2) / k, 0, 1)
        return slab * (1 - hm) + d2 * hm + k * hm * (1 - hm)

    def local_fields(self, X, Y, Z, which):
        q = np.stack([X - self.C[0], Y - self.C[1], Z - self.C[2]])
        rho = np.sqrt((q ** 2).sum(0)) + 1e-6
        phi = np.degrees(np.arctan2(q[0], -q[1]))
        theta = np.degrees(np.arcsin(np.clip(q[2] / rho, -1, 1)))
        if which == "skin":
            d = rho - self._lookup(self.R, phi, theta)
            for sg in (-1, 1):
                ed = self.ear(X, Y, Z, sg)
                h = np.clip(0.5 + 0.5 * (ed - d) / 3.0, 0, 1)  # smooth union, k = 3 mm
                d = ed * (1 - h) + d * h - 3.0 * h * (1 - h)
            # neck stub (radius / length are set when the head is attached to a body)
            nx, ny = X, Y - (self.C[1] - 12.0)
            neck = np.maximum(np.sqrt(nx ** 2 + ny ** 2) - self.neck_r, np.maximum(Z - (-60.0), self.neck_bottom - Z))
            h = np.clip(0.5 + 0.5 * (neck - d) / 12.0, 0, 1)
            return neck * (1 - h) + d * h - 12.0 * h * (1 - h)
        if which == "glasses":
            Rr = self._lookup(self.R, phi, theta)
            Rg = self._lookup(self.RG, phi, theta)
            t = self.P.glasses_t / self.s
            radial = np.maximum(rho - (Rg + t), (Rr - 1.0) - rho)
            return np.maximum(radial, self._lookup(self.G, phi, theta))
        if which == "hair":
            Rh = self._lookup(self.RH, phi, theta)
            Rr = self._lookup(self.R_smooth, phi, theta)
            hl = self._lookup(self.HL, phi, theta)
            return np.maximum(np.maximum(rho - Rh, (Rr - 3.0) - rho), np.radians(hl - theta) * rho)
        raise ValueError(which)

    def node(self, pl: Placement, which, name=None):
        lo = self.C - np.array([110.0, 120.0, 250.0])
        hi = self.C + np.array([110.0, 120.0, 125.0])
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
