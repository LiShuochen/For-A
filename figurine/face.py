"""Photos -> fused, frontalized 3D face landmarks (real-world millimetres).

Coordinate frame of the output (head-local, real mm):
    +x = viewer's right, +z = up, -y = toward the viewer (the face looks along -y).
Origin = midpoint between the two pupils.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "assets", "models", "face_landmarker.task")
PHOTOS = os.path.join(ROOT, "assets", "photos")
CACHE = os.path.join(ROOT, "assets", "face_model.npz")

IPD_MM = 63.0  # adult interpupillary distance used to give landmarks real-world scale

# (photo id, approx face centre in full-res px or None for whole image, crop radius, weight)
# Neutral, near-frontal views of the person on the far right of the group photo, plus the
# high-resolution 3/4 close-up. Looking-up / looking-down / laughing shots are excluded.
SOURCES = [
    ("7b85dcec", (463, 644), 170, 1.0),
    ("411ee641", (1940, 1087), 170, 1.0),
    ("87dff726", (515, 636), 170, 0.8),
    ("39503021", None, 0, 0.35),
]

# MediaPipe landmark indices
RIGHT_PUPIL, LEFT_PUPIL = 468, 473  # subject's right = viewer's left
NOSE_TIP, CHIN, FOREHEAD = 1, 152, 10
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
LEFT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


@dataclass
class FaceModel:
    points: np.ndarray  # (478, 3) real mm, head-local frame
    per_photo: list  # [(photo id, (478,3) aligned points)] for diagnostics

    def idx(self, ids):
        return self.points[np.asarray(ids)]


def _landmarker():
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        num_faces=1,
        min_face_detection_confidence=0.2,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def detect(lm, photo_id, center, radius):
    """Return (pts (478,3) camera-like frame in px, rotation 3x3) or None."""
    import mediapipe as mp

    im = Image.open(os.path.join(PHOTOS, f"{photo_id}-image.jpg")).convert("RGB")
    if center is not None:
        cx, cy = center
        im = im.crop((cx - radius, cy - radius, cx + radius, cy + radius)).resize((680, 680), Image.LANCZOS)
    arr = np.ascontiguousarray(np.asarray(im))
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if not res.face_landmarks:
        return None
    W, H = im.size
    f = res.face_landmarks[0]
    # camera-like frame: x right, y up, z toward the viewer
    pts = np.array([[l.x * W, -l.y * H, -l.z * W] for l in f], dtype=np.float64)
    M = np.array(res.facial_transformation_matrixes[0], dtype=np.float64)
    R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0, keepdims=True)
    return pts, R


def frontalize(pts, R):
    """Undo head rotation; return points in head-local frame (x right, y = -toward viewer, z up)."""
    c = pts.mean(axis=0)
    p = (pts - c) @ R  # R^T applied to column vectors
    # canonical frame is x right, y up, z toward viewer -> head-local x right, z up, -y toward viewer
    out = np.stack([p[:, 0], -p[:, 2], p[:, 1]], axis=1)
    mid = 0.5 * (out[RIGHT_PUPIL] + out[LEFT_PUPIL])
    out -= mid
    ipd = np.linalg.norm(out[LEFT_PUPIL] - out[RIGHT_PUPIL])
    return out * (IPD_MM / ipd)


def _similarity(src, dst, w=None):
    """Least-squares similarity transform mapping src -> dst (Umeyama). Returns transformed src."""
    w = np.ones(len(src)) if w is None else w
    w = w / w.sum()
    ms, md = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    a, b = src - ms, dst - md
    cov = (w[:, None] * b).T @ a
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    Rm = U @ D @ Vt
    var = (w * (a ** 2).sum(1)).sum()
    s = np.trace(np.diag(S) @ D) / var
    return (s * (Rm @ a.T)).T + md


def symmetrize(points):
    """Average each landmark with the mirror of its partner to remove pose/lens noise.

    MediaPipe's topology is left/right symmetric; partners are found by nearest mirrored point.
    Keeps 35% of the original asymmetry so the face is not uncannily perfect.
    """
    mirrored = points * np.array([-1, 1, 1])
    d = ((points[:, None, :] - mirrored[None, :, :]) ** 2).sum(-1)
    partner = d.argmin(1)
    sym = 0.5 * (points + mirrored[partner])
    return 0.35 * points + 0.65 * sym


def build(force=False) -> FaceModel:
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        return FaceModel(z["points"], list(z["per_photo"]))
    lm = _landmarker()
    got = []
    for pid, center, radius, weight in SOURCES:
        r = detect(lm, pid, center, radius)
        if r is None:
            print(f"[face] no face found in {pid}, skipped")
            continue
        got.append((pid, frontalize(*r), weight))
    if not got:
        raise RuntimeError("no usable face photos")
    ref = got[0][1]
    aligned = [(pid, _similarity(p, ref), w) for pid, p, w in got]
    W = np.array([w for _, _, w in aligned])
    fused = sum(w * p for (_, p, _), w in zip(aligned, W)) / W.sum()
    mid = 0.5 * (fused[RIGHT_PUPIL] + fused[LEFT_PUPIL])
    # scale stays that of the frontal reference photo (normalised by its own IPD); the 3/4 close-up
    # has unreliable iris points, so re-normalising by the fused IPD would inflate the face
    fused = symmetrize(fused - mid)
    per = [(pid, p) for pid, p, _ in aligned]
    np.savez(CACHE, points=fused, per_photo=np.array(per, dtype=object))
    return FaceModel(fused, per)


if __name__ == "__main__":
    fm = build(force=True)
    p = fm.points
    print("pupil dist", np.linalg.norm(p[LEFT_PUPIL] - p[RIGHT_PUPIL]))
    print("nose tip", p[NOSE_TIP], "chin", p[CHIN], "forehead", p[FOREHEAD])
    print("x(33) %.1f x(263) %.1f" % (p[33, 0], p[263, 0]))
    print("face width %.1f  height %.1f" % (np.ptp(fm.idx(FACE_OVAL)[:, 0]), p[FOREHEAD, 2] - p[CHIN, 2]))
