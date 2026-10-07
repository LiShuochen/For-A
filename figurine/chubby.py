"""Chubby Q-version doll body in a JK sailor uniform, built from smooth SDF primitives.

Proportions follow figure-style chibi dolls (~2.3 heads): a round belly, puffy short sleeves,
stubby arms with round hands clasped in front, a flared pleated skirt, short chubby legs in knee
socks and round-toed shoes. Every colour region is its own node so it can be printed in colour.
All sizes are in printed mm for a figure whose body (soles to neck) is `body_h` tall.
"""
from __future__ import annotations

import numpy as np

from . import sdf
from .skirt import pleated_skirt


def chubby_body(body_h=66.0, base_h=3.0):
    """Returns ({group: node}, attach) with attach = dict(neck_top, neck_r, chin_z)."""
    u = body_h / 66.0  # all numbers below are designed for a 66 mm body
    z0 = base_h

    def P(x, y, z):
        return np.array([x * u, y * u, z0 + z * u])

    def r(v):
        return v * u

    k = r(2.2)  # blend radius for soft, toy-like joins
    # ---- skin: neck, forearms, hands, thighs/knees above the socks
    neck = sdf.capsule(P(0, 1.0, 55.5), P(0, 1.5, 60.5), r(4.9))
    hands = []
    arms_skin = []
    for sg in (-1, 1):
        elbow = P(sg * 13.2, -2.0, 47.0)
        hand = P(sg * 4.0, -11.8, 38.6)
        arms_skin.append(sdf.round_cone(elbow, hand, r(3.9), r(3.5)))
        hands.append(sdf.ellipsoid(hand + np.array([0, -r(0.4), 0]), [r(4.4), r(3.6), r(4.0)]))
    legs = [sdf.round_cone(P(sg * 5.6, 0.0, 26.0), P(sg * 5.3, 0.4, 10.0), r(5.4), r(4.6)) for sg in (-1, 1)]
    skin = sdf.Union([neck] + arms_skin + hands + legs, k=r(1.6), name="skin")

    # ---- uniform top: round belly + chest, puffy short sleeves (white)
    belly = sdf.ellipsoid(P(0, 0.4, 45.5), [r(11.2), r(10.2), r(12.6)])
    chest = sdf.ellipsoid(P(0, 0.8, 53.0), [r(11.0), r(9.0), r(7.4)])
    sleeves = []
    for sg in (-1, 1):
        sh = P(sg * 10.4, 0.6, 54.6)
        sleeves.append(sdf.round_cone(sh, P(sg * 12.6, -0.8, 49.0), r(5.1), r(4.5)))
        sleeves.append(sdf.round_cone(P(sg * 12.6, -0.8, 49.0), P(sg * 13.0, -1.4, 47.4), r(4.6), r(4.3)))  # cuff
    top = sdf.Union([belly, chest] + sleeves, k=k, name="top")

    # ---- sailor collar (navy): a raised flap over the shoulders/back + front lapels, with a stripe
    torso = sdf.Union([belly, chest], k=k)
    collar_shell = sdf.Offset(torso, r(0.7))
    zc = z0 + 58.6 * u
    yfront = 0.0

    def collar_region(x, y, z):
        """Negative inside: back flap over the shoulders + two front lapels forming a V."""
        X, Y, Z = np.broadcast_arrays(x, y, z)
        back = np.maximum(np.maximum(np.abs(X) - r(9.4), (z0 + 47.5 * u) - Z), (yfront - r(1.0)) - Y)
        vz = z0 + 48.5 * u + 0.95 * np.abs(X)
        lapel = np.maximum(np.maximum(Z - (vz + r(3.6)), vz - Z), Y - (yfront + r(2.0)))
        return np.maximum(np.minimum(back, lapel), Z - zc)

    region = sdf.Func(collar_region, (P(-20, -16, 44), P(20, 16, 62)))
    collar = sdf.Intersect(collar_shell, region, name="collar")
    # neck opening so the collar doesn't cover the neck
    collar = sdf.Subtract(collar, sdf.capsule(P(0, 1.0, 54.0), P(0, 1.2, 70.0), r(5.6)), name="collar")

    # ---- bow / scarf (red) at the V of the collar
    bz = 49.4
    bow = sdf.Union([
        sdf.ellipsoid(P(-3.6, -10.6, bz), [r(3.6), r(1.8), r(2.5)], R=sdf.rot("y", 12)),
        sdf.ellipsoid(P(3.6, -10.6, bz), [r(3.6), r(1.8), r(2.5)], R=sdf.rot("y", -12)),
        sdf.sphere(P(0, -11.4, bz), r(1.9)),
        sdf.round_cone(P(-0.8, -11.0, bz - 1.0), P(-2.4, -10.6, bz - 6.2), r(1.3), r(1.1)),
        sdf.round_cone(P(0.8, -11.0, bz - 1.0), P(2.4, -10.6, bz - 6.2), r(1.3), r(1.1)),
    ], k=r(0.6), name="bow")

    # ---- pleated skirt (navy), flared, self-supporting underside
    skirt = pleated_skirt(z_top=z0 + 37.5 * u, z_hem=z0 + 23.0 * u, top_r=(r(11.0), r(9.8)),
                          hem_r=(r(16.4), r(14.6)), y_c=0.3 * u, pleats=18, pleat_amp=r(1.1),
                          band_h=r(1.8), hem_ring=r(1.0), name="skirt")

    # ---- knee socks (navy) with a cuff, and round-toed shoes (black/brown)
    socks, shoes = [], []
    for sg in (-1, 1):
        socks.append(sdf.round_cone(P(sg * 5.4, 0.3, 17.0), P(sg * 5.3, 0.4, 9.0), r(5.05), r(4.85)))
        socks.append(sdf.round_cone(P(sg * 5.4, 0.3, 17.6), P(sg * 5.4, 0.3, 16.4), r(5.35), r(5.35)))  # cuff
        shoe = sdf.ellipsoid(P(sg * 5.6, -2.6, 3.6), [r(5.9), r(8.6), r(4.6)], R=sdf.rot("z", sg * 8))
        flat = sdf.Func(lambda x, y, z, zz=z0: zz - z, (P(-20, -20, -5), P(20, 20, 20)))
        shoes.append(sdf.Intersect(shoe, flat))
        shoes.append(sdf.capsule(P(sg * 1.0, -4.0, 7.2), P(sg * 10.2, -4.0, 7.2), r(0.9)))  # strap
    socks = sdf.Union(socks, k=r(0.4), name="socks")
    shoes = sdf.Union(shoes, k=r(0.8), name="shoes")

    groups = {"skin": skin, "top": top, "collar": collar, "bow": bow, "skirt": skirt, "socks": socks, "shoes": shoes}
    attach = {"neck_top": P(0, 1.5, 60.5), "neck_r": r(4.9), "chin_z": z0 + 61.0 * u}
    return groups, attach
