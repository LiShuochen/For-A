"""JK pleated skirt as an SDF: elliptical bell with knife pleats and a self-supporting underside."""
from __future__ import annotations

import numpy as np

from .sdf import Func


def pleated_skirt(z_top, z_hem, top_r, hem_r, y_c=0.0, pleats=24, pleat_amp=1.0, band_h=1.5,
                  hem_ring=1.0, under_slope=1.25, z_floor=None, name="skirt"):
    """Skirt between z_hem (bottom) and z_top (waist).

    top_r / hem_r: (half-width x, half-depth y) of the ellipse at the waist / hem, in mm.
    pleat_amp: pleat depth (mm) at the hem; pleats fade toward the waistband.
    under_slope: the underside rises inward at this slope (dz per mm), ~51 deg -> no supports.
    z_floor: if given, the skirt is a solid pedestal standing on z_floor (bust version).
    """
    (ta, tb), (ha, hb) = top_r, hem_r
    z_top, z_hem, y_c = float(z_top), float(z_hem), float(y_c)
    L = z_top - z_hem
    n = float(pleats)

    def f(x, y, z):
        X, Y, Z = np.broadcast_arrays(x, y - y_c, z)
        t = np.clip((z_top - Z) / L, 0.0, 1.0)
        flare = t ** 1.25  # slightly bell-shaped
        a = ta + (ha - ta) * flare
        b = tb + (hb - tb) * flare
        theta = np.arctan2(Y, X)
        rho = np.sqrt((X / a) ** 2 + (Y / b) ** 2)
        r_eff = np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)
        # knife pleats: sawtooth around the waist, softened fall; depth grows toward the hem
        ph = np.mod(theta * n / (2 * np.pi), 1.0)
        saw = np.where(ph < 0.82, ph / 0.82, (1.0 - ph) / 0.18)
        grow = np.clip((z_top - band_h - Z) / max(L - band_h, 1e-6), 0, 1) ** 0.8
        d_side = (rho - 1.0) * r_eff - pleat_amp * grow * (saw - 0.5)
        d_top = Z - z_top
        if z_floor is None:
            inward = np.maximum((1.0 - rho) * r_eff - hem_ring, 0.0)
            d_bot = (z_hem + under_slope * inward) - Z
        else:
            d_bot = float(z_floor) - Z
        return np.maximum(np.maximum(d_side, d_top), d_bot * 0.7)

    ext_a, ext_b = max(ta, ha) + pleat_amp + 1, max(tb, hb) + pleat_amp + 1
    lo_z = z_hem if z_floor is None else float(z_floor)
    return Func(f, ((-ext_a, y_c - ext_b, lo_z - 1), (ext_a, y_c + ext_b, z_top + 1)), name)
