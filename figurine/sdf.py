"""Signed-distance-field modelling on a voxel grid (negative = inside, millimetres).

Nodes are evaluated over *regions* of a global grid. Every node has a bounding box, and
unions/subtractions only evaluate a child inside the child's box, so a 100M-voxel figure
made of a few hundred primitives stays fast and fits in memory.
"""
from __future__ import annotations

import numpy as np

BIG = np.float32(1e3)
F32 = np.float32


# ---------------------------------------------------------------- grid & regions
class Grid:
    def __init__(self, lo, hi, h):
        self.h = float(h)
        self.lo = np.asarray(lo, dtype=np.float64)
        self.shape = tuple(int(np.ceil((hi[i] - lo[i]) / h)) + 1 for i in range(3))

    def axes(self, reg):
        """Broadcastable coordinate arrays (x:(n,1,1), y:(1,n,1), z:(1,1,n)) for a region."""
        (i0, i1), (j0, j1), (k0, k1) = reg
        x = (self.lo[0] + self.h * np.arange(i0, i1)).astype(F32)[:, None, None]
        y = (self.lo[1] + self.h * np.arange(j0, j1)).astype(F32)[None, :, None]
        z = (self.lo[2] + self.h * np.arange(k0, k1)).astype(F32)[None, None, :]
        return x, y, z

    def full(self):
        return tuple((0, n) for n in self.shape)

    def region_of(self, box, within):
        """Index region covering world box, clipped to `within`; None if empty."""
        if box is None:
            return within
        lo, hi = box
        out = []
        for a in range(3):
            i0 = int(np.floor((lo[a] - self.lo[a]) / self.h))
            i1 = int(np.ceil((hi[a] - self.lo[a]) / self.h)) + 1
            i0, i1 = max(i0, within[a][0]), min(i1, within[a][1])
            if i1 <= i0:
                return None
            out.append((i0, i1))
        return tuple(out)


def _local(sub, reg):
    return tuple(slice(sub[a][0] - reg[a][0], sub[a][1] - reg[a][0]) for a in range(3))


def _shape(reg):
    return tuple(r[1] - r[0] for r in reg)


def _grow(box, m):
    if box is None:
        return None
    return (np.asarray(box[0]) - m, np.asarray(box[1]) + m)


def _merge(boxes):
    if any(b is None for b in boxes):
        return None
    return (np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0))


def _inter(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (np.maximum(a[0], b[0]), np.minimum(a[1], b[1]))


# ---------------------------------------------------------------- smooth ops
def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (b * (1 - h) + a * h - k * h * (1 - h)).astype(F32)


def smax(a, b, k):
    return -smin(-a, -b, k)


# ---------------------------------------------------------------- node base
class Node:
    box = None  # world AABB of the solid (plus margin); None = unbounded
    name = ""

    def eval(self, grid, reg):  # -> float32 array of _shape(reg)
        raise NotImplementedError

    def __or__(self, other):
        return Union([self, other])

    def __sub__(self, other):
        return Subtract(self, other)

    def __and__(self, other):
        return Intersect(self, other)


class Func(Node):
    """Leaf from a vectorised function f(x, y, z) -> distance, with a bounding box."""

    def __init__(self, f, box, name=""):
        self.f, self.box, self.name = f, (np.asarray(box[0], float), np.asarray(box[1], float)), name

    CHUNK = 6_000_000  # voxels per slab, bounds temporary memory

    def eval(self, grid, reg):
        shp = _shape(reg)
        plane = shp[1] * shp[2]
        step = max(1, self.CHUNK // max(plane, 1))
        out = np.empty(shp, dtype=F32)
        for i0 in range(reg[0][0], reg[0][1], step):
            sub = ((i0, min(i0 + step, reg[0][1])), reg[1], reg[2])
            x, y, z = grid.axes(sub)
            out[i0 - reg[0][0]:sub[0][1] - reg[0][0]] = np.broadcast_to(self.f(x, y, z), _shape(sub))
        return out


class Union(Node):
    def __init__(self, children, k=0.0, name=""):
        self.children, self.k, self.name = list(children), float(k), name
        self.box = _merge([_grow(c.box, self.k) for c in self.children])

    def eval(self, grid, reg):
        out = np.full(_shape(reg), BIG, dtype=F32)
        for c in self.children:
            sub = grid.region_of(_grow(c.box, self.k + 2 * grid.h), reg)
            if sub is None:
                continue
            sl = _local(sub, reg)
            out[sl] = smin(out[sl], c.eval(grid, sub), self.k)
        return out


class Subtract(Node):
    def __init__(self, a, b, k=0.0, name=""):
        self.a, self.b, self.k, self.name = a, b, float(k), name
        self.box = a.box

    def eval(self, grid, reg):
        out = self.a.eval(grid, reg)
        sub = grid.region_of(_grow(self.b.box, self.k + 2 * grid.h), reg)
        if sub is not None:
            sl = _local(sub, reg)
            out[sl] = smax(out[sl], -self.b.eval(grid, sub), self.k)
        return out


class Intersect(Node):
    def __init__(self, a, b, k=0.0, name=""):
        self.a, self.b, self.k, self.name = a, b, float(k), name
        self.box = _inter(a.box, b.box)

    def eval(self, grid, reg):
        out = self.a.eval(grid, reg)
        sub = grid.region_of(_grow(self.b.box, 2 * grid.h), reg)
        res = np.full(_shape(reg), BIG, dtype=F32)
        if sub is not None:
            sl = _local(sub, reg)
            res[sl] = smax(out[sl], self.b.eval(grid, sub), self.k)
        return res


class Offset(Node):
    """Grow (r>0) or shrink a solid; also used for shells via abs()."""

    def __init__(self, a, r, name=""):
        self.a, self.r, self.name = a, float(r), name
        self.box = _grow(a.box, max(self.r, 0))

    def eval(self, grid, reg):
        return (self.a.eval(grid, reg) - self.r).astype(F32)


# ---------------------------------------------------------------- primitive helpers
# Constants are converted to Python floats so that numpy keeps the float32 grid arrays float32.
def _v(p):
    return np.asarray(p, dtype=np.float64)


def _f(p):
    return [float(t) for t in np.ravel(p)]


def sphere(c, r, name=""):
    cx, cy, cz = _f(c)
    r = float(r)
    return Func(lambda x, y, z: np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - r,
                (_v(c) - r, _v(c) + r), name)


def ellipsoid(c, radii, name="", R=None):
    """Ellipsoid with semi-axes `radii`, optionally rotated by matrix R (columns = local axes in world)."""
    c, rad = _v(c), _v(radii)
    R = np.eye(3) if R is None else np.asarray(R, float)
    ext = np.sqrt(((R * rad[None, :]) ** 2).sum(1))  # world AABB half-extent
    cx, cy, cz = _f(c)
    r0, r1, r2 = _f(rad)
    Rf = [_f(R[i]) for i in range(3)]

    def f(x, y, z):
        dx, dy, dz = x - cx, y - cy, z - cz
        u = (dx * Rf[0][0] + dy * Rf[1][0] + dz * Rf[2][0]) / r0
        v = (dx * Rf[0][1] + dy * Rf[1][1] + dz * Rf[2][1]) / r1
        w = (dx * Rf[0][2] + dy * Rf[1][2] + dz * Rf[2][2]) / r2
        k0 = np.sqrt(u * u + v * v + w * w)
        k1 = np.sqrt((u / r0) ** 2 + (v / r1) ** 2 + (w / r2) ** 2)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)

    return Func(f, (c - ext, c + ext), name)


def round_cone(a, b, ra, rb, name=""):
    """Capsule from a to b whose radius goes from ra to rb (exact SDF, after Inigo Quilez)."""
    a_, b_ = _v(a), _v(b)
    ax, ay, az = _f(a_)
    bax, bay, baz = _f(b_ - a_)
    ra, rb = float(ra), float(rb)
    l2 = bax * bax + bay * bay + baz * baz
    rr = ra - rb
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    srr = float(np.sign(rr)) * rr * rr

    def f(x, y, z):
        px, py, pz = x - ax, y - ay, z - az
        yv = px * bax + py * bay + pz * baz
        zv = yv - l2
        cx, cy, cz = px * l2 - bax * yv, py * l2 - bay * yv, pz * l2 - baz * yv
        x2 = cx * cx + cy * cy + cz * cz
        y2 = yv * yv * l2
        z2 = zv * zv * l2
        k = srr * x2
        d_b = np.sqrt(x2 + z2) * il2 - rb
        d_a = np.sqrt(x2 + y2) * il2 - ra
        d_s = (np.sqrt(x2 * a2 * il2) + yv * rr) * il2 - ra
        return np.where(np.sign(zv) * a2 * z2 > k, d_b, np.where(np.sign(yv) * a2 * y2 < k, d_a, d_s))

    lo = np.minimum(a_ - ra, b_ - rb)
    hi = np.maximum(a_ + ra, b_ + rb)
    return Func(f, (lo, hi), name)


def capsule(a, b, r, name=""):
    return round_cone(a, b, r, r, name)


def tube(points, radii, k=0.0, name=""):
    """Smooth chain of round cones through `points` with per-point radii."""
    pts = [_v(p) for p in points]
    segs = [round_cone(pts[i], pts[i + 1], radii[i], radii[i + 1]) for i in range(len(pts) - 1)]
    return Union(segs, k=k, name=name)


def box(c, half, r=0.0, name="", R=None):
    """Rounded box centred at c with half-extents `half` (rounding r), optional rotation R."""
    c, hb = _v(c), _v(half)
    R = np.eye(3) if R is None else np.asarray(R, float)
    ext = np.abs(R) @ (hb + r)
    cx, cy, cz = _f(c)
    h0, h1, h2 = _f(hb)
    Rf = [_f(R[i]) for i in range(3)]
    r = float(r)

    def f(x, y, z):
        dx, dy, dz = x - cx, y - cy, z - cz
        u = np.abs(dx * Rf[0][0] + dy * Rf[1][0] + dz * Rf[2][0]) - h0
        v = np.abs(dx * Rf[0][1] + dy * Rf[1][1] + dz * Rf[2][1]) - h1
        w = np.abs(dx * Rf[0][2] + dy * Rf[1][2] + dz * Rf[2][2]) - h2
        out = np.sqrt(np.maximum(u, 0) ** 2 + np.maximum(v, 0) ** 2 + np.maximum(w, 0) ** 2)
        return out + np.minimum(np.maximum(u, np.maximum(v, w)), 0) - r

    return Func(f, (c - ext, c + ext), name)


def cylinder_z(c, r, h0, h1, round_=0.0, name=""):
    """Vertical capped cylinder (axis z) from z=h0 to z=h1, edges rounded by round_."""
    cx, cy = _f(c)[:2]
    r, h0, h1, ro = float(r), float(h0), float(h1), float(round_)

    def f(x, y, z):
        d_r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - (r - ro)
        d_z = np.abs(z - 0.5 * (h0 + h1)) - (0.5 * (h1 - h0) - ro)
        return (np.minimum(np.maximum(d_r, d_z), 0)
                + np.sqrt(np.maximum(d_r, 0) ** 2 + np.maximum(d_z, 0) ** 2) - ro)

    return Func(f, ((cx - r, cy - r, h0), (cx + r, cy + r, h1)), name)


def rot(axis, deg):
    """Rotation matrix about x/y/z by degrees."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def evaluate(node, h, pad=1.0):
    """Evaluate node on a fresh grid around its bounding box. Returns (field, grid)."""
    lo, hi = node.box
    grid = Grid(np.asarray(lo) - pad, np.asarray(hi) + pad, h)
    return node.eval(grid, grid.full()), grid
