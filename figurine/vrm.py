"""Minimal VRM / glTF reader with linear-blend skinning, so T-posed avatars can be re-posed
for a figurine, and with per-primitive access so parts (hair, face, body...) can be swapped.

Coordinates stay in glTF space (metres, Y up, model faces +Z for VRM 1.0 / -Z for VRM 0.x).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import pygltflib

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


@dataclass
class Part:
    name: str  # "<mesh name>/<primitive index>:<material name>"
    vertices: np.ndarray  # (n,3) posed, glTF space
    faces: np.ndarray  # (m,3)


class VRM:
    def __init__(self, path):
        self.g = pygltflib.GLTF2().load_binary(path)
        self.blob = self.g.binary_blob()
        self.n_nodes = len(self.g.nodes)
        self.parent = [-1] * self.n_nodes
        for i, n in enumerate(self.g.nodes):
            for c in n.children or []:
                self.parent[c] = i
        self.local = [self._node_local(n) for n in self.g.nodes]
        self.bones = self._humanoid()
        self.morph_names = {}

    # ---------------------------------------------------------------- accessors
    def acc(self, idx):
        a = self.g.accessors[idx]
        bv = self.g.bufferViews[a.bufferView]
        dt = _CT[a.componentType]
        nc = _NC[a.type]
        off = (bv.byteOffset or 0) + (a.byteOffset or 0)
        stride = bv.byteStride or (np.dtype(dt).itemsize * nc)
        n = a.count
        if stride == np.dtype(dt).itemsize * nc:
            arr = np.frombuffer(self.blob, dtype=dt, count=n * nc, offset=off).reshape(n, nc)
        else:
            raw = np.frombuffer(self.blob, dtype=np.uint8, count=stride * (n - 1) + np.dtype(dt).itemsize * nc, offset=off)
            arr = np.stack([np.frombuffer(raw[i * stride:i * stride + np.dtype(dt).itemsize * nc].tobytes(), dtype=dt)
                            for i in range(n)])
        arr = arr.astype(np.float64) if dt == np.float32 else arr
        if a.normalized and dt != np.float32:
            arr = arr / np.iinfo(dt).max
        return arr

    def _node_local(self, n):
        M = np.eye(4)
        if n.matrix:
            return np.array(n.matrix, float).reshape(4, 4).T
        if n.scale:
            M = np.diag(list(n.scale) + [1.0]) @ M
        if n.rotation:
            R = np.eye(4)
            R[:3, :3] = _quat_to_mat(n.rotation)
            M = R @ M
        if n.translation:
            T = np.eye(4)
            T[:3, 3] = n.translation
            M = T @ M
        return M

    def _humanoid(self):
        ext = self.g.extensions or {}
        bones = {}
        if "VRMC_vrm" in ext:
            for k, v in ext["VRMC_vrm"]["humanoid"]["humanBones"].items():
                bones[k] = v["node"]
            self.version = 1
        elif "VRM" in ext:
            for b in ext["VRM"]["humanoid"]["humanBones"]:
                bones[b["bone"]] = b["node"]
            self.version = 0
        else:
            self.version = None
        return bones

    # ---------------------------------------------------------------- posing
    def globals(self, pose):
        """pose: {node index: 3x3 extra local rotation}. Returns list of 4x4 global matrices."""
        G = [None] * self.n_nodes

        def get(i):
            if G[i] is not None:
                return G[i]
            L = self.local[i].copy()
            if i in pose:
                R = np.eye(4)
                R[:3, :3] = pose[i]
                L = L @ R
            G[i] = L if self.parent[i] < 0 else get(self.parent[i]) @ L
            return G[i]

        for i in range(self.n_nodes):
            get(i)
        return G

    def world_rotation_pose(self, rest_globals, bone_rots):
        """Convert {bone name: world-space 3x3 rotation} to local extra rotations at rest pose."""
        pose = {}
        for name, Rw in bone_rots.items():
            if name not in self.bones:
                continue
            i = self.bones[name]
            Gr = rest_globals[i][:3, :3]
            Gr = Gr / np.linalg.norm(Gr, axis=0, keepdims=True)
            pose[i] = Gr.T @ Rw @ Gr  # rotate about world axes through the joint
        return pose

    def scaled_pose(self, world_scales, world_rots=None):
        """Pose from desired *cumulative* world scales per humanoid bone (+ world rotations).

        VRoid rest poses have identity joint rotations, so world and local axes coincide. For each
        bone the local extra transform is P = S_parent_cum^-1 @ S_desired, then the rotation:
        children inherit the parent's scale, so this keeps every bone at exactly its own scale.
        """
        world_rots = world_rots or {}
        want = {self.bones[b]: np.asarray(s, float) for b, s in world_scales.items() if b in self.bones}
        rots = {self.bones[b]: R for b, R in world_rots.items() if b in self.bones}
        cum = {}

        def cumulative(i):
            if i in cum:
                return cum[i]
            if i in want:
                cum[i] = want[i]
            else:
                cum[i] = np.ones(3) if self.parent[i] < 0 else cumulative(self.parent[i])
            return cum[i]

        pose = {}
        for i in range(self.n_nodes):
            sc = cumulative(i)
            par = np.ones(3) if self.parent[i] < 0 else cumulative(self.parent[i])
            P = np.diag(sc / par)
            if i in rots:
                P = P @ rots[i]
            if i in rots or not np.allclose(P, np.eye(3)):
                pose[i] = P
        return pose

    def parts(self, pose=None, morphs=None):
        """All mesh primitives, skinned with `pose`. morphs: {mesh index: {target index: weight}}."""
        pose = pose or {}
        morphs = morphs or {}
        G = self.globals(pose)
        out = []
        for ni, node in enumerate(self.g.nodes):
            if node.mesh is None:
                continue
            mesh = self.g.meshes[node.mesh]
            skin = self.g.skins[node.skin] if node.skin is not None else None
            if skin is not None:
                ibm = self.acc(skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
                J = np.stack([G[j] @ ibm[k] for k, j in enumerate(skin.joints)])
            for pi, prim in enumerate(mesh.primitives):
                a = prim.attributes
                V = self.acc(a.POSITION).copy()
                for ti, w in morphs.get(node.mesh, {}).items():
                    if prim.targets and ti < len(prim.targets) and w:
                        V += w * self.acc(prim.targets[ti]["POSITION"])  # targets share the buffer layout
                if prim.indices is None:
                    Fc = np.arange(len(V)).reshape(-1, 3)
                else:
                    Fc = self.acc(prim.indices).reshape(-1, 3).astype(np.int64)
                # VRoid primitives share one big vertex buffer: keep only referenced vertices
                used, Fc = np.unique(Fc, return_inverse=True)
                Fc = Fc.reshape(-1, 3)
                V = V[used]
                sel = used
                if skin is not None and a.JOINTS_0 is not None:
                    jt = self.acc(a.JOINTS_0).astype(np.int64)[sel]
                    wt = self.acc(a.WEIGHTS_0).astype(np.float64)[sel]
                    wt = wt / np.maximum(wt.sum(1, keepdims=True), 1e-9)
                    Vh = np.concatenate([V, np.ones((len(V), 1))], 1)
                    P = np.zeros((len(V), 3))
                    for k in range(jt.shape[1]):
                        M = J[jt[:, k]]  # (n,4,4)
                        P += wt[:, k:k + 1] * np.einsum("nij,nj->ni", M, Vh)[:, :3]
                    V = P
                else:
                    V = (G[ni] @ np.concatenate([V, np.ones((len(V), 1))], 1).T).T[:, :3]
                mat = ""
                if prim.material is not None and self.g.materials:
                    mat = self.g.materials[prim.material].name or ""
                out.append(Part(f"{mesh.name}/{pi}:{mat}", V, Fc))
        return out

    def bone_position(self, name, pose=None):
        G = self.globals(pose or {})
        return G[self.bones[name]][:3, 3]
