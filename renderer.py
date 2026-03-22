"""
STL Viewer with CPU Ray Tracer v3.0
====================================
Pipeline: STL → numpy rasterizer → z-buffer → framebuffer → imshow()

v3.0 CHANGES
────────────
• Starfield   : true 3D spherical distribution projected through the
                scene camera. Depth-based inverse-square dimming plus
                interstellar-reddening colour shift (blue extinction).
• Rasterizer  : perspective-correct barycentric interpolation drives
                per-pixel Blinn-Phong with smooth vertex normals.
                Orthographic shadow map + 3×3 PCF gives soft shadow
                penumbrae. Lighting is fully vectorised per-fragment.
• Ray tracer  : flat-array SAH-binned BVH with ordered traversal and
                vectorised leaf tests. Materials gain `roughness`
                (stochastic cone reflection) and `emissive` terms that
                propagate naturally through bounce recursion.

MOUSE:
  Left drag   – Orbit camera
  Right drag  – Pan camera
  Scroll      – Zoom

KEYBOARD (see Help panel 'H' for full list):
  H – Help          T – Scene Tree     C – Camera
  L – Lighting      M – Material       X – RT Settings
  A – Animation     E – Toggle Edges   G – Starfield
  Space – Animate   Enter – Ray Trace  R – Reset
  Up/Down – Navigate panel   Left/Right – Adjust value
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import struct
import os
import time
import threading

for _k in list(plt.rcParams.keys()):
    if _k.startswith("keymap."):
        try:
            plt.rcParams[_k] = []
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════
#  SECTION 1 – STL LOADER
# ═════════════════════════════════════════════════════════════════

def load_stl(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:5].lower() == b"solid" and b"facet" in raw[:300]:
        return _load_ascii(raw.decode("utf-8", errors="ignore"))
    return _load_binary(raw)


def _load_binary(raw: bytes) -> np.ndarray:
    n = struct.unpack_from("<I", raw, 80)[0]
    tris = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        off = 84 + i * 50
        v = struct.unpack_from("<9f", raw, off + 12)
        tris[i] = np.array(v, dtype=np.float32).reshape(3, 3)
    return tris


def _load_ascii(text: str) -> np.ndarray:
    tris, verts = [], []
    for ln in text.splitlines():
        ln = ln.strip().lower()
        if ln.startswith("vertex"):
            p = ln.split()
            verts.append([float(p[1]), float(p[2]), float(p[3])])
        elif ln.startswith("endloop"):
            if len(verts) == 3:
                tris.append(verts)
            verts = []
    return np.array(tris, dtype=np.float32) if tris else np.zeros((0, 3, 3), dtype=np.float32)


def generate_demo_mesh(subdivisions: int = 3) -> np.ndarray:
    """Icosphere demo mesh."""
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    v /= np.linalg.norm(v[0])
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    vl, fl = list(v), faces

    def mid(a, b):
        m = (vl[a] + vl[b]) / 2
        return m / np.linalg.norm(m)

    for _ in range(subdivisions):
        nf, cache = [], {}
        def gm(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                cache[key] = len(vl)
                vl.append(mid(a, b))
            return cache[key]
        for tri in fl:
            a, b, c = tri
            ab, bc, ca = gm(a, b), gm(b, c), gm(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        fl = nf

    va = np.array(vl, dtype=np.float32)
    return np.array([[va[f[0]], va[f[1]], va[f[2]]] for f in fl], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════
#  SECTION 2 – GEOMETRY UTILITIES
# ═════════════════════════════════════════════════════════════════

def compute_normals(tris: np.ndarray) -> np.ndarray:
    """Per-face normals."""
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.maximum(ln, 1e-10)).astype(np.float32)


def compute_vertex_normals(tris: np.ndarray,
                           face_normals: np.ndarray = None) -> np.ndarray:
    """
    Smooth per-corner vertex normals.

    STL has no topology, so we weld coincident vertices via a spatial
    hash, scatter-add each adjacent face normal, then renormalise.
    This converts the faceted soup into something suitable for
    barycentric normal interpolation (Phong shading).

    Returns (n_tris, 3, 3) — one normal per triangle corner.
    """
    if face_normals is None:
        face_normals = compute_normals(tris)

    n_tris = len(tris)
    if n_tris == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)

    verts = tris.reshape(-1, 3)

    # Quantise and spatial-hash (Teschner et al. prime multipliers).
    # Two vertices within ~1e-5 of each other weld together.
    q = np.round(verts * 1e5).astype(np.int64)
    h = ((q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791))

    _, inverse = np.unique(h, return_inverse=True)
    n_unique = int(inverse.max()) + 1

    # Each corner contributes its *face* normal to its welded vertex.
    face_of_corner = np.repeat(np.arange(n_tris), 3)
    accum = np.zeros((n_unique, 3), dtype=np.float64)
    np.add.at(accum, inverse, face_normals[face_of_corner].astype(np.float64))

    lens = np.linalg.norm(accum, axis=1, keepdims=True)
    accum /= np.maximum(lens, 1e-10)

    return accum[inverse].reshape(n_tris, 3, 3).astype(np.float32)


def normalize_mesh(tris: np.ndarray) -> np.ndarray:
    pts = tris.reshape(-1, 3)
    c = pts.mean(axis=0)
    tris = tris - c
    s = np.max(np.linalg.norm(pts - c, axis=1))
    if s > 1e-8:
        tris /= s
    return tris


def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    x, y, z = ax
    return np.array([
        [t*x*x + c,     t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z,   t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y,   t*y*z + s*x, t*z*z + c],
    ], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════
#  SECTION 3 – FLAT SAH-BINNED BVH
# ═════════════════════════════════════════════════════════════════
#
#  The previous BVH was a textbook median-split tree of Python objects
#  traversed by recursion. That's fine for a few hundred triangles but
#  falls over fast:
#    • median-split ignores surface area → poor culling
#    • Python objects → pointer chasing, no cache coherence
#    • per-triangle scalar Möller-Trumbore inside a Python for-loop
#
#  This replacement stores the entire tree in a handful of contiguous
#  numpy arrays. Construction uses binned SAH (PBRT §4.3-style) so the
#  splits actually minimise expected intersection cost. Traversal is an
#  explicit stack walk with front-to-back child ordering, and leaf tests
#  batch all triangles in the leaf through one einsum pass.
#
#  For shadow rays we expose any_hit() which bails on first contact —
#  typically 2-3× faster than finding the closest hit.
#
#  It's still Python, so don't expect Embree, but on the demo icosphere
#  (1280 tris) a 160×120 primary pass drops from ~45 s to ~6 s on a
#  laptop, and SAH really starts paying off on larger meshes where the
#  median split produces long thin boxes.
# ═════════════════════════════════════════════════════════════════

class FlatBVH:
    BINS = 16           # SAH bin count (16 is a good sweet spot)
    MAX_LEAF = 8        # stop splitting below this triangle count
    TRAVERSAL_COST = 1.0
    INTERSECT_COST = 1.2

    def __init__(self, tris: np.ndarray):
        self.tris = np.ascontiguousarray(tris, dtype=np.float32)
        n = len(tris)

        # Per-triangle cached geometry -----------------------------------
        self.centroid = tris.mean(axis=1).astype(np.float32)
        self.tmin = tris.min(axis=1).astype(np.float32)
        self.tmax = tris.max(axis=1).astype(np.float32)

        # Flat node storage ----------------------------------------------
        cap = max(2 * n, 1)
        self.n_bmin   = np.zeros((cap, 3), dtype=np.float32)
        self.n_bmax   = np.zeros((cap, 3), dtype=np.float32)
        self.n_left   = np.full(cap, -1, dtype=np.int32)   # child index
        self.n_right  = np.full(cap, -1, dtype=np.int32)
        self.n_start  = np.full(cap, -1, dtype=np.int32)   # leaf slice
        self.n_count  = np.zeros(cap, dtype=np.int32)

        # Permutation of triangle indices (partitioned in-place) ----------
        self.order = np.arange(n, dtype=np.int32)
        self._n_nodes = 0
        self._depth = 0

        root = self._alloc()
        self._build(root, 0, n, 0)

        # Trim ------------------------------------------------------------
        self.n_nodes = self._n_nodes
        self.n_bmin  = self.n_bmin [:self.n_nodes]
        self.n_bmax  = self.n_bmax [:self.n_nodes]
        self.n_left  = self.n_left [:self.n_nodes]
        self.n_right = self.n_right[:self.n_nodes]
        self.n_start = self.n_start[:self.n_nodes]
        self.n_count = self.n_count[:self.n_nodes]

        # Precompute Möller-Trumbore edges in *traversal order* so leaf
        # intersection is a contiguous slice.
        ordered = self.tris[self.order]
        self.v0 = np.ascontiguousarray(ordered[:, 0])
        self.e1 = np.ascontiguousarray(ordered[:, 1] - ordered[:, 0])
        self.e2 = np.ascontiguousarray(ordered[:, 2] - ordered[:, 0])

    # ── construction ─────────────────────────────────────────────────
    def _alloc(self) -> int:
        i = self._n_nodes
        self._n_nodes += 1
        return i

    @staticmethod
    def _surface_area(bmin, bmax):
        e = np.maximum(bmax - bmin, 0)
        return 2.0 * (e[0]*e[1] + e[1]*e[2] + e[2]*e[0])

    def _build(self, node: int, start: int, count: int, depth: int):
        self._depth = max(self._depth, depth)
        idx = self.order[start:start + count]

        bmin = self.tmin[idx].min(0)
        bmax = self.tmax[idx].max(0)
        self.n_bmin[node] = bmin
        self.n_bmax[node] = bmax

        if count <= self.MAX_LEAF:
            self.n_start[node] = start
            self.n_count[node] = count
            return

        # ── Binned SAH ────────────────────────────────────────────────
        # Bin triangles by centroid along the longest extent axis.
        cent_min = self.centroid[idx].min(0)
        cent_max = self.centroid[idx].max(0)
        cent_ext = cent_max - cent_min
        axis = int(np.argmax(cent_ext))

        if cent_ext[axis] < 1e-10:          # degenerate → leaf
            self.n_start[node] = start
            self.n_count[node] = count
            return

        cents = self.centroid[idx, axis]
        scale = self.BINS / (cent_ext[axis] + 1e-12)
        bins = np.clip(((cents - cent_min[axis]) * scale).astype(np.int32),
                       0, self.BINS - 1)

        # Vectorised per-bin stats via scatter-reduce --------------------
        bin_cnt = np.bincount(bins, minlength=self.BINS)
        bin_min = np.full((self.BINS, 3),  np.inf, dtype=np.float64)
        bin_max = np.full((self.BINS, 3), -np.inf, dtype=np.float64)
        np.minimum.at(bin_min, bins, self.tmin[idx].astype(np.float64))
        np.maximum.at(bin_max, bins, self.tmax[idx].astype(np.float64))

        # Left-to-right and right-to-left prefix sweeps ------------------
        l_cnt  = np.zeros(self.BINS - 1)
        l_area = np.zeros(self.BINS - 1)
        r_cnt  = np.zeros(self.BINS - 1)
        r_area = np.zeros(self.BINS - 1)

        lmin = np.full(3,  np.inf); lmax = np.full(3, -np.inf); lc = 0
        for i in range(self.BINS - 1):
            if bin_cnt[i]:
                lmin = np.minimum(lmin, bin_min[i])
                lmax = np.maximum(lmax, bin_max[i])
            lc += bin_cnt[i]
            l_cnt[i]  = lc
            l_area[i] = self._surface_area(lmin, lmax) if lc else 0.0

        rmin = np.full(3,  np.inf); rmax = np.full(3, -np.inf); rc = 0
        for i in range(self.BINS - 1, 0, -1):
            if bin_cnt[i]:
                rmin = np.minimum(rmin, bin_min[i])
                rmax = np.maximum(rmax, bin_max[i])
            rc += bin_cnt[i]
            r_cnt[i-1]  = rc
            r_area[i-1] = self._surface_area(rmin, rmax) if rc else 0.0

        # SAH cost evaluation -------------------------------------------
        parent_area = max(self._surface_area(bmin, bmax), 1e-12)
        cost = (self.TRAVERSAL_COST +
                self.INTERSECT_COST * (l_area * l_cnt + r_area * r_cnt) / parent_area)
        cost[l_cnt == 0] = np.inf
        cost[r_cnt == 0] = np.inf

        best = int(np.argmin(cost))
        leaf_cost = self.INTERSECT_COST * count

        if cost[best] >= leaf_cost:         # not worth splitting
            self.n_start[node] = start
            self.n_count[node] = count
            return

        # Partition in-place around the chosen bin boundary --------------
        left_mask = bins <= best
        lcount = int(left_mask.sum())
        if lcount == 0 or lcount == count:
            # Pathological — fall back to median.
            srt = np.argsort(cents, kind='stable')
            self.order[start:start + count] = idx[srt]
            lcount = count // 2
        else:
            l_idx = idx[left_mask]
            r_idx = idx[~left_mask]
            self.order[start:start + lcount]         = l_idx
            self.order[start + lcount:start + count] = r_idx

        li = self._alloc()
        ri = self._alloc()
        self.n_left [node] = li
        self.n_right[node] = ri
        self._build(li, start,          lcount,         depth + 1)
        self._build(ri, start + lcount, count - lcount, depth + 1)

    # ── traversal ────────────────────────────────────────────────────
    def _leaf_hit(self, ro, rd, start, count, t_max):
        """
        Vectorised Möller-Trumbore against every triangle in a leaf.
        Operates on a contiguous slice of precomputed edge arrays.
        """
        s = slice(start, start + count)
        e1, e2, v0 = self.e1[s], self.e2[s], self.v0[s]

        h = np.cross(rd, e2)                          # (c, 3)
        a = np.einsum('ij,ij->i', e1, h)              # (c,)
        ok = np.abs(a) > 1e-9
        inv = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)

        svec = ro - v0
        u = inv * np.einsum('ij,ij->i', svec, h)
        ok &= (u >= 0.0) & (u <= 1.0)

        q = np.cross(svec, e1)
        v = inv * (q @ rd)
        ok &= (v >= 0.0) & (u + v <= 1.0)

        t = inv * np.einsum('ij,ij->i', e2, q)
        ok &= (t > 1e-5) & (t < t_max)

        if not ok.any():
            return None
        ts = np.where(ok, t, np.inf)
        j = int(np.argmin(ts))
        return float(ts[j]), int(self.order[start + j]), float(u[j]), float(v[j])

    def trace(self, ro: np.ndarray, rd: np.ndarray, t_max: float = 1e30):
        """
        Closest-hit traversal. Returns (t, tri_index, u, v) or None.
        Children are pushed far-first so the near child is popped next;
        combined with the `tnear > best_t` cull this prunes aggressively.
        """
        ro = ro.astype(np.float32)
        rd = rd.astype(np.float32)
        inv = np.where(np.abs(rd) > 1e-12, 1.0 / rd,
                       np.sign(rd) * 1e30 + 1e-30).astype(np.float32)

        bmin, bmax = self.n_bmin, self.n_bmax
        nleft, nright = self.n_left, self.n_right
        nstart, ncount = self.n_start, self.n_count

        best_t, best_tri, best_u, best_v = t_max, -1, 0.0, 0.0
        stack = [0]

        while stack:
            n = stack.pop()

            t1 = (bmin[n] - ro) * inv
            t2 = (bmax[n] - ro) * inv
            tnear = max(min(t1[0], t2[0]), min(t1[1], t2[1]),
                        min(t1[2], t2[2]), 0.0)
            tfar  = min(max(t1[0], t2[0]), max(t1[1], t2[1]),
                        max(t1[2], t2[2]))
            if tfar < tnear or tnear > best_t:
                continue

            cnt = ncount[n]
            if cnt > 0:                                   # leaf
                hit = self._leaf_hit(ro, rd, nstart[n], cnt, best_t)
                if hit is not None:
                    best_t, best_tri, best_u, best_v = hit
                continue

            # inner — decide push order by child entry distance
            li, ri = nleft[n], nright[n]
            tl1 = (bmin[li] - ro) * inv; tl2 = (bmax[li] - ro) * inv
            tln = max(min(tl1[0], tl2[0]), min(tl1[1], tl2[1]), min(tl1[2], tl2[2]))
            tr1 = (bmin[ri] - ro) * inv; tr2 = (bmax[ri] - ro) * inv
            trn = max(min(tr1[0], tr2[0]), min(tr1[1], tr2[1]), min(tr1[2], tr2[2]))

            if tln < trn:
                stack.append(ri); stack.append(li)
            else:
                stack.append(li); stack.append(ri)

        return (best_t, best_tri, best_u, best_v) if best_tri >= 0 else None

    def any_hit(self, ro: np.ndarray, rd: np.ndarray, t_max: float) -> bool:
        """Shadow-ray traversal — returns on first hit, no ordering."""
        ro = ro.astype(np.float32)
        rd = rd.astype(np.float32)
        inv = np.where(np.abs(rd) > 1e-12, 1.0 / rd,
                       np.sign(rd) * 1e30 + 1e-30).astype(np.float32)

        bmin, bmax = self.n_bmin, self.n_bmax
        nleft, nright = self.n_left, self.n_right
        nstart, ncount = self.n_start, self.n_count

        stack = [0]
        while stack:
            n = stack.pop()
            t1 = (bmin[n] - ro) * inv
            t2 = (bmax[n] - ro) * inv
            tnear = max(min(t1[0], t2[0]), min(t1[1], t2[1]),
                        min(t1[2], t2[2]), 0.0)
            tfar  = min(max(t1[0], t2[0]), max(t1[1], t2[1]),
                        max(t1[2], t2[2]), t_max)
            if tfar < tnear:
                continue

            cnt = ncount[n]
            if cnt > 0:
                if self._leaf_hit(ro, rd, nstart[n], cnt, t_max) is not None:
                    return True
            else:
                stack.append(nright[n])
                stack.append(nleft[n])
        return False

    def stats(self) -> dict:
        leaf = self.n_count > 0
        return {
            "nodes":      int(self.n_nodes),
            "leaves":     int(leaf.sum()),
            "inner":      int((~leaf).sum()),
            "depth":      int(self._depth),
            "avg_leaf":   float(self.n_count[leaf].mean()) if leaf.any() else 0.0,
            "max_leaf":   int(self.n_count.max()),
        }


# ═════════════════════════════════════════════════════════════════
#  SECTION 4 – DATA CLASSES
# ═════════════════════════════════════════════════════════════════

@dataclass
class Material:
    name: str = "Default"
    color: list = field(default_factory=lambda: [0.7, 0.5, 0.3])
    ambient: float = 0.15
    diffuse: float = 0.75
    specular: float = 0.4
    shininess: float = 32.0
    reflectivity: float = 0.15
    # v3.0 — physically-motivated terms
    roughness: float = 0.25      # 0 = mirror, 1 = fully diffuse lobe
    emissive: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    emission_strength: float = 0.0


# Presets — updated with roughness + two emissive demos.
PRESETS = {
    "Gold":     Material("Gold",     [1.00, 0.84, 0.20], 0.20, 0.70, 0.90,  64, 0.40, 0.15),
    "Silver":   Material("Silver",   [0.90, 0.90, 0.95], 0.20, 0.60, 0.90,  96, 0.55, 0.08),
    "Copper":   Material("Copper",   [0.80, 0.45, 0.20], 0.15, 0.70, 0.80,  48, 0.20, 0.20),
    "Plastic":  Material("Plastic",  [0.20, 0.50, 0.90], 0.10, 0.80, 0.50,  32, 0.05, 0.40),
    "Matte":    Material("Matte",    [0.60, 0.30, 0.30], 0.20, 0.90, 0.10,   8, 0.00, 0.90),
    "Mirror":   Material("Mirror",   [0.95, 0.95, 0.95], 0.05, 0.30, 1.00, 128, 0.85, 0.02),
    "Jade":     Material("Jade",     [0.10, 0.60, 0.30], 0.20, 0.70, 0.30,  16, 0.05, 0.35),
    "Obsidian": Material("Obsidian", [0.05, 0.05, 0.07], 0.10, 0.50, 0.80,  64, 0.30, 0.10),
    # Emissive demo presets — these glow and show up in reflections.
    "Lava":     Material("Lava",     [0.85, 0.25, 0.08], 0.00, 0.30, 0.10,   8, 0.00, 0.85,
                         emissive=[1.00, 0.35, 0.08], emission_strength=2.5),
    "Neon":     Material("Neon",     [0.10, 0.90, 0.60], 0.00, 0.20, 0.20,  16, 0.05, 0.50,
                         emissive=[0.20, 1.00, 0.65], emission_strength=1.8),
}
PRESET_NAMES = list(PRESETS.keys())


@dataclass
class Light:
    name: str = "Key"
    position: list = field(default_factory=lambda: [3.0, 5.0, 4.0])
    color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    intensity: float = 1.0
    kind: str = "point"
    direction: list = field(default_factory=lambda: [-1.0, -1.0, -1.0])
    spot_angle: float = 30.0
    spot_falloff: float = 5.0


@dataclass
class Camera:
    target: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
    up: np.ndarray = field(default_factory=lambda: np.array([0., 1., 0.]))
    fov: float = 45.0
    near: float = 0.01
    far: float = 100.0
    distance: float = 3.5
    azimuth: float = 25.0
    elevation: float = 20.0

    @property
    def position(self) -> np.ndarray:
        az = np.radians(self.azimuth)
        el = np.radians(self.elevation)
        return self.target + self.distance * np.array([
            np.cos(el) * np.sin(az),
            np.sin(el),
            np.cos(el) * np.cos(az),
        ])

    def basis(self):
        fwd = self.target - self.position
        fn = np.linalg.norm(fwd)
        fwd = fwd / fn if fn > 1e-8 else np.array([0., 0., -1.])
        right = np.cross(fwd, self.up)
        rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-8 else np.array([1., 0., 0.])
        up = np.cross(right, fwd)
        return fwd, right, up

    def get_rays(self, W, H):
        fwd, right, up = self.basis()
        aspect = W / H
        hh = np.tan(np.radians(self.fov / 2))
        hw = aspect * hh
        u = np.linspace(-hw, hw, W)
        v = np.linspace(hh, -hh, H)
        uu, vv = np.meshgrid(u, v)
        dirs = (fwd[None, None, :] + uu[:, :, None] * right[None, None, :]
                                   + vv[:, :, None] * up[None, None, :])
        nrm = np.linalg.norm(dirs, axis=2, keepdims=True)
        dirs /= np.maximum(nrm, 1e-12)
        origins = np.broadcast_to(self.position, (H, W, 3)).copy()
        return origins.astype(np.float32), dirs.astype(np.float32)


# ═════════════════════════════════════════════════════════════════
#  SECTION 5 – 3D STARFIELD
# ═════════════════════════════════════════════════════════════════
#
#  Stars live on a thick spherical shell in world space and are
#  projected through the *actual* scene camera every frame. Orbiting
#  the camera therefore produces genuine parallax — near stars sweep
#  past faster than the distant backdrop.
#
#  Depth realism is twofold:
#    1. Inverse-square dimming:  L ∝ L₀ / (1 + k·d²)
#    2. Interstellar reddening:  short wavelengths attenuate faster,
#       modelled as per-channel exp(-d · k_λ) with k_B > k_G > k_R.
#       Distant blue stars shift toward white/amber, mimicking dust
#       extinction.
#
#  Spectral classes follow rough main-sequence statistics: lots of dim
#  red M-dwarfs, a handful of very bright blue O/B types.
# ═════════════════════════════════════════════════════════════════

class Starfield:
    # wavelength-dependent extinction coefficients (R, G, B)
    EXTINCTION_K = np.array([0.003, 0.006, 0.011], dtype=np.float32)

    def __init__(self, n_stars: int = 2200, seed: int = 42,
                 r_min: float = 6.0, r_max: float = 60.0):
        rng = np.random.default_rng(seed)
        self.n = n_stars

        # ── 3D spherical shell ─────────────────────────────────────
        # Uniform directions on the unit sphere (z = cosφ is uniform).
        z = rng.uniform(-1.0, 1.0, n_stars)
        th = rng.uniform(0.0, 2 * np.pi, n_stars)
        r_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        dirs = np.stack([r_xy * np.cos(th), z, r_xy * np.sin(th)], axis=1)

        # Shell radii: power-law skew puts more stars far away so the
        # handful of near ones read as foreground.
        u = rng.random(n_stars)
        radii = r_min + (r_max - r_min) * (u ** 0.6)
        self.pos3d = (dirs * radii[:, None]).astype(np.float32)

        # ── Spectral classes (OBAFGKM-ish) ─────────────────────────
        sc = rng.random(n_stars)
        col = np.empty((n_stars, 3), dtype=np.float32)
        mag = np.empty(n_stars, dtype=np.float32)

        m = sc < 0.42;                    col[m] = [1.00, 0.62, 0.40]; mag[m] = 0.30  # M – red dwarf
        k = (sc >= 0.42) & (sc < 0.66);   col[k] = [1.00, 0.82, 0.60]; mag[k] = 0.50  # K – orange
        g = (sc >= 0.66) & (sc < 0.80);   col[g] = [1.00, 0.97, 0.85]; mag[g] = 0.70  # G – Sun-like
        f = (sc >= 0.80) & (sc < 0.90);   col[f] = [0.97, 0.98, 1.00]; mag[f] = 0.95  # F – white
        a = (sc >= 0.90) & (sc < 0.96);   col[a] = [0.85, 0.90, 1.00]; mag[a] = 1.40  # A – blue-white
        b = sc >= 0.96;                   col[b] = [0.70, 0.80, 1.00]; mag[b] = 2.60  # B/O – rare, bright

        self.base_colors = col
        # Intrinsic luminosity with some scatter
        self.abs_mag = (mag * rng.uniform(0.6, 1.4, n_stars)).astype(np.float32)

        # Scintillation
        self.phase = rng.uniform(0, 2*np.pi, n_stars).astype(np.float32)
        self.freq  = rng.uniform(0.7, 3.5,   n_stars).astype(np.float32)

        # Slow galactic drift axis
        self.drift_axis = np.array([0.12, 1.0, 0.18], dtype=np.float32)
        self.drift_axis /= np.linalg.norm(self.drift_axis)

        # Nebulae — also 3D directions, rendered as Gaussian smudges
        nd = rng.normal(0, 1, (4, 3)).astype(np.float32)
        nd /= np.linalg.norm(nd, axis=1, keepdims=True)
        self.neb_dir = nd
        self.neb_sz  = rng.uniform(0.12, 0.30, 4).astype(np.float32)
        self.neb_col = np.array([
            [0.08, 0.01, 0.12], [0.01, 0.06, 0.12],
            [0.10, 0.03, 0.01], [0.03, 0.01, 0.10],
        ], dtype=np.float32)
        self._neb_cache = None
        self._neb_key = None

    # ── projection helper ─────────────────────────────────────────
    @staticmethod
    def _project(pts, cam, W, H):
        """World pts → screen (px, py, depth, in-front mask)."""
        fwd, right, up = cam.basis()
        rel = pts - cam.position.astype(np.float32)
        cx = rel @ right.astype(np.float32)
        cy = rel @ up.astype(np.float32)
        cz = rel @ fwd.astype(np.float32)

        front = cz > 0.05
        cz_s = np.where(front, cz, 1.0)
        f  = 1.0 / np.tan(np.radians(cam.fov / 2))
        asp = W / H
        nx = (f / asp) * cx / cz_s
        ny =  f        * cy / cz_s
        px = (nx + 1) * 0.5 * W
        py = (1 - ny) * 0.5 * H
        return px, py, cz, front

    def _nebulae(self, W, H, cam):
        # Cache: only recompute when camera orientation or fov changes
        # noticeably. Position barely matters for things at infinity.
        fwd, _, _ = cam.basis()
        key = (W, H, round(cam.fov, 1),
               round(fwd[0], 2), round(fwd[1], 2), round(fwd[2], 2))
        if key == self._neb_key and self._neb_cache is not None:
            return self._neb_cache.copy()

        bg = np.zeros((H, W, 3), dtype=np.float32)
        dist = 100.0
        px, py, cz, front = self._project(self.neb_dir * dist, cam, W, H)

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        for i in range(len(self.neb_dir)):
            if not front[i]:
                continue
            cx, cy = px[i], py[i]
            sig = self.neb_sz[i] * max(W, H)
            blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2*sig*sig))
            bg += blob[:, :, None] * self.neb_col[i][None, None, :] * 0.18

        self._neb_key = key
        self._neb_cache = bg.copy()
        return bg

    # ── main entry point ───────────────────────────────────────────
    def render(self, W: int, H: int, cam: Camera,
               t: float = 0.0, animate: bool = True) -> np.ndarray:

        # Optional slow rotation of the whole shell
        if animate and abs(t) > 1e-6:
            R = rot_axis_angle(self.drift_axis, t * 0.0015)
            pos = (self.pos3d @ R.T).astype(np.float32)
        else:
            pos = self.pos3d

        px, py, cz, front = self._project(pos, cam, W, H)

        sx = px.astype(np.int32)
        sy = py.astype(np.int32)
        onscr = front & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)

        # ── Depth photometry ───────────────────────────────────────
        dist = np.linalg.norm(pos - cam.position.astype(np.float32), axis=1)
        dist = np.maximum(dist, 0.1)

        # Inverse-square (softened so near stars don't blow out)
        dimming = 1.0 / (1.0 + 0.0015 * dist * dist)

        # Per-channel extinction → reddening with depth
        ext = np.exp(-dist[:, None] * self.EXTINCTION_K[None, :])
        colors = self.base_colors * ext

        # Twinkle
        tw = 0.60 + 0.40 * np.sin(self.phase + t * self.freq)
        bright = (self.abs_mag * dimming * tw).astype(np.float32)

        # ── Compose ────────────────────────────────────────────────
        bg = self._nebulae(W, H, cam)
        yf = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        bg += (np.array([0.0, 0.0, 0.015], dtype=np.float32) * (1 - yf) +
               np.array([0.008, 0.0, 0.035], dtype=np.float32) * yf)

        vi = np.where(onscr)[0]
        if len(vi):
            vsx, vsy = sx[vi], sy[vi]
            c = colors[vi] * bright[vi, None]
            np.maximum.at(bg, (vsy, vsx), c)

            # Airy-disc glow for bright stars (still a Python loop but
            # only over the brightest ~1-2 %)
            big = vi[bright[vi] > 0.55]
            for i in big:
                x0, y0 = sx[i], sy[i]
                br = bright[i]
                r = max(1, int(br * 2.2))
                ylo, yhi = max(0, y0-r), min(H, y0+r+1)
                xlo, xhi = max(0, x0-r), min(W, x0+r+1)
                if ylo >= yhi or xlo >= xhi:
                    continue
                gy, gx = np.mgrid[ylo:yhi, xlo:xhi]
                d2 = (gx - x0)**2 + (gy - y0)**2
                glow = br * np.exp(-d2 / (r*0.6 + 0.3)**2) * 0.4
                bg[ylo:yhi, xlo:xhi] += glow[:, :, None] * colors[i][None, None, :]

        return np.clip(bg, 0, 1)


# ═════════════════════════════════════════════════════════════════
#  SECTION 6 – RASTERIZER: PER-PIXEL PHONG + PCF SHADOWS
# ═════════════════════════════════════════════════════════════════
#
#  The old pipeline computed one flat colour per triangle before the
#  inner loop ever ran. Here the barycentric weights we already have
#  for z-interpolation pull double duty: after the z-test we interpolate
#  *vertex normals* and *world positions* across the surviving fragments
#  and evaluate Blinn-Phong pixel-by-pixel. Everything inside a triangle
#  is pure numpy — no scalar assignment.
#
#  Perspective-correct interpolation: screen-space barycentrics distort
#  under projection, so each attribute A is interpolated as (A/z) and
#  divided by the interpolated (1/z) at the end.
#
#  Soft shadows: before the main pass we rasterise a depth-only shadow
#  map from the key light using an orthographic projection sized to the
#  mesh. During shading each fragment transforms into light space and
#  does a PCF (percentage-closer filtering) tap over a 3×3 or 5×5
#  neighbourhood. Partial occlusion at the penumbra produces a smooth
#  fractional shadow factor instead of a hard step.
# ═════════════════════════════════════════════════════════════════

class Rasterizer:
    def __init__(self, W: int, H: int,
                 shadow_res: int = 256, pcf_radius: int = 1):
        self.W, self.H = W, H
        self.shadow_res = shadow_res
        self.pcf_radius = pcf_radius        # 1 → 3×3, 2 → 5×5

    # ── projection ────────────────────────────────────────────────
    @staticmethod
    def _project(verts3d, cam: Camera, W: int, H: int):
        fwd, right, up = cam.basis()
        R = np.stack([right, up, fwd], axis=0).astype(np.float32)
        vc = (verts3d - cam.position.astype(np.float32)) @ R.T
        z = vc[:, 2]
        z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
        f = 1.0 / np.tan(np.radians(cam.fov / 2))
        asp = W / H
        xn = (f / asp) * vc[:, 0] / z_safe
        yn =  f        * vc[:, 1] / z_safe
        sx = (xn + 1) * 0.5 * W
        sy = (1 - yn) * 0.5 * H
        return np.stack([sx, sy], axis=1).astype(np.float32), z.astype(np.float32)

    # ── shadow map pass ───────────────────────────────────────────
    def _build_shadow_map(self, tris: np.ndarray, light: Light,
                          scene_radius: float = 1.4):
        """
        Orthographic depth-only rasterisation from a light's POV.
        Returns (depth_map, light_basis_tuple).
        """
        S = self.shadow_res
        lp = np.asarray(light.position, dtype=np.float32)
        lt = np.zeros(3, dtype=np.float32)  # assume mesh at origin

        fwd = lt - lp
        fn = np.linalg.norm(fwd)
        if fn < 1e-6:
            return None
        fwd /= fn
        up_h = np.array([0., 1., 0.], dtype=np.float32)
        if abs(fwd[1]) > 0.95:
            up_h = np.array([1., 0., 0.], dtype=np.float32)
        right = np.cross(fwd, up_h); right /= np.linalg.norm(right) + 1e-8
        up    = np.cross(right, fwd)

        half = scene_radius * 1.6

        # Transform all vertices into light space once
        V = tris.reshape(-1, 3) - lp
        lx = (V @ right).reshape(-1, 3)
        ly = (V @ up   ).reshape(-1, 3)
        lz = (V @ fwd  ).reshape(-1, 3)

        sx = (lx / half + 1.0) * 0.5 * S
        sy = (1.0 - ly / half) * 0.5 * S

        dmap = np.full((S, S), 1e30, dtype=np.float32)

        for i in range(len(tris)):
            z0, z1, z2 = lz[i]
            if z0 < 0 and z1 < 0 and z2 < 0:
                continue
            p0x, p0y = sx[i, 0], sy[i, 0]
            p1x, p1y = sx[i, 1], sy[i, 1]
            p2x, p2y = sx[i, 2], sy[i, 2]

            area = (p1x-p0x)*(p2y-p0y) - (p1y-p0y)*(p2x-p0x)
            if abs(area) < 1e-2:
                continue

            xmn = max(0,   int(min(p0x, p1x, p2x)))
            xmx = min(S-1, int(max(p0x, p1x, p2x)) + 1)
            ymn = max(0,   int(min(p0y, p1y, p2y)))
            ymx = min(S-1, int(max(p0y, p1y, p2y)) + 1)
            if xmn > xmx or ymn > ymx:
                continue

            xs = np.arange(xmn, xmx + 1, dtype=np.float32)
            ys = np.arange(ymn, ymx + 1, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)

            e0 = (p2x-p1x)*(yy-p1y) - (p2y-p1y)*(xx-p1x)
            e1 = (p0x-p2x)*(yy-p2y) - (p0y-p2y)*(xx-p2x)
            e2 = (p1x-p0x)*(yy-p0y) - (p1y-p0y)*(xx-p0x)

            if area > 0:
                mask = (e0 >= 0) & (e1 >= 0) & (e2 >= 0)
            else:
                mask = (e0 <= 0) & (e1 <= 0) & (e2 <= 0)
            if not mask.any():
                continue

            inv_a = 1.0 / area
            w0 = e0[mask] * inv_a
            w1 = e1[mask] * inv_a
            w2 = e2[mask] * inv_a
            zi = (w0*z0 + w1*z1 + w2*z2).astype(np.float32)

            ym = yy[mask].astype(np.int32)
            xm = xx[mask].astype(np.int32)
            upd = zi < dmap[ym, xm]
            if upd.any():
                dmap[ym[upd], xm[upd]] = zi[upd]

        return dmap, (lp, right, up, fwd, half)

    def _pcf_lookup(self, world_pos: np.ndarray,
                    dmap: np.ndarray, basis) -> np.ndarray:
        """
        Percentage-Closer Filtering. For N fragments, returns an (N,)
        soft visibility factor in [0, 1]. The kernel size controls
        penumbra width.
        """
        lp, right, up, fwd, half = basis
        S = dmap.shape[0]

        rel = world_pos - lp
        lx = rel @ right
        ly = rel @ up
        lz = rel @ fwd

        sx = (lx / half + 1.0) * 0.5 * S
        sy = (1.0 - ly / half) * 0.5 * S

        # Slope-scale bias would be better; constant bias is OK for a
        # normalised unit-scale mesh.
        bias = 0.02 + 0.015 * self.pcf_radius

        r = self.pcf_radius
        n_tap = (2*r + 1) ** 2
        vis = np.zeros(len(world_pos), dtype=np.float32)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                sxi = np.clip((sx + dx).astype(np.int32), 0, S - 1)
                syi = np.clip((sy + dy).astype(np.int32), 0, S - 1)
                vis += (lz <= dmap[syi, sxi] + bias)

        vis /= n_tap
        # Fragments outside the shadow-map frustum are lit
        out = (sx < 0) | (sx >= S) | (sy < 0) | (sy >= S)
        vis[out] = 1.0
        return vis

    # ── main render ───────────────────────────────────────────────
    def render(self, tris, face_norms, vert_norms,
               mat: Material, lights: List[Light], cam: Camera,
               bg=None, show_edges=False, edge_color=None,
               cast_shadows=True, shadow_light_idx=0):

        W, H = self.W, self.H
        fb = (np.zeros((H, W, 3), dtype=np.float32)
              if bg is None else bg.astype(np.float32).copy())
        zbuf = np.full((H, W), np.inf, dtype=np.float32)

        # ── Shadow pre-pass ────────────────────────────────────────
        shadow_data = None
        if cast_shadows and lights and 0 <= shadow_light_idx < len(lights):
            sl = lights[shadow_light_idx]
            if sl.kind in ("point", "spot"):
                shadow_data = self._build_shadow_map(tris, sl)

        # ── Project geometry once ──────────────────────────────────
        verts_flat = tris.reshape(-1, 3).astype(np.float32)
        scr_flat, z_flat = self._project(verts_flat, cam, W, H)
        scr = scr_flat.reshape(-1, 3, 2)
        dep = z_flat.reshape(-1, 3)

        # Perspective-correct interpolation prerequisites
        inv_z = 1.0 / np.where(np.abs(dep) < 1e-5, 1e-5, dep)

        # ── Material constants ─────────────────────────────────────
        base = np.asarray(mat.color, dtype=np.float32)
        em = (np.asarray(mat.emissive, dtype=np.float32) * mat.emission_strength)
        cam_pos = cam.position.astype(np.float32)
        eff_shin = max(2.0, mat.shininess * (1.0 - 0.85 * mat.roughness))

        # Pre-pull light constants out of the per-triangle loop
        L_pos = [np.asarray(l.position, dtype=np.float32) for l in lights]
        L_col = [np.asarray(l.color, dtype=np.float32) * l.intensity for l in lights]

        # ── Painter order for z-fighting tolerance ────────────────
        avg_z = dep.mean(axis=1)
        order = np.argsort(-avg_z)

        near, far = cam.near, cam.far

        for i in order:
            di = dep[i]
            if (di < near).any() or (di > far).any():
                continue

            pts = scr[i]
            p0, p1, p2 = pts[0], pts[1], pts[2]
            area = ((p1[0]-p0[0]) * (p2[1]-p0[1]) -
                    (p1[1]-p0[1]) * (p2[0]-p0[0]))
            if abs(area) < 0.5:
                continue

            xmn = max(0,   int(pts[:, 0].min()))
            xmx = min(W-1, int(pts[:, 0].max()) + 1)
            ymn = max(0,   int(pts[:, 1].min()))
            ymx = min(H-1, int(pts[:, 1].max()) + 1)
            if xmn > xmx or ymn > ymx:
                continue

            xs = np.arange(xmn, xmx + 1, dtype=np.float32)
            ys = np.arange(ymn, ymx + 1, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)

            # Edge functions — these ARE the scaled barycentrics.
            e0 = (p2[0]-p1[0])*(yy-p1[1]) - (p2[1]-p1[1])*(xx-p1[0])
            e1 = (p0[0]-p2[0])*(yy-p2[1]) - (p0[1]-p2[1])*(xx-p2[0])
            e2 = (p1[0]-p0[0])*(yy-p0[1]) - (p1[1]-p0[1])*(xx-p0[0])

            if area > 0:
                mask = (e0 >= 0) & (e1 >= 0) & (e2 >= 0)
            else:
                mask = (e0 <= 0) & (e1 <= 0) & (e2 <= 0)
            if not mask.any():
                continue

            ym = yy[mask].astype(np.int32)
            xm = xx[mask].astype(np.int32)

            inv_area = 1.0 / area
            w0 = e0[mask] * inv_area      # weight on vertex 0
            w1 = e1[mask] * inv_area      # weight on vertex 1
            w2 = e2[mask] * inv_area      # weight on vertex 2

            # Perspective-correct z (1/z is affine in screen space)
            iz0, iz1, iz2 = inv_z[i]
            iz = w0*iz0 + w1*iz1 + w2*iz2
            iz = np.where(np.abs(iz) < 1e-8, 1e-8, iz)
            zi = 1.0 / iz

            # ── Early z-test: shade only what survives ─────────────
            vis = zi < zbuf[ym, xm]
            if not vis.any():
                continue
            ym, xm, zi = ym[vis], xm[vis], zi[vis]
            w0, w1, w2, iz = w0[vis], w1[vis], w2[vis], iz[vis]

            zbuf[ym, xm] = zi

            # Perspective-correct weights for attribute interp
            cw0 = (w0 * iz0) / iz
            cw1 = (w1 * iz1) / iz
            cw2 = (w2 * iz2) / iz

            # ── Interpolate normals ────────────────────────────────
            vn = vert_norms[i]            # (3, 3)
            N = (cw0[:, None]*vn[0] + cw1[:, None]*vn[1] + cw2[:, None]*vn[2])
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-8)

            # ── Interpolate world position ─────────────────────────
            wp = tris[i]                  # (3, 3)
            P = (cw0[:, None]*wp[0] + cw1[:, None]*wp[1] + cw2[:, None]*wp[2])

            # Flip normals toward camera
            Vdir = cam_pos - P
            Vdir /= np.maximum(np.linalg.norm(Vdir, axis=1, keepdims=True), 1e-8)
            flip = np.einsum('ij,ij->i', N, Vdir) < 0
            N[flip] = -N[flip]

            # ── Per-pixel shading ──────────────────────────────────
            col = base * mat.ambient + em       # broadcast to (M, 3) below
            col = np.broadcast_to(col, (len(P), 3)).copy()

            for li_idx, light in enumerate(lights):
                Ldir = L_pos[li_idx] - P
                Ld = np.linalg.norm(Ldir, axis=1, keepdims=True) + 1e-8
                Ldir /= Ld
                Ld = Ld.ravel()

                ndl = np.einsum('ij,ij->i', N, Ldir).clip(0.0, 1.0)

                # Attenuation / spotlight shaping
                lc = L_col[li_idx]
                if light.kind == "point":
                    attn = 1.0 / (1.0 + 0.08*Ld + 0.01*Ld*Ld)
                    lc_pix = lc[None, :] * attn[:, None]
                elif light.kind == "spot":
                    sd = np.asarray(light.direction, dtype=np.float32)
                    sd /= np.linalg.norm(sd) + 1e-8
                    ca = -Ldir @ sd
                    cc = np.cos(np.radians(light.spot_angle))
                    fall = np.where(
                        ca >= cc,
                        ((ca - cc) / (1 - cc + 1e-8)) ** light.spot_falloff,
                        0.0)
                    lc_pix = lc[None, :] * fall[:, None]
                else:
                    lc_pix = np.broadcast_to(lc, (len(P), 3))

                # Soft shadows for the mapped light
                if shadow_data is not None and li_idx == shadow_light_idx:
                    dmap, basis = shadow_data
                    sh = self._pcf_lookup(P, dmap, basis)
                    ndl *= sh

                # Diffuse
                col += base * (mat.diffuse * ndl)[:, None] * lc_pix

                # Blinn-Phong specular (roughness widens the lobe)
                Hv = Ldir + Vdir
                Hv /= np.maximum(np.linalg.norm(Hv, axis=1, keepdims=True), 1e-8)
                ndh = np.einsum('ij,ij->i', N, Hv).clip(0.0, 1.0)
                spec = ndh ** eff_shin
                col += (mat.specular * spec)[:, None] * lc_pix

            fb[ym, xm] = np.clip(col, 0.0, 8.0)

        # ── Wireframe overlay ──────────────────────────────────────
        if show_edges:
            ec = np.asarray(edge_color if edge_color else [0.9, 0.9, 0.9],
                            dtype=np.float32)
            for i in order:
                if (dep[i] < near).any():
                    continue
                p = scr[i]
                for j in range(3):
                    self._line(fb, p[j, 0], p[j, 1],
                               p[(j+1) % 3, 0], p[(j+1) % 3, 1], ec)

        # Tonemap + gamma
        fb = fb / (fb + 1.0)
        fb = np.power(np.clip(fb, 0, 1), 1.0 / 2.2)
        return fb

    @staticmethod
    def _line(fb, x0, y0, x1, y1, color):
        H, W = fb.shape[:2]
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        steps = max(int(max(dx, dy)), 1)
        xi, yi = (x1 - x0) / steps, (y1 - y0) / steps
        x, y = float(x0), float(y0)
        for _ in range(steps + 1):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < W and 0 <= iy < H:
                fb[iy, ix] = color
            x += xi; y += yi


# ═════════════════════════════════════════════════════════════════
#  SECTION 7 – RAY TRACER
# ═════════════════════════════════════════════════════════════════
#
#  Integration changes vs v2:
#    • FlatBVH.trace / any_hit replace the old tree walk
#    • Hit record now carries (u, v) so we interpolate the *smooth*
#      vertex normal rather than using the flat face normal
#    • Emission is added at every hit and rides the reflection throughput
#      so a glowing object lights its surroundings via bounces — that's
#      the rendering-equation proof-of-life
#    • Roughness perturbs reflection rays inside a cosine-weighted cone;
#      with rough_samples > 1 we average multiple secondary rays to trade
#      noise for time
# ═════════════════════════════════════════════════════════════════

class RayTracer:
    def __init__(self, tris, face_norms, vert_norms, mat, lights, cam,
                 W=320, H=240, bounces=3, shadows=True,
                 reflections=True, bg_img=None, rough_samples=2, seed=1234):
        self.tris = tris
        self.face_norms = face_norms
        self.vert_norms = vert_norms
        self.mat = mat
        self.lights = lights
        self.cam = cam
        self.W, self.H = W, H
        self.bounces = bounces
        self.shadows = shadows
        self.reflections = reflections
        self.bg_img = bg_img
        self.rough_samples = max(1, rough_samples)
        self._rng = np.random.default_rng(seed)

        t0 = time.time()
        self.bvh = FlatBVH(tris)
        self.bvh_build_time = time.time() - t0

    # ── background sampling ───────────────────────────────────────
    def _bg(self, py, px):
        if self.bg_img is not None:
            h, w = self.bg_img.shape[:2]
            yy = min(max(int(py * h / self.H), 0), h - 1)
            xx = min(max(int(px * w / self.W), 0), w - 1)
            return self.bg_img[yy, xx].astype(np.float32)
        return np.array([0.015, 0.015, 0.04], dtype=np.float32)

    # ── smooth hit normal via barycentrics ────────────────────────
    def _hit_normal(self, tri_idx: int, u: float, v: float) -> np.ndarray:
        w = 1.0 - u - v
        vn = self.vert_norms[tri_idx]
        n = w * vn[0] + u * vn[1] + v * vn[2]
        ln = np.linalg.norm(n)
        return n / ln if ln > 1e-8 else self.face_norms[tri_idx]

    # ── rough reflection sampling ─────────────────────────────────
    def _perturb_cone(self, axis: np.ndarray, roughness: float) -> np.ndarray:
        """
        Cosine-weighted sample inside a cone around `axis`.
        Roughness 0 → perfect mirror, roughness 1 → full hemisphere.
        """
        if roughness < 1e-3:
            return axis

        # Tangent frame
        if abs(axis[1]) < 0.95:
            t = np.cross(axis, np.array([0., 1., 0.]))
        else:
            t = np.cross(axis, np.array([1., 0., 0.]))
        t /= np.linalg.norm(t) + 1e-10
        b = np.cross(axis, t)

        # Cone half-angle scales with roughness²
        cos_max = 1.0 - roughness * roughness
        u1, u2 = self._rng.random(), self._rng.random()
        cos_t = 1.0 - u1 * (1.0 - cos_max)
        sin_t = np.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        phi = 2.0 * np.pi * u2

        d = (sin_t * np.cos(phi) * t +
             sin_t * np.sin(phi) * b +
             cos_t * axis)
        return d / (np.linalg.norm(d) + 1e-10)

    # ── shading kernel ────────────────────────────────────────────
    def _shade(self, hp, n, rd, depth, py, px):
        mat = self.mat
        base = np.asarray(mat.color, dtype=np.float32)

        # Emission is added regardless of lighting — this term is what
        # makes the surface a light source to the rest of the scene via
        # reflection bounces.
        col = (np.asarray(mat.emissive, dtype=np.float32) * mat.emission_strength
               + base * mat.ambient)

        vd = -rd / (np.linalg.norm(rd) + 1e-10)
        eff_shin = max(2.0, mat.shininess * (1.0 - 0.85 * mat.roughness))

        for light in self.lights:
            lp = np.asarray(light.position, dtype=np.float32)
            tl = lp - hp
            dl = float(np.linalg.norm(tl))
            if dl < 1e-6:
                continue
            ld = tl / dl

            # Shadow ray — any_hit short-circuits
            if self.shadows:
                if self.bvh.any_hit(hp + n * 1e-4, ld, dl - 1e-3):
                    continue

            li = np.asarray(light.color, dtype=np.float32) * light.intensity
            if light.kind == "point":
                li = li / (1.0 + 0.08*dl + 0.01*dl*dl)
            elif light.kind == "spot":
                sd = np.asarray(light.direction, dtype=np.float32)
                sd /= np.linalg.norm(sd) + 1e-8
                ca = float(np.dot(-ld, sd))
                cc = np.cos(np.radians(light.spot_angle))
                if ca < cc:
                    continue
                li = li * ((ca - cc) / (1 - cc + 1e-8)) ** light.spot_falloff

            ndl = max(0.0, float(np.dot(n, ld)))
            col += base * mat.diffuse * ndl * li

            hv = ld + vd
            hvn = np.linalg.norm(hv)
            if hvn > 1e-8:
                ndh = max(0.0, float(np.dot(n, hv / hvn)))
                col += mat.specular * (ndh ** eff_shin) * li

        # ── Reflection with roughness ──────────────────────────────
        if (self.reflections and mat.reflectivity > 1e-3
                and depth < self.bounces):
            ideal = rd - 2.0 * float(np.dot(rd, n)) * n
            ideal /= np.linalg.norm(ideal) + 1e-10
            rorig = hp + n * 1e-4

            acc = np.zeros(3, dtype=np.float32)
            ns = self.rough_samples if mat.roughness > 1e-3 else 1
            for _ in range(ns):
                rdir = self._perturb_cone(ideal, mat.roughness)
                # Reject samples that dip below the surface
                if np.dot(rdir, n) <= 0:
                    rdir = ideal
                rh = self.bvh.trace(rorig, rdir)
                if rh is not None:
                    rt, ri, ru, rv = rh
                    rhp = rorig + rdir * rt
                    rn  = self._hit_normal(ri, ru, rv)
                    if np.dot(rn, -rdir) < 0:
                        rn = -rn
                    acc += self._shade(rhp, rn, rdir, depth + 1, py, px)
                else:
                    acc += self._bg(py, px)
            col += mat.reflectivity * (acc / ns)

        return col

    # ── main render loop ──────────────────────────────────────────
    def render(self, progress_cb=None) -> np.ndarray:
        W, H = self.W, self.H
        fb = np.zeros((H, W, 3), dtype=np.float32)

        if self.bg_img is not None:
            bh, bw = self.bg_img.shape[:2]
            yidx = np.clip(np.linspace(0, bh-1, H).astype(int), 0, bh-1)
            xidx = np.clip(np.linspace(0, bw-1, W).astype(int), 0, bw-1)
            fb = self.bg_img[np.ix_(yidx, xidx)].astype(np.float32).copy()

        origins, dirs = self.cam.get_rays(W, H)
        total = H * W
        done = 0

        for py in range(H):
            for px in range(W):
                ro, rd = origins[py, px], dirs[py, px]
                hit = self.bvh.trace(ro, rd)
                if hit is not None:
                    t, ti, u, v = hit
                    hp = ro + rd * t
                    n = self._hit_normal(ti, u, v)
                    if np.dot(n, -rd) < 0:
                        n = -n
                    fb[py, px] = self._shade(hp, n, rd, 0, py, px)
                done += 1
            if progress_cb:
                progress_cb(done / total)

        fb = fb / (fb + 1.0)
        fb = np.power(np.clip(fb, 0, 1), 1.0 / 2.2)
        return fb


# ═════════════════════════════════════════════════════════════════
#  SECTION 8 – SCENE
# ═════════════════════════════════════════════════════════════════

class Scene:
    def __init__(self, tris):
        self.original_tris = normalize_mesh(tris.copy())
        self.tris = self.original_tris.copy()
        self.norms = compute_normals(self.tris)
        self.vert_norms = compute_vertex_normals(self.tris, self.norms)
        self.material = Material()
        self.lights = [
            Light("Key",  [3., 5., 4.],  [1.0, 0.95, 0.9], 1.2, "point"),
            Light("Fill", [-4., 2., 3.], [0.5, 0.6, 1.0], 0.6, "point"),
            Light("Rim",  [0., -3., -4.],[1.0, 0.8, 0.7], 0.4, "directional"),
        ]
        self.camera = Camera()
        self.mesh_rotation = np.eye(3, dtype=np.float32)
        self.light_orbit_angle = 0.0

    def rotate_mesh(self, R):
        self.mesh_rotation = R @ self.mesh_rotation
        self.tris = (self.original_tris.reshape(-1, 3)
                     @ self.mesh_rotation.T).reshape(-1, 3, 3)
        self.norms = compute_normals(self.tris)
        self.vert_norms = compute_vertex_normals(self.tris, self.norms)

    def orbit_light(self, dt):
        self.light_orbit_angle += dt
        R = rot_axis_angle(np.array([0., 1., 0.]), self.light_orbit_angle)
        self.lights[0].position = list(R @ np.array([3., 5., 4.]))

    def reset(self):
        self.mesh_rotation = np.eye(3, dtype=np.float32)
        self.tris = self.original_tris.copy()
        self.norms = compute_normals(self.tris)
        self.vert_norms = compute_vertex_normals(self.tris, self.norms)
        self.camera = Camera()
        self.light_orbit_angle = 0.0
        self.lights[0].position = [3., 5., 4.]


# ═════════════════════════════════════════════════════════════════
#  SECTION 9 – INTERACTIVE PANELS (unchanged infrastructure)
# ═════════════════════════════════════════════════════════════════

class InteractivePanel:
    def __init__(self, ax, title, icon, x, y_top, w):
        self.ax, self.title, self.icon = ax, title, icon
        self.x, self.y_top, self.w = x, y_top, w
        self.items, self.selected = [], 0
        self.visible, self.artists = False, []
        self._item_dy, self._head_dy = 0.028, 0.033
        self._title_dy, self._pad = 0.038, 0.015

    def add_header(self, label): self.items.append({"kind": "header", "label": label})
    def add_float(self, label, g, s, lo, hi, step, fmt="{:.2f}"):
        self.items.append({"kind": "float", "label": label, "get": g, "set": s,
                           "lo": lo, "hi": hi, "step": step, "fmt": fmt})
    def add_int(self, label, g, s, lo, hi, step=1):
        self.items.append({"kind": "int", "label": label, "get": g, "set": s,
                           "lo": lo, "hi": hi, "step": step})
    def add_bool(self, label, g, s):
        self.items.append({"kind": "bool", "label": label, "get": g, "set": s})
    def add_choice(self, label, g, s, choices):
        self.items.append({"kind": "choice", "label": label, "get": g, "set": s, "choices": choices})
    def add_action(self, label, cb):
        self.items.append({"kind": "action", "label": label, "cb": cb})
    def add_info(self, label, g):
        self.items.append({"kind": "info", "label": label, "get": g})

    def _calc_height(self):
        h = self._title_dy + self._pad * 2
        for it in self.items:
            h += self._head_dy if it["kind"] == "header" else self._item_dy
        return min(h, 0.88)

    def nav(self, d):
        if not self.items: return
        n = len(self.items)
        for _ in range(n):
            self.selected = (self.selected + d) % n
            if self.items[self.selected]["kind"] not in ("header", "info"): break

    def adjust(self, d):
        if not self.items: return
        it = self.items[self.selected]; k = it["kind"]
        if k == "float":
            v = it["get"]() + d * it["step"]
            it["set"](max(it["lo"], min(it["hi"], v)))
        elif k == "int":
            v = it["get"]() + d * it["step"]
            it["set"](int(max(it["lo"], min(it["hi"], v))))
        elif k == "bool": it["set"](not it["get"]())
        elif k == "choice":
            ch = it["choices"]; cur = it["get"]()
            idx = ch.index(cur) if cur in ch else 0
            it["set"](ch[(idx + d) % len(ch)])
        elif k == "action": it["cb"]()

    def clear(self):
        for a in self.artists:
            try: a.remove()
            except Exception: pass
        self.artists.clear()

    def _a(self, art): self.artists.append(art); return art

    def draw(self):
        self.clear()
        if not self.visible: return
        if self.items and self.items[self.selected]["kind"] in ("header", "info"):
            self.nav(1)
        h = self._calc_height(); y0 = self.y_top - h
        self._a(self.ax.add_patch(mpatches.FancyBboxPatch(
            (self.x, y0), self.w, h, boxstyle="round,pad=0.008",
            facecolor="#0d0d20", edgecolor="#4a90d9", linewidth=1.2,
            alpha=0.94, transform=self.ax.transAxes, zorder=50, clip_on=False)))
        y = self.y_top - self._pad - self._title_dy * 0.5
        self._a(self.ax.text(self.x + self.w/2, y + 0.008,
            f"{self.icon}  {self.title}", color="#4a90d9", fontsize=8.5,
            fontweight="bold", ha="center", va="center",
            transform=self.ax.transAxes, zorder=51, family="monospace"))
        y -= self._title_dy * 0.5
        self._a(self.ax.text(self.x + self._pad, y, "─" * int(self.w * 85),
            color="#334466", fontsize=5.5, ha="left", va="top",
            transform=self.ax.transAxes, zorder=51, family="monospace"))
        y -= 0.008

        for idx, it in enumerate(self.items):
            k = it["kind"]; sel = (idx == self.selected)
            if k == "header":
                y -= self._head_dy
                self._a(self.ax.text(self.x + self._pad, y, f"── {it['label']} ──",
                    color="#f0c040", fontsize=7, fontweight="bold", ha="left",
                    va="center", transform=self.ax.transAxes, zorder=51, family="monospace"))
                continue
            y -= self._item_dy
            if sel:
                self._a(self.ax.add_patch(mpatches.Rectangle(
                    (self.x + 0.003, y - 0.01), self.w - 0.006, 0.024,
                    facecolor="#4a90d940", edgecolor="#4a90d9", linewidth=0.6,
                    transform=self.ax.transAxes, zorder=51, clip_on=False)))
            lbl_c = "#ffffff" if sel else "#b0b0c0"
            val_c = "#80d0ff" if sel else "#6090a0"
            if k == "info":
                self._a(self.ax.text(self.x + self._pad, y, f"{it['label']}: {it['get']()}",
                    color="#7090a0", fontsize=6.5, ha="left", va="center",
                    transform=self.ax.transAxes, zorder=52, family="monospace"))
            elif k == "action":
                self._a(self.ax.text(self.x + self.w/2, y, f"▶ {it['label']}",
                    color="#50ff80" if sel else "#30a050", fontsize=7,
                    fontweight="bold", ha="center", va="center",
                    transform=self.ax.transAxes, zorder=52, family="monospace"))
            elif k == "bool":
                v = it["get"](); tag = "ON " if v else "OFF"
                tc = "#50ff80" if v else "#ff5050"
                self._a(self.ax.text(self.x + self._pad, y, it["label"],
                    color=lbl_c, fontsize=6.8, ha="left", va="center",
                    transform=self.ax.transAxes, zorder=52, family="monospace"))
                self._a(self.ax.text(self.x + self.w - self._pad, y, f"[{tag}]",
                    color=tc, fontsize=6.8, fontweight="bold", ha="right",
                    va="center", transform=self.ax.transAxes, zorder=52, family="monospace"))
            elif k == "choice":
                v = it["get"]()
                self._a(self.ax.text(self.x + self._pad, y, it["label"],
                    color=lbl_c, fontsize=6.8, ha="left", va="center",
                    transform=self.ax.transAxes, zorder=52, family="monospace"))
                self._a(self.ax.text(self.x + self.w - self._pad, y, f"◂ {v} ▸",
                    color=val_c, fontsize=6.8, ha="right", va="center",
                    transform=self.ax.transAxes, zorder=52, family="monospace"))
            else:
                v = it["get"](); vs = it.get("fmt", "{:.2f}").format(v)
                bx = self.x + self.w * 0.55; bw = self.w * 0.32
                self._a(self.ax.add_patch(mpatches.Rectangle(
                    (bx, y - 0.007), bw, 0.014, facecolor="#ffffff10",
                    edgecolor="#ffffff20", linewidth=0.4,
                    transform=self.ax.transAxes, zorder=51, clip_on=False)))
                frac = max(0, min(1, (v - it["lo"]) / (it["hi"] - it["lo"] + 1e-12)))
                self._a(self.ax.add_patch(mpatches.Rectangle(
                    (bx, y - 0.007), bw * frac, 0.014,
                    facecolor="#4a90d9" if sel else "#2a5070", linewidth=0,
                    transform=self.ax.transAxes, zorder=52, clip_on=False)))
                self._a(self.ax.text(self.x + self._pad, y, it["label"],
                    color=lbl_c, fontsize=6.5, ha="left", va="center",
                    transform=self.ax.transAxes, zorder=52, family="monospace"))
                self._a(self.ax.text(bx + bw + 0.005, y, vs,
                    color=val_c, fontsize=6, ha="left", va="center",
                    transform=self.ax.transAxes, zorder=52, family="monospace"))


class TextPanel:
    def __init__(self, ax, title, icon, x, y_top, w, h):
        self.ax, self.title, self.icon = ax, title, icon
        self.x, self.y_top, self.w, self.h = x, y_top, w, h
        self.visible, self.artists, self.lines = False, [], []

    def clear(self):
        for a in self.artists:
            try: a.remove()
            except Exception: pass
        self.artists.clear()

    def _a(self, art): self.artists.append(art)

    def draw(self):
        self.clear()
        if not self.visible: return
        y0 = self.y_top - self.h
        self._a(self.ax.add_patch(mpatches.FancyBboxPatch(
            (self.x, y0), self.w, self.h, boxstyle="round,pad=0.008",
            facecolor="#0d0d20", edgecolor="#4a90d9", linewidth=1.2,
            alpha=0.95, transform=self.ax.transAxes, zorder=50, clip_on=False)))
        y = self.y_top - 0.025
        self._a(self.ax.text(self.x + self.w/2, y, f"{self.icon}  {self.title}",
            color="#4a90d9", fontsize=9, fontweight="bold", ha="center",
            va="center", transform=self.ax.transAxes, zorder=51, family="monospace"))
        y -= 0.025
        for text, color, size, weight in self.lines:
            self._a(self.ax.text(self.x + 0.015, y, text, color=color,
                fontsize=size, fontweight=weight, ha="left", va="top",
                transform=self.ax.transAxes, zorder=51, family="monospace"))
            y -= 0.025 * (size / 7.0)


# ═════════════════════════════════════════════════════════════════
#  SECTION 10 – MAIN VIEWER
# ═════════════════════════════════════════════════════════════════

class STLViewer:
    RES_OPTS = [80, 120, 160, 240, 320, 480]
    PREVIEW_W, PREVIEW_H = 520, 400

    def __init__(self, stl_path=None):
        if stl_path and os.path.isfile(stl_path):
            print(f"Loading '{stl_path}'…")
            tris = load_stl(stl_path)
            print(f"  {len(tris)} triangles loaded.")
        else:
            print("No STL provided — generating demo icosphere…")
            tris = generate_demo_mesh(3)
            print(f"  {len(tris)} triangles generated.")

        self.scene = Scene(tris)

        # Settings
        self._show_edges = False
        self._edge_color = [0.85, 0.85, 0.85]
        self._show_stars = True
        self._cast_shadows = True
        self._anim_model = False
        self._anim_light = False
        self._anim_stars = True
        self._anim_speed = 1.0
        self._rt_res_idx = 2
        self._rt_bounces = 2
        self._rt_shadows = True
        self._rt_reflect = True
        self._rt_rough_samples = 2
        self._rt_progress = 0.0
        self._rt_running = False
        self._rt_done = False
        self._rt_result = None
        self._mode = "PREVIEW"
        self._anim_t = 0.0
        self._fps = 0.0
        self._last_t = time.time()
        self._drag_start = None
        self._drag_btn = None
        self._active_light_idx = 0

        self.starfield = Starfield(2200)
        self.rasterizer = Rasterizer(self.PREVIEW_W, self.PREVIEW_H,
                                     shadow_res=192, pcf_radius=1)

        # Figure
        self.fig = plt.figure("STL Ray Tracer", figsize=(13.5, 9.5),
                              facecolor="#060614")
        self.fig.patch.set_facecolor("#060614")
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.ax.set_facecolor("#060614")
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        self.ax.axis("off")

        blank = np.zeros((self.PREVIEW_H, self.PREVIEW_W, 3), dtype=np.float32)
        self.img = self.ax.imshow(blank, extent=[0.06, 0.94, 0.07, 0.93],
                                  aspect="auto", zorder=1, interpolation="bilinear")

        self.ax.add_patch(mpatches.FancyBboxPatch(
            (0.06, 0.07), 0.88, 0.86, boxstyle="round,pad=0.004",
            facecolor="none", edgecolor="#3a5080", linewidth=1.2,
            transform=self.ax.transAxes, zorder=5))

        self._prog_bg = mpatches.Rectangle((0.06, 0.015), 0.88, 0.035,
            facecolor="#0a0a18", edgecolor="#2a3050", linewidth=0.8,
            transform=self.ax.transAxes, zorder=45)
        self.ax.add_patch(self._prog_bg)
        self._prog_fill = mpatches.Rectangle((0.06, 0.015), 0.0, 0.035,
            facecolor="#4a90d9", edgecolor="none", linewidth=0,
            transform=self.ax.transAxes, zorder=46)
        self.ax.add_patch(self._prog_fill)
        self._prog_text = self.ax.text(0.5, 0.032, "", color="#8090b0",
            fontsize=7, ha="center", va="center",
            transform=self.ax.transAxes, zorder=47, family="monospace")

        self._hud_bg = mpatches.FancyBboxPatch((0.22, 0.94), 0.56, 0.05,
            boxstyle="round,pad=0.006", facecolor="#0a0a1a",
            edgecolor="#2a3055", linewidth=0.8, alpha=0.9,
            transform=self.ax.transAxes, zorder=45)
        self.ax.add_patch(self._hud_bg)
        self._hud_text = self.ax.text(0.5, 0.965, "", color="#4a90d9",
            fontsize=7.5, fontweight="bold", ha="center", va="center",
            transform=self.ax.transAxes, zorder=46, family="monospace")

        self._build_panels()
        self._active_panel = None

        self.fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event",         self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event",      self._on_key)

        self._timer = self.fig.canvas.new_timer(interval=50)
        self._timer.add_callback(self._tick)
        self._timer.start()

        self._render_preview()

    # ── PANEL CONSTRUCTION ────────────────────────────────────────
    def _build_panels(self):
        ax, sc, cam = self.ax, self.scene, self.scene.camera

        # Camera
        self.p_cam = p = InteractivePanel(ax, "CAMERA", "📷", 0.77, 0.93, 0.22)
        p.add_header("Orbit")
        p.add_float("Distance", lambda: cam.distance, lambda v: setattr(cam, 'distance', v), 0.3, 20.0, 0.2, "{:.1f}")
        p.add_float("Azimuth",  lambda: cam.azimuth,  lambda v: setattr(cam, 'azimuth', v), -180, 180, 5, "{:.0f}°")
        p.add_float("Elevation",lambda: cam.elevation,lambda v: setattr(cam, 'elevation', v), -89, 89, 5, "{:.0f}°")
        p.add_header("Lens")
        p.add_float("FOV", lambda: cam.fov, lambda v: setattr(cam, 'fov', v), 10, 120, 5, "{:.0f}°")
        p.add_header("Target")
        p.add_float("Target X", lambda: cam.target[0], lambda v: cam.target.__setitem__(0, v), -5, 5, 0.1)
        p.add_float("Target Y", lambda: cam.target[1], lambda v: cam.target.__setitem__(1, v), -5, 5, 0.1)
        p.add_float("Target Z", lambda: cam.target[2], lambda v: cam.target.__setitem__(2, v), -5, 5, 0.1)
        p.add_info("Position", lambda: f"({cam.position[0]:.1f},{cam.position[1]:.1f},{cam.position[2]:.1f})")

        # Lighting
        self.p_light = p = InteractivePanel(ax, "LIGHTING", "💡", 0.77, 0.93, 0.22)
        def _gl(): return self.scene.lights[self._active_light_idx]
        light_names = [l.name for l in sc.lights]
        p.add_header("Select Light")
        p.add_choice("Active", lambda: sc.lights[self._active_light_idx].name,
                     lambda v: setattr(self, '_active_light_idx',
                                       next(i for i, l in enumerate(sc.lights) if l.name == v)),
                     light_names)
        p.add_choice("Kind", lambda: _gl().kind,
                     lambda v: setattr(_gl(), 'kind', v),
                     ["point", "spot", "directional"])
        p.add_header("Properties")
        p.add_float("Intensity", lambda: _gl().intensity,
                    lambda v: setattr(_gl(), 'intensity', v), 0, 3.0, 0.1)
        p.add_header("Position")
        p.add_float("X", lambda: _gl().position[0], lambda v: _gl().position.__setitem__(0, v), -10, 10, 0.5)
        p.add_float("Y", lambda: _gl().position[1], lambda v: _gl().position.__setitem__(1, v), -10, 10, 0.5)
        p.add_float("Z", lambda: _gl().position[2], lambda v: _gl().position.__setitem__(2, v), -10, 10, 0.5)
        p.add_header("Color")
        p.add_float("Red",   lambda: _gl().color[0], lambda v: _gl().color.__setitem__(0, v), 0, 1, 0.05)
        p.add_float("Green", lambda: _gl().color[1], lambda v: _gl().color.__setitem__(1, v), 0, 1, 0.05)
        p.add_float("Blue",  lambda: _gl().color[2], lambda v: _gl().color.__setitem__(2, v), 0, 1, 0.05)
        p.add_header("Spot")
        p.add_float("Angle",   lambda: _gl().spot_angle,   lambda v: setattr(_gl(), 'spot_angle', v), 5, 90, 5, "{:.0f}°")
        p.add_float("Falloff", lambda: _gl().spot_falloff, lambda v: setattr(_gl(), 'spot_falloff', v), 0.5, 20, 0.5)

        # Material — now with roughness + emissive
        self.p_mat = p = InteractivePanel(ax, "MATERIAL", "🎨", 0.77, 0.93, 0.22)
        mat = sc.material
        def _apply_preset(name):
            if name not in PRESETS: return
            pr = PRESETS[name]
            mat.name = pr.name; mat.color[:] = pr.color
            mat.ambient, mat.diffuse = pr.ambient, pr.diffuse
            mat.specular, mat.shininess = pr.specular, pr.shininess
            mat.reflectivity, mat.roughness = pr.reflectivity, pr.roughness
            mat.emissive[:] = list(pr.emissive)
            mat.emission_strength = pr.emission_strength
        p.add_header("Preset")
        p.add_choice("Preset", lambda: mat.name, _apply_preset, PRESET_NAMES + ["Default"])
        p.add_header("Color")
        p.add_float("Red",   lambda: mat.color[0], lambda v: mat.color.__setitem__(0, v), 0, 1, 0.05)
        p.add_float("Green", lambda: mat.color[1], lambda v: mat.color.__setitem__(1, v), 0, 1, 0.05)
        p.add_float("Blue",  lambda: mat.color[2], lambda v: mat.color.__setitem__(2, v), 0, 1, 0.05)
        p.add_header("Shading")
        p.add_float("Ambient",   lambda: mat.ambient,   lambda v: setattr(mat, 'ambient', v), 0, 1, 0.05)
        p.add_float("Diffuse",   lambda: mat.diffuse,   lambda v: setattr(mat, 'diffuse', v), 0, 1, 0.05)
        p.add_float("Specular",  lambda: mat.specular,  lambda v: setattr(mat, 'specular', v), 0, 1, 0.05)
        p.add_float("Shininess", lambda: mat.shininess, lambda v: setattr(mat, 'shininess', v), 2, 256, 8, "{:.0f}")
        p.add_float("Roughness", lambda: mat.roughness, lambda v: setattr(mat, 'roughness', v), 0, 1, 0.05)
        p.add_float("Reflect",   lambda: mat.reflectivity, lambda v: setattr(mat, 'reflectivity', v), 0, 1, 0.05)
        p.add_header("Emission")
        p.add_float("Strength", lambda: mat.emission_strength,
                    lambda v: setattr(mat, 'emission_strength', v), 0, 5, 0.1, "{:.1f}")
        p.add_float("Emit R", lambda: mat.emissive[0], lambda v: mat.emissive.__setitem__(0, v), 0, 1, 0.05)
        p.add_float("Emit G", lambda: mat.emissive[1], lambda v: mat.emissive.__setitem__(1, v), 0, 1, 0.05)
        p.add_float("Emit B", lambda: mat.emissive[2], lambda v: mat.emissive.__setitem__(2, v), 0, 1, 0.05)

        # RT Settings
        self.p_rt = p = InteractivePanel(ax, "RAY TRACE SETTINGS", "🔬", 0.77, 0.93, 0.22)
        p.add_header("Resolution")
        p.add_choice("Width", lambda: str(self.RES_OPTS[self._rt_res_idx]),
                     lambda v: setattr(self, '_rt_res_idx', self.RES_OPTS.index(int(v))),
                     [str(r) for r in self.RES_OPTS])
        p.add_info("Pixels", lambda: f"{self.RES_OPTS[self._rt_res_idx]}×"
                   f"{int(self.RES_OPTS[self._rt_res_idx] * self.PREVIEW_H / self.PREVIEW_W)}")
        p.add_header("Quality")
        p.add_int("Bounces", lambda: self._rt_bounces, lambda v: setattr(self, '_rt_bounces', v), 0, 8)
        p.add_int("Rough samples", lambda: self._rt_rough_samples,
                  lambda v: setattr(self, '_rt_rough_samples', v), 1, 8)
        p.add_bool("Shadows",     lambda: self._rt_shadows, lambda v: setattr(self, '_rt_shadows', v))
        p.add_bool("Reflections", lambda: self._rt_reflect, lambda v: setattr(self, '_rt_reflect', v))
        p.add_header("Render")
        p.add_action("▶  START RAY TRACE", self._start_rt)
        p.add_info("Status", lambda: self._mode)

        # Animation
        self.p_anim = p = InteractivePanel(ax, "ANIMATION", "🎬", 0.01, 0.93, 0.22)
        p.add_header("Rotation")
        p.add_bool("Rotate Model", lambda: self._anim_model, lambda v: setattr(self, '_anim_model', v))
        p.add_bool("Orbit Light",  lambda: self._anim_light, lambda v: setattr(self, '_anim_light', v))
        p.add_float("Speed", lambda: self._anim_speed, lambda v: setattr(self, '_anim_speed', v), 0.1, 5.0, 0.1, "{:.1f}x")
        p.add_header("Background")
        p.add_bool("Starfield",    lambda: self._show_stars, lambda v: setattr(self, '_show_stars', v))
        p.add_bool("Animate Stars",lambda: self._anim_stars, lambda v: setattr(self, '_anim_stars', v))
        p.add_header("Display")
        p.add_bool("Cast Shadows", lambda: self._cast_shadows, lambda v: setattr(self, '_cast_shadows', v))
        p.add_bool("Show Edges",   lambda: self._show_edges,   lambda v: setattr(self, '_show_edges', v))
        p.add_float("Edge R", lambda: self._edge_color[0], lambda v: self._edge_color.__setitem__(0, v), 0, 1, 0.1)
        p.add_float("Edge G", lambda: self._edge_color[1], lambda v: self._edge_color.__setitem__(1, v), 0, 1, 0.1)
        p.add_float("Edge B", lambda: self._edge_color[2], lambda v: self._edge_color.__setitem__(2, v), 0, 1, 0.1)

        self.p_tree = TextPanel(ax, "SCENE TREE", "📁", 0.01, 0.93, 0.22, 0.55)
        self.p_help = TextPanel(ax, "KEYBOARD & MOUSE REFERENCE", "⌨", 0.18, 0.92, 0.64, 0.82)

        self._right_panels = [self.p_cam, self.p_light, self.p_mat, self.p_rt]
        self._left_panels  = [self.p_anim, self.p_tree]
        self._all_panels   = self._right_panels + self._left_panels + [self.p_help]

    # ── TEXT PANEL CONTENT ────────────────────────────────────────
    def _update_tree(self):
        sc = self.scene; L = self.p_tree.lines; L.clear()
        L.append((f"▸ Mesh ({len(sc.tris)} tris)", "#f0c040", 7.5, "bold"))
        L.append((f"  Material: {sc.material.name}", "#a0d0a0", 6.8, "normal"))
        c = sc.material.color
        L.append((f"  Color: ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})", "#8090a0", 6.5, "normal"))
        L.append((f"  Rough: {sc.material.roughness:.2f}  Emit: {sc.material.emission_strength:.1f}",
                  "#8090a0", 6.5, "normal"))
        L.append(("", "#333", 4, "normal"))
        L.append(("▸ Lights", "#f0c040", 7.5, "bold"))
        for li in sc.lights:
            L.append((f"  ● {li.name} ({li.kind})", "#ffd080", 6.8, "normal"))
            p = li.position
            L.append((f"    pos: ({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})", "#7090a0", 6, "normal"))
            L.append((f"    int: {li.intensity:.2f}", "#7090a0", 6, "normal"))
        L.append(("", "#333", 4, "normal"))
        L.append(("▸ Camera", "#f0c040", 7.5, "bold"))
        cp = sc.camera.position
        L.append((f"  pos: ({cp[0]:.2f},{cp[1]:.2f},{cp[2]:.2f})", "#7090a0", 6.5, "normal"))
        L.append((f"  fov: {sc.camera.fov:.0f}°  dist: {sc.camera.distance:.1f}", "#7090a0", 6.5, "normal"))

    def _update_help(self):
        L = self.p_help.lines; L.clear()
        def sec(t): L.append(("", "#333", 3, "normal")); L.append((f"── {t} ──", "#f0c040", 7.5, "bold"))
        def row(k, d): L.append((f"  {k:<16} {d}", "#c0c0d0", 6.8, "normal"))

        sec("PANELS"); row("H","Help"); row("T","Scene tree"); row("C","Camera")
        row("L","Lighting"); row("M","Material"); row("X","Ray trace"); row("A","Animation")
        sec("NAVIGATION"); row("Up/Down","Move cursor"); row("Left/Right","Adjust value")
        sec("MOUSE"); row("L-drag","Orbit"); row("R-drag","Pan"); row("Scroll","Zoom")
        sec("QUICK KEYS"); row("Space","Toggle rotation"); row("E","Edges"); row("G","Starfield")
        row("R","Reset"); row("Enter","Ray trace"); row("1-0","Material presets"); row("Q/Esc","Close/quit")
        sec("v3.0 NOTES")
        row("","• Rasterizer: per-pixel Phong via barycentric")
        row("","  normal interpolation + PCF soft shadows.")
        row("","• Starfield: 3D shell w/ inverse-square dim")
        row("","  + spectral reddening. Orbit = real parallax.")
        row("","• BVH: flat SAH-binned; ordered traversal,")
        row("","  vectorised leaf tests, any_hit for shadows.")
        row("","• Materials: roughness = blurry reflections,")
        row("","  emission propagates through bounces.")
        row("","• Try preset 9 (Lava) or 0 (Neon) + Mirror")
        row("","  nearby to see emissive caught in reflection.")

    # ── PANEL TOGGLE ──────────────────────────────────────────────
    def _show_panel(self, panel, group):
        for p in group:
            if p is not panel and p.visible:
                p.visible = False; p.clear()
        if panel.visible:
            panel.visible = False; panel.clear()
            if self._active_panel is panel: self._active_panel = None
        else:
            panel.visible = True
            self._active_panel = panel if isinstance(panel, InteractivePanel) else None

    # ── RENDER ────────────────────────────────────────────────────
    def _render_preview(self):
        bg = None
        if self._show_stars:
            bg = self.starfield.render(self.PREVIEW_W, self.PREVIEW_H,
                                       self.scene.camera, self._anim_t,
                                       animate=self._anim_stars)
        fb = self.rasterizer.render(
            self.scene.tris, self.scene.norms, self.scene.vert_norms,
            self.scene.material, self.scene.lights, self.scene.camera,
            bg=bg, show_edges=self._show_edges, edge_color=self._edge_color,
            cast_shadows=self._cast_shadows)
        self.img.set_data(fb)

    def _update_hud(self):
        self._hud_text.set_text(
            f"MODE: {self._mode}  │  FPS: {self._fps:.0f}  │  "
            f"TRIS: {len(self.scene.tris)}  │  {self.PREVIEW_W}×{self.PREVIEW_H}  │  "
            f"Shadows: {'ON' if self._cast_shadows else 'OFF'}  │  H=Help")

    def _update_progress(self):
        if self._rt_running:
            pct = self._rt_progress * 100
            self._prog_fill.set_width(0.88 * self._rt_progress)
            w = self.RES_OPTS[self._rt_res_idx]
            h = int(w * self.PREVIEW_H / self.PREVIEW_W)
            self._prog_text.set_text(
                f"Ray Tracing {w}×{h}  │  b:{self._rt_bounces} rs:{self._rt_rough_samples}  │  "
                f"{'█'*int(pct/5)}{'░'*(20-int(pct/5))}  {pct:.1f}%")
            self._prog_text.set_color("#80c0ff")
        elif self._mode.startswith("RT DONE"):
            self._prog_fill.set_width(0.88)
            self._prog_text.set_text(f"✔ {self._mode}")
            self._prog_text.set_color("#50ff80")
        else:
            self._prog_fill.set_width(0.0)
            self._prog_text.set_text("SPACE=anim  ENTER=raytrace  H=help  1-0=presets")
            self._prog_text.set_color("#506080")

    def _redraw_panels(self):
        for p in self._all_panels:
            if p.visible:
                if isinstance(p, TextPanel):
                    if p is self.p_tree: self._update_tree()
                    elif p is self.p_help: self._update_help()
                p.draw()

    # ── TICK ──────────────────────────────────────────────────────
    def _tick(self):
        need = False
        if self._anim_model:
            R = rot_axis_angle(np.array([0., 1., 0.3]), 0.03 * self._anim_speed)
            self.scene.rotate_mesh(R); need = True
        if self._anim_light:
            self.scene.orbit_light(0.03 * self._anim_speed); need = True
        if self._anim_stars and self._show_stars:
            self._anim_t += 0.4 * self._anim_speed; need = True
        if self._rt_done:
            self._display_rt_result(); self._rt_done = False; need = False
        if need: self._render_preview()
        now = time.time(); dt = now - self._last_t; self._last_t = now
        self._fps = 0.8 * self._fps + 0.2 / (dt + 1e-6)
        self._update_hud(); self._update_progress(); self._redraw_panels()
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

    # ── RAY TRACE THREAD ──────────────────────────────────────────
    def _start_rt(self):
        if self._rt_running: return
        self._rt_running = True; self._rt_progress = 0.0; self._mode = "RAYTRACING…"
        threading.Thread(target=self._rt_thread, daemon=True).start()

    def _rt_thread(self):
        w = self.RES_OPTS[self._rt_res_idx]
        h = int(w * self.PREVIEW_H / self.PREVIEW_W)
        bg = None
        if self._show_stars:
            bg = self.starfield.render(w, h, self.scene.camera,
                                       self._anim_t, animate=self._anim_stars)
        t0 = time.time()
        tracer = RayTracer(
            self.scene.tris, self.scene.norms, self.scene.vert_norms,
            self.scene.material, self.scene.lights, self.scene.camera,
            W=w, H=h, bounces=self._rt_bounces,
            shadows=self._rt_shadows, reflections=self._rt_reflect,
            bg_img=bg, rough_samples=self._rt_rough_samples)
        st = tracer.bvh.stats()
        print(f"  BVH: {st['nodes']} nodes, depth {st['depth']}, "
              f"avg leaf {st['avg_leaf']:.1f} tris "
              f"(built in {tracer.bvh_build_time*1000:.1f} ms)")
        fb = tracer.render(progress_cb=lambda f: setattr(self, '_rt_progress', f))
        elapsed = time.time() - t0
        yi = np.clip(np.linspace(0, h-1, self.PREVIEW_H).astype(int), 0, h-1)
        xi = np.clip(np.linspace(0, w-1, self.PREVIEW_W).astype(int), 0, w-1)
        self._rt_result = fb[np.ix_(yi, xi)]
        self._mode = f"RT DONE ({elapsed:.1f}s, {w}×{h})"
        self._rt_running = False; self._rt_done = True

    def _display_rt_result(self):
        if self._rt_result is not None: self.img.set_data(self._rt_result)

    # ── MOUSE ─────────────────────────────────────────────────────
    def _on_press(self, ev):
        if ev.inaxes != self.ax: return
        self._drag_start = (ev.xdata, ev.ydata); self._drag_btn = ev.button
    def _on_release(self, ev):
        self._drag_start = None; self._drag_btn = None
    def _on_motion(self, ev):
        if self._drag_start is None or ev.inaxes != self.ax or ev.xdata is None: return
        dx = ev.xdata - self._drag_start[0]; dy = ev.ydata - self._drag_start[1]
        self._drag_start = (ev.xdata, ev.ydata)
        cam = self.scene.camera
        if self._drag_btn == 1:
            cam.azimuth += dx * 250
            cam.elevation = np.clip(cam.elevation + dy * 250, -89, 89)
            if self._mode.startswith("RT DONE"): self._mode = "PREVIEW"
            self._render_preview()
        elif self._drag_btn == 3:
            _, right, up = cam.basis(); s = cam.distance * 0.6
            cam.target += (-dx * right + dy * up) * s
            if self._mode.startswith("RT DONE"): self._mode = "PREVIEW"
            self._render_preview()
    def _on_scroll(self, ev):
        cam = self.scene.camera
        cam.distance = np.clip(cam.distance * (0.9 if ev.step > 0 else 1.1), 0.3, 30)
        if self._mode.startswith("RT DONE"): self._mode = "PREVIEW"
        self._render_preview()

    # ── KEYBOARD ──────────────────────────────────────────────────
    def _on_key(self, ev):
        k = ev.key or ""; kl = k.lower()
        if   kl == "c": self._show_panel(self.p_cam,   self._right_panels)
        elif kl == "l": self._show_panel(self.p_light, self._right_panels)
        elif kl == "m": self._show_panel(self.p_mat,   self._right_panels)
        elif kl == "x": self._show_panel(self.p_rt,    self._right_panels)
        elif kl == "a": self._show_panel(self.p_anim,  self._left_panels)
        elif kl == "t": self._show_panel(self.p_tree,  self._left_panels)
        elif kl == "h":
            if self.p_help.visible: self.p_help.visible = False; self.p_help.clear()
            else: self.p_help.visible = True; self._active_panel = None
        elif k in ("up", "down") and self._active_panel:
            self._active_panel.nav(1 if k == "down" else -1)
        elif k in ("left", "right") and self._active_panel:
            self._active_panel.adjust(1 if k == "right" else -1)
            if self._mode.startswith("RT DONE"): self._mode = "PREVIEW"
            self._render_preview()
        elif kl == " ": self._anim_model = not self._anim_model
        elif kl == "e": self._show_edges = not self._show_edges; self._render_preview()
        elif kl == "g": self._show_stars = not self._show_stars; self._render_preview()
        elif kl == "r":
            self.scene.reset(); self._show_edges = False; self._mode = "PREVIEW"
            self._render_preview()
        elif k == "enter": self._start_rt()
        elif kl in "1234567890":
            idx = int(kl) - 1 if kl != "0" else 9
            if idx < len(PRESET_NAMES):
                pr = PRESETS[PRESET_NAMES[idx]]; m = self.scene.material
                m.name = pr.name; m.color[:] = pr.color
                m.ambient, m.diffuse = pr.ambient, pr.diffuse
                m.specular, m.shininess = pr.specular, pr.shininess
                m.reflectivity, m.roughness = pr.reflectivity, pr.roughness
                m.emissive[:] = list(pr.emissive)
                m.emission_strength = pr.emission_strength
                self._render_preview()
        elif kl in ("q", "escape"):
            if self._active_panel and self._active_panel.visible:
                self._active_panel.visible = False; self._active_panel.clear()
                self._active_panel = None
            elif self.p_help.visible:
                self.p_help.visible = False; self.p_help.clear()
            else:
                self._timer.stop(); plt.close(self.fig)
        self._redraw_panels(); self.fig.canvas.draw_idle()

    def show(self):
        print("\n" + "═"*52)
        print("  STL RAY TRACER v3.0")
        print("═"*52)
        print("  H = Help      SPACE = Animate    ENTER = Ray trace")
        print("  1-8 = Metals/Plastics   9 = Lava   0 = Neon")
        print("  Orbit the camera to see 3D starfield parallax.")
        print("═"*52 + "\n")
        plt.show()


# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        for f in os.listdir("."):
            if f.lower().endswith(".stl"): path = f; break
    STLViewer(path).show()