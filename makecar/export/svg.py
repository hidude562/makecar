"""SVG blueprint exporter: multi-view orthographic engineering sheets (vector only).

Each view is a painter's-algorithm flat-shaded rendering.  Faces are projected
orthographically, depth-sorted far-to-near and written as ``<polygon>`` elements
whose colour is the material colour times a Lambert term (translucent materials
use ``fill-opacity``).  Silhouette, crease (dihedral > 35 deg) and boundary edges
are written as ``<path>`` elements *interleaved* with the polygons at the depth
rank of their nearer face, so nearer faces occlude them: a cheap hidden-line
approximation that costs nothing extra.  Coordinates are written as integers in
tenths of a pixel inside a ``scale(0.1)`` group, which keeps the file compact.

Public API
    write_blueprint(path, mesh, connectors=None, measurements=None, ...)  -> Path
    build_blueprint(mesh, ...)                                            -> (svg_text, info)
    write_connector_map(path, mesh, connectors, title="")                 -> Path
    build_connector_map(mesh, connectors, ...)                            -> (svg_text, info)

World frame: +X forward, +Y left, +Z up, ground at z = 0.  All views put the
nose on the LEFT (side/plan); the side view is the car's left side (camera at +Y).
"""
from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..geometry.mesh import Mesh, Material
from ..connectors import Connector, PointConnector, PolygonConnector

__all__ = ["write_blueprint", "build_blueprint", "write_connector_map", "build_connector_map", "THEMES"]

# ----------------------------------------------------------------------------- views
# Rows of R are (screen-right, screen-up, toward-camera) expressed in world coordinates.
# All are proper rotations (det = +1) so face winding is preserved.
_R_SIDE = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])     # left side, nose left, camera at +Y
_R_TOP = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])     # plan, nose left, camera at +Z
_R_FRONT = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])     # camera at +X looking aft
_R_REAR = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])    # camera at -X looking forward

_COS_CREASE = math.cos(math.radians(35.0))
_SHADE_LEVELS = 20
_SHEET_W, _SHEET_H = 1600, 1100
_SHEET_MM_W = 420.0                                   # printed width (A3 landscape) -> exact printed scale
_STD_SCALES = (5, 10, 12.5, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200)

THEMES = {
    "blueprint": dict(
        bg="#1d3b66", grid="#ffffff", grid_minor_op=0.055, grid_major_op=0.11,
        frame="#cfe6fa", ink="#eaf6ff", ink_dim="#b6d6ef", ink_faint="#7fa9cc",
        edge="#f4fbff", edge_op=0.85, seam="#ffffff", seam_op=0.75,
        dim="#a9dcff", dim_text="#eaf6ff", ground="#cfe6fa", centre="#8fc7f2",
        title_bg="#173257", label_bg="#1d3b66",
        conn=dict(point="#ffd23f", circle="#4de3ff", rectangle="#ff5fe0", polygon="#5dff8f"),
        ghost_fill_op=0.22, ghost_edge_op=0.55,
    ),
    "paper": dict(
        bg="#fbfbf8", grid="#000000", grid_minor_op=0.045, grid_major_op=0.09,
        frame="#222222", ink="#1b1b1b", ink_dim="#4a4a4a", ink_faint="#8a8a8a",
        edge="#141414", edge_op=0.9, seam="#101010", seam_op=0.8,
        dim="#1f4e8c", dim_text="#1f4e8c", ground="#333333", centre="#1f4e8c",
        title_bg="#f1f1ea", label_bg="#fbfbf8",
        conn=dict(point="#c98f00", circle="#0a8ea8", rectangle="#b8179a", polygon="#1e9e3e"),
        ghost_fill_op=0.16, ghost_edge_op=0.6,
    ),
}

_FONT = "'Roboto Condensed','Arial Narrow','Helvetica Neue',Arial,sans-serif"
_MONO = "'DejaVu Sans Mono','Menlo','Consolas',monospace"


# ----------------------------------------------------------------------------- helpers
def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / max(np.linalg.norm(v), 1e-12)


def _hex(rgb) -> str:
    r, g, b = (int(round(float(min(max(c, 0.0), 1.0)) * 255)) for c in rgb)
    return "#%02x%02x%02x" % (r, g, b)


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _fmt(v: float) -> str:
    """Compact float for sheet-space coordinates (1 decimal)."""
    s = "%.1f" % v
    return s[:-2] if s.endswith(".0") else s


def _text_width(text: str, size: float, weight: str = "normal", spacing: float = 0.0) -> float:
    k = 0.62 if weight == "bold" else 0.54
    return k * size * len(text) + spacing * max(len(text) - 1, 0)


# ----------------------------------------------------------------------------- mesh data
class _MeshData:
    """Per-mesh precomputation shared by all views: faces grouped by polygon size,
    normals, centroids, material table and the edge table with adjacency."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.V = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
        faces = mesh.faces
        self.n = len(faces)
        sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=self.n) if self.n else np.zeros(0, np.int64)
        self.groups: List[Tuple[np.ndarray, np.ndarray]] = []
        for s in np.unique(sizes):
            if s < 3:
                continue
            idx = np.flatnonzero(sizes == s)
            F = np.asarray([faces[i] for i in idx], dtype=np.int64)
            self.groups.append((idx, F))
        # normals (Newell) + centroids
        self.normals = np.zeros((self.n, 3))
        self.centroids = np.zeros((self.n, 3))
        V = self.V
        for idx, F in self.groups:
            P = V[F]
            Q = np.roll(P, -1, axis=1)
            nx = np.sum((P[:, :, 1] - Q[:, :, 1]) * (P[:, :, 2] + Q[:, :, 2]), axis=1)
            ny = np.sum((P[:, :, 2] - Q[:, :, 2]) * (P[:, :, 0] + Q[:, :, 0]), axis=1)
            nz = np.sum((P[:, :, 0] - Q[:, :, 0]) * (P[:, :, 1] + Q[:, :, 1]), axis=1)
            nn = np.column_stack([nx, ny, nz])
            self.normals[idx] = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
            self.centroids[idx] = P.mean(axis=1)
        # materials
        names = sorted(set(mesh.face_materials) | set(mesh.materials))
        lut = {nm: i for i, nm in enumerate(names)}
        mats = [mesh.materials.get(nm, Material(nm)) for nm in names]
        self.mat_color = np.array([m.color for m in mats], dtype=float).reshape(-1, 3) if mats else np.zeros((0, 3))
        self.mat_alpha = np.array([float(m.alpha) for m in mats]) if mats else np.zeros(0)
        self.mat_emis = np.array([float(getattr(m, "emissive", 0.0)) for m in mats]) if mats else np.zeros(0)
        self.mat_index = np.fromiter((lut[nm] for nm in mesh.face_materials), dtype=np.int64, count=self.n)
        self._build_edges()
        self._vn: Optional[np.ndarray] = None

    def _build_edges(self):
        if not self.groups:
            self.E = np.zeros((0, 2), np.int64)
            self.EF = np.zeros((0, 2), np.int64)
            self.crease = np.zeros(0, bool)
            self.same_dir = np.zeros(0, bool)
            self.flipped = np.zeros(self.n, bool)
            return
        a_, b_, f_ = [], [], []
        for idx, F in self.groups:
            a_.append(F.ravel())
            b_.append(np.roll(F, -1, axis=1).ravel())
            f_.append(np.repeat(idx, F.shape[1]))
        a, b, f = np.concatenate(a_), np.concatenate(b_), np.concatenate(f_)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        ok = lo != hi
        lo, hi, f = lo[ok], hi[ok], f[ok]
        key = lo * np.int64(len(self.V)) + hi
        order = np.argsort(key, kind="stable")
        ks = key[order]
        first = np.r_[True, ks[1:] != ks[:-1]]
        starts = np.flatnonzero(first)
        counts = np.diff(np.r_[starts, len(ks)])
        i1 = order[starts]
        i2 = order[np.minimum(starts + 1, len(ks) - 1)]
        f1 = f[i1]
        f2 = np.where(counts >= 2, f[i2], -1)
        self.E = np.column_stack([lo[i1], hi[i1]])
        self.EF = np.column_stack([f1, f2])
        # do both faces traverse the edge in the same direction?  (consistent meshes: opposite)
        d1 = a[i1] == lo[i1]
        d2 = a[i2] == lo[i2]
        self.same_dir = (counts == 2) & (d1 == d2)
        self._orient()
        dots = np.einsum("ij,ij->i", self.normals[f1], self.normals[np.maximum(f2, 0)])
        self.crease = (f2 >= 0) & (dots < _COS_CREASE)

    def _orient(self):
        """Make face orientation consistent per connected component (parity propagation over
        shared edges), then pick the outward sign from the signed volume (majority vote for open
        sheets).  Robust to meshes whose caps or patches are wound the wrong way."""
        n = self.n
        EF = self.EF
        manifold = EF[:, 1] >= 0
        if n == 0 or not manifold.any():
            self.flipped = np.zeros(n, bool)
            return
        f1, f2 = EF[manifold, 0], EF[manifold, 1]
        par = np.where(self.same_dir[manifold], -1, 1).astype(np.int8)
        src = np.concatenate([f1, f2])
        dst = np.concatenate([f2, f1])
        pp = np.concatenate([par, par])
        o = np.argsort(src, kind="stable")
        src, dst, pp = src[o], dst[o], pp[o]
        indptr = np.zeros(n + 1, np.int64)
        np.add.at(indptr, src + 1, 1)
        indptr = np.cumsum(indptr)
        nbr, parl, ptr = dst.tolist(), pp.tolist(), indptr.tolist()
        sign = [0] * n
        comp = [-1] * n
        ncomp = 0
        for seed in range(n):
            if sign[seed]:
                continue
            sign[seed] = 1
            comp[seed] = ncomp
            stack = [seed]
            while stack:
                fi = stack.pop()
                sf = sign[fi]
                for k in range(ptr[fi], ptr[fi + 1]):
                    g = nbr[k]
                    if not sign[g]:
                        sign[g] = sf * parl[k]
                        comp[g] = ncomp
                        stack.append(g)
            ncomp += 1
        sign = np.asarray(sign, np.int64)
        comp = np.asarray(comp, np.int64)
        # signed volume contribution per face (fan tetrahedra from the origin)
        vol = np.zeros(n)
        V = self.V
        for idx, F in self.groups:
            P = V[F]
            for k in range(1, F.shape[1] - 1):
                vol[idx] += np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, k], P[:, k + 1])) / 6.0
        cvol = np.bincount(comp, weights=vol * sign, minlength=ncomp)
        cmaj = np.bincount(comp, weights=sign, minlength=ncomp)
        cscale = np.bincount(comp, weights=np.abs(vol), minlength=ncomp)
        closed = np.abs(cvol) > 0.05 * np.maximum(cscale, 1e-12)
        gsign = np.where(closed, np.sign(cvol), np.where(cmaj < 0, -1.0, 1.0))
        self.flipped = (sign * gsign[comp]) < 0
        if self.flipped.any():
            self.normals[self.flipped] *= -1.0

    def vertex_normals(self) -> np.ndarray:
        if self._vn is None:
            vn = np.zeros_like(self.V)
            for idx, F in self.groups:
                np.add.at(vn, F.ravel(), np.repeat(self.normals[idx], F.shape[1], axis=0))
            self._vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        return self._vn

    def face_area2d(self, P2: np.ndarray) -> np.ndarray:
        """Signed projected area per face for 2-D vertex positions P2 (nv,2)."""
        area = np.zeros(self.n)
        for idx, F in self.groups:
            P = P2[F]
            Q = np.roll(P, -1, axis=1)
            area[idx] = 0.5 * np.sum(P[:, :, 0] * Q[:, :, 1] - Q[:, :, 0] * P[:, :, 1], axis=1)
        return area


# ----------------------------------------------------------------------------- style registry
class _Styles:
    """Collects fill classes so polygons carry a short class name instead of colours."""

    def __init__(self):
        self.classes: Dict[Tuple, str] = {}
        self.rules: List[str] = []

    def fill(self, hexcol: str, alpha: float, layer_op: float = 1.0) -> str:
        key = (hexcol, round(alpha, 3), round(layer_op, 3))
        c = self.classes.get(key)
        if c is None:
            c = "f%d" % len(self.classes)
            self.classes[key] = c
            op = alpha * layer_op
            if op >= 0.999:
                self.rules.append(".%s{fill:%s;stroke:%s;stroke-width:4;stroke-linejoin:round}" % (c, hexcol, hexcol))
            else:  # translucent: a same-colour stroke would draw at full opacity and read as a grid
                self.rules.append(".%s{fill:%s;fill-opacity:%.3g;stroke:none}" % (c, hexcol, op))
        return c

    def stroke(self, hexcol: str, width_px: float, opacity: float) -> str:
        key = ("s", hexcol, round(width_px, 3), round(opacity, 3))
        c = self.classes.get(key)
        if c is None:
            c = "e%d" % len(self.classes)
            self.classes[key] = c
            self.rules.append(".%s{fill:none;stroke:%s;stroke-width:%.3g;stroke-opacity:%.3g;stroke-linecap:round;stroke-linejoin:round}"
                              % (c, hexcol, width_px * 10.0, opacity))
        return c

    def css(self) -> str:
        return "\n".join(self.rules)


# ----------------------------------------------------------------------------- view rendering
class _Layer:
    """One mesh (or subset of it) to draw in a view."""

    def __init__(self, data: _MeshData, face_mask: Optional[np.ndarray] = None, two_sided: bool = False,
                 fill_op: float = 1.0, edge_op: Optional[float] = None, edge_width: float = 0.55, shade: bool = True):
        self.data = data
        self.mask = np.ones(data.n, bool) if face_mask is None else np.asarray(face_mask, bool)
        self.two_sided = two_sided
        self.fill_op = fill_op
        self.edge_op = edge_op
        self.edge_width = edge_width
        self.shade = shade


class _Viewport:
    """Maps view-space metres (right, up) to sheet pixels."""

    def __init__(self, R: np.ndarray, S: float, x0: float, y0: float, sx_min: float, sy_max: float):
        self.R, self.S, self.x0, self.y0, self.sx_min, self.sy_max = R, S, x0, y0, sx_min, sy_max

    def to_sheet(self, pts_world) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_world, dtype=float)) @ self.R.T
        x = self.x0 + (p[:, 0] - self.sx_min) * self.S
        y = self.y0 + (self.sy_max - p[:, 1]) * self.S
        return np.column_stack([x, y])

    @property
    def cam_dir(self) -> np.ndarray:
        return self.R[2]


def _shade_factor(normals_view: np.ndarray, front: np.ndarray, light=(-0.45, 0.62, 0.64)) -> np.ndarray:
    L = _unit(light)
    n = normals_view * np.where(front, 1.0, -1.0)[:, None]        # two-sided lighting
    ndl = np.clip(n @ L, 0.0, 1.0)
    k = 0.40 + 0.60 * ndl
    k *= 0.86 + 0.14 * np.clip(n[:, 2], 0.0, 1.0)                  # faces turned toward the camera a bit brighter
    k = np.where(front, k, k * 0.82)
    return k


def _render_view(vp: _Viewport, layers: Sequence[_Layer], styles: _Styles, theme: dict,
                 min_area_px: float = 0.15, area_thresholds: Optional[Dict[int, float]] = None) -> Tuple[str, int, int]:
    """Painter's-algorithm rendering of the layers into one <g>. Returns (svg, n_polys, n_edges)."""
    R, S = vp.R, vp.S
    poly_items: List[str] = []          # polygon element strings
    poly_keys: List[np.ndarray] = []
    edge_items: List[str] = []
    edge_keys: List[np.ndarray] = []
    edge_cls_of: List[str] = []
    # global depth sort across layers: gather depths first
    depths, layer_of, face_of = [], [], []
    per_layer = []
    for li, L in enumerate(layers):
        d = L.data
        pv = d.V @ R.T                                          # (nv,3) view space
        P2 = np.column_stack([vp.x0 + (pv[:, 0] - vp.sx_min) * S, vp.y0 + (vp.sy_max - pv[:, 1]) * S])
        depth = d.centroids @ R[2]
        nz = d.normals @ R[2]
        front = nz > 1e-9
        visible = L.mask.copy()
        if not L.two_sided:
            visible &= front
        area = d.face_area2d(P2)
        thr = min_area_px if not area_thresholds or li not in area_thresholds else area_thresholds[li]
        emit = visible & (np.abs(area) >= thr)
        per_layer.append((d, P2, depth, front, visible, emit, nz))
        depths.append(depth[visible])
        layer_of.append(np.full(int(visible.sum()), li))
        face_of.append(np.flatnonzero(visible))
    if not depths:
        return "", 0, 0
    depth_all = np.concatenate(depths)
    order = np.argsort(depth_all, kind="stable")                 # far -> near
    rank_all = np.empty(len(order), np.int64)
    rank_all[order] = np.arange(len(order))
    layer_all = np.concatenate(layer_of)
    face_all = np.concatenate(face_of)
    n_poly = n_edge = 0
    for li, L in enumerate(layers):
        d, P2, depth, front, visible, emit, nz = per_layer[li]
        sel = layer_all == li
        rank = np.full(d.n, -1, np.int64)
        rank[face_all[sel]] = rank_all[sel]
        # ---- polygons
        P10 = np.rint(P2 * 10.0).astype(np.int64)
        nview = d.normals @ R.T
        k = _shade_factor(nview, front) if L.shade else np.full(d.n, 0.75)
        if len(d.mat_color):
            base = d.mat_color[d.mat_index]
            emis = d.mat_emis[d.mat_index]
            alpha = d.mat_alpha[d.mat_index]
        else:
            base = np.full((d.n, 3), 0.6)
            emis = np.zeros(d.n)
            alpha = np.ones(d.n)
        q = np.clip(np.rint(k * _SHADE_LEVELS), 0, _SHADE_LEVELS + 2).astype(np.int64)
        strs = np.empty(d.n, dtype=object)
        for idx, F in d.groups:
            m = emit[idx]
            if not m.any():
                continue
            idx_e = idx[m]
            coords = P10[F[m]].reshape(len(idx_e), -1)
            fmt = " ".join(["%d,%d"] * F.shape[1])
            pts = [fmt % t for t in map(tuple, coords.tolist())]
            # class per face (material colour x quantised shade)
            kk = q[idx_e] / _SHADE_LEVELS
            rgb = np.clip(base[idx_e] * (kk + emis[idx_e])[:, None], 0, 1)
            rgb8 = np.rint(rgb * 255).astype(np.int64)
            keys = rgb8[:, 0] * 65536 + rgb8[:, 1] * 256 + rgb8[:, 2]
            akey = np.rint(alpha[idx_e] * 1000).astype(np.int64)
            combo = keys * 1001 + akey
            uniq, inv = np.unique(combo, return_inverse=True)
            cls_names = []
            for u in uniq.tolist():
                col = u // 1001
                a = (u % 1001) / 1000.0
                cls_names.append(styles.fill("#%06x" % col, a, L.fill_op))
            cls_arr = np.asarray(cls_names, dtype=object)[inv]
            strs[idx_e] = ['<polygon class="%s" points="%s"/>' % (c, p) for c, p in zip(cls_arr.tolist(), pts)]
        emitted = np.flatnonzero(emit)
        poly_items.extend(strs[emitted].tolist())
        poly_keys.append(rank[emitted] * 2)
        n_poly += len(emitted)
        # ---- edges: silhouette / crease / boundary, at the rank of the nearer visible face
        E, EF = d.E, d.EF
        if len(E):
            f1, f2 = EF[:, 0], EF[:, 1]
            has2 = f2 >= 0
            v1 = visible[f1]
            v2 = np.where(has2, visible[np.maximum(f2, 0)], False)
            fr1 = front[f1]
            fr2 = np.where(has2, front[np.maximum(f2, 0)], False)
            sil = has2 & v1 & v2 & (fr1 != fr2)
            one_side = has2 & (v1 != v2)                          # one face culled/clipped -> silhouette
            crease = has2 & d.crease & (v1 | v2)
            boundary = (~has2) & v1
            draw = sil | one_side | crease | boundary
            if draw.any():
                ei = np.flatnonzero(draw)
                r1 = np.where(v1[ei], rank[f1[ei]], -1)
                r2 = np.where(v2[ei], rank[np.maximum(f2[ei], 0)], -1)
                er = np.maximum(r1, r2)
                A = P10[E[ei, 0]]
                B = P10[E[ei, 1]]
                coords = np.column_stack([A, B])
                edge_items.extend(["M%d %dL%d %d" % t for t in map(tuple, coords.tolist())])
                edge_keys.append(er * 2 + 1)
                eop = theme["edge_op"] if L.edge_op is None else L.edge_op
                ecls = styles.stroke(theme["edge"], L.edge_width, eop)
                edge_cls_of.extend([ecls] * len(ei))
                n_edge += len(ei)
    # ---- interleave by sort key (polygons first at equal rank)
    if not poly_items and not edge_items:
        return "", 0, 0
    keys = np.concatenate((poly_keys or [np.zeros(0, np.int64)]) + (edge_keys or [np.zeros(0, np.int64)]))
    order = np.argsort(keys, kind="stable")
    npoly = len(poly_items)
    out: List[str] = ['<g transform="scale(0.1)">']
    buf: List[str] = []
    buf_cls = None
    for o in order.tolist():
        if o >= npoly:
            e = o - npoly
            c = edge_cls_of[e]
            if buf and c != buf_cls:
                out.append('<path class="%s" d="%s"/>' % (buf_cls, "".join(buf)))
                buf = []
            buf.append(edge_items[e])
            buf_cls = c
        else:
            if buf:
                out.append('<path class="%s" d="%s"/>' % (buf_cls, "".join(buf)))
                buf = []
            out.append(poly_items[o])
    if buf:
        out.append('<path class="%s" d="%s"/>' % (buf_cls, "".join(buf)))
    out.append("</g>")
    return "\n".join(out), n_poly, n_edge


def _seam_paths(vp: _Viewport, data: _MeshData, theme: dict, styles: _Styles, max_ref: int = 12000) -> str:
    """Draw mesh.lines segments whose local outward normal faces the camera."""
    lines = data.mesh.lines
    if not lines or data.n == 0:
        return ""
    vn = data.vertex_normals()
    V = data.V
    if len(V) > max_ref:
        rng = np.random.default_rng(0)
        ref = rng.choice(len(V), max_ref, replace=False)
    else:
        ref = np.arange(len(V))
    Vr, Nr = V[ref], vn[ref]
    cam = vp.cam_dir
    parts: List[str] = []
    for pts in lines:
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        if len(pts) < 2:
            continue
        mid = 0.5 * (pts[1:] + pts[:-1])
        d2 = ((mid[:, None, :] - Vr[None, :, :]) ** 2).sum(axis=2)
        nn = Nr[np.argmin(d2, axis=1)]
        vis = (nn @ cam) > 0.05
        if not vis.any():
            continue
        P = vp.to_sheet(pts)
        # emit runs of visible segments as polylines
        i = 0
        n = len(vis)
        while i < n:
            if not vis[i]:
                i += 1
                continue
            j = i
            while j < n and vis[j]:
                j += 1
            seg = P[i:j + 1]
            parts.append("M" + "L".join("%s %s" % (_fmt(x), _fmt(y)) for x, y in seg))
            i = j
    if not parts:
        return ""
    return '<path fill="none" stroke="%s" stroke-opacity="%.2f" stroke-width="0.6" stroke-linecap="round" stroke-linejoin="round" d="%s"/>' % (
        theme["seam"], theme["seam_op"], "".join(parts))


# ----------------------------------------------------------------------------- sheet primitives
class _Sheet:
    def __init__(self, width: int, height: int, theme: dict):
        self.w, self.h, self.t = width, height, theme
        self.parts: List[str] = []
        self.styles = _Styles()

    # basic elements ---------------------------------------------------------------
    def add(self, s: str):
        self.parts.append(s)

    def line(self, x1, y1, x2, y2, color=None, width=0.7, opacity=1.0, dash=None):
        color = color or self.t["ink"]
        d = ' stroke-dasharray="%s"' % dash if dash else ""
        op = "" if opacity >= 0.999 else ' stroke-opacity="%.2f"' % opacity
        self.add('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%.3g"%s%s/>'
                 % (_fmt(x1), _fmt(y1), _fmt(x2), _fmt(y2), color, width, op, d))

    def polyline(self, pts, color, width=0.8, opacity=1.0, closed=False, fill="none", fill_op=None, dash=None):
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return
        d = "M" + "L".join("%s %s" % (_fmt(x), _fmt(y)) for x, y in pts) + ("Z" if closed else "")
        op = "" if opacity >= 0.999 else ' stroke-opacity="%.2f"' % opacity
        fo = "" if fill_op is None else ' fill-opacity="%.2f"' % fill_op
        da = ' stroke-dasharray="%s"' % dash if dash else ""
        self.add('<path d="%s" fill="%s"%s stroke="%s" stroke-width="%.3g"%s%s stroke-linejoin="round"/>' % (d, fill, fo, color, width, op, da))

    def rect(self, x, y, w, h, fill="none", stroke=None, width=0.8, opacity=1.0, rx=0.0):
        st = "" if stroke is None else ' stroke="%s" stroke-width="%.3g"' % (stroke, width)
        op = "" if opacity >= 0.999 else ' opacity="%.2f"' % opacity
        r = "" if rx <= 0 else ' rx="%s"' % _fmt(rx)
        self.add('<rect x="%s" y="%s" width="%s" height="%s" fill="%s"%s%s%s/>' % (_fmt(x), _fmt(y), _fmt(w), _fmt(h), fill, st, op, r))

    def circle(self, cx, cy, r, fill="none", stroke=None, width=0.8, opacity=1.0):
        st = "" if stroke is None else ' stroke="%s" stroke-width="%.3g"' % (stroke, width)
        op = "" if opacity >= 0.999 else ' opacity="%.2f"' % opacity
        self.add('<circle cx="%s" cy="%s" r="%s" fill="%s"%s%s/>' % (_fmt(cx), _fmt(cy), _fmt(r), fill, st, op))

    def text(self, x, y, s, size=9.0, color=None, anchor="start", weight="normal", family=None, spacing=0.0,
             opacity=1.0, rotate=None, italic=False, baseline=None):
        color = color or self.t["ink"]
        fam = family or _FONT
        attrs = ['x="%s"' % _fmt(x), 'y="%s"' % _fmt(y), 'font-size="%.3g"' % size, 'fill="%s"' % color,
                 'font-family="%s"' % fam]
        if anchor != "start":
            attrs.append('text-anchor="%s"' % anchor)
        if weight != "normal":
            attrs.append('font-weight="%s"' % weight)
        if spacing:
            attrs.append('letter-spacing="%.3g"' % spacing)
        if opacity < 0.999:
            attrs.append('opacity="%.2f"' % opacity)
        if italic:
            attrs.append('font-style="italic"')
        if baseline:
            attrs.append('dominant-baseline="%s"' % baseline)
        if rotate is not None:
            attrs.append('transform="rotate(%s %s %s)"' % (_fmt(rotate), _fmt(x), _fmt(y)))
        self.add("<text %s>%s</text>" % (" ".join(attrs), _esc(s)))

    def label_box(self, x, y, text, size, color, anchor="start", pad=1.5, bg_op=0.72):
        """Text with a background box so it stays legible over linework."""
        w = _text_width(text, size) + 2 * pad
        h = size + 2 * pad
        bx = x - pad if anchor == "start" else x - w / 2 if anchor == "middle" else x - w + pad
        self.rect(bx, y - size + pad * 0.2 - pad, w, h, fill=self.t["label_bg"], opacity=bg_op, rx=1.0)
        self.text(x, y, text, size=size, color=color, anchor=anchor)

    # drafting primitives -------------------------------------------------------------
    def arrow(self, tip, direction, size=5.0, color=None):
        color = color or self.t["dim"]
        d = _unit(direction)
        n = np.array([-d[1], d[0]])
        p0 = np.asarray(tip, float)
        p1 = p0 - d * size + n * size * 0.32
        p2 = p0 - d * size - n * size * 0.32
        self.add('<polygon points="%s,%s %s,%s %s,%s" fill="%s"/>' % (_fmt(p0[0]), _fmt(p0[1]), _fmt(p1[0]), _fmt(p1[1]), _fmt(p2[0]), _fmt(p2[1]), color))

    def dim_h(self, x1, x2, y, label, ext_y1=None, ext_y2=None, text_above=True, size=8.0):
        """Horizontal dimension between sheet x1..x2 at sheet y. ext_y*: feature y for extension lines."""
        c = self.t["dim"]
        if x2 < x1:
            x1, x2 = x2, x1
            ext_y1, ext_y2 = ext_y2, ext_y1
        for x, ey in ((x1, ext_y1), (x2, ext_y2)):
            if ey is not None:
                gap = 2.5 if ey < y else -2.5
                over = 3.0 if ey < y else -3.0
                self.line(x, ey + gap, x, y + over, color=c, width=0.45, opacity=0.9)
        inside = (x2 - x1) > 26
        if inside:
            self.line(x1, y, x2, y, color=c, width=0.6)
            self.arrow((x1, y), (-1, 0), color=c)
            self.arrow((x2, y), (1, 0), color=c)
        else:  # arrows outside for short dimensions
            self.line(x1 - 12, y, x2 + 12, y, color=c, width=0.6)
            self.arrow((x1, y), (1, 0), color=c)
            self.arrow((x2, y), (-1, 0), color=c)
        ty = y - 2.5 if text_above else y + size + 1.5
        self.label_box((x1 + x2) / 2, ty, label, size, self.t["dim_text"], anchor="middle", bg_op=0.0)

    def dim_v(self, y1, y2, x, label, ext_x1=None, ext_x2=None, text_left=False, size=8.0):
        """Vertical dimension between sheet y1..y2 at sheet x; text rotated along the line."""
        c = self.t["dim"]
        if y2 < y1:
            y1, y2 = y2, y1
            ext_x1, ext_x2 = ext_x2, ext_x1
        for y, ex in ((y1, ext_x1), (y2, ext_x2)):
            if ex is not None:
                gap = 2.5 if ex < x else -2.5
                over = 3.0 if ex < x else -3.0
                self.line(ex + gap, y, x + over, y, color=c, width=0.45, opacity=0.9)
        inside = (y2 - y1) > 26
        if inside:
            self.line(x, y1, x, y2, color=c, width=0.6)
            self.arrow((x, y1), (0, -1), color=c)
            self.arrow((x, y2), (0, 1), color=c)
        else:
            self.line(x, y1 - 12, x, y2 + 12, color=c, width=0.6)
            self.arrow((x, y1), (0, 1), color=c)
            self.arrow((x, y2), (0, -1), color=c)
        tx = x - 3.0 if text_left else x + size + 0.5
        self.text(tx, (y1 + y2) / 2, label, size=size, color=self.t["dim_text"], anchor="middle", rotate=-90)

    def ground_line(self, x1, x2, y, hatch=True):
        c = self.t["ground"]
        self.line(x1, y, x2, y, color=c, width=1.1)
        if hatch:
            parts = []
            x = x1 + 4
            while x < x2 - 2:
                parts.append("M%s %sL%s %s" % (_fmt(x), _fmt(y + 0.8), _fmt(x - 5), _fmt(y + 6)))
                x += 7
            self.add('<path d="%s" stroke="%s" stroke-width="0.5" stroke-opacity="0.65" fill="none"/>' % ("".join(parts), c))

    def centre_mark(self, x, y, r=4.0, color=None):
        c = color or self.t["centre"]
        self.line(x - r * 1.6, y, x + r * 1.6, y, color=c, width=0.5)
        self.line(x, y - r * 1.6, x, y + r * 1.6, color=c, width=0.5)
        self.circle(x, y, r, stroke=c, width=0.5)

    def centre_line(self, x1, y1, x2, y2, color=None):
        self.line(x1, y1, x2, y2, color=color or self.t["centre"], width=0.5, opacity=0.8, dash="14 3 2 3")

    def view_label(self, x, y, title, sub=None):
        self.text(x, y, title, size=11.5, color=self.t["ink"], weight="bold", spacing=1.6)
        tw = _text_width(title, 11.5, "bold", 1.6)
        if sub:
            self.text(x + tw + 14, y, sub, size=8.5, color=self.t["ink_dim"], spacing=0.6)
        self.line(x, y + 4, x + tw + (_text_width(sub, 8.5, spacing=0.6) + 14 if sub else 0), y + 4,
                  color=self.t["ink"], width=0.8)

    # assembly -----------------------------------------------------------------------
    def background(self, grid_minor=20, grid_major=100):
        t = self.t
        self.add(
            '<defs>'
            '<pattern id="gminor" width="%d" height="%d" patternUnits="userSpaceOnUse">'
            '<path d="M %d 0 L 0 0 0 %d" fill="none" stroke="%s" stroke-opacity="%.3f" stroke-width="0.6"/></pattern>'
            '<pattern id="gmajor" width="%d" height="%d" patternUnits="userSpaceOnUse">'
            '<rect width="%d" height="%d" fill="url(#gminor)"/>'
            '<path d="M %d 0 L 0 0 0 %d" fill="none" stroke="%s" stroke-opacity="%.3f" stroke-width="0.8"/></pattern>'
            '</defs>' % (grid_minor, grid_minor, grid_minor, grid_minor, t["grid"], t["grid_minor_op"],
                         grid_major, grid_major, grid_major, grid_major, grid_major, grid_major, t["grid"], t["grid_major_op"]))
        self.rect(0, 0, self.w, self.h, fill=t["bg"])
        self.rect(0, 0, self.w, self.h, fill="url(#gmajor)")

    def frame(self, margin=18, inner=8):
        c = self.t["frame"]
        self.rect(margin, margin, self.w - 2 * margin, self.h - 2 * margin, stroke=c, width=1.6)
        self.rect(margin + inner, margin + inner, self.w - 2 * (margin + inner), self.h - 2 * (margin + inner), stroke=c, width=0.6)
        # zone ticks along the frame (A, B, C ... / 1, 2, 3 ...) like a real drawing border
        n_cols, n_rows = 8, 6
        for i in range(1, n_cols):
            x = margin + (self.w - 2 * margin) * i / n_cols
            self.line(x, margin, x, margin + inner, color=c, width=0.6)
            self.line(x, self.h - margin - inner, x, self.h - margin, color=c, width=0.6)
        for i in range(n_cols):
            x = margin + (self.w - 2 * margin) * (i + 0.5) / n_cols
            self.text(x, margin + inner - 1.5, str(i + 1), size=6, color=c, anchor="middle")
            self.text(x, self.h - margin - 1.5, str(i + 1), size=6, color=c, anchor="middle")
        for j in range(1, n_rows):
            y = margin + (self.h - 2 * margin) * j / n_rows
            self.line(margin, y, margin + inner, y, color=c, width=0.6)
            self.line(self.w - margin - inner, y, self.w - margin, y, color=c, width=0.6)
        for j in range(n_rows):
            y = margin + (self.h - 2 * margin) * (j + 0.5) / n_rows
            self.text(margin + inner / 2, y + 2, "ABCDEFGH"[j], size=6, color=c, anchor="middle")
            self.text(self.w - margin - inner / 2, y + 2, "ABCDEFGH"[j], size=6, color=c, anchor="middle")

    def render(self) -> str:
        head = ('<svg xmlns="http://www.w3.org/2000/svg" width="%smm" height="%smm" viewBox="0 0 %d %d">\n'
                % (_fmt(_SHEET_MM_W * self.w / _SHEET_W), _fmt(_SHEET_MM_W * self.h / _SHEET_W), self.w, self.h))
        css = self.styles.css()
        style = "<style>\n%s\n</style>\n" % css if css else ""
        return head + style + "\n".join(self.parts) + "\n</svg>\n"


# ----------------------------------------------------------------------------- label placement
class _Labeler:
    """Greedy label placement: tries a ring of offsets around the anchor and keeps the first
    that does not collide with an earlier label; returns the text origin and whether a leader is needed."""

    def __init__(self, bounds: Tuple[float, float, float, float]):
        self.boxes: List[Tuple[float, float, float, float]] = []
        self.bounds = bounds

    @staticmethod
    def _hit(a, b) -> bool:
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    def place(self, ax: float, ay: float, text: str, size: float):
        w = _text_width(text, size) + 3
        h = size + 2
        cands = [(5, -3), (5, 9), (-w - 5, -3), (-w - 5, 9), (5, -14), (5, 20), (-w / 2, -16), (-w / 2, 20),
                 (-w - 5, -14), (-w - 5, 20), (14, -26), (14, 32), (-w - 14, -26), (-w - 14, 32),
                 (5, -38), (5, 44), (-w - 5, -38), (-w - 5, 44), (30, -3), (-w - 30, -3)]
        chosen = None
        for dx, dy in cands:
            x, y = ax + dx, ay + dy                        # (x, y) = text baseline origin (start anchor)
            box = (x - 1.5, y - size - 1, x + w, y + 2)
            bx0, by0, bx1, by1 = self.bounds
            if box[0] < bx0 or box[2] > bx1 or box[1] < by0 or box[3] > by1:
                continue
            if any(self._hit(box, b) for b in self.boxes):
                continue
            chosen = (x, y, box, dx, dy)
            break
        if chosen is None:
            dx, dy = cands[0]
            x, y = ax + dx, ay + dy
            chosen = (x, y, (x - 1.5, y - size - 1, x + w, y + 2), dx, dy)
        x, y, box, dx, dy = chosen
        self.boxes.append(box)
        leader = (abs(dx) > 8 and dx > 0) or (dx < 0 and abs(dx + w) > 8) or abs(dy) > 12
        # leader target: nearest box edge midpoint
        lx = box[0] if ax < box[0] else box[2] if ax > box[2] else ax
        ly = box[3] if ay > box[3] else box[1] if ay < box[1] else (box[1] + box[3]) / 2
        return x, y, leader, (lx, ly)


# ----------------------------------------------------------------------------- connectors overlay
def _connector_style(kind: str, theme: dict) -> str:
    return theme["conn"].get(kind, theme["conn"]["polygon"])


def _is_mirror_twin(c: Connector) -> bool:
    """True for *_R connectors (mirror twins hidden in the side view)."""
    return c.name.endswith("_R") or c.name.endswith("_r") or c.meta.get("side") == "right"


def _draw_connectors(sheet: _Sheet, vp: _Viewport, connectors: Sequence[Connector], bounds, facing_only=True,
                     skip_twins=False, label=True, font=6.5, min_facing=0.15, opacity=0.95):
    t = sheet.t
    cam = vp.cam_dir
    lab = _Labeler(bounds)
    drawn = 0
    # draw geometry first, labels after (so labels sit on top)
    label_jobs = []
    for c in connectors:
        n = np.asarray(c.frame.z_axis, float)
        facing = float(n @ cam)
        if facing_only and facing < min_facing:
            continue
        if skip_twins and _is_mirror_twin(c):
            continue
        col = _connector_style(c.kind, t)
        o = vp.to_sheet(c.frame.origin)[0]
        if isinstance(c, PolygonConnector):
            P = vp.to_sheet(c.points)
            sheet.polyline(P, col, width=0.9, opacity=opacity, closed=True, fill=col, fill_op=0.10)
            # centre mark + normal tick
            sheet.circle(o[0], o[1], 1.6, fill=col, opacity=opacity)
        else:
            sheet.circle(o[0], o[1], 2.6, stroke=col, width=1.0, opacity=opacity)
            sheet.circle(o[0], o[1], 0.9, fill=col, opacity=opacity)
        tip = vp.to_sheet(np.asarray(c.frame.origin) + n * 0.12)[0]
        if np.hypot(*(tip - o)) > 1.0:
            sheet.line(o[0], o[1], tip[0], tip[1], color=col, width=0.9, opacity=opacity)
            sheet.arrow(tip, tip - o, size=3.5, color=col)
        drawn += 1
        if label:
            label_jobs.append((o, c.name, col))
    for o, name, col in label_jobs:
        x, y, leader, (lx, ly) = lab.place(o[0], o[1], name, font)
        if leader:
            sheet.line(o[0], o[1], lx, ly, color=col, width=0.4, opacity=0.7)
        sheet.label_box(x, y, name, font, col, bg_op=0.7)
    return drawn


def _legend(sheet: _Sheet, x, y, title="CONNECTOR KINDS", columns=1, col_width=190.0):
    """Connector-kind legend; `columns=2` gives a compact 2x2 grid."""
    t = sheet.t
    sheet.text(x, y, title, size=7.5, color=t["ink_dim"], weight="bold", spacing=1.0)
    entries = (("point", "point - location + mount normal"), ("circle", "circle - radius in frame XY"),
               ("rectangle", "rectangle - width x height"), ("polygon", "polygon - free outline"))
    yy = y + 12
    for i, (kind, desc) in enumerate(entries):
        col = _connector_style(kind, t)
        cx = x + (i % columns) * col_width
        cy = yy + (i // columns) * 12
        if kind == "point":
            sheet.circle(cx + 6, cy - 3, 2.6, stroke=col, width=1.0)
            sheet.circle(cx + 6, cy - 3, 0.9, fill=col)
        elif kind == "circle":
            sheet.circle(cx + 6, cy - 3, 4, stroke=col, width=0.9, fill=col)
        elif kind == "rectangle":
            sheet.rect(cx + 1, cy - 7, 10, 8, stroke=col, width=0.9)
        else:
            sheet.polyline([(cx + 1, cy), (cx + 4, cy - 8), (cx + 11, cy - 6), (cx + 9, cy + 1)], col, width=0.9, closed=True)
        sheet.text(cx + 18, cy, desc, size=7, color=t["ink_dim"])
    return yy + 12 * ((len(entries) + columns - 1) // columns)


def _title_block(sheet: _Sheet, x, y, w, h, title, subtitle, info_rows, credit="makecar"):
    t = sheet.t
    sheet.rect(x, y, w, h, fill=t["title_bg"], stroke=t["frame"], width=1.2)
    # left column: title
    sheet.rect(x, y, w * 0.56, h, stroke=t["frame"], width=0.6)
    sheet.text(x + 10, y + 13, "TITLE", size=6, color=t["ink_faint"], spacing=1.2)
    title_size = 20 if _text_width(title.upper(), 20) < w * 0.56 - 20 else max(9.0, 20 * (w * 0.56 - 20) / max(_text_width(title.upper(), 20), 1))
    sheet.text(x + 10, y + 13 + title_size + 4, title.upper(), size=title_size, color=t["ink"], weight="bold", spacing=1.0)
    if subtitle:
        sheet.text(x + 10, y + h - 26, subtitle, size=9, color=t["ink_dim"])
    sheet.text(x + 10, y + h - 9, "%s  ·  general arrangement drawing  ·  orthographic views, nose to the left" % credit,
               size=6.5, color=t["ink_faint"])
    # right column: grid of fields
    gx = x + w * 0.56
    gw = w - w * 0.56
    cols = 2
    rows = int(math.ceil(len(info_rows) / cols))
    cw, ch = gw / cols, h / rows
    for i, (k, v) in enumerate(info_rows):
        cx = gx + (i % cols) * cw
        cy = y + (i // cols) * ch
        sheet.rect(cx, cy, cw, ch, stroke=t["frame"], width=0.5)
        sheet.text(cx + 6, cy + 9, k.upper(), size=5.8, color=t["ink_faint"], spacing=1.0)
        big = k.lower() == "scale"
        sheet.text(cx + 6, cy + ch - 6, str(v), size=12 if big else 8.5, color=t["ink"], weight="bold" if big else "normal",
                   family=_MONO if not big else None)


# ----------------------------------------------------------------------------- scale / layout
def _mm_per_px() -> float:
    return _SHEET_MM_W / _SHEET_W


def _px_per_m(ratio: float) -> float:
    return 1000.0 / (ratio * _mm_per_px())


def _resolve_scale(scale, s_fit: float) -> Tuple[float, float]:
    """Return (px_per_metre, ratio).  scale: None -> largest standard scale that fits,
    'fit' -> exact fit, number or '1:25' -> forced."""
    if scale is None or scale == "auto":
        for r in _STD_SCALES:
            s = _px_per_m(r)
            if s <= s_fit:
                return s, float(r)
        return s_fit, 1000.0 / (s_fit * _mm_per_px())
    if isinstance(scale, str):
        if scale == "fit":
            return s_fit, 1000.0 / (s_fit * _mm_per_px())
        scale = float(scale.split(":")[-1])
    r = float(scale)
    return _px_per_m(r), r


def _scale_text(ratio: float) -> str:
    return "1:%s" % (("%d" % ratio) if abs(ratio - round(ratio)) < 1e-6 else ("%.1f" % ratio))


def _mm(v: float) -> str:
    return "%d" % int(round(v * 1000.0))


def _visible_areas(vp: _Viewport, layers: Sequence[_Layer]) -> np.ndarray:
    out = []
    for L in layers:
        d = L.data
        pv = d.V @ vp.R.T
        P2 = np.column_stack([pv[:, 0] * vp.S, pv[:, 1] * vp.S])
        vis = L.mask.copy()
        if not L.two_sided:
            vis &= (d.normals @ vp.R[2]) > 1e-9
        out.append(np.abs(d.face_area2d(P2))[vis])
    return np.concatenate(out) if out else np.zeros(0)


def _fallback_measurements(mesh: Mesh, measurements: Optional[dict]) -> dict:
    lo, hi = mesh.bounds()
    m = dict(measurements or {})
    m.setdefault("x_front", float(hi[0]))
    m.setdefault("x_rear", float(lo[0]))
    m.setdefault("length", m["x_front"] - m["x_rear"])
    m.setdefault("width", float(hi[1] - lo[1]))
    m.setdefault("height", float(hi[2]))
    if lo[2] > 1e-3:
        m.setdefault("z_floor", float(lo[2]))
    return m


class _Cell:
    """Placement of one view: car box top-left (x0,y0) in sheet px, extents in metres."""

    def __init__(self, name, x0, y0, wm, hm, S):
        self.name, self.x0, self.y0, self.wm, self.hm, self.S = name, x0, y0, wm, hm, S

    @property
    def w(self):
        return self.wm * self.S

    @property
    def h(self):
        return self.hm * self.S

    @property
    def x1(self):
        return self.x0 + self.w

    @property
    def y1(self):
        return self.y0 + self.h


def _viewport(name: str, cell: _Cell, lo, hi) -> _Viewport:
    if name in ("side", "section"):
        return _Viewport(_R_SIDE, cell.S, cell.x0, cell.y0, -hi[0], hi[2])
    if name in ("top", "plan"):
        return _Viewport(_R_TOP, cell.S, cell.x0, cell.y0, -hi[0], -lo[1])
    if name == "front":
        return _Viewport(_R_FRONT, cell.S, cell.x0, cell.y0, lo[1], hi[2])
    return _Viewport(_R_REAR, cell.S, cell.x0, cell.y0, -hi[1], hi[2])


# ----------------------------------------------------------------------------- blueprint
def build_blueprint(mesh: Mesh, connectors: Optional[Sequence[Connector]] = None, measurements: Optional[dict] = None,
                    title: str = "", subtitle: str = "", show_connectors: bool = False, interior_mesh: Optional[Mesh] = None,
                    scale=None, theme: str = "blueprint", max_polygons: int = 60000, date: Optional[str] = None,
                    credit: str = "makecar") -> Tuple[str, dict]:
    """Build the four-view (plus optional interior) blueprint sheet. Returns (svg_text, info)."""
    th = THEMES[theme] if isinstance(theme, str) else dict(theme)
    connectors = list(connectors or [])
    data = _MeshData(mesh)
    idata = _MeshData(interior_mesh) if interior_mesh is not None and interior_mesh.n_faces else None
    if mesh.n_vertices == 0 or mesh.n_faces == 0:
        raise ValueError("write_blueprint: the mesh has no faces")
    lo, hi = mesh.bounds()
    zmin = min(0.0, float(lo[2]))
    L, W, Hs = float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - zmin)
    L, W, Hs = max(L, 1e-3), max(W, 1e-3), max(Hs, 1e-3)
    meas = _fallback_measurements(mesh, measurements)
    has_interior = idata is not None

    # ---- layout constants (px).  Rows: [SIDE, FRONT] / [PLAN, REAR] / [INTERIOR PLAN, INTERIOR SECTION]
    SW, SH = _SHEET_W, _SHEET_H
    X0, X1, Y0, Y1 = 50.0, 1550.0, 66.0, 1050.0
    TB_W, TB_H = 470.0, 118.0
    gap = 36.0
    pad_side = (34, 74, 28, 80)      # left, right, top, bottom
    pad_front = (10, 44, 28, 80)
    pad_rear = (10, 10, 28, 34)
    pad_top = (10, 10, 28, 22)
    pad_sec = (10, 10, 28, 34)
    row_gap = 16.0
    s_w1 = ((X1 - X0) - sum(pad_side[:2]) - sum(pad_front[:2]) - gap) / (L + W)
    s_w2 = ((X1 - X0) - sum(pad_top[:2]) - sum(pad_rear[:2]) - gap) / (L + W)
    h_pads = (pad_side[2] + pad_side[3]) + (pad_top[2] + pad_top[3]) + row_gap
    if has_interior:
        s_w3 = ((X1 - X0) - sum(pad_top[:2]) - sum(pad_sec[:2]) - gap) / (2 * L)
        s_h = ((Y1 - Y0) - h_pads - (pad_top[2] + pad_top[3]) - row_gap) / (Hs + W + max(W, Hs))
        s_fit = min(s_w1, s_w2, s_w3, s_h)
    else:
        s_h = ((Y1 - Y0) - h_pads) / (Hs + W)
        s_fit = min(s_w1, s_w2, s_h)
    S, ratio = _resolve_scale(scale, s_fit)

    # ---- cells
    y = Y0
    side = _Cell("side", X0 + pad_side[0], y + pad_side[2], L, Hs, S)
    front = _Cell("front", side.x1 + pad_side[1] + gap + pad_front[0], side.y0, W, Hs, S)
    y = side.y1 + pad_side[3] + row_gap
    top = _Cell("top", X0 + pad_top[0], y + pad_top[2], L, W, S)
    rear = _Cell("rear", top.x1 + pad_top[1] + gap + pad_rear[0], top.y0, W, Hs, S)
    plan = sec = None
    if has_interior:
        y = max(top.y1 + pad_top[3], rear.y1 + pad_rear[3]) + row_gap
        plan = _Cell("plan", X0 + pad_top[0], y + pad_top[2], L, W, S)
        sec = _Cell("section", plan.x1 + pad_top[1] + gap + pad_sec[0], plan.y0, L, Hs, S)

    sheet = _Sheet(SW, SH, th)
    sheet.background()
    sheet.frame()
    styles = sheet.styles
    info: Dict[str, object] = {"scale": _scale_text(ratio), "px_per_m": S, "views": {}}

    # ---- polygon budget: raise the tiny-face threshold globally if needed
    vps = {
        "side": _viewport("side", side, lo, hi), "front": _viewport("front", front, lo, hi),
        "rear": _viewport("rear", rear, lo, hi), "top": _viewport("top", top, lo, hi),
    }
    main_layers = [_Layer(data)]
    areas = [_visible_areas(vp, main_layers) for vp in vps.values()]
    extra_layers = {}
    if has_interior:
        vps["plan"] = _viewport("plan", plan, lo, hi)
        vps["section"] = _viewport("section", sec, lo, hi)
        z_belt = meas.get("z_belt")
        body_plan_mask = data.centroids[:, 2] < z_belt if z_belt is not None else np.ones(data.n, bool)
        body_sec_mask = data.centroids[:, 1] < 0.0
        extra_layers["plan_body"] = [_Layer(data, body_plan_mask, two_sided=True, fill_op=th["ghost_fill_op"],
                                            edge_op=th["ghost_edge_op"], edge_width=0.45)]
        extra_layers["plan_int"] = [_Layer(idata)]
        extra_layers["section"] = [_Layer(data, body_sec_mask, two_sided=True), _Layer(idata)]
        areas.append(_visible_areas(vps["plan"], extra_layers["plan_body"]))
        areas.append(_visible_areas(vps["plan"], extra_layers["plan_int"]))
        areas.append(_visible_areas(vps["section"], extra_layers["section"]))
    all_areas = np.concatenate(areas) if areas else np.zeros(0)
    min_area = 0.15
    n_total = int((all_areas >= min_area).sum())
    if n_total > max_polygons and len(all_areas):
        k = len(all_areas) - max_polygons
        min_area = max(min_area, float(np.partition(all_areas, k)[k]) + 1e-9)
    info["min_face_area_px"] = min_area

    # ---- header
    sheet.text(X0, 46, ("%s  ·  GENERAL ARRANGEMENT" % (title or mesh.name or "untitled")).upper(), size=10.5,
               color=th["ink"], weight="bold", spacing=1.8)
    hdr_r = "SCALE %s   ·   A3 LANDSCAPE   ·   %s" % (_scale_text(ratio), date or _dt.date.today().isoformat())
    sheet.text(X1, 46, hdr_r, size=8.5, color=th["ink_dim"], anchor="end", spacing=1.0)

    total_polys = total_edges = 0

    def render(name, cell, layers, label, sub=None, seams=True, ground=True):
        nonlocal total_polys, total_edges
        vp = vps[name]
        sheet.view_label(cell.x0 - 4, cell.y0 - 12, label, sub)
        if ground:
            gy = vp.to_sheet([[0.0, 0.0, 0.0]])[0][1]
            sheet.ground_line(cell.x0 - 14, cell.x1 + 14, gy)
        svg, npoly, nedge = _render_view(vp, layers, styles, th, min_area_px=min_area)
        sheet.add(svg)
        if seams:
            sheet.add(_seam_paths(vp, data, th, styles))
        total_polys += npoly
        total_edges += nedge
        info["views"][name] = {"polygons": npoly, "edges": nedge}
        return vp

    # ---- SIDE
    vp = render("side", side, main_layers, "SIDE ELEVATION", "left-hand side, nose left")
    gy = vp.to_sheet([[0, 0, 0]])[0][1]
    xr, xf = meas["x_rear"], meas["x_front"]
    # feature points for extension lines: extreme vertices
    V = data.V
    p_front = V[np.argmax(V[:, 0])]
    p_rear = V[np.argmin(V[:, 0])]
    p_top = V[np.argmax(V[:, 2])]
    sf, sr, stp = vp.to_sheet(p_front)[0], vp.to_sheet(p_rear)[0], vp.to_sheet(p_top)[0]
    sheet.dim_h(sr[0], sf[0], gy + 66, _mm(xf - xr), ext_y1=sr[1], ext_y2=sf[1])
    if "wheel_front_x" in meas and "wheel_rear_x" in meas:
        wz = meas.get("wheel_front_z", meas.get("arch_front_r", 0.33))
        wf = vp.to_sheet([[meas["wheel_front_x"], 0, wz]])[0]
        wr = vp.to_sheet([[meas["wheel_rear_x"], 0, meas.get("wheel_rear_z", wz)]])[0]
        for w in (wf, wr):
            sheet.centre_mark(w[0], w[1])
            sheet.centre_line(w[0], w[1], w[0], gy + 40)
        sheet.dim_h(wr[0], wf[0], gy + 38, _mm(meas["wheel_front_x"] - meas["wheel_rear_x"]))
        sheet.dim_h(sr[0], wr[0], gy + 38, _mm(meas["wheel_rear_x"] - xr))
        sheet.dim_h(wf[0], sf[0], gy + 38, _mm(xf - meas["wheel_front_x"]))
    # height + ground clearance on the tail side (right)
    xd = side.x1 + 52
    sheet.dim_v(gy, stp[1], xd, _mm(meas["height"]), ext_x1=stp[0], ext_x2=None)
    if "z_floor" in meas and meas["z_floor"] > 0.02:
        zf = vp.to_sheet([[xf, 0, meas["z_floor"]]])[0]
        sheet.dim_v(gy, zf[1], side.x0 - 22, "GC %s" % _mm(meas["z_floor"]), ext_x1=side.x0 - 2, ext_x2=None, size=7, text_left=True)
    if show_connectors and connectors:
        _draw_connectors(sheet, vp, connectors, (side.x0 - 60, side.y0 - 40, side.x1 + 20, side.y1 + 30), skip_twins=True)

    # ---- FRONT
    vp = render("front", front, main_layers, "FRONT")
    gy = vp.to_sheet([[0, 0, 0]])[0][1]
    p_l = V[np.argmax(V[:, 1])]
    p_r = V[np.argmin(V[:, 1])]
    sl, srr = vp.to_sheet(p_l)[0], vp.to_sheet(p_r)[0]
    if "track" in meas:
        tr = meas["track"]
        wz = meas.get("wheel_front_z", 0.33)
        a = vp.to_sheet([[0, -tr / 2, wz]])[0]
        b = vp.to_sheet([[0, tr / 2, wz]])[0]
        sheet.centre_mark(a[0], a[1])
        sheet.centre_mark(b[0], b[1])
        sheet.dim_h(a[0], b[0], gy + 38, _mm(tr), ext_y1=a[1], ext_y2=b[1])
    sheet.dim_h(srr[0], sl[0], gy + 66, _mm(meas["width"]), ext_y1=srr[1], ext_y2=sl[1])
    sheet.dim_v(gy, stp[1], front.x1 + 30, _mm(meas["height"]), ext_x1=front.x0 + front.w / 2, ext_x2=None)
    cx = vp.to_sheet([[0, 0, 0]])[0][0]
    sheet.centre_line(cx, front.y0 - 6, cx, gy + 8)
    if show_connectors and connectors:
        _draw_connectors(sheet, vp, connectors, (front.x0 - 30, front.y0 - 40, front.x1 + 30, front.y1 + 30))

    # ---- REAR
    vp = render("rear", rear, main_layers, "REAR")
    gy = vp.to_sheet([[0, 0, 0]])[0][1]
    cx = vp.to_sheet([[0, 0, 0]])[0][0]
    sheet.centre_line(cx, rear.y0 - 6, cx, gy + 8)
    if show_connectors and connectors:
        _draw_connectors(sheet, vp, connectors, (rear.x0 - 30, rear.y0 - 40, rear.x1 + 30, rear.y1 + 30))

    # ---- TOP (plan)
    vp = render("top", top, main_layers, "PLAN", "viewed from above, nose left", ground=False)
    a = vp.to_sheet([[xf + 0.25, 0, 0]])[0]
    b = vp.to_sheet([[xr - 0.25, 0, 0]])[0]
    sheet.centre_line(a[0], a[1], b[0], b[1])
    if "wheel_front_x" in meas and "track" in meas:
        for wx in (meas["wheel_front_x"], meas["wheel_rear_x"]):
            for sy in (-1, 1):
                p = vp.to_sheet([[wx, sy * meas["track"] / 2, 0]])[0]
                sheet.centre_mark(p[0], p[1], r=3.0)
    if show_connectors and connectors:
        _draw_connectors(sheet, vp, connectors, (top.x0 - 40, top.y0 - 40, top.x1 + 40, top.y1 + 40))

    # ---- INTERIOR views
    if has_interior:
        vp = vps["plan"]
        sheet.view_label(plan.x0 - 4, plan.y0 - 12, "INTERIOR PLAN", "body cut at belt line" if meas.get("z_belt") else "body ghosted")
        svg, n1, e1 = _render_view(vp, extra_layers["plan_body"], styles, th, min_area_px=min_area)
        sheet.add(svg)
        svg, n2, e2 = _render_view(vp, extra_layers["plan_int"], styles, th, min_area_px=min_area)
        sheet.add(svg)
        total_polys += n1 + n2
        total_edges += e1 + e2
        info["views"]["plan"] = {"polygons": n1 + n2, "edges": e1 + e2}
        a = vp.to_sheet([[xf + 0.25, 0, 0]])[0]
        b = vp.to_sheet([[xr - 0.25, 0, 0]])[0]
        sheet.centre_line(a[0], a[1], b[0], b[1])
        if show_connectors and connectors:
            _draw_connectors(sheet, vp, connectors, (plan.x0 - 40, plan.y0 - 40, plan.x1 + 40, plan.y1 + 40))
        vp = vps["section"]
        sheet.view_label(sec.x0 - 4, sec.y0 - 12, "INTERIOR SECTION", "cutaway at the centre plane, near half removed")
        gy = vp.to_sheet([[0, 0, 0]])[0][1]
        sheet.ground_line(sec.x0 - 14, sec.x1 + 14, gy)
        svg, n3, e3 = _render_view(vp, extra_layers["section"], styles, th, min_area_px=min_area)
        sheet.add(svg)
        total_polys += n3
        total_edges += e3
        info["views"]["section"] = {"polygons": n3, "edges": e3}
        if show_connectors and connectors:
            cam = vp.cam_dir
            sel = [c for c in connectors if float(np.asarray(c.frame.z_axis) @ cam) > -0.2 and float(c.frame.origin[1]) < 0.5]
            _draw_connectors(sheet, vp, sel, (sec.x0 - 40, sec.y0 - 40, sec.x1 + 40, sec.y1 + 40), facing_only=False)

    # ---- notes, legend, title block
    tb_x, tb_y = X1 - TB_W, Y1 - TB_H
    if has_interior:
        nx, ny = sec.x0 - 4, sec.y1 + pad_sec[3] + 4
        n_ok = (tb_x - nx) > 300 and (Y1 - ny) > 60
    else:
        nx, ny = X0, max(top.y1 + pad_top[3], rear.y1 + pad_rear[3]) + 12
        n_ok = (Y1 - ny) > 60
    if n_ok:
        sheet.text(nx, ny, "NOTES", size=7.5, color=th["ink_dim"], weight="bold", spacing=1.0)
        notes = ["1. All dimensions in millimetres, taken from the body mesh.",
                 "2. Orthographic projection. Front of vehicle to the left in side and plan views.",
                 "3. Ground plane at Z = 0. Height and ground clearance (GC) measured from ground.",
                 "4. Seams shown are body feature lines (doors, hood, deck lid)."]
        if show_connectors:
            notes.append("5. Connector overlay: outline colour = kind, arrow = mount normal (Z axis).")
        if has_interior:
            notes.append("%d. Interior plan drops body faces above the belt line; section keeps the far half only." % (len(notes) + 1))
        yy = ny + 13
        for s_ in notes:
            sheet.text(nx, yy, s_, size=7, color=th["ink_dim"])
            yy += 10.5
        if show_connectors and not has_interior:
            _legend(sheet, nx + 470, ny)
    if show_connectors and has_interior:   # compact legend in the strip under the REAR view
        _legend(sheet, rear.x0 - 6, rear.y1 + pad_rear[3] + 12, columns=2, col_width=max(150.0, (rear.w + 40) / 2))
    st = mesh.stats()
    rows = [("scale", _scale_text(ratio)), ("sheet", "A3 · 1 / 1"), ("date", date or _dt.date.today().isoformat()),
            ("units", "mm"), ("drawn by", "%s" % credit), ("mesh", "%s f / %s v" % (format(st["faces"], ","), format(st["vertices"], ",")))]
    _title_block(sheet, tb_x, tb_y, TB_W, TB_H, title or mesh.name or "untitled", subtitle, rows, credit)

    svg_text = sheet.render()
    info.update({"polygons": total_polys, "edges": total_edges, "svg_size_bytes": len(svg_text.encode("utf-8")),
                 "sheet_px": (SW, SH), "interior": has_interior})
    return svg_text, info


def write_blueprint(path, mesh: Mesh, connectors: Optional[Sequence[Connector]] = None, measurements: Optional[dict] = None,
                    title: str = "", subtitle: str = "", show_connectors: bool = False, interior_mesh: Optional[Mesh] = None,
                    scale=None, theme: str = "blueprint", max_polygons: int = 60000) -> Path:
    """Write a multi-view blueprint sheet to `path` (SVG). Returns the Path.

    theme: "blueprint" (dark blue, light linework) or "paper" (white).
    scale: None -> largest standard scale that fits; "fit"; a number (25) or "1:25".
    The last build's statistics are available as ``write_blueprint.last_info``.
    """
    svg_text, info = build_blueprint(mesh, connectors, measurements, title, subtitle, show_connectors, interior_mesh,
                                     scale, theme, max_polygons)
    path = Path(path)
    path.write_text(svg_text, encoding="utf-8")
    write_blueprint.last_info = info  # type: ignore[attr-defined]
    return path


write_blueprint.last_info = {}  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------- connector map
def build_connector_map(mesh: Mesh, connectors: Sequence[Connector], title: str = "", theme: str = "paper",
                        scale=None, date: Optional[str] = None, credit: str = "makecar") -> Tuple[str, dict]:
    """Two-view (side + plan) sheet that labels every connector: the 'where do components attach' diagram."""
    th = THEMES[theme] if isinstance(theme, str) else dict(theme)
    connectors = list(connectors or [])
    if mesh.n_vertices == 0 or mesh.n_faces == 0:
        raise ValueError("write_connector_map: the mesh has no faces")
    data = _MeshData(mesh)
    lo, hi = mesh.bounds()
    zmin = min(0.0, float(lo[2]))
    L, W, Hs = max(float(hi[0] - lo[0]), 1e-3), max(float(hi[1] - lo[1]), 1e-3), max(float(hi[2] - zmin), 1e-3)
    SW, SH = _SHEET_W, _SHEET_H
    X0, X1, Y0, Y1 = 50.0, 1550.0, 66.0, 1050.0
    TB_W, TB_H = 470.0, 96.0
    pad_x, pad_top, pad_bot = 60.0, 34.0, 40.0
    s_w = ((X1 - X0) - 2 * pad_x) / L
    s_h = ((Y1 - Y0) - 2 * (pad_top + pad_bot) - 16) / (Hs + W)
    S, ratio = _resolve_scale(scale, min(s_w, s_h))
    side = _Cell("side", X0 + pad_x, Y0 + pad_top, L, Hs, S)
    top = _Cell("top", X0 + pad_x, side.y1 + pad_bot + 16 + pad_top, L, W, S)
    sheet = _Sheet(SW, SH, th)
    sheet.background()
    sheet.frame()
    sheet.text(X0, 46, ("%s  ·  CONNECTOR MAP" % (title or mesh.name or "untitled")).upper(), size=10.5, color=th["ink"],
               weight="bold", spacing=1.8)
    sheet.text(X1, 46, "SCALE %s   ·   %d CONNECTORS   ·   %s" % (_scale_text(ratio), len(connectors), date or _dt.date.today().isoformat()),
               size=8.5, color=th["ink_dim"], anchor="end", spacing=1.0)
    ghost = [_Layer(data, fill_op=0.38, edge_op=0.7, edge_width=0.45)]
    info = {"scale": _scale_text(ratio), "views": {}}
    total = 0
    for name, cell, label, sub, skip in (("side", side, "SIDE ELEVATION", "left-hand side; right-hand twins (*_R) omitted", True),
                                          ("top", top, "PLAN", "all connectors, viewed from above", False)):
        vp = _viewport(name, cell, lo, hi)
        sheet.view_label(cell.x0 - 4, cell.y0 - 14, label, sub)
        if name == "side":
            gy = vp.to_sheet([[0, 0, 0]])[0][1]
            sheet.ground_line(cell.x0 - 20, cell.x1 + 20, gy)
        else:
            a = vp.to_sheet([[hi[0] + 0.3, 0, 0]])[0]
            b = vp.to_sheet([[lo[0] - 0.3, 0, 0]])[0]
            sheet.centre_line(a[0], a[1], b[0], b[1])
        svg, npoly, nedge = _render_view(vp, ghost, sheet.styles, th)
        sheet.add(svg)
        sheet.add(_seam_paths(vp, data, th, sheet.styles))
        total += npoly
        info["views"][name] = {"polygons": npoly, "edges": nedge}
        bounds = (X0 - 20, cell.y0 - 30, X1 + 20, cell.y1 + pad_bot)
        n = _draw_connectors(sheet, vp, connectors, bounds, facing_only=False, skip_twins=skip, font=7.0, opacity=0.95)
        info["views"][name]["connectors"] = n
    # legend + title block
    tb_x, tb_y = X1 - TB_W, Y1 - TB_H
    _legend(sheet, X0 + 10, tb_y + 12)
    rows = [("scale", _scale_text(ratio)), ("connectors", str(len(connectors))), ("date", date or _dt.date.today().isoformat()),
            ("drawn by", credit)]
    _title_block(sheet, tb_x, tb_y, TB_W, TB_H, title or mesh.name or "untitled", "connector map - attachment sites by kind",
                 rows, credit)
    svg_text = sheet.render()
    info.update({"polygons": total, "svg_size_bytes": len(svg_text.encode("utf-8"))})
    return svg_text, info


def write_connector_map(path, mesh: Mesh, connectors: Sequence[Connector], title: str = "", theme: str = "paper",
                        scale=None) -> Path:
    """Write the two-view connector map to `path` (SVG). Returns the Path."""
    svg_text, info = build_connector_map(mesh, connectors, title, theme, scale)
    path = Path(path)
    path.write_text(svg_text, encoding="utf-8")
    write_connector_map.last_info = info  # type: ignore[attr-defined]
    return path


write_connector_map.last_info = {}  # type: ignore[attr-defined]
