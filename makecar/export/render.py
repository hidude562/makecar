"""Tiny software rasteriser (numpy z-buffer) for preview images.

Supports perspective/orthographic cameras, flat shading with two lights,
per-material colour/alpha (transparent materials are blended in a second,
back-to-front pass), a ground shadow, and overlay polylines for seams.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np

from ..geometry.mesh import Mesh, Material
from .png import write_png


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray = None  # type: ignore
    fov_deg: float = 32.0
    ortho: bool = False
    ortho_height: float = 3.0  # world units visible vertically in ortho mode

    def __post_init__(self):
        self.eye = np.asarray(self.eye, dtype=float)
        self.target = np.asarray(self.target, dtype=float)
        self.up = np.array([0.0, 0.0, 1.0]) if self.up is None else np.asarray(self.up, dtype=float)

    def view_matrix(self) -> np.ndarray:
        f = self.target - self.eye
        f /= np.linalg.norm(f)
        up = self.up
        if abs(np.dot(f, up / np.linalg.norm(up))) > 0.999:
            up = np.array([0.0, 1.0, 0.0])
        s = np.cross(f, up)
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        m = np.eye(4)
        m[0, :3], m[1, :3], m[2, :3] = s, u, -f
        m[:3, 3] = -m[:3, :3] @ self.eye
        return m

    @classmethod
    def orbit(cls, target, distance, azimuth_deg, elevation_deg, **kw) -> "Camera":
        az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
        d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]) * distance
        return cls(np.asarray(target) + d, np.asarray(target), **kw)


class Renderer:
    def __init__(self, width=1200, height=800, supersample=2, background=((0.93, 0.95, 0.98), (0.78, 0.82, 0.87))):
        self.w, self.h, self.ss = width, height, supersample
        self.bg = background
        self.light_dir = self._unit(np.array([-0.5, 0.6, 1.0]))
        self.fill_dir = self._unit(np.array([0.7, -0.4, 0.4]))

    @staticmethod
    def _unit(v):
        return v / np.linalg.norm(v)

    # ------------------------------------------------------------ projection
    NEAR = 0.05

    def _view(self, cam: Camera, pts: np.ndarray) -> np.ndarray:
        V = cam.view_matrix()
        return pts @ V[:3, :3].T + V[:3, 3]

    def _project_view(self, cam: Camera, p: np.ndarray, W: int, H: int):
        """Project view-space points (camera looks down -Z)."""
        aspect = W / H
        if cam.ortho:
            hh = cam.ortho_height / 2
            x = p[:, 0] / (hh * aspect)
            y = p[:, 1] / hh
            depth = -p[:, 2]
        else:
            f = 1.0 / np.tan(np.radians(cam.fov_deg) / 2)
            z = np.maximum(-p[:, 2], 1e-4)
            x = (p[:, 0] * f / aspect) / z
            y = (p[:, 1] * f) / z
            depth = z
        sx = (x * 0.5 + 0.5) * W
        sy = (1 - (y * 0.5 + 0.5)) * H
        return np.column_stack([sx, sy]), depth

    def _project(self, cam: Camera, pts: np.ndarray, W: int, H: int):
        p = self._view(cam, pts)
        scr, depth = self._project_view(cam, p, W, H)
        return scr, depth, p

    def _clip_near(self, cam: Camera, view: np.ndarray, tris: np.ndarray, owner: np.ndarray):
        """Sutherland-Hodgman clip of triangles against the near plane (perspective only).
        Returns (view_points, tris, owner) with clipped triangles re-tessellated."""
        if cam.ortho or len(tris) == 0:
            return view, tris, owner
        near = -self.NEAR
        behind = view[:, 2] > near          # z > -near  -> behind the near plane
        tb = behind[tris]
        n_behind = tb.sum(axis=1)
        keep = n_behind == 0
        mixed = np.where((n_behind > 0) & (n_behind < 3))[0]
        if len(mixed) == 0:
            return view, tris[keep], owner[keep]
        new_pts, new_tris, new_owner = [], [], []
        base = len(view)
        for t in mixed:
            poly = []
            idx = tris[t]
            for k in range(3):
                a, b = idx[k], idx[(k + 1) % 3]
                pa, pb = view[a], view[b]
                ina, inb = not behind[a], not behind[b]
                if ina:
                    poly.append(pa)
                if ina != inb:
                    s = (near - pa[2]) / (pb[2] - pa[2])
                    poly.append(pa + (pb - pa) * s)
            if len(poly) < 3:
                continue
            start = base + len(new_pts)
            new_pts.extend(poly)
            for k in range(1, len(poly) - 1):
                new_tris.append((start, start + k, start + k + 1))
                new_owner.append(owner[t])
        if new_pts:
            view = np.vstack([view, np.asarray(new_pts)])
            tris = np.vstack([tris[keep], np.asarray(new_tris, dtype=int)])
            owner = np.concatenate([owner[keep], np.asarray(new_owner, dtype=int)])
        else:
            tris, owner = tris[keep], owner[keep]
        return view, tris, owner

    # ------------------------------------------------------------ rasterise
    def render(self, mesh: Mesh, cam: Camera, ground: bool = True, lines: bool = True,
               line_color=(0.08, 0.08, 0.1), clip_z: Optional[float] = None, clip_y: Optional[float] = None) -> np.ndarray:
        W, H = self.w * self.ss, self.h * self.ss
        img = self._background(W, H)
        zbuf = np.full((H, W), np.inf)
        tris, owner = mesh.triangulated()
        if len(tris) == 0:
            return self._downsample(img)
        verts = mesh.vertices
        view = self._view(cam, verts)
        fn = mesh.face_normals()
        cent = mesh.face_centroids()
        mats = [mesh.materials.get(n, Material(n)) for n in mesh.face_materials]
        colors = np.array([m.color for m in mats])
        alphas = np.array([m.alpha for m in mats])
        shin = np.array([m.shininess for m in mats])
        emis = np.array([m.emissive for m in mats])
        # optional clipping (cutaway views): drop faces whose centroid is beyond the clip
        keep = np.ones(len(tris), dtype=bool)
        if clip_z is not None:
            keep &= cent[owner][:, 2] < clip_z
        if clip_y is not None:
            keep &= cent[owner][:, 1] < clip_y
        tris, owner = tris[keep], owner[keep]
        view, tris, owner = self._clip_near(cam, view, tris, owner)
        scr, depth = self._project_view(cam, view, W, H)
        # shading per face
        view_dir = self._unit(cam.target - cam.eye)
        ndl = fn @ self.light_dir
        ndf = fn @ self.fill_dir
        facing = -(fn @ view_dir)
        diffuse = 0.42 + 0.48 * np.clip(ndl, 0, 1) + 0.18 * np.clip(ndf, 0, 1)
        # specular highlight (Blinn-ish with a fixed half vector)
        half = self._unit(self.light_dir - view_dir)
        spec = np.clip(fn @ half, 0, 1) ** 40 * shin
        shade = colors * diffuse[:, None] + spec[:, None] * 0.6 + emis[:, None] * colors
        # two-sided: darken back faces slightly
        shade[facing < 0] *= 0.85
        # ground shadow first
        if ground:
            self._ground_shadow(img, zbuf, cam, verts, W, H)
        # opaque pass
        opaque = alphas[owner] >= 0.999
        self._raster(img, zbuf, scr, depth, tris[opaque], owner[opaque], shade, None)
        # transparent pass, back to front
        trans_idx = np.where(~opaque)[0]
        if len(trans_idx):
            d = depth[tris[trans_idx]].mean(axis=1)
            order = trans_idx[np.argsort(-d)]
            self._raster(img, zbuf, scr, depth, tris[order], owner[order], shade, alphas, write_z=False)
        if lines and mesh.lines:
            ls = mesh.lines
            if clip_z is not None:
                ls = [l for l in ls if l[:, 2].max() < clip_z]
            if clip_y is not None:
                ls = [l for l in ls if l[:, 1].max() < clip_y]
            self._draw_lines(img, zbuf, cam, ls, W, H, line_color)
        return self._downsample(img)

    def _background(self, W, H):
        top, bot = np.array(self.bg[0]), np.array(self.bg[1])
        t = np.linspace(0, 1, H)[:, None, None]
        return (top * (1 - t) + bot * t) * np.ones((H, W, 3))

    def _downsample(self, img):
        s = self.ss
        if s == 1:
            return (img * 255)
        H, W = img.shape[0] // s, img.shape[1] // s
        out = img[: H * s, : W * s].reshape(H, s, W, s, 3).mean(axis=(1, 3))
        return out * 255

    def _raster(self, img, zbuf, scr, depth, tris, owner, shade, alphas, write_z=True):
        H, W = zbuf.shape
        if len(tris) == 0:
            return
        P = scr[tris]                       # (T,3,2)
        D = depth[tris]                     # (T,3)
        ok = np.all(D > 0, axis=1)
        mins = np.floor(P.min(axis=1)).astype(int)
        maxs = np.ceil(P.max(axis=1)).astype(int)
        ok &= (maxs[:, 0] >= 0) & (maxs[:, 1] >= 0) & (mins[:, 0] < W) & (mins[:, 1] < H)
        mins = np.clip(mins, 0, [W - 1, H - 1])
        maxs = np.clip(maxs, 0, [W - 1, H - 1])
        ok &= (maxs[:, 0] >= mins[:, 0]) & (maxs[:, 1] >= mins[:, 1])
        det = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1])
        ok &= np.abs(det) > 1e-9
        # tiny triangles (bounding box <= 1 px): a single depth-tested pixel
        tiny = ok & ((maxs[:, 0] - mins[:, 0]) <= 1) & ((maxs[:, 1] - mins[:, 1]) <= 1)
        if tiny.any():
            idx = np.where(tiny)[0]
            px = np.clip(np.round(P[idx].mean(axis=1)).astype(int), 0, [W - 1, H - 1])
            zc = D[idx].mean(axis=1)
            cur = zbuf[px[:, 1], px[:, 0]]
            hit = zc < cur
            if hit.any():
                sel = idx[hit]
                col = shade[owner[sel]]
                if alphas is not None:
                    a = alphas[owner[sel]][:, None]
                    col = img[px[hit, 1], px[hit, 0]] * (1 - a) + col * a
                img[px[hit, 1], px[hit, 0]] = col
                if write_z:
                    zbuf[px[hit, 1], px[hit, 0]] = zc[hit]
        big = np.where(ok & ~tiny)[0]
        for t in big:
            (x0, y0), (x1, y1), (x2, y2) = P[t]
            d = D[t]
            xmin, ymin = mins[t]
            xmax, ymax = maxs[t]
            xs = np.arange(xmin, xmax + 1) + 0.5
            ys = np.arange(ymin, ymax + 1) + 0.5
            X = xs[None, :]
            Y = ys[:, None]
            dt = det[t]
            l1 = ((X - x0) * (y2 - y0) - (x2 - x0) * (Y - y0)) / dt
            l2 = ((x1 - x0) * (Y - y0) - (X - x0) * (y1 - y0)) / dt
            l0 = 1 - l1 - l2
            inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            if not inside.any():
                continue
            zi = 1.0 / (l0 / d[0] + l1 / d[1] + l2 / d[2])
            sub = zbuf[ymin : ymax + 1, xmin : xmax + 1]
            mask = inside & (zi < sub)
            if not mask.any():
                continue
            o = owner[t]
            col = shade[o]
            region = img[ymin : ymax + 1, xmin : xmax + 1]
            if alphas is None:
                region[mask] = col
                sub[mask] = zi[mask]
            else:
                a = alphas[o]
                region[mask] = region[mask] * (1 - a) + col * a
                if write_z:
                    sub[mask] = zi[mask]

    def _ground_shadow(self, img, zbuf, cam, verts, W, H):
        """Soft blob shadow: project the mesh footprint onto z=0 and darken it."""
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        rx, ry = (hi[0] - lo[0]) / 2 * 1.02, (hi[1] - lo[1]) / 2 * 1.08
        n = 64
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = np.column_stack([cx + rx * np.cos(a), cy + ry * np.sin(a), np.zeros(n) + 0.002])
        center = np.array([[cx, cy, 0.002]])
        pts = np.vstack([ring, center])
        scr, depth, _ = self._project(cam, pts, W, H)
        tris = np.array([[n, i, (i + 1) % n] for i in range(n)])
        shade = np.zeros((1, 3))
        # draw with alpha into a temp copy for a soft look
        for k, s in ((1.0, 0.35), (0.8, 0.25), (0.6, 0.2)):
            pts_k = np.vstack([center + (ring - center) * k, center])
            scr_k, depth_k, _ = self._project(cam, pts_k, W, H)
            self._raster(img, zbuf, scr_k, depth_k, tris, np.zeros(n, dtype=int), img.mean() * 0.0 + np.array([[0.55, 0.57, 0.6]]),
                         np.array([s]), write_z=False)

    def _draw_lines(self, img, zbuf, cam, lines, W, H, color):
        col = np.array(color)
        for pts in lines:
            scr, depth, _ = self._project(cam, pts, W, H)
            for k in range(len(pts) - 1):
                self._line(img, zbuf, scr[k], scr[k + 1], depth[k], depth[k + 1], col)

    def _line(self, img, zbuf, a, b, da, db, col):
        H, W = zbuf.shape
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        t = np.linspace(0, 1, n)
        xs = np.round(a[0] + (b[0] - a[0]) * t).astype(int)
        ys = np.round(a[1] + (b[1] - a[1]) * t).astype(int)
        ds = da + (db - da) * t
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs, ys, ds = xs[ok], ys[ok], ds[ok]
        vis = ds <= zbuf[ys, xs] * 1.003 + 0.004
        img[ys[vis], xs[vis]] = img[ys[vis], xs[vis]] * 0.35 + col * 0.65

    # --------------------------------------------------------------- utils
    def save(self, img: np.ndarray, path):
        return write_png(path, img)

    @staticmethod
    def hstack(images: Sequence[np.ndarray], pad=8, bg=240) -> np.ndarray:
        h = max(i.shape[0] for i in images)
        out = []
        for i in images:
            if i.shape[0] < h:
                padrow = np.full((h - i.shape[0], i.shape[1], 3), bg, dtype=float)
                i = np.vstack([i, padrow])
            out.append(i)
            out.append(np.full((h, pad, 3), bg, dtype=float))
        return np.hstack(out[:-1])

    @staticmethod
    def vstack(images: Sequence[np.ndarray], pad=8, bg=240) -> np.ndarray:
        w = max(i.shape[1] for i in images)
        out = []
        for i in images:
            if i.shape[1] < w:
                padcol = np.full((i.shape[0], w - i.shape[1], 3), bg, dtype=float)
                i = np.hstack([i, padcol])
            out.append(i)
            out.append(np.full((pad, w, 3), bg, dtype=float))
        return np.vstack(out[:-1])


def default_materials() -> dict:
    return {
        "paint": Material("paint", (0.72, 0.12, 0.12), 1.0, 0.8),
        "aperture": Material("aperture", (0.35, 0.55, 0.7), 0.5, 0.9),
        "underbody": Material("underbody", (0.12, 0.12, 0.13), 1.0, 0.05),
        "trim": Material("trim", (0.10, 0.10, 0.11), 1.0, 0.2),
        "bed": Material("bed", (0.15, 0.15, 0.16), 1.0, 0.1),
        "default": Material("default", (0.6, 0.6, 0.6)),
    }
