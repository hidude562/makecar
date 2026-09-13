"""Perspective camera pose and silhouette matching (analysis by synthesis).

A 3/4 reference photo cannot be traced like a side elevation, but a body mesh
that already matches the side profile can be *projected* into it.  Solving the
camera (yaw, pitch, roll, distance, focal length, principal point) by
maximising silhouette IoU against a segmented photo gives a calibrated view in
which plan-view and cross-section parameters can then be fitted the same way.

World frame: +X forward, +Y left, +Z up.  Image frame: u right, v down.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence, Tuple
import numpy as np


@dataclass
class PinholeCamera:
    yaw: float        # degrees; 0 = camera in front of the car, 90 = on the car's left side
    pitch: float      # degrees above the horizon
    roll: float       # degrees about the viewing axis
    distance: float   # metres from the target point
    focal: float      # pixels
    cx: float         # principal point, pixels
    cy: float
    tx: float = 0.0   # target point the camera looks at (world)
    ty: float = 0.0
    tz: float = 0.7

    def basis(self):
        y, p = np.radians(self.yaw), np.radians(self.pitch)
        target = np.array([self.tx, self.ty, self.tz])
        eye = target + self.distance * np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
        f = target - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, np.array([0.0, 0.0, 1.0]))
        nr = np.linalg.norm(r)
        r = r / nr if nr > 1e-9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(r, f)
        return eye, r, u, f

    def project(self, pts) -> np.ndarray:
        """World points -> (u, v, depth)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        eye, r, u, f = self.basis()
        d = pts - eye
        xc, yc, zc = d @ r, -(d @ u), d @ f
        a = np.radians(self.roll)
        ca, sa = np.cos(a), np.sin(a)
        xr, yr = ca * xc - sa * yc, sa * xc + ca * yc
        zs = np.maximum(zc, 1e-6)
        return np.column_stack([self.cx + self.focal * xr / zs, self.cy + self.focal * yr / zs, zc])


def silhouette(vertices: np.ndarray, tris: np.ndarray, cam: PinholeCamera, size: Tuple[int, int], near: float = 0.05) -> np.ndarray:
    """Binary mask (H, W) of the projected triangle soup."""
    from PIL import Image, ImageDraw

    W, H = size
    uvz = cam.project(vertices)
    P = uvz[tris]
    ok = (P[:, :, 2] > near).all(axis=1)
    uv = P[ok][:, :, :2]
    lo, hi = uv.min(axis=1), uv.max(axis=1)
    inside = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < W) & (lo[:, 1] < H)
    uv = uv[inside]
    im = Image.new("1", (W, H), 0)
    dr = ImageDraw.Draw(im)
    for tri in uv.reshape(-1, 6).tolist():
        dr.polygon(tri, fill=1)
    return np.asarray(im, dtype=bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def resize_mask(mask: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    from PIL import Image

    H, W = mask.shape
    s = min(1.0, max_side / max(H, W))
    if s >= 1.0:
        return mask.copy(), 1.0
    im = Image.fromarray((mask * 255).astype(np.uint8)).resize((max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR)
    return np.asarray(im) > 127, s


def align_to_mask(vertices, tris, cam: PinholeCamera, mask: np.ndarray, iters: int = 3) -> PinholeCamera:
    """Match silhouette area (focal) and centroid (principal point) to the mask."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    area, mcx, mcy = len(xs), xs.mean(), ys.mean()
    for _ in range(iters):
        s = silhouette(vertices, tris, cam, (W, H))
        if s.sum() < 20:
            cam = replace(cam, cx=W / 2, cy=H / 2, focal=cam.focal * 1.5)
            continue
        cam = replace(cam, focal=cam.focal * float(np.sqrt(area / s.sum())))
        s = silhouette(vertices, tris, cam, (W, H))
        if s.sum() < 20:
            continue
        ys2, xs2 = np.nonzero(s)
        cam = replace(cam, cx=cam.cx + (mcx - xs2.mean()), cy=cam.cy + (mcy - ys2.mean()))
    return cam


def search_pose(vertices, tris, mask, yaws: Iterable[float], pitches: Iterable[float], distance: float = 8.0,
                focal: Optional[float] = None) -> Tuple[PinholeCamera, float]:
    H, W = mask.shape
    focal = focal or 1.2 * W
    tz = float(0.5 * (vertices[:, 2].min() + vertices[:, 2].max()))
    best: Optional[Tuple[PinholeCamera, float]] = None
    for yw in yaws:
        for pt in pitches:
            c = PinholeCamera(float(yw), float(pt), 0.0, distance, focal, W / 2, H / 2, tz=tz)
            c = align_to_mask(vertices, tris, c, mask, iters=2)
            sc = iou(silhouette(vertices, tris, c, (W, H)), mask)
            if best is None or sc > best[1]:
                best = (c, sc)
    return best  # type: ignore[return-value]


FREE_POSE = ("yaw", "pitch", "roll", "distance", "focal", "cx", "cy")
_STEP = {"yaw": 5.0, "pitch": 2.0, "roll": 1.5, "distance": 1.5, "focal": 0.08, "cx": 0.02, "cy": 0.02, "tz": 0.1}


def refine_pose(vertices, tris, cam: PinholeCamera, mask: np.ndarray, free: Sequence[str] = FREE_POSE,
                maxiter: int = 400) -> Tuple[PinholeCamera, float]:
    from scipy.optimize import minimize

    H, W = mask.shape
    x0 = np.array([getattr(cam, k) for k in free], dtype=float)
    scale = np.array([_STEP[k] * (cam.focal if k == "focal" else W if k == "cx" else H if k == "cy" else 1.0) for k in free])

    def make(z):
        return replace(cam, **{k: float(v) for k, v in zip(free, x0 + z * scale)})

    def obj(z):
        c = make(z)
        if c.distance < 2.0 or c.focal < 50 or not (-5 <= c.pitch <= 60):
            return 1.0
        return 1.0 - iou(silhouette(vertices, tris, c, (W, H)), mask)

    n = len(free)
    res = minimize(obj, np.zeros(n), method="Nelder-Mead",
                   options={"initial_simplex": np.vstack([np.zeros(n), np.eye(n)]), "maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5})
    return make(res.x), 1.0 - float(res.fun)
