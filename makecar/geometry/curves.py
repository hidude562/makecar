"""Curve utilities: monotone cubic profiles, splines through control points,
polyline resampling, and simple shaping functions."""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------- shaping
def smoothstep(e0: float, e1: float, x):
    t = np.clip((np.asarray(x, dtype=float) - e0) / max(e1 - e0, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


# -------------------------------------------------------- monotone profiles
class Profile:
    """A 1-D function defined by keypoints, interpolated with a monotone cubic
    (Fritsch-Carlson) so it never overshoots — ideal for side-view silhouettes
    and plan-view width curves.  Keypoints with `sharp=True` get a C0 corner.
    Evaluation outside the keypoint range clamps to the end values.
    """

    def __init__(self, keypoints, sharp=None):
        pts = sorted(keypoints, key=lambda p: p[0])
        # merge duplicate x by nudging (keeps strict monotonic x)
        xs, ys = [], []
        for x, y in pts:
            if xs and x <= xs[-1]:
                x = xs[-1] + 1e-6
            xs.append(float(x))
            ys.append(float(y))
        self.x = np.asarray(xs)
        self.y = np.asarray(ys)
        n = len(self.x)
        self.sharp = np.zeros(n, dtype=bool)
        if sharp is not None:
            for i, flag in enumerate(sharp):
                if i < n:
                    self.sharp[i] = bool(flag)
        self.m = self._tangents()

    def _tangents(self) -> np.ndarray:
        x, y = self.x, self.y
        n = len(x)
        if n == 1:
            return np.zeros(1)
        h = np.diff(x)
        delta = np.diff(y) / h
        m = np.zeros(n)
        if n == 2:
            m[:] = delta[0]
            return m
        for k in range(1, n - 1):
            if delta[k - 1] * delta[k] <= 0:
                m[k] = 0.0
            else:
                w1 = 2 * h[k] + h[k - 1]
                w2 = h[k] + 2 * h[k - 1]
                m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        # end tangents (one-sided, shape preserving)
        m[0] = self._end_tangent(h[0], h[1], delta[0], delta[1])
        m[-1] = self._end_tangent(h[-1], h[-2], delta[-1], delta[-2])
        return m

    @staticmethod
    def _end_tangent(h0, h1, d0, d1):
        d = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(d) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(d) > abs(3 * d0):
            return 3 * d0
        return d

    def __call__(self, xq):
        xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
        x, y, m = self.x, self.y, self.m
        if len(x) == 1:
            out = np.full_like(xq_arr, y[0])
            return out if np.ndim(xq) else float(out[0])
        idx = np.clip(np.searchsorted(x, xq_arr, side="right") - 1, 0, len(x) - 2)
        h = x[idx + 1] - x[idx]
        t = np.clip((xq_arr - x[idx]) / h, 0.0, 1.0)
        # sharp keypoints: linear interpolation on the adjacent intervals
        m0 = np.where(self.sharp[idx], (y[idx + 1] - y[idx]) / h, m[idx])
        m1 = np.where(self.sharp[idx + 1], (y[idx + 1] - y[idx]) / h, m[idx + 1])
        t2, t3 = t * t, t * t * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        out = h00 * y[idx] + h10 * h * m0 + h01 * y[idx + 1] + h11 * h * m1
        return out if np.ndim(xq) else float(out[0])


# ------------------------------------------------------ splines through points
def catmull_rom_closed(points: np.ndarray, samples_per_segment, sharpness=None) -> np.ndarray:
    """Sample a closed Catmull-Rom spline through `points` with a fixed number
    of samples per segment (topology-stable).  `sharpness[i]` in [0,1] scales
    the tangent at joint i towards zero (1 = perfectly sharp corner).
    Returns the concatenated samples (segment start points included, end
    points excluded so the ring has sum(samples_per_segment) points)."""
    p = np.asarray(points, dtype=float)
    n = len(p)
    if np.isscalar(samples_per_segment):
        samples_per_segment = [int(samples_per_segment)] * n
    if sharpness is None:
        sharpness = np.zeros(n)
    sharpness = np.asarray(sharpness, dtype=float)
    prev = np.roll(p, 1, axis=0)
    nxt = np.roll(p, -1, axis=0)
    tangents = 0.5 * (nxt - prev) * (1.0 - sharpness)[:, None]
    out = []
    for i in range(n):
        j = (i + 1) % n
        k = samples_per_segment[i]
        t = np.linspace(0.0, 1.0, k, endpoint=False)
        out.append(_hermite(p[i], p[j], tangents[i], tangents[j], t))
    return np.vstack(out)


def catmull_rom_open(points: np.ndarray, samples_per_segment, sharpness=None) -> np.ndarray:
    """Open spline through points; returns sum(samples)+1 points including both ends."""
    p = np.asarray(points, dtype=float)
    n = len(p)
    if np.isscalar(samples_per_segment):
        samples_per_segment = [int(samples_per_segment)] * (n - 1)
    if sharpness is None:
        sharpness = np.zeros(n)
    sharpness = np.asarray(sharpness, dtype=float)
    tangents = np.zeros_like(p)
    tangents[1:-1] = 0.5 * (p[2:] - p[:-2])
    tangents[0] = p[1] - p[0]
    tangents[-1] = p[-1] - p[-2]
    tangents *= (1.0 - sharpness)[:, None]
    out = []
    for i in range(n - 1):
        k = samples_per_segment[i]
        t = np.linspace(0.0, 1.0, k, endpoint=False)
        out.append(_hermite(p[i], p[i + 1], tangents[i], tangents[i + 1], t))
    out.append(p[-1:])
    return np.vstack(out)


def _hermite(p0, p1, m0, m1, t):
    t = np.asarray(t)[:, None]
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1


# ------------------------------------------------------------- polylines
def polyline_length(pts: np.ndarray, closed=False) -> float:
    p = np.asarray(pts, dtype=float)
    d = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    if closed:
        d += np.linalg.norm(p[0] - p[-1])
    return float(d)


def resample_polyline(pts: np.ndarray, n: int, closed=False) -> np.ndarray:
    """Resample a polyline to n points spaced uniformly by arc-length."""
    p = np.asarray(pts, dtype=float)
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.repeat(p[:1], n, axis=0)
    s = np.linspace(0.0, total, n, endpoint=not closed)
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    t = (s - cum[idx]) / np.maximum(seg[idx], 1e-12)
    return p[idx] + (p[idx + 1] - p[idx]) * t[:, None]


def circle_points(radius: float, n: int, start_angle=0.0) -> np.ndarray:
    a = start_angle + np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)


def rounded_rect_points(width: float, height: float, radius: float, n_corner: int = 4) -> np.ndarray:
    """Closed 2-D outline of a rounded rectangle centred at the origin (CCW)."""
    r = max(0.0, min(radius, width / 2, height / 2))
    hw, hh = width / 2, height / 2
    pts = []
    centers = [(hw - r, hh - r), (-hw + r, hh - r), (-hw + r, -hh + r), (hw - r, -hh + r)]
    starts = [0.0, np.pi / 2, np.pi, 1.5 * np.pi]
    for (cx, cy), a0 in zip(centers, starts):
        if r <= 1e-9:
            pts.append([cx + (hw if cx > 0 else -hw) * 0, cy])
            # plain corner
            pts[-1] = [np.sign(cx) * hw if cx != 0 else hw, np.sign(cy) * hh if cy != 0 else hh]
            continue
        for a in np.linspace(a0, a0 + np.pi / 2, n_corner, endpoint=True):
            pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.asarray(pts)


def polygon_area_2d(pts: np.ndarray) -> float:
    p = np.asarray(pts, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_centroid_2d(pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=float)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = 0.5 * cross.sum()
    if abs(a) < 1e-12:
        return p.mean(axis=0)
    cx = ((x + xn) * cross).sum() / (6 * a)
    cy = ((y + yn) * cross).sum() / (6 * a)
    return np.array([cx, cy])


def point_in_polygon_2d(pt, poly: np.ndarray) -> bool:
    x, y = pt
    p = np.asarray(poly)
    inside = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside
