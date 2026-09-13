"""From calibrated measurements to generator parameters, and back onto the photo.

The hub similarity is exact at the wheel faces but the body ends, which wrap
inboard and away from the camera, are foreshortened.  `AnchoredRemap` corrects
that with a monotone curve through published hard points (tail tip, both hubs,
nose tip for x; ground and roof top for z), preserving the traced shape between
anchors.  `mesh_side_outline` + `project_outline` draw a generated body back
onto the reference image so a fit can be judged against the real car.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np

from ..geometry.curves import Profile


class AnchoredRemap:
    """Monotone 1-D map through (measured, true) anchor pairs, with an exact inverse."""

    def __init__(self, measured: Sequence[float], true: Sequence[float], samples: int = 4001):
        m = np.asarray(measured, dtype=float)
        t = np.asarray(true, dtype=float)
        order = np.argsort(m)
        m, t = m[order], t[order]
        if np.any(np.diff(m) <= 0) or np.any(np.diff(t) <= 0):
            raise ValueError("anchors must be strictly increasing in both measured and true values")
        self._prof = Profile(list(zip(m, t)))
        self._m0, self._m1 = m[0], m[-1]
        self._slope0 = (t[1] - t[0]) / (m[1] - m[0])
        self._slope1 = (t[-1] - t[-2]) / (m[-1] - m[-2])
        pad = 0.25 * (m[-1] - m[0])
        self._grid = np.linspace(m[0] - pad, m[-1] + pad, samples)
        self._vals = self(self._grid)

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        out = np.asarray(self._prof(np.clip(v, self._m0, self._m1)), dtype=float)
        out = np.where(v < self._m0, self._prof(self._m0) + (v - self._m0) * self._slope0, out)
        out = np.where(v > self._m1, self._prof(self._m1) + (v - self._m1) * self._slope1, out)
        return out

    def inverse(self, v):
        return np.interp(np.asarray(v, dtype=float), self._vals, self._grid)


def _edge_crossings(mesh, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact surface heights where mesh edges cross each x plane in `xs`.

    Every triangle edge (including fan diagonals, which lie on the surface) is
    intersected with all grid planes it spans; vertices are added in their
    nearest plane so vertical faces register their full height.  The side
    silhouette of a piecewise-linear surface is attained on edges, so the
    per-plane extremes of these samples are exact (no sampling holes).
    """
    tris, _ = mesh.triangulated()
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    v = mesh.vertices
    xa, xb = v[e[:, 0], 0], v[e[:, 1], 0]
    za, zb = v[e[:, 0], 2], v[e[:, 1], 2]
    swap = xa > xb
    xa, xb = np.where(swap, xb, xa), np.where(swap, xa, xb)
    za, zb = np.where(swap, zb, za), np.where(swap, za, zb)
    i0 = np.searchsorted(xs, xa, side="left")
    i1 = np.searchsorted(xs, xb, side="right") - 1
    cnt = np.clip(i1 - i0 + 1, 0, None)
    span = xb - xa
    keep = (cnt > 0) & (span > 1e-12)
    i0, cnt, xa, za, zb, span = i0[keep], cnt[keep], xa[keep], za[keep], zb[keep], span[keep]
    total = int(cnt.sum())
    rep = np.repeat(np.arange(len(cnt)), cnt)
    starts = np.repeat(np.cumsum(cnt) - cnt, cnt)
    idx = np.repeat(i0, cnt) + (np.arange(total) - starts)
    t = (xs[idx] - xa[rep]) / span[rep]
    z = za[rep] + t * (zb[rep] - za[rep])
    vi = np.clip(np.searchsorted(xs, v[:, 0]), 0, len(xs) - 1)
    vi_prev = np.clip(vi - 1, 0, len(xs) - 1)
    nearer = np.abs(xs[vi_prev] - v[:, 0]) < np.abs(xs[vi] - v[:, 0])
    vi = np.where(nearer, vi_prev, vi)
    return np.concatenate([idx, vi]), np.concatenate([z, v[:, 2]])


def mesh_side_outline(mesh, bins: int = 400, exclude_x: Sequence[Tuple[float, float]] = ()) -> Dict[str, np.ndarray]:
    """Side-view outline of a body mesh in world (x, z).

    top/bottom: exact per-plane extreme z of the surface (bottom skips
    `exclude_x` ranges such as wheel arches).  Glass apertures come from the
    mesh zones as closed (x, z) loops, which is what a side photo shows as the
    daylight opening.
    """
    v = mesh.vertices
    xs = np.linspace(v[:, 0].min(), v[:, 0].max(), bins)
    idx, z = _edge_crossings(mesh, xs)
    zmax = np.full(bins, -np.inf)
    zmin = np.full(bins, np.inf)
    np.maximum.at(zmax, idx, z)
    np.minimum.at(zmin, idx, z)
    filled = np.isfinite(zmax)
    top = np.column_stack([xs[filled], zmax[filled]])
    keep = filled.copy()
    for a0, a1 in exclude_x:
        keep &= ~((xs >= a0) & (xs <= a1))
    bottom = np.column_stack([xs[keep], zmin[keep]])
    out = {"top": top, "bottom": bottom}
    for zone, faces in mesh.zones.items():
        if not zone.startswith("aperture/glass") and zone not in ("aperture/windshield", "aperture/rear_window"):
            continue
        if zone.endswith("_R"):
            continue
        loops = mesh.boundary_loops(faces)
        if loops:
            loop = max(loops, key=len)
            out[zone.split("/", 1)[1]] = v[loop][:, [0, 2]]
    return out


def project_outline(outline: Dict[str, np.ndarray], calib, xremap: AnchoredRemap, zremap: AnchoredRemap) -> Dict[str, np.ndarray]:
    """World (x, z) outline -> image pixels via the inverse remaps and the hub calibration."""
    out = {}
    for k, xz in outline.items():
        xz = np.asarray(xz, dtype=float)
        meas = np.column_stack([xremap.inverse(xz[:, 0]), zremap.inverse(xz[:, 1])])
        out[k] = calib.to_pixel(meas)
    return out


# ---------------------------------------------------------------------------
# Perspective depth model
# ---------------------------------------------------------------------------
class DepthModel:
    """Correct hub-plane measurements for depth behind that plane.

    A point `d` metres further from the camera than the wheel faces is imaged
    smaller about the principal point (x_c, z_c) by D / (D + d), where D is the
    camera distance to the hub plane.  D follows from the published length
    against the measured centreline tip span; z_c from the published height
    against the measured roof crown (both tips and crown sit at the centreline
    depth d_c).  x_c is solved from a published overhang split when available,
    otherwise taken from the image centre column.
    """

    def __init__(self, D: float, x_c: float, z_c: float | None, d_c: float):
        self.D, self.x_c, self.z_c, self.d_c = float(D), float(x_c), (None if z_c is None else float(z_c)), float(d_c)

    def k(self, d: float) -> float:
        return (self.D + d) / self.D

    def correct(self, xz, d: float) -> np.ndarray:
        xz = np.atleast_2d(np.asarray(xz, dtype=float)).copy()
        k = self.k(d)
        xz[:, 0] = self.x_c + (xz[:, 0] - self.x_c) * k
        if self.z_c is not None:
            xz[:, 1] = self.z_c + (xz[:, 1] - self.z_c) * k
        return xz

    @classmethod
    def solve(cls, nose_m: float, tail_m: float, roof_m: float, length: float, height: float, d_c: float,
              x_centre_m: float, front_overhang: float | None = None, wheelbase: float | None = None) -> "DepthModel":
        span = nose_m - tail_m
        k = length / span
        if k <= 1.0 + 1e-4:
            raise ValueError(f"measured span {span:.3f} m is not shorter than the published {length:.3f} m; "
                             "the depth model does not apply (camera very far away or a wrong tip)")
        D = d_c / (k - 1.0)
        if front_overhang is not None and wheelbase is not None:
            nose_t = wheelbase / 2 + front_overhang
            x_c = (nose_t - k * nose_m) / (1.0 - k)
        else:
            x_c = x_centre_m
        z_c = (height - k * roof_m) / (1.0 - k)
        return cls(D, x_c, z_c, d_c)


# ---------------------------------------------------------------------------
# Greenhouse features from a measured centreline top curve
# ---------------------------------------------------------------------------
def resample_curve(xz: np.ndarray, step: float = 0.01, smooth_m: float = 0.09) -> Tuple[np.ndarray, np.ndarray]:
    xz = np.asarray(xz, dtype=float)
    order = np.argsort(xz[:, 0])
    x, z = xz[order, 0], xz[order, 1]
    x, idx = np.unique(np.round(x, 4), return_index=True)
    z = z[idx]
    xs = np.arange(x[0], x[-1] + 1e-9, step)
    zs = np.interp(xs, x, z)
    n = max(1, int(round(smooth_m / step)))
    if n > 1:
        zs = np.convolve(np.pad(zs, n // 2, mode="edge"), np.ones(n) / n, mode="valid")[: len(xs)]
    return xs, zs


def greenhouse_features(xz_top: np.ndarray, frac_start: float = 0.35, frac_end: float = 0.45) -> Dict[str, float]:
    """Roof peak, windshield top/base (roof_front, cowl), rear glass top/base (roof_rear, deck).

    Threshold rule on the smoothed slope, relative to each car's own glass
    steepness so it works for a Civic, a Model 3 and an Aventador alike:
    the windshield *starts* where the slope ahead of the roof peak first
    exceeds `frac_start` of the steepest windshield slope, and *ends* (cowl)
    where, past that steepest point, it relaxes below `frac_end` of it.  The
    rear glass is found the same way behind the peak.  (A max-curvature rule
    latches onto small kinks next to the steepest point and was abandoned.)
    """
    x, z = resample_curve(xz_top, step=0.01, smooth_m=0.15)
    s = np.gradient(z, x)
    ip = int(np.argmax(z))
    out = {"x_peak": float(x[ip]), "z_peak": float(z[ip])}
    n = len(x)
    # ---- front: steepest downward slope between the peak and 25 cm short of the nose
    fr = np.where((x > x[ip] + 0.10) & (x < x[-1] - 0.25))[0]
    i_ws = int(fr[np.argmin(s[fr])])
    s_ws = -float(s[i_ws])
    i_rf = next((i for i in range(ip, i_ws + 1) if -s[i] >= frac_start * s_ws), ip)
    i_cowl = next((i for i in range(i_ws, n) if -s[i] <= frac_end * s_ws), n - 1)
    nose_lim = int(np.searchsorted(x, x[-1] - 0.15))
    if i_cowl > nose_lim:
        win = np.arange(max(0, int(np.searchsorted(x, x[-1] - 0.40))), nose_lim + 1)
        i_cowl = int(win[np.argmax(s[win])])
    # ---- rear: steepest upward slope between 25 cm ahead of the tail and the peak
    rr = np.where((x < x[ip] - 0.10) & (x > x[0] + 0.25))[0]
    i_rg = int(rr[np.argmax(s[rr])])
    s_rg = float(s[i_rg])
    i_rr = next((i for i in range(ip, i_rg - 1, -1) if s[i] >= frac_start * s_rg), ip)
    i_deck = next((i for i in range(i_rg, -1, -1) if s[i] <= frac_end * s_rg), 0)
    tail_lim = int(np.searchsorted(x, x[0] + 0.15))
    if i_deck < tail_lim:  # never relaxed: take the flattest point in the last 40 cm before the tail
        win = np.arange(tail_lim, max(tail_lim + 1, int(np.searchsorted(x, x[0] + 0.40))))
        i_deck = int(win[np.argmin(s[win])])
    out.update({
        "x_roof_front": float(x[i_rf]), "z_roof_front": float(z[i_rf]),
        "x_cowl": float(x[i_cowl]), "z_cowl": float(z[i_cowl]),
        "x_roof_rear": float(x[i_rr]), "z_roof_rear": float(z[i_rr]),
        "x_deck": float(x[i_deck]), "z_deck": float(z[i_deck]),
        "windshield_slope": s_ws, "rear_glass_slope": s_rg,
        "windshield_run": float(x[i_cowl] - x[i_rf]), "rear_glass_run": float(x[i_rr] - x[i_deck]),
        "z_front_end": float(z[-1]), "z_rear_end": float(z[0]), "x_front_end": float(x[-1]), "x_rear_end": float(x[0]),
    })
    return out
