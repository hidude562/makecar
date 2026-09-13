"""Calibration and edge tracing for side-profile reference images."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np


def load_rgb(path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def tyre_radius(width_mm: float, aspect_pct: float, rim_in: float) -> float:
    """Nominal unloaded tyre radius in metres from an ISO size like 235/40 R18."""
    return (rim_in * 25.4 + 2 * width_mm * aspect_pct / 100.0) / 2000.0


@dataclass
class Calibration:
    """Pixel (u right, v down) <-> car world (x forward, z up) similarity transform."""

    scale: float          # metres per pixel
    angle: float          # image roll, radians
    origin_px: np.ndarray  # pixel that maps to world (0, 0) before z offset
    sign_x: float          # +1 if the car's front is to the right in the image
    hubs_px: Tuple[Tuple[float, float], Tuple[float, float]]
    hubs_world: Tuple[Tuple[float, float], Tuple[float, float]]

    @classmethod
    def from_hubs(cls, front_px, rear_px, wheelbase: float, r_front: float, r_rear: float) -> "Calibration":
        f = np.asarray(front_px, dtype=float)
        r = np.asarray(rear_px, dtype=float)
        sign_x = 1.0 if f[0] > r[0] else -1.0
        # world points in a (x, -z) frame so both systems have y pointing down
        fw = np.array([wheelbase / 2, -r_front])
        rw = np.array([-wheelbase / 2, -r_rear])
        # mirror pixel u when the front is on the left, so x grows with u'
        fp = np.array([f[0] * sign_x, f[1]])
        rp = np.array([r[0] * sign_x, r[1]])
        dp, dw = fp - rp, fw - rw
        scale = np.linalg.norm(dw) / np.linalg.norm(dp)
        angle = np.arctan2(dw[1], dw[0]) - np.arctan2(dp[1], dp[0])
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        origin = rw - scale * rot @ rp  # world = scale*R*p + origin
        cal = cls(scale, angle, origin, sign_x, (tuple(f), tuple(r)), (tuple(fw * [1, -1]), tuple(rw * [1, -1])))
        return cal

    def _rot(self):
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array([[c, -s], [s, c]])

    def to_world(self, uv) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        p = np.column_stack([uv[:, 0] * self.sign_x, uv[:, 1]])
        w = (self.scale * (self._rot() @ p.T)).T + self.origin_px
        return np.column_stack([w[:, 0], -w[:, 1]])  # (x, z)

    def to_pixel(self, xz) -> np.ndarray:
        xz = np.atleast_2d(np.asarray(xz, dtype=float))
        w = np.column_stack([xz[:, 0], -xz[:, 1]]) - self.origin_px
        p = (self._rot().T @ w.T).T / self.scale
        return np.column_stack([p[:, 0] * self.sign_x, p[:, 1]])

    def residual(self) -> float:
        """Round-trip error of the hub fit in metres (should be ~0)."""
        got = self.to_world(np.array(self.hubs_px))
        return float(np.abs(got - np.array(self.hubs_world)).max())


def _signal(img: np.ndarray, kind: str, ref_rgb: Optional[Sequence[float]]) -> np.ndarray:
    if kind == "luma":
        return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    if kind == "colour":
        ref = np.asarray(ref_rgb, dtype=np.float32)
        return -np.linalg.norm(img - ref, axis=-1)  # high = close to the body colour
    raise ValueError(kind)


def trace_edge(img: np.ndarray, u0: int, u1: int, band: Union[Tuple[float, float], Callable[[float], Tuple[float, float]]],
               polarity: str = "rise", signal: str = "luma", ref_rgb=None, step: int = 4, smooth: int = 3,
               min_contrast: float = 12.0, pick: str = "strongest") -> np.ndarray:
    """Trace a roughly horizontal edge between columns u0..u1.

    For each sampled column the signal is scanned downward inside `band`
    (v_min, v_max) — a constant or a function of u.  `polarity` "rise" finds
    the largest increase going down (dark above, light below), "fall" the
    largest decrease.  `pick` "first"/"last" takes the first/last edge above
    `min_contrast` instead of the strongest.  Returns (N, 2) pixel coordinates
    with sub-pixel v; columns without a confident edge are omitted.
    """
    sig = _signal(img, signal, ref_rgb)
    h = sig.shape[0]
    out = []
    lo_u, hi_u = sorted((int(u0), int(u1)))
    for u in range(lo_u, hi_u + 1, step):
        vmin, vmax = band(u) if callable(band) else band
        a, b = int(max(1, vmin)), int(min(h - 2, vmax))
        if b - a < 4:
            continue
        col = sig[a:b, max(0, u - 1): u + 2].mean(axis=1)
        if smooth > 1:
            k = np.ones(smooth) / smooth
            col = np.convolve(col, k, mode="same")
        g = np.diff(col)
        if polarity == "fall":
            g = -g
        if pick == "strongest":
            i = int(np.argmax(g))
        else:
            idx = np.where(g > min_contrast)[0]
            if not len(idx):
                continue
            # the local maximum of the first / last qualifying run
            j = idx[0] if pick == "first" else idx[-1]
            run = [j]
            direction = 1 if pick == "first" else -1
            while 0 <= run[-1] + direction < len(g) and g[run[-1] + direction] > min_contrast:
                run.append(run[-1] + direction)
            i = max(run, key=lambda t: g[t])
        if g[i] < min_contrast:
            continue
        # parabolic sub-pixel refinement
        if 0 < i < len(g) - 1:
            y0, y1, y2 = g[i - 1], g[i], g[i + 1]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
        else:
            off = 0.0
        out.append((u, a + i + 0.5 + off))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def clean_trace(uv: np.ndarray, window: int = 7, max_dev_px: float = 6.0) -> np.ndarray:
    """Drop points that deviate from a rolling median (reflections, clutter)."""
    if len(uv) < window:
        return uv
    v = uv[:, 1]
    half = window // 2
    med = np.array([np.median(v[max(0, i - half): i + half + 1]) for i in range(len(v))])
    keep = np.abs(v - med) <= max_dev_px
    return uv[keep]


def resample_xz(xz: np.ndarray, n: int = 60, x_range: Optional[Tuple[float, float]] = None) -> List[List[float]]:
    """Sort by x, average duplicates and resample to n evenly spaced samples."""
    xz = np.asarray(xz, dtype=float)
    order = np.argsort(xz[:, 0])
    x, z = xz[order, 0], xz[order, 1]
    lo, hi = x_range if x_range else (x[0], x[-1])
    xs = np.linspace(lo, hi, n)
    zs = np.interp(xs, x, z)
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(xs, zs)]


def draw_overlay(image_path, out_path, traces: dict, calib: Optional[Calibration] = None, crop=None, scale: float = 0.5,
                 markers: Optional[dict] = None):
    """Draw pixel-space traces (name -> (N,2) uv) and hub marks for review."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    colours = [(255, 60, 60), (60, 220, 90), (70, 150, 255), (255, 210, 0), (255, 80, 255), (0, 230, 230), (255, 140, 0)]
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    for k, (name, uv) in enumerate(traces.items()):
        uv = np.asarray(uv)
        if not len(uv):
            continue
        col = colours[k % len(colours)]
        d.line([tuple(p) for p in uv], fill=col, width=4)
        d.text((uv[len(uv) // 2][0], uv[len(uv) // 2][1] - 34), name, fill=col, font=font)
    if calib is not None:
        for (u, v) in calib.hubs_px:
            d.ellipse([u - 9, v - 9, u + 9, v + 9], outline=(255, 255, 0), width=4)
    for name, (u, v) in (markers or {}).items():
        d.line([(u, v - 40), (u, v + 40)], fill=(255, 255, 255), width=3)
        d.text((u + 6, v - 40), name, fill=(255, 255, 255), font=font)
    if crop:
        im = im.crop(crop)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    im.save(out_path, quality=90)
    return out_path
