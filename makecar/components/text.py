"""Text as geometry: fonts, layout and filled outlines that wrap onto a body panel.

Fonts are the JSON files in makecar/data/fonts (made from TrueType files by
scripts/extract_font.py): flattened glyph outlines, advances and kerning.
`layout` places glyphs for a block of text (lines, alignment, kerning, letter
spacing, a synthetic slant).  `fill` cuts each glyph into trapezoids with the
nonzero winding rule, so counters (the hole in an O) come out right without any
triangulation library.  `text_mesh` lifts the result onto a surface through a
mapper, so the same text sits flat on a plate or follows the curve of a door.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..geometry.mesh import Mesh
from ..geometry.frame import Frame

FONT_DIR = Path(__file__).resolve().parent.parent / "data" / "fonts"
DEFAULT_FONT = "sans-bold"

Mapper = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]   # (N,2) text-plane metres -> (points (N,3), normals (N,3))


@dataclass
class Glyph:
    adv: float
    contours: List[np.ndarray]       # each (K, 2) in font units, closed


class Font:
    def __init__(self, data: dict):
        self.name = data["name"]
        self.source = data.get("source", "")
        self.units_per_em = float(data["units_per_em"])
        self.ascender = float(data["ascender"])
        self.descender = float(data["descender"])
        self.cap_height = float(data["cap_height"]) or 0.7 * self.units_per_em
        self.x_height = float(data.get("x_height") or 0.5 * self.units_per_em)
        self.glyphs: Dict[str, Glyph] = {
            ch: Glyph(float(g["adv"]), [np.asarray(c, dtype=float) for c in g["contours"] if len(c) >= 3])
            for ch, g in data["glyphs"].items()
        }
        self.kerning: Dict[str, float] = {k: float(v) for k, v in data.get("kerning", {}).items()}

    def glyph(self, ch: str) -> Optional[Glyph]:
        g = self.glyphs.get(ch)
        if g is None and ch.upper() != ch:
            g = self.glyphs.get(ch.upper())
        return g


def font_names() -> List[str]:
    return sorted(p.stem for p in FONT_DIR.glob("*.json"))


@lru_cache(maxsize=16)
def load_font(name: str = DEFAULT_FONT) -> Font:
    """A shipped font by name, or any JSON made by scripts/extract_font.py by path."""
    path = Path(name)
    if not path.suffix == ".json" or not path.exists():
        path = FONT_DIR / f"{name}.json"
    if not path.exists():
        raise KeyError(f"unknown font {name!r}; shipped fonts: {font_names()}")
    return Font(json.loads(path.read_text()))


# ------------------------------------------------------------------ layout
@dataclass
class PlacedGlyph:
    char: str
    contours: List[np.ndarray]       # (K, 2) metres in the text plane (x right, y up), block centred on the origin


@dataclass
class Layout:
    glyphs: List[PlacedGlyph]
    width: float
    height: float
    line_widths: List[float]
    size: float = 0.1                # cap height the block was laid out with


def layout(text: str, font: Font, size: float, align: str = "center", line_spacing: float = 1.15,
           letter_spacing: float = 0.0, slant_deg: float = 0.0, kerning: bool = True) -> Layout:
    """Place the glyphs of `text` (lines split on newlines).  `size` is the cap height in metres.
    The block is centred on the origin: x spans [-w/2, w/2]; y runs from the last baseline
    (descenders hang below) up to the cap line of the first line."""
    scale = size / font.cap_height
    lines = text.split("\n")
    line_h = size * line_spacing * (font.ascender - font.descender) / font.cap_height
    placed_lines: List[List[Tuple[str, Glyph, float]]] = []
    widths: List[float] = []
    for line in lines:
        pen = 0.0
        row: List[Tuple[str, Glyph, float]] = []
        prev = None
        for ch in line:
            g = font.glyph(ch)
            if g is None:
                prev = None
                continue
            if kerning and prev is not None:
                pen += font.kerning.get(prev + ch, 0.0) * scale
            row.append((ch, g, pen))
            pen += g.adv * scale + letter_spacing
            prev = ch
        if row:
            pen -= letter_spacing
        placed_lines.append(row)
        widths.append(max(pen, 0.0))
    width = max(widths) if widths else 0.0
    height = size + line_h * (len(lines) - 1)
    shear = np.tan(np.radians(slant_deg))
    glyphs: List[PlacedGlyph] = []
    for i, row in enumerate(placed_lines):
        baseline = height / 2 - size - i * line_h
        dx = {"left": -width / 2, "right": width / 2 - widths[i]}.get(align, -widths[i] / 2)
        for ch, g, pen in row:
            contours = []
            for c in g.contours:
                pts = c * scale
                pts = np.column_stack([pts[:, 0] + pen + dx + pts[:, 1] * shear, pts[:, 1] + baseline])
                contours.append(pts)
            glyphs.append(PlacedGlyph(ch, contours))
    return Layout(glyphs, width, height, widths, size)


# ------------------------------------------------------------------ filling
def _intervals_at(E: np.ndarray, direction: np.ndarray, y: float) -> List[Tuple[int, int]]:
    """Filled x-intervals of the scanline at `y` as (left edge, right edge) indices, nonzero winding."""
    y0, y1 = E[:, 1], E[:, 3]
    crossing = np.where(((y0 < y) & (y <= y1)) | ((y1 < y) & (y <= y0)))[0]
    if len(crossing) < 2:
        return []
    t = (y - y0[crossing]) / (y1[crossing] - y0[crossing])
    x = E[crossing, 0] + t * (E[crossing, 2] - E[crossing, 0])
    order = np.argsort(x, kind="stable")
    out: List[Tuple[int, int]] = []
    winding, start = 0, -1
    for k in order:
        before = winding
        winding += int(direction[crossing[k]])
        if before == 0 and winding != 0:
            start = int(crossing[k])
        elif before != 0 and winding == 0 and start >= 0:
            out.append((start, int(crossing[k])))
            start = -1
    return out


def _x_on(E: np.ndarray, e: int, y: float) -> float:
    x0, y0, x1, y1 = E[e]
    return float(x0 + (y - y0) / (y1 - y0) * (x1 - x0))


def fill(contours: Sequence[np.ndarray], band: float) -> List[np.ndarray]:
    """Trapezoids (4, 2, counter-clockwise) covering the nonzero-winding interior of closed contours.

    Scan bands are about `band` tall and are cut wherever an outline turns around in y, so
    inside a band the crossings pair up and each filled span is one quad.  Curved outlines
    are chords across the band (well under a millimetre off at the sizes cars use)."""
    edges, critical = [], []
    for c in contours:
        q = np.asarray(c, dtype=float)
        y = q[:, 1]
        dy_in, dy_out = y - np.roll(y, 1), np.roll(y, -1) - y
        critical.extend(y[(dy_in * dy_out) <= 0].tolist())
        nxt = np.roll(q, -1, axis=0)
        for (x0, y0), (x1, y1) in zip(q, nxt):
            if y0 != y1:
                edges.append((x0, y0, x1, y1))
    if not edges:
        return []
    E = np.asarray(edges)
    direction = np.where(E[:, 3] > E[:, 1], 1, -1)
    y_min, y_max = float(E[:, 1:4:2].min()), float(E[:, 1:4:2].max())
    n_bands = max(1, int(np.ceil((y_max - y_min) / max(band, 1e-9))))
    ys = np.unique(np.concatenate([np.linspace(y_min, y_max, n_bands + 1), np.asarray(critical)]))
    out: List[np.ndarray] = []

    def emit(ya: float, yb: float, depth: int = 0):
        if yb - ya < 1e-9:
            return
        delta = (yb - ya) * 1e-3
        ia, ib = _intervals_at(E, direction, ya + delta), _intervals_at(E, direction, yb - delta)
        if len(ia) != len(ib):
            if depth < 6:
                mid = 0.5 * (ya + yb)
                emit(ya, mid, depth + 1)
                emit(mid, yb, depth + 1)
            return
        for (la, ra), (lb, rb) in zip(ia, ib):
            out.append(np.array([[_x_on(E, la, ya), ya], [_x_on(E, ra, ya), ya], [_x_on(E, rb, yb), yb], [_x_on(E, lb, yb), yb]]))

    for ya, yb in zip(ys[:-1], ys[1:]):
        emit(float(ya), float(yb))
    return out


# ------------------------------------------------------------------ mesh
def text_mesh(lay: Layout, mapper: Mapper, thickness: float, material: str, base: float = 0.0,
              name: str = "text") -> Mesh:
    """Solid raised letters on a surface: filled tops plus walls around every quad, grouped per glyph.
    Walls between stacked quads end up inside the letter, so the outside reads as one solid."""
    band = max(lay.size, 1e-3) / 16.0
    verts: List[np.ndarray] = []
    faces: List[Tuple[int, ...]] = []
    groups: List[Tuple[str, int, int, int, int]] = []
    n_v = 0
    for k, g in enumerate(lay.glyphs):
        quads = fill(g.contours, band)
        if not quads:
            continue
        v0, f0 = n_v, len(faces)
        P = np.vstack(quads)
        pts, nrm = mapper(P)
        bot, top = pts + nrm * base, pts + nrm * (base + thickness)
        for q in range(len(quads)):
            i = 4 * q
            a, b, c = top[i], top[i + 1], top[i + 2]
            flip = np.dot(np.cross(b - a, c - a), nrm[i]) < 0
            ids = [n_v + i, n_v + i + 1, n_v + i + 2, n_v + i + 3]
            if flip:
                ids = ids[::-1]
            faces.append(tuple(ids))
            B = n_v + 4 * len(quads)   # bottom copies follow all the tops
            for e in range(4):
                p, r = ids[e], ids[(e + 1) % 4]
                faces.append((p - n_v + B, r - n_v + B, r, p))
        verts.append(top)
        verts.append(bot)
        n_v += 2 * len(top)
        groups.append((f"glyph_{k}", v0, n_v, f0, len(faces)))
    if not verts:
        return Mesh(name=name)
    m = Mesh(np.vstack(verts), faces, [material] * len(faces), name=name)
    for gname, va, vb, fa, fb in groups:
        m.add_group(gname, range(va, vb))
        m.add_zone(gname, range(fa, fb))
    return m


def outlined(lay: Layout, width: float, directions: int = 12) -> Layout:
    """The layout with every glyph grown by `width`: the union of copies shifted around a circle.
    Filled with the nonzero rule, the copies never cancel, so this is a clean outline layer."""
    angles = np.linspace(0.0, 2 * np.pi, directions, endpoint=False)
    shifts = np.column_stack([np.cos(angles), np.sin(angles)]) * width
    glyphs = [PlacedGlyph(g.char, [c + sh for sh in shifts for c in g.contours] + list(g.contours)) for g in lay.glyphs]
    return Layout(glyphs, lay.width + 2 * width, lay.height + 2 * width, lay.line_widths, lay.size)


def text_layers(lay: Layout, mapper: Mapper, thickness: float, material: str, base: float = 0.0,
                outline: float = 0.0, outline_material: Optional[str] = None, name: str = "text") -> Mesh:
    """Letters with an optional outline layer underneath (groups: glyph_k and outline/glyph_k)."""
    m = Mesh(name=name)
    if outline > 0 and outline_material:
        m.merge(text_mesh(outlined(lay, outline), mapper, thickness, outline_material, base, name + "_outline"), group_prefix="outline")
        base += thickness + 0.0002
    m.merge(text_mesh(lay, mapper, thickness, material, base, name))
    return m


def flat_text(text: str, font: str, size: float, material: str, *, origin=(0.0, 0.0, 0.0), advance=(1.0, 0.0), up=(0.0, 1.0),
              thickness: float = 0.001, base: float = 0.0, fit_width: Optional[float] = None, outline: float = 0.0,
              outline_material: Optional[str] = None, **layout_kw) -> Mesh:
    """Text in a local XY plane (letters rise along +z), for lettering built inside a component's own frame.
    `advance` / `up` are the reading directions in that plane; `origin` is the block centre."""
    fnt = load_font(font)
    lay_fn = lambda sz: layout(text, fnt, sz, **layout_kw)  # noqa: E731
    lay = sized_to_fit(lay_fn, size, fit_width) if fit_width else lay_fn(size)
    mapper = planar_mapper(Frame.identity(origin), np.asarray(advance, dtype=float), np.asarray(up, dtype=float), np.zeros(3))
    return text_layers(lay, mapper, thickness, material, base, outline, outline_material)


# ------------------------------------------------------------------ mappers
def planar_mapper(frame, advance_local: np.ndarray, up_local: np.ndarray, origin_local: np.ndarray) -> Mapper:
    """Text plane -> world through a connector frame: `advance_local` / `up_local` are unit
    vectors in the frame's XY plane; `origin_local` is the block centre."""

    def mapper(P: np.ndarray):
        local = origin_local[None, :2] + P[:, :1] * advance_local[None, :] + P[:, 1:2] * up_local[None, :]
        pts = frame.to_world(np.column_stack([local, np.zeros(len(local))]))
        return pts, np.tile(frame.z_axis, (len(pts), 1))

    return mapper


def sized_to_fit(lay_fn: Callable[[float], Layout], size: float, room: float) -> Layout:
    """Shrink the cap height until the block fits in `room` metres."""
    lay = lay_fn(size)
    if lay.width > room > 0:
        lay = lay_fn(size * room / lay.width)
    return lay
