"""Fleet and police equipment, plus decals (lettering, stripes) on body panels.

Exterior parts attach to the equipment mounts every body emits (`roof_mount`,
`bumper_front`, `spotlight_L/R`, `antenna_aux_L/R`, the `panel_*` grids); the
cabin parts attach to `cabin_partition` and to sub-connectors that the console,
headliner and parcel shelf emit when asked to (`laptop_mount`, `visor_lights`,
`deck_lights`).  None of these is a default, so a plain car never grows a light bar.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from ..geometry.mesh import Mesh, Material
from ..geometry.frame import Frame
from ..geometry import primitives as P
from ..geometry.curves import rounded_rect_points
from ..connectors import PointConnector, PolygonConnector, RectangleConnector
from .base import CarComponent, ComponentResult, BuildContext, register
from .exterior import _part, _pipe
from .fills import grid_mesh
from .text import DEFAULT_FONT, load_font, layout as text_layout, planar_mapper, sized_to_fit, text_layers
from . import interior_helpers as H

LENS_COLORS = {"red": "#d8262d", "blue": "#2557d6", "amber": "#e9a33a", "clear": "#e8eef5"}


def _lens(name: str, alpha=0.85, emissive=0.35) -> Material:
    return Material(f"lens_{name}", Material.parse_color(LENS_COLORS[name]), alpha, 0.95, emissive)


def _materials(ctx: BuildContext) -> Dict[str, Material]:
    return {
        "equip_black": ctx.material("equip_black", "#191b1e", shininess=0.3),
        "equip_steel": ctx.material("equip_steel", "#2c2f33", shininess=0.45, metallic=0.6),
        "equip_grey": ctx.material("equip_grey", "#4a4d52", shininess=0.35),
        "chrome": ctx.material("chrome", ctx.palette.chrome, shininess=0.85, metallic=0.9),
        "screen_dark": ctx.material("screen_dark", "#0e1118", shininess=0.9, emissive=0.15),
        "poly_clear": Material("poly_clear", (0.75, 0.82, 0.9), 0.3, 0.95),
        "lens_red": _lens("red"), "lens_blue": _lens("blue"), "lens_amber": _lens("amber"),
        "lens_clear": _lens("clear", 0.6, 0.25),
    }


def _finish(m: Mesh, ctx: BuildContext, extra: Optional[Dict[str, Material]] = None) -> Mesh:
    mats = _materials(ctx)
    if extra:
        mats.update(extra)
    used = set(m.face_materials)
    m.materials.update({k: v for k, v in mats.items() if k in used})
    return m


def _lens_for(side: str) -> str:
    """US convention: red on the driver's (left) side, blue on the right."""
    return "lens_red" if side == "left" else "lens_blue"


# =============================================================== ROOF LIGHT BAR
@register
class LightBar(CarComponent):
    """Roof light bar on two feet.  The `roof_mount` connector carries the roof
    crown profile, so the feet reach the curved roof and the bar stays level."""

    name = "light.bar"
    accepts = (RectangleConnector,)
    default_for = ()
    options = {"length": None, "height": 0.06, "depth": 0.30, "colors": "red_blue", "takedown": True, "low_profile": False}
    description = "roof-mounted emergency light bar (red/blue, all blue, all red or amber) with takedown lights"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        prof = conn.meta.get("profile")
        lp = conn.frame.to_local(np.asarray(prof, dtype=float)) if prof is not None else None
        L = float(opts["length"] or min(1.25, 0.82 * conn.height))
        Hb = 0.045 if opts["low_profile"] else float(opts["height"])
        D = float(opts["depth"])

        def roof_z(y: float) -> float:
            if lp is None:
                return 0.0
            order = np.argsort(lp[:, 1])
            return float(np.interp(y, lp[order, 1], lp[order, 2]))

        z_bot = max(roof_z(y) for y in np.linspace(-L / 2, L / 2, 25)) + 0.035
        m = Mesh(name="light_bar")
        _part(m, P.rounded_box(D * 0.9, L, 0.014, 0.03, material="equip_black", center=(0, 0, z_bot + 0.007)), "bar_base")
        n = max(4, int(L / 0.11))
        edges = np.linspace(-L / 2, L / 2, n + 1)
        scheme = str(opts["colors"])
        for k in range(n):
            y0, y1 = edges[k], edges[k + 1]
            yc = 0.5 * (y0 + y1)
            if opts["takedown"] and abs(yc) < L * 0.12:
                mat = "lens_clear"
            elif scheme == "red_blue":
                mat = "lens_red" if yc > 0 else "lens_blue"
            else:
                mat = {"blue": "lens_blue", "red": "lens_red", "amber": "lens_amber"}.get(scheme, "lens_blue")
            _part(m, P.box(D * 0.86, (y1 - y0) - 0.006, Hb - 0.022, material=mat, bevel=0.005,
                           center=(0, yc, z_bot + 0.014 + (Hb - 0.022) / 2)), f"module_{k}")
            if 0 < k:
                _part(m, P.box(D * 0.88, 0.006, Hb - 0.022, material="equip_black",
                               center=(0, y0, z_bot + 0.014 + (Hb - 0.022) / 2)), f"divider_{k}")
        _part(m, P.rounded_box(D * 0.9, L, 0.008, 0.03, material="equip_black", center=(0, 0, z_bot + Hb - 0.004)), "bar_top")
        for y in (-0.36 * L, 0.36 * L):
            zr = roof_z(y)
            foot_h = z_bot - zr
            _part(m, P.box(0.10, 0.05, foot_h, material="equip_steel", center=(0, y, zr + foot_h / 2)), "foot")
            _part(m, P.rounded_box(0.14, 0.08, 0.006, 0.02, material="equip_black", center=(0, y, zr + 0.003)), "foot_pad")
        return ComponentResult(_finish(m, ctx), [], {"length": L, "height": Hb})


# =============================================================== PUSH BUMPER
@register
class PushBar(CarComponent):
    """Tubular push bumper on the front fascia.  Local frame of `bumper_front`:
    x lateral (car right), y down, z forward."""

    name = "bumper.push_bar"
    accepts = (RectangleConnector,)
    default_for = ()
    options = {"width": None, "wrap": True, "grille_guard": True, "siren": False}
    description = "black tubular push bumper with uprights, a lower wrap bar and an optional siren speaker"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        W = float(opts["width"] or min(0.62, conn.width * 0.55))
        h = conn.height
        z0 = float(conn.origin[2])
        top = float(conn.meta.get("grille_top_z", z0 + h / 2 + 0.25)) - z0
        bot = -h / 2 + 0.02
        mid = min(h / 2 * 0.15, top - 0.12)
        stand, r = 0.085, 0.022

        def pt(lat, up, fwd):
            return [lat, -up, fwd]

        m = Mesh(name="push_bar")
        for lat in (-W / 2, W / 2):
            path = [pt(lat, bot, 0.035), pt(lat, bot, stand), pt(lat, top - 0.03, stand), pt(lat, top, stand - 0.025)]
            _part(m, _pipe(path, r, "equip_black"), f"upright_{lat:+.2f}")
            _part(m, P.box(0.05, 0.03, stand, material="equip_steel", center=(lat, -mid, stand / 2)), f"bracket_{lat:+.2f}")
        _part(m, _pipe([pt(-W / 2, top - 0.03, stand), pt(W / 2, top - 0.03, stand)], r, "equip_black"), "top_rail")
        _part(m, _pipe([pt(-W / 2, mid, stand), pt(W / 2, mid, stand)], r, "equip_black"), "mid_rail")
        if opts["wrap"]:
            Ww = min(conn.width * 0.92, W + 0.55)
            path = [pt(-Ww / 2, bot + 0.05, 0.012), pt(-Ww / 2 + 0.09, bot + 0.05, stand),
                    pt(Ww / 2 - 0.09, bot + 0.05, stand), pt(Ww / 2, bot + 0.05, 0.012)]
            _part(m, _pipe(path, r * 1.15, "equip_black"), "wrap_bar")
        if opts["grille_guard"]:
            for lat in np.linspace(-W / 4, W / 4, 3):
                _part(m, _pipe([pt(lat, mid + 0.02, stand + 0.004), pt(lat, top - 0.05, stand + 0.004)], 0.009, "equip_black"),
                      f"guard_{lat:+.2f}")
        if opts["siren"]:
            _part(m, P.box(0.17, 0.085, 0.10, material="equip_black", center=(W * 0.22, -(mid - 0.075), stand - 0.055)), "siren_box")
            _part(m, P.cylinder(0.035, 0.012, 20, material="equip_grey", center=(W * 0.22, -(mid - 0.075), stand - 0.002)), "siren_horn")
        return ComponentResult(_finish(m, ctx), [], {"width": W, "top": top})


# =============================================================== A-PILLAR SPOTLIGHT
@register
class Spotlight(CarComponent):
    """Pillar-mounted spotlight: shaft through the A pillar, knuckle, lamp head facing forward."""

    name = "light.spotlight"
    accepts = (PointConnector,)
    default_for = ()
    options = {"finish": "chrome", "radius": 0.072}
    description = "A-pillar spotlight on a through-pillar handle shaft"

    def build_local(self, conn: PointConnector, opts, ctx) -> ComponentResult:
        R = float(opts["radius"])
        mat = "chrome" if opts["finish"] == "chrome" else "equip_black"
        m = Mesh(name="spotlight")
        _part(m, P.cylinder(0.011, 0.20, 12, material="equip_steel", center=(0, 0, 0.04)), "shaft")
        _part(m, P.cylinder(0.021, 0.04, 12, material="equip_black", center=(0, 0, 0.012)), "boot")
        _part(m, P.cylinder(0.024, 0.05, 12, material=mat, center=(0, 0, 0.13)), "knuckle")
        _part(m, _pipe([[0, 0, 0.13], [0.075, 0, 0.13]], 0.012, mat), "arm")
        head = P.cylinder(R, 0.11, 24, material=mat, radius_top=R * 1.05).transform(H.rot_y(np.pi / 2)).translate([0.135, 0, 0.13])
        _part(m, head, "lamp_head")
        ring = P.tube(R * 1.05 + 0.004, R * 1.05 - 0.008, 0.012, 24, material="chrome").transform(H.rot_y(np.pi / 2)).translate([0.195, 0, 0.13])
        _part(m, ring, "bezel")
        lens = P.cylinder(R * 1.05 - 0.007, 0.006, 24, material="lens_clear").transform(H.rot_y(np.pi / 2)).translate([0.199, 0, 0.13])
        _part(m, lens, "lens")
        return ComponentResult(_finish(m, ctx))


# =============================================================== WHIP ANTENNA
@register
class WhipAntenna(CarComponent):
    name = "antenna.whip"
    accepts = (PointConnector,)
    default_for = ()
    options = {"length": 0.75, "rake_deg": 8.0}
    description = "radio whip antenna on a spring base (fits the roof antenna mount or the aux mounts)"

    def build_local(self, conn: PointConnector, opts, ctx) -> ComponentResult:
        L = float(opts["length"])
        m = Mesh(name="whip")
        _part(m, P.cylinder(0.02, 0.018, 16, material="equip_black", center=(0, 0, 0.009)), "base")
        mast = Mesh(name="mast")
        _part(mast, P.cylinder(0.008, 0.05, 12, material="equip_steel", center=(0, 0, 0.043)), "spring")
        _part(mast, P.cylinder(0.003, L, 8, material="equip_black", radius_top=0.0015, center=(0, 0, 0.068 + L / 2)), "rod")
        _part(mast, P.cylinder(0.005, 0.01, 8, material="equip_black", center=(0, 0, 0.068 + L + 0.005)), "tip")
        mast.translate([0, 0, -0.018]).transform(H.rot_y(-np.radians(float(opts["rake_deg"])))).translate([0, 0, 0.018])
        m.merge(mast)
        return ComponentResult(_finish(m, ctx), [], {"length": L})


# =============================================================== CABIN PARTITION
@register
class Partition(CarComponent):
    """Prisoner partition behind the front seats.  Local frame of `cabin_partition`:
    x lateral (car left), y down, z forward."""

    name = "partition.cage"
    accepts = (RectangleConnector,)
    default_for = ()
    options = {"style": "bars", "bar_pitch": 0.065}
    description = "cabin partition: solid lower panel, then steel bars or a clear polycarbonate pane"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        W, Hc = conn.width, conn.height
        z_c = float(conn.origin[2])
        y_floor = Hc / 2
        y_belt = -(float(conn.meta.get("belt_z", z_c)) - z_c) + 0.06
        y_top = -Hc / 2 + 0.02
        m = Mesh(name="partition")
        _part(m, P.box(W - 0.04, y_floor - y_belt, 0.02, material="equip_grey",
                       center=(0, 0.5 * (y_floor + y_belt), 0)), "lower_panel")
        _part(m, P.box(W - 0.04, 0.03, 0.03, material="equip_steel", center=(0, y_belt - 0.015, 0)), "belt_rail")
        _part(m, P.box(W - 0.04, 0.03, 0.03, material="equip_steel", center=(0, y_top + 0.015, 0)), "top_rail")
        for x in (-(W / 2 - 0.035), W / 2 - 0.035):
            _part(m, P.box(0.03, y_floor - y_top, 0.03, material="equip_steel", center=(x, 0.5 * (y_floor + y_top), 0)), "post")
        if opts["style"] == "clear":
            _part(m, P.box(W - 0.10, y_belt - y_top - 0.03, 0.008, material="poly_clear",
                           center=(0, 0.5 * (y_belt + y_top), 0)), "pane")
        else:
            pitch = float(opts["bar_pitch"])
            n = max(3, int((W - 0.12) / pitch))
            for x in np.linspace(-(W - 0.12) / 2, (W - 0.12) / 2, n):
                _part(m, _pipe([[x, y_belt - 0.02, 0], [x, y_top + 0.02, 0]], 0.007, "equip_steel"), "bar")
        return ComponentResult(_finish(m, ctx), [], {"style": opts["style"]})


# =============================================================== LAPTOP MOUNT
@register
class LaptopMount(CarComponent):
    """Pedestal computer mount on the console, tray turned toward the driver."""

    name = "mount.laptop"
    accepts = (RectangleConnector,)
    default_for = ("laptop",)
    options = {"height": 0.28, "turn_deg": 22.0, "screen_open_deg": 105.0}
    description = "pedestal laptop mount with an open laptop, turned toward the driver"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        side = 1.0 if conn.meta.get("driver_y", 0.4) > 0 else -1.0
        hp = float(opts["height"])
        m = Mesh(name="laptop_mount")
        _part(m, P.rounded_box(0.12, 0.10, 0.012, 0.02, material="equip_steel", center=(0, 0, 0.006)), "base_plate")
        _part(m, P.cylinder(0.022, hp, 14, material="equip_steel", center=(0, 0, 0.012 + hp / 2)), "post")
        top = np.array([-0.06, side * 0.15, hp + 0.03])
        _part(m, _pipe([[0, 0, hp], [top[0] * 0.5, top[1] * 0.5, hp + 0.02], top], 0.014, "equip_steel"), "arm")
        tray = Mesh(name="tray")
        _part(tray, P.rounded_box(0.34, 0.25, 0.012, 0.02, material="equip_black", center=(0, 0, 0.006)), "tray")
        lap = Mesh(name="laptop")
        _part(lap, P.rounded_box(0.33, 0.23, 0.022, 0.012, material="equip_grey", center=(0, 0, 0.023)), "laptop_base")
        _part(lap, P.box(0.29, 0.10, 0.002, material="equip_black", center=(-0.01, 0, 0.035)), "keyboard")
        screen = Mesh(name="screen")
        _part(screen, P.box(0.008, 0.33, 0.22, material="equip_grey", center=(-0.004, 0, 0.11)), "lid")
        _part(screen, P.box(0.002, 0.30, 0.19, material="screen_dark", center=(-0.009, 0, 0.11)), "display")
        screen.transform(H.rot_y(np.radians(float(opts["screen_open_deg"]) - 90.0))).translate([0.115, 0, 0.034])
        lap.merge(screen)
        tray.merge(lap)
        tray.transform(H.rot_z(side * np.radians(float(opts["turn_deg"])))).translate(top + [0, 0, 0.007])
        m.merge(tray)
        return ComponentResult(_finish(m, ctx), [], {"height": hp})


# =============================================================== INTERIOR LIGHT BARS
def _led_modules(m: Mesh, width: float, y_face: float, z: float, n: int, red_sign: float, name: str):
    """Alternating lens modules along x on a face at y_face; `red_sign` says which x half is red."""
    edges = np.linspace(-width / 2, width / 2, n + 1)
    for k in range(n):
        xc = 0.5 * (edges[k] + edges[k + 1])
        mat = "lens_red" if xc * red_sign > 0 else "lens_blue"
        _part(m, P.box(edges[k + 1] - edges[k] - 0.006, 0.008, 0.022, material=mat, center=(xc, y_face, z)), f"{name}_{k}")


@register
class VisorLight(CarComponent):
    """Slim light bar under the windshield header.  Local frame: x car right, y backward, z down."""

    name = "light.visor"
    accepts = (RectangleConnector,)
    default_for = ("visor_light",)
    options = {"modules": 8}
    description = "interior windshield light bar (red/blue LED modules facing forward)"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        W = conn.width
        m = Mesh(name="visor_light")
        _part(m, P.box(W, 0.035, 0.032, material="equip_black", center=(0, 0, 0.018)), "housing")
        _led_modules(m, W - 0.03, -0.0215, 0.018, int(opts["modules"]), red_sign=-1.0, name="module")
        for x in (-W / 2 + 0.05, W / 2 - 0.05):
            _part(m, P.box(0.04, 0.03, 0.004, material="equip_steel", center=(x, 0, 0.002)), "bracket")
        return ComponentResult(_finish(m, ctx))


@register
class DeckLight(CarComponent):
    """Light bar on the parcel shelf.  Local frame: x car left, y backward, z up."""

    name = "light.deck"
    accepts = (RectangleConnector,)
    default_for = ("deck_light",)
    options = {"modules": 6}
    description = "rear deck light bar behind the rear window (red/blue LED modules facing rearward)"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        W = conn.width
        m = Mesh(name="deck_light")
        for x in (-W / 2 + 0.05, W / 2 - 0.05):
            _part(m, P.box(0.04, 0.03, 0.03, material="equip_steel", center=(x, 0, 0.015)), "stand")
        _part(m, P.box(W, 0.035, 0.032, material="equip_black", center=(0, 0, 0.046)), "housing")
        _led_modules(m, W - 0.03, 0.0215, 0.046, int(opts["modules"]), red_sign=1.0, name="module")
        return ComponentResult(_finish(m, ctx))


# =============================================================== DECALS
class SurfaceGrid:
    """Bilinear surface over a (rows, cols, 3) point grid: axis 0 (u) runs along rows, axis 1 (v) along columns.

    The loft samples the skin unevenly, so positions are given as *arc-length* fractions
    and converted to raw grid parameters through the cumulative length of the middle row/column."""

    def __init__(self, grid: np.ndarray, outward: np.ndarray):
        self.g = np.asarray(grid, dtype=float)
        self.rows, self.cols = self.g.shape[:2]
        self.out = np.asarray(outward, dtype=float)
        mid_u, mid_v = self.g[:, self.cols // 2], self.g[self.rows // 2, :]
        self.arc = []
        self.length = []
        for line in (mid_u, mid_v):
            d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
            self.length.append(float(d[-1]))
            self.arc.append(d / max(d[-1], 1e-9))
        self.len_u, self.len_v = self.length

    def to_param(self, axis: int, frac) -> np.ndarray:
        """Raw grid parameter for an arc-length fraction along axis 0 (u) or 1 (v)."""
        n = self.rows if axis == 0 else self.cols
        return np.interp(np.clip(frac, 0.0, 1.0), self.arc[axis], np.linspace(0.0, 1.0, n))

    def point(self, u, v) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0) * (self.rows - 1)
        v = np.clip(np.asarray(v, dtype=float), 0.0, 1.0) * (self.cols - 1)
        i0 = np.clip(np.floor(u).astype(int), 0, self.rows - 2)
        j0 = np.clip(np.floor(v).astype(int), 0, self.cols - 2)
        fu, fv = (u - i0)[..., None], (v - j0)[..., None]
        return ((1 - fu) * (1 - fv) * self.g[i0, j0] + fu * (1 - fv) * self.g[i0 + 1, j0]
                + (1 - fu) * fv * self.g[i0, j0 + 1] + fu * fv * self.g[i0 + 1, j0 + 1])

    def normal(self, u, v) -> np.ndarray:
        e = 1e-3
        du = self.point(np.minimum(u + e, 1), v) - self.point(np.maximum(u - e, 0), v)
        dv = self.point(u, np.minimum(v + e, 1)) - self.point(u, np.maximum(v - e, 0))
        n = np.cross(du, dv)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        flip = (n @ self.out) < 0
        n[flip] *= -1
        return n

    def axis_for(self, world_dir) -> Tuple[int, float]:
        """(0 for u / 1 for v, sign) of the grid axis that best follows a world direction."""
        d = np.asarray(world_dir, dtype=float)
        du = self.g[-1].mean(axis=0) - self.g[0].mean(axis=0)
        dv = self.g[:, -1].mean(axis=0) - self.g[:, 0].mean(axis=0)
        su, sv = float(du @ d) / max(np.linalg.norm(du), 1e-9), float(dv @ d) / max(np.linalg.norm(dv), 1e-9)
        return (0, float(np.sign(su) or 1.0)) if abs(su) >= abs(sv) else (1, float(np.sign(sv) or 1.0))

    def slab(self, u0: float, u1: float, v0: float, v1: float, thickness: float, material: str,
             nu: int = 2, nv: int = 2, base: float = 0.0, name="slab") -> Mesh:
        """Thin solid raised off the surface between raw parameters (u0..u1, v0..v1)."""
        us, vs = np.linspace(u0, u1, nu), np.linspace(v0, v1, nv)
        U, V = np.meshgrid(us, vs, indexing="ij")
        p, n = self.point(U, V), self.normal(U, V)
        top = grid_mesh(p + n * (base + thickness), material, name)
        bottom = grid_mesh(p + n * base, material, name + "_back", flip=True)
        ring = lambda G: np.vstack([G[:, 0], G[-1, 1:], G[-2::-1, -1], G[0, -2:0:-1]])  # noqa: E731
        walls = P.loft([ring(p + n * base), ring(p + n * (base + thickness))], closed_rings=True, material=material, name=name + "_wall")
        m = top
        m.merge(bottom).merge(walls)
        if P.signed_volume(m) < 0:
            m.flip_normals()
        return m


VIEW_DIRS = {  # (text advance, text up) in world space for each way of reading a panel:
    # facing the driver's door the hood is on your left, so text on the left side runs rearwards
    "left": ((-1, 0, 0), (0, 0, 1)), "right": ((1, 0, 0), (0, 0, 1)),
    "front": ((0, 1, 0), (-1, 0, 0)), "rear": ((0, -1, 0), (1, 0, 0)),
}


def _reading(conn: PolygonConnector, orient: str) -> str:
    if orient in VIEW_DIRS:
        return orient
    if "panel_hood" in conn.tags:
        return "front"
    if conn.meta.get("side") in ("left", "right"):
        return conn.meta["side"]
    return "rear"


def _text_directions(conn: PolygonConnector, orient: str) -> Tuple[np.ndarray, np.ndarray]:
    """World-space (advance, up) for reading text on this connector."""
    if orient in VIEW_DIRS:
        return tuple(np.asarray(v, dtype=float) for v in VIEW_DIRS[orient])  # type: ignore[return-value]
    n = conn.frame.z_axis
    if conn.meta.get("grid_points") is None and abs(n[2]) < 0.5:
        # any upright surface (a plate, a fascia rectangle): read it facing the surface
        up = np.array([0.0, 0.0, 1.0])
        adv = np.cross(-n, up)
        return adv / np.linalg.norm(adv), up
    return tuple(np.asarray(v, dtype=float) for v in VIEW_DIRS[_reading(conn, orient)])  # type: ignore[return-value]


class PanelFrame:
    """Panel coordinates on a surface grid: x runs along the car (0 = rear edge, 1 = front edge);
    y runs up a side panel (0 = sill) or left-to-right across a top panel."""

    def __init__(self, S: SurfaceGrid, side_panel: bool):
        self.S = S
        self.ax_x, self.sx = S.axis_for((1.0, 0.0, 0.0))
        self.ax_y, self.sy = S.axis_for((0.0, 0.0, 1.0) if side_panel else (0.0, -1.0, 0.0))
        if self.ax_x == self.ax_y:
            self.ax_y = 1 - self.ax_x
            self.sy = 1.0
        self.len_x, self.len_y = S.length[self.ax_x], S.length[self.ax_y]

    def arc(self, x_frac: float, y_frac: float) -> List[float]:
        q = [0.0, 0.0]
        q[self.ax_x] = x_frac if self.sx > 0 else 1.0 - x_frac
        q[self.ax_y] = y_frac if self.sy > 0 else 1.0 - y_frac
        return q

    def raw(self, arc) -> np.ndarray:
        return np.array([self.S.to_param(0, arc[0]), self.S.to_param(1, arc[1])], dtype=float)


def text_mapper(conn: PolygonConnector, orient: str, fx: float, fy: float):
    """(mapper, room) for a text block centred at panel fractions (fx, fy): text-plane metres ->
    world points and normals, on the panel's surface grid when it has one, else flat in its plane.
    `room` is the panel's extent along the reading direction."""
    adv_dir, up_dir = _text_directions(conn, orient)
    grid = conn.meta.get("grid_points")
    if grid is not None:
        S = SurfaceGrid(grid, conn.frame.z_axis)
        F = PanelFrame(S, side_panel=abs(float(conn.frame.z_axis[2])) < 0.5)
        (ax_a, sa), (ax_u, su) = S.axis_for(adv_dir), S.axis_for(up_dir)
        if ax_a == ax_u:
            raise ValueError(f"{conn.name}: panel grid too skewed to place text")
        centre = np.asarray(F.arc(fx, fy), dtype=float)

        def mapper(P: np.ndarray):
            q = np.tile(centre, (len(P), 1))
            q[:, ax_a] += sa * P[:, 0] / S.length[ax_a]
            q[:, ax_u] += su * P[:, 1] / S.length[ax_u]
            u, v = S.to_param(0, q[:, 0]), S.to_param(1, q[:, 1])
            return S.point(u, v), S.normal(u, v)

        return mapper, S.length[ax_a]
    fr = conn.frame
    adv = np.array([adv_dir @ fr.x_axis, adv_dir @ fr.y_axis])
    up = np.array([up_dir @ fr.x_axis, up_dir @ fr.y_axis])
    if np.linalg.norm(adv) < 1e-6 or np.linalg.norm(up) < 1e-6:
        raise ValueError(f"{conn.name}: text direction lies out of the connector's plane; set orient")
    adv, up = adv / np.linalg.norm(adv), up / np.linalg.norm(up)
    lp = conn.local_points()
    a, b = lp @ adv, lp @ up
    origin = (a.min() + fx * (a.max() - a.min())) * adv + (b.min() + fy * (b.max() - b.min())) * up
    return planar_mapper(fr, adv, up, origin), float(a.max() - a.min())


def _colour_material(kind: str, color: str) -> str:
    return f"{kind}_{str(color).lstrip('#').lower()}"


def add_text(m: Mesh, conn: PolygonConnector, ctx: BuildContext, text: str, *, font: str = DEFAULT_FONT, size: float = 0.12,
             x: float = 0.5, y: float = 0.5, color: str = "#f4f4f0", align: str = "center", line_spacing: float = 1.15,
             letter_spacing: float = 0.0, slant_deg: float = 0.0, kerning: bool = True, orient: str = "auto",
             fit: bool = True, thickness: float = 0.0015, case: Optional[str] = None, outline: float = 0.0,
             outline_color: str = "#14213d", material: Optional[str] = None, group: Optional[str] = None) -> dict:
    """Lay `text` out on the connector and merge the letters into `m` (world space).
    The material is named by colour, so decals in different colours survive being merged into one car."""
    fnt = load_font(font)
    material = material or _colour_material("text", color)
    outline = float(outline or 0.0)
    outline_material = _colour_material("text", outline_color) if outline > 0 else None
    text = {"upper": text.upper(), "lower": text.lower()}.get(str(case or "").lower(), text)
    mapper, room = text_mapper(conn, orient, float(x), float(y))
    lay_fn = lambda s: text_layout(text, fnt, s, align, float(line_spacing), float(letter_spacing), float(slant_deg), bool(kerning))  # noqa: E731
    lay = sized_to_fit(lay_fn, float(size), room * 0.85) if fit else lay_fn(float(size))
    letters = text_layers(lay, mapper, float(thickness), material, base=0.0005, outline=outline,
                          outline_material=outline_material, name="text")
    m.materials[material] = ctx.material(material, color, shininess=0.5)
    if outline_material:
        m.materials[outline_material] = ctx.material(outline_material, outline_color, shininess=0.5)
    m.merge(letters, group_prefix=group)
    return {"text": text, "font": fnt.name, "size": lay.size, "width": lay.width, "height": lay.height,
            "glyphs": len(lay.glyphs), "outline": outline}


TEXT_KEYS = ("font", "size", "x", "y", "color", "align", "line_spacing", "letter_spacing", "slant_deg", "kerning", "orient",
             "fit", "thickness", "case", "outline", "outline_color")


@register
class TextBox(CarComponent):
    """A block of text on any polygon connector: curved onto a body panel's surface grid, or
    flat in the connector's plane (a plate, a fascia rectangle, the roof mount).

    `size` is the cap height in metres; `x`/`y` place the block centre as fractions of the
    panel (x along the car from the rear edge, y up a side panel or across a top one)."""

    name = "text.box"
    accepts = (PolygonConnector,)
    default_for = ()
    options = {"text": "TEXT", "font": DEFAULT_FONT, "size": 0.12, "color": "#f4f4f0", "x": 0.5, "y": 0.5,
               "align": "center", "line_spacing": 1.15, "letter_spacing": 0.0, "slant_deg": 0.0, "kerning": True,
               "orient": "auto", "fit": True, "thickness": 0.0015, "case": None, "outline": 0.0, "outline_color": "#14213d"}
    description = "text block with real fonts (see makecar/data/fonts), optional outline, wrapped onto the connector"

    def build_local(self, conn: PolygonConnector, opts, ctx) -> ComponentResult:
        text = str(opts["text"] or "")
        if not text.strip():
            raise ValueError(f"{conn.name}: text.box needs some text")
        m = Mesh(name="text_box")
        info = add_text(m, conn, ctx, text, **{k: opts[k] for k in TEXT_KEYS})
        m.transform(np.linalg.inv(conn.frame.matrix))
        return ComponentResult(m, [], info)


@register
class PanelDecal(CarComponent):
    """Lettering and/or a stripe on a body panel, wrapped onto the panel's surface grid.

    Positions are panel fractions: x along the car (0 rear, 1 front); y up a side panel
    (0 sill, 1 belt) or across a top panel (0 left, 1 right).  Sizes are metres."""

    name = "decal.panel"
    accepts = (PolygonConnector,)
    default_for = ()
    options = {"text": None, "font": DEFAULT_FONT, "text_height": 0.15, "text_x": 0.5, "text_y": 0.42, "text_color": "#f4f4f0",
               "align": "center", "line_spacing": 1.15, "letter_spacing": 0.0, "slant_deg": 0.0, "orient": "auto",
               "case": None, "outline": 0.0, "outline_color": "#14213d", "texts": (),
               "stripe": False, "stripe_y": 0.66, "stripe_height": 0.06, "stripe_x": (0.02, 0.98),
               "stripe_color": "#1b3a8a", "stripe_accent": "#d9b13b", "thickness": 0.0015}
    description = "lettering (real fonts, outlines, extra text blocks) and a reflective stripe, conforming to a body panel"

    def build_local(self, conn: PolygonConnector, opts, ctx) -> ComponentResult:
        grid = conn.meta.get("grid_points")
        if grid is None:
            raise ValueError(f"{conn.name}: decal.panel needs a panel connector with a surface grid")
        S = SurfaceGrid(grid, conn.frame.z_axis)
        F = PanelFrame(S, side_panel=abs(float(conn.frame.z_axis[2])) < 0.5)
        t = float(opts["thickness"])
        m = Mesh(name="decal")
        mats: Dict[str, Material] = {}
        info: Dict[str, object] = {}
        if opts["stripe"]:
            x0, x1 = (float(x) for x in opts["stripe_x"])
            hy = float(opts["stripe_height"]) / F.len_y
            yc = float(opts["stripe_y"])
            nu = max(6, int(F.len_x * abs(x1 - x0) / 0.03))
            stripe_mat = _colour_material("stripe", opts["stripe_color"])
            mats[stripe_mat] = ctx.material(stripe_mat, opts["stripe_color"], shininess=0.6)
            bands = [(stripe_mat, yc - hy / 2, yc + hy / 2, "stripe")]
            if opts["stripe_accent"]:
                accent_mat = _colour_material("stripe", opts["stripe_accent"])
                mats[accent_mat] = ctx.material(accent_mat, opts["stripe_accent"], shininess=0.6)
                bands.append((accent_mat, yc - hy * 0.95, yc - hy * 0.65, "accent"))
            for mat, y_lo, y_hi, nm in bands:
                a, b = F.raw(F.arc(x0, y_lo)), F.raw(F.arc(x1, y_hi))
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                nus = (nu, 2) if F.ax_x == 0 else (2, nu)
                _part(m, S.slab(lo[0], hi[0], lo[1], hi[1], t, mat, nu=nus[0], nv=nus[1], name=nm), nm)
            info["stripe_y"] = yc
        if opts["text"]:
            info.update(add_text(m, conn, ctx, str(opts["text"]), font=opts["font"], size=float(opts["text_height"]),
                                 x=float(opts["text_x"]), y=float(opts["text_y"]), color=opts["text_color"], align=opts["align"],
                                 line_spacing=opts["line_spacing"], letter_spacing=opts["letter_spacing"],
                                 slant_deg=opts["slant_deg"], orient=str(opts["orient"]), thickness=t, case=opts["case"],
                                 outline=opts["outline"], outline_color=opts["outline_color"]))
            info["reading"] = _reading(conn, str(opts["orient"]))
        blocks = []
        for i, block in enumerate(opts["texts"] or ()):
            kw = dict(block)
            text = str(kw.pop("text", "") or "")
            bad = set(kw) - set(TEXT_KEYS)
            if bad:
                raise ValueError(f"{conn.name}: texts[{i}] has unknown keys {sorted(bad)}; use {TEXT_KEYS}")
            if text.strip():
                kw.setdefault("thickness", t)
                blocks.append(add_text(m, conn, ctx, text, group=f"text{i}", **kw))
        if blocks:
            info["texts"] = blocks
        if m.n_faces == 0:
            raise ValueError(f"{conn.name}: decal.panel needs text and/or stripe")
        m.materials.update(mats)
        m.transform(np.linalg.inv(conn.frame.matrix))
        return ComponentResult(m, [], info)
