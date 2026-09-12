"""Exterior components: wheels, glazing, lights, grille, plates, mirrors ..."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from ..geometry.mesh import Mesh, Material
from ..geometry.frame import Frame
from ..geometry import primitives as P
from ..geometry.curves import circle_points, rounded_rect_points
from ..connectors import Connector, PointConnector, PolygonConnector, RectangleConnector, CircleConnector
from ..morph.morphable import ModifierSpec
from .base import CarComponent, MorphableComponent, ComponentResult, BuildContext, register
from .fills import fill_connector


# =============================================================== WHEEL
@dataclass
class WheelParams:
    radius: float = 0.33       # tyre outer radius
    width: float = 0.225       # tyre width
    rim_ratio: float = 0.62    # rim radius / tyre radius
    dish: float = 0.04         # how deep the spokes sit inside the rim
    sidewall_bulge: float = 0.012


@register
class Wheel(MorphableComponent):
    """Tyre + rim + spokes, fitted to a CircleConnector (hub).  Local frame:
    +Z is the axle direction pointing outboard, the hub is at the origin."""

    name = "wheel.alloy"
    accepts = (CircleConnector,)
    default_for = ("wheel",)
    params_cls = WheelParams
    options = {"spokes": 5, "spoke_width": 0.55, "hubcap": True, "tread_grooves": 3}
    description = "tyre with alloy rim; radius/width/rim size fitted from the hub connector"
    modifier_specs = [
        ModifierSpec("radius", 0.10, 0.14, "size", "tyre radius"),
        ModifierSpec("width", 0.08, 0.12, "size", "tyre width"),
        ModifierSpec("rim_ratio", 0.15, 0.20, "style", "low-profile <-> balloon tyre"),
        ModifierSpec("dish", 0.03, 0.05, "style", "spoke dish depth"),
    ]

    N = 36  # tyre segments

    def generate(self, p: WheelParams) -> Mesh:
        n = self.N
        R, w, rr = p.radius, p.width, p.rim_ratio * p.radius
        b = p.sidewall_bulge
        # tyre as a revolved profile (r, z) about +Z; z from -w/2 (inboard) to +w/2 (outboard)
        # R is the outer (tread) radius; the sidewall bulge is kept inside R so the tyre touches the ground
        prof = np.array([
            [rr + 0.005, -w / 2 + 0.015],
            [rr + 0.02, -w / 2 + 0.008],
            [R - 0.05, -w / 2 + 0.004],
            [R - 0.02, -w / 2 + 0.008],
            [R - 0.006, -w / 2 + 0.03],
            [R, -w * 0.25],
            [R, w * 0.25],
            [R - 0.006, w / 2 - 0.03],
            [R - 0.02, w / 2 - 0.008],
            [R - 0.05, w / 2 - 0.004],
            [rr + 0.02, w / 2 - 0.008],
            [rr + 0.005, w / 2 - 0.015],
        ])
        tyre = P.revolve(prof, n, material="tyre", name="tyre")
        # rim barrel (inner cylinder) + lip
        barrel = P.tube(rr + 0.006, rr - 0.02, w - 0.02, n, material="rim", name="barrel")
        lip = P.tube(rr + 0.012, rr - 0.005, 0.02, n, material="rim", center=(0, 0, w / 2 - 0.02), name="lip")
        # spoke face: a dished disc built as a revolve (fixed topology; spoke cut-outs are done by material)
        zf = w / 2 - 0.02 - p.dish
        face_prof = np.array([[0.0, zf + 0.01], [0.05, zf + 0.012], [0.09, zf + 0.0], [rr * 0.55, zf - 0.005],
                              [rr - 0.02, zf + 0.004], [rr - 0.02, zf - 0.02], [0.0, zf - 0.02]])
        face = P.revolve(face_prof, n, material="rim", name="spokes")
        hub = P.cylinder(0.055, 0.03, 16, material="rim_dark", center=(0, 0, zf + 0.012), name="hub")
        m = tyre.merge(barrel).merge(lip).merge(face).merge(hub)
        m.add_group("face_ring", list(range(tyre.n_vertices + barrel.n_vertices + lip.n_vertices, m.n_vertices - hub.n_vertices)))
        return m

    def fit(self, conn: CircleConnector, opts, ctx) -> Dict[str, float]:
        v = {"radius": self.value_for("radius", conn.radius)}
        tw = conn.meta.get("tire_width")
        if tw:
            v["width"] = self.value_for("width", tw)
        # bigger wheels get lower profile tyres
        v["rim_ratio"] = self.value_for("rim_ratio", float(opts.get("rim_ratio", 0.62 + (conn.radius - 0.33) * 0.6)))
        return v

    def materials(self, ctx, opts):
        return {
            "tyre": ctx.material("tyre", "#1a1a1c", shininess=0.08),
            "rim": ctx.material("rim", opts.get("rim_color", ctx.palette.rim), shininess=0.7, metallic=0.8),
            "rim_dark": ctx.material("rim_dark", "#3a3c40", shininess=0.4),
        }

    def build_local(self, conn, opts, ctx) -> ComponentResult:
        res = super().build_local(conn, opts, ctx)
        m = res.mesh
        # spoke cut-outs: paint the faces of the dished disc between the spokes as dark
        n_spokes = int(opts.get("spokes", 5))
        sw = float(opts.get("spoke_width", 0.55))
        spokes_zone = [i for i, f in enumerate(m.faces) if m.face_materials[i] == "rim" and m.name and True]
        cent = m.face_centroids()
        rr = conn.radius * (0.62 + (conn.radius - 0.33) * 0.6)
        for i in range(m.n_faces):
            if m.face_materials[i] != "rim":
                continue
            x, y, z = cent[i]
            r = np.hypot(x, y)
            if r < 0.07 or r > rr * 0.9 or z < 0:
                continue
            ang = (np.arctan2(y, x) * n_spokes / (2 * np.pi)) % 1.0
            if abs(ang - 0.5) < 0.5 * (1 - sw) * (0.55 + 0.45 * (r / rr)):
                m.face_materials[i] = "rim_dark"
        return res


# =============================================================== GLASS
@register
class Glass(CarComponent):
    """Glazing that fills any polygon aperture (windshield, side, rear).  Adds a
    dark ceramic frit / frame border so pillars read correctly."""

    name = "glass.tinted"
    accepts = (PolygonConnector,)
    default_for = ("glass",)
    options = {"tint": None, "alpha": 0.42, "border": 0.045, "inset": 0.004}
    description = "curved glazing following the aperture (Coons patch) with a black frame band"

    def build_local(self, conn: PolygonConnector, opts, ctx) -> ComponentResult:
        # build directly in world space then pull back to local (fills are world-space by nature)
        color = opts.get("tint") or ctx.palette.glass
        mats = {
            "glass": Material("glass", Material.parse_color(color), float(opts["alpha"]), 0.95),
            "glass_frame": ctx.material("glass_frame", "#0c0d0f", shininess=0.35),
        }
        m = fill_connector(conn, "glass", offset=-float(opts["inset"]), border=float(opts["border"]),
                           border_material="glass_frame", thickness=0.0, name="glass")
        m.materials.update(mats)
        m.transform(np.linalg.inv(conn.frame.matrix))  # to local; build() maps back to world
        return ComponentResult(m, [], {"area": conn.area})


# =============================================================== LIGHTS
@register
class Headlight(CarComponent):
    name = "light.headlight"
    accepts = (PolygonConnector,)
    default_for = ("headlight",)
    options = {"lens_alpha": 0.55, "projector": True}
    description = "wrap-around headlamp: clear lens over a chrome bezel with a projector unit"

    def build_local(self, conn: PolygonConnector, opts, ctx) -> ComponentResult:
        mats = {
            "lens": Material("lens", (0.82, 0.88, 0.92), float(opts["lens_alpha"]), 0.95),
            "bezel": ctx.material("bezel", "#d8dde2", shininess=0.8, metallic=0.9),
            "lamp_dark": ctx.material("lamp_dark", "#1b1c1f", shininess=0.4),
            "lamp_led": Material("lamp_led", (0.95, 0.97, 1.0), 1.0, 0.5, emissive=0.8),
        }
        housing = fill_connector(conn, "lamp_dark", offset=-0.05, name="housing")
        bezel = fill_connector(conn, "bezel", offset=-0.03, border=0.03, border_material="lamp_dark", name="bezel")
        lens = fill_connector(conn, "lens", offset=-0.004, thickness=0.0, name="lens")
        m = housing.merge(bezel).merge(lens)
        # projector: a short cylinder near the outer/forward end of the lamp
        lp = conn.local_points()
        c = conn.centroid
        i = int(np.argmax(lp[:, 0]))  # major direction ~ car forward
        p_world = conn.points[i] * 0.45 + c * 0.55
        proj = P.cylinder(0.055, 0.03, 20, material="lamp_dark", name="projector")
        proj_face = P.cylinder(0.04, 0.012, 20, material="lamp_led", center=(0, 0, 0.02), name="projector_led")
        proj.merge(proj_face)
        proj.apply_frame(Frame.from_normal(p_world - conn.frame.z_axis * 0.035, conn.frame.z_axis, conn.frame.x_axis))
        # DRL strip along the top edge
        top = lp[:, 1].max()
        strip_pts = conn.points[lp[:, 1] > top - 0.02]
        m.merge(proj)
        m.materials.update(mats)
        m.transform(np.linalg.inv(conn.frame.matrix))
        return ComponentResult(m, [], {})


@register
class Taillight(CarComponent):
    name = "light.taillight"
    accepts = (PolygonConnector,)
    default_for = ("taillight",)
    options = {"lens_alpha": 0.7}
    description = "red lens tail lamp with an inner reflector and reversing-light band"

    def build_local(self, conn: PolygonConnector, opts, ctx) -> ComponentResult:
        red = Material.parse_color(ctx.palette.taillight)
        mats = {
            "tail_lens": Material("tail_lens", red, float(opts["lens_alpha"]), 0.95, emissive=0.25),
            "tail_reflector": Material("tail_reflector", (0.9, 0.35, 0.3), 1.0, 0.6),
            "tail_reverse": Material("tail_reverse", (0.9, 0.92, 0.95), 1.0, 0.6),
            "lamp_dark": ctx.material("lamp_dark", "#1b1c1f", shininess=0.4),
        }
        housing = fill_connector(conn, "lamp_dark", offset=-0.04, name="housing")
        inner = fill_connector(conn, "tail_reflector", offset=-0.02, border=0.03, border_material="lamp_dark", name="inner")
        # reversing band on the lower third
        lp = conn.local_points()
        lo, hi = lp[:, 1].min(), lp[:, 1].max()
        cent = inner.face_centroids()
        cl = conn.frame.to_local(cent)[:, :2]
        for i in range(inner.n_faces):
            if inner.face_materials[i] == "tail_reflector" and cl[i, 1] < lo + 0.3 * (hi - lo):
                inner.face_materials[i] = "tail_reverse"
        lens = fill_connector(conn, "tail_lens", offset=-0.004, thickness=0.0, name="lens")
        m = housing.merge(inner).merge(lens)
        m.materials.update(mats)
        m.transform(np.linalg.inv(conn.frame.matrix))
        return ComponentResult(m, [], {})


# =============================================================== GRILLE / INTAKE / PLATE / BADGE
@register
class Grille(CarComponent):
    name = "grille.slats"
    accepts = (RectangleConnector,)
    default_for = ("grille",)
    options = {"slats": 5, "frame": True, "mesh": False}
    description = "horizontal-slat radiator grille in a chrome frame (fits any rectangle)"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        w, h = conn.width, conn.height
        mats = {"grille_dark": ctx.material("grille_dark", "#0e0f11", shininess=0.2),
                "chrome": ctx.material("chrome", ctx.palette.chrome, shininess=0.85, metallic=0.9)}
        m = P.rounded_box(w, h, 0.02, min(0.04, h / 3), material="grille_dark", center=(0, 0, 0.0), name="grille_back")
        if opts.get("frame", True):
            fr = P.rounded_box(w + 0.03, h + 0.03, 0.016, min(0.05, h / 2.5), material="chrome", center=(0, 0, 0.006), name="frame")
            m.merge(fr)
            back = P.rounded_box(w - 0.0, h - 0.0, 0.014, min(0.04, h / 3), material="grille_dark", center=(0, 0, 0.012), name="grille_inner")
            m.merge(back)
        n = int(opts.get("slats", 5))
        if opts.get("mesh"):
            n = max(n, 8)
        for k in range(n):
            y = -h / 2 + (k + 0.5) * h / n
            slat = P.box(w - 0.02, h / n * 0.45, 0.02, material="chrome", center=(0, y, 0.02), name="slat")
            m.merge(slat)
        m.materials.update(mats)
        return ComponentResult(m)


@register
class Intake(CarComponent):
    name = "grille.intake"
    accepts = (RectangleConnector,)
    default_for = ("intake",)
    priority = 1
    options = {"mesh": True}
    description = "lower bumper air intake with a honeycomb/mesh insert"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        w, h = conn.width, conn.height
        mats = {"grille_dark": ctx.material("grille_dark", "#0e0f11", shininess=0.2),
                "mesh_dark": ctx.material("mesh_dark", "#26282c", shininess=0.35)}
        m = P.rounded_box(w, h, 0.03, min(0.05, h / 2.5), material="grille_dark", center=(0, 0, -0.005), name="intake")
        # mesh bars
        n = max(3, int(w / 0.08))
        for k in range(n):
            x = -w / 2 + (k + 0.5) * w / n
            m.merge(P.box(0.012, h - 0.02, 0.02, material="mesh_dark", center=(x, 0, 0.012), name="bar"))
        for k in range(max(2, int(h / 0.05))):
            y = -h / 2 + (k + 0.5) * h / max(2, int(h / 0.05))
            m.merge(P.box(w - 0.02, 0.01, 0.02, material="mesh_dark", center=(0, y, 0.012), name="bar"))
        m.materials.update(mats)
        return ComponentResult(m)


@register
class LicensePlate(CarComponent):
    name = "plate.standard"
    accepts = (RectangleConnector,)
    default_for = ("plate",)
    options = {"region": "eu", "color": "#f2f2ee", "text_color": "#1a1a1a"}
    description = "license plate (EU 520x110 or US 300x150) with a dark text band"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        w, h = (0.52, 0.11) if opts.get("region", "eu") == "eu" else (0.305, 0.152)
        mats = {"plate": ctx.material("plate", opts["color"], shininess=0.4),
                "plate_text": ctx.material("plate_text", opts["text_color"], shininess=0.2),
                "plate_blue": ctx.material("plate_blue", "#1c3d8f", shininess=0.3)}
        m = P.box(w, h, 0.006, material="plate", center=(0, 0, 0.003), name="plate")
        # crude "characters": 7 dark blocks
        n = 7
        cw = w * 0.09
        for k in range(n):
            x = -w * 0.36 + k * (w * 0.78) / (n - 1)
            m.merge(P.box(cw, h * 0.55, 0.002, material="plate_text", center=(x, 0, 0.007), name="char"))
        if opts.get("region", "eu") == "eu":
            m.merge(P.box(w * 0.08, h - 0.01, 0.002, material="plate_blue", center=(-w / 2 + w * 0.05, 0, 0.007), name="euband"))
        m.materials.update(mats)
        return ComponentResult(m)


@register
class Badge(CarComponent):
    name = "badge.roundel"
    accepts = (PointConnector,)
    default_for = ("badge",)
    options = {"radius": 0.045}
    description = "chrome roundel emblem"

    def build_local(self, conn: PointConnector, opts, ctx) -> ComponentResult:
        r = float(opts["radius"])
        mats = {"chrome": ctx.material("chrome", ctx.palette.chrome, shininess=0.85, metallic=0.9),
                "badge_dark": ctx.material("badge_dark", "#1d1f24", shininess=0.5)}
        m = P.cylinder(r, 0.008, 24, material="chrome", center=(0, 0, 0.004), name="badge")
        m.merge(P.cylinder(r * 0.7, 0.004, 24, material="badge_dark", center=(0, 0, 0.01), name="badge_inner"))
        m.merge(P.cylinder(r * 0.28, 0.003, 16, material="chrome", center=(0, 0, 0.0135), name="badge_dot"))
        m.materials.update(mats)
        return ComponentResult(m)


# =============================================================== EXHAUST / FUEL / ANTENNA / HANDLES / MIRRORS / RAILS
@register
class ExhaustTip(CarComponent):
    name = "exhaust.tip"
    accepts = (CircleConnector,)
    default_for = ("exhaust",)
    options = {"dual": False, "length": 0.08}
    description = "chrome exhaust tip (single or twin) protruding from the rear valance"

    def build_local(self, conn: CircleConnector, opts, ctx) -> ComponentResult:
        r = conn.radius
        L = float(opts["length"])
        mats = {"chrome": ctx.material("chrome", ctx.palette.chrome, shininess=0.85, metallic=0.9),
                "exhaust_dark": ctx.material("exhaust_dark", "#151517", shininess=0.2)}
        def tip(cy):
            t = P.tube(r, r - 0.006, L, 20, material="chrome", center=(0, cy, L / 2 - 0.03), name="tip")
            inner = P.cylinder(r - 0.006, 0.004, 20, material="exhaust_dark", center=(0, cy, L - 0.03 - 0.01), name="tip_inner")
            return t.merge(inner)
        m = tip(0.0)
        if opts.get("dual"):
            m = tip(-r * 1.2).merge(tip(r * 1.2))
        m.materials.update(mats)
        return ComponentResult(m)


@register
class FuelCap(CarComponent):
    name = "fuel.cap"
    accepts = (CircleConnector,)
    default_for = ("fuel_cap",)
    options = {}
    description = "fuel filler door flush with the rear quarter"

    def build_local(self, conn: CircleConnector, opts, ctx) -> ComponentResult:
        r = conn.radius
        mats = {"paint": ctx.material("paint", ctx.palette.paint, shininess=0.85),
                "trim_dark": ctx.material("trim_dark", "#101113", shininess=0.2)}
        m = P.cylinder(r + 0.006, 0.004, 24, material="trim_dark", center=(0, 0, 0.002), name="cap_gap")
        m.merge(P.cylinder(r, 0.006, 24, material="paint", center=(0, 0, 0.006), name="cap"))
        m.materials.update(mats)
        return ComponentResult(m)


@register
class SharkFinAntenna(CarComponent):
    name = "antenna.sharkfin"
    accepts = (PointConnector,)
    default_for = ("antenna",)
    options = {"length": 0.18, "height": 0.055}
    description = "shark-fin roof antenna"

    def build_local(self, conn: PointConnector, opts, ctx) -> ComponentResult:
        L, H = float(opts["length"]), float(opts["height"])
        mats = {"paint": ctx.material("paint", ctx.palette.paint, shininess=0.85)}
        # fin profile in the XZ plane, extruded in Y (thin)
        prof = np.array([[-L * 0.55, 0], [L * 0.45, 0], [L * 0.45, H * 0.15], [L * 0.1, H], [-L * 0.25, H * 0.75], [-L * 0.55, 0.01]])
        base_w = 0.06
        rings = []
        for y, s in ((-base_w / 2, 0.85), (-base_w * 0.3, 1.0), (base_w * 0.3, 1.0), (base_w / 2, 0.85)):
            pts = np.column_stack([prof[:, 0], np.full(len(prof), y), prof[:, 1] * s])
            rings.append(pts)
        m = P.loft(rings, closed_rings=True, cap_start=True, cap_end=True, material="paint", name="fin")
        m.materials.update(mats)
        return ComponentResult(m)


@register
class DoorHandle(CarComponent):
    name = "handle.pull"
    accepts = (PointConnector,)
    default_for = ("handle",)
    options = {"length": 0.17, "chrome": False}
    description = "flush pull-type door handle"

    def build_local(self, conn: PointConnector, opts, ctx) -> ComponentResult:
        L = float(opts["length"])
        mat = "chrome" if opts.get("chrome") else "paint"
        mats = {"paint": ctx.material("paint", ctx.palette.paint, shininess=0.85),
                "chrome": ctx.material("chrome", ctx.palette.chrome, shininess=0.85, metallic=0.9),
                "trim_dark": ctx.material("trim_dark", "#101113", shininess=0.2)}
        m = P.rounded_box(L + 0.02, 0.045, 0.004, 0.015, material="trim_dark", center=(0, 0, 0.002), name="recess")
        m.merge(P.rounded_box(L, 0.028, 0.018, 0.012, material=mat, center=(0.0, 0.004, 0.014), name="handle"))
        m.materials.update(mats)
        return ComponentResult(m)


@register
class SideMirror(CarComponent):
    name = "mirror.side"
    accepts = (PointConnector,)
    default_for = ("mirror",)
    options = {"housing_color": None}
    description = "door mirror on a short stalk, housing in body colour, mirror glass facing rearwards"

    def build_local(self, conn: PointConnector, opts, ctx) -> ComponentResult:
        # local: +Z outboard, +X car-forward, +Y up (right-handed with z out, x fwd -> y = z × x)
        col = opts.get("housing_color") or ctx.palette.paint
        mats = {"mirror_housing": ctx.material("mirror_housing", col, shininess=0.85),
                "trim_dark": ctx.material("trim_dark", "#101113", shininess=0.2),
                "mirror_glass": Material("mirror_glass", (0.75, 0.8, 0.85), 1.0, 0.95)}
        stalk = P.box(0.05, 0.03, 0.06, material="trim_dark", center=(0.0, -0.01, 0.03), name="stalk")
        # housing: rounded box, 0.10 deep (x, fore-aft), 0.11 tall (y), 0.13 outboard (z)
        housing = P.rounded_box(0.10, 0.11, 0.13, 0.03, material="mirror_housing", center=(0.02, 0.02, 0.05 + 0.065), name="housing")
        glass = P.box(0.004, 0.09, 0.11, material="mirror_glass", center=(-0.031, 0.02, 0.115), name="mglass")
        m = stalk.merge(housing).merge(glass)
        m.materials.update(mats)
        return ComponentResult(m)


@register
class RoofRails(CarComponent):
    name = "roof.rails"
    accepts = (RectangleConnector,)
    default_for = ()   # opt-in via config (wagons / SUVs)
    options = {"height": 0.05}
    description = "longitudinal roof rail following the roof edge"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        path = conn.meta.get("path")
        mats = {"rail": ctx.material("rail", "#2a2c30", shininess=0.5, metallic=0.5)}
        if path is None:
            m = P.box(conn.width, 0.04, 0.05, material="rail", center=(0, 0, 0.03))
        else:
            pts = np.asarray(path, dtype=float).copy()
            pts = conn.frame.to_local(pts)
            pts[:, 2] += float(opts["height"])
            prof = rounded_rect_points(0.035, 0.05, 0.012, 3)
            m = P.sweep_profile(pts, prof, material="rail", up_hint=(0, 0, 1))
            for k in (0, len(pts) - 1):
                foot = P.box(0.08, 0.05, float(opts["height"]), material="rail", center=(pts[k, 0], pts[k, 1], pts[k, 2] - float(opts["height"]) / 2 + 0.005))
                m.merge(foot)
        m.materials.update(mats)
        return ComponentResult(m)
