"""Interior components: seats, steering wheel, dashboard (+ cluster, screen,
vents, glovebox, HVAC), pedals, centre console (+ shifter, cupholders), carpet
floor, bulkheads, shelves / cargo floors, headliner (+ dome light, grab
handles), door cards (+ speakers) and the rear-view mirror.

Local frame conventions follow `base.py`: +Z is the connector normal (parts
grow along +Z), +X the connector's major direction.  For floor-mounted parts
(seats, console, pedals, floor) +Z is up and +X car-forward; for the dashboard
+Z points rearward into the cabin and +Y is up; for the headliner +Z points
*down* into the cabin.

Depth budget of the dashboard: the body hands us the dash outline (a polygon
in a vertical plane just behind the windshield base) and a `depth` hint that
is the design envelope between that plane and the steering wheel.  The brow
(top pad) reaches 42% of it, the fascia 30%, the knee bolster 23% and the
centre stack 46%, so the wheel and the driver's knees always stay clear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from ..geometry.mesh import Mesh, Material
from ..geometry.frame import Frame
from ..geometry import primitives as P
from ..geometry.curves import rounded_rect_points
from ..connectors import Connector, PointConnector, PolygonConnector, RectangleConnector, CircleConnector
from ..morph.morphable import ModifierSpec
from .base import CarComponent, MorphableComponent, ComponentResult, BuildContext, register
from . import interior_helpers as H


# =============================================================== MATERIALS
def _mats(ctx: BuildContext) -> Dict[str, Material]:
    """Interior material set (names prefixed to avoid clashes with exterior parts)."""
    pal = ctx.palette
    return {
        "int_stitch": ctx.material("int_stitch", "#807466", shininess=0.08),
        "int_belt": ctx.material("int_belt", "#292c30", shininess=0.04),
        "int_plastic": ctx.material("int_plastic", pal.interior, shininess=0.2),
        "int_soft": ctx.material("int_soft", pal.interior, shininess=0.12),
        "int_accent": ctx.material("int_accent", pal.interior_accent, shininess=0.25),
        "int_leather": ctx.material("int_leather", pal.seat, shininess=0.35),
        "int_carpet": ctx.material("int_carpet", pal.carpet, shininess=0.04),
        "int_chrome": ctx.material("int_chrome", pal.chrome, shininess=0.85, metallic=0.9),
        "int_trim": ctx.material("int_trim", pal.trim, shininess=0.6),
        "int_wood": ctx.material("int_wood", pal.wood, shininess=0.6),
        "int_fabric": ctx.material("int_fabric", "#cbc6be", shininess=0.05),
        "int_rubber": ctx.material("int_rubber", "#141414", shininess=0.08),
        "int_metal": ctx.material("int_metal", "#4a4d52", shininess=0.5, metallic=0.6),
        "int_gauge": ctx.material("int_gauge", "#0d0e10", shininess=0.6),
        "int_grille": ctx.material("int_grille", "#202225", shininess=0.15),
        "int_screen": Material("int_screen", (0.05, 0.07, 0.10), 1.0, 0.9, emissive=0.15),
        "int_ui": Material("int_ui", (0.23, 0.43, 0.66), 1.0, 0.7, emissive=0.35),
        "int_needle": Material("int_needle", (0.9, 0.25, 0.2), 1.0, 0.4, emissive=0.4),
        "int_lens": Material("int_lens", (0.95, 0.94, 0.88), 0.92, 0.7, emissive=0.5),
        "int_mirror": Material("int_mirror", (0.72, 0.78, 0.84), 1.0, 0.95),
    }


def _finish(m: Mesh, ctx: BuildContext, extra: Dict[str, Material] | None = None) -> Mesh:
    """Attach the interior materials actually used by `m`."""
    mats = _mats(ctx)
    if extra:
        mats.update(extra)
    used = set(m.face_materials)
    m.materials.update({k: v for k, v in mats.items() if k in used})
    return m


# =============================================================== SEATS
def _cushion_section(w: float, t: float, b: float, drop: float = 0.0, lift: float = 0.0) -> np.ndarray:
    """Closed (y, z) outline of a seat-cushion cross-section (18 points):
    flat underside at `lift`, top at `t - drop`, side bolsters `b` higher."""
    hw = w / 2
    zt = t - drop
    return np.array([
        (-hw + 0.03, lift), (0.0, lift), (hw - 0.03, lift),
        (hw, lift + 0.3 * (zt - lift)), (hw, zt * 0.72),
        (hw - 0.025, zt + b), (hw - 0.09, zt + 0.55 * b),
        (w * 0.22 + 0.003, zt - 0.002), (w * 0.22, zt - 0.006), (w * 0.22 - 0.003, zt - 0.002),
        (0.0, zt - 0.012),
        (-w * 0.22 + 0.003, zt - 0.002), (-w * 0.22, zt - 0.006), (-w * 0.22 - 0.003, zt - 0.002),
        (-(hw - 0.09), zt + 0.55 * b), (-(hw - 0.025), zt + b),
        (-hw, zt * 0.72), (-hw, lift + 0.3 * (zt - lift)),
    ])


def _backrest_section(w: float, front: float, back: float, b: float) -> np.ndarray:
    """Closed (y, t) outline of a backrest cross-section (18 points); t is the
    forward direction (towards the occupant)."""
    hw = w / 2
    f = front
    return np.array([
        (-hw + 0.03, -back), (0.0, -back - 0.005), (hw - 0.03, -back),
        (hw, -back * 0.4), (hw, f * 0.35),
        (hw - 0.025, f + b), (hw - 0.09, f + 0.5 * b),
        (w * 0.22 + 0.003, f - 0.003), (w * 0.22, f - 0.007), (w * 0.22 - 0.003, f - 0.003),
        (0.0, f - 0.012),
        (-w * 0.22 + 0.003, f - 0.003), (-w * 0.22, f - 0.007), (-w * 0.22 - 0.003, f - 0.003),
        (-(hw - 0.09), f + 0.5 * b), (-(hw - 0.025), f + b),
        (-hw, f * 0.35), (-hw, -back * 0.4),
    ])


def _headrest(centre: np.ndarray, u: np.ndarray, t: np.ndarray, width: float, height: float, gap: float,
              material="int_leather") -> Mesh:
    """Headrest slab + two posts.  `centre` is the backrest top point, `u` the
    backrest axis (unit, pointing up along the reclined back), `t` forward."""
    ey = np.array([0.0, 1.0, 0.0])
    R = np.eye(4)
    R[:3, :3] = np.column_stack([t, ey, u])
    slab = P.rounded_box(0.085, width, height, 0.03, material=material, name="headrest")
    slab.transform(R).translate(centre + u * (gap + height / 2) - t * 0.005)
    H.named(slab, "headrest")
    for y in (-0.055, 0.055):
        post = P.cylinder(0.006, gap + 0.05, 8, material="int_metal", name="post")
        post.transform(R).translate(centre + u * (gap / 2 - 0.01) + ey * y)
        slab.merge(H.named(post, "post_left" if y > 0 else "post_right"))
    return slab


@dataclass
class SeatParams:
    cushion_width: float = 0.50
    cushion_depth: float = 0.50
    cushion_height: float = 0.22      # cushion top above the floor (H-point - 0.05)
    cushion_thickness: float = 0.11
    backrest_height: float = 0.62
    recline_deg: float = 22.0
    bolster: float = 0.035
    headrest_height: float = 0.23
    footprint_depth: float = 0.55     # connector depth; cushion sits at its rear


@register
class BucketSeat(MorphableComponent):
    """Front bucket seat on a riser and rails: cushion with side bolsters at
    the rear of the footprint, reclined backrest rising from its rear edge,
    headrest on posts.  Fitted from the footprint rectangle and the H-point /
    headroom hints."""

    name = "seat.bucket"
    accepts = (RectangleConnector,)
    default_for = ("seat_front",)
    params_cls = SeatParams
    options = {"recline_deg": 22.0, "backrest_height": 0.62, "bolster": 0.035, "headrest": True}
    description = "bucket seat (cushion + bolsters, reclined backrest, headrest) fitted to the seat footprint and H-point"
    modifier_specs = [
        ModifierSpec("cushion_width", 0.10, 0.10, "size", "cushion width"),
        ModifierSpec("cushion_depth", 0.10, 0.10, "size", "cushion depth"),
        ModifierSpec("cushion_height", 0.12, 0.16, "fit", "cushion top height (H-point)"),
        ModifierSpec("cushion_thickness", 0.03, 0.04, "style", "cushion thickness"),
        ModifierSpec("backrest_height", 0.15, 0.12, "size", "backrest height"),
        ModifierSpec("recline_deg", 10.0, 12.0, "pose", "backrest recline"),
        ModifierSpec("bolster", 0.03, 0.04, "style", "side bolster size"),
        ModifierSpec("footprint_depth", 0.12, 0.15, "fit", "footprint depth (cushion placed at its rear)"),
        ModifierSpec("headrest_height", 0.04, 0.05, "size", "headrest height"),
    ]

    def macros(self):
        return {"sport": SeatParams(bolster=0.07, cushion_thickness=0.09, backrest_height=0.68, recline_deg=20.0)}

    def generate(self, p: SeatParams) -> Mesh:
        cw, cd, ch, ct = p.cushion_width, p.cushion_depth, p.cushion_height, p.cushion_thickness
        x0 = -p.footprint_depth / 2 + 0.02          # cushion rear edge
        zb = ch - ct                                # cushion underside
        ey, ez = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        # ---- cushion: sections along x (rear -> front), nose rounded off
        secs, origins = [], []
        for f, scale, drop, lift in ((0.0, 0.94, 0.012, 0.0), (0.05, 1.0, 0.0, 0.0), (0.18, 1.0, 0.0, 0.0),
                                     (0.45, 1.0, 0.0, 0.0), (0.72, 1.0, 0.0, 0.0), (0.9, 1.0, 0.004, 0.0),
                                     (0.97, 0.99, 0.025, 0.006), (1.0, 0.96, 0.07, 0.04)):
            secs.append(_cushion_section(cw * scale, ct, p.bolster, drop, lift))
            origins.append(np.array([x0 + f * cd, 0.0, zb]))
        cushion = H.orient_outward(H.section_loft(secs, origins, ey, ez, "int_leather", name="cushion"))
        # ---- backrest: sections along the reclined axis
        r = np.radians(p.recline_deg)
        u = np.array([-np.sin(r), 0.0, np.cos(r)])
        t = np.array([np.cos(r), 0.0, np.sin(r)])
        hinge = np.array([x0 + 0.07, 0.0, ch - 0.035])
        bh = p.backrest_height
        secs, origins = [], []
        for f, wf, ff, bf in ((0.0, 1.0, 0.9, 1.0), (0.05, 1.0, 1.0, 1.0), (0.22, 1.0, 1.0, 1.0), (0.5, 0.97, 1.0, 1.0),
                              (0.72, 0.93, 1.0, 1.0), (0.88, 0.86, 0.95, 0.95), (0.96, 0.78, 0.7, 0.8), (1.0, 0.66, 0.25, 0.3)):
            secs.append(_backrest_section(cw * wf, 0.05 * ff, 0.055 * bf, p.bolster * 1.3 * ff))
            origins.append(hinge + u * (f * bh))
        back = H.orient_outward(H.section_loft(secs, origins, ey, t, "int_leather", name="backrest"))
        # ---- headrest, riser, rails
        top = hinge + u * bh
        head = _headrest(top, ez, np.array([1.0, 0.0, 0.0]), 0.24, p.headrest_height, 0.10)
        riser = P.box(cd * 0.55, cw * 0.55, max(zb - 0.02, 0.02), material="int_plastic",
                      center=(x0 + cd * 0.5, 0, max(zb - 0.02, 0.02) / 2 + 0.01), name="riser")
        # The two 6 mm-wide V grooves are part of the upholstery surface, not
        # floating polylines.  Each loft ring keeps identical point ordering.
        for part in (cushion, back):
            stride = 18
            for ring in range(7):
                for j in (7, 8, 11, 12):
                    part.face_materials[ring * stride + j] = "int_stitch"
                for j in (9, 10):
                    part.face_materials[ring * stride + j] = "int_accent"
        m = H.named(cushion, "cushion").merge(H.named(back, "backrest")).merge(head).merge(riser)
        for i, y in enumerate((-(cw / 2 - 0.08), cw / 2 - 0.08)):
            rail = P.box(p.footprint_depth - 0.06, 0.035, 0.03, material="int_metal", center=(0, y, 0.015), name="rail")
            m.merge(H.named(rail, f"rail_{i}"))
        return m

    def fit(self, conn: RectangleConnector, opts, ctx) -> Dict[str, float]:
        hp = float(conn.meta.get("h_point_height", 0.27))
        ch = hp - 0.05
        # Do not shorten a production backrest to compensate for a short
        # headrest.  Keep the torso support at adult scale in low-roof styles.
        bh = float(np.clip(opts["backrest_height"], 0.60, 0.65))
        return {
            "cushion_width": self.value_for("cushion_width", np.clip(conn.height - 0.02, 0.48, 0.52)),
            "cushion_depth": self.value_for("cushion_depth", 0.50),
            "footprint_depth": self.value_for("footprint_depth", conn.width),
            "cushion_height": self.value_for("cushion_height", ch),
            "backrest_height": self.value_for("backrest_height", bh),
            "recline_deg": self.value_for("recline_deg", float(opts["recline_deg"])),
            "bolster": self.value_for("bolster", float(opts["bolster"])),
        }

    def materials(self, ctx, opts):
        return _mats(ctx)

    def build_local(self, conn, opts, ctx) -> ComponentResult:
        res = super().build_local(conn, opts, ctx)
        p = H.params_from_values(self, res.info["modifier_values"])
        x0 = -p.footprint_depth / 2 + 0.02
        hp = np.array([x0 + 0.095, 0.0, p.cushion_height + 0.05])
        res.info["h_point_local"] = hp.tolist()
        res.info["backrest_length"] = p.backrest_height
        # The footprint is a floor mount, not the hip centre: the H-point is
        # 95 mm ahead of the cushion's rear edge, over the back/cushion junction.
        side = 1.0 if conn.meta.get("side", "left") == "left" else -1.0
        buckle = H.bucket_buckle(p, side)
        res.mesh.merge(H.ribbon([buckle - [0.05, 0, 0.15], buckle], 0.018, 0.009, "int_belt", "buckle_stalk"))
        res.mesh.merge(H.named(P.rounded_box(0.055, 0.032, 0.045, 0.008, material="int_plastic", center=buckle), "buckle"))
        res.mesh.merge(P.box(0.023, 0.022, 0.006, material="int_needle", center=buckle + [0.006, 0, 0.025]))
        res.info["buckle_local"] = buckle.tolist()
        res.mesh.merge(H.named(P.box(0.028, 0.004, 0.013, material="int_fabric",
                                    center=(x0 - 0.015, side * p.cushion_width / 2, p.cushion_height + 0.26)), "airbag_tag"))
        res.mesh.merge(H.named(P.rounded_box(0.10, 0.023, 0.025, 0.009, material="int_plastic",
                                            center=(x0 + 0.13, side * (p.cushion_width / 2 + 0.012), p.cushion_height - 0.075)), "recline_lever"))
        if not opts.get("headrest", True):
            for zone in ("headrest", "post_left", "post_right"):
                res.mesh.remove_faces(res.mesh.zone_faces(zone))
            res.mesh.prune_unused_vertices()
        if "headroom" in conn.meta:
            clearance = float(conn.meta["headroom"]) - float(res.mesh.vertices[:, 2].max())
            res.info["headroom_clearance"] = clearance
            if clearance < 0:
                res.info["fit_warnings"] = [f"Seat exceeds supplied headroom by {-clearance:.3f} m; raise the roof or lower the H-point."]
        _finish(res.mesh, ctx)
        res.mesh.materials = {k: v for k, v in res.mesh.materials.items() if k in set(res.mesh.face_materials)}
        return res


@register
class SportSeat(BucketSeat):
    """Deeper bolsters and a continuous shoulder-to-head shell, no metal posts."""
    name = "seat.sport"
    default_for = ()
    options = dict(BucketSeat.options, bolster=0.07, recline_deg=20.0)
    description = "sport bucket with deep bolsters and an integrated headrest"

    def generate(self, p: SeatParams) -> Mesh:
        m = super().generate(p)
        for zone in ("headrest", "post_left", "post_right"):
            m.remove_faces(m.zone_faces(zone))
        m.prune_unused_vertices()
        r = np.radians(p.recline_deg)
        u, t = np.array([-np.sin(r), 0, np.cos(r)]), np.array([np.cos(r), 0, np.sin(r)])
        hinge = np.array([-p.footprint_depth / 2 + 0.09, 0, p.cushion_height - 0.035])
        secs, origins = [], []
        for h, w in ((p.backrest_height - 0.14, 0.34), (p.backrest_height, 0.28),
                     (p.backrest_height + 0.28, 0.25), (p.backrest_height + 0.34, 0.21)):
            secs.append(rounded_rect_points(w, 0.095, 0.025, 3))
            origins.append(hinge + u * h)
        head = H.orient_outward(H.section_loft(secs, origins, np.array([0, 1, 0]), t, "int_leather"))
        m.merge(H.named(head, "headrest"))
        return m


def _bench_cushion_section(x0: float, d: float, t: float, shrink: float = 0.0) -> np.ndarray:
    """Closed (x, z) outline of a bench cushion (10 points): rear at x0, nose at x0+d."""
    x1 = x0 + d - shrink
    zt = t - shrink
    return np.array([
        (x0, 0.0), (x0 + d * 0.5, 0.0), (x1 - 0.05, 0.0),
        (x1, 0.35 * zt), (x1 - 0.01, 0.85 * zt), (x1 - 0.06, zt),
        (x0 + d * 0.5, zt + 0.006), (x0 + 0.10, zt - 0.004), (x0 + 0.02, zt - 0.03), (x0, 0.75 * zt),
    ])


def _bench_back_section(bh: float, front: float, back: float, shrink: float = 0.0) -> np.ndarray:
    """Closed (t, h) outline of a bench backrest (8 points), t forward, h up the back."""
    f = front - shrink
    h = bh - shrink
    return np.array([
        (-back, 0.0), (-back, h * 0.95), (-back * 0.5, h), (f * 0.4, h),
        (f, h * 0.9), (f + 0.012, h * 0.55), (f, h * 0.15), (f * 0.6, 0.0),
    ])


@dataclass
class BenchParams:
    width: float = 1.44
    depth: float = 0.45
    cushion_height: float = 0.20
    cushion_thickness: float = 0.11
    backrest_height: float = 0.60
    recline_deg: float = 24.0
    footprint_depth: float = 0.50
    split: float = 0.6                # fold seam position (fraction of width from the left)


@register
class BenchSeat(MorphableComponent):
    """Rear bench spanning the full connector width, with a 60/40 fold seam,
    reclined backrest and two or three headrests (build-time option)."""

    name = "seat.bench"
    accepts = (RectangleConnector,)
    default_for = ("seat_bench",)
    params_cls = BenchParams
    options = {"recline_deg": 24.0, "backrest_height": 0.60, "headrests": "auto", "split": 0.6}
    description = "rear bench seat with fold seam and headrests, spanning the bench connector"
    modifier_specs = [
        ModifierSpec("width", 0.40, 0.40, "size", "bench width"),
        ModifierSpec("depth", 0.08, 0.10, "size", "cushion depth"),
        ModifierSpec("cushion_height", 0.08, 0.16, "fit", "cushion top height"),
        ModifierSpec("cushion_thickness", 0.03, 0.04, "style", "cushion thickness"),
        ModifierSpec("backrest_height", 0.15, 0.12, "size", "backrest height"),
        ModifierSpec("recline_deg", 10.0, 12.0, "pose", "backrest recline"),
        ModifierSpec("footprint_depth", 0.10, 0.15, "fit", "footprint depth"),
        ModifierSpec("split", 0.15, 0.15, "style", "fold seam position"),
    ]

    def _stations(self, p: BenchParams) -> List[Tuple[float, float]]:
        """(y, shrink) pairs across the bench: rounded ends + seam groove."""
        w, s = p.width, p.split
        fr = [(0.0, 0.03), (0.02, 0.008), (0.10, 0.0), (s - 0.02, 0.0), (s - 0.006, 0.02), (s + 0.006, 0.02),
              (s + 0.02, 0.0), (0.90, 0.0), (0.98, 0.008), (1.0, 0.03)]
        return [(w / 2 - f * w, sh) for f, sh in fr]

    def _axes(self, p: BenchParams):
        r = np.radians(p.recline_deg)
        u = np.array([-np.sin(r), 0.0, np.cos(r)])
        t = np.array([np.cos(r), 0.0, np.sin(r)])
        x0 = -p.footprint_depth / 2 + 0.02
        hinge = np.array([x0 + 0.06, 0.0, p.cushion_height - 0.06])
        return x0, u, t, hinge

    def generate(self, p: BenchParams) -> Mesh:
        x0, u, t, hinge = self._axes(p)
        zb = p.cushion_height - p.cushion_thickness
        ex, ez = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        secs, origins = [], []
        for y, sh in self._stations(p):
            secs.append(_bench_cushion_section(x0, p.depth, p.cushion_thickness, sh))
            origins.append(np.array([0.0, y, zb]))
        cushion = H.orient_outward(H.section_loft(secs, origins, ex, ez, "int_leather", name="bench_cushion"))
        secs, origins = [], []
        for y, sh in self._stations(p):
            secs.append(_bench_back_section(p.backrest_height, 0.06, 0.05, sh))
            origins.append(hinge + np.array([0.0, y, 0.0]))
        back = H.orient_outward(H.section_loft(secs, origins, t, u, "int_leather", name="bench_back"))
        riser_h = max(zb - 0.02, 0.02)
        riser = P.box(p.depth * 0.8, p.width - 0.12, riser_h, material="int_plastic",
                      center=(x0 + p.depth * 0.45, 0, riser_h / 2 + 0.01), name="riser")
        return cushion.merge(back).merge(riser)

    def fit(self, conn: RectangleConnector, opts, ctx) -> Dict[str, float]:
        hp = float(conn.meta.get("h_point_height", 0.25))
        headroom = float(conn.meta.get("headroom", 1.2))
        ch = hp - 0.05
        bh = min(float(opts["backrest_height"]), headroom - ch - 0.24)
        return {
            "width": self.value_for("width", conn.height - 0.06),
            "depth": self.value_for("depth", conn.width - 0.05),
            "footprint_depth": self.value_for("footprint_depth", conn.width),
            "cushion_height": self.value_for("cushion_height", ch),
            "backrest_height": self.value_for("backrest_height", bh),
            "recline_deg": self.value_for("recline_deg", float(opts["recline_deg"])),
            "split": self.value_for("split", float(opts["split"])),
        }

    def materials(self, ctx, opts):
        return _mats(ctx)

    def build_local(self, conn, opts, ctx) -> ComponentResult:
        res = super().build_local(conn, opts, ctx)
        p = H.params_from_values(self, res.info["modifier_values"])
        x0, u, t, hinge = self._axes(p)
        n = opts.get("headrests", "auto")
        n = 3 if n == "auto" else int(np.clip(int(n), 0, 3))
        ys = [0.0] if n == 1 else np.linspace(-(p.width / 2 - 0.30), p.width / 2 - 0.30, n)
        top = hinge + u * p.backrest_height
        for k, y in enumerate(ys):
            res.mesh.merge(_headrest(top + np.array([0.0, y, 0.0]), u, t, 0.22, 0.18, 0.05), group_prefix=f"headrest_{k}")
        res.info["headrests"] = n
        if ctx.measurements.get("has_bed", 0.0) > 0.5:
            # Custom recline can exceed the default bench envelope reserved by
            # the pickup's mount allocator. Report the completed seat's actual
            # clearance to the measured cab wall + its default 20 mm trim;
            # never shorten the seat or silently override its requested pose.
            wall_x = min(ctx.measurements["x_deck"] - 0.05,
                         ctx.measurements["x_roof_rear"] + 0.15) + 0.02
            clearance = float(conn.frame.to_world(res.mesh.vertices)[:, 0].min() - wall_x)
            res.info["cab_rear_clearance"] = clearance
            if clearance < 0:
                res.info.setdefault("fit_warnings", []).append(
                    f"Bench exceeds the pickup cab rear clearance by {-clearance:.3f} m; "
                    "reduce recline or use seat_rows: 2.")
        _finish(res.mesh, ctx)
        res.mesh.materials = {k: v for k, v in res.mesh.materials.items() if k in set(res.mesh.face_materials)}
        return res


# =============================================================== BELTS / PILLAR CLOSURE
@register
class ShoulderBelt(CarComponent):
    name = "belt.shoulder"
    accepts = (PointConnector,)
    default_for = ("belt_anchor",)
    description = "B-pillar height adjuster, D-ring and 47 mm webbing to the seat buckle"

    def build_local(self, conn, opts, ctx):
        seat = RectangleConnector(conn.meta["seat"], Frame.from_normal(conn.meta["seat_origin"], [0, 0, 1], x_hint=(1, 0, 0)),
                                  *conn.meta["seat_size"], meta={"h_point_height": conn.meta["h_point_height"]})
        component = BucketSeat()
        params = H.params_from_values(component, component.fit(seat, component.resolve_options(None), ctx))
        side = 1 if conn.meta["side"] == "L" else -1
        buckle_world = seat.frame.to_world(H.bucket_buckle(params, side))[0]
        buckle = conn.frame.to_local(buckle_world)[0]
        m = H.named(P.rounded_box(0.045, 0.14, 0.014, 0.012, material="int_plastic"), "belt_adjuster")
        m.merge(P.rounded_box(0.064, 0.023, 0.012, 0.008, material="int_metal", center=(0, 0, 0.014)))
        # Flat webbing, not a cable: slight forward bow clears the seat bolster.
        start = np.array([0.0, 0.0, 0.025])
        mid = start * 0.45 + buckle * 0.55 + np.array([0.085, 0.0, 0.015])
        m.merge(H.ribbon([start, mid, buckle], float(conn.meta["belt_width"]), 0.0014,
                         "int_belt", "shoulder_webbing", up=(0, 0, 1)))
        return ComponentResult(_finish(m, ctx), info={"buckle_world": buckle_world.tolist()})


@register
class PillarTrim(CarComponent):
    name = "pillar.trim"
    accepts = (PointConnector,)
    default_for = ("pillar_trim",)
    description = "closed A/B-pillar fabric trim following the morphed aperture and pillar"

    def build_local(self, conn, opts, ctx):
        m = Mesh(name="pillar_trim")
        if "grid_points" in conn.meta:
            grid = np.asarray(conn.meta["grid_points"])
            material = "int_carpet" if conn.meta["pillar"] == "lower" else "int_plastic" if conn.meta["pillar"] == "belt" else "int_fabric"
            skin = H.patch_surface(grid, conn.normal, material, thickness=0.008, name="pillar_skin")
            skin.transform(np.linalg.inv(conn.frame.matrix))
            m.merge(H.named(skin, "pillar_skin"))
        if "path" in conn.meta:
            path = conn.frame.to_local(np.asarray(conn.meta["path"]))
            m.merge(H.ribbon(path, float(conn.meta["width"]), 0.018, "int_fabric", "pillar_trim", up=(0, 0, 1)))
        return ComponentResult(_finish(m, ctx))


@register
class CowlTrim(CarComponent):
    name = "cowl.trim"
    accepts = (PointConnector,)
    default_for = ("cowl_trim",)
    description = "windshield-base cowl closure and demister slots"

    def build_local(self, conn, opts, ctx):
        path = conn.frame.to_local(np.asarray(conn.meta["path"]))
        m = H.ribbon(path, 0.085, 0.014, "int_soft", "cowl_strip")
        for i, p in enumerate(path[1:-1]):
            m.merge(H.named(P.box(0.025, 0.003, 0.002, material="int_gauge", center=p + [0, 0, 0.008]), f"demister_{i}"))
        return ComponentResult(_finish(m, ctx))


# =============================================================== STEERING WHEEL
@register
class SteeringWheel(CarComponent):
    """Rim (torus), three spokes, airbag hub, stalks and the column tube going
    along -Z into the dashboard.  Local: +Z towards the driver, +X car-left,
    +Y down the wheel plane."""

    name = "steering.wheel"
    accepts = (CircleConnector,)
    default_for = ("steering_wheel",)
    options = {"rim_thickness": 0.016, "spokes": 3, "flat_bottom": "auto"}
    description = "steering wheel with airbag hub, stalks and column, sized from the connector radius"

    def build_local(self, conn: CircleConnector, opts, ctx) -> ComponentResult:
        R = conn.radius
        rt = float(opts["rim_thickness"])
        L = float(conn.meta.get("column_length", 0.35))
        flat = opts["flat_bottom"]
        flat = ctx.hints.get("style") == "sports" if flat == "auto" else bool(flat)
        if flat:
            a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            path = np.column_stack([(R - rt) * np.cos(a), np.minimum((R - rt) * np.sin(a), R * 0.74), np.zeros_like(a)])
            b = np.linspace(0, 2 * np.pi, 10, endpoint=False)
            tangent = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
            radial = np.column_stack([tangent[:, 1], -tangent[:, 0], np.zeros(len(path))])
            radial /= np.linalg.norm(radial, axis=1)[:, None]
            rings = [p + rt * np.cos(b)[:, None] * n + rt * np.sin(b)[:, None] * [0, 0, 1] for p, n in zip(path, radial)]
            # Wrap tangents too: a generic open-end sweep leaves a wedge at
            # the closed path's seam, even when its endpoint is duplicated.
            rim = P.loft(rings + [rings[0]], material="int_leather", name="rim").flip_normals()
        else:
            rim = P.torus(R - rt, rt, 48, 10, material="int_leather", name="rim")
        m = H.named(rim, "rim")
        # spokes at 3, 9 o'clock (±x) and 6 o'clock (+y is "down" in this frame)
        angles = [0.0, np.pi, np.pi / 2] if int(opts["spokes"]) == 3 else [0.0, np.pi, np.pi / 2 - 0.5, np.pi / 2 + 0.5]
        for i, a in enumerate(angles):
            reach = min(R - rt, R * 0.74 / max(np.sin(a), 1e-9)) if flat else R - rt
            ln = reach - 0.04
            spoke = P.box(ln, 0.032, 0.02, material="int_plastic", center=(0.04 + ln / 2, 0, 0.0), name="spoke", bevel=0.004)
            spoke.transform(H.rot_z(a))
            m.merge(H.named(spoke, f"spoke_{i}"))
        m.merge(H.named(P.rounded_box(0.179, 0.134, 0.042, 0.04, material="int_trim", center=(0, 0.005, 0.012)), "horn_edge"))
        hub = P.rounded_box(0.17, 0.125, 0.05, 0.04, material="int_soft", center=(0, 0.005, 0.012), name="hub")
        m.merge(H.named(hub, "airbag_pad"))
        for s in (-1.0, 1.0):
            m.merge(H.named(P.rounded_box(0.025, 0.05, 0.027, 0.012, material="int_leather", center=(s * (R - 0.052), -0.027, 0.005)),
                            "thumb_left" if s > 0 else "thumb_right"))
        m.merge(P.cylinder(0.02, 0.004, 20, material="int_chrome", center=(0, -0.01, 0.039), name="emblem"))
        # column: shroud behind the wheel, tube to the dash
        shroud = P.rounded_box(0.10, 0.12, 0.17, 0.03, material="int_plastic", center=(0, 0.01, -0.13), name="shroud")
        m.merge(shroud)
        m.merge(P.cylinder(0.024, L - 0.2, 16, material="int_plastic", center=(0, 0.0, -0.2 - (L - 0.2) / 2), name="column"))
        for s in (-1.0, 1.0):
            stalk = P.cylinder(0.007, 0.11, 8, material="int_plastic", center=(0, 0, 0.055), name="stalk")
            stalk.transform(H.rot_y(s * np.pi / 2)).translate((s * 0.05, 0.0, -0.085))
            stalk.merge(P.rounded_box(0.025, 0.02, 0.04, 0.008, material="int_plastic", center=(s * 0.15, 0, -0.085)))
            m.merge(H.named(stalk, "stalk_left" if s > 0 else "stalk_right"))
        return ComponentResult(_finish(m, ctx), [], {"radius": R, "flat_bottom": flat})


def _console_nose_rise(height: float) -> float:
    """Console nose above its horizontal shifter deck, shared with dash packing."""
    return float(np.clip(0.35 - height, 0.0, 0.11))


# =============================================================== DASHBOARD
@dataclass
class DashParams:
    width: float = 1.65
    height: float = 0.43
    depth: float = 0.55        # design envelope (dash plane -> steering wheel)
    brow: float = 0.42         # brow (top pad) depth as a fraction of `depth`
    stack_width: float = 0.36
    chamfer: float = 0.12      # top-corner chamfer (matches the body polygon)


def _dash_depths(p: DashParams, x: float):
    """(brow, fascia, knee, stack_blend) depths at lateral position x."""
    d = p.depth
    s = float(H.bump(x, 0.0, p.stack_width / 2 - 0.02, 0.09))
    brow = p.brow * d
    stack = 0.46 * d
    fascia = 0.30 * d + (stack - 0.30 * d) * s
    knee = 0.23 * d + (stack - 0.23 * d) * s
    return brow, fascia, knee, s


def _dash_top(p: DashParams, x: float) -> float:
    half = p.width / 2
    return p.height / 2 - p.chamfer * float(np.clip((abs(x) - (half - 0.10)) / 0.10, 0, 1))


def _dash_section(p: DashParams, x: float) -> np.ndarray:
    """Closed (y, z) profile at lateral x: back face at z=0, knee bolster,
    fascia, brow lip and the top pad sloping down to the windshield base."""
    T = _dash_top(p, x)
    B = -p.height / 2
    brow, f, k, _ = _dash_depths(p, x)
    yk = min(B + 0.16, T - 0.15)                        # knee section top
    return np.array([
        (B, 0.0),
        (B, k),
        (B + 0.05, k + 0.02),
        (yk, k + 0.015),
        (yk + 0.03, f - 0.008),
        (0.5 * (yk + 0.03 + T - 0.11), f),
        (T - 0.11, f),
        (T - 0.065, brow - 0.015),
        (T - 0.025, brow),
        (T, brow - 0.045),
        (T - 0.004, brow * 0.55),
        (T - 0.02, brow * 0.2),
        (T - 0.045, 0.0),
    ])


def _dash_surface_z(p: DashParams, x: float, y: float) -> float:
    """Depth of the rear-facing dash surface at (x, y) (piecewise-linear)."""
    sec = _dash_section(p, x)[1:10]
    ys, zs = sec[:, 0], sec[:, 1]
    return float(np.interp(y, ys, zs))


def _dash_stations(p: DashParams) -> np.ndarray:
    half, s = p.width / 2, p.stack_width / 2
    xs = [0.0, s - 0.08, s - 0.02, s + 0.02, s + 0.07]
    a, b = s + 0.07, half - 0.10
    xs += [a + (b - a) / 3, a + 2 * (b - a) / 3, b, half]
    xs = sorted(set(xs))
    return np.array([-x for x in xs[::-1] if x > 0] + xs)


@register
class Dashboard(MorphableComponent):
    """Sculpted dashboard adapting to the body's dash polygon: soft top pad
    with a brow, a fascia band, a protruding centre stack and a knee-height
    lower section.  Emits cluster / screen / vent / glovebox / hvac connectors."""

    name = "dashboard.sculpted"
    accepts = (PolygonConnector,)
    default_for = ("dashboard",)
    params_cls = DashParams
    options = {"two_tone": True, "gloss_stack": True}
    description = "sculpted dashboard (top pad, fascia, centre stack, knee bolster) fitted to the dash outline; emits cluster/screen/vent/glovebox/hvac connectors"
    modifier_specs = [
        ModifierSpec("width", 0.40, 0.40, "fit", "dash width (cabin width)"),
        ModifierSpec("height", 0.12, 0.20, "fit", "dash height (floor+0.32 .. cowl)"),
        ModifierSpec("depth", 0.15, 0.15, "size", "depth envelope"),
        ModifierSpec("brow", 0.10, 0.12, "style", "top pad overhang"),
        ModifierSpec("stack_width", 0.10, 0.16, "style", "centre stack width"),
    ]

    def generate(self, p: DashParams) -> Mesh:
        ey, ez = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        secs, origins = [], []
        for x in _dash_stations(p):
            secs.append(_dash_section(p, x))
            origins.append(np.array([x, 0.0, 0.0]))
        return H.orient_outward(H.section_loft(secs, origins, ey, ez, "int_soft", name="dash"))

    def fit(self, conn: PolygonConnector, opts, ctx) -> Dict[str, float]:
        return {
            "width": self.value_for("width", conn.width - 0.02),
            "height": self.value_for("height", conn.height),
            "depth": self.value_for("depth", float(conn.meta.get("depth", 0.55))),
        }

    def materials(self, ctx, opts):
        return _mats(ctx)

    # ---- layout shared by build_local (painting) and sub_connectors
    def _layout(self, p: DashParams, conn: PolygonConnector):
        sc = conn.meta.get("steering_center")
        if sc is not None:
            x_d = float(conn.frame.to_local(np.asarray(sc, dtype=float))[0, 0])
        else:
            x_d = -float(conn.meta.get("driver_y", 0.42))
        T0 = _dash_top(p, 0.0)
        B = -p.height / 2
        yk = min(B + 0.16, T0 - 0.15)
        band_lo, band_hi = yk + 0.03, T0 - 0.11
        y_band = 0.5 * (band_lo + band_hi)
        stack_top, stack_bot = T0 - 0.11, B + 0.01
        A = stack_top - stack_bot
        h_s = float(np.clip(A - 0.20, 0.07, 0.15))
        y_vent = stack_top - 0.058
        y_screen = y_vent - 0.058 - h_s / 2 - 0.008
        y_hvac = stack_bot + 0.028
        if sc is not None:
            wheel_y = float(conn.frame.to_local(np.asarray(sc))[0, 1])
            # The level shifter deck is wheel centre - 260 mm, but in low
            # cabins the console nose rises above it. Reserve the full 50 mm
            # HVAC panel plus a 10 mm gap above that nose, not just the deck.
            y_hvac = max(y_hvac, wheel_y - 0.205)
            if "floor" in conn.meta:
                console_height = float(np.asarray(sc)[2]) - float(conn.meta["floor"]) - 0.26
                nose_y = wheel_y - 0.26 + _console_nose_rise(console_height)
                y_hvac = max(y_hvac, nose_y + 0.025 + 0.010)
        y_screen = max(y_screen, y_hvac + 0.053 + h_s / 2)
        y_vent = max(y_vent, y_screen + h_s / 2 + 0.076)
        y_cluster = y_band + 0.01
        if sc is not None:
            y_cluster = max(y_cluster, float(conn.frame.to_local(np.asarray(sc))[0, 1]) + 0.075)
        return dict(x_d=x_d, T=T0, B=B, yk=yk, band=(band_lo, band_hi), y_band=y_band, y_cluster=y_cluster, h_s=h_s,
                    y_vent=y_vent, y_screen=y_screen, y_hvac=y_hvac, half=p.width / 2, s_half=p.stack_width / 2)

    def build_local(self, conn, opts, ctx) -> ComponentResult:
        res = super().build_local(conn, opts, ctx)
        m = res.mesh
        p = H.params_from_values(self, res.info["modifier_values"])
        L = self._layout(p, conn)
        cent = m.face_centroids()
        tops = np.array([_dash_top(p, x) for x in cent[:, 0]])
        for i in range(m.n_faces):
            x, y, z = cent[i]
            if z < 0.004:
                m.face_materials[i] = "int_plastic"          # back face (against the firewall)
            elif y > tops[i] - 0.075:
                m.face_materials[i] = "int_soft"             # top pad + brow
            elif abs(x) < L["s_half"] + 0.03 and opts.get("gloss_stack", True):
                m.face_materials[i] = "int_trim"             # centre stack (gloss)
            elif y > L["yk"] + 0.012:
                m.face_materials[i] = "int_accent" if opts.get("two_tone", True) else "int_plastic"
            else:
                m.face_materials[i] = "int_plastic"          # knee bolster
        # Passenger airbag tear seam and a continuous fascia accent.  Sample
        # the fitted dash surface so these do not hover over its curved brow.
        xp = -L["x_d"]
        y0, y1 = L["T"] - 0.060, L["T"] - 0.026
        outline = [(xp - 0.14, y0), (xp + 0.14, y0), (xp + 0.14, y1), (xp - 0.14, y1), (xp - 0.14, y0)]
        path = [[x, y, _dash_surface_z(p, x, y) + 0.001] for x, y in outline]
        m.merge(H.ribbon(path, 0.0015, 0.001, "int_trim", "passenger_airbag_seam"))
        for sign in (-1, 1):
            xs = np.linspace(sign * (L["s_half"] + 0.05), sign * (L["half"] - 0.06), 12)
            path = [[x, _dash_top(p, x) - 0.09, _dash_surface_z(p, x, _dash_top(p, x) - 0.09) + 0.002] for x in xs]
            m.merge(H.ribbon(path, 0.005, 0.002, "int_chrome", f"fascia_strip_{sign}"))
        _finish(m, ctx)
        m.materials = {k: v for k, v in m.materials.items() if k in set(m.face_materials)}
        return res

    def sub_connectors(self, conn: PolygonConnector, opts, ctx, values) -> List[Connector]:
        p = H.params_from_values(self, values)
        L = self._layout(p, conn)
        x_d = L["x_d"]
        out: List[Connector] = []

        def rect(name, x, y, w, h, tilt, tags, **meta):
            z = _dash_surface_z(p, x, y) + 0.004
            if name == "cluster":
                z = max(_dash_surface_z(p, x, yy) for yy in np.linspace(y - h / 2, y + h / 2, 9)) + 0.008 + abs(np.sin(tilt)) * h / 2
            fr = Frame.from_normal([x, y, z], [0.0, np.sin(tilt), np.cos(tilt)], x_hint=(1, 0, 0))
            out.append(RectangleConnector(name, fr, w, h, tags=tags, meta=meta))

        def circ(name, x, y, r, tilt, tags, **meta):
            z = _dash_surface_z(p, x, y) + 0.004
            fr = Frame.from_normal([x, y, z], [0.0, np.sin(tilt), np.cos(tilt)], x_hint=(1, 0, 0))
            out.append(CircleConnector(name, fr, r, tags=tags, meta=meta))

        band_h = L["band"][1] - L["band"][0]
        rect("cluster", x_d, L["y_cluster"], 0.26, min(0.12, band_h + 0.02), np.radians(12), ["cluster"],
             hood_height=L["T"] - L["y_cluster"], driver_x=x_d)
        rect("screen", 0.0, L["y_screen"], min(0.30, p.stack_width - 0.05), L["h_s"], np.radians(6), ["screen"])
        rect("hvac", 0.0, L["y_hvac"], min(0.24, p.stack_width - 0.08), 0.05, 0.0, ["hvac"])
        rect("glovebox", -x_d, 0.5 * (L["B"] + L["yk"]) + 0.005, min(0.40, L["half"] - abs(x_d) - 0.02) * 2 - 0.16,
             (L["yk"] - L["B"]) - 0.04, 0.0, ["glovebox"])
        xo = L["half"] - 0.13
        circ("vent_L", -xo, L["y_band"], 0.045, np.radians(8), ["vent"], side="left")
        circ("vent_R", xo, L["y_band"], 0.045, np.radians(8), ["vent"], side="right")
        xc = min(0.09, L["s_half"] - 0.07)
        circ("vent_center_L", -xc, L["y_vent"], 0.045, np.radians(4), ["vent"], side="left")
        circ("vent_center_R", xc, L["y_vent"], 0.045, np.radians(4), ["vent"], side="right")
        return out


@register
class AnalogCluster(CarComponent):
    """Hooded binnacle with two round gauges (chrome bezels, dark faces, lit
    rings, needles) and a small centre display."""

    name = "cluster.analog"
    accepts = (RectangleConnector,)
    default_for = ("cluster",)
    options = {"hood": True, "hood_depth": 0.10}
    description = "instrument binnacle: hood over two analogue gauges with needles and a centre display"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        w, h = conn.width, conn.height
        m = P.rounded_box(w, h, 0.012, 0.02, material="int_gauge", center=(0, 0, 0.006), name="cluster_plate")
        m.merge(P.rounded_box(w + 0.014, h + 0.012, 0.065, 0.02, material="int_plastic", center=(0, 0, -0.027), name="cluster_pod"))
        rg = min(h * 0.42, w * 0.2)
        for k, (x, ang) in enumerate(((-w * 0.24, -0.7), (w * 0.24, 0.45))):
            m.merge(P.tube(rg + 0.006, rg - 0.004, 0.012, 24, material="int_chrome", center=(x, 0, 0.018), name="bezel"))
            face = P.cylinder(rg - 0.004, 0.003, 24, material="int_gauge", center=(x, 0, 0.0135), name="face")
            m.merge(H.named(face, "tachometer_face" if k == 0 else "speedometer_face"))
            for j, a in enumerate(np.linspace(-2.35, 2.35, 25)):
                length = 0.006 if j % 2 == 0 else 0.003
                tick = P.box(0.0015, length, 0.001, material="int_lens", center=(0, rg - 0.012 - length / 2, 0.018))
                m.merge(H.named(tick.transform(H.rot_z(a)).translate((x, 0, 0)), f"gauge_{k}_tick_{j}"))
            for j, a in enumerate(np.linspace(-2.15, 2.15, 5)):
                marking = str(j * (2 if k == 0 else 40))
                # Labels read upright across the instrument, not rotated text.
                at = (x + (rg - 0.025) * np.sin(a), (rg - 0.025) * np.cos(a), 0.018)
                m.merge(H.digits(marking, 0.0055, center=at))
            m.merge(P.tube(rg - 0.006, rg - 0.010, 0.001, 24, material="int_lens", center=(x, 0, 0.0155), name="lit_ring"))
            needle = P.box(0.003, rg * 0.85, 0.002, material="int_needle", center=(0, rg * 0.32, 0.0165), name="needle")
            needle.transform(H.rot_z(ang)).translate((x, 0, 0))
            m.merge(H.named(needle, f"needle_{k}"))
            m.merge(P.cylinder(0.006, 0.003, 12, material="int_chrome", center=(x, 0, 0.017), name="cap"))
        m.merge(P.box(w * 0.2, h * 0.5, 0.003, material="int_screen", center=(0, 0, 0.0135), name="centre_display"))
        if opts.get("hood", True):
            # thin "C"-shaped shell lofted from inside the dash out to the visor lip
            a, b = w / 2 + 0.035, h / 2 + 0.06
            th = np.linspace(np.pi, 0.0, 9)
            outer = np.column_stack([a * np.cos(th), b * np.sin(th) - 0.005])
            inner = np.column_stack([(a - 0.007) * np.cos(th), (b - 0.007) * np.sin(th) - 0.005])
            prof = np.vstack([outer, inner[::-1]])
            rear_outline = np.vstack([outer, [[a, -h / 2 - 0.01], [-a, -h / 2 - 0.01]]])
            m.merge(H.named(P.extrude_polygon(rear_outline, 0.006, material="int_gauge", z0=-0.036), "binnacle_back"))
            depth = float(opts["hood_depth"])
            secs, origins = [], []
            for z, sx, dy in ((-0.03, 1.0, 0.0), (0.04, 1.02, 0.004), (depth, 0.95, -0.012)):
                secs.append(prof * np.array([sx, 1.0]) + np.array([0.0, dy]))
                origins.append(np.array([0.0, 0.0, z]))
            hood = H.section_loft(secs, origins, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), "int_soft",
                                  cap_start=False, cap_end=False, name="hood")
            # A C-shaped profile is concave: a fan cap would bridge its opening
            # and hide the gauges.  Join each outer arc segment to its inner one.
            for off, sign in ((0, -1), (2 * len(prof), 1)):
                for j in range(len(th) - 1):
                    face = (off + j, off + j + 1, off + len(prof) - 2 - j, off + len(prof) - 1 - j)
                    hood.faces.append(face if sign < 0 else tuple(reversed(face)))
                    hood.face_materials.append("int_soft")
            m.merge(H.named(H.orient_outward(hood), "binnacle_hood"))
        return ComponentResult(_finish(m, ctx))


@register
class InfotainmentScreen(CarComponent):
    name = "screen.infotainment"
    accepts = (RectangleConnector,)
    default_for = ("screen",)
    options = {"ui": True}
    description = "glossy dark infotainment slab in a thin bezel"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        w, h = conn.width, conn.height
        m = P.rounded_box(w + 0.02, h + 0.02, 0.014, 0.012, material="int_trim", center=(0, 0, 0.007), name="bezel")
        m.merge(P.box(w - 0.006, h - 0.006, 0.003, material="int_screen", center=(0, 0, 0.0155), name="screen"))
        if opts.get("ui", True):
            m.merge(P.box(w * 0.34, h * 0.62, 0.001, material="int_ui", center=(-w * 0.24, h * 0.05, 0.0175), name="ui_map"))
            m.merge(P.box(w * 0.8, h * 0.1, 0.001, material="int_ui", center=(0, -h * 0.36, 0.0175), name="ui_bar"))
        return ComponentResult(_finish(m, ctx))


@register
class RoundVent(CarComponent):
    name = "vent.round"
    accepts = (CircleConnector,)
    default_for = ("vent",)
    options = {"slats": 4, "chrome_ring": True}
    description = "round air vent: bezel ring, recessed dark barrel, tilted louvre slats and a centre knob"

    def build_local(self, conn: CircleConnector, opts, ctx) -> ComponentResult:
        r = conn.radius
        m = P.tube(r + 0.012, r - 0.002, 0.018, 24, material="int_plastic", center=(0, 0, 0.009), name="vent_ring")
        if opts.get("chrome_ring", True):
            m.merge(P.tube(r + 0.014, r + 0.010, 0.02, 24, material="int_chrome", center=(0, 0, 0.01), name="vent_bezel"))
        m.merge(P.cylinder(r, 0.004, 24, material="int_gauge", center=(0, 0, -0.004), name="vent_back"))
        n = int(opts["slats"])
        for k in range(n):
            y = (k - (n - 1) / 2) * (2 * r * 0.8 / n)
            half_chord = np.sqrt(max(r * r - y * y, 1e-6)) - 0.004
            slat = P.box(2 * half_chord, 0.005, 0.014, material="int_plastic", center=(0, 0, 0), name="slat")
            slat.transform(H.rot_x(np.radians(20))).translate((0, y, 0.004))
            m.merge(slat)
        m.merge(P.cylinder(r * 0.16, 0.008, 12, material="int_chrome", center=(0, 0, 0.012), name="vent_knob"))
        return ComponentResult(_finish(m, ctx))


@register
class GloveboxLid(CarComponent):
    name = "glovebox.lid"
    accepts = (RectangleConnector,)
    default_for = ("glovebox",)
    options = {}
    description = "glovebox lid with a shadow-gap seam and a chrome latch"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        w, h = conn.width, conn.height
        m = P.rounded_box(w + 0.014, h + 0.014, 0.006, 0.02, material="int_gauge", center=(0, 0, 0.003), name="seam")
        m.merge(P.rounded_box(w, h, 0.012, 0.018, material="int_plastic", center=(0, 0, 0.012), name="lid"))
        m.merge(P.box(0.07, 0.012, 0.005, material="int_chrome", center=(0, h / 2 - 0.022, 0.02), name="latch"))
        return ComponentResult(_finish(m, ctx))


@register
class HvacKnobs(CarComponent):
    name = "hvac.knobs"
    accepts = (RectangleConnector,)
    default_for = ("hvac",)
    options = {"knobs": 3}
    description = "climate panel with three rotary knobs and button rows"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        w, h = conn.width, conn.height
        m = P.rounded_box(w, h, 0.01, 0.012, material="int_trim", center=(0, 0, 0.005), name="hvac_panel")
        n = int(opts["knobs"])
        rk = min(0.014, h * 0.28)
        for k in range(n):
            x = (k - (n - 1) / 2) * (w * 0.68 / max(n - 1, 1))
            m.merge(H.knob(rk, 0.018, "int_plastic", "int_chrome", 20, center=(x, 0.008, 0.01), pointer="int_lens"))
        for k, x in enumerate(np.linspace(-w * 0.38, w * 0.38, 6)):
            button = P.box(0.012, 0.010, 0.004, material="int_plastic", center=(x, -0.016, 0.012))
            m.merge(H.named(button, f"hvac_button_{k}"))
            m.merge(P.box(0.004, 0.0015, 0.001, material="int_lens", center=(x, -0.016, 0.0145)))
        return ComponentResult(_finish(m, ctx))


# =============================================================== PEDALS
@register
class Pedals(CarComponent):
    """Hinged pedal plates leaning ~60 deg from the floor on stalks: accelerator
    (organ type, right), brake (wide pad), clutch for manuals, and a dead pedal."""

    name = "pedals.hinged"
    accepts = (RectangleConnector,)
    default_for = ("pedals",)
    options = {"lean_deg": 60.0, "dead_pedal": True}
    description = "accelerator / brake (/ clutch) pedals on stalks plus a footrest, laid out across the footwell"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        lean = np.radians(90.0 - float(opts["lean_deg"]))   # rotation about +y tilting the pad forward
        manual = conn.meta.get("transmission", "automatic") == "manual"
        R = H.rot_y(lean)
        m = Mesh(name="pedals")

        def pad(y, width, height, z_base, thick=0.012, mat="int_rubber", stalk_h=0.0):
            p = P.box(thick, width, height, material=mat, center=(0, 0, z_base + height / 2), name="pad", bevel=0.003)
            if stalk_h > 0:
                p.merge(P.box(0.012, 0.022, stalk_h + 0.02, material="int_metal", center=(0.008, 0, stalk_h / 2), name="stalk"))
            for z in np.linspace(z_base + 0.012, z_base + height - 0.012, max(3, int(height / 0.02))):
                p.merge(P.box(0.003, width * 0.8, 0.004, material="int_metal", center=(-thick / 2 - 0.001, 0, z)))
            p.transform(R).translate((0.0, y, 0.0))
            m.merge(p)

        pad(-0.09, 0.055, 0.17, 0.02)                              # accelerator (floor hinged)
        if manual:
            pad(0.0, 0.085, 0.06, 0.15, stalk_h=0.15)              # brake
            pad(0.11, 0.075, 0.06, 0.15, stalk_h=0.15)             # clutch
        else:
            pad(0.02, 0.11, 0.06, 0.15, stalk_h=0.15)              # wide brake pad
            if opts.get("dead_pedal", True):
                pad(0.13, 0.075, 0.16, 0.02, thick=0.018)          # footrest
        if manual and opts.get("dead_pedal", True):
            start = m.n_vertices
            pad(0.22, 0.065, 0.16, 0.02, thick=0.018)
            m.add_group("footrest", range(start, m.n_vertices))
        # Heel patch establishes the accelerator heel reference on the carpet.
        heel = np.array([-0.025, -0.09, 0.0])
        m.merge(H.named(P.box(0.05, 0.07, 0.004, material="int_rubber", center=heel), "heel_patch"))
        m.merge(P.box(0.03, conn.height * 0.7, 0.02, material="int_metal", center=(-0.005, 0.01, 0.01), name="hinge"))
        return ComponentResult(_finish(m, ctx), info={"heel_local": heel.tolist()})


# =============================================================== CONSOLE
@register
class CenterConsole(CarComponent):
    """Centre console between the front seats: rounded body rising to meet the
    dash, wood/accent shifter panel, padded armrest lid at the rear.  Emits
    `shifter` (point) and two `cupholder` (circle) connectors."""

    name = "console.center"
    accepts = (RectangleConnector,)
    default_for = ("console",)
    options = {"armrest": True, "panel_material": "int_wood", "cup_cover": 0.35}
    description = "centre console with armrest lid and trim panel; emits shifter and cupholder connectors"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        Lc, Wc = conn.width, conn.height
        hc = float(conn.meta.get("height", 0.33))
        xr, xf = -Lc / 2, Lc / 2 + 0.03                       # nose extends slightly towards the dash
        ey, ez = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        # body: rounded-rect sections along x, height profile rises towards the dash
        secs, origins = [], []
        # Keep the nose below the centre-stack controls in high-H-point cars.
        rise = _console_nose_rise(hc)
        for x, hgt, wsc in ((xr, hc - 0.03, 0.92), (xr + 0.02, hc - 0.02, 1.0), (-0.16, hc - 0.02, 1.0), (-0.14, hc, 1.0),
                            (0.25, hc, 1.0), (0.42, hc + rise * 0.18, 1.0), (xf - 0.02, hc + rise * 0.91, 0.98), (xf, hc + rise, 0.9)):
            sec = rounded_rect_points(Wc * wsc, hgt, min(0.035, Wc * 0.15), 3)
            sec[:, 1] += hgt / 2
            secs.append(sec)
            origins.append(np.array([x, 0.0, 0.0]))
        m = H.named(H.orient_outward(H.section_loft(secs, origins, ey, ez, "int_plastic", name="console_body")), "console_body")
        # shifter trim panel + armrest
        panel = P.rounded_box(0.28, Wc - 0.06, 0.006, 0.02, material=str(opts.get("panel_material", "int_wood")),
                              center=(0.27, 0, hc + 0.003), name="shift_panel")
        m.merge(panel)
        if opts.get("armrest", True):
            arm = P.rounded_box(0.30, Wc - 0.04, 0.06, 0.028, material="int_leather", center=(xr + 0.16, 0, hc - 0.02 + 0.03), name="armrest")
            m.merge(H.named(P.rounded_box(0.306, Wc - 0.034, 0.006, 0.028, material="int_gauge",
                                          center=(xr + 0.16, 0, hc - 0.020)), "armrest_seam"))
            m.merge(H.named(arm, "armrest_lid"))
        # Sliding tambour cover: a partly retracted shutter on two guide rails.
        fraction = float(np.clip(opts["cup_cover"], 0, 1))
        for sign in (-1, 1):
            m.merge(P.box(0.21, 0.007, 0.012, material="int_plastic", center=(-0.045, sign * 0.056, hc + 0.006)))
        if fraction > 0:
            length = 0.20 * fraction
            cover = P.box(length, 0.104, 0.004, material="int_trim", center=(-0.145 + length / 2, 0, hc + 0.016))
            for x in np.arange(-0.14, -0.145 + length, 0.01):
                cover.merge(P.box(0.0012, 0.10, 0.001, material="int_metal", center=(x, 0, hc + 0.0185)))
            cover.merge(P.box(0.006, 0.045, 0.006, material="int_chrome", center=(-0.15 + length, 0, hc + 0.021)))
            m.merge(H.named(cover, "cup_sliding_cover"))
        side = 1 if conn.meta.get("driver_y", 0.42) > 0 else -1
        m.merge(H.named(P.rounded_box(0.025, 0.021, 0.004, 0.004, material="int_gauge", center=(0.15, side * 0.085, hc + 0.006)), "ebrake_surround"))
        m.merge(H.named(P.box(0.014, 0.012, 0.006, material="int_plastic", center=(0.15, side * 0.085, hc + 0.010)), "ebrake_switch"))
        m.merge(P.box(0.005, 0.002, 0.001, material="int_lens", center=(0.15, side * 0.085, hc + 0.0135)))
        m.merge(H.button_row(2, 0.025, (0.014, 0.012), 0.004, "int_trim", center=(0.10, 0.0, hc), along="x"))
        subs = [
            PointConnector("shifter", Frame.from_normal([0.27, 0.0, hc + 0.006], [0, 0, 1], x_hint=(1, 0, 0)), tags=["shifter"],
                           meta={"transmission": conn.meta.get("transmission", "automatic")}),
            CircleConnector("cupholder_1", Frame.from_normal([0.0, 0.0, hc], [0, 0, 1], x_hint=(1, 0, 0)), 0.04, tags=["cupholder"]),
            CircleConnector("cupholder_2", Frame.from_normal([-0.09, 0.0, hc], [0, 0, 1], x_hint=(1, 0, 0)), 0.04, tags=["cupholder"]),
        ]
        return ComponentResult(_finish(m, ctx), subs, {"height": hc})


@register
class ShifterLever(CarComponent):
    name = "shifter.lever"
    accepts = (PointConnector,)
    default_for = ("shifter",)
    options = {"height": 0.15}
    description = "gear lever: leather boot, chrome stalk and a leather knob"

    def build_local(self, conn: PointConnector, opts, ctx) -> ComponentResult:
        hgt = float(opts["height"])
        m = P.cylinder(0.045, 0.05, 16, material="int_leather", center=(0, 0, 0.025), radius_top=0.018, name="boot")
        m.merge(P.cylinder(0.011, hgt - 0.05, 12, material="int_chrome", center=(0, 0, 0.05 + (hgt - 0.05) / 2), name="stalk"))
        prof = np.array([[0.0, -0.03], [0.02, -0.028], [0.03, -0.01], [0.028, 0.015], [0.018, 0.028], [0.0, 0.03]])
        knob = P.revolve(prof, 16, material="int_leather", name="knob", center=(0, 0, hgt + 0.02))
        m.merge(knob)
        m.merge(P.cylinder(0.008, 0.002, 12, material="int_chrome", center=(0, 0, hgt + 0.051), name="knob_cap"))
        # gate indicator beside the lever
        m.merge(P.box(0.10, 0.012, 0.003, material="int_gauge", center=(0, -0.06, 0.0015), name="gate"))
        return ComponentResult(_finish(m, ctx))


@register
class RoundCupholder(CarComponent):
    name = "cupholder.round"
    accepts = (CircleConnector,)
    default_for = ("cupholder",)
    options = {}
    description = "recessed cupholder: chrome ring around a dark well"

    def build_local(self, conn: CircleConnector, opts, ctx) -> ComponentResult:
        r = conn.radius
        m = P.tube(r + 0.008, r, 0.008, 24, material="int_chrome", center=(0, 0, 0.004), name="cup_ring")
        m.merge(P.cylinder(r, 0.002, 24, material="int_gauge", center=(0, 0, 0.001), name="cup_well"))
        return ComponentResult(_finish(m, ctx))


# =============================================================== FLOOR / PANELS / SHELVES
def _front_wheelhouse(ctx):
    """Rear X, crown Z and inboard Y of the measured front wheel-house envelope."""
    meas = ctx.measurements
    if not all(k in meas for k in ("wheel_front_x", "wheel_front_y", "wheel_front_z", "arch_front_r")):
        return None
    radius = meas["arch_front_r"]
    tyre_width = float(ctx.hints.get("tire_width", 0.225))
    return (meas["wheel_front_x"] - radius - 0.025,
            meas["wheel_front_z"] + radius + 0.025,
            meas["wheel_front_y"] - tyre_width - 0.035)


@register
class CarpetFloor(CarComponent):
    """Carpet slab with a raised transmission tunnel (tapering under the rear
    seats) and a toe-board ramp at the front of the footwells."""

    name = "floor.carpet"
    accepts = (RectangleConnector,)
    default_for = ("floor",)
    options = {"tunnel_half_width": 0.17, "toe_ramp": 0.10}
    description = "cabin carpet with transmission tunnel and footwell toe-board shaping"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        Lx, Wy = conn.width, conn.height
        th = float(conn.meta.get("tunnel_height", 0.16))
        hw = float(opts["tunnel_half_width"])
        ramp = float(opts["toe_ramp"])
        x_front = Lx / 2
        xs = np.unique(np.concatenate([np.linspace(-Lx / 2, Lx / 2, 22), [x_front - 0.25, x_front - 1.3, x_front - 1.6]]))
        wheelhouse = _front_wheelhouse(ctx)
        if wheelhouse is not None:
            rear_x = wheelhouse[0] - conn.origin[0]
            # Finish the inset before the arch begins; explicit stations keep
            # a long floor quad from cutting diagonally through the tyre.
            xs = np.unique(np.r_[xs, np.clip([rear_x - 0.08, rear_x], -Lx / 2, Lx / 2)])
        ys = np.unique(np.concatenate([np.linspace(-Wy / 2, Wy / 2, 15), np.linspace(-hw, hw, 9)]))

        def z_top(X, Y):
            tunnel = np.where(np.abs(Y) < hw, np.cos(np.pi * Y / (2 * hw)) ** 2, 0.0)
            # taper the tunnel to ~0.07 under the rear seats
            hfac = 0.07 + (th - 0.07) * H.smoothstep(x_front - 1.6, x_front - 1.3, X)
            toe = ramp * np.clip((X - (x_front - 0.25)) / 0.25, 0, 1) ** 2
            return 0.01 + tunnel * hfac + toe

        m = H.heightfield_slab(xs, ys, z_top, -0.01, "int_carpet", name="carpet")
        if wheelhouse is not None and wheelhouse[2] < Wy / 2:
            inner = wheelhouse[2]
            fixed_half = min(0.60, inner - 0.04)
            blend = H.smoothstep(rear_x - 0.08, rear_x, m.vertices[:, 0])
            lateral = np.abs(m.vertices[:, 1])
            inset = np.clip((lateral - fixed_half) / (Wy / 2 - fixed_half), 0, 1)
            m.vertices[:, 1] -= np.sign(m.vertices[:, 1]) * inset * blend * (Wy / 2 - inner)
        xh = ctx.measurements.get("x_cowl", conn.origin[0] + Lx / 2) - float(ctx.hints.get("front_h_point_offset", 0.95))
        lateral = min(0.42, Wy / 2 - 0.30)
        for row, (ahead, length) in enumerate(((0.58, 0.50), (-0.28, 0.40))):
            x = float(np.clip(xh + ahead - conn.origin[0], -Lx / 2 + length / 2 + 0.02, Lx / 2 - length / 2 - 0.02))
            for side, sign in (("L", 1), ("R", -1)):
                mat = P.rounded_box(length, 0.36, 0.005, 0.035, material="int_rubber", center=(x, sign * lateral, 0.003))
                mat.merge(P.rounded_box(length - 0.016, 0.344, 0.001, 0.03, material="int_carpet", center=(x, sign * lateral, 0.0065)))
                mat.vertices[:, 2] += z_top(mat.vertices[:, 0], mat.vertices[:, 1]) + 0.002
                m.merge(H.named(mat, f"floor_mat_{row + 1}_{side}"))
        return ComponentResult(_finish(m, ctx))


@register
class BulkheadTrim(CarComponent):
    name = "bulkhead.trim"
    accepts = (RectangleConnector,)
    default_for = ("bulkhead",)
    options = {"thickness": 0.02}
    description = "dark trimmed bulkhead panel (firewall / rear bulkhead)"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        t = float(opts["thickness"])
        m = P.rounded_box(conn.width - 0.01, conn.height - 0.01, t, 0.03, material="int_carpet" if conn.meta.get("position") == "front" else "int_plastic",
                          center=(0, 0, t / 2), name="bulkhead")
        if conn.meta.get("position") == "front":
            # Close the cowl, but wrap the lower firewall around the front
            # wheel houses rather than carrying a full-width slab through the
            # tyre and rim. The upper cowl stays at its original full width.
            top = ctx.measurements["z_cowl"] - 0.025 - conn.origin[2]
            bottom = -conn.height / 2
            upper_bottom = conn.height / 2 - 0.015
            wheelhouse = _front_wheelhouse(ctx)
            intrudes = wheelhouse is not None and conn.origin[0] > wheelhouse[0]

            def levels(lo, hi):
                if not intrudes:
                    return [lo, hi]
                crown = wheelhouse[1] - conn.origin[2]
                return np.unique(np.r_[lo, np.clip([crown, crown + 0.04], lo, hi), hi])

            def width_at(y, width):
                if not intrudes:
                    return width
                blend = H.smoothstep(wheelhouse[1], wheelhouse[1] + 0.04, y + conn.origin[2])
                inner = min(width, 2 * wheelhouse[2])
                return inner + (width - inner) * blend

            def panel(lo, hi, width, material):
                heights = levels(lo, hi)
                sections = [rounded_rect_points(width_at(y, width), t, min(0.008, t / 3), 3) for y in heights]
                origins = [[0, y, t / 2] for y in heights]
                return H.orient_outward(H.section_loft(sections, origins, np.array([1, 0, 0]), np.array([0, 0, 1]), material))

            m = panel(bottom + 0.005, conn.height / 2 - 0.005, conn.width - 0.01, "int_carpet")
            if top > upper_bottom:
                m.merge(H.named(panel(upper_bottom, top, conn.width, "int_plastic"), "upper_firewall"))
            for sign in (-1, 1):
                heights = levels(bottom, top)
                sections = [rounded_rect_points(0.024, 0.16, 0.004, 3) for _ in heights]
                origins = [[sign * (width_at(y, conn.width) / 2 - 0.012), y, 0.08] for y in heights]
                side = H.orient_outward(H.section_loft(sections, origins, np.array([1, 0, 0]), np.array([0, 0, 1]), "int_plastic"))
                m.merge(H.named(side, f"firewall_return_{sign}"))
        return ComponentResult(_finish(m, ctx))


@register
class ParcelShelf(CarComponent):
    name = "shelf.parcel"
    accepts = (RectangleConnector,)
    default_for = ("shelf",)
    options = {"speakers": True}
    description = "carpeted parcel shelf with two speaker grilles"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        w, h = conn.width, conn.height
        m = P.box(w, h, 0.03, material="int_carpet", center=(0, 0, 0.015), name="shelf")
        if opts.get("speakers", True) and h > 0.9:
            for y in (-(h / 2 - 0.30), h / 2 - 0.30):
                m.merge(H.grille(0.075, f"shelf_speaker_{y:+.2f}").translate((0, y, 0.035)))
        return ComponentResult(_finish(m, ctx))


@register
class CargoFloor(CarComponent):
    name = "floor.cargo"
    accepts = (RectangleConnector,)
    default_for = ("cargo_floor",)
    options = {}
    description = "lined cargo floor, side returns and closed inner rear wheel houses"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        length, width = conn.width, conn.height
        m = P.box(length, width, 0.03, material="int_carpet", center=(0, 0, 0.015), name="cargo_floor")
        height = max(0.20, ctx.measurements["z_belt"] - conn.origin[2] - 0.04)
        for sign in (-1, 1):
            m.merge(H.named(P.box(length, 0.025, height, material="int_carpet",
                                  center=(0, sign * (width / 2 - 0.0125), height / 2)), f"cargo_side_{sign}"))
        m.merge(H.named(P.box(0.025, width, height, material="int_plastic",
                              center=(-length / 2 + 0.0125, 0, height / 2)), "cargo_rear_return"))
        return ComponentResult(_finish(m, ctx))


@register
class CargoWheelhouse(CarComponent):
    name = "cargo.wheelhouse"
    accepts = (PointConnector,)
    default_for = ("cargo_wheelhouse",)
    description = "closed carpeted inner wheel house fitted to the measured rear arch"

    def build_local(self, conn, opts, ctx):
        radius, height = float(conn.meta["radius"]), float(conn.meta["height"])
        sign = 1 if conn.meta["side"] == "L" else -1
        a = np.linspace(np.pi, 0, 17)
        profile = np.column_stack([radius * np.cos(a), height * np.sin(a)])
        secs, origins = [], []
        for inset, scale in ((float(conn.meta["depth"]), 0.86), (0.21, 0.98), (0.0, 1.0)):
            secs.append(profile * [1, scale])
            origins.append([0, -sign * inset, 0])
        m = H.orient_outward(H.section_loft(secs, origins, np.array([1, 0, 0]), np.array([0, 0, 1]), "int_carpet"))
        return ComponentResult(_finish(H.named(m, "inner_wheelhouse"), ctx))


@register
class BedLiner(CarComponent):
    name = "floor.bedliner"
    accepts = (RectangleConnector,)
    default_for = ("bed",)
    priority = 1
    options = {"rib_pitch": 0.10, "rib_height": 0.012}
    description = "ribbed dark plastic pickup bed liner"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        Lx, Wy = conn.width, conn.height
        pitch, rh = float(opts["rib_pitch"]), float(opts["rib_height"])
        n_ribs = max(3, int(Wy / pitch))
        xs = np.linspace(-Lx / 2, Lx / 2, 6)
        ys = np.linspace(-Wy / 2, Wy / 2, n_ribs * 4 + 1)

        def z_top(X, Y):
            return 0.01 + rh * 0.5 * (1 + np.cos(2 * np.pi * Y / (Wy / n_ribs)))

        m = H.heightfield_slab(xs, ys, z_top, -0.005, "int_rubber", name="bed_liner")
        return ComponentResult(_finish(m, ctx))


# =============================================================== HEADLINER
@register
class Headliner(CarComponent):
    """Light fabric ceiling following the roof (Coons patch), sun visors near
    the header; emits `dome_light` and `grab_handle_L/R` connectors."""

    name = "headliner.fabric"
    accepts = (PolygonConnector,)
    default_for = ("headliner",)
    options = {"color": None, "visors": True}
    description = "fabric headliner with sun visors; emits dome light and grab handle connectors"

    def build_local(self, conn: PolygonConnector, opts, ctx) -> ComponentResult:
        w, h = conn.width, conn.height
        res_u = int(np.clip(w / 0.12, 8, 24))
        res_v = int(np.clip(h / 0.12, 8, 16))
        grid = H.connector_patch(conn, res_u, res_v)
        m = H.patch_surface(grid, conn.frame.z_axis, "int_fabric", offset=0.0, thickness=0.008, name="headliner")
        m.transform(np.linalg.inv(conn.frame.matrix))
        lp = conn.local_points()
        x_min, x_max = lp[:, 0].min(), lp[:, 0].max()
        y_half = 0.5 * (lp[:, 1].max() - lp[:, 1].min())
        y_mid = 0.5 * (lp[:, 1].max() + lp[:, 1].min())
        subs: List[Connector] = []
        x_dome = x_max - min(0.32, 0.3 * (x_max - x_min))
        z_dome = H.grid_local_z(grid, conn.frame, x_dome, y_mid)
        subs.append(CircleConnector("dome_light", Frame.from_normal([x_dome, y_mid, z_dome], [0, 0, 1], x_hint=(1, 0, 0)), 0.06, tags=["dome_light"]))
        for tag, s in (("L", -1.0), ("R", 1.0)):  # local +y is car-right (frame z points down)
            y = y_mid + s * (y_half - 0.10)
            x = 0.5 * (x_min + x_max)
            z = H.grid_local_z(grid, conn.frame, x, y)
            subs.append(PointConnector(f"grab_handle_{tag}", Frame.from_normal([x, y, z], [0, 0, 1], x_hint=(1, 0, 0)),
                                       tags=["grab_handle"], meta={"side": "left" if s < 0 else "right"}))
        if opts.get("visors", True):
            for s in (-1.0, 1.0):
                y = y_mid + s * min(0.40, y_half - 0.20)
                x = x_max - 0.17
                z = H.grid_local_z(grid, conn.frame, x, y)
                visor = P.rounded_box(0.14, 0.30, 0.012, 0.02, material="int_fabric", center=(x, y, z + 0.010), name="visor")
                visor.merge(P.rounded_box(0.075, 0.18, 0.002, 0.008, material="int_plastic", center=(x, y, z + 0.017)))
                visor.merge(H.named(P.box(0.060, 0.16, 0.001, material="int_mirror", center=(x, y, z + 0.0185)),
                                    "vanity_mirror_L" if s < 0 else "vanity_mirror_R"))
                m.merge(H.named(visor, "visor_L" if s < 0 else "visor_R"))
                for dy in (-0.11, 0.11):
                    m.merge(P.box(0.025, 0.025, 0.026, material="int_plastic", center=(x + 0.065, y + dy, z + 0.012), name="visor_clip"))
        extra = {}
        if opts.get("color"):
            extra["int_fabric"] = ctx.material("int_fabric", opts["color"], shininess=0.05)
        return ComponentResult(_finish(m, ctx, extra), subs)


@register
class DomeLight(CarComponent):
    name = "light.dome"
    accepts = (CircleConnector,)
    default_for = ("dome_light",)
    options = {}
    description = "dome light: housing ring, warm lens and two switch buttons"

    def build_local(self, conn: CircleConnector, opts, ctx) -> ComponentResult:
        r = conn.radius
        m = P.tube(r + 0.02, r - 0.002, 0.014, 24, material="int_fabric", center=(0, 0, 0.007), name="dome_housing")
        m.merge(P.cylinder(r, 0.006, 24, material="int_lens", center=(0, 0, 0.011), radius_top=r * 0.9, name="dome_lens"))
        m.merge(H.button_row(2, 0.03, (0.018, 0.012), 0.005, "int_plastic", center=(0, -(r + 0.008), 0.014), along="x"))
        return ComponentResult(_finish(m, ctx))


@register
class GrabHandle(CarComponent):
    name = "handle.grab"
    accepts = (PointConnector,)
    default_for = ("grab_handle",)
    options = {"length": 0.15}
    description = "roof grab handle: bar on two short posts"

    def build_local(self, conn: PointConnector, opts, ctx) -> ComponentResult:
        L = float(opts["length"])
        m = P.rounded_box(L, 0.026, 0.02, 0.008, material="int_fabric", center=(0, 0, 0.045), name="grab_bar")
        for i, x in enumerate((-L / 2 + 0.02, L / 2 - 0.02)):
            m.merge(H.named(P.rounded_box(0.045, 0.038, 0.01, 0.010, material="int_plastic", center=(x, 0, 0.005)), f"grab_base_{i}"))
            m.merge(P.box(0.03, 0.026, 0.04, material="int_fabric", center=(x, 0, 0.02), name="grab_post"))
        return ComponentResult(_finish(m, ctx))


# =============================================================== DOOR CARDS
@register
class DoorCard(CarComponent):
    """Door trim panel following the door aperture: bordered Coons patch with
    upper / insert / lower zones, padded armrest with window switches, door
    pull and release lever, map pocket (front doors); emits a `speaker`."""

    name = "door.card"
    accepts = (PolygonConnector,)
    default_for = ("door_card",)
    options = {"border": 1, "pocket": True}
    description = "door card with armrest, pull, release lever, switches and map pocket; emits a speaker connector"

    def build_local(self, conn: PolygonConnector, opts, ctx) -> ComponentResult:
        fr = conn.frame
        pts = conn.points.copy()
        # the body's front-door outline runs forward of the firewall (to the arch seam); clip it
        x_lim = ctx.measurements.get("x_cowl")
        clip_x = None if x_lim is None else float(x_lim) + 0.02
        if clip_x is not None:
            pts[:, 0] = np.minimum(pts[:, 0], clip_x)
        lp = fr.to_local(pts)
        x_min, x_max = float(lp[:, 0].min()), float(lp[:, 0].max())
        span = x_max - x_min
        res_u = int(np.clip(span / 0.10, 8, 24))
        res_v = int(np.clip(conn.height / 0.08, 6, 12))
        grid = H.connector_patch(conn, res_u, res_v, clip_x_max=clip_x)
        m = H.patch_surface(grid, fr.z_axis, "int_plastic", offset=0.0, thickness=0.02, name="door_card")
        belt = float(conn.meta.get("belt_z", pts[:, 2].max()))
        z_bot = float(pts[:, 2].min())
        # zones by world height, border by grid position
        cent = m.face_centroids()
        nb = int(opts.get("border", 1))
        n_front = (res_u - 1) * (res_v - 1)
        for i in range(m.n_faces):
            z = cent[i, 2]
            if i < n_front:
                iu, iv = divmod(i, res_v - 1)
                if iu < nb or iu >= res_u - 1 - nb or iv < nb or iv >= res_v - 1 - nb:
                    m.face_materials[i] = "int_trim"
                    continue
            if z > belt - 0.09:
                m.face_materials[i] = "int_plastic"
            elif z > belt - 0.42:
                m.face_materials[i] = "int_accent"
            else:
                m.face_materials[i] = "int_plastic"
        m.transform(np.linalg.inv(fr.matrix))
        # ---- local placement helpers (local: x forward, z inward; y is ±up)
        ey_z = float(fr.y_axis[2])
        sgn = 1.0 if ey_z >= 0 else -1.0
        oz = float(fr.origin[2])

        def yl(zw: float) -> float:
            return (zw - oz) / ey_z

        def zs(x: float, y: float) -> float:
            return H.grid_local_z(grid, fr, x, y)

        La = float(np.clip(0.42 * span, 0.28, 0.55))
        x_a = x_min + 0.45 * span
        y_a = yl(belt - 0.25 - 0.03)
        z_a = zs(x_a, y_a)
        arm = P.rounded_box(La, 0.05, 0.085, 0.02, material="int_accent", center=(x_a, y_a, z_a + 0.04), name="armrest")
        arm.merge(P.box(La + 0.05, 0.035, 0.055, material="int_plastic", center=(x_a, y_a - sgn * 0.04, z_a + 0.027), name="armrest_base"))
        m.merge(arm)
        # window switches on top of the armrest
        y_top = y_a + sgn * 0.025
        m.merge(P.box(0.11, 0.006, 0.045, material="int_gauge", center=(x_a + La * 0.05, y_top + sgn * 0.003, z_a + 0.045), name="switch_panel"))
        n_sw = 4 if span > 1.0 else 2
        for k in range(n_sw):
            xb = x_a + La * 0.05 + (k - (n_sw - 1) / 2) * 0.022
            m.merge(P.box(0.016, 0.008, 0.014, material="int_plastic", center=(xb, y_top + sgn * 0.007, z_a + 0.045), name="switch"))
        # door pull (grab handle at the front of the armrest)
        x_p = x_a + La / 2 - 0.09
        m.merge(P.box(0.15, 0.05, 0.004, material="int_gauge", center=(x_p, y_a + sgn * 0.045, z_a + 0.002), name="pull_recess"))
        m.merge(P.rounded_box(0.13, 0.028, 0.03, 0.01, material="int_plastic", center=(x_p, y_a + sgn * 0.052, z_a + 0.045), name="pull"))
        # release lever forward/up of the armrest
        x_l = min(x_a + La / 2 + 0.10, x_max - 0.10)
        y_l = yl(belt - 0.13)
        z_l = zs(x_l, y_l)
        m.merge(P.rounded_box(0.12, 0.045, 0.004, 0.015, material="int_gauge", center=(x_l, y_l, z_l + 0.002), name="lever_recess"))
        m.merge(P.rounded_box(0.09, 0.018, 0.012, 0.006, material="int_chrome", center=(x_l - 0.005, y_l, z_l + 0.01), name="lever"))
        # Open map pocket: front wall, bottom and end cheeks; no top cap.
        if opts.get("pocket", True) and span > 0.45:
            width = min(0.32, span * 0.32)
            x_k = x_min + width / 2 + 0.08
            y_k = yl(z_bot + 0.10)
            z_k = zs(x_k, y_k)
            pocket = P.rounded_box(width, 0.10, 0.008, 0.014, material="int_plastic", center=(x_k, y_k, z_k + 0.060))
            pocket.merge(P.box(width, 0.008, 0.06, material="int_plastic", center=(x_k, y_k - sgn * 0.046, z_k + 0.030)))
            for sign in (-1, 1):
                pocket.merge(P.box(0.008, 0.10, 0.06, material="int_plastic", center=(x_k + sign * (width / 2 - 0.004), y_k, z_k + 0.030)))
            pocket.merge(P.box(width - 0.016, 0.002, 0.046, material="int_gauge", center=(x_k, y_k - sgn * 0.039, z_k + 0.031)))
            m.merge(H.named(pocket, "map_pocket"))
        # Sill return closes the carpet-to-door gap; the metal scuff plate is
        # horizontal in world space even on the right-hand door's inverted Y.
        floor_z = ctx.measurements.get("z_floor", z_bot - 0.15) + 0.05
        sill_h = max(z_bot - floor_z, 0.02)
        xc = (x_min + x_max) / 2
        z_sill = zs(xc, yl(z_bot))
        m.merge(H.named(P.box(span, sill_h, 0.07, material="int_plastic",
                              center=(xc, yl(z_bot - sill_h / 2), z_sill + 0.025)), "sill_return"))
        plate = P.box(min(0.65, span * 0.8), 0.004, 0.065, material="int_metal",
                      center=(xc, yl(z_bot + 0.005), z_sill + 0.030))
        m.merge(H.named(plate, "sill_scuff_plate"))
        # speaker connector low and forward, clear of the map-pocket opening
        x_sp = x_max - 0.15
        z_w = max(belt - 0.48, z_bot + 0.12)
        y_sp = yl(z_w)
        sp = CircleConnector("speaker", Frame.from_normal([x_sp, y_sp, zs(x_sp, y_sp) + 0.004], [0, 0, 1], x_hint=(1, 0, 0)), 0.08,
                             tags=["speaker"], meta={"side": conn.meta.get("side"), "door": conn.meta.get("door")})
        return ComponentResult(_finish(m, ctx), [sp], {"span": span})


@register
class RoundSpeaker(CarComponent):
    name = "speaker.round"
    accepts = (CircleConnector,)
    default_for = ("speaker",)
    options = {}
    description = "round speaker grille with a trim ring"

    def build_local(self, conn: CircleConnector, opts, ctx) -> ComponentResult:
        m = H.grille(conn.radius, "speaker_grille")
        return ComponentResult(_finish(m, ctx))


# =============================================================== REAR-VIEW MIRROR
@register
class RearviewMirror(CarComponent):
    """Stalk from the windshield header and a mirror slab facing the driver.
    Local: +Z rearward, +X car-left, +Y down."""

    name = "mirror.rearview"
    accepts = (PointConnector,)
    default_for = ("rearview_mirror",)
    options = {"width": 0.24, "height": 0.07, "fit_header": "auto"}
    description = "interior rear-view mirror on a stalk"

    def build_local(self, conn: PointConnector, opts, ctx) -> ComponentResult:
        w, h = float(opts["width"]), float(opts["height"])
        base = P.cylinder(0.02, 0.012, 16, material="int_plastic", center=(0, 0, 0.006), name="mirror_base")
        stalk = P.cylinder(0.009, 0.10, 10, material="int_plastic", center=(0, 0, 0.05), name="stalk")
        stalk.transform(H.rot_x(-np.radians(40))).translate((0, 0.0, 0.01))   # heads down (+y) and rearward (+z)
        m = base.merge(stalk)
        cy, cz = 0.10 * np.sin(np.radians(40)) + 0.02, 0.10 * np.cos(np.radians(40)) + 0.02
        m.merge(P.rounded_box(w, h, 0.025, 0.02, material="int_plastic", center=(0, cy, cz), name="mirror_housing"))
        m.merge(P.box(w - 0.02, h - 0.012, 0.003, material="int_mirror", center=(0, cy, cz + 0.014), name="mirror_glass"))
        # Fit only the legacy body root; explicit/custom mounts still follow
        # their connector.  fit_header=False also preserves the old root frame.
        fit = opts["fit_header"]
        if fit == "auto":
            fit = (conn.name == "rearview_mirror" and conn.owner == "body"
                   and "x_roof_front" in ctx.measurements and "z_roof_front" in ctx.measurements
                   and conn.origin[0] < ctx.measurements["x_roof_front"] - 0.35)
        mount = conn.origin.copy()
        if fit:
            mount = np.array([ctx.measurements["x_roof_front"] + 0.015, 0, ctx.measurements["z_roof_front"] - 0.055])
            m.translate(conn.frame.to_local(mount)[0])
        return ComponentResult(_finish(m, ctx), info={"header_mount_world": mount.tolist(), "fitted_header": bool(fit)})
