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
from .fills import fill_connector, coons_patch


def _part(mesh, part, name):
    """Keep detail groups/zones addressable for inspection and export."""
    part.add_group(name, range(part.n_vertices))
    part.add_zone(name, range(part.n_faces))
    mesh.merge(part)
    return mesh


def _surround(width, height, band, depth, material, z=0.0):
    """An open rounded rectangular bezel, not overlapping solid plates."""
    band = min(band, width * .18, height * .18)
    radius = min(.035, height * .24, width * .24)
    outer = rounded_rect_points(width, height, radius, 4)
    inner = rounded_rect_points(width - 2 * band, height - 2 * band,
                                max(.001, radius - band), 4)
    rings = [np.column_stack([p, np.full(len(p), h)]) for p, h in
             ((inner, z), (outer, z), (outer, z + depth), (inner, z + depth), (inner, z))]
    return P.loft(rings, material=material)


def _pipe(path, radius, material, closed=False, sides=8):
    m = P.sweep_profile(np.asarray(path), circle_points(radius, sides), material=material,
                        closed_path=closed)
    # sweep_profile's (binormal, normal) basis has negative handedness.
    return m.flip_normals()


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
    options = {"spokes": 5, "spoke_width": 0.55, "hubcap": True, "tread_grooves": 3,
               "tread_blocks": 40, "dish": 0.04, "caliper_color": "#b52d22"}
    description = "tyre with alloy rim; radius/width/rim size fitted from the hub connector"
    modifier_specs = [
        ModifierSpec("radius", 0.10, 0.14, "size", "tyre radius"),
        ModifierSpec("width", 0.08, 0.12, "size", "tyre width"),
        ModifierSpec("rim_ratio", 0.15, 0.20, "style", "low-profile <-> balloon tyre"),
        ModifierSpec("dish", 0.03, 0.05, "style", "spoke dish depth"),
    ]

    N = 48

    def generate(self, p: WheelParams) -> Mesh:
        # Only the carcass, barrel, lips and hub belong to the target library.
        # Count-dependent tread/spokes/hardware are fitted AFTER morphing it.
        R, w, rr = p.radius, p.width, p.rim_ratio * p.radius
        gap = R - rr
        profile = np.array([
            [rr + .004, -w / 2 + .016], [rr + gap * .18, -w / 2 + .006],
            [rr + gap * .60, -w / 2], [R - .025, -w / 2 + .009],
            [R - .008, -w * .37], [R - .007, -w * .32],
            [R - .007, w * .32], [R - .008, w * .37],
            [R - .025, w / 2 - .009], [rr + gap * .60, w / 2],
            [rr + gap * .18, w / 2 - .006], [rr + .004, w / 2 - .016],
        ])
        m = Mesh(name="wheel")
        _part(m, P.revolve(profile, self.N, material="tyre"), "carcass")
        _part(m, P.tube(rr + .004, rr - .012, w - .035, self.N, material="rim"), "barrel")
        for side in (-1, 1):
            _part(m, P.torus(rr + .004, .007, self.N, 6, material="rim",
                            center=(0, 0, side * (w / 2 - .019))), f"lip_{side}")
        zf = w / 2 - .024 - p.dish
        _part(m, P.cylinder(.052, .028, 24, material="rim_dark", center=(0, 0, zf)), "face_ring")
        return m

    def fit(self, conn: CircleConnector, opts, ctx) -> Dict[str, float]:
        v = {"radius": self.value_for("radius", conn.radius)}
        tw = conn.meta.get("tire_width")
        if tw:
            v["width"] = self.value_for("width", tw)
        # bigger wheels get lower profile tyres
        v["rim_ratio"] = self.value_for("rim_ratio", float(opts.get("rim_ratio", 0.62 + (conn.radius - 0.33) * 0.6)))
        v["dish"] = self.value_for("dish", float(opts["dish"]))
        return v

    def materials(self, ctx, opts):
        return {
            "tyre": ctx.material("tyre", "#1a1a1c", shininess=0.08),
            "rim": ctx.material("rim", opts.get("rim_color", ctx.palette.rim), shininess=0.7, metallic=0.8),
            "rim_dark": ctx.material("rim_dark", "#3a3c40", shininess=0.4),
            "brake_disc": ctx.material("brake_disc", "#777a7d", metallic=.8, shininess=.5),
            "caliper": ctx.material("caliper", opts["caliper_color"], shininess=.5),
        }

    def build_local(self, conn, opts, ctx) -> ComponentResult:
        res = super().build_local(conn, opts, ctx)
        p = self.canonical()
        for key, value in res.info["modifier_values"].items():
            spec = self.spec(key)
            setattr(p, key, getattr(p, key) + value * (spec.delta_plus if value >= 0 else spec.delta_minus))
        # Differential targets add independently; use the same fitted rim radius
        # as the library (rather than introducing a radius*ratio cross-term).
        rr = .33 * p.rim_ratio + (p.radius - .33) * .62
        self._tread(res.mesh, p, opts)
        self._wheel_face(res.mesh, p, rr, opts)
        self._hardware(res.mesh, p, rr, opts)
        res.info.update({"tread_blocks": int(np.clip(opts["tread_blocks"], 0, 64)),
                         "spokes": int(np.clip(opts["spokes"], 3, 12)), "rim_radius": rr})
        return res

    def _tread(self, m, p, opts):
        blocks = int(np.clip(opts["tread_blocks"], 0, 64))
        grooves = int(np.clip(opts["tread_grooves"], 0, 4))
        lane = p.width * .70 / (grooves + 1)
        for j in range(grooves + 1):
            z = -p.width * .35 + (j + .5) * lane
            half = (lane - min(.006, lane * .18)) / 2
            prof = [[p.radius - .007, z - half], [p.radius - .003, z - half],
                    [p.radius - .003, z + half], [p.radius - .007, z + half]]
            _part(m, P.revolve(prof, self.N, material="tyre"), f"tread_rib_{j}")
            for k in range(blocks):
                a = 2 * np.pi * (k + .5 * (j % 2)) / blocks
                da = 2 * np.pi / blocks * .40
                skew = .018 / p.radius * (1 if j % 2 else -1)
                # Four-sided angled tread blocks, with an actual 4mm valley.
                angles = np.array([a - da - skew, a + da - skew, a + da + skew, a - da + skew])
                zz = np.array([z - half, z - half, z + half, z + half])
                rings = [np.column_stack([r * np.cos(angles), r * np.sin(angles), zz])
                         for r in (p.radius - .004, p.radius)]
                block = P.loft(rings, cap_start=False, cap_end=False, material="tyre")
                block.faces += [(3, 2, 1, 0), (4, 5, 6, 7)]
                block.face_materials += ["tyre", "tyre"]
                _part(m, block, f"tread_block_{j}_{k}")

    def _wheel_face(self, m, p, rr, opts):
        n = int(np.clip(opts["spokes"], 3, 12))
        width = float(np.clip(opts["spoke_width"], .2, .85))
        zf = p.width / 2 - .024 - p.dish
        for k in range(n):
            rings = []
            for r, sweep, z in ((.042, -.02, zf), (rr * .52, .04, zf - .008),
                                (rr - .008, .075, p.width / 2 - .028)):
                half = min(.026, r * np.pi / n * width)
                # A radial loft with a 22mm section and swept outer attachment.
                rings.append(np.array([[r, -half + sweep * r, z - .012],
                                       [r, half + sweep * r, z - .012],
                                       [r, half + sweep * r, z + .010],
                                       [r, -half + sweep * r, z + .010]]))
            spoke = P.loft(rings, cap_start=True, cap_end=True, material="rim")
            a = k * 2 * np.pi / n
            spoke.apply_frame(Frame.from_normal([0, 0, 0], [0, 0, 1], [np.cos(a), np.sin(a), 0]))
            _part(m, spoke, f"spoke_{k}")

    def _hardware(self, m, p, rr, opts):
        zf = p.width / 2 - .024 - p.dish
        # Disc and caliper stay behind even the deepest part of each spoke.
        disc_z = zf - .033
        _part(m, P.tube(rr * .79, .046, .014, self.N, material="brake_disc",
                        center=(0, 0, disc_z)), "brake_disc")
        _part(m, P.tube(.070, .032, .026, 24, material="rim_dark",
                        center=(0, 0, disc_z - .001)), "disc_hat")
        for k in range(16):
            a = k * 2 * np.pi / 16
            # Radial ventilation slots on the disc's edge, not painted spoke gaps.
            slot = P.box(.010, .018, .010, material="rim_dark",
                         center=(rr * .75, 0, disc_z))
            slot.apply_frame(Frame.from_normal([0, 0, 0], [0, 0, 1], [np.cos(a), np.sin(a), 0]))
            _part(m, slot, f"disc_vent_{k}")
        # Forged caliper follows the disc arc rather than reading as a red box.
        section = np.array([[-.019, -.016], [.017, -.016], [.025, -.008], [.025, .008],
                            [.017, .016], [-.019, .016], [-.025, .008], [-.025, -.008]])
        rings = [np.column_stack([(rr * .77 + section[:, 0]) * np.cos(a),
                                  (rr * .77 + section[:, 0]) * np.sin(a),
                                  disc_z - .008 + section[:, 1]])
                 for a in np.linspace(np.pi - .42, np.pi + .42, 9)]
        caliper = P.loft(rings, cap_start=True, cap_end=True, material="caliper")
        if P.signed_volume(caliper) < 0:
            caliper.flip_normals()
        _part(m, caliper, "caliper")
        for k in range(5):
            a = k * 2 * np.pi / 5
            _part(m, P.cylinder(.007, .012, 6, material="rim", center=(.038 * np.cos(a), .038 * np.sin(a), zf + .019)),
                  f"lug_{k}")
        if opts["hubcap"]:
            _part(m, P.cylinder(.025, .009, 24, material="rim", center=(0, 0, zf + .022)), "centre_cap")
            _part(m, P.cylinder(.016, .003, 16, material="rim_dark", center=(0, 0, zf + .028)), "cap_badge")
        _part(m, P.cylinder(.004, .024, 8, material="rim_dark",
                            center=(rr * .70, -rr * .60, p.width / 2 - .022)), "valve_stem")


@register
class SteelWheel(Wheel):
    name = "wheel.steel"
    default_for = ()
    description = "stamped steel wheel with open ventilation slots and a domed hubcap"

    def _wheel_face(self, m, p, rr, opts):
        zf = p.width / 2 - .024 - p.dish
        # Pressed dish: annular inner web plus separated outer webs. The gaps
        # between them are ventilation holes, open all the way to the brake.
        profile = [[.045, zf], [rr * .52, zf], [rr * .55, zf - .006],
                   [rr * .55, zf - .015], [.045, zf - .015], [.045, zf]]
        _part(m, P.revolve(profile, self.N, material="rim"), "steel_dish")
        for k in range(12):
            a = k * 2 * np.pi / 12
            section = [[rr * .52, zf - .010], [rr - .006, p.width / 2 - .035],
                       [rr - .006, p.width / 2 - .024], [rr * .52, zf], [rr * .52, zf - .010]]
            web = P.revolve(section, 4, angle=2 * np.pi / 12 * .55, material="rim")
            # Cap the two exposed ends of each stamped web.
            for ids in (list(range(5)), list(range(web.n_vertices - 5, web.n_vertices))):
                web.faces.append(tuple(ids[:4]) if ids[0] == 0 else tuple(ids[:4][::-1]))
                web.face_materials.append("rim")
            web.apply_frame(Frame.from_normal([0, 0, 0], [0, 0, 1], [np.cos(a), np.sin(a), 0]))
            _part(m, web, f"steel_web_{k}")
        if opts["hubcap"]:
            rings = [np.column_stack([circle_points(radius, 32), np.full(32, z)])
                     for radius, z in ((rr * .46, zf + .004), (rr * .43, zf + .020),
                                       (rr * .30, zf + .035), (.018, zf + .041))]
            cap = P.loft(rings, cap_end=True, material="rim")
            _part(m, cap, "steel_hubcap")


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
        if conn.name.startswith("glass_"):
            # Separate 8mm rubber moulding follows the actual aperture boundary.
            trim = _pipe(conn.points + conn.normal * .001, .004, "glass_frame", closed=True, sides=6)
            _part(m, trim, "window_moulding")
        m.materials.update(mats)
        m.transform(np.linalg.inv(conn.frame.matrix))  # to local; build() maps back to world
        return ComponentResult(m, [], {"area": conn.area})


# =============================================================== LIGHTS
class _LampSurface:
    """Decorations in aperture UVs, following the exact (possibly warped) grid.

    U runs inboard -> outboard; V runs bottom -> top in WORLD space. Neither
    handedness nor the lamp's steep corner rake is guessed from local Y.
    """
    def __init__(self, conn):
        grid = conn.meta.get("grid_points")
        shape = conn.meta.get("grid")
        if grid is None and shape and len(conn.points) == 2 * (shape[0] + shape[1] - 2):
            grid = coons_patch(conn.points, *shape, max(shape[0], 3), max(shape[1], 3))
        if grid is None:
            # Generic convex polygon: an inscribed square for the internals;
            # the pane itself still uses fill_connector's full outline.
            lp = conn.local_points()
            c = lp.mean(axis=0)
            edges = np.roll(lp[:, :2], -1, axis=0) - lp[:, :2]
            distances = np.abs(np.cross(edges, c[:2] - lp[:, :2])) / np.maximum(np.linalg.norm(edges, axis=1), 1e-9)
            r = distances.min() * .65
            grid = conn.frame.to_world(np.array([[[c[0] - r, c[1] - r, c[2]], [c[0] - r, c[1] + r, c[2]]],
                                                [[c[0] + r, c[1] - r, c[2]], [c[0] + r, c[1] + r, c[2]]]]))
        grid = np.asarray(grid, dtype=float)
        lengths = [np.linalg.norm(np.diff(grid, axis=i), axis=2).sum(axis=i).mean() for i in (0, 1)]
        if lengths[1] > lengths[0]:
            grid = grid.transpose(1, 0, 2)
        if grid[:, -1, 2].mean() < grid[:, 0, 2].mean():
            grid = grid[:, ::-1]
        if abs(grid[-1, :, 1].mean()) < abs(grid[0, :, 1].mean()):
            grid = grid[::-1]
        self.grid = conn.frame.to_local(grid.reshape(-1, 3)).reshape(grid.shape)
        du = np.linalg.norm(np.diff(grid, axis=0), axis=2).mean(axis=1)
        dv = np.linalg.norm(np.diff(grid, axis=1), axis=2).mean(axis=0)
        self.width, self.height = du.sum(), dv.sum()
        self.knots = (np.r_[0., np.cumsum(du)] / self.width, np.r_[0., np.cumsum(dv)] / self.height)

    def at(self, uv, depth=0.0):
        uv = np.asarray(uv, dtype=float)
        shape = uv.shape[:-1]
        # Body grid indices are semantic, not uniformly spaced. Arc-length
        # coordinates keep projectors on the vertical fascia rather than
        # crowding them onto the densely sampled hood-side edge of the lens.
        flat = uv.reshape(-1, 2)
        ij = np.column_stack([np.interp(flat[:, k], self.knots[k], np.arange(self.grid.shape[k])) for k in (0, 1)])
        base = np.minimum(ij.astype(int), np.array(self.grid.shape[:2]) - 2)
        f = ij - base
        i, j = base.T
        u, v = f[:, 0:1], f[:, 1:2]
        pts = ((1 - u) * (1 - v) * self.grid[i, j] + u * (1 - v) * self.grid[i + 1, j]
               + (1 - u) * v * self.grid[i, j + 1] + u * v * self.grid[i + 1, j + 1])
        pts[:, 2] += depth
        return pts.reshape(*shape, 3)

    def ribbon(self, uv, width, material, depth=-.008, closed=False):
        uv = np.asarray(uv, dtype=float)
        dense = []
        ends = np.roll(uv, -1, axis=0) if closed else uv[1:]
        for a, b in zip(uv, ends):
            count = max(1, int(np.ceil(np.linalg.norm((b - a) * [self.width, self.height]) / .008)))
            dense.extend(a + (b - a) * t for t in np.linspace(0, 1, count, endpoint=False))
        if not closed:
            dense.append(uv[-1])
        return _pipe(self.at(dense, depth), width / 2, material, closed, sides=6)

    def patch(self, u0, u1, v0, v1, material, depth=-.014):
        from .fills import grid_mesh
        u, v = np.meshgrid(np.linspace(u0, u1, 9), np.linspace(v0, v1, 4), indexing="ij")
        mesh = grid_mesh(self.at(np.stack([u, v], axis=-1), depth), material)
        if mesh.face_normals()[:, 2].mean() < 0:
            mesh.flip_normals()
        return mesh

    def border(self, material, depth=-.005):
        uv = np.array([(u, .035) for u in np.linspace(.025, .975, 17)]
                      + [(.975, v) for v in np.linspace(.035, .965, 9)[1:]]
                      + [(u, .965) for u in np.linspace(.975, .025, 17)[1:]]
                      + [(.025, v) for v in np.linspace(.965, .035, 9)[1:-1]])
        return self.ribbon(uv, min(.010, self.height * .08), material, depth, True)

    def projector(self, u, radius):
        a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        rings = []
        for size, depth in ((.30, -.048), (.62, -.041), (.88, -.026), (1., -.017), (1.04, -.012)):
            uv = np.column_stack([u + radius / self.width * size * np.cos(a),
                                  .49 + radius / self.height * size * np.sin(a)])
            rings.append(self.at(uv, depth))
        bowl = P.loft(rings, material="bezel")
        if bowl.face_normals()[:, 2].mean() < 0:
            bowl.flip_normals()
        # Small convex lens within the reflector, with no collapsed pole quads.
        uv = np.column_stack([u + radius / self.width * .62 * np.cos(a),
                              .49 + radius / self.height * .62 * np.sin(a)])
        edge = self.at(uv, -.019)
        center = self.at([u, .49], -.011)
        lens = Mesh(np.vstack([edge, center]), [(k, (k + 1) % 24, 24) for k in range(24)], ["projector_glass"] * 24)
        if lens.face_normals()[:, 2].mean() < 0:
            lens.flip_normals()
        return bowl.merge(lens)


@register
class Headlight(CarComponent):
    name = "light.headlight"
    accepts = (PolygonConnector,)
    default_for = ("headlight",)
    options = {"lens_alpha": 0.28, "projector": True}
    description = "twin projector bowls, upper DRL and outboard indicator under a curved clear pane"

    def build_local(self, conn: PolygonConnector, opts, ctx) -> ComponentResult:
        mats = {
            "lens": Material("lens", (0.82, 0.88, 0.92), float(opts["lens_alpha"]), 0.95),
            "bezel": ctx.material("bezel", "#b9c3ce", shininess=0.8, metallic=0.9),
            "lamp_dark": ctx.material("lamp_dark", "#12161d", shininess=0.4),
            "lamp_led": Material("lamp_led", (0.95, 0.97, 1.0), 1.0, 0.5, emissive=0.65),
            "projector_glass": ctx.material("projector_glass", "#8caac1", shininess=.95),
            "indicator": ctx.material("indicator", "#ee8c14", emissive=.3, shininess=.65),
        }
        inv = np.linalg.inv(conn.frame.matrix)
        m = Mesh(name="headlight")
        _part(m, fill_connector(conn, "lamp_dark", offset=-.056).transform(inv), "housing")
        surf = _LampSurface(conn)
        _part(m, surf.border("bezel"), "bezel")
        radius = min(.052, surf.height * .27, surf.width * .15)
        for k, u in enumerate((.28, .61) if opts["projector"] else (.37,)):
            _part(m, surf.projector(u, radius), f"projector_{k}")
        if not opts["projector"]:
            _part(m, surf.patch(.58, .78, .35, .60, "lamp_led"), "led_bar")
        uv = np.column_stack([np.linspace(.08, .84, 25), np.full(25, .85)])
        _part(m, surf.ribbon(uv, min(.008, surf.height * .065), "lamp_led"), "drl")
        _part(m, surf.patch(.86, .95, .18, .77, "indicator", -.010), "indicator")
        _part(m, fill_connector(conn, "lens", offset=.001, bulge=.008, upsample=3).transform(inv), "lens")
        m.materials.update(mats)
        return ComponentResult(m, [], {"projectors": 2 if opts["projector"] else 1})


@register
class Taillight(CarComponent):
    name = "light.taillight"
    accepts = (PolygonConnector,)
    default_for = ("taillight",)
    options = {"lens_alpha": 0.32}
    description = "curved tail lamp with light-guide ring, clear reverse segment and painted bezel"

    def build_local(self, conn: PolygonConnector, opts, ctx) -> ComponentResult:
        mats = {
            "tail_lens": Material("tail_lens", Material.parse_color(ctx.palette.taillight), float(opts["lens_alpha"]), .95),
            "tail_guide": ctx.material("tail_guide", "#f12837", emissive=.6, shininess=.7),
            "tail_reflector": ctx.material("tail_reflector", "#b92231", shininess=.8),
            "tail_reverse": ctx.material("tail_reverse", "#e5e7ee", shininess=.8),
            "tail_bezel": ctx.material("tail_bezel", ctx.palette.paint, shininess=.85),
            "lamp_dark": ctx.material("lamp_dark", "#17131a", shininess=.4),
        }
        inv = np.linalg.inv(conn.frame.matrix)
        m = Mesh(name="taillight")
        _part(m, fill_connector(conn, "lamp_dark", offset=-.038).transform(inv), "housing")
        surf = _LampSurface(conn)
        _part(m, surf.border("tail_bezel", .001), "bezel")
        uv = rounded_rect_points(.78, .54, .10, 6) + [.5, .57]
        _part(m, surf.ribbon(uv, min(.012, surf.height * .08), "tail_guide", -.012, True), "light_guide")
        _part(m, surf.patch(.22, .55, .41, .57, "tail_reverse", -.015), "reverse")
        _part(m, surf.patch(.12, .88, .12, .23, "tail_reflector", -.009), "reflex_strip")
        # Small prismatic flutes are geometry, rather than a single red patch.
        for u in np.linspace(.14, .86, 19):
            _part(m, surf.ribbon([[u, .135], [u, .215]], .0025, "tail_reflector", -.007), f"reflex_flute_{u:.2f}")
        _part(m, fill_connector(conn, "tail_lens", offset=.001, bulge=.006, upsample=3).transform(inv), "lens")
        m.materials.update(mats)
        return ComponentResult(m)


# =============================================================== GRILLE / INTAKE / PLATE / BADGE
@register
class Grille(CarComponent):
    name = "grille.slats"
    accepts = (RectangleConnector,)
    default_for = ("grille",)
    options = {"slats": 5, "frame": True, "mesh": False, "badge": True}
    description = "deep horizontal slats interrupted around an inset roundel, with an open surround"
    pattern = "slats"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        w, h = conn.width, conn.height
        band = min(.012, h * .09)
        mats = {"grille_dark": ctx.material("grille_dark", "#101317", shininess=.2),
                "mesh_dark": ctx.material("mesh_dark", "#42484d", shininess=.5, metallic=.5),
                "chrome": ctx.material("chrome", ctx.palette.chrome, shininess=.85, metallic=.9)}
        m = Mesh(name="grille")
        _part(m, P.rounded_box(w, h, .010, min(.03, h * .2), material="grille_dark", center=(0, 0, -.022)), "back")
        if opts["frame"]:
            _part(m, _surround(w, h, band, .022, "chrome", -.012), "surround")
        iw, ih = w - 2 * band, h - 2 * band
        pattern = "mesh" if opts.get("mesh") else self.pattern
        badge_r = min(.040, h * .29) if opts.get("badge") and pattern == "slats" else 0
        if pattern == "slats":
            n = int(np.clip(opts["slats"], 1, 14))
            for k in range(n):
                y = -ih / 2 + (k + .5) * ih / n
                thick = min(.012, ih / n * .32)
                cut = np.sqrt(max(0, (badge_r + .006) ** 2 - max(0, abs(y) - thick / 2) ** 2))
                spans = [(-iw / 2, -cut), (cut, iw / 2)] if cut else [(-iw / 2, iw / 2)]
                for j, (lo, hi) in enumerate(spans):
                    _part(m, P.box(hi - lo, thick, .030, material="chrome", bevel=.002,
                                   center=((lo + hi) / 2, y, -.001)), f"slat_{k}_{j}")
        elif pattern == "honeycomb":
            # Individual open hexagonal cells. Cell size is bounded to keep a
            # wide van grille inexpensive even at the smallest option value.
            r = max(float(opts.get("cell_size", .025)), iw / 48, .018)
            row = np.sqrt(3) * r
            for i, x in enumerate(np.arange(-iw / 2 + r, iw / 2 - r, 1.5 * r)):
                for j, y in enumerate(np.arange(-ih / 2 + row / 2 + (i % 2) * row / 2, ih / 2 - row / 2, row)):
                    _part(m, P.tube(r, r - .0025, .022, 6, material="mesh_dark", center=(x, y, -.001)), f"cell_{i}_{j}")
        else:
            # Intersecting diagonal wires clipped analytically to the insert.
            for slope in (-1, 1):
                for k, intercept in enumerate(np.arange(-ih / 2 - iw / 2, ih / 2 + iw / 2, .030)):
                    ends = []
                    for x in (-iw / 2, iw / 2):
                        y = slope * x + intercept
                        if -ih / 2 <= y <= ih / 2:
                            ends.append([x, y, .003])
                    for y in (-ih / 2, ih / 2):
                        x = (y - intercept) / slope
                        if -iw / 2 < x < iw / 2:
                            ends.append([x, y, .003])
                    if len(ends) == 2 and np.linalg.norm(np.subtract(*ends)) > .001:
                        _part(m, _pipe(ends, .0022, "mesh_dark", sides=4), f"wire_{slope}_{k}")
        if badge_r:
            _part(m, P.tube(badge_r, badge_r - .004, .012, 24, material="chrome", center=(0, 0, .003)), "badge_rim")
            _part(m, P.cylinder(badge_r - .005, .008, 24, material="grille_dark", center=(0, 0, .004)), "badge")
        # These mounts sit on uncut fascia, unlike the lamp apertures. Keep the
        # radiator backing in front of the skin so paint cannot fill the cells.
        m.translate([0, 0, .022])
        if conn.meta.get("lower"):
            # The rounded lower bumper projects past the average fascia plane.
            # Rake the insert's backing and cells together, anchored at its top,
            # rather than leaving blue bodywork visible through the lower rows.
            rake = .75 * abs(conn.normal[2]) / max(abs(conn.normal[0]), .5)
            down = -1 if conn.frame.y_axis[2] > 0 else 1
            m.vertices[:, 2] += .007 + rake * (down * m.vertices[:, 1] + h / 2)
        m.materials.update(mats)
        return ComponentResult(m)


@register
class HoneycombGrille(Grille):
    name = "grille.honeycomb"
    default_for = ()
    pattern = "honeycomb"
    options = dict(Grille.options, cell_size=.025, badge=False)
    description = "open hexagonal grille cells with 22mm wall depth"


@register
class MeshGrille(Grille):
    name = "grille.mesh"
    default_for = ()
    pattern = "mesh"
    options = dict(Grille.options, badge=False)
    description = "clipped diagonal woven-wire grille insert"


@register
class Intake(HoneycombGrille):
    name = "grille.intake"
    default_for = ("intake",)
    priority = 1
    options = dict(HoneycombGrille.options)
    description = "lower bumper honeycomb intake in a three-dimensional surround"


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
        if conn.meta.get("position") == "front":
            # The lower intake and plate overlap in elevation on short fascias;
            # a 65mm plinth puts the plate ahead of, not behind, the insert.
            _part(m, P.box(w * .82, h * .72, .065, material="plate_text", center=(0, 0, -.0325)), "mounting_plinth")
            m.translate([0, 0, .065])
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
        _part(m, P.rounded_box(L * 1.02, .068, .007, .026, material="antenna_gasket", center=(0, 0, .0035)), "antenna_base")
        mats["antenna_gasket"] = ctx.material("antenna_gasket", "#181a1d", shininess=.2)
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
        m = Mesh(name="handle")
        outer = rounded_rect_points(L + .025, .054, .020, 5)
        inner = outer * [.72, .52]
        rings = [np.column_stack([outer, np.full(len(outer), .006)]),
                 np.column_stack([inner, np.full(len(inner), .001)])]
        bowl = P.loft(rings, material="trim_dark", cap_end=True)
        if bowl.face_normals()[:, 2].mean() < 0:
            bowl.flip_normals()
        _part(m, bowl, "finger_recess")
        path = np.array([[-L / 2, .006, .006], [-L * .38, .008, .025],
                         [0, .008, .030], [L * .38, .008, .025], [L / 2, .006, .006]])
        handle = P.sweep_profile(path, rounded_rect_points(.022, .013, .004, 3), material=mat).flip_normals()
        _part(m, handle, "pull_bridge")
        _part(m, P.cylinder(.003, .002, 12, material="trim_dark", center=(L * .39, .006, .034)), "keyhole")
        # The left connector's local Y points down, unlike the right one.
        m.scale([1, 1 if conn.frame.y_axis[2] >= 0 else -1, 1])
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
        col = opts.get("housing_color") or ctx.palette.paint
        mats = {"mirror_housing": ctx.material("mirror_housing", col, shininess=.85),
                "trim_dark": ctx.material("trim_dark", "#101113", shininess=.2),
                "mirror_glass": Material("mirror_glass", (.62, .73, .82), 1., .95),
                "mirror_signal": ctx.material("mirror_signal", "#edb66b", shininess=.85)}
        m = Mesh(name="mirror")
        _part(m, P.rounded_box(.065, .095, .016, .022, material="trim_dark", center=(0, .005, .008)), "base_plinth")
        _part(m, _pipe([[0, -.015, .012], [0, -.009, .045], [-.005, .009, .085]], .016, "trim_dark"), "stalk")
        outline = rounded_rect_points(.20, .112, .035, 5)
        rings = [np.column_stack([np.full(len(outline), x), outline[:, 1] * scale + .025,
                                  outline[:, 0] * scale + .145])
                 for x, scale in ((-.045, 1.), (.010, 1.03), (.066, .77), (.090, .30))]
        housing = P.loft(rings, cap_end=True, material="mirror_housing")
        if P.signed_volume(housing) < 0:
            housing.flip_normals()
        _part(m, housing, "housing")
        rear_frame = Frame.from_normal([-.046, .025, .145], [-1, 0, 0], [0, 0, 1])
        _part(m, _surround(.202, .114, .007, .008, "trim_dark").apply_frame(rear_frame), "glass_gasket")
        glass = P.grid_fill_polygon(rounded_rect_points(.186, .098, .026, 5), material="mirror_glass", z=.003, bulge=.003)
        _part(m, glass.apply_frame(rear_frame), "mirror_pane")
        signal_edge = outline[10:20] * .77
        path = np.column_stack([np.full(len(signal_edge), .0695), signal_edge[:, 1] + .025,
                                signal_edge[:, 0] + .145])
        _part(m, _pipe(path, .0035, "mirror_signal"), "turn_signal")
        m.scale([1, 1 if conn.frame.y_axis[2] >= 0 else -1, 1])
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


# =============================================================== FASCIA FURNITURE
@register
class FogLamp(CarComponent):
    name = "light.fog"
    accepts = (PointConnector,)
    default_for = ("fog",)
    options = {"color": "#e8eef3"}
    description = "bumper-corner projector fog lamp in a recessed black plinth"

    def build_local(self, conn, opts, ctx):
        w, h = conn.meta.get("width", .14), conn.meta.get("height", .09)
        r = min(h * .36, w * .28)
        m = Mesh(name="fog")
        _part(m, P.rounded_box(w, h, .018, min(.025, h * .24), material="fog_trim", center=(0, 0, .009)), "fog_plinth")
        _part(m, P.tube(r + .005, r - .002, .018, 24, material="fog_chrome", center=(0, 0, .024)), "fog_bezel")
        _part(m, P.cylinder(r * .90, .010, 24, radius_top=r * .8, material="fog_lens", center=(0, 0, .030)), "fog_projector")
        m.materials.update({"fog_trim": ctx.material("fog_trim", "#15191e"),
                            "fog_chrome": ctx.material("fog_chrome", ctx.palette.chrome, metallic=.8, shininess=.8),
                            "fog_lens": ctx.material("fog_lens", opts["color"], shininess=.95)})
        return ComponentResult(m)


class _SmallLamp(CarComponent):
    accepts = (PointConnector,)
    color = "#d72732"

    def build_local(self, conn, opts, ctx):
        w, h, d = (conn.meta.get(k, v) for k, v in (("width", .13), ("height", .04), ("depth", .008)))
        kind = conn.meta.get("kind", "")
        col = "#e0e9ed" if kind == "reverse" else self.color
        key = self.name.replace(".", "_") + kind
        m = Mesh(name=key)
        _part(m, P.rounded_box(w, h, d, min(.012, h * .23), material="aux_trim", center=(0, 0, d / 2)), "plinth")
        _part(m, P.rounded_box(w - .009, h - .008, .004, min(.009, h * .18), material=key,
                              center=(0, 0, d + .001)), "lens")
        for k, x in enumerate(np.linspace(-w * .40, w * .40, max(3, int(w / .008)))):
            _part(m, P.box(.0012, h - .012, .0015, material=key, center=(x, 0, d + .0035)), f"prism_{k}")
        m.materials.update({"aux_trim": ctx.material("aux_trim", "#191b1e"),
                            key: ctx.material(key, col, shininess=.85)})
        return ComponentResult(m)


@register
class RearReflector(_SmallLamp):
    name = "light.rear_reflector"
    default_for = ("rear_reflector",)
    description = "red prismatic reflex reflector on the rear bumper"


@register
class RearAuxLamp(_SmallLamp):
    name = "light.rear_aux"
    default_for = ("rear_aux",)
    description = "rear fog or reversing lamp selected by connector kind"


@register
class SideMarker(_SmallLamp):
    name = "light.side_marker"
    default_for = ("side_marker",)
    color = "#e99721"
    description = "amber side marker with geometric lens flutes"


@register
class PlateLamp(_SmallLamp):
    name = "light.plate"
    default_for = ("plate_lamp",)
    color = "#e5ecf0"
    description = "licence plate lamp with a shielded downward-facing lens"

    def build_local(self, conn, opts, ctx):
        m = super().build_local(conn, opts, ctx).mesh
        # Rotate the fixture down toward the plate; retain its mounting lip.
        m.apply_frame(Frame.identity().rotated_about_x(np.radians(-35)))
        return ComponentResult(m)


@register
class TowHookCover(CarComponent):
    name = "bumper.tow_cover"
    accepts = (PointConnector,)
    default_for = ("tow_hook_cover",)
    description = "painted removable tow-hook cover with a narrow shutline"

    def build_local(self, conn, opts, ctx):
        w, h = conn.meta.get("width", .065), conn.meta.get("height", .055)
        m = P.rounded_box(w, h, .003, .012, material="tow_gap", center=(0, 0, .0015))
        _part(m, P.rounded_box(w - .004, h - .004, .004, .010, material="paint", center=(0, 0, .003)), "cover")
        m.materials.update({"tow_gap": ctx.material("tow_gap", "#191b1e"), "paint": ctx.material("paint", shininess=.85)})
        return ComponentResult(m)


# =============================================================== WIPERS / UNDERSIDE
@register
class Wipers(CarComponent):
    name = "wipers.parked"
    accepts = (RectangleConnector,)
    default_for = ("wipers",)
    description = "two parked sprung blades and articulated arms following the windshield crown"

    def build_local(self, conn, opts, ctx):
        paths = conn.meta.get("paths")
        if paths is None:
            paths = [conn.frame.to_world([[s * conn.width * .05, 0, .01], [s * conn.width * .43, .025, .01]])
                     for s in (1, -1)]
        pivots = conn.meta.get("pivots", [p[0] for p in paths])
        normals = conn.meta.get("blade_normals")
        m = Mesh(name="wipers")
        for k, (path, pivot) in enumerate(zip(paths, pivots)):
            pts = conn.frame.to_local(path)
            p = conn.frame.to_local(pivot)[0]
            n = np.asarray(normals[k]) @ conn.frame.rotation if normals is not None else np.tile([0, 0, 1], (len(pts), 1))
            _part(m, _pipe(pts, .0035, "wiper_rubber", sides=6), f"blade_{k}")
            _part(m, _pipe(pts + .005 * n, .004, "wiper_arm", sides=6), f"blade_spine_{k}")
            attach = pts[len(pts) // 2] + .011 * n[len(pts) // 2]
            elbow = p * .40 + attach * .60 + [.0, .0, .014]
            _part(m, _pipe([p, elbow, attach], .005, "wiper_arm", sides=6), f"arm_{k}")
            _part(m, P.cylinder(.014, .009, 16, material="wiper_arm", center=p), f"pivot_{k}")
        m.materials.update({"wiper_rubber": ctx.material("wiper_rubber", "#080a0c"),
                            "wiper_arm": ctx.material("wiper_arm", "#292d33", shininess=.5)})
        return ComponentResult(m)


@register
class ExhaustSystem(CarComponent):
    name = "exhaust.system"
    accepts = (PointConnector,)
    default_for = ("exhaust_system",)
    options = {"muffler": True}
    description = "underfloor tunnel pipe, catalyst and rear branches terminating at the exhaust tips"

    def build_local(self, conn, opts, ctx):
        paths = conn.meta.get("paths")
        if paths is None:
            length = ctx.measurements.get("wheelbase", 2.7)
            paths = [conn.frame.to_world([[length / 2, 0, 0], [-length / 2, 0, 0]])]
        r = float(conn.meta.get("pipe_radius", .025))
        m = Mesh(name="exhaust_system")
        local = [conn.frame.to_local(path) for path in paths]
        for k, path in enumerate(local):
            _part(m, _pipe(path, r, "exhaust_steel", sides=10), f"pipe_run_{k}")
        main = local[0]
        if opts["muffler"]:
            for name, t, length, radius in (("catalyst", .20, .25, .047), ("silencer", .77, .34, .059)):
                index = min(len(main) - 2, max(0, int(t * (len(main) - 1))))
                point = main[index]
                axis = main[index + 1] - main[index]
                can = P.cylinder(radius, length, 16, radius_top=radius * .90, material="exhaust_steel")
                can.apply_frame(Frame.from_normal(point, axis))
                # Low sports floors need an oval can, not a round muffler that
                # almost scrapes the ground. Flatten only its vertical section.
                if conn.meta.get("paths_space") == "world":
                    up = conn.frame.rotation[2]
                    height = conn.frame.to_world(point)[0, 2]
                    vertical = (can.vertices - point) @ up
                    factor = min(1., max(.020, height - .050) / max(-vertical.min(), 1e-6))
                    can.vertices += vertical[:, None] * (factor - 1) * up
                _part(m, can, name)
        m.materials["exhaust_steel"] = ctx.material("exhaust_steel", "#787e83", metallic=.7, shininess=.45)
        return ComponentResult(m)


@register
class Diffuser(CarComponent):
    name = "bumper.diffuser"
    accepts = (RectangleConnector,)
    default_for = ("diffuser",)
    options = {"fins": 5}
    description = "rear undertray with tapered longitudinal diffuser strakes"

    def build_local(self, conn, opts, ctx):
        w, h = conn.width, conn.height
        depth = float(conn.meta.get("depth", .045))
        m = P.box(w, h, .007, material="diffuser", center=(0, 0, .0035))
        for k, x in enumerate(np.linspace(-w * .42, w * .42, int(np.clip(opts["fins"], 2, 9)))):
            rings = [np.array([[x - .004, y, .006], [x + .004, y, .006],
                               [x + .004, y, z], [x - .004, y, z]])
                     for y, z in ((-h / 2, depth), (0, depth * .76), (h / 2, .013))]
            fin = P.loft(rings, cap_start=True, cap_end=True, material="diffuser")
            if P.signed_volume(fin) < 0:
                fin.flip_normals()
            _part(m, fin, f"fin_{k}")
        m.materials["diffuser"] = ctx.material("diffuser", "#252930", shininess=.35)
        return ComponentResult(m)


@register
class MudFlap(CarComponent):
    name = "mud_flap.standard"
    accepts = (RectangleConnector,)
    default_for = ("mud_flap",)
    description = "optional rubber mud flap with mounting rivets (body.hints.mud_flaps)"

    def build_local(self, conn, opts, ctx):
        w, h = conn.width, conn.height
        m = P.rounded_box(w, h, .006, min(.025, h * .18), material="mud_rubber", center=(0, 0, .003))
        for k, x in enumerate((-.36 * w, .36 * w)):
            _part(m, P.cylinder(.004, .003, 8, material="mud_fastener", center=(x, h / 2 - .017, .007)), f"rivet_{k}")
        m.materials.update({"mud_rubber": ctx.material("mud_rubber", "#1c1d20"),
                            "mud_fastener": ctx.material("mud_fastener", "#92989d", metallic=.7)})
        return ComponentResult(m)


@register
class TowEye(CarComponent):
    name = "bumper.tow_eye"
    accepts = (PointConnector,)
    default_for = ("tow_eye",)
    description = "small forged towing eye below the rear valance"

    def build_local(self, conn, opts, ctx):
        m = P.cylinder(.010, .040, 12, material="tow_steel", center=(0, 0, .020))
        eye = P.torus(.024, .006, 20, 8, material="tow_steel")
        eye.apply_frame(Frame.from_normal([0, 0, .065], [0, 1, 0], [1, 0, 0]))
        _part(m, eye, "eye")
        m.materials["tow_steel"] = ctx.material("tow_steel", "#505761", metallic=.8)
        return ComponentResult(m)
