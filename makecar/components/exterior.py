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
    groove_depth: float = 0.004


@register
class Wheel(MorphableComponent):
    """Tyre + rim + spokes, fitted to a CircleConnector (hub).  Local frame:
    +Z is the axle direction pointing outboard, the hub is at the origin."""

    name = "wheel.alloy"
    accepts = (CircleConnector,)
    default_for = ("wheel",)
    params_cls = WheelParams
    options = {"spokes": 5, "spoke_width": 0.55, "hubcap": True, "tread_grooves": 3,
               "tread_blocks": 40, "dish": 0.04, "caliper_color": "#b52d22",
               "aspect_ratio": None, "tread_pattern": "asymmetric_block", "groove_depth": None,
               "shoulder_blocks": None, "lettering": False, "spoke_family": "five_spoke",
               "disc_pattern": "plain", "disc_vanes": 16}
    description = "tyre with alloy rim; radius/width/rim size fitted from the hub connector"
    modifier_specs = [
        ModifierSpec("radius", 0.10, 0.14, "size", "tyre radius"),
        ModifierSpec("width", 0.08, 0.12, "size", "tyre width"),
        ModifierSpec("rim_ratio", 0.15, 0.20, "style", "low-profile <-> balloon tyre"),
        ModifierSpec("dish", 0.03, 0.05, "style", "spoke dish depth"),
    ]

    N = 64
    SPOKE_FAMILIES = ("mesh", "five_spoke", "twin_five", "multi_spoke", "turbine", "dish", "steel_cap")
    TREAD_PATTERNS = ("directional_v", "asymmetric_block", "all_terrain", "slick")

    def generate(self, p: WheelParams) -> Mesh:
        # Fixed topology: bead seats, rounded sidewalls and a recessed tread bed.
        # The visible tread is fitted later, so groove depth never changes OD.
        R, w, rr = p.radius, p.width, p.rim_ratio * p.radius
        gap = R - rr
        side = [[rr, w * .40], [rr + gap * .05, w * .435],
                [rr + gap * .18, w * .475], [rr + gap * .40, w * .50],
                [rr + gap * .65, w * .485], [R - gap * .12, w * .44],
                [R - gap * .055, w * .39], [R - p.groove_depth, w * .35]]
        profile = [[r, -z] for r, z in side] + side[::-1]
        m = Mesh(name="wheel")
        _part(m, P.revolve(profile, self.N, material="tyre"), "carcass")
        # Rolled lips are integral with the open barrel; a drop centre clears
        # the tyre bead during mounting. No circular plate covers the openings.
        barrel = [[rr - .010, -w * .42], [rr + .004, -w * .45],
                  [rr + .008, -w * .43], [rr - .009, -w * .22],
                  [rr - .009, w * .22], [rr + .008, w * .43],
                  [rr + .004, w * .45], [rr - .010, w * .42],
                  [rr - .015, w * .22], [rr - .015, -w * .22],
                  [rr - .010, -w * .42]]
        shell = P.revolve(barrel, 48, material="rim")
        shell.set_material("rim_dark", [i for i in range(shell.n_faces) if i % 10 in (7, 8, 9)])
        for side in (-1, 1):
            shell.add_group(f"lip_{side}", np.flatnonzero(shell.vertices[:, 2] * side >= w * .42 - 1e-9))
            shell.add_zone(f"lip_{side}", np.flatnonzero(shell.face_centroids()[:, 2] * side >= w * .42 - 1e-9))
        _part(m, shell, "barrel")
        zf = w / 2 - .024 - p.dish
        _part(m, P.tube(.061, .026, .018, 8, material="rim_dark", center=(0, 0, zf)), "face_ring")
        return m

    def _fitted_params(self, conn, opts):
        R = float(conn.radius)
        w = float(conn.meta.get("tire_width", .225))
        if not np.isfinite([R, w]).all() or R <= .12 or w <= .06:
            raise ValueError("wheel needs a finite tyre radius > .12m and section width > .06m")
        aspect = opts.get("aspect_ratio")
        if aspect is not None:
            aspect = float(aspect)
            if not np.isfinite(aspect) or not 20 <= aspect <= 90:
                raise ValueError("aspect_ratio is a percentage in [20, 90], e.g. 35 for 245/35")
            rr = R - w * aspect / 100
        else:
            rr = R * float(opts.get("rim_ratio", .62 + (R - .33) * .6))
        if not np.isfinite(rr) or not .085 < rr < R - .018:
            raise ValueError("tyre size leaves no usable rim or sidewall; check connector and aspect_ratio")
        dish = float(opts["dish"])
        depth = float(opts["groove_depth"] if opts["groove_depth"] is not None
                      else (.010 if opts["tread_pattern"] == "all_terrain" else .004))
        if not np.isfinite([dish, depth]).all():
            raise ValueError("dish and groove_depth must be finite metres")
        # Negative dish makes a convex face; connector OD/section width win over
        # the finite slider library range, including staggered synthetic mounts.
        return WheelParams(R, w, rr / R, float(np.clip(dish, -.025, .09)),
                           groove_depth=float(np.clip(depth, 0, min(.018, (R - rr) * .25))))

    def fit(self, conn: CircleConnector, opts, ctx) -> Dict[str, float]:
        p = self._fitted_params(conn, opts)
        return {key: self.value_for(key, getattr(p, key)) for key in ("radius", "width", "rim_ratio", "dish")}

    def materials(self, ctx, opts):
        return {
            "tyre": ctx.material("tyre", "#282a2d", shininess=0.08),
            "rim": ctx.material("rim", opts.get("rim_color", ctx.palette.rim), shininess=0.7, metallic=0.8),
            "rim_dark": ctx.material("rim_dark", "#3a3c40", shininess=0.4),
            "brake_disc": ctx.material("brake_disc", "#8a8e92", metallic=.8, shininess=.5),
            "brake_hat": ctx.material("brake_hat", "#454a50", metallic=.6, shininess=.35),
            "brake_pad": ctx.material("brake_pad", "#303237", shininess=.15),
            "brake_shield": ctx.material("brake_shield", "#383d44", metallic=.5, shininess=.3),
            "caliper": ctx.material("caliper", opts["caliper_color"], shininess=.5),
        }

    def build_local(self, conn, opts, ctx) -> ComponentResult:
        for option, choices in (("tread_pattern", self.TREAD_PATTERNS), ("spoke_family", self.SPOKE_FAMILIES),
                                ("disc_pattern", ("plain", "drilled", "slotted"))):
            if opts[option] not in choices:
                raise ValueError(f"{option} must be one of {', '.join(choices)}")
        p = self._fitted_params(conn, opts)
        if opts["tread_pattern"] == "slick":
            p.groove_depth = 0.
        res = super().build_local(conn, opts, ctx)
        # Targets remain useful for editing/preview, but radius*rim_ratio and
        # width*aspect are coupled physical quantities. Correct the fitted
        # vertices analytically (same topology), not with the old missing
        # cross-term or clipped large/small axle dimensions.
        res.mesh.vertices = self.generate(p).vertices
        rr = p.radius * p.rim_ratio
        self._tread(res.mesh, p, opts)
        if opts["lettering"]:
            self._sidewall_lettering(res.mesh, p)
        self._wheel_face(res.mesh, p, rr, opts)
        self._hardware(res.mesh, p, rr, opts)
        res.info.update({"tread_blocks": int(np.clip(opts["tread_blocks"], 0, 64)),
                         "spokes": int(np.clip(opts["spokes"], 3, 12)), "rim_radius": rr,
                         "radius": p.radius, "section_width": p.width,
                         "sidewall_height": p.radius - rr,
                         "aspect_ratio": (p.radius - rr) / p.width * 100,
                         "rim_diameter_inches": 2 * rr / .0254,
                         "dish": p.dish, "groove_depth": p.groove_depth,
                         "tread_pattern": opts["tread_pattern"], "spoke_family": opts["spoke_family"]})
        return res

    @staticmethod
    def _rubber_block(angles, zz, bottom, top):
        angles, zz = np.asarray(angles), np.asarray(zz)
        rings = [np.column_stack([r * np.cos(angles), r * np.sin(angles), zz]) for r in (bottom, top)]
        block = P.loft(rings, material="tyre")
        block.faces += [(3, 2, 1, 0), (4, 5, 6, 7)]
        block.face_materials += ["tyre", "tyre"]
        return block

    def _tread(self, m, p, opts):
        blocks = int(np.clip(opts["tread_blocks"], 0, 64))
        grooves = int(np.clip(opts["tread_grooves"], 0, 4))
        pattern, depth = opts["tread_pattern"], p.groove_depth
        if pattern == "slick" or depth == 0:
            # With zero depth the carcass itself reaches the connector radius.
            return
        lanes = grooves + 1 if pattern != "all_terrain" else 3
        weights = np.linspace(.88, 1.12, lanes) if pattern == "asymmetric_block" else np.ones(lanes)
        edges = np.r_[0., np.cumsum(weights / weights.sum())] * p.width * .70 - p.width * .35
        for j, (lo, hi) in enumerate(zip(edges, edges[1:])):
            z = (lo + hi) / 2
            half = ((hi - lo) * .34 if pattern == "all_terrain"
                    else (hi - lo - min(.006, (hi - lo) * .18)) / 2)
            crest = p.radius if blocks == 0 else p.radius - depth * .75
            prof = [[p.radius - depth, z - half], [crest, z - half],
                    [crest, z + half], [p.radius - depth, z + half]]
            if pattern != "all_terrain" or blocks == 0:
                rib = P.revolve(prof, self.N, material="tyre")
                # Match the carcass circumference: a coarser crest disappears
                # into it between samples. Adjacent COPLANAR side-wall quads
                # can share one hexagon without changing any triangles/curves.
                faces = [rib.faces[i * 3 + 1] for i in range(self.N)]
                for wall in (0, 2):
                    for i in range(0, self.N, 2):
                        pair = [rib.faces[i * 3 + wall], rib.faces[(i + 1) * 3 + wall]]
                        edges = [(a, b) for face in pair for a, b in zip(face, face[1:] + face[:1])]
                        boundary = {a: b for a, b in edges if (b, a) not in edges}
                        start = (i + 1) * 4 + (0 if wall == 0 else 3)
                        loop = [start]
                        while boundary[loop[-1]] != start:
                            loop.append(boundary[loop[-1]])
                        faces.append(tuple(loop))
                rib.faces, rib.face_materials = faces, ["tyre"] * len(faces)
                _part(m, rib, f"tread_rib_{j}")
            for k in range(blocks):
                phase = .5 * (j % 2) if pattern != "directional_v" else 0
                a = 2 * np.pi * (k + phase) / blocks
                da = 2 * np.pi / blocks * (.34 if pattern == "all_terrain" else .43)
                skew = (.022 if pattern == "all_terrain" else .014) / p.radius * (1 if j % 2 else -1)
                if pattern == "directional_v":
                    a += abs(z) * .85 / p.radius
                    skew = half * .85 / p.radius * np.sign(z)
                angles = [a - da - skew, a + da - skew, a + da + skew, a - da + skew]
                zz = [z - half, z - half, z + half, z + half]
                _part(m, self._rubber_block(angles, zz, p.radius - depth, p.radius), f"tread_block_{j}_{k}")
        # Shoulder lugs wrap onto the sidewall instead of stopping at a square
        # tread edge. Their outer surfaces remain inside the section envelope.
        shoulders = int(np.clip(opts["shoulder_blocks"] if opts["shoulder_blocks"] is not None
                                else (32 if pattern == "all_terrain" else 16), 0, 40))
        gap = p.radius * (1 - p.rim_ratio)
        for side in (-1, 1):
            for k in range(shoulders):
                a = 2 * np.pi * (k + .25 * side) / shoulders
                da = 2 * np.pi / shoulders * (.32 if pattern == "all_terrain" else .39)
                angles = np.array([a - da, a + da, a + da + .018, a - da + .018])
                zz = p.width * np.array([.35, .35, .435, .435])
                top = p.radius - gap * np.array([.018, .018, .11, .11])
                block = self._rubber_block(angles, zz, top - min(depth, .004), top)
                _part(m, block.scale([1, 1, side]), f"shoulder_{side}_{k}")

    def _sidewall_lettering(self, m, p):
        # Tiny moulded block glyphs, not textures/fonts or a runtime dependency.
        glyphs = {"0": "111101101101111", "1": "010110010010111", "2": "110001010100111",
                  "3": "110001010001110", "4": "101101111001001", "5": "111100110001110",
                  "6": "011100111101111", "7": "111001010010010", "8": "111101111101111",
                  "9": "111101111001110", "R": "110101110101101", "A": "010101111101101",
                  "D": "110101101101110", "I": "111010010010111", "L": "100100100100111",
                  "/": "001001010100100", " ": "000000000000000"}
        rr, gap = p.radius * p.rim_ratio, p.radius * (1 - p.rim_ratio)
        cell = min(.0032, gap * .045)
        radius = rr + gap * .40
        size = f"{round(p.width * 1000)}/{round(gap / p.width * 100)} R{round(rr * 2 / .0254)}"
        for label, text, direction in (("brand", "RADIAL", -1), ("size", size, 1)):
            part = Mesh()
            for i, char in enumerate(text):
                for row in range(5):
                    for col in range(3):
                        if glyphs[char][row * 3 + col] != "1":
                            continue
                        r = radius - direction * (2 - row) * cell
                        a = direction * np.pi / 2 + direction * (i * 4 + col - len(text) * 2) * cell / radius
                        z = p.width * np.interp((r - rr) / gap, [.18, .40, .65], [.475, .50, .485])
                        pixel = P.box(cell * .88, cell * .88, .0008, material="tyre",
                                      center=(0, 0, z + .0003))
                        pixel.apply_frame(Frame.from_normal([r * np.cos(a), r * np.sin(a), 0],
                                                            [0, 0, 1], [np.cos(a), np.sin(a), 0]))
                        part.merge(pixel)
            _part(m, part, f"sidewall_{label}")

    def _wheel_face(self, m, p, rr, opts):
        n = int(np.clip(opts["spokes"], 3, 12))
        width = float(np.clip(opts["spoke_width"], .2, .85))
        family = opts["spoke_family"]
        if family == "steel_cap":
            self._steel_face(m, p, rr, opts)
            return
        zf, lip_z = p.width / 2 - .024 - p.dish, p.width / 2 - .028
        if family == "dish":
            radii = np.linspace(.054, rr * .76, 4)
            front = [[r, zf + .010 + (p.dish - .004) * t] for r, t in zip(radii, (0., .3, .8, 1.))]
            # Both skins follow the same radial graph; a constant axial wall
            # cannot cross itself when the dish becomes convex.
            profile = front + [[r, z - .012] for r, z in front[::-1]] + [front[0]]
            _part(m, P.revolve(profile, self.N, material="rim"), "aero_dish")
        count = {"five_spoke": n, "twin_five": 10, "multi_spoke": 18,
                 "mesh": 24, "turbine": 9, "dish": 8}[family]
        for k in range(count):
            if family == "twin_five":
                a, branch = (k // 2) * 2 * np.pi / 5, (-1 if k % 2 else 1)
            elif family == "mesh":
                a, branch = (k // 2) * 2 * np.pi / 12, (-1 if k % 2 else 1)
            else:
                a, branch = k * 2 * np.pi / count, 0
            rings = []
            for t, flare in ((0., 1.45), (.12, 1.05), (.48, 1.), (.87, 1.12), (1., 1.7)):
                start = rr * .735 if family == "dish" else .054
                r = start + t * (rr - .006 - start)
                sweep = .055 * t
                half = min(.026, r * np.pi / n * width) * flare
                if family in ("twin_five", "multi_spoke", "mesh"):
                    half = {"twin_five": .009, "multi_spoke": .007, "mesh": .0045}[family] * flare * width / .55
                    sweep = branch * (.035 + .105 * t) if family == "twin_five" else branch * .37 * t
                elif family == "turbine":
                    sweep, half = .40 * t ** .8, (.010 + .024 * t) * flare
                elif family == "dish":
                    half = .022 * flare
                # Smooth dish rise and widening roots approximate forged
                # fillets. The octagon has a 2mm edge break and 12% wall draft,
                # not a rectangular slab with painted shading.
                z = zf + (lip_z - zf) * (t * t * (3 - 2 * t))
                if family == "dish":
                    z = lip_z - .008 + .004 * t
                h, bevel = .010, min(.002, half * .22)
                yz = np.array([[-half + bevel, -h], [half - bevel, -h],
                               [half, -h + bevel], [half * .88, h - bevel],
                               [half * .88 - bevel, h], [-half * .88 + bevel, h],
                               [-half * .88, h - bevel], [-half, -h + bevel]])
                angle = a + sweep
                rings.append(np.column_stack([r * np.cos(angle) - yz[:, 0] * np.sin(angle),
                                              r * np.sin(angle) + yz[:, 0] * np.cos(angle), z + yz[:, 1]]))
            spoke = P.loft(rings, cap_start=True, cap_end=True, material="rim")
            _part(m, spoke, f"spoke_{k}")

    @staticmethod
    def _pierced_ring(inner, outer, bottom, top, count, hole, material, *, pocket=0., taper=1., phase=0.,
                      round_outer=False, inner_wall=True, outer_wall=True):
        """Annulus tiled around real octagonal holes (or blind machined slots).

        Shared sector boundaries are welded; no coplanar discs hide the holes.
        The same construction makes conical lug seats without a boolean library.
        """
        m = Mesh()
        middle = (inner + outer) / 2
        for k in range(count):
            a = 2 * np.pi * k / count + phase
            da = np.pi / count
            radii = np.array([inner, middle, outer, outer, outer, middle, inner, inner])
            angles = a + np.array([-da, -da, -da, 0, da, da, da, 0])
            boundary = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            phi = np.arange(8) * np.pi / 4 + 5 * np.pi / 4
            xy = np.column_stack([hole[0] * np.cos(phi), hole[1] * np.sin(phi)])
            rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            center = middle * np.array([np.cos(a), np.sin(a)])
            opening = center + xy @ rotation.T
            floor = center + taper * xy @ rotation.T
            rings = [np.column_stack([xy, np.full(8, z)]) for xy, z in
                     ((boundary, bottom), (boundary, top), (opening, top),
                      (floor, top - pocket if pocket else bottom), (boundary, bottom))]
            tile = P.loft(rings, material=material)
            # Adjacent cells share these radial walls; omit them altogether.
            drop = {0, 1, 4, 5}
            if not outer_wall:
                drop.update((2, 3))
            if not inner_wall:
                drop.update((6, 7))
            if pocket:
                # Replace the diagonal pocket-to-back strip with its floor and
                # an uninterrupted back face. Both loops are star-shaped.
                drop.update(range(24, 32))
            tile.remove_faces(list(drop))
            if pocket:
                for ring, flip in ((24, True), (0, True)):
                    ids = list(range(ring, ring + 8))
                    if ring == 24:
                        flip = False  # pocket floor faces into the opening (+Z)
                    center_id = tile.n_vertices
                    tile.vertices = np.vstack([tile.vertices, tile.vertices[ids].mean(axis=0)])
                    tile.faces.extend((center_id, ids[(j + 1) % 8], ids[j]) if flip
                                      else (center_id, ids[j], ids[(j + 1) % 8]) for j in range(8))
                    tile.face_materials.extend([material] * 8)
            m.merge(tile)
        _, first, inverse = np.unique(np.round(m.vertices, 12), axis=0, return_index=True, return_inverse=True)
        m.vertices = m.vertices[first]
        m.faces = [tuple(inverse[list(face)]) for face in m.faces]
        if round_outer:
            # A five-bolt hub still needs a round perimeter, not a decagon.
            # Refine the curved wall and its planar annulus boundary together.
            vertices, faces, midpoints = m.vertices.tolist(), [], {}
            outer_ids = set(np.flatnonzero(np.isclose(np.linalg.norm(m.vertices[:, :2], axis=1), outer)))
            def midpoint(a, b):
                key = tuple(sorted((a, b)))
                if key not in midpoints:
                    v = (m.vertices[a] + m.vertices[b]) / 2
                    v[:2] *= outer / np.linalg.norm(v[:2])
                    midpoints[key] = len(vertices)
                    vertices.append(v.tolist())
                return midpoints[key]
            for face in m.faces:
                edges = [(a, b) for a, b in zip(face, face[1:] + face[:1])]
                refine = [a in outer_ids and b in outer_ids and abs(m.vertices[a, 2] - m.vertices[b, 2]) < 1e-12
                          for a, b in edges]
                if all(v in outer_ids for v in face):
                    i = refine.index(True)
                    a, b, c, d = face[i:] + face[:i]
                    ab, cd = midpoint(a, b), midpoint(c, d)
                    faces.extend([(a, ab, cd, d), (ab, b, c, cd)])
                else:
                    faces.append(tuple(v for (a, b), split in zip(edges, refine)
                                       for v in ((a, midpoint(a, b)) if split else (a,))))
            m.vertices, m.faces = np.asarray(vertices), faces
            m.face_materials = [material] * len(faces)
        return m

    @staticmethod
    def _arc(inner, outer, bottom, top, material, angle=.72, start=np.pi - .36, n=9):
        profile = [[inner, bottom], [outer, bottom], [outer, top], [inner, top], [inner, bottom]]
        m = P.revolve(profile, n, angle=angle, material=material)
        m.faces.extend([(0, 1, 2, 3), tuple(range(m.n_vertices - 2, m.n_vertices - 6, -1))])
        m.face_materials.extend([material, material])
        m.apply_frame(Frame.from_normal([0, 0, 0], [0, 0, 1], [np.cos(start), np.sin(start), 0]))
        if P.signed_volume(m) < 0:
            m.flip_normals()
        return m

    def _hardware(self, m, p, rr, opts):
        zf = p.width / 2 - .024 - p.dish
        # Reserve a 6mm air gap even between the convex face and the caliper.
        spoke_back = min(zf, p.width / 2 - .028) - .010
        disc_z, rd = spoke_back - .036, rr * .79
        inner = min(.066, rd * .47)
        disc = Mesh()
        pattern = opts["disc_pattern"]
        lo, hi = disc_z + .004, disc_z + .010
        if pattern == "plain":
            plate = P.tube(rd, inner, .006, 32, material="brake_disc", center=(0, 0, (lo + hi) / 2))
        else:
            plate = Mesh()
            spans = np.linspace(inner, rd, 3) if pattern == "drilled" else [inner, rd]
            for j, (ri, ro) in enumerate(zip(spans, spans[1:])):
                hole = (.003, .003) if pattern == "drilled" else ((ro - ri) * .32, .0022)
                band = self._pierced_ring(ri, ro, lo, hi, 12, hole, "brake_disc",
                                         pocket=.0015 if pattern == "slotted" else 0,
                                         phase=j * np.pi / 12, inner_wall=j == 0,
                                         outer_wall=j == len(spans) - 2)
                plate.merge(band)
            # Join adjacent drilled bands without internal coincident walls.
            _, first, inverse = np.unique(np.round(plate.vertices, 12), axis=0, return_index=True, return_inverse=True)
            plate.vertices = plate.vertices[first]
            plate.faces = [tuple(inverse[list(face)]) for face in plate.faces]
        _part(disc, plate.copy().scale([1, 1, -1]).translate([0, 0, 2 * disc_z]), "disc_face_-1")
        _part(disc, plate, "disc_face_1")
        _part(m, disc, "brake_disc")
        _part(m, P.tube(inner + .004, .027, .020, 16, material="brake_hat",
                        center=(0, 0, disc_z + .005)), "disc_hat")
        for k in range(int(np.clip(opts["disc_vanes"], 0, 48))):
            a = k * 2 * np.pi / int(np.clip(opts["disc_vanes"], 0, 48))
            vane = P.box(rd - inner - .008, .003, .008, material="brake_hat",
                         center=((rd + inner) / 2, 0, disc_z))
            vane.apply_frame(Frame.from_normal([0, 0, 0], [0, 0, 1], [np.cos(a), np.sin(a), 0]))
            _part(m, vane, f"disc_vane_{k}")
        shield = P.revolve([[.031, disc_z - .030], [rd * .91, disc_z - .030],
                            [rd + .006, disc_z - .024], [rd + .006, disc_z - .026],
                            [rd * .91, disc_z - .032], [.031, disc_z - .032], [.031, disc_z - .030]],
                           18, angle=2 * np.pi - .94, material="brake_shield")
        # Leave an actual caliper cutout; the rolled edge must not pass through
        # the inboard cheek. Cap the two thin bent sections with simple quads.
        end = shield.n_vertices - 7
        shield.faces.extend([(5, 4, 1, 0), (4, 3, 2, 1),
                             (end, end + 1, end + 4, end + 5),
                             (end + 1, end + 2, end + 3, end + 4)])
        shield.face_materials.extend(["brake_shield"] * 4)
        a = np.pi + .47
        shield.apply_frame(Frame.from_normal([0, 0, 0], [0, 0, 1], [np.cos(a), np.sin(a), 0]))
        _part(m, shield, "dust_shield")
        # A U-shaped casting: cheeks sit outside both friction faces; only the
        # outer-radius bridge crosses the rotor's axial interval. It is NOT a
        # red toroidal box passing through the swept friction surface.
        caliper = Mesh()
        for side in (-1, 1):
            lo, hi = sorted((disc_z + side * .015, disc_z + side * .024))
            cheek = self._arc(rd - .036, rd + .014, lo, hi, "caliper")
            rings = cheek.vertices.reshape(9, 5, 3)
            for i, inset in enumerate((.009, .003, 0., 0., 0., 0., 0., .003, .009)):
                radii = np.linalg.norm(rings[i, :, :2], axis=1)
                change = np.array([inset, -inset, -inset, inset, inset])
                rings[i, :, :2] *= ((radii + change) / radii)[:, None]
            _part(caliper, cheek, f"caliper_cheek_{side}")
            lo, hi = sorted((disc_z + side * .012, disc_z + side * .014))
            _part(m, self._arc(rd - .035, rd - .005, lo, hi, "brake_pad", angle=.56,
                               start=np.pi - .28, n=7), f"pad_backing_{side}")
            lo, hi = sorted((disc_z + side * .0106, disc_z + side * .012))
            _part(m, self._arc(rd - .034, rd - .006, lo, hi, "brake_pad", angle=.54,
                               start=np.pi - .27, n=7), f"brake_pad_{side}")
        bridge = self._arc(rd + .004, rd + .014, disc_z - .015, disc_z + .015,
                           "caliper", angle=.48, start=np.pi - .24, n=7)
        rings = bridge.vertices.reshape(7, 5, 3)
        for i, inset in enumerate((.003, 0., .003, .006, .003, 0., .003)):
            radii = np.linalg.norm(rings[i, :, :2], axis=1)
            rings[i, [1, 2], :2] *= ((radii[[1, 2]] - inset) / radii[[1, 2]])[:, None]
        _part(caliper, bridge, "caliper_bridge")
        for k, a in enumerate((np.pi - .18, np.pi + .18)):
            # Raised piston-housing bosses break up the forged cheek. Their
            # tops stay inside the existing brake/spoke clearance envelope.
            boss = P.cylinder(.016, .005, 8, radius_top=.012, material="caliper",
                              center=((rd - .018) * np.cos(a), (rd - .018) * np.sin(a), disc_z + .0265))
            _part(caliper, boss, f"caliper_boss_{k}")
        _part(m, caliper, "caliper")
        # Machined bolt-circle recesses, with the hex heads below the mouths.
        hub = self._pierced_ring(.026, .067, zf + .004, zf + .018, 5, (.010, .010), "rim",
                                 taper=.64, round_outer=True)
        _part(m, hub, "lug_seats")
        for k in range(5):
            a = k * 2 * np.pi / 5
            _part(m, P.cylinder(.0055, .007, 6, material="rim_dark",
                                center=(.0465 * np.cos(a), .0465 * np.sin(a), zf + .0105)), f"lug_{k}")
        if opts["hubcap"]:
            _part(m, P.cylinder(.025, .009, 24, material="rim", center=(0, 0, zf + .022)), "centre_cap")
            _part(m, P.cylinder(.016, .003, 12, material="rim_dark", center=(0, 0, zf + .028)), "cap_badge")
        _part(m, P.cylinder(.004, .024, 8, material="rim_dark",
                            center=(rr * .70, -rr * .60, p.width / 2 - .022)), "valve_stem")

    def _steel_face(self, m, p, rr, opts):
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


@register
class SteelWheel(Wheel):
    name = "wheel.steel"
    default_for = ()
    options = dict(Wheel.options, spoke_family="steel_cap")
    description = "stamped steel wheel with open ventilation slots and a domed hubcap"


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
def _lamp_grid(conn):
    """Fit only sub-3mm boundary notches under the lamp's aperture seal.

    The stock sports fit stays below 2mm; mild supported modifiers need up
    to 3mm, with the physical seal expanded by the same measured allowance.
    The sports fascia has a noncorner sample that doubles back by 0.8mm in
    height. Its boundary crosses itself in the mount-plane projection, so no
    outward pane can follow it exactly. Straighten only a crossed cell's tiny
    boundary notch onto its neighboring chord; retain all four corners and
    leave the body/connector grid untouched. Larger changes are not a fit.
    """
    source = conn.meta.get("grid_points")
    if source is None:
        return None
    source = np.asarray(source, dtype=float)
    grid = source.copy()
    rows, cols = grid.shape[:2]

    def crossed(g):
        q = np.stack([g[:-1, :-1], g[1:, :-1], g[1:, 1:], g[:-1, 1:]], axis=-2)
        def area(a, b, c):
            return np.cross(q[..., b, :] - q[..., a, :], q[..., c, :] - q[..., a, :]) @ conn.normal
        a, b, c, d = area(0, 1, 2), area(0, 2, 3), area(0, 1, 3), area(1, 2, 3)
        sign = 1 if (a + b).sum() >= 0 else -1
        return (np.minimum(sign * a, sign * b) <= 1e-12) & (np.minimum(sign * c, sign * d) <= 1e-12)

    bad = crossed(grid)
    while bad.any():
        candidates = []
        for i, j in zip(*np.where(bad)):
            for r, c in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                if c in (0, cols - 1) and 0 < r < rows - 1:
                    a, b = grid[r - 1, c], grid[r + 1, c]
                elif r in (0, rows - 1) and 0 < c < cols - 1:
                    a, b = grid[r, c - 1], grid[r, c + 1]
                else:
                    continue  # Never move a corner or an interior sample.
                edge = b - a
                t = np.clip((grid[r, c] - a) @ edge / max(edge @ edge, 1e-20), 0., 1.)
                point = a + t * edge
                distance = np.linalg.norm(point - source[r, c])
                if distance > .003:
                    continue
                trial = grid.copy()
                trial[r, c] = point
                remaining = crossed(trial)
                if remaining.sum() < bad.sum():
                    candidates.append((distance, r, c, point, remaining))
        if not candidates:
            break
        _, r, c, point, bad = min(candidates, key=lambda candidate: candidate[:3])
        grid[r, c] = point
    return grid


def _lamp_wrapped_patch(conn, grid, material, offset, bulge):
    """An embedded graph for an extreme aperture, not a flipped folded grid.

    Keep the boundary and well-separated interior source samples that can
    lie on the same graph. Search outward viewing directions nearest the mount normal,
    ear-clip only a simple projected outline, then insert interior samples.
    All triangles are positive in that projection, which prevents crossings
    in 3-D too. This deliberately does not claim an extreme folded aperture
    has the same local outward direction as its single average mount frame.
    """
    if conn.side() == "right":
        # Canonicalize reflection BEFORE choosing a projection/ear order, so
        # floating-point ties cannot give left/right lamps different surfaces.
        reflect = np.array([1., -1., 1.])
        left = PolygonConnector(conn.name, Frame.from_normal(conn.origin * reflect,
                                conn.normal * reflect, conn.frame.x_axis * reflect),
                                conn.points[::-1] * reflect, meta={"side": "left"})
        mesh = _lamp_wrapped_patch(left, grid[::-1] * reflect, material, offset, bulge).mirrored()
        mesh.meta["lamp_projection_normal"] = (np.asarray(mesh.meta["lamp_projection_normal"]) * reflect).tolist()
        return mesh
    rows, cols = grid.shape[:2]
    vertices = grid.reshape(-1, 3)
    boundary = ([i * cols for i in range(rows)]
                + [(rows - 1) * cols + j for j in range(1, cols)]
                + [i * cols + cols - 1 for i in range(rows - 2, -1, -1)]
                + list(range(cols - 2, 0, -1)))

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    # A deterministic hemisphere, sampled from closest to furthest from the
    # mount normal. This is component geometry, never connector-frame repair.
    k = np.arange(384)
    z = 1 - (k + .5) / len(k)
    angle = k * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z * z)
    directions = np.vstack([[0., 0., 1.], np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])])
    for normal in directions @ conn.frame.rotation.T:
        frame = Frame.from_normal(conn.origin, normal, conn.frame.x_axis)
        xy = frame.to_local(vertices)[:, :2]
        a, b = xy[boundary], np.roll(xy[boundary], -1, axis=0)
        if cross(a, b).sum() <= 1e-12:
            continue
        edge = b - a
        c, d = a[:, None], b[:, None]
        if np.any((cross(edge, c - a) * cross(edge, d - a) < -1e-24)
                  & (cross(d - c, a - c) * cross(d - c, b - c) < -1e-24)):
            continue
        loop, faces = list(boundary), []
        while len(loop) > 3:
            ears = []
            for i, middle in enumerate(loop):
                first, last = loop[i - 1], loop[(i + 1) % len(loop)]
                a, b, c = xy[[first, middle, last]]
                if cross(b - a, c - a) <= 1e-12:
                    continue
                other = xy[[v for v in loop if v not in (first, middle, last)]]
                inside = ((cross(b - a, other - a) >= -1e-12)
                          & (cross(c - b, other - b) >= -1e-12)
                          & (cross(a - c, other - c) >= -1e-12))
                if not inside.any():
                    ears.append((np.linalg.norm(c - a), i, (first, middle, last)))
            if not ears:
                break
            _, i, face = min(ears)
            faces.append(face)
            loop.pop(i)
        if len(loop) == 3 and cross(xy[loop[1]] - xy[loop[0]], xy[loop[2]] - xy[loop[0]]) > 1e-12:
            faces.append(tuple(loop))
            break
    else:
        # A documented scalar modifier should not abort the whole assembly.
        # An unprojectable boundary gets a conservative convex-envelope pane,
        # explicitly warned below, rather than an intersecting legacy fill.
        normal = conn.normal
        xy = conn.frame.to_local(vertices)[:, :2]
        order = np.lexsort((xy[:, 1], xy[:, 0])).tolist()
        halves = []
        for sequence in (order, order[::-1]):
            hull = []
            for index in sequence:
                while len(hull) > 1 and cross(xy[hull[-1]] - xy[hull[-2]], xy[index] - xy[hull[-1]]) <= 1e-12:
                    hull.pop()
                hull.append(index)
            halves.append(hull[:-1])
        hull = halves[0] + halves[1]
        faces = [(hull[0], hull[i], hull[i + 1]) for i in range(1, len(hull) - 1)]
        mesh = Mesh(vertices.copy(), faces, [material] * len(faces), name="lamp_fit")
        mesh.prune_unused_vertices().translate(offset * normal)
        mesh.meta["lamp_fit_warning"] = "Infeasible lamp aperture: conservative convex-envelope pane; exact boundary fit is unavailable."
        mesh.meta["lamp_projection_normal"] = normal.tolist()
        return mesh

    # Preserve source curvature where its projection lies strictly inside the
    # polygon. A sample on the boundary must not pull that boundary off its
    # exact 3-D segment; projected duplicate/outer samples are not inserted.
    for index in (i * cols + j for i in range(1, rows - 1) for j in range(1, cols - 1)):
        point = xy[index]
        used = np.unique(np.asarray(faces))
        if np.linalg.norm(xy[used] - point, axis=1).min() < 1e-10:
            continue
        a, b, c = xy[np.asarray(faces)].transpose(1, 0, 2)
        areas = np.column_stack([cross(b - a, point - a), cross(c - b, point - b), cross(a - c, point - c)])
        hit = np.flatnonzero(np.all(areas > 1e-10, axis=1))
        if len(hit) != 1:
            continue  # Do not create sliver triangles on an existing edge.
        first, middle, last = faces.pop(int(hit[0]))
        faces.extend([(first, middle, index), (middle, last, index), (last, first, index)])
    u, v = np.meshgrid(np.linspace(0, 1, rows), np.linspace(0, 1, cols), indexing="ij")
    dome = (np.sin(np.pi * u) * np.sin(np.pi * v)).reshape(-1, 1)
    mesh = Mesh(vertices + offset * conn.normal + bulge * dome * normal, faces, [material] * len(faces), name="lamp_fit")
    mesh.prune_unused_vertices()
    turn = np.degrees(np.arccos(np.clip(normal @ conn.normal, -1., 1.)))
    mesh.meta["lamp_fit_warning"] = (f"Wrapped lamp aperture: embedded boundary-fit pane uses a {turn:.1f}-degree "
                                     "local projection change; extreme-aperture optics are approximate.")
    mesh.meta["lamp_projection_normal"] = normal.tolist()
    return mesh


def _lamp_fill(conn, material, offset, bulge=0.0, upsample=2):
    """Fit a pane, choosing the noncrossing diagonal of concave fascia cells."""
    from .fills import coons_fill
    grid = _lamp_grid(conn)
    if grid is None:
        return fill_connector(conn, material, offset=offset, bulge=bulge, upsample=upsample)
    mesh = coons_fill(conn.points, *grid.shape[:2], material, conn.normal, offset=offset,
                      bulge=bulge, upsample=upsample, grid_points=grid)
    quads = np.asarray(mesh.faces)
    a, b, c, d = mesh.vertices[quads].transpose(1, 0, 2)
    default = np.minimum(np.cross(b - a, c - a) @ conn.normal,
                         np.cross(c - a, d - a) @ conn.normal) > 1e-12
    alternate = np.minimum(np.cross(b - a, d - a) @ conn.normal,
                           np.cross(c - b, d - b) @ conn.normal) > 1e-12
    if not np.all(default | alternate):
        return _lamp_wrapped_patch(conn, grid, material, offset, bulge)
    faces = []
    for keep, face in zip(default, mesh.faces):
        if keep:
            faces.append(face)
        else:
            # A concave quad's default fan can invert a triangle even when its
            # Newell normal points out. Change the diagonal, not the winding.
            faces.extend([(face[0], face[1], face[3]), (face[1], face[2], face[3])])
    mesh.faces = faces
    mesh.face_materials = [material] * len(faces)
    return mesh


class _LampSurface:
    """Decorations in aperture UVs, following the exact (possibly warped) grid.

    U runs inboard -> outboard; V runs bottom -> top in WORLD space. Neither
    handedness nor the lamp's steep corner rake is guessed from local Y.
    """
    def __init__(self, conn):
        grid = _lamp_grid(conn)
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
        # The pickup's lamp is taller than it is wide. Choose the vertical
        # axis by world elevation, not by assuming the longest axis is U.
        rise = [np.ptp(grid[:, :, 2], axis=i).mean() for i in (0, 1)]
        if rise[0] > rise[1]:
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
        _part(m, _lamp_fill(conn, "lamp_dark", offset=-.056).transform(inv), "housing")
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
        pane = _lamp_fill(conn, "lens", offset=.001, bulge=.008, upsample=3)
        _part(m, pane.transform(inv), "lens")
        fitted = _lamp_grid(conn)
        deviation = float(np.linalg.norm(fitted - conn.meta["grid_points"], axis=2).max()) if fitted is not None else 0.
        seal_radius = max(.0025, (deviation + .0003) / np.cos(np.pi / 8))
        _part(m, _pipe(conn.points + conn.normal * .001, seal_radius, "lamp_dark", closed=True,
                       sides=8).transform(inv), "aperture_seal")
        m.materials.update(mats)
        info = {"projectors": 2 if opts["projector"] else 1}
        if "lamp_fit_warning" in pane.meta:
            info["fit_warnings"] = [pane.meta["lamp_fit_warning"]]
        if deviation > .002:
            info.setdefault("fit_warnings", []).append(f"Lamp boundary notch fitted by {deviation * 1000:.2f}mm under an adaptive seal.")
        return ComponentResult(m, [], info)


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
        _part(m, _lamp_fill(conn, "lamp_dark", offset=-.038).transform(inv), "housing")
        surf = _LampSurface(conn)
        _part(m, surf.border("tail_bezel", .001), "bezel")
        uv = rounded_rect_points(.78, .54, .10, 6) + [.5, .57]
        _part(m, surf.ribbon(uv, min(.012, surf.height * .08), "tail_guide", -.012, True), "light_guide")
        _part(m, surf.patch(.22, .55, .41, .57, "tail_reverse", -.015), "reverse")
        _part(m, surf.patch(.12, .88, .12, .23, "tail_reflector", -.009), "reflex_strip")
        # Small prismatic flutes are geometry, rather than a single red patch.
        for u in np.linspace(.14, .86, 19):
            _part(m, surf.ribbon([[u, .135], [u, .215]], .0025, "tail_reflector", -.007), f"reflex_flute_{u:.2f}")
        pane = _lamp_fill(conn, "tail_lens", offset=.001, bulge=.006, upsample=3)
        _part(m, pane.transform(inv), "lens")
        fitted = _lamp_grid(conn)
        deviation = float(np.linalg.norm(fitted - conn.meta["grid_points"], axis=2).max()) if fitted is not None else 0.
        if deviation > 0:
            radius = max(.0025, (deviation + .0003) / np.cos(np.pi / 8))
            _part(m, _pipe(conn.points + conn.normal * .001, radius, "lamp_dark", closed=True,
                           sides=8).transform(inv), "aperture_seal")
        m.materials.update(mats)
        info = {}
        if "lamp_fit_warning" in pane.meta:
            info["fit_warnings"] = [pane.meta["lamp_fit_warning"]]
        if deviation > .002:
            info.setdefault("fit_warnings", []).append(f"Lamp boundary notch fitted by {deviation * 1000:.2f}mm under an adaptive seal.")
        return ComponentResult(m, [], info)


# =============================================================== GRILLE / INTAKE / PLATE / BADGE
@register
class Grille(CarComponent):
    name = "grille.slats"
    accepts = (RectangleConnector,)
    default_for = ("grille",)
    options = {"slats": None, "frame": True, "mesh": False, "badge": False, "emergency_lights": False, "siren": False}
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
            # Five slats crowd the new 45mm sports opening into a chrome strip.
            # Fit the default spacing; explicit counts retain their old range.
            requested = opts["slats"]
            n = int(np.clip(min(5, int(ih / .014)) if requested is None else requested, 1, 14))
            for k in range(n):
                y = -ih / 2 + (k + .5) * ih / n
                thick = min(.012, ih / n * .32)
                cut = np.sqrt(max(0, (badge_r + .006) ** 2 - max(0, abs(y) - thick / 2) ** 2)) if badge_r else 0
                spans = [(-iw / 2, -cut), (cut, iw / 2)] if cut else [(-iw / 2, iw / 2)]
                for j, (lo, hi) in enumerate(spans):
                    _part(m, P.box(hi - lo, thick, .030, material="chrome", bevel=min(.002, thick * .2),
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
        if opts.get("emergency_lights"):
            # concealed LED modules at each end (local +x is the car's right: blue there, red on the driver's side)
            from .equipment import _lens
            lh = min(.03, ih * .5)
            for sign, lens in ((-1, "lens_red"), (1, "lens_blue")):
                x = sign * (iw / 2 - .06)
                _part(m, P.box(.095, lh + .01, .010, material="grille_dark", center=(x, 0, -.004)), f"led_bezel_{lens}")
                _part(m, P.box(.085, lh, .016, material=lens, center=(x, 0, .005)), f"led_{lens}")
                mats[lens] = _lens(lens.split("_")[1])
        if opts.get("siren"):
            # siren speaker horn behind the insert, offset to one side
            r = min(.06, ih * .45, iw * .12)
            x = iw / 2 - r - .02
            _part(m, P.cylinder(r, .03, 20, material="grille_dark", center=(x, 0, .003)), "siren_horn")
            for k in range(3):
                _part(m, P.tube(r * (.85 - .25 * k) + .004, r * (.85 - .25 * k), .006, 20, material="mesh_dark",
                                center=(x, 0, .021)), f"siren_ring_{k}")
        # These mounts clear the WHOLE stepped fascia footprint. Seat the back
        # of the 10mm radiator backing at the mount, with no obsolete rake or
        # negative offset burying it in the uncut skin.
        m.translate([0, 0, .027])
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
    options = {"region": "eu", "color": "#f2f2ee", "text_color": "#1a1a1a", "style": "standard"}
    description = "license plate (EU 520x110 or US 300x150) with a dark text band; style 'government' for fleet plates"

    def build_local(self, conn: RectangleConnector, opts, ctx) -> ComponentResult:
        opts = dict(opts)
        government = opts.get("style") == "government"
        if government:
            opts.update(region="us", color="#f3f4f0", text_color="#1d2b53")
        w, h = (0.52, 0.11) if opts.get("region", "eu") == "eu" else (0.305, 0.152)
        mats = {"plate": ctx.material("plate", opts["color"], shininess=0.4),
                "plate_text": ctx.material("plate_text", opts["text_color"], shininess=0.2),
                "plate_blue": ctx.material("plate_blue", "#1c3d8f", shininess=0.3)}
        m = P.box(w, h, 0.006, material="plate", center=(0, 0, 0.003), name="plate")
        if government:
            m.merge(P.box(w * 0.92, h * 0.14, 0.002, material="plate_text", center=(0, h * 0.38, 0.007), name="govband"))
            m.merge(P.box(w * 0.5, h * 0.09, 0.002, material="plate_text", center=(0, -h * 0.40, 0.007), name="govtext"))
        # crude "characters": 7 dark blocks
        n = 7
        cw = w * 0.09
        for k in range(n):
            x = -w * 0.36 + k * (w * 0.78) / (n - 1)
            m.merge(P.box(cw, h * 0.55, 0.002, material="plate_text", center=(x, 0, 0.007), name="char"))
        if opts.get("region", "eu") == "eu":
            m.merge(P.box(w * 0.08, h - 0.01, 0.002, material="plate_blue", center=(-w / 2 + w * 0.05, 0, 0.007), name="euband"))
        if conn.meta.get("position") == "front":
            # The new bumper separates the EU plate from both inserts. Its
            # bracket reaches back through the mount's 12mm skin clearance;
            # the old 65mm pedestal needlessly floated the plate off the car.
            projection = .006
            # A raised bumper can bring even the EU plate into the intake's
            # footprint. Recover its fitted vertical span from the measured
            # bottom and this plate's crease-relative mount (not a style name).
            intake_z = ctx.measurements.get("nose_z_bottom", -np.inf) + .105
            crease_z = conn.origin[2] + .075
            intake_h = max(.05, min(.14, 2 * (crease_z - .145 - intake_z)))
            grille_lo = crease_z + .030
            grille_hi = ctx.measurements.get("nose_z_top", np.inf) - .045
            grille_z = (grille_lo + grille_hi) / 2
            grille_h = max(.045, min(.17, grille_hi - grille_lo))
            overlaps_lower = conn.origin[2] - h / 2 < intake_z + intake_h / 2 + .005
            overlaps_upper = conn.origin[2] + h / 2 > grille_z - grille_h / 2 - .005
            if h > conn.height or overlaps_lower or overlaps_upper:
                # Clear the entire 41mm insert depth, including its foremost
                # bumper step, when either plate format overlaps an insert.
                projection = max(projection, ctx.measurements.get("x_front", conn.origin[0])
                                 - conn.origin[0] + .050)
            depth = projection + .012
            _part(m, P.box(w * .82, h * .72, depth, material="plate_text", center=(0, 0, -depth / 2)), "mounting_plinth")
            m.translate([0, 0, projection])
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
        def tip(cx):
            t = P.tube(r, r - 0.006, L, 20, material="chrome", center=(cx, 0, L / 2 - 0.03), name="tip")
            inner = P.cylinder(r - 0.006, 0.004, 20, material="exhaust_dark", center=(cx, 0, L - 0.03 - 0.01), name="tip_inner")
            return t.merge(inner)
        m = tip(0.0)
        dual = bool(opts.get("dual"))
        if dual:
            # Rear local X is lateral; local Y points down. Stacking along Y
            # buries the upper twin in the skirt even when the single tip fits.
            m = tip(-r * 1.2).merge(tip(r * 1.2))
        # Join the unchanged central pipe inlet to the sleeve(s): a circular
        # ferrule for one outlet, an oval collector for two separate mouths.
        # Twice the circumference budget for twins retains the public 2:1
        # single/dual topology contract, including these inlet connections.
        n = 20 if dual else 10
        circle = circle_points(1., n)
        rings = [np.column_stack([circle[:, 0] * rx, circle[:, 1] * ry, np.full(n, z)])
                 for z, rx, ry in ((-.045, r * .78, r * .78),
                                   (.005, r * (1.95 if dual else .84), r * .84))]
        _part(m, P.loft(rings, cap_start=True, cap_end=True, material="exhaust_dark"),
              "collector" if dual else "inlet")
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
    options = {"housing_color": None, "emergency_light": False}
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
        if opts.get("emergency_light"):
            from .equipment import _lens, _lens_for
            lens = _lens_for(conn.meta.get("side", "left"))
            _part(m, P.box(.014, .022, .048, material=lens, center=(.079, .025, .192)), "emergency_lens")
            mats[lens] = _lens(lens.split("_")[1])
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
                # Fit an oval can between the road and the measured tunnel,
                # not just above the road: a 118mm round silencer otherwise
                # pokes through a crown only 33mm above the pipe centreline.
                if conn.meta.get("paths_space") == "world":
                    up = conn.frame.rotation[2]
                    height = conn.frame.to_world(point)[0, 2]
                    vertical = (can.vertices - point) @ up
                    factor = min(1., max(0., height - .050) / max(-vertical.min(), 1e-6))
                    if "floor_grid" in conn.meta:
                        grid = np.asarray(conn.meta["floor_grid"])
                        world = conn.frame.to_world(can.vertices)
                        lo, hi = world[:, 0].min(), world[:, 0].max()
                        lateral = np.abs(world[:, 1]).max()
                        # The lowest roof over the entire can footprint,
                        # including station breaks between its end rings.
                        xs = grid[:, 0, 0]
                        samples = np.r_[lo, xs[(xs > lo) & (xs < hi)], hi]
                        roof = []
                        for x in samples:
                            row = np.array([[np.interp(x, chain[:, 0], chain[:, k]) for k in (1, 2)]
                                            for chain in grid.transpose(1, 0, 2)])
                            roof.append(np.interp(lateral, row[:, 0], row[:, 1]))
                        room = max(0., min(roof) - .008 - height)
                        factor = min(factor, room / max(vertical.max(), 1e-6))
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
