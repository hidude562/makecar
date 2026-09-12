"""Fixed-topology parametric car-body loft.

The body is a loft of cross-section *rings* along the length of the car.  Each
ring is sampled from a small set of semantic control points (floor edge, sill,
belt line, shoulder, roof rail, roof centre ...) with a fixed number of samples
per segment, so vertex `j` of every ring always represents the same feature and
the whole mesh has identical topology for any parameter set.  That is what makes
the MakeHuman-style target system possible: any two parameterisations differ
only by a per-vertex displacement.

Two independent longitudinal "chains" of feature positions are used: the lower
part of each ring (floor, wheel wells, sill) follows the *arch chain* (axles and
arch chords) while the upper part (glass base, pillars, roof) follows the
*greenhouse chain* (deck, roof ends, pillars, cowl).  Both chains have the same
number of stations; a ring simply connects station i of one chain to station i
of the other, which lets wheel arches and window apertures both have clean
edges without a variable topology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from ..geometry.mesh import Mesh, Material
from ..geometry.curves import Profile, catmull_rom_closed, smoothstep
from .params import BodyParams

# ----------------------------------------------------------------------------
# Ring layout: semantic control points of a half section and samples/segment.
# ----------------------------------------------------------------------------
SEGMENTS = [
    ("A", "B", 4),   # tunnel crown -> tunnel shoulder -> flat floor edge
    ("B", "C", 2),   # floor edge -> sill bottom / wheel-well inner top
    ("C", "D", 3),   # sill face / wheel-well ceiling -> sill top (arch lip)
    ("D", "E", 8),   # arch flange / door hem, then door skin -> belt
    ("E", "F", 5),   # three-sample shoulder radius -> glass base / hood crease
    ("F", "G", 10),  # frame, recessed glass / pillar, frame -> roof rail
    ("G", "G2", 4),  # roof edge radius / A-pillar band -> roof (or bed rail inner)
    ("G2", "G3", 3),
    ("G3", "H", 4),  # -> roof centre
]
CTRL_NAMES = ["A", "B", "C", "D", "E", "F", "G", "G2", "G3", "H"]


def _ring_indices():
    idx, acc = {}, 0
    for a, b, n in SEGMENTS:
        idx[a] = acc
        acc += n
    idx["H"] = acc
    return idx, acc


RING, HALF_N = _ring_indices()      # RING["F"] = half-ring index of F ; HALF_N = 32
RING_N = 2 * HALF_N                  # 64 vertices per full ring


def mirror_index(j: int) -> int:
    """Ring index of the mirror (y -> -y) of half-ring index j."""
    return j if j in (0, HALF_N) else RING_N - j


# ----------------------------------------------------------------------------
# Station allocation (fixed counts per chain interval)
# ----------------------------------------------------------------------------
UPPER_COUNTS = [6, 8, 4, 1, 8, 1, 8, 10, 12]  # glass boundaries have dedicated frame rows
LOWER_COUNTS = [6, 6, 6, 22, 6, 6, 6]          # arch chords retain independent stations
assert sum(UPPER_COUNTS) == sum(LOWER_COUNTS)
N_STATIONS = sum(UPPER_COUNTS) + 1              # 59 rings in the main loft
N_FASCIA = 6                                    # corner, face, pocket rim and pocket floor
UPPER_NAMES = ["rear", "deck", "roof_rear", "cp_r", "cp_f", "bp_r", "bp_f", "roof_front", "cowl", "front"]
LOWER_NAMES = ["rear", "ra_start", "ra", "ra_end", "fa_start", "fa", "fa_end", "front"]


def _station_positions(breakpoints: List[float], counts: List[int]) -> np.ndarray:
    xs = []
    for k, n in enumerate(counts):
        a, b = breakpoints[k], breakpoints[k + 1]
        xs.extend(np.linspace(a, b, n, endpoint=False))
    xs.append(breakpoints[-1])
    return np.asarray(xs)


def station_index(name: str, chain: str = "upper") -> int:
    """Ring index of a named breakpoint ('cowl', 'ra', ...)."""
    names, counts = (UPPER_NAMES, UPPER_COUNTS) if chain == "upper" else (LOWER_NAMES, LOWER_COUNTS)
    k = names.index(name)
    return int(sum(counts[:k]))


PAINT, GLASS_ZONE, UNDERBODY, TRIM = "paint", "aperture", "underbody", "trim"


class BodyGenerator:
    """params -> fixed-topology body Mesh (with zones, groups, landmarks)."""

    def __init__(self, params: BodyParams):
        self.p = params
        self.L = params.layout()
        self.x_lo = _station_positions(self.L["lower"], LOWER_COUNTS)
        self.x_hi = _station_positions(self.L["upper"], UPPER_COUNTS)
        # Reserve a 25 mm painted frame at both ends of every glass patch.
        # Interior samples still follow the roof curve, rather than a long quad
        # whose top edge would be a straight chord across the whole door.
        for start, end in (("deck", "roof_rear"), ("roof_rear", "cp_r"),
                           ("cp_f", "bp_r"), ("bp_f", "roof_front"), ("roof_front", "cowl")):
            a, b = station_index(start), station_index(end)
            frame = min(0.025, (self.x_hi[b] - self.x_hi[a]) * 0.18)
            self.x_hi[a + 1:b] = np.linspace(self.x_hi[a] + frame, self.x_hi[b] - frame, b - a - 1)
        assert len(self.x_lo) == len(self.x_hi) == N_STATIONS
        self._make_profiles()

    # ------------------------------------------------------------ profiles
    def _make_profiles(self):
        p, L = self.p, self.L
        xf, xr = L["x_front"], L["x_rear"]
        half_w = p.width / 2

        # centreline silhouette (hood - windshield - roof - rear window - deck)
        self.z_center = Profile(
            [
                (xr, p.deck_rear_height),
                (L["u_deck"], p.deck_height),
                (L["u_roof_rear"], p.roof_height - p.roof_drop),
                (L["u_roof_rear"] + 0.35 * (L["u_roof_front"] - L["u_roof_rear"]), p.roof_height - 0.4 * p.roof_drop),
                (L["u_roof_front"], p.roof_height),
                (L["u_cowl"], p.cowl_height),
                (xf - 0.18, p.hood_front_height + 0.035),
                (xf, p.hood_front_height),
            ],
            sharp=[False, True, False, False, False, True, False, False],
        )
        # transverse crown of the top surface
        self.crown = Profile(
            [
                (xr, p.deck_crown),
                (L["u_deck"], p.deck_crown),
                (L["u_deck"] + 0.5 * (L["u_roof_rear"] - L["u_deck"]), p.roof_crown * 1.3),
                (L["u_roof_rear"], p.roof_crown),
                (L["u_roof_front"], p.roof_crown),
                (L["u_roof_front"] + 0.5 * (L["u_cowl"] - L["u_roof_front"]), p.roof_crown * 1.4),
                (L["u_cowl"], p.hood_crown),
                (xf, p.hood_crown),
            ]
        )
        # belt line (door shoulder) before clamping to the hood/deck
        self.belt = Profile(
            [
                (xr, p.belt_height + p.belt_rake),
                (L["u_bp_r"], p.belt_height + p.belt_rake),
                (L["u_bp_f"], p.belt_height + 0.3 * p.belt_rake),
                (L["u_cowl"], p.belt_height),
                (xf, p.belt_height),
            ]
        )
        # rocker / lower body edge
        self.sill = Profile(
            [
                (xr, p.rear_bumper_bottom + 0.10),
                (xr + p.rear_corner_length, p.sill_height),
                (xf - p.front_corner_length, p.sill_height),
                (xf, p.front_bumper_bottom + 0.10),
            ]
        )
        # underbody
        self.floor = Profile(
            [
                (xr, p.rear_bumper_bottom - p.rear_valance_height),
                (xr + max(0.3, p.rear_corner_length * 0.7), p.ground_clearance),
                (xf - max(0.3, p.front_corner_length * 0.7), p.ground_clearance),
                (xf, p.front_bumper_bottom - p.air_dam_height),
            ]
        )
        # roof rail / top-corner half width
        roof_y = half_w * p.roof_width_ratio
        hood_y = half_w - p.shoulder_inset - 0.10
        self.roof_edge = Profile(
            [
                (xr, hood_y),
                (L["u_deck"], hood_y),
                (L["u_roof_rear"], roof_y),
                (L["u_roof_front"], roof_y),
                (L["u_cowl"], hood_y),
                (xf, hood_y),
            ]
        )
        # pillar lean (x shift of the glass base relative to the roof rail)
        lean_a = max(0.0, min(p.a_pillar_lean, 1.0) * p.windshield_length - p.a_pillar_width)
        lean_c = p.c_pillar_lean * p.rear_window_length
        self.lean = Profile(
            [
                (xr, 0.0),
                (L["u_deck"], 0.0),
                (L["u_roof_rear"], -lean_c),
                (min(L["u_roof_rear"] + 0.6, L["u_bp_r"]), 0.0),
                (max(L["u_roof_front"] - 0.6, L["u_bp_f"]), 0.0),
                (L["u_roof_front"], lean_a),
                (L["u_cowl"], 0.0),
                (xf, 0.0),
            ],
            sharp=[True] * 8,  # a narrow pillar must not fold back at the cowl
        )
        # pickup bed weight/depth
        if p.bed_depth > 0:
            x0, x1 = xr + 0.06, L["u_deck"] - 0.04
            self.bed = Profile([(xr, 0.0), (x0, 1.0), (x1, 1.0), (L["u_deck"] + 0.02, 0.0), (xf, 0.0)],
                               sharp=[True, True, True, True, True])
        else:
            self.bed = None

    # ------------------------------------------------------------ helpers
    def plan_half_width(self, x) -> np.ndarray:
        """Half width of the body at belt height, including nose/tail taper."""
        p, L = self.p, self.L
        x = np.asarray(x, dtype=float)
        hw = p.width / 2
        y = np.full_like(x, hw)
        # gentle mid-body taper towards the ends (2%)
        xf, xr = L["x_front"], L["x_rear"]
        tf = np.clip((x - (xf - p.front_corner_length)) / max(p.front_corner_length, 1e-6), 0, 1)
        tr = np.clip(((xr + p.rear_corner_length) - x) / max(p.rear_corner_length, 1e-6), 0, 1)
        ell_f = np.sqrt(np.clip(1 - tf ** 2.2, 0, 1))
        ell_r = np.sqrt(np.clip(1 - tr ** 2.2, 0, 1))
        y = hw * (p.nose_width_ratio + (1 - p.nose_width_ratio) * ell_f)
        y = np.minimum(y, hw * (p.tail_width_ratio + (1 - p.tail_width_ratio) * ell_r))
        mid = 0.5 * (xf + xr)
        y = y * (1 - 0.02 * np.abs((x - mid) / ((xf - xr) / 2)) ** 2)
        return y

    def arch(self, x_lo: float) -> Tuple[float, float]:
        """(weight, z of arch lip) for the lower-chain position x_lo."""
        p, L = self.p, self.L
        r, half = L["arch_radius"], L["arch_half"]
        for xa in (L["x_fa"], L["x_ra"]):
            dx = x_lo - xa
            if abs(dx) <= half + 1e-9:
                z = p.axle_height + np.sqrt(max(r * r - dx * dx, 0.0))
                prof = np.sqrt(max(1 - (dx / half) ** 2, 0.0))
                return 1.0, max(z, self.sill(x_lo)), prof
        return 0.0, self.sill(x_lo), 0.0

    # --------------------------------------------------------------- rings
    def control_points(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Half-section control points for station i -> (pts (10,3), sharpness (10,))."""
        p = self.p
        xl, xh = float(self.x_lo[i]), float(self.x_hi[i])
        xm = 0.5 * (xl + xh)
        yb_lo = float(self.plan_half_width(xl))
        yb_mid = float(self.plan_half_width(xm))
        yb_hi = float(self.plan_half_width(xh))

        zf = float(self.floor(xl))
        z_sill_base = float(self.sill(xl))
        y_sill = yb_lo - p.sill_inset
        well_w = p.tire_width + 0.12
        y_fe = max(0.30, y_sill - well_w)
        w_arch, z_arch, arch_prof = self.arch(xl)

        zc = float(self.z_center(xh))
        cr = float(self.crown(xh))
        z_belt = min(float(self.belt(xm)), zc - cr - 0.02)
        z_shoulder = min(z_belt + max(p.shoulder_rise, p.shoulder_radius), zc - cr - 0.005)
        y_belt = yb_mid + (p.side_bulge if w_arch == 0 else 0.0) * 0.5
        y_F = yb_hi - p.shoulder_inset
        y_G = min(float(self.roof_edge(xh)), yb_hi - p.shoulder_inset - 0.04)
        z_G = zc - cr
        lean = float(self.lean(xh))
        bed_w = float(self.bed(xh)) if self.bed is not None else 0.0

        # control points --------------------------------------------------
        A = (xl, 0.0, zf)
        B = (xl, y_fe, zf)
        if w_arch > 0:
            C = (xl, y_fe, z_arch - 0.01)
            D = (xl, y_sill + p.fender_flare * arch_prof, z_arch)
        else:
            C = (xl, y_sill, max(zf + 0.015, z_sill_base - p.rocker_height))
            D = (xl, y_sill, z_sill_base)
        E = (xm, y_belt, z_belt)
        F = (xh + lean, y_F, z_shoulder)
        G = (xh, y_G, z_G)
        # roof arc points (parabolic crown) vs. bed rail
        def arc(t):
            return (xh, y_G * (1 - t), z_G + cr * (1 - (1 - t) ** 2))
        G2a, G3a = arc(1 / 3), arc(2 / 3)
        if bed_w > 0:
            rail_w = 0.07
            z_bed = zc - p.bed_depth * bed_w
            G2b = (xh, y_G - rail_w, z_G)
            G3b = (xh, y_G - rail_w, z_bed)
            Hb = (xh, 0.0, z_bed)
            G2 = tuple(np.add(np.multiply(G2a, 1 - bed_w), np.multiply(G2b, bed_w)))
            G3 = tuple(np.add(np.multiply(G3a, 1 - bed_w), np.multiply(G3b, bed_w)))
            H = tuple(np.add(np.multiply((xh, 0.0, zc), 1 - bed_w), np.multiply(Hb, bed_w)))
        else:
            G2, G3, H = G2a, G3a, (xh, 0.0, zc)
        pts = np.array([A, B, C, D, E, F, G, G2, G3, H], dtype=float)
        sharp = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          0.45 + 0.55 * bed_w, bed_w, bed_w, 1.0])
        return pts, sharp

    def ring(self, i: int) -> np.ndarray:
        pts, sharp = self.control_points(i)
        # mirror to a closed control polygon: A .. H, then H's mirror side back to A
        mir = pts[1:-1][::-1].copy()
        mir[:, 1] *= -1
        full = np.vstack([pts, mir])
        full_sharp = np.concatenate([sharp, sharp[1:-1][::-1]])
        counts = [n for _, _, n in SEGMENTS]
        full_counts = counts + counts[::-1]
        yz = catmull_rom_closed(full[:, 1:3], full_counts, full_sharp)
        # x: linear along each segment (keeps the pillar lean exactly where intended)
        xs = []
        n_ctrl = len(full)
        for k in range(n_ctrl):
            k2 = (k + 1) % n_ctrl
            t = np.linspace(0, 1, full_counts[k], endpoint=False)
            xs.append(full[k, 0] + (full[k2, 0] - full[k, 0]) * t)
        x = np.concatenate(xs)
        ring = np.column_stack([x, yz[:, 0], yz[:, 1]])
        self._section_details(ring, i)
        # force exact symmetry and exact centre points
        ring[0, 1] = 0.0
        ring[HALF_N, 1] = 0.0
        return ring

    def _section_details(self, ring: np.ndarray, i: int):
        """Explicit small-radius sheet-metal details, sampled at fixed indices.

        Splines suit the big panels, but overshoot a folded flange or a sill.
        Those sections use straight faces and circular arcs instead.
        """
        p = self.p
        jC, jD, jE, jF, jG = (RING[k] for k in ("C", "D", "E", "F", "G"))
        xl, xh = self.x_lo[i], self.x_hi[i]
        tunnel = (smoothstep(self.L["x_rear"] + 0.30, self.L["x_ra"], xl) *
                  (1 - smoothstep(self.L["x_fa"], self.L["x_front"] - 0.30, xl)))
        floor = ring[RING["B"]].copy()
        half_tunnel = min(p.tunnel_width / 2, floor[1] * 0.75)
        ring[RING["A"], 2] = floor[2] + p.tunnel_height * tunnel
        ring[1] = [xl, half_tunnel * 0.65, floor[2] + p.tunnel_height * tunnel]
        ring[2] = [xl, half_tunnel, floor[2]]
        ring[3] = [xl, (half_tunnel + floor[1]) / 2, floor[2]]
        arch = self.arch(xl)[0] > 0
        D, E, F, G = ring[[jD, jE, jF, jG]].copy()
        if arch:
            axle = min((self.L["x_ra"], self.L["x_fa"]), key=lambda x: abs(xl - x))
            radial = D[[0, 2]] - [axle, p.axle_height]
            radial /= max(np.linalg.norm(radial), 1e-9)
            # D remains the inner lip circle used by wheel connectors. The
            # first strip is a flange of uniform radial width; the next returns
            # inward to the fender skin rather than inflating the whole arch.
            outer = D.copy()
            outer[[0, 2]] += p.arch_lip_width * radial
            ring[jD + 1] = outer
            root = outer.copy()
            root[1] -= p.arch_lip_width * 0.6
            root[[0, 2]] += 0.004 * radial
            ring[jD + 2] = root
        else:
            # Vertical rocker, then the small outward step of the door hem.
            for k in range(jC + 1, jD):
                t = (k - jC) / (jD - jC)
                ring[k] = ring[jC] * (1 - t) + D * t
            ring[jD + 1] = D + [0, p.door_step, 0.009]
            ring[jD + 2] = D + [0, p.door_step + 0.008, 0.030]
            root = ring[jD + 2].copy()
        for k in range(jD + 3, jE):
            t = (k - jD - 2) / (jE - jD - 2)
            ring[k] = root * (1 - t) + E * t
            ring[k, 1] += p.side_bulge * np.sin(np.pi * t) * (0.0 if arch else 0.5)

        radius = max(0.001, min(p.shoulder_radius, F[2] - E[2], E[1] - F[1]))
        # Quarter circle turning the vertical door skin onto the shoulder.
        for k, angle in enumerate((np.pi / 6, np.pi / 3, np.pi / 2), start=1):
            t = k / (jF - jE)
            ring[jE + k] = E * (1 - t) + F * t
            ring[jE + k, 1] = E[1] - radius * (1 - np.cos(angle))
            ring[jE + k, 2] = E[2] + radius * np.sin(angle)
        ring[jF - 1] = (ring[jE + 3] + F) / 2

        on_hood = xh >= self.L["u_cowl"]
        on_deck = xh <= self.L["u_deck"] and self.bed is None
        if on_hood or on_deck:
            ring[jF, 2] += p.fender_crease
            # A 6 mm-wide fold, followed by a smooth hood/deck crown.
            root = F.copy()
            root[1] -= 0.006
            ring[jF + 1] = root
            for k in range(jF + 2, jG):
                t = (k - jF - 1) / (jG - jF - 1)
                ring[k] = root * (1 - t) + G * t
        elif self.L["u_deck"] < xh < self.L["u_cowl"]:
            t_roof = np.clip((xh - self.L["u_roof_rear"]) /
                             max(self.L["u_roof_front"] - self.L["u_roof_rear"], 1e-9), 0, 1)
            rail_arch = 0.018 * np.sin(np.pi * t_roof)
            G[2] += rail_arch
            ring[RING["G2"], 2] += rail_arch * 0.6
            ring[RING["G3"], 2] += rail_arch * 0.15
            radius = p.roof_edge_radius
            ring[jG] = G - [0, 0, radius / 2]
            # Circular rolled edge and a raised drip bead. G2 remains on the
            # roof crown so changing the edge never inflates the whole roof.
            ring[jG + 1] = G + [0, -radius * (1 - np.cos(np.pi / 4)),
                                    radius * (np.sin(np.pi / 4) - 0.5)]
            ring[jG + 1, 1] += p.drip_rail * 0.5
            ring[jG + 1, 2] += p.drip_rail
            ring[jG + 2] = G + [0, -max(radius, p.a_pillar_width * 0.5), radius * 0.5]
            ring[jG + 3] = (ring[jG + 2] + ring[RING["G2"]]) / 2
            # Glass boundaries are 20 mm from the belt and roof rail. The
            # remaining eight vertices trace the curved strip, not the frame.
            rail = ring[jG].copy()
            band = min(0.08, 0.020 / max(np.linalg.norm(rail - F), 1e-9))
            ts = np.r_[0.0, np.linspace(band, 1 - band, jG - jF - 1)]
            for k, t in enumerate(ts):
                ring[jF + k] = F * (1 - t) + rail * t
                ring[jF + k, 1] += 0.003 * np.sin(np.pi * t)
        # Do not let independent spline evaluation introduce asymmetric flanges.
        for j in range(1, HALF_N):
            ring[mirror_index(j)] = ring[j] * [1, -1, 1]

    def _glass_regions(self):
        """Rectangular aperture cells, excluding the painted perimeter bands."""
        jF, jG = RING["F"], RING["G"]
        regions = {}
        top = range(jG + 2, RING_N - jG - 2)
        for name, start, end in (("windshield", "roof_front", "cowl"), ("rear_window", "deck", "roof_rear")):
            regions[f"aperture/{name}"] = (range(station_index(start) + 1, station_index(end) - 1), top)
        for name, start, end in (("front", "bp_f", "roof_front"), ("rear", "cp_f", "bp_r"),
                                  ("quarter", "roof_rear", "cp_r")):
            if name == "quarter" and self.p.quarter_window_length <= 0.15:
                continue
            rows = range(station_index(start) + 1, station_index(end) - 1)
            regions[f"aperture/glass_{name}_L"] = (rows, range(jF + 1, jG - 1))
            regions[f"aperture/glass_{name}_R"] = (rows, range(RING_N - jG + 1, RING_N - jF - 1))
        return regions

    def _recess_glass(self, rings):
        """Sink every glass vertex, including its boundary, along its normal.

        Adjacent frame vertices are left on the original shell. The intervening
        narrow face is a real reveal, so the aperture connector and its grid
        inherit the recess without a component-specific displacement trick.
        """
        surface = np.asarray(rings)
        along = np.gradient(surface, axis=0)
        around = np.roll(surface, -1, axis=1) - np.roll(surface, 1, axis=1)
        normals = np.cross(around, along)
        normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-9)
        glass = np.zeros(surface.shape[:2], dtype=bool)
        for rows, cols in self._glass_regions().values():
            glass[rows.start:rows.stop + 1, cols.start:cols.stop + 1] = True
        surface[glass] -= self.p.glass_recess * normals[glass]
        # Exact mirror symmetry is part of the fixed-target contract.
        for j in range(1, HALF_N):
            surface[:, mirror_index(j)] = surface[:, j] * [1, -1, 1]
        return list(surface)

    # --------------------------------------------------------------- build
    def build(self) -> Mesh:
        p, L = self.p, self.L
        rings = [self.ring(i) for i in range(N_STATIONS)]
        # nose / tail rake: shift the top rearwards progressively over the corner run
        rings = self._apply_rake(rings)
        rings = self._recess_glass(rings)
        # fascia rings and flat pocket centres
        rear_fascia, rear_apex = self._fascia(rings[0], forward=False)
        front_fascia, front_apex = self._fascia(rings[-1], forward=True)
        all_rings = rear_fascia[::-1] + rings + front_fascia
        n_rings = len(all_rings)
        verts = np.vstack(all_rings + [rear_apex[None, :], front_apex[None, :]])
        rear_apex_i, front_apex_i = n_rings * RING_N, n_rings * RING_N + 1
        off = len(rear_fascia)  # index offset of main ring 0 inside all_rings

        faces: List[Tuple[int, ...]] = []
        mats: List[str] = []
        face_grid: Dict[Tuple[int, int], int] = {}
        for r in range(n_rings - 1):
            a, b = r * RING_N, (r + 1) * RING_N
            for j in range(RING_N):
                j2 = (j + 1) % RING_N
                face_grid[(r, j)] = len(faces)
                faces.append((a + j, b + j, b + j2, a + j2))
                mats.append(PAINT)
        # caps
        for j in range(RING_N):
            j2 = (j + 1) % RING_N
            faces.append((rear_apex_i, j2, j))
            mats.append(PAINT)
        last = (n_rings - 1) * RING_N
        for j in range(RING_N):
            j2 = (j + 1) % RING_N
            faces.append((front_apex_i, last + j, last + j2))
            mats.append(PAINT)

        mesh = Mesh(verts, faces, mats, name="body")
        n_loft = (n_rings - 1) * RING_N
        mesh.meta.update({"ring_offset": off, "n_rings": n_rings, "n_main": N_STATIONS, "ring_n": RING_N,
                          "n_loft_faces": n_loft})
        # orientation: make the loft normals point outward, then wind each cap to face its apex direction
        c = verts.mean(axis=0)
        fn = mesh.face_normals()
        fc = mesh.face_centroids()
        if np.mean(np.sum(fn[:n_loft] * (fc[:n_loft] - c), axis=1)) < 0:
            mesh.flip_normals()
        fn = mesh.face_normals()
        for cap, sign in ((range(n_loft, n_loft + RING_N), -1.0), (range(n_loft + RING_N, n_loft + 2 * RING_N), 1.0)):
            idx = list(cap)
            if np.mean(fn[idx, 0]) * sign < 0:
                for i in idx:
                    mesh.faces[i] = tuple(reversed(mesh.faces[i]))
        self._assign_zones(mesh, face_grid, off)
        self._assign_groups(mesh, off, n_rings, rear_apex_i, front_apex_i)
        return mesh

    # ------------------------------------------------------------ details
    def _apply_rake(self, rings: List[np.ndarray]) -> List[np.ndarray]:
        """Keep the bumper upright; rake only the upper nose and tail panels.

        The nominal ends include the bumper/lip, not an extra cap extension.
        Arch vertices are untouched because their stations are outside the corner.
        """
        p, L = self.p, self.L
        out = []
        for i, ring in enumerate(rings):
            r = ring.copy()
            # Each vertex follows its own longitudinal chain (important at arches).
            for forward, end, run in ((True, L["x_front"], p.front_corner_length),
                                      (False, L["x_rear"], p.rear_corner_length)):
                sign = 1 if forward else -1
                w = np.clip(1 - sign * (end - r[:, 0]) / max(run, 1e-6), 0, 1) ** 2
                reserve = p.bumper_projection + (p.front_splitter if forward else 0.0)
                r[:, 0] -= sign * reserve * w
                crease = p.bumper_crease_height if forward else p.rear_bumper_crease_height
                top = p.hood_front_height if forward else p.deck_rear_height
                rake = p.front_fascia_rake if forward else -p.rear_fascia_rake
                upper = np.clip((r[:, 2] - crease) / max(top - crease, 0.08), 0, 1)
                r[:, 0] -= sign * rake * w * upper
            out.append(r)
        return out

    def _fascia_x(self, y, z, forward: bool):
        """Piecewise bumper section: lip, upright face, upper setback, hood lip."""
        p, L = self.p, self.L
        if forward:
            bottom, top, crease = p.front_bumper_bottom, p.hood_front_height, p.bumper_crease_height
            # A narrow crease, not a rake through the entire bumper volume.
            upper = smoothstep(crease, crease + 0.025, z)
            hood = smoothstep(top - 0.035, top - 0.008, z)
            lip = 1 - smoothstep(bottom + 0.012, bottom + 0.045, z)
            return (L["x_front"] - p.front_splitter + p.front_splitter * lip
                    - (p.front_fascia_rake + p.hood_overhang) * upper + p.hood_overhang * hood)
        bottom, crease = p.rear_bumper_bottom, p.rear_bumper_crease_height
        valance = 1 - smoothstep(bottom + 0.04, bottom + 0.10, z)
        panel = smoothstep(crease, crease + 0.035, z)
        return (L["x_rear"] + p.diffuser_step * valance
                + (p.tailgate_panel - p.rear_fascia_rake) * panel)

    def _fascia(self, ring: np.ndarray, forward: bool) -> Tuple[List[np.ndarray], np.ndarray]:
        """Rounded bumper perimeter closing onto a *flat* rectangular pocket.

        The final fan is coplanar with the pocket floor: its apex is a useful
        centre landmark, not a pointed nose.  Corresponding rings exist even
        when a recess or splitter has zero depth, preserving target topology.
        """
        p = self.p
        bottom = (p.front_bumper_bottom - p.air_dam_height if forward
                  else p.rear_bumper_bottom - p.rear_valance_height)
        top = p.hood_front_height if forward else p.deck_rear_height
        crease = p.bumper_crease_height if forward else p.rear_bumper_crease_height
        hw = max(abs(ring[:, 1]))
        zmid = (bottom + top) / 2

        def outline(width, lo, hi, shoulder):
            # Semantic samples retain the lamp corner (D..G) and flat top/bottom.
            radius = min(0.045, width * 0.12, (hi - lo) * 0.18)
            pts = np.array([(0, lo), (width - radius, lo), (width, lo + radius),
                            (width, lo + 2 * radius), (width, np.clip(shoulder, lo + 2.5 * radius, hi - radius)),
                            (width - radius * 0.3, hi - radius * 0.3), (width - radius, hi),
                            (width * 2 / 3, hi), (width / 3, hi), (0, hi)])
            full = np.vstack([pts, pts[1:-1][::-1] * [-1, 1]])
            counts = [n for _, _, n in SEGMENTS]
            return catmull_rom_closed(full, counts + counts[::-1], np.ones(len(full)))

        face_yz = outline(hw * 0.985, bottom, top, crease)
        face = np.column_stack([self._fascia_x(face_yz[:, 0], face_yz[:, 1], forward), face_yz])
        # The first roll carries wraparound lamps into the face. The next roll
        # gives the bumper a rounded edge without a pillow-shaped central cap.
        corner = ring.copy()
        corner[:, 1] *= 0.995
        corner[:, 2] = zmid + (corner[:, 2] - zmid) * 0.995
        corner[:, 0] += (1 if forward else -1) * p.bumper_projection * 0.55
        out = [corner, face]
        for k, (width, lo, hi, depth) in enumerate((
                (hw * 0.87, bottom + 0.035, top - 0.025, 0.0),
                (0.32, zmid - 0.115, zmid + 0.115, 0.0),
                (0.29, zmid - 0.09, zmid + 0.09, p.plate_recess),
                (0.20, zmid - 0.06, zmid + 0.06, p.plate_recess))):
            if forward and k > 0:
                # Full-width upper grille landing, then the bumper crease.
                # No artificial licence-plate-shaped dent in the front face.
                width = hw * (0.80, 0.77, 0.46)[k - 1]
                lo = bottom + (0.07, 0.09, 0.13)[k - 1]
                hi = min(top - 0.04, crease + (0.025, 0.0, -0.025)[k - 1])
                hi = max(lo + 0.02, hi)
            yz = outline(width, lo, hi, crease if k == 0 else zmid)
            x = self._fascia_x(yz[:, 0], yz[:, 1], forward)
            if not forward and k >= 1:
                x = np.full(len(yz), self.L["x_rear"] + depth)
            out.append(np.column_stack([x, yz]))
        if forward:
            zmid = float((out[-1][:, 2].min() + out[-1][:, 2].max()) / 2)
        apex_x = float(self._fascia_x(0.0, zmid, forward)) if forward else self.L["x_rear"] + p.plate_recess
        return out, np.array([apex_x, 0.0, zmid])

    def _assign_zones(self, mesh: Mesh, face_grid, off: int):
        """Classify faces into zones (apertures, underbody, trim)."""
        p, L = self.p, self.L
        u = lambda name: station_index(name, "upper") + off  # noqa: E731
        l = lambda name: station_index(name, "lower") + off  # noqa: E731
        zones: Dict[str, List[int]] = {}
        rects: Dict[str, Tuple[int, int, int, int]] = {}

        def add(zone, rows, cols):
            rows, cols = list(rows), list(cols)
            for r in rows:
                for j in cols:
                    fi = face_grid.get((r, j))
                    if fi is not None:
                        zones.setdefault(zone, []).append(fi)
            if rows and cols:
                rects[zone] = (min(rows), max(rows), min(cols), max(cols))

        jF, jG, jH, jE, jD, jC = RING["F"], RING["G"], RING["H"], RING["E"], RING["D"], RING["C"]
        for name, (rows, cols) in self._glass_regions().items():
            add(name, range(rows.start + off, rows.stop + off), cols)
        # lights: wrap-around corner strips at the nose/tail
        n_front = 3
        n_rear = 3
        jL = jE - 3  # lights extend down the fender side to mid height of the D->E strip
        # End at the fender fold: glass/light fills must not bridge the folded
        # hood seam, whose return face intentionally has an opposing normal.
        add("aperture/headlight_L", range(u("front") - n_front, u("front") + 1), range(jL, jF))
        add("aperture/headlight_R", range(u("front") - n_front, u("front") + 1), range(RING_N - jF, RING_N - jL))
        add("aperture/taillight_L", range(off - 1, off + n_rear), range(jL, jF))
        add("aperture/taillight_R", range(off - 1, off + n_rear), range(RING_N - jF, RING_N - jL))
        # underbody / wells
        all_rows = range(off, off + N_STATIONS - 1)
        add("underbody", all_rows, list(range(0, jC)) + list(range(RING_N - jC, RING_N)))
        arch_rows = [i + off for i in range(N_STATIONS) if self.arch(self.x_lo[i])[0] > 0]
        add("wheel_well", arch_rows, list(range(jC, jD)) + list(range(RING_N - jD, RING_N - jC)))
        # Only the outer skirt bands are dark. Ring columns on an inner
        # fascia encircle the plate, so calling them "underbody" would paint
        # triangular wedges halfway up the bumper.
        skirt_rows = list(range(off - 3, off)) + list(range(off + N_STATIONS - 1, off + N_STATIONS + 2))
        add("trim", skirt_rows, list(range(0, RING["C"])) + list(range(RING_N - RING["C"], RING_N)))
        if self.bed is not None:
            bed_rows = [i + off for i in range(N_STATIONS) if self.bed(self.x_hi[i]) > 0.5]
            add("bed", bed_rows, range(RING["G2"], RING_N - RING["G2"]))
        for name, idx in zones.items():
            mesh.add_zone(name, idx)
            if name.startswith("aperture/"):
                mesh.set_material(GLASS_ZONE, idx)
            elif name in ("underbody", "wheel_well"):
                mesh.set_material(UNDERBODY, idx)
            elif name == "trim":
                mesh.set_material(TRIM, idx)
            elif name == "bed":
                mesh.set_material("bed", idx)
        mesh.meta["aperture_rects"] = {k: v for k, v in rects.items() if k.startswith("aperture/")}
        mesh.meta["arch_rows"] = [int(r) for r in arch_rows]

    def _assign_groups(self, mesh: Mesh, off: int, n_rings: int, rear_apex_i: int, front_apex_i: int):
        """Vertex groups / landmarks that connectors are derived from."""
        L = self.L

        def v(ring_i, j):
            return (ring_i + off) * RING_N + j

        def ring_group(name, ring_i, js):
            mesh.add_group(name, [v(ring_i, j) for j in js])

        half_left = range(0, HALF_N + 1)
        for nm in UPPER_NAMES:
            i = station_index(nm, "upper")
            ring_group(f"ring/{nm}", i, range(RING_N))
        for nm in LOWER_NAMES:
            i = station_index(nm, "lower")
            ring_group(f"lring/{nm}", i, range(RING_N))
        # arch lips (D points) for circle fitting
        for tag, a, b in (("front", "fa_start", "fa_end"), ("rear", "ra_start", "ra_end")):
            i0, i1 = station_index(a, "lower"), station_index(b, "lower")
            mesh.add_group(f"arch_{tag}_L", [v(i, RING["D"]) for i in range(i0, i1 + 1)])
            mesh.add_group(f"arch_{tag}_R", [v(i, mirror_index(RING["D"])) for i in range(i0, i1 + 1)])
        # longitudinal feature lines
        for key, j in (("floor", RING["A"]), ("sill_L", RING["D"]), ("belt_L", RING["E"]), ("shoulder_L", RING["F"]),
                       ("rail_L", RING["G"]), ("center_top", RING["H"])):
            mesh.add_group(f"line/{key}", [v(i, j) for i in range(N_STATIONS)])
        mesh.add_group("line/sill_R", [v(i, mirror_index(RING["D"])) for i in range(N_STATIONS)])
        mesh.add_group("line/belt_R", [v(i, mirror_index(RING["E"])) for i in range(N_STATIONS)])
        mesh.add_group("line/shoulder_R", [v(i, mirror_index(RING["F"])) for i in range(N_STATIONS)])
        mesh.add_group("line/rail_R", [v(i, mirror_index(RING["G"])) for i in range(N_STATIONS)])
        mesh.add_group("apex_front", [front_apex_i])
        mesh.add_group("apex_rear", [rear_apex_i])
        mesh.add_group("nose_ring", [v(N_STATIONS - 1, j) for j in range(RING_N)])
        mesh.add_group("tail_ring", [v(0, j) for j in range(RING_N)])
        for end, rows in (("front", range(off + N_STATIONS, n_rings)), ("rear", range(off - 1, -1, -1))):
            rows = list(rows)
            mesh.add_group(f"fascia/{end}", [r * RING_N + j for r in rows for j in range(RING_N)])
            for name, k in (("face", 1), ("pocket_rim", 3), ("pocket_floor", 4)):
                mesh.add_group(f"fascia/{end}_{name}", [rows[k] * RING_N + j for j in range(RING_N)])


def vertex_index(station: int, j: int, ring_offset: int) -> int:
    return (station + ring_offset) * RING_N + j
