"""Canonical parameters of the body generator.

All lengths in metres.  The canonical values describe a mid-size sedan; the
style presets in `styles.py` are full parameter sets for other body types and
become *macro targets* (like MakeHuman's gender/age/weight macros).

Coordinate conventions: +X forward, +Y left, +Z up, ground at Z=0, origin at
the middle of the wheelbase.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import math


@dataclass
class BodyParams:
    # ---- overall proportions -------------------------------------------------
    wheelbase: float = 2.75
    front_overhang: float = 0.95
    rear_overhang: float = 1.05
    width: float = 1.84            # maximum body width (at the belt line)
    ground_clearance: float = 0.15
    # ---- wheels / arches -----------------------------------------------------
    wheel_diameter: float = 0.66
    tire_width: float = 0.225
    arch_gap: float = 0.06         # radial gap between tyre and arch lip
    fender_flare: float = 0.025    # outward bulge of the arch lip
    # ---- lower body ----------------------------------------------------------
    sill_height: float = 0.36      # top of the rocker panel
    sill_inset: float = 0.05       # rocker sits this far inboard of the belt line
    belt_height: float = 0.93      # door shoulder / belt line height
    belt_rake: float = 0.03        # belt rises this much from front door to rear door
    side_bulge: float = 0.02       # convexity of the door skin
    # ---- front end -----------------------------------------------------------
    front_bumper_bottom: float = 0.30
    hood_front_height: float = 0.76
    cowl_height: float = 0.98      # windshield base
    cowl_offset: float = 0.35      # windshield base is this far behind the front axle
    hood_crown: float = 0.03
    nose_width_ratio: float = 0.72  # flat nose width / body width
    front_corner_length: float = 0.55  # plan-view corner rounding run
    front_fascia_rake: float = 0.03    # upper fascia setback (bumper stays vertical)
    bumper_projection: float = 0.035  # bumper face proud of the end ring
    bumper_crease_height: float = 0.60
    hood_overhang: float = 0.03       # hood lip proud of the upper grille face
    front_splitter: float = 0.025    # lower lip projection beyond the bumper
    rear_bumper_crease_height: float = 0.62
    plate_recess: float = 0.025      # depth of the rear licence-plate pocket
    diffuser_step: float = 0.012     # lower rear valance setback
    tailgate_panel: float = 0.0      # lower tailgate panel setback (two-box bodies)
    # ---- greenhouse ----------------------------------------------------------
    windshield_length: float = 0.90    # horizontal run of the windshield
    roof_height: float = 1.45
    roof_drop: float = 0.02            # roof is lower at the rear
    roof_rear_offset: float = 0.33     # roof ends this far ahead of (+) the rear axle
    rear_window_length: float = 0.70
    roof_crown: float = 0.035
    roof_width_ratio: float = 0.68     # roof rail half width / body half width
    shoulder_inset: float = 0.06       # glass base sits inboard of the belt line
    shoulder_rise: float = 0.03
    a_pillar_lean: float = 0.45        # as fraction of windshield_length
    c_pillar_lean: float = 0.35        # as fraction of rear_window_length
    front_door_length: float = 1.32    # cowl -> B pillar centre
    b_pillar_width: float = 0.10
    quarter_window_length: float = 0.03
    c_pillar_width: float = 0.08
    # ---- rear end ------------------------------------------------------------
    deck_height: float = 0.98          # rear window base / trunk lid (or bed rail)
    deck_rear_height: float = 0.94     # trunk lid at the rear edge
    rear_bumper_bottom: float = 0.32
    tail_width_ratio: float = 0.78
    rear_corner_length: float = 0.45
    rear_fascia_rake: float = 0.02     # >0: tail top protrudes over the bumper
    deck_crown: float = 0.02
    bed_depth: float = 0.0             # >0 turns the deck into an open pickup bed
    # ---- non-geometric hints (do not affect topology) -------------------------
    door_count: int = 4

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BodyParams":
        valid = {f.name for f in fields(cls)}
        unknown = set(d) - valid
        if unknown:
            raise KeyError(f"unknown body parameter(s): {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def numeric_fields(cls):
        return [f.name for f in fields(cls) if f.type in ("float", float)]

    # derived layout -----------------------------------------------------------
    @property
    def length(self) -> float:
        return self.wheelbase + self.front_overhang + self.rear_overhang

    @property
    def arch_radius(self) -> float:
        return self.wheel_diameter / 2 + self.arch_gap

    @property
    def axle_height(self) -> float:
        return self.wheel_diameter / 2

    def layout(self) -> dict:
        """Key longitudinal positions.  Greenhouse breakpoints are forced to be
        strictly ordered (min gap 2 cm) so the loft topology is always valid."""
        x_fa = self.wheelbase / 2
        x_ra = -self.wheelbase / 2
        x_front = x_fa + self.front_overhang
        x_rear = x_ra - self.rear_overhang
        gap = 0.02
        # ---- upper (greenhouse) chain, front to back ----
        x_cowl = x_fa - self.cowl_offset
        x_roof_front = x_cowl - self.windshield_length
        x_bp = x_cowl - self.front_door_length
        x_bp_f = x_bp + self.b_pillar_width / 2
        x_bp_r = x_bp - self.b_pillar_width / 2
        x_roof_rear = x_ra + self.roof_rear_offset
        x_cp_r = x_roof_rear + self.quarter_window_length
        x_cp_f = x_cp_r + self.c_pillar_width
        x_deck = x_roof_rear - self.rear_window_length
        upper = [x_rear, x_deck, x_roof_rear, x_cp_r, x_cp_f, x_bp_r, x_bp_f, x_roof_front, x_cowl, x_front]
        # enforce ordering front->back (front fixed): push items rearward if needed
        for k in range(len(upper) - 2, 0, -1):
            upper[k] = min(upper[k], upper[k + 1] - gap)
        for k in range(1, len(upper) - 1):
            upper[k] = max(upper[k], upper[k - 1] + gap)
        # ---- lower (arch) chain ----
        r = self.arch_radius
        dz = self.sill_height - self.axle_height
        half = math.sqrt(max(r * r - dz * dz, 1e-6)) if dz > 0 else r
        half = min(half, self.wheelbase / 2 - gap, self.front_overhang - 0.1, self.rear_overhang - 0.1)
        lower = [x_rear, x_ra - half, x_ra, x_ra + half, x_fa - half, x_fa, x_fa + half, x_front]
        names_u = ["rear", "deck", "roof_rear", "cp_r", "cp_f", "bp_r", "bp_f", "roof_front", "cowl", "front"]
        names_l = ["rear", "ra_start", "ra", "ra_end", "fa_start", "fa", "fa_end", "front"]
        d = {f"u_{n}": v for n, v in zip(names_u, upper)}
        d.update({f"l_{n}": v for n, v in zip(names_l, lower)})
        d.update(dict(x_front=x_front, x_rear=x_rear, x_fa=x_fa, x_ra=x_ra, arch_half=half, arch_radius=r))
        d["upper"] = upper
        d["lower"] = lower
        return d
