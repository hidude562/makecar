"""Additive exterior archetypes, expressed as dimensional generator changes.

Each family is a simplex with a neutral remainder: a total weight below one
leaves some of the base shape; a total above one becomes a convex blend. These
are ordinary differential targets, not mutually exclusive style switches.
"""
from __future__ import annotations

from dataclasses import replace

from .params import BodyParams


SHAPE_DELTAS = {
    "face/wedge": dict(hood_front_height=-0.16, hood_straightness=1.0,
                       bumper_crease_height=-0.10, front_bumper_bottom=-0.065,
                       fascia_slope=0.15,
                       hood_overhang=-0.02, fender_crown_height=0.055),
    "face/upright": dict(hood_front_height=0.19, hood_straightness=1.0,
                         front_fascia_rake=-0.025, hood_overhang=-0.02,
                         front_splitter=-0.012),
    "face/shark": dict(hood_straightness=0.7, nose_center_drop=0.12,
                       nose_center_extension=0.16, fender_crown_height=0.10,
                       fascia_slope=0.13, hood_overhang=-0.015),
    "face/snub": dict(cowl_offset=-0.40, windshield_length=0.18,
                      hood_front_height=0.16, hood_straightness=0.35,
                      nose_width_ratio=-0.06, front_corner_length=0.12,
                      fascia_roundness=0.10, front_fascia_rake=-0.015,
                      hood_overhang=-0.02),
    "face/cabforward": dict(cowl_offset=-0.68, windshield_length=0.50,
                            front_door_length=0.40, cowl_height=0.10,
                            hood_front_height=0.14, hood_straightness=0.65,
                            fascia_roundness=0.065, fascia_slope=0.07,
                            hood_overhang=-0.02),
    "face/longhood": dict(cowl_offset=0.32, hood_front_height=0.15,
                          hood_straightness=1.0, front_fascia_rake=-0.015,
                          hood_overhang=-0.02),
    "plan/pointed": dict(nose_width_ratio=-0.36, tail_width_ratio=-0.26,
                         nose_taper_length=0.67, tail_taper_length=0.58,
                         nose_taper_exponent=-1.1, tail_taper_exponent=-1.1),
    "plan/square": dict(nose_width_ratio=0.20, tail_width_ratio=0.16,
                        front_corner_length=-0.16, rear_corner_length=-0.16,
                        nose_taper_exponent=5.8, tail_taper_exponent=5.8),
    "plan/cokebottle": dict(waist_depth=0.14, nose_taper_length=0.37,
                            tail_taper_length=0.38, nose_width_ratio=-0.07,
                            tail_width_ratio=-0.04),
    "section/tumblehome": dict(tumblehome=0.10),
    "section/slabside": dict(roof_width_ratio=0.16, shoulder_inset=-0.03,
                             shoulder_radius=-0.012, side_bulge=-0.012),
    "section/domed": dict(roof_crown=0.095, roof_width_ratio=-0.035,
                          shoulder_radius=0.035, shoulder_inset=0.025),
}

SHAPE_DESCRIPTIONS = {
    "face/wedge": "low straight hood and raked upper face over a forward splitter",
    "face/upright": "high flat hood over a near-vertical wall of fascia",
    "face/shark": "projecting low centre tip between proud crowned fenders",
    "face/snub": "short high rounded city-car nose",
    "face/cabforward": "forward cowl and extended windscreen, nearly eliminating the hood",
    "face/longhood": "rearward cowl and long high flat hood",
    "plan/pointed": "strong taper to narrow rounded nose and tail",
    "plan/square": "full-width ends with tight corners",
    "plan/cokebottle": "waisted doors between full-width axle shoulders",
    "section/tumblehome": "inward-leaning glasshouse above wide hips",
    "section/slabside": "broad roof, vertical sides and square shoulders",
    "section/domed": "domed roof and rounded shoulders",
}

SHAPE_FAMILIES = ("face", "plan", "section")


def normalize_shape_values(values: dict) -> dict:
    """Copy, clamp, and normalize each family independently; preserve other keys."""
    resolved = dict(values)
    for family in SHAPE_FAMILIES:
        names = [name for name in SHAPE_DELTAS if name.startswith(family + "/") and name in resolved]
        weights = {name: max(0.0, min(1.0, float(resolved[name]))) for name in names}
        total = max(1.0, sum(weights.values()))
        resolved.update({name: weight / total for name, weight in weights.items()})
    return resolved


def shape_params(base: BodyParams, values: dict) -> BodyParams:
    """Apply normalized dimensional archetype deltas to a generator preset."""
    changes = {}
    for name, weight in normalize_shape_values(values).items():
        if name not in SHAPE_DELTAS:
            continue
        for field, delta in SHAPE_DELTAS[name].items():
            changes[field] = changes.get(field, getattr(base, field)) + weight * delta
    return replace(base, **changes)


def resolve_shape_values(values: dict) -> dict:
    """Inherit style mixes, then overlay explicit archetypes (including zero).

    Runs at body-library displacement time because CarBody is outside this
    task's ownership. Requested modifier metadata is deliberately not mutated.
    """
    from .styles import STYLE_SHAPE_DEFAULTS

    inherited = {}
    for style, defaults in STYLE_SHAPE_DEFAULTS.items():
        weight = max(0.0, min(1.0, float(values.get("style/" + style, 0.0))))
        if weight:
            for name, value in defaults.items():
                inherited[name] = inherited.get(name, 0.0) + weight * value
    inherited.update(values)
    return normalize_shape_values(inherited)
