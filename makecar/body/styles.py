"""Body-style macros, dimensioned against stock production-car references.

Wheelbase, width and nominal tyre diameter follow the reference specification;
overhang splits and greenhouse hard points are modeling estimates. Roof height
is the skin centre, not the antenna/roof-rack extremity. Sources and limitations
are recorded in docs/agent_reports/01_body_surfacing/REPORT.md.
"""
from __future__ import annotations

import dataclasses
from typing import Dict
from .params import BodyParams

STYLE_OVERRIDES: Dict[str, dict] = {
    "sedan": {},  # 2024 Toyota Camry LE (canonical BodyParams)
    "hatchback": dict(  # 2020 VW Golf Life 1.0 eTSI
        wheelbase=2.619, front_overhang=0.850, rear_overhang=0.815, width=1.789, ground_clearance=0.14,
        wheel_diameter=0.6319, tire_width=0.205, sill_height=0.35, belt_height=0.98,
        hood_front_height=0.78, cowl_height=1.04, cowl_offset=0.28, windshield_length=0.72,
        roof_height=1.491, roof_drop=0.03, roof_rear_offset=-0.38, rear_window_length=0.32,
        deck_height=1.06, deck_rear_height=0.96, rear_corner_length=0.25,
        tail_width_ratio=0.80, rear_fascia_rake=0.0, nose_width_ratio=0.73, front_corner_length=0.27,
        front_door_length=1.22, c_pillar_width=0.12, roof_width_ratio=0.72, c_pillar_lean=0.25,
        tailgate_panel=0.025, bumper_crease_height=0.59, rear_bumper_crease_height=0.62,
        shoulder_radius=0.022, fender_crease=0.010, front_splitter=0.020,
    ),
    "wagon": dict(  # 2024 Subaru Outback Base (roof rails excluded from skin)
        wheelbase=2.746, front_overhang=0.995, rear_overhang=1.113, width=1.854,
        ground_clearance=0.22, wheel_diameter=0.7243, tire_width=0.225, sill_height=0.43,
        belt_height=1.09, hood_front_height=0.90, cowl_height=1.15, cowl_offset=0.30,
        windshield_length=0.84, roof_height=1.64, roof_rear_offset=-0.65,
        rear_window_length=0.36, deck_height=1.12, deck_rear_height=1.06,
        quarter_window_length=0.48, c_pillar_width=0.10, roof_drop=0.03, roof_width_ratio=0.72,
        tail_width_ratio=0.80, rear_fascia_rake=0.0, rear_corner_length=0.27, c_pillar_lean=0.2,
        front_bumper_bottom=0.36, rear_bumper_bottom=0.39, bumper_crease_height=0.63,
        rear_bumper_crease_height=0.67, tailgate_panel=0.030, shoulder_radius=0.025,
        fender_crease=0.012, front_splitter=0.020,
    ),
    "suv": dict(  # 2024 Toyota RAV4 LE FWD
        wheelbase=2.690, front_overhang=0.915, rear_overhang=0.990, width=1.854, ground_clearance=0.21,
        wheel_diameter=0.7243, tire_width=0.225, sill_height=0.44, belt_height=1.10,
        hood_front_height=0.94, cowl_height=1.16, cowl_offset=0.20, windshield_length=0.65,
        roof_height=1.66, roof_drop=0.025, roof_rear_offset=-0.48, rear_window_length=0.39,
        deck_height=1.12, deck_rear_height=1.06, roof_width_ratio=0.74, roof_crown=0.03,
        front_bumper_bottom=0.40, rear_bumper_bottom=0.43, nose_width_ratio=0.76,
        front_corner_length=0.28, front_fascia_rake=0.03, quarter_window_length=0.42,
        c_pillar_width=0.13, front_door_length=1.22, fender_flare=0.035, tail_width_ratio=0.81,
        rear_corner_length=0.26, rear_fascia_rake=0.0, c_pillar_lean=0.2,
        bumper_crease_height=0.65, rear_bumper_crease_height=0.69, tailgate_panel=0.035,
        shoulder_radius=0.018, fender_crease=0.014, a_pillar_width=0.08, front_splitter=0.020,
    ),
    "pickup": dict(  # 2021 Ford F-150 XL SuperCrew 4x2, 5.5-ft bed
        wheelbase=3.693, front_overhang=0.955, rear_overhang=1.237, width=2.029, ground_clearance=0.22,
        wheel_diameter=0.7748, tire_width=0.245, sill_height=0.50, belt_height=1.28,
        hood_front_height=1.17, cowl_height=1.38, cowl_offset=0.16, windshield_length=0.66,
        roof_height=1.920, roof_drop=0.02, roof_rear_offset=0.80, rear_window_length=0.11,
        deck_height=1.30, deck_rear_height=1.30, bed_depth=0.55,
        front_door_length=1.26, quarter_window_length=0.03, c_pillar_width=0.15,
        roof_width_ratio=0.78, tail_width_ratio=0.83, rear_corner_length=0.25,
        rear_fascia_rake=0.0, front_bumper_bottom=0.45, rear_bumper_bottom=0.45,
        nose_width_ratio=0.78, front_corner_length=0.28, front_fascia_rake=0.025,
        fender_flare=0.035, c_pillar_lean=0.15, deck_crown=0.0,
        bumper_crease_height=0.70, rear_bumper_crease_height=0.70, tailgate_panel=0.025,
        shoulder_radius=0.018, fender_crease=0.016, a_pillar_width=0.085, front_splitter=0.015,
    ),
    "coupe": dict(  # 2025 Ford Mustang EcoBoost Fastback
        wheelbase=2.718, front_overhang=0.965, rear_overhang=1.128, width=1.915, ground_clearance=0.13,
        wheel_diameter=0.6903, tire_width=0.235, sill_height=0.33, belt_height=0.93, belt_rake=0.05,
        hood_front_height=0.75, cowl_height=0.99, cowl_offset=0.58, windshield_length=0.86,
        roof_height=1.397, roof_drop=0.035, roof_rear_offset=0.35, rear_window_length=0.83,
        deck_height=0.98, deck_rear_height=0.94, roof_width_ratio=0.68,
        front_door_length=1.50, door_count=2, c_pillar_lean=0.45,
        bumper_crease_height=0.58, rear_bumper_crease_height=0.61, front_splitter=0.045,
        diffuser_step=0.065, shoulder_radius=0.020, fender_crease=0.016, fender_flare=0.035,
    ),
    "sports": dict(  # 2025 Mazda MX-5 RF Grand Touring, closed-roof approximation
        wheelbase=2.309, front_overhang=0.780, rear_overhang=0.825, width=1.735, ground_clearance=0.12,
        wheel_diameter=0.6163, tire_width=0.205, sill_height=0.30, belt_height=0.81, belt_rake=0.03,
        hood_front_height=0.66, cowl_height=0.86, cowl_offset=0.48, windshield_length=0.59,
        roof_height=1.245, roof_drop=0.02, roof_rear_offset=0.43, rear_window_length=0.30,
        deck_height=0.87, deck_rear_height=0.83, roof_width_ratio=0.67,
        front_door_length=1.20, door_count=2, roof_crown=0.03, front_bumper_bottom=0.24,
        rear_bumper_bottom=0.28, nose_width_ratio=0.73, front_corner_length=0.27, front_fascia_rake=0.025,
        fender_flare=0.035, shoulder_inset=0.07, tail_width_ratio=0.76, c_pillar_lean=0.35,
        bumper_crease_height=0.55, rear_bumper_crease_height=0.57, hood_overhang=0.025,
        front_splitter=0.055, diffuser_step=0.045, shoulder_radius=0.018, fender_crease=0.018,
        a_pillar_width=0.06, rocker_height=0.14, air_dam_height=0.025, rear_valance_height=0.035,
    ),
    "van": dict(  # 2024 Toyota Sienna LE FWD, without roof rails
        wheelbase=3.061, front_overhang=0.970, rear_overhang=1.143, width=1.994, ground_clearance=0.16,
        wheel_diameter=0.7373, tire_width=0.235, sill_height=0.40, belt_height=1.09,
        hood_front_height=0.91, cowl_height=1.15, cowl_offset=-0.05, windshield_length=0.98,
        roof_height=1.74, roof_drop=0.02, roof_rear_offset=-0.63, rear_window_length=0.37,
        deck_height=1.10, deck_rear_height=1.04, roof_width_ratio=0.79,
        front_door_length=1.28, quarter_window_length=0.50, c_pillar_width=0.12, tail_width_ratio=0.82,
        rear_corner_length=0.27, nose_width_ratio=0.77, front_corner_length=0.29, front_fascia_rake=0.03,
        front_bumper_bottom=0.34, rear_bumper_bottom=0.37, rear_fascia_rake=0.0, c_pillar_lean=0.15,
        bumper_crease_height=0.62, rear_bumper_crease_height=0.65, tailgate_panel=0.030,
        shoulder_radius=0.030, fender_crease=0.010, a_pillar_width=0.080, front_splitter=0.015,
    ),
}

# Archetype defaults are inherited by style macros at morph time. Explicit
# face/plan/section config values replace individual defaults, including zero.
STYLE_SHAPE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "sedan": {},
    "hatchback": {"face/snub": 0.65, "section/domed": 0.5, "plan/pointed": 0.15},
    "wagon": {"face/upright": 0.3, "plan/square": 0.3, "plan/pointed": 0.35},
    "suv": {"face/upright": 0.85, "plan/square": 0.55, "plan/cokebottle": 0.15, "section/slabside": 0.6},
    "pickup": {"face/upright": 1.0, "plan/square": 1.0, "section/slabside": 0.8},
    "coupe": {"face/longhood": 0.65, "plan/cokebottle": 0.8, "section/tumblehome": 0.5},
    "sports": {"face/wedge": 0.8, "plan/pointed": 0.85, "section/tumblehome": 0.7},
    "van": {"face/cabforward": 0.85, "plan/square": 0.85, "section/slabside": 0.5},
}

STYLE_DESCRIPTIONS = {
    "sedan": "three-box family sedan (canonical base)",
    "hatchback": "compact two-box hatchback",
    "wagon": "estate / station wagon",
    "suv": "tall sport-utility vehicle",
    "pickup": "crew-cab pickup truck with open bed",
    "coupe": "two-door fastback coupe",
    "sports": "low, wide two-seat sports car",
    "van": "one-box minivan / MPV",
}


def style_names():
    return list(STYLE_OVERRIDES)


def style_params(name: str, base: BodyParams | None = None) -> BodyParams:
    """Dimensional preset including its default face, plan and section mix."""
    from .shapes import shape_params

    if name not in STYLE_OVERRIDES:
        raise KeyError(f"unknown style {name!r}; choose from {style_names()}")
    base = base or BodyParams()
    return shape_params(dataclasses.replace(base, **STYLE_OVERRIDES[name]), STYLE_SHAPE_DEFAULTS[name])
