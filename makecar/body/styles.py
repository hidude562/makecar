"""Body-style presets: full parameter sets that become macro targets.

Each preset is expressed as *overrides* of the canonical sedan so that the
differences are readable.  `style_params(name)` returns a BodyParams.
"""
from __future__ import annotations

import dataclasses
from typing import Dict
from .params import BodyParams

STYLE_OVERRIDES: Dict[str, dict] = {
    "sedan": {},
    "hatchback": dict(
        wheelbase=2.62, front_overhang=0.85, rear_overhang=0.72, width=1.80, ground_clearance=0.14,
        wheel_diameter=0.64, sill_height=0.35, belt_height=0.92, hood_front_height=0.74, cowl_height=0.98,
        cowl_offset=0.30, windshield_length=0.85, roof_height=1.47, roof_drop=0.03, roof_rear_offset=-0.02,
        rear_window_length=0.50, deck_height=1.02, deck_rear_height=0.80, rear_corner_length=0.32,
        tail_width_ratio=0.86, rear_fascia_rake=0.0, nose_width_ratio=0.70, front_corner_length=0.5,
        front_door_length=1.22, c_pillar_width=0.10, roof_width_ratio=0.70, c_pillar_lean=0.25,
    ),
    "wagon": dict(
        rear_overhang=1.12, roof_rear_offset=-0.72, rear_window_length=0.30, deck_height=1.02,
        deck_rear_height=0.97, quarter_window_length=0.48, c_pillar_width=0.10, roof_drop=0.03,
        tail_width_ratio=0.86, rear_fascia_rake=0.0, rear_corner_length=0.32, c_pillar_lean=0.2,
    ),
    "suv": dict(
        wheelbase=2.85, front_overhang=0.9, rear_overhang=1.0, width=1.93, ground_clearance=0.22,
        wheel_diameter=0.76, tire_width=0.255, sill_height=0.45, belt_height=1.05, hood_front_height=0.90,
        cowl_height=1.12, cowl_offset=0.25, windshield_length=0.75, roof_height=1.75, roof_drop=0.03,
        roof_rear_offset=-0.6, rear_window_length=0.30, deck_height=1.10, deck_rear_height=1.05,
        roof_width_ratio=0.72, roof_crown=0.03, front_bumper_bottom=0.42, rear_bumper_bottom=0.45,
        nose_width_ratio=0.78, front_corner_length=0.45, front_fascia_rake=0.05, quarter_window_length=0.45,
        c_pillar_width=0.10, front_door_length=1.22, fender_flare=0.035, tail_width_ratio=0.88,
        rear_corner_length=0.3, rear_fascia_rake=0.0, c_pillar_lean=0.2,
    ),
    "pickup": dict(
        wheelbase=3.4, front_overhang=0.95, rear_overhang=1.2, width=2.0, ground_clearance=0.24,
        wheel_diameter=0.80, tire_width=0.265, sill_height=0.50, belt_height=1.08, hood_front_height=0.95,
        cowl_height=1.18, cowl_offset=0.30, windshield_length=0.70, roof_height=1.85, roof_drop=0.02,
        roof_rear_offset=1.0, rear_window_length=0.15, deck_height=1.15, deck_rear_height=1.15, bed_depth=0.55,
        front_door_length=1.15, quarter_window_length=0.03, c_pillar_width=0.15, roof_width_ratio=0.75,
        tail_width_ratio=0.92, rear_corner_length=0.2, rear_fascia_rake=0.0, front_bumper_bottom=0.45,
        rear_bumper_bottom=0.45, nose_width_ratio=0.85, front_corner_length=0.4, front_fascia_rake=0.03,
        fender_flare=0.04, c_pillar_lean=0.15, deck_crown=0.0,
    ),
    "coupe": dict(
        wheelbase=2.7, front_overhang=0.95, rear_overhang=1.0, width=1.86, ground_clearance=0.13,
        wheel_diameter=0.68, sill_height=0.33, belt_height=0.90, belt_rake=0.06, hood_front_height=0.72,
        cowl_height=0.95, cowl_offset=0.45, windshield_length=1.0, roof_height=1.36, roof_drop=0.04,
        roof_rear_offset=0.55, rear_window_length=0.85, deck_height=0.92, deck_rear_height=0.90,
        roof_width_ratio=0.64, front_door_length=1.55, door_count=2, c_pillar_lean=0.45,
    ),
    "sports": dict(
        wheelbase=2.5, front_overhang=0.9, rear_overhang=0.85, width=1.95, ground_clearance=0.11,
        wheel_diameter=0.68, tire_width=0.265, sill_height=0.30, belt_height=0.85, belt_rake=0.05,
        hood_front_height=0.62, cowl_height=0.88, cowl_offset=0.65, windshield_length=0.95, roof_height=1.22,
        roof_drop=0.03, roof_rear_offset=0.40, rear_window_length=0.70, deck_height=0.86, deck_rear_height=0.84,
        roof_width_ratio=0.62, front_door_length=1.45, door_count=2, roof_crown=0.03, front_bumper_bottom=0.20,
        rear_bumper_bottom=0.24, nose_width_ratio=0.75, front_corner_length=0.6, front_fascia_rake=0.15,
        fender_flare=0.045, shoulder_inset=0.08, tail_width_ratio=0.82, c_pillar_lean=0.5,
    ),
    "van": dict(
        wheelbase=3.0, front_overhang=0.95, rear_overhang=1.05, width=1.95, ground_clearance=0.17,
        wheel_diameter=0.70, sill_height=0.40, belt_height=1.05, hood_front_height=0.85, cowl_height=1.15,
        cowl_offset=0.05, windshield_length=1.05, roof_height=1.80, roof_drop=0.02, roof_rear_offset=-0.75,
        rear_window_length=0.25, deck_height=1.05, deck_rear_height=1.0, roof_width_ratio=0.78,
        front_door_length=1.2, quarter_window_length=0.5, c_pillar_width=0.12, tail_width_ratio=0.9,
        rear_corner_length=0.3, nose_width_ratio=0.8, front_corner_length=0.45, front_fascia_rake=0.05,
        front_bumper_bottom=0.35, rear_bumper_bottom=0.38, rear_fascia_rake=0.0, c_pillar_lean=0.15,
    ),
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
    if name not in STYLE_OVERRIDES:
        raise KeyError(f"unknown style {name!r}; choose from {style_names()}")
    base = base or BodyParams()
    return dataclasses.replace(base, **STYLE_OVERRIDES[name])
