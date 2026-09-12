"""The body's modifier library (MakeHuman analogue of the targets folder).

Three kinds of targets are used:

* **differential targets** – one bipolar modifier per generator parameter,
  computed by re-running the fixed-topology generator with the parameter moved
  by a physically meaningful amount (e.g. wheelbase ± 0.4 m);
* **macro targets** – one unipolar modifier per body style (hatchback, wagon,
  suv, ...) computed as `generate(style) - generate(sedan)`;
* **sculpt targets** – hand-authored procedural displacement fields that are not
  expressible through the generator parameters (rear haunches, side crease,
  wedge stance, roof bubble).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List
import numpy as np

from ..geometry.mesh import Mesh
from ..morph.morphable import MorphableMesh, DifferentialTargetBuilder, ModifierSpec
from ..morph.target import Target, Modifier
from ..geometry.curves import smoothstep
from .params import BodyParams
from .generator import BodyGenerator, RING, RING_N, mirror_index
from .styles import STYLE_OVERRIDES, style_params

# (param, delta-, delta+, group, description)
BODY_MODIFIER_SPECS: List[ModifierSpec] = [
    ModifierSpec("wheelbase", 0.40, 0.55, "proportions", "distance between axles"),
    ModifierSpec("front_overhang", 0.20, 0.25, "proportions", "body ahead of the front axle"),
    ModifierSpec("rear_overhang", 0.30, 0.35, "proportions", "body behind the rear axle"),
    ModifierSpec("width", 0.15, 0.18, "proportions", "overall body width"),
    ModifierSpec("ground_clearance", 0.05, 0.09, "stance", "ride height"),
    ModifierSpec("wheel_diameter", 0.08, 0.14, "stance", "wheel/tyre diameter and arch size"),
    ModifierSpec("tire_width", 0.05, 0.06, "stance", "tyre width / wheel-well depth"),
    ModifierSpec("fender_flare", 0.02, 0.05, "stance", "arch lip flare"),
    ModifierSpec("arch_lip_width", 0.012, 0.015, "stance", "radial width of the folded wheel arch flange"),
    ModifierSpec("rocker_height", 0.04, 0.04, "lower_body", "height of the flat vertical sill face"),
    ModifierSpec("door_step", 0.008, 0.012, "lower_body", "door bottom hem proud of the sill"),
    ModifierSpec("shoulder_radius", 0.015, 0.015, "lower_body", "rolled shoulder radius"),
    ModifierSpec("fender_crease", 0.010, 0.012, "front", "hood-to-fender folded step"),
    ModifierSpec("tunnel_height", 0.04, 0.045, "lower_body", "floor pan tunnel rise"),
    ModifierSpec("tunnel_width", 0.08, 0.10, "lower_body", "floor pan tunnel width"),
    ModifierSpec("air_dam_height", 0.025, 0.025, "front", "front skirt depth below bumper"),
    ModifierSpec("rear_valance_height", 0.025, 0.030, "rear", "rear skirt depth below bumper"),
    ModifierSpec("sill_height", 0.06, 0.09, "lower_body", "rocker panel height"),
    ModifierSpec("belt_height", 0.08, 0.10, "lower_body", "belt line / shoulder height"),
    ModifierSpec("belt_rake", 0.03, 0.06, "lower_body", "belt line rising towards the rear"),
    ModifierSpec("side_bulge", 0.02, 0.04, "lower_body", "door skin convexity"),
    ModifierSpec("hood_front_height", 0.10, 0.12, "front", "hood leading edge height"),
    ModifierSpec("cowl_height", 0.08, 0.10, "front", "windshield base height"),
    ModifierSpec("cowl_offset", 0.25, 0.30, "front", "windshield base position (cab-forward <-> long hood)"),
    ModifierSpec("hood_crown", 0.02, 0.03, "front", "transverse hood curvature"),
    ModifierSpec("nose_width_ratio", 0.12, 0.15, "front", "flat nose width"),
    ModifierSpec("front_corner_length", 0.20, 0.25, "front", "plan-view rounding of the nose"),
    ModifierSpec("front_fascia_rake", 0.06, 0.08, "front", "nose top set back over the bumper"),
    ModifierSpec("front_bumper_bottom", 0.08, 0.10, "front", "front bumper lower edge"),
    ModifierSpec("bumper_projection", 0.025, 0.035, "front", "bumper stand-off from body end"),
    ModifierSpec("bumper_crease_height", 0.05, 0.05, "front", "top of the upright front bumper face"),
    ModifierSpec("hood_overhang", 0.025, 0.025, "front", "hood leading lip over the grille"),
    ModifierSpec("front_splitter", 0.025, 0.05, "front", "front lower lip projection"),
    ModifierSpec("rear_bumper_crease_height", 0.05, 0.07, "rear", "rear bumper shoulder height"),
    ModifierSpec("plate_recess", 0.025, 0.025, "rear", "licence plate pocket depth"),
    ModifierSpec("diffuser_step", 0.012, 0.06, "rear", "lower rear valance setback"),
    ModifierSpec("tailgate_panel", 0.0, 0.04, "rear", "lower tailgate panel inset"),
    ModifierSpec("windshield_length", 0.25, 0.30, "greenhouse", "windshield rake / horizontal run"),
    ModifierSpec("roof_height", 0.15, 0.28, "greenhouse", "roof height"),
    ModifierSpec("roof_drop", 0.02, 0.06, "greenhouse", "roof falling towards the rear"),
    ModifierSpec("roof_rear_offset", 0.45, 0.45, "greenhouse", "where the roof ends (cab length)"),
    ModifierSpec("rear_window_length", 0.30, 0.35, "greenhouse", "rear window rake / horizontal run"),
    ModifierSpec("roof_width_ratio", 0.08, 0.10, "greenhouse", "tumblehome (narrow <-> wide roof)"),
    ModifierSpec("roof_crown", 0.02, 0.04, "greenhouse", "transverse roof curvature"),
    ModifierSpec("shoulder_inset", 0.03, 0.04, "greenhouse", "glass base inboard of the belt"),
    ModifierSpec("a_pillar_width", 0.025, 0.04, "greenhouse", "A-pillar painted band width"),
    ModifierSpec("c_pillar_width", 0.03, 0.08, "greenhouse", "C-pillar longitudinal width"),
    ModifierSpec("glass_recess", 0.006, 0.008, "greenhouse", "glass below the pillar surface"),
    ModifierSpec("roof_edge_radius", 0.010, 0.015, "greenhouse", "rolled roof-side edge radius"),
    ModifierSpec("drip_rail", 0.003, 0.004, "greenhouse", "roof-edge drip bead projection"),
    ModifierSpec("a_pillar_lean", 0.25, 0.30, "greenhouse", "A pillar diagonal"),
    ModifierSpec("c_pillar_lean", 0.25, 0.30, "greenhouse", "C pillar diagonal"),
    ModifierSpec("front_door_length", 0.25, 0.30, "greenhouse", "B pillar position"),
    ModifierSpec("quarter_window_length", 0.0, 0.45, "greenhouse", "rear quarter window"),
    ModifierSpec("deck_height", 0.10, 0.12, "rear", "trunk lid / rear window base height"),
    ModifierSpec("deck_rear_height", 0.10, 0.10, "rear", "tail height"),
    ModifierSpec("tail_width_ratio", 0.10, 0.12, "rear", "flat tail width"),
    ModifierSpec("rear_corner_length", 0.15, 0.20, "rear", "plan-view rounding of the tail"),
    ModifierSpec("rear_fascia_rake", 0.06, 0.08, "rear", "tail top protruding over the bumper"),
    ModifierSpec("rear_bumper_bottom", 0.08, 0.10, "rear", "rear bumper lower edge"),
    ModifierSpec("bed_depth", 0.0, 0.55, "rear", "open pickup bed depth"),
]


# ---------------------------------------------------------------- sculpts
def _side_weight(base: Mesh):
    """Signed lateral direction (+1 left side, -1 right side, 0 centre) per vertex."""
    y = base.vertices[:, 1]
    return np.sign(y) * np.clip(np.abs(y) / 0.3, 0, 1)


def sculpt_haunches(base: Mesh, params: BodyParams) -> Target:
    """Muscular bulge over the rear wheel arches (like a Coke-bottle shoulder)."""
    v = base.vertices
    L = params.layout()
    xr = L["x_ra"]
    side = _side_weight(base)
    ax = np.exp(-((v[:, 0] - xr - 0.1) / 0.55) ** 2)
    az = smoothstep(params.sill_height + 0.1, params.belt_height - 0.15, v[:, 2]) * (1 - smoothstep(params.belt_height + 0.02, params.belt_height + 0.2, v[:, 2]))
    amp = 0.05 * ax * az * side
    off = np.zeros_like(v)
    off[:, 1] = amp
    return Target("sculpt-haunches", off, "rear fender muscle bulge")


def sculpt_side_crease(base: Mesh, params: BodyParams) -> Target:
    """A sharp character line just below the belt (pushes the skin out above it, in below)."""
    v = base.vertices
    side = _side_weight(base)
    zc = params.belt_height - 0.12
    d = v[:, 2] - zc
    prof = np.where(d > 0, np.exp(-(d / 0.05) ** 2), -0.6 * np.exp(-(d / 0.08) ** 2))
    L = params.layout()
    ax = smoothstep(L["x_rear"] + 0.4, L["x_rear"] + 0.9, v[:, 0]) * (1 - smoothstep(L["x_front"] - 0.9, L["x_front"] - 0.4, v[:, 0]))
    off = np.zeros_like(v)
    off[:, 1] = 0.02 * prof * side * ax
    return Target("sculpt-side_crease", off, "character crease below the belt line")


def sculpt_wedge(base: Mesh, params: BodyParams) -> Target:
    """Wedge stance: nose down, tail up (rotation about the lateral axis)."""
    v = base.vertices
    ang = np.radians(1.6)
    dz = -np.sin(ang) * v[:, 0]
    # keep the wheel arches at the same height by counter-lifting the underbody
    off = np.zeros_like(v)
    off[:, 2] = dz * smoothstep(params.ground_clearance + 0.15, params.sill_height + 0.2, v[:, 2])
    return Target("sculpt-wedge", off, "nose-down wedge stance")


def sculpt_roof_bubble(base: Mesh, params: BodyParams) -> Target:
    v = base.vertices
    L = params.layout()
    xc = 0.5 * (L["u_roof_front"] + L["u_roof_rear"])
    ax = np.exp(-((v[:, 0] - xc) / 0.7) ** 2)
    az = smoothstep(params.roof_height - 0.25, params.roof_height - 0.03, v[:, 2])
    off = np.zeros_like(v)
    off[:, 2] = 0.06 * ax * az
    return Target("sculpt-roof_bubble", off, "raised roof bubble")


SCULPTS = {
    "haunches": sculpt_haunches,
    "side_crease": sculpt_side_crease,
    "wedge": sculpt_wedge,
    "roof_bubble": sculpt_roof_bubble,
}


# ------------------------------------------------------------------ build
def _generate(params: BodyParams) -> Mesh:
    return BodyGenerator(params).build()


LIBRARY_VERSION = 9


def _cache_dir():
    import os
    from pathlib import Path
    d = Path(os.environ.get("MAKECAR_CACHE", Path.home() / ".cache" / "makecar"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(params: BodyParams) -> str:
    import hashlib, json
    from .generator import SEGMENTS, UPPER_COUNTS, LOWER_COUNTS
    payload = json.dumps([LIBRARY_VERSION, params.to_dict(), SEGMENTS, UPPER_COUNTS, LOWER_COUNTS,
                          [(m.param, m.delta_minus, m.delta_plus) for m in BODY_MODIFIER_SPECS], sorted(STYLE_OVERRIDES.items())],
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _save_library(mm: MorphableMesh, path):
    arrays = {}
    for name, t in mm.targets.items():
        arrays[f"t:{name}"] = t.offsets.astype(np.float32)
    np.savez_compressed(path, **arrays)


def _load_library(mm_skeleton: MorphableMesh, path) -> bool:
    try:
        data = np.load(path)
    except Exception:
        return False
    for name, t in mm_skeleton.targets.items():
        key = f"t:{name}"
        if key not in data or data[key].shape != t.offsets.shape:
            return False
        t.offsets = data[key].astype(float)
    return True


def _skeleton(params: BodyParams) -> MorphableMesh:
    """Modifier structure with zero targets (filled from cache or by generation)."""
    base = _generate(params)
    mm = MorphableMesh(base, "car_body")
    zeros = np.zeros_like(base.vertices)
    for spec in BODY_MODIFIER_SPECS:
        name = spec.name or spec.param
        incr = Target(f"{name}-incr", zeros.copy()) if spec.delta_plus else None
        decr = Target(f"{name}-decr", zeros.copy()) if spec.delta_minus else None
        mm.add_modifier(Modifier(name, incr, decr, -1.0 if decr is not None else 0.0, 1.0, 0.0, spec.group,
                                 spec.description, unit_scale=spec.delta_plus))
    for style in STYLE_OVERRIDES:
        if style == "sedan":
            continue
        mm.add_modifier(Modifier(f"style/{style}", Target(f"style-{style}", zeros.copy()), None, 0.0, 1.0, 0.0, "style", f"macro target: {style}"))
    for name in SCULPTS:
        mm.add_modifier(Modifier(f"sculpt/{name}", Target(f"sculpt-{name}", zeros.copy()), None, -1.0, 1.0, 0.0, "sculpt", name))
    return mm


@lru_cache(maxsize=4)
def _cached_library(params_key: tuple) -> MorphableMesh:
    params = BodyParams(**dict(params_key))
    cache_file = _cache_dir() / f"body_targets_{_cache_key(params)}.npz"
    skel = _skeleton(params)
    if cache_file.exists() and _load_library(skel, cache_file):
        return skel
    builder = DifferentialTargetBuilder(_generate, params, name="car_body")
    macros = {name: style_params(name, params) for name in STYLE_OVERRIDES if name != "sedan"}
    mm = builder.build(BODY_MODIFIER_SPECS, macros=macros, macro_group="style")
    for name, fn in SCULPTS.items():
        t = fn(mm.base, params)
        mm.add_modifier(Modifier(f"sculpt/{name}", t, None, -1.0, 1.0, 0.0, "sculpt", t.description))
    try:
        _save_library(mm, cache_file)
    except OSError:
        pass
    return mm


def body_library(params: BodyParams | None = None) -> MorphableMesh:
    """Morphable body for the given base parameters (cached)."""
    params = params or BodyParams()
    key = tuple(sorted(params.to_dict().items()))
    return _cached_library(key)
