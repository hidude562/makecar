"""Whole-car fit invariants.

These guard the geometry that a purely visual review misses: interior parts
must stay inside the body shell, exterior parts may only protrude where a real
car protrudes, and nothing may sink below the road.  They run over every style
so a change tuned on the sedan cannot quietly break the van or the pickup.
"""
import numpy as np
import pytest

from makecar.assembly import _is_interior
from makecar.body.styles import style_names
from makecar.config import CarConfig
from makecar.pipeline import build_car

# exterior parts that legitimately stand outside the painted shell
PROTRUDING = ("wheel", "mirror", "exhaust", "tow", "plate", "badge", "intake", "grille",
              "antenna", "rail", "wiper", "spoiler", "diffuser", "splitter", "flap")


@pytest.fixture(scope="module")
def built():
    return {s: build_car(CarConfig.from_dict({"name": s, "body": {"style": s}})) for s in style_names()}


def _outside(inst, blo, bhi):
    lo, hi = inst.result.mesh.bounds()
    return max(blo[0] - lo[0], hi[0] - bhi[0], blo[1] - lo[1], hi[1] - bhi[1], blo[2] - lo[2], hi[2] - bhi[2])


@pytest.mark.parametrize("style", style_names())
def test_interior_parts_stay_inside_the_body(built, style):
    res, asm, _ = built[style]
    blo, bhi = res.full_mesh.bounds()
    bad = [(inst.id, round(_outside(inst, blo, bhi), 3)) for inst in asm.instances
           if _is_interior(inst.connector) and _outside(inst, blo, bhi) > 0.02]
    assert not bad, f"{style}: interior parts poke through the body shell: {bad}"


@pytest.mark.parametrize("style", style_names())
def test_exterior_parts_only_protrude_where_expected(built, style):
    res, asm, _ = built[style]
    blo, bhi = res.full_mesh.bounds()
    bad = [(inst.id, round(_outside(inst, blo, bhi), 3)) for inst in asm.instances
           if not _is_interior(inst.connector)
           and not any(k in inst.id for k in PROTRUDING)
           and _outside(inst, blo, bhi) > 0.05]
    assert not bad, f"{style}: unexpected exterior protrusions: {bad}"


@pytest.mark.parametrize("style", style_names())
def test_nothing_sinks_below_the_road(built, style):
    _, asm, _ = built[style]
    lo, _ = asm.mesh().bounds()
    assert lo[2] > -0.005, f"{style}: geometry below the ground plane at z={lo[2]:.3f}"


@pytest.mark.parametrize("style", style_names())
def test_wheels_touch_the_road(built, style):
    res, asm, _ = built[style]
    for inst in asm.instances:
        if not inst.component.startswith("wheel."):  # road wheels only, not steering.wheel
            continue
        lo, _ = inst.result.mesh.bounds()
        assert lo[2] < 0.012, f"{style}: {inst.id} floats {lo[2]:.3f} m above the road"
