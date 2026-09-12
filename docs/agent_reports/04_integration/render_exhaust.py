"""Reproduce the matched exhaust fitting diagnostics (not full-car beauty views).

Run from any directory: python3 <this-file> [--phase before|after|both]
The scene contains the CURRENT shell, four wheels, phase-specific tip mounts and
selected exhaust system only. Lamps, glazing, interior and other furniture are
intentionally omitted. The pan is light grey with below-directed lighting;
geometry is not displaced or cut away. Ground/shadows and seam overlays are off.
Before exhaust code is read from the fixed, local integration baseline commit;
after uses the working tree. Cameras and diagnostic materials are identical.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from makecar.body import CarBody
from makecar.components import BuildContext, Palette, get_component
from makecar.components import exterior
from makecar.export.render import Camera, Renderer, default_materials
from makecar.geometry.mesh import Material

BASELINE = "c8a4407"
OUTPUT = Path(__file__).resolve().parent


def baseline_exhaust():
    """Load this repository's trusted baseline, without changing the checkout."""
    def source(path):
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{BASELINE}:{path}"], text=True)

    connectors = {"__package__": "makecar.body"}
    mounts = {"__package__": "makecar.body"}
    exec(source("makecar/body/connectors_extra.py"), connectors)
    exec(source("makecar/body/connectors.py"), mounts)
    # Define only the old exhaust class, without executing registry decorators
    # or replacing any current wheel/tip implementations in the shared registry.
    code = source("makecar/components/exterior.py")
    component = dict(exterior.__dict__)
    for name in ("ExhaustSystem", "ExhaustTip"):
        definition = code[code.index(f"class {name}("):]
        exec(definition[:definition.index("\n\n@register")], component)
    return (connectors["extra_exterior_connectors"], component["ExhaustSystem"](),
            mounts["emit_connectors"], component["ExhaustTip"]())


def render(phase):
    builder = CarBody()
    renderer = Renderer(800, 600, supersample=2)
    renderer.light_dir = renderer._unit(np.array([-.5, .6, -1.]))
    renderer.fill_dir = renderer._unit(np.array([.7, -.4, -.4]))
    old_connectors, old_component, old_mounts, old_tip = baseline_exhaust() if phase != "after" else (None,) * 4
    phases = ("before", "after") if phase == "both" else (phase,)
    for style in ("sedan", "sports"):
        body = builder.build(style=style)
        ctx = BuildContext(body.measurements, body.hints, Palette(), np.random.default_rng(0))
        fixed = body.mesh.copy()
        fixed.materials.update(default_materials())
        fixed.materials["underbody"] = Material("underbody", (.40, .42, .44), 1., .2)
        for name in ("wheel_front_L", "wheel_front_R", "wheel_rear_L", "wheel_rear_R"):
            fixed.merge(get_component("wheel.alloy").build(body.connector(name), {}, ctx).mesh)
        for selected in phases:
            if selected == "before":
                conn = next(c for c in old_connectors(body.full_mesh, body.measurements, body.hints)
                            if c.name == "exhaust_system")
                component = old_component
                tips = [c for c in old_mounts(body.full_mesh, body.measurements, body.hints)
                        if c.name in ("exhaust_L", "exhaust_R")]
            else:
                conn = body.connector("exhaust_system")
                component = get_component("exhaust.system")
                tips = [body.connector(name) for name in ("exhaust_L", "exhaust_R")]
            scene = fixed.copy().merge(component.build(conn, {}, ctx).mesh)
            tip_component = old_tip if selected == "before" else get_component("exhaust.tip")
            for tip in tips:
                scene.merge(tip_component.build(tip, {}, ctx).mesh)
            meas = body.measurements
            views = (
                ("underbody", [0, 0, .2], 145, -35, meas["length"] * .66),
                ("side", [0, 0, .2], 90, -8, meas["length"] * .76),
                ("rear_closeup", [meas["x_rear"] + .30, 0, .29], 160, -15, 1.35),
            )
            for view, target, azimuth, elevation, height in views:
                camera = Camera.orbit(target, 8, azimuth, elevation, ortho=True, ortho_height=height)
                path = OUTPUT / f"exhaust_{selected}_{style}_{view}.png"
                renderer.save(renderer.render(scene, camera, ground=False, lines=False), path)
                print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after", "both"), default="both")
    render(parser.parse_args().phase)
