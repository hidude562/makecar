"""Render actual before/after lamp geometry with matched diagnostic cameras.

Run from the worktree root:
    python3 docs/agent_reports/04_integration/render_lamps.py

Only the three original lamp classes are loaded from the integration baseline;
other clusters' current geometry is not mislabeled as a historical snapshot.
These are geometric renderings, not plots. Context views show the unchanged body.
"""
import argparse
from pathlib import Path
import ast
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from makecar.body import CarBody
from makecar.components import BuildContext, Palette
from makecar.components import exterior
from makecar.export.render import Camera, Renderer, default_materials

BASELINE = "c8a4407"
OUTPUT = Path(__file__).resolve().parent / "lamps"


def baseline_classes():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:makecar/components/exterior.py"], cwd=ROOT, text=True)
    module = ast.parse(source)
    module.body = [node for node in module.body if isinstance(node, ast.ClassDef)
                   and node.name in {"_LampSurface", "Headlight", "Taillight"}]
    namespace = dict(vars(exterior))
    namespace["register"] = lambda cls: cls  # Never replace registered components.
    exec(compile(module, f"{BASELINE}:lamp-classes", "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modifiers", action="store_true", help="render supported-modifier compatibility cases")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = baseline_classes()
    generator = CarBody()
    renderer = Renderer(800, 600, supersample=2)
    cases = [(style, kind, {}, style) for style, kind in
             (("pickup", "headlight"), ("sports", "headlight"), ("wagon", "taillight"))]
    if args.modifiers:
        cases = [(style, kind, {key: value}, f"{style}_{key}_{value:g}") for style, kind, key, value in [
            ("sports", "headlight", "hood_front_height", -.25),
            ("sports", "headlight", "hood_front_height", -1),
            ("sports", "headlight", "bumper_crease_height", 1),
            ("sports", "headlight", "fender_crease", 1),
            ("pickup", "taillight", "tailgate_panel", 1),
            ("suv", "taillight", "rear_fascia_rake", -1),
        ]]
    for style, kind, modifiers, label in cases:
        body = generator.build(style=style, modifiers=modifiers)
        connector = body.connector(f"{kind}_L")
        context = BuildContext(dict(body.measurements), dict(body.hints), Palette(), np.random.default_rng(0))
        cls_name = "Headlight" if kind == "headlight" else "Taillight"
        lamps = {"before": before[cls_name]().build(connector, None, context).mesh,
                 "after": getattr(exterior, cls_name)().build(connector, None, context).mesh}
        target = (connector.points.min(axis=0) + connector.points.max(axis=0)) / 2
        sign = 1 if kind == "headlight" else -1
        azimuth = 0 if sign > 0 else 180
        # Three angles; third view exposes the lower outboard pane/body seam.
        cameras = {
            "front": Camera.orbit(target, 2., azimuth, 8, ortho=True,
                                  ortho_height=max(.20, np.ptp(connector.points[:, 2]) * 1.4)),
            "oblique": Camera.orbit(target, 2., 42 if sign > 0 else 138, 28,
                                    ortho=True, ortho_height=max(.25, np.ptp(connector.points[:, 2]) * 1.4)),
        }
        # Fit BOTH snapshots to the same camera, including wide sports lamps
        # and the housing depth visible at an oblique angle.
        union = np.vstack([lamp.vertices for lamp in lamps.values()])
        for camera in cameras.values():
            view = camera.view_matrix()
            projected = union @ view[:3, :3].T + view[:3, 3]
            extent = np.abs(projected[:, :2]).max(axis=0)
            camera.ortho_height = 2.5 * max(extent[1], extent[0] / (800 / 600))
        if style == "sports" and not modifiers:
            seam = np.asarray(connector.meta["grid_points"])[1, 0]
            seam_height = .075
        else:
            seam = target
            seam_height = max(.25, np.ptp(connector.points[:, 2]) * 1.4)
        cameras["seam_context"] = Camera.orbit(seam, 2., 70 if sign > 0 else 110, 5,
                                               ortho=True, ortho_height=seam_height)
        for phase, lamp in lamps.items():
            for name, camera in cameras.items():
                mesh = lamp.copy()
                if name == "seam_context":
                    shell = body.mesh.copy()
                    shell.materials.update(default_materials())
                    shell.materials["paint"] = context.material("paint")
                    mesh = shell.merge(mesh)
                filename = OUTPUT / f"{label}_{kind}_{phase}_{name}.png"
                renderer.save(renderer.render(mesh, camera, ground=False, lines=False), filename)
                print(filename.relative_to(ROOT))


if __name__ == "__main__":
    main()
