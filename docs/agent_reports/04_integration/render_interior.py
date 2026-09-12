#!/usr/bin/env python3
"""Supplemental, explicitly isolated interior diagnostics and full-context wheels.

python3 docs/agent_reports/04_integration/render_interior.py --phase before
python3 docs/agent_reports/04_integration/render_interior.py --phase after

Before imports the actual integration baseline c8a4407 from a temporary git
archive, never current geometry relabelled as before. Controls isolate the
complete dashboard/console trees. Footwell views isolate floor/firewall only;
wheel closeups retain every assembled part and reuse cameras.json. Matched
views use identical cameras/materials, with no hidden-surface corrections.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
BASELINE = "c8a4407"


def render(phase, source):
    sys.path.insert(0, str(source))
    import numpy as np
    from makecar.assembly import assemble
    from makecar.body.body import CarBody
    from makecar.export.render import Camera, Renderer
    from makecar.geometry.mesh import Mesh

    def selected(assembly, roots):
        mesh = Mesh(name="isolated_interior_diagnostic")
        for part in assembly.instances:
            if part.connector.name.split("/", 1)[0] in roots:
                mesh.merge(part.result.mesh)
        return mesh

    renderer = Renderer(960, 640, supersample=2)

    def save(mesh, camera, filename, ground=False):
        path = OUTPUT / filename
        renderer.save(renderer.render(mesh, camera, ground=ground, lines=False), path)
        print(path, flush=True)

    sports = assemble(CarBody().build("sports"))
    controls = selected(sports, {"dashboard", "console"})
    for name, eye, fov in (("front", [-.95, 0, .83], 55),
                           ("oblique", [-.65, -.85, 1.03], 55),
                           ("closeup", [-.2, -.22, .7], 46)):
        save(controls, Camera(eye, [.35, 0, .64], fov_deg=fov), f"sports_controls_{phase}_{name}.png")

    settings = json.loads((OUTPUT / "cameras.json").read_text())
    for style in ("pickup", "suv"):
        body = CarBody().build(style)
        assembly = assemble(body)
        renderer = Renderer(800, 600, supersample=2)
        camera = Camera(**settings["styles"][style]["wheel_closeup"]["camera"])
        save(assembly.mesh(), camera, f"{style}_wheel_footwell_{phase}.png", ground=True)
        footwell = selected(assembly, {"floor", "firewall"})
        target = np.array([body.measurements["x_cowl"] - .15, 0, body.measurements["z_floor"] + .4])
        for name, delta in (("oblique", [-1.6, -1.6, 1.15]), ("side", [0, -2.8, .1])):
            save(footwell, Camera(target + delta, target, fov_deg=52), f"{style}_footwell_{phase}_{name}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after"), default="after")
    args = parser.parse_args()
    if args.phase == "after":
        render(args.phase, ROOT)
        return
    archive = subprocess.check_output(["git", "-C", str(ROOT), "archive", BASELINE, "makecar"])
    with tempfile.TemporaryDirectory(prefix="makecar-interior-integration-before-") as directory:
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(directory, filter="data")
        render(args.phase, Path(directory))


if __name__ == "__main__":
    main()
