#!/usr/bin/env python3
"""Matched shape/exhaust renders with the actual makecar NumPy Renderer.

From the assigned worktree (runtime: standard library + NumPy/PyYAML):
  OPENBLAS_NUM_THREADS=1 python3 docs/agent_reports/06_shape_integration/render_shape_integration.py before
  OPENBLAS_NUM_THREADS=1 python3 docs/agent_reports/06_shape_integration/render_shape_integration.py after

BEFORE always imports a git archive of 2a4a5e7 in /tmp, NEVER live code.
AFTER freezes the current worktree's makecar package in /tmp and refuses mixed
or concurrently changed source. Both have independent fresh target caches.
BEFORE locks cameras/materials/lights in cameras.json. AFTER reuses them exactly.
Existing before artifacts are never overwritten; reproduce in --output /tmp/...
To render AFTER elsewhere, copy cameras.json to that output directory first.

underbody/side/rear_closeup are SELECTED-PART DIAGNOSTICS: actual shell, four
wheels, exhaust system and both tips; other assembled components are omitted.
Only the underbody material is recolored grey, with below-directed lighting,
no ground/shadow or seam overlay. No geometry is displaced or clipped. Each
manifest lists all selected/omitted instances and exact material overrides.
Sports rear_three_quarter is the COMPLETE assembly with original materials,
above-directed lighting, ground/shadow and seams; no parts removed or recolored.
These pictures expose routing, but cannot establish a 4 mm numerical clearance.
Adapted from task 04 render_exhaust.py and render_integration.py.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

import numpy as np

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
BASELINE = "2a4a5e786783ad5d3440029297f6b082d41fa209"
STYLES = ("sports", "sedan", "pickup")
DIAGNOSTIC_PARTS = ("wheel_front_L", "wheel_front_R", "wheel_rear_L", "wheel_rear_R",
                    "exhaust_system", "exhaust_L", "exhaust_R")
PAINT = "#426b88"
PAN = {"name": "underbody", "color": [.40, .42, .44], "alpha": 1., "shininess": .2}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(source):
    return {str(p.relative_to(source)): sha(p)
            for p in sorted((source / "makecar").rglob("*.py"))}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def snapshot(phase, destination):
    if phase == "before":
        archive = git("archive", "--format=tar", BASELINE, "makecar")
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            tf.extractall(destination, filter="data")
        return BASELINE, hashes(destination)
    before = hashes(ROOT)
    shutil.copytree(ROOT / "makecar", destination / "makecar",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    captured = hashes(destination)
    if before != captured or hashes(ROOT) != captured:
        raise RuntimeError("Source changed while freezing AFTER; retry when edits finish")
    return git("rev-parse", "HEAD").decode().strip(), captured


def verify_imports(source):
    imported = {}
    for name, module in sorted(sys.modules.items()):
        if name == "makecar" or name.startswith("makecar."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(source):
                raise RuntimeError(f"Mixed source import: {name} came from {path}")
            imported[name] = str(path.relative_to(source))
    return imported


def camera_dict(camera):
    return {"eye": camera.eye.tolist(), "target": camera.target.tolist(),
            "up": camera.up.tolist(), "fov_deg": camera.fov_deg,
            "ortho": camera.ortho, "ortho_height": camera.ortho_height}


def fitted_camera(points, azimuth, elevation, margin, Camera):
    target = (points.min(axis=0) + points.max(axis=0)) * .5
    cam = Camera.orbit(target, 10, azimuth, elevation, ortho=True)
    local = (points - target) @ cam.view_matrix()[:3, :3].T
    cam.ortho_height = float(2 * margin * max(np.abs(local[:, 1]).max(),
                                              np.abs(local[:, 0]).max() / (800 / 600)))
    return camera_dict(cam)


def camera_settings(cars, Camera):
    settings = {"baseline_revision": BASELINE, "width": 800, "height": 600,
                "supersample": 2, "palette": {"paint": PAINT}, "seed": 0,
                "background": [[.93, .95, .98], [.78, .82, .87]],
                "pan_material_override": PAN, "styles": {}}
    for style, car in cars.items():
        vertices = car["diagnostic"].vertices
        rear = vertices[(vertices[:, 0] < car["assembly"].body.measurements["x_rear"] + .85)
                        & (vertices[:, 2] < .65)]
        definitions = [("underbody", vertices, 145, -35, 1.12),
                       ("side", vertices, 90, -8, 1.12),
                       ("rear_closeup", rear, 160, -15, 1.12)]
        views = {}
        for name, points, azimuth, elevation, margin in definitions:
            views[name] = {"camera": fitted_camera(points, azimuth, elevation, margin, Camera),
                           "scene": "diagnostic", "ground": False, "lines": False,
                           "light_dir": [-.5, .6, -1.], "fill_dir": [.7, -.4, -.4]}
        if style == "sports":
            views["rear_three_quarter"] = {
                "camera": fitted_camera(car["full"].vertices, 145, 18, 1.12, Camera),
                "scene": "full", "ground": True, "lines": True,
                "light_dir": [-.5, .6, 1.], "fill_dir": [.7, -.4, .4]}
        settings["styles"][style] = views
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("before", "after"))
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cameras_path = output / "cameras.json"
    manifest_path = output / f"{args.phase}_manifest.json"
    if args.phase == "before" and (manifest_path.exists() or (output / "before").exists()):
        parser.error("BEFORE is immutable; choose a new --output directory to reproduce")
    if args.phase == "after" and not cameras_path.exists():
        parser.error("Render BEFORE first (or copy its cameras.json) to lock matched settings")
    settings = json.loads(cameras_path.read_text()) if cameras_path.exists() else None
    if settings and settings["baseline_revision"] != BASELINE:
        raise RuntimeError("Camera baseline revision does not match")

    with tempfile.TemporaryDirectory(prefix=f"makecar-shape-integration-{args.phase}-", dir="/tmp") as temp:
        source = Path(temp).resolve()
        revision, source_hashes = snapshot(args.phase, source)
        sys.dont_write_bytecode = True
        sys.path[:] = [p for p in sys.path if p and not Path(p).resolve().is_relative_to(ROOT)]
        sys.path.insert(0, str(source))
        os.environ["PYTHONPATH"] = str(source)
        os.environ["MAKECAR_CACHE"] = str(source / ".cache")
        from makecar.body import CarBody
        from makecar.body.shapes import resolve_shape_values
        from makecar.assembly import assemble
        from makecar.components.base import Palette
        from makecar.export.render import Camera, Renderer
        from makecar.geometry.mesh import Material

        palette = Palette(**(settings["palette"] if settings else {"paint": PAINT}))
        seed = settings["seed"] if settings else 0
        pan = settings["pan_material_override"] if settings else PAN
        builder, cars, records = CarBody(), {}, {}
        for style in STYLES:
            body = builder.build(style=style, paint=palette.paint)
            assembly = assemble(body, palette=palette, seed=seed)
            full, diagnostic = assembly.mesh(), body.mesh.copy()
            selected, omitted = [], []
            for instance in assembly.instances:
                if instance.connector.name in DIAGNOSTIC_PARTS:
                    diagnostic.merge(instance.result.mesh, group_prefix=instance.id)
                    selected.append(instance.id)
                else:
                    omitted.append(instance.id)
            if {i.connector.name for i in assembly.instances
                if i.connector.name in DIAGNOSTIC_PARTS} != set(DIAGNOSTIC_PARTS):
                raise RuntimeError(f"Missing selected diagnostic parts in {style}")
            original_pan = vars(diagnostic.materials["underbody"]).copy()
            diagnostic.materials["underbody"] = Material(**pan)
            cars[style] = {"assembly": assembly, "full": full, "diagnostic": diagnostic}
            records[style] = {
                "configuration": {"body": {"style": style, "modifiers": {}},
                                  "components": {"defaults": True}, "seed": seed,
                                  "palette": {"paint": palette.paint}},
                "requested_modifiers": body.modifier_values,
                "effective_shape_values": resolve_shape_values(body.modifier_values),
                "measurements": body.measurements,
                "components": [i.to_dict() for i in assembly.instances],
                "scenes": {
                    "full": {"stats": full.stats(), "selection": "complete default assembly",
                             "omitted_instances": [], "faces_removed": 0,
                             "geometry_displaced": False, "geometry_clipped": False,
                             "material_overrides": {}},
                    "diagnostic": {"stats": diagnostic.stats(),
                                   "selection": "actual body shell + selected assembled components",
                                   "selected_instances": selected, "omitted_instances": omitted,
                                   "faces_omitted_from_full_assembly": full.n_faces - diagnostic.n_faces,
                                   "faces_removed_from_selected_parts": 0,
                                   "geometry_displaced": False, "geometry_clipped": False,
                                   "material_overrides": {"underbody": {"original": original_pan, "rendered": pan}}}},
                "images": {}}
            print(f"Captured {args.phase}/{style}: {full.n_faces:,} full / {diagnostic.n_faces:,} diagnostic faces", flush=True)
        if settings is None:
            settings = camera_settings(cars, Camera)
            cameras_path.write_text(json.dumps(settings, indent=2) + "\n")
        manifest = {"phase": args.phase, "baseline_revision": BASELINE, "source_head": revision,
                    "source_kind": "git archive" if args.phase == "before" else "frozen working tree",
                    "source_sha256": source_hashes, "imported_modules": verify_imports(source),
                    "script_sha256": sha(Path(__file__)), "cameras_sha256": sha(cameras_path),
                    "python": sys.version.split()[0], "numpy": np.__version__,
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "settings": settings, "styles": records,
                    "limitations": "Diagnostic views omit unselected furniture and recolor the pan; images do not certify millimetre clearance."}
        for style, car in cars.items():
            destination = output / args.phase / style
            destination.mkdir(parents=True, exist_ok=True)
            for view, spec in settings["styles"][style].items():
                renderer = Renderer(settings["width"], settings["height"], settings["supersample"],
                                    background=settings["background"])
                renderer.light_dir = renderer._unit(np.asarray(spec["light_dir"]))
                renderer.fill_dir = renderer._unit(np.asarray(spec["fill_dir"]))
                mesh = car[spec["scene"]]
                start = time.monotonic()
                image = renderer.render(mesh, Camera(**spec["camera"]),
                                        ground=spec["ground"], lines=spec["lines"])
                if image.shape != (600, 800, 3) or not np.isfinite(image).all():
                    raise RuntimeError(f"Invalid render: {style}/{view}")
                path = destination / f"{view}.png"
                renderer.save(image, path)
                records[style]["images"][view] = {
                    "path": str(path.relative_to(output)), "sha256": sha(path),
                    "faces_rendered": mesh.n_faces, "seconds": round(time.monotonic() - start, 3), **spec}
                print(f"Rendered {path}", flush=True)
        verify_imports(source)
        if hashes(source) != source_hashes:
            raise RuntimeError("Frozen source changed while building/rendering")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
