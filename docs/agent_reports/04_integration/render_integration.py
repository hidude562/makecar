#!/usr/bin/env python3
"""Matched integration screenshots using makecar's real NumPy Renderer.

Run from the assigned integration worktree (standard library + NumPy only):
  python3 docs/agent_reports/04_integration/render_integration.py baseline
  python3 docs/agent_reports/04_integration/render_integration.py after

Baseline source is ALWAYS git archive c8a4407, never the live working tree.
AFTER copies current makecar source, checking for concurrent edits. Both use an
isolated /tmp source/cache, and verify every imported makecar module came from
that source. No pickle snapshots or external renderer dependencies are used.
The baseline locks cameras; AFTER refuses to run without those camera settings.
Every view renders the COMPLETE assembly, including closeups: no components,
materials, transparent panes or offending geometry are removed or displaced.
--styles / --views allow selective rerenders; manifests are per-style so a
partial style rerender does not invalidate unrelated style records.
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
BASELINE = "c8a44076540c091a88970cf1022c56c894d97ffe"
STYLES = ("sedan", "sports", "pickup", "suv")
VIEWS = ("front_three_quarter", "rear_three_quarter", "side",
         "nose_closeup", "rear_bumper_exhaust_closeup", "wheel_closeup")
PAINT = "#426b88"


def hashes(source):
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source / "makecar").rglob("*.py"))}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def snapshot(phase, destination):
    if phase == "baseline":
        archive = git("archive", "--format=tar", BASELINE, "makecar")
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            tf.extractall(destination, filter="data")
        return BASELINE, hashes(destination)
    before = hashes(ROOT)
    shutil.copytree(ROOT / "makecar", destination / "makecar",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    captured = hashes(destination)
    if before != captured or hashes(ROOT) != captured:
        raise RuntimeError("Source changed while freezing AFTER; retry after editors finish")
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


def fitted_camera(points, azimuth, elevation, margin, width, height, Camera):
    target = (points.min(axis=0) + points.max(axis=0)) * 0.5
    cam = Camera.orbit(target, 1.0, azimuth, elevation, fov_deg=30)
    local = (points - target) @ cam.view_matrix()[:3, :3].T
    tangent = np.tan(np.radians(cam.fov_deg / 2))
    demand = np.maximum(np.abs(local[:, 0]) / (tangent * width / height),
                        np.abs(local[:, 1]) / tangent)
    distance = max(float(np.max(local[:, 2] + margin * demand)), 0.4)
    cam.eye = target + (cam.eye - target) * distance
    return {"eye": cam.eye.tolist(), "target": target.tolist(),
            "up": cam.up.tolist(), "fov_deg": cam.fov_deg}


def camera_settings(cars, Camera):
    settings = {"baseline_revision": BASELINE, "width": 800, "height": 600,
                "supersample": 2, "palette": {"paint": PAINT}, "seed": 0,
                "background": [[0.93, 0.95, 0.98], [0.78, 0.82, 0.87]],
                "light_dir": [-0.5, 0.6, 1.0], "fill_dir": [0.7, -0.4, 0.4],
                "selection": "complete assembly; no clipping or omissions", "styles": {}}
    for style, car in cars.items():
        mesh, parts = car["mesh"], car["parts"]
        vertices = mesh.vertices
        # Focus bounds affect the camera ONLY; Renderer still receives full mesh.
        front = vertices[vertices[:, 0] >= vertices[:, 0].max() - 0.62]
        rear = vertices[vertices[:, 0] <= vertices[:, 0].min() + 0.65]
        rear = rear[rear[:, 2] <= max(parts["taillight_L"].vertices[:, 2]) + 0.08]
        wheel = parts["wheel_front_L"].vertices
        definitions = (
            ("front_three_quarter", vertices, 35, 17, 1.13),
            ("rear_three_quarter", vertices, 145, 18, 1.13),
            ("side", vertices, 90, 3, 1.13),
            ("nose_closeup", front, 15, 9, 1.16),
            ("rear_bumper_exhaust_closeup", rear, 165, 4, 1.16),
            ("wheel_closeup", wheel, 65, 10, 1.85),
        )
        settings["styles"][style] = {
            name: {"camera": fitted_camera(points, az, el, margin, 800, 600, Camera),
                   "ground": True, "lines": True}
            for name, points, az, el, margin in definitions}
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("baseline", "after"))
    parser.add_argument("--styles", nargs="+", choices=STYLES, default=list(STYLES))
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    cameras_path = REPORT / "cameras.json"
    if args.phase == "after" and not cameras_path.exists():
        parser.error("Render baseline first to lock the matched cameras")
    with tempfile.TemporaryDirectory(prefix=f"makecar-integration-{args.phase}-", dir="/tmp") as temp:
        source = Path(temp).resolve()
        revision, source_hashes = snapshot(args.phase, source)
        # Imports are delayed until the trusted source snapshot is installed.
        # Discard worktree/PYTHONPATH entries as additional protection against mixing.
        sys.dont_write_bytecode = True
        sys.path[:] = [p for p in sys.path if p and not Path(p).resolve().is_relative_to(ROOT)]
        sys.path.insert(0, str(source))
        os.environ["PYTHONPATH"] = str(source)
        os.environ["MAKECAR_CACHE"] = str(source / ".cache")
        from makecar.body import CarBody
        from makecar.assembly import assemble
        from makecar.components.base import Palette
        from makecar.export.render import Camera, Renderer

        body, palette, cars = CarBody(), Palette(paint=PAINT), {}
        for style in STYLES:
            assembly = assemble(body.build(style=style, paint=palette.paint), palette=palette, seed=0)
            cars[style] = {"mesh": assembly.mesh(),
                           "parts": {i.connector.name: i.result.mesh for i in assembly.instances},
                           "components": {i.connector.name: {"component": i.component,
                               "faces": i.result.mesh.n_faces, "options": i.options}
                               for i in assembly.instances},
                           "measurements": assembly.body.measurements}
            print(f"Captured {args.phase}/{style}: {cars[style]['mesh'].n_faces:,} faces", flush=True)
        imported = verify_imports(source)
        if hashes(source) != source_hashes:
            raise RuntimeError("Frozen source changed while assembling")
        if cameras_path.exists():
            settings = json.loads(cameras_path.read_text())
            if settings["baseline_revision"] != BASELINE:
                raise RuntimeError("Camera baseline does not match fixed baseline revision")
        else:
            settings = camera_settings(cars, Camera)
            cameras_path.write_text(json.dumps(settings, indent=2) + "\n")
        captured_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for style in args.styles:
            car = cars[style]
            output = REPORT / args.phase / style
            output.mkdir(parents=True, exist_ok=True)
            manifest_path = output / "manifest.json"
            manifest = {"phase": args.phase, "style": style, "captured_at": captured_at,
                        "baseline_revision": BASELINE, "source_head": revision,
                        "source_kind": "git archive" if args.phase == "baseline" else "frozen working tree",
                        "source_sha256": source_hashes, "imported_modules": imported,
                        "python": sys.version.split()[0], "numpy": np.__version__,
                        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "settings": settings | {"styles": {style: settings["styles"][style]}},
                        "full_mesh": car["mesh"].stats(), "components": car["components"],
                        "images": {}}
            if manifest_path.exists() and set(args.views) != set(VIEWS):
                old = json.loads(manifest_path.read_text())
                if old["source_sha256"] != source_hashes or old["settings"] != manifest["settings"]:
                    raise RuntimeError("Partial rerender requires unchanged source and cameras")
                manifest["images"] = old["images"]
            for view in args.views:
                spec = settings["styles"][style][view]
                renderer = Renderer(settings["width"], settings["height"], settings["supersample"],
                                    background=settings["background"])
                renderer.light_dir = renderer._unit(np.array(settings["light_dir"]))
                renderer.fill_dir = renderer._unit(np.array(settings["fill_dir"]))
                start = time.monotonic()
                image = renderer.render(car["mesh"], Camera(**spec["camera"]),
                                        ground=spec["ground"], lines=spec["lines"])
                if image.shape != (settings["height"], settings["width"], 3) or not np.isfinite(image).all():
                    raise RuntimeError(f"Invalid image: {style}/{view}")
                path = output / f"{view}.png"
                renderer.save(image, path)
                manifest["images"][view] = {"path": str(path.relative_to(REPORT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "faces_rendered": car["mesh"].n_faces, "faces_removed": 0,
                    "seconds": round(time.monotonic() - start, 3), **spec}
                print(f"Rendered {path.relative_to(REPORT)}", flush=True)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        verify_imports(source)
        if hashes(source) != source_hashes:
            raise RuntimeError("Frozen source changed during rendering")


if __name__ == "__main__":
    main()
