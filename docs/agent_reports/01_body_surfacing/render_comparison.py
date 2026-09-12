#!/usr/bin/env python3
"""Render actual build_car assemblies, with identical before/after cameras.

From any working directory:
  python3 /path/to/render_comparison.py before
  python3 /path/to/render_comparison.py after

All PNGs are 800x600. Four overview views include EVERY assembled component;
nose and A-pillar closeups show only the unchanged shell plus assembled glass
(no mirrors, wipers, grille or plates hiding the geometry). Underbody is a
full-assembly view without a ground shadow. The additional underbody_diagnostic
uses neutral clay materials and lights below the car to expose the floor tunnel;
this is explicitly a geometry diagnostic, not a beauty render. No charts or
synthetic car shapes.

The original baseline dimensions are embedded as JSON so an `after` run stays
reproducible even if /tmp has been cleaned. A `before` run saves exact cameras,
configs, measurements and face counts to --state (outside this report folder).
Subsequent runs reuse that JSON when available, never fitting to after bounds.
Use --state to keep a new comparison's state somewhere persistent if desired.

--snapshots reads trusted LOCAL pickle files from an earlier build_car snapshot
(before only); this lets geometry editing start before rasterisation completes.
Never use pickle snapshots received from an untrusted source.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from makecar.config import CarConfig
from makecar.export.render import Camera, Renderer
from makecar.pipeline import build_car

STYLES = ("sedan", "hatchback", "suv", "pickup", "sports")
VIEWS = ("three_quarter_front", "three_quarter_rear", "side", "front",
         "nose_closeup", "a_pillar_closeup", "underbody", "underbody_diagnostic")
DEFAULT_STATE = Path("/tmp/makecar-body-surfacing-baseline/cameras.json")
PAINT = "#91acc1"  # light silver-blue; deliberately identical across styles/phases

# Frozen pre-surfacing geometry, 2026-09-12. Bounds include the complete assembly.
# Only camera landmarks are embedded; the external state retains all measurements.
BASELINE = json.loads(r'''
{
  "sedan": {
    "bounds": [[-2.510650,-1.039900,0],[2.348805,1.039900,1.486488]],
    "measurements": {"x_front":2.300735,"x_cowl":1.025,"z_cowl":0.98,"x_roof_front":0.125,"z_roof_front":1.45,"y_shoulder":0.859716,"roof_half_width":0.6256,"nose_z_top":0.76,"nose_z_bottom":0.3}
  },
  "hatchback": {
    "bounds": [[-2.108,-1.019963,-0.000024],[2.182967,1.019963,1.496919]],
    "measurements": {"x_front":2.135744,"x_cowl":1.01,"z_cowl":0.98,"x_roof_front":0.16,"z_roof_front":1.47,"y_shoulder":0.839567,"roof_half_width":0.63,"nose_z_top":0.74,"nose_z_bottom":0.3}
  },
  "suv": {
    "bounds": [[-2.575,-1.084228,0],[2.360739,1.084228,1.788374]],
    "measurements": {"x_front":2.320329,"x_cowl":1.175,"z_cowl":1.12,"x_roof_front":0.425,"z_roof_front":1.75,"y_shoulder":0.904993,"roof_half_width":0.6948,"nose_z_top":0.9,"nose_z_bottom":0.42}
  },
  "pickup": {
    "bounds": [[-2.978,-1.118232,0],[2.693341,1.118232,1.886668]],
    "measurements": {"x_front":2.658395,"x_cowl":1.4,"z_cowl":1.18,"x_roof_front":0.7,"z_roof_front":1.85,"y_shoulder":0.939726,"roof_half_width":0.75,"nose_z_top":0.95,"nose_z_bottom":0.45}
  },
  "sports": {
    "bounds": [[-2.185800,-1.074393,0],[2.172,1.074393,1.247523]],
    "measurements": {"x_front":2.080039,"x_cowl":0.6,"z_cowl":0.88,"x_roof_front":-0.35,"z_roof_front":1.22,"y_shoulder":0.891305,"roof_half_width":0.6045,"nose_z_top":0.62,"nose_z_bottom":0.2}
  }
}
''')


def config_for(style):
    return CarConfig.from_dict({
        "name": style, "seed": 0, "body": {"style": style},
        "palette": {"paint": PAINT}, "components": {"defaults": True},
    })


def capture(style, config=None):
    cfg = CarConfig.from_dict(config) if config else config_for(style)
    body, assembly, _ = build_car(cfg)
    full = assembly.mesh()
    detail = body.mesh.copy()
    for instance in assembly.instances:
        if "glass" in instance.connector.tags:
            detail.merge(instance.result.mesh)
    return {"config": cfg.raw, "measurements": body.measurements,
            "full": full, "detail": detail, "body_faces": body.mesh.n_faces,
            "full_faces": full.n_faces}


def cameras_for(bounds, m):
    lo, hi = np.asarray(bounds, dtype=float)
    length, width, height = hi - lo
    center = (lo + hi) / 2
    center[2] = height * 0.46
    nose = [m["x_front"] - 0.16, 0.10,
            (m["nose_z_top"] + m["nose_z_bottom"]) / 2 + 0.12]
    pillar = [(m["x_cowl"] + m["x_roof_front"]) / 2,
              (m["y_shoulder"] + m["roof_half_width"]) / 2,
              (m["z_cowl"] + m["z_roof_front"]) / 2]
    return {
        "three_quarter_front": Camera.orbit(center, 1.65 * length, 35, 16, fov_deg=30),
        "three_quarter_rear": Camera.orbit(center, 1.65 * length, -140, 18, fov_deg=30),
        "side": Camera.orbit(center, 4 * length, 90, 0, ortho=True,
                             ortho_height=length * 0.86),
        "front": Camera.orbit(center, 4 * length, 0, 0, ortho=True,
                              ortho_height=max(height * 1.3, width * 0.95)),
        "nose_closeup": Camera.orbit(nose, 8, 32, 12, ortho=True,
                                     ortho_height=max(1.6, width * 0.77)),
        "a_pillar_closeup": Camera.orbit(pillar, 8, 60, 12, ortho=True,
                                         ortho_height=1.05),
        "underbody": Camera.orbit(center, 4 * length, 40, -18, ortho=True,
                                  ortho_height=length * 0.74),
        "underbody_diagnostic": Camera.orbit(center, 4 * length, 40, -32, ortho=True,
                                             ortho_height=length * 0.74),
    }


def camera_dict(cam):
    return {"eye": cam.eye.tolist(), "target": cam.target.tolist(),
            "up": cam.up.tolist(), "fov_deg": cam.fov_deg,
            "ortho": cam.ortho, "ortho_height": cam.ortho_height}


def baseline_state():
    return {
        "version": 1, "size": [800, 600], "supersample": 2,
        "styles": {
            style: {**item, "config": config_for(style).raw,
                    "cameras": {view: camera_dict(cam) for view, cam in
                                cameras_for(item["bounds"], item["measurements"]).items()}}
            for style, item in BASELINE.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("before", "after"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--snapshots", type=Path,
                        help="trusted LOCAL directory containing STYLE.pickle (before only)")
    parser.add_argument("--styles", nargs="+", choices=STYLES, default=list(STYLES))
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    args = parser.parse_args()
    if args.snapshots and args.phase != "before":
        parser.error("--snapshots is before-only: after must use build_car on current code")
    state = (json.loads(args.state.read_text()) if args.state.exists()
             else baseline_state())
    if state.get("version") != 1:
        parser.error("unsupported camera state version")
    args.output.mkdir(parents=True, exist_ok=True)

    # Capture EVERY requested car before starting the slower PNG rasterisation.
    snapshots = {}
    for style in args.styles:
        if args.snapshots:
            with (args.snapshots / f"{style}.pickle").open("rb") as stream:
                data = pickle.load(stream)
        else:
            data = capture(style, state["styles"][style]["config"])
        snapshots[style] = data
        print(f"{args.phase} {style}: body={data['body_faces']:,} faces; "
              f"full={data['full_faces']:,} faces", flush=True)
        if args.phase == "before":
            bounds = [v.tolist() for v in data["full"].bounds()]
            state["styles"][style] = {
                "bounds": bounds, "measurements": data["measurements"],
                "config": data["config"], "body_faces": data["body_faces"],
                "full_faces": data["full_faces"],
                "cameras": {v: camera_dict(c) for v, c in
                            cameras_for(bounds, data["measurements"]).items()},
            }
    if args.phase == "before":
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(state, indent=2) + "\n")
    print(f"All {args.phase} meshes captured; fixed cameras: {args.state}", flush=True)
    renderer = Renderer(*state["size"], supersample=state["supersample"])
    for style, data in snapshots.items():
        item = state["styles"][style]
        # Older saved camera sets predate the diagnostic underside view.
        for view, cam in cameras_for(item["bounds"], item["measurements"]).items():
            item["cameras"].setdefault(view, camera_dict(cam))
        for view in args.views:
            cam = Camera(**item["cameras"][view])
            detail = view.endswith("closeup")
            render_mesh = data["detail"] if detail else data["full"]
            view_renderer = renderer
            if view == "underbody_diagnostic":
                render_mesh = render_mesh.copy()
                render_mesh.materials = {name: mat.with_color("#a5adb5") for name, mat in render_mesh.materials.items()}
                view_renderer = Renderer(*state["size"], supersample=state["supersample"])
                view_renderer.light_dir = np.array([0.3, 0.6, -1.0]) / np.linalg.norm([0.3, 0.6, -1.0])
                view_renderer.fill_dir = np.array([-0.4, -0.6, -0.7]) / np.linalg.norm([-0.4, -0.6, -0.7])
            image = view_renderer.render(render_mesh, cam,
                                         ground=not detail and not view.startswith("underbody"))
            path = args.output / f"{args.phase}_{style}_{view}.png"
            renderer.save(image, path)
            print(path, flush=True)


if __name__ == "__main__":
    main()
