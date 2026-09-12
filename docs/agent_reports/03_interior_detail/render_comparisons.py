#!/usr/bin/env python3
"""Render reproducible interior verification views (not production artwork).

    python3 docs/agent_reports/03_interior_detail/render_comparisons.py --phase before
    python3 docs/agent_reports/03_interior_detail/render_comparisons.py --phase after

BEFORE requires the immutable /tmp/interior_before_{sedan,suv}.pickle snapshots:
(BodyResult, {connector_name: world_mesh}, {name: connector}). AFTER builds the
current default assembly and needs NO snapshots. Both phases read cameras.json
beside this script: original root-derived cameras and belt cutoff, independent
of the current components' bounds. Snapshots are opened read-only, never saved.

Baseline geometry is commit f937856. If snapshots are lost, regenerate them in
an independent checkout of that commit using CarBody().build(style), then
assemble(body); pickle (body, {i.connector.name: i.result.mesh for i in
asm.instances}, {c.name: c for c in asm.connectors}). Never substitute current
geometry for BEFORE. To explicitly regenerate cameras.json from those snapshots
without rendering: run this script with --freeze-cameras [--snapshot-dir DIR].

PIL is used only to compose labelled verification contact sheets, not to alter
the renderer's individual images. Outputs are written beside this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from makecar.assembly import assemble
from makecar.body import CarBody
from makecar.export.render import Camera, Renderer
from makecar.geometry.mesh import Mesh

OUTPUT = Path(__file__).resolve().parent
CAMERA_FILE = OUTPUT / "cameras.json"
BASELINE_COMMIT = "f937856"
VIEWS = (
    "driver-eye", "passenger-eye", "roof-off-cutaway", "rear-seat",
    "seat-closeup", "cluster-closeup", "door-card-closeup",
)
HIDE_ROOF_TAGS = {
    "glass", "headliner", "antenna", "roof_rail", "rearview_mirror",
    "dome_light", "grab_handle", "mirror", "sun_visor",
}


def load_baseline(style: str, snapshot_dir: Path):
    # These are trusted, locally created snapshots; never unpickle downloads.
    path = snapshot_dir / f"interior_before_{style}.pickle"
    if not path.is_file():
        raise FileNotFoundError(f"Original BEFORE snapshot required ({BASELINE_COMMIT}): {path}")
    with path.open("rb") as stream:
        return pickle.load(stream)


def full_mesh(body, parts: dict[str, Mesh]) -> Mesh:
    mesh = body.mesh.copy()
    for part in parts.values():
        mesh.merge(part)
    return mesh


def below_belt(mesh: Mesh, z_cut: float) -> Mesh:
    """Drop any face with ANY vertex above the cut, rather than centroid clips."""
    z = mesh.vertices[:, 2]
    return mesh.subset([i for i, face in enumerate(mesh.faces)
                        if np.all(z[list(face)] < z_cut)])


def cutaway_mesh(body, parts, connectors, z_cut: float) -> Mesh:
    mesh = below_belt(body.mesh, z_cut)
    for name, part in parts.items():
        root_name = name.split("/", 1)[0]
        root = connectors[root_name]
        connector = connectors[name]
        tags = root.tags | connector.tags
        if tags & HIDE_ROOF_TAGS:
            continue
        # New A/B pillar trim is deliberately cut only in this roof-off view.
        if "pillar_trim" in tags or root_name.startswith("pillar_trim_"):
            part = below_belt(part, z_cut)
        mesh.merge(part)
    return mesh


def isolated_tree(parts: dict[str, Mesh], root: str) -> Mesh:
    """Parents do not contain recursive children: merge the entire named tree."""
    if root not in parts:
        raise KeyError(f"Missing close-up root component: {root}")
    mesh = Mesh(name=f"isolated_{root}")
    for name, part in parts.items():
        if name == root or name.startswith(root + "/"):
            mesh.merge(part)
    return mesh


def cameras(original_body, original_connectors) -> dict[str, Camera]:
    """Frozen packaging anchors, in metres (+X forward, +Y left, +Z up)."""
    result = {}
    for view, name in (("driver-eye", "seat_front_driver"),
                       ("passenger-eye", "seat_front_passenger")):
        seat = original_connectors[name]
        o = seat.origin
        hp = seat.meta.get("h_point_height", 0.27)
        # The original pipeline's driver camera, mirrored for the passenger.
        eye = [o[0] - 0.15, o[1], o[2] + hp + 0.64]
        target = [o[0] + 2.5, o[1] * 0.35, o[2] + hp + 0.42]
        result[view] = Camera(eye, target, fov_deg=78)

    driver = original_connectors["seat_front_driver"]
    rear = original_connectors["seat_row2"]
    o = driver.origin
    hp = driver.meta.get("h_point_height", 0.27)
    m = original_body.measurements
    # High rear-quarter view exposes the dashboard and cabin layout.
    center = np.array([(o[0] + rear.origin[0]) * 0.5, 0.0, m["z_belt"] - 0.18])
    result["roof-off-cutaway"] = Camera.orbit(
        center, m["length"] * 1.06, 142, 52, fov_deg=45)

    # Rearward from between the front seatbacks, showing rear bench/closure.
    rear_hp = rear.meta.get("h_point_height", 0.25)
    result["rear-seat"] = Camera(
        [o[0] - 0.50, 0.0, o[2] + hp + 0.66],
        [rear.origin[0] - 0.13, 0.0, rear.origin[2] + rear_hp + 0.32],
        fov_deg=92,
    )

    # Front-inboard three-quarter seat view; fixed original floor/H-point anchor.
    seat_target = o + np.array([-0.09, 0.0, hp + 0.21])
    result["seat-closeup"] = Camera(
        seat_target + [1.23, -1.05, 0.79], seat_target, fov_deg=46)

    dash = original_connectors["dashboard"]
    # Aim at the original root's driver gauge area, not the evolving child mesh.
    cluster_target = dash.origin + np.array([-0.18, dash.meta["driver_y"], 0.08])
    result["cluster-closeup"] = Camera(
        cluster_target + [-0.63, -0.035, 0.12], cluster_target, fov_deg=42)

    door = original_connectors["door_card_front_L"]
    # Inboard view, slightly above the armrest, wide enough for the entire card.
    door_target = door.origin + np.array([0.0, -0.025, -0.015])
    # Include the sill return down to the original carpet datum in BOTH phases.
    door_target[2] = (door.points[:, 2].max() + m["z_floor"] + 0.05) / 2
    result["door-card-closeup"] = Camera(
        door_target + [-0.12, -door.width * 1.22, door.height * 0.43],
        door_target, fov_deg=42)
    return result


def freeze_cameras(snapshot_dir: Path):
    """Explicit regeneration only; normal renders never change the camera file."""
    data = {"schema_version": 1, "baseline_commit": BASELINE_COMMIT, "styles": {}}
    for style in ("sedan", "suv"):
        body, _, connectors = load_baseline(style, snapshot_dir)
        data["styles"][style] = {
            "z_cut": body.measurements["z_belt"] + 0.06,
            "views": {
                name: {"eye": camera.eye.tolist(), "target": camera.target.tolist(),
                       "fov_deg": camera.fov_deg}
                for name, camera in cameras(body, connectors).items()
            },
        }
    CAMERA_FILE.write_text(json.dumps(data, indent=2) + "\n")
    print(CAMERA_FILE, flush=True)


def frozen_cameras(style: str) -> tuple[dict[str, Camera], float]:
    """Load durable original cameras and the cutaway plane, never /tmp data."""
    data = json.loads(CAMERA_FILE.read_text())
    if data["schema_version"] != 1 or data["baseline_commit"] != BASELINE_COMMIT:
        raise ValueError("Unexpected camera schema or baseline commit")
    saved = data["styles"][style]
    if set(saved["views"]) != set(VIEWS):
        raise ValueError(f"Frozen cameras must contain exactly the seven views: {style}")
    return {name: Camera(**saved["views"][name]) for name in VIEWS}, saved["z_cut"]


def contact_sheet(style: str, phase: str, image_paths: list[Path]) -> Path:
    """PIL layout only: captions clearly identify open and isolated views."""
    tile_w, tile_h, caption, pad = 480, 320, 30, 12
    cols, rows = 2, 4
    sheet = Image.new("RGB", (cols * (tile_w + pad) + pad,
                              rows * (tile_h + caption + pad) + pad), "#e6e9ed")
    draw = ImageDraw.Draw(sheet)
    for i, (view, path) in enumerate(zip(VIEWS, image_paths)):
        x = pad + (i % cols) * (tile_w + pad)
        y = pad + (i // cols) * (tile_h + caption + pad)
        label = f"{style.upper()} / {phase.upper()} / {view}"
        if "closeup" in view:
            label += " (isolated)"
        draw.text((x + 4, y + 8), label, fill="#20252a")
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS),
                        (x, y + caption))
    x = pad + tile_w + pad
    y = pad + 3 * (tile_h + caption + pad)
    draw.multiline_text(
        (x + 12, y + 46),
        f"{style.upper()} - {phase.upper()}\n\n"
        "Fixed original connector cameras.\n"
        "Eye and rear-seat views: complete closed cabin.\n"
        "Cutaway: roof removed at belt + 60 mm;\n"
        "pillar trims also cut at that height.\n"
        "Seat, cluster and door: isolated component trees.\n\n"
        "Renderer: 960 x 640; ground=False; lines=False.\n"
        "Individual PNGs are unaltered renderer outputs.",
        fill="#20252a", spacing=7,
    )
    path = OUTPUT / f"{style}_{phase}_contact-sheet.png"
    sheet.save(path)
    return path


def render_style(style: str, phase: str, snapshot_dir: Path, supersample: int):
    views, z_cut = frozen_cameras(style)
    if phase == "before":
        body, parts, connectors = load_baseline(style, snapshot_dir)
    else:
        body = CarBody().build(style)
        assembly = assemble(body)
        parts = {i.connector.name: i.result.mesh for i in assembly.instances}
        connectors = {c.name: c for c in body.connectors}
        connectors.update({i.connector.name: i.connector for i in assembly.instances})

    full = full_mesh(body, parts)
    cutaway = cutaway_mesh(body, parts, connectors, z_cut)
    subjects = {
        "driver-eye": full,
        "passenger-eye": full,
        "roof-off-cutaway": cutaway,
        "rear-seat": full,
        "seat-closeup": isolated_tree(parts, "seat_front_driver"),
        # Dashboard root plus children preserves the real cluster's occlusion by
        # its fascia/hood without the steering wheel hiding the gauge details.
        "cluster-closeup": isolated_tree(parts, "dashboard"),
        "door-card-closeup": isolated_tree(parts, "door_card_front_L"),
    }
    renderer = Renderer(960, 640, supersample=supersample)
    image_paths = []
    for view in VIEWS:
        path = OUTPUT / f"{style}_{phase}_{view}.png"
        image = renderer.render(subjects[view], views[view], ground=False, lines=False)
        renderer.save(image, path)
        image_paths.append(path)
        print(path, flush=True)
    print(contact_sheet(style, phase, image_paths), flush=True)


def render_extras(supersample: int):
    """After-only inspections of details hidden in the matched eye views."""
    renderer = Renderer(960, 640, supersample=supersample)
    sports = assemble(CarBody().build("sports"), {"assign": {"seat_front": "seat.sport"}})
    parts = {i.connector.name: i.result.mesh for i in sports.instances}
    seat = sports.body.connector("seat_front_driver")
    target = seat.origin + [-0.13, 0, seat.meta["h_point_height"] + 0.32]
    subjects = [("sports_after_integrated-seat", isolated_tree(parts, "seat_front_driver"),
                 Camera(target + [1.25, -1.05, 0.73], target, fov_deg=46))]
    wheel = sports.body.connector("steering_wheel")
    subjects.append(("sports_after_flat-wheel", parts["steering_wheel"],
                     Camera(wheel.origin + wheel.normal * 0.65 - wheel.frame.y_axis * 0.06, wheel.origin, fov_deg=44)))
    suv = assemble(CarBody().build("suv"))
    parts = {i.connector.name: i.result.mesh for i in suv.instances}
    ceiling = isolated_tree(parts, "headliner")
    lo, hi = ceiling.bounds()
    subjects.append(("suv_after_ceiling", ceiling,
                     Camera.orbit((lo + hi) / 2, 3.8, 160, -60, fov_deg=48)))
    cargo = Mesh(name="cargo_inspection")
    for name in ("cargo_wheelhouse_L", "cargo_wheelhouse_R", "trunk_floor"):
        cargo.merge(parts[name])
    floor = parts["floor"]
    cargo.merge(floor.subset([i for i, f in enumerate(floor.faces)
                             if np.all(floor.vertices[list(f), 0] < suv.body.measurements["wheel_rear_x"] + 0.50)]))
    target = np.array([suv.body.measurements["wheel_rear_x"] - 0.20, 0, suv.body.measurements["z_floor"] + 0.30])
    subjects.append(("suv_after_cargo-liners", cargo, Camera.orbit(target, 2.8, 145, 35, fov_deg=45)))
    for name, mesh, camera in subjects:
        path = OUTPUT / f"{name}.png"
        renderer.save(renderer.render(mesh, camera, ground=False, lines=False), path)
        print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after"), default="after")
    parser.add_argument("--styles", nargs="+", choices=("sedan", "suv"), default=["sedan", "suv"])
    parser.add_argument("--snapshot-dir", type=Path, default=Path("/tmp"),
                        help="BEFORE/freeze only; unused for AFTER")
    parser.add_argument("--freeze-cameras", action="store_true",
                        help="write both styles' cameras.json from original snapshots, then exit without rendering")
    parser.add_argument("--supersample", type=int, choices=(1, 2), default=2)
    parser.add_argument("--extras", action="store_true", help="render only four supplemental AFTER detail inspections")
    args = parser.parse_args()
    if args.extras:
        render_extras(args.supersample)
        return
    if args.freeze_cameras:
        freeze_cameras(args.snapshot_dir)
        return
    for style in args.styles:
        render_style(style, args.phase, args.snapshot_dir, args.supersample)


if __name__ == "__main__":
    main()
