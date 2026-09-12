#!/usr/bin/env python3
"""Frozen-mesh exterior inspection with makecar's numpy Renderer (not charts).

Run from any directory, using only Python's standard library + numpy:
  python3 docs/agent_reports/02_exterior_detail/render_exterior.py before
  python3 docs/agent_reports/02_exterior_detail/render_exterior.py after
  python3 docs/agent_reports/02_exterior_detail/render_exterior.py after --reuse
  python3 docs/agent_reports/02_exterior_detail/render_exterior.py before --only wheel
  python3 docs/agent_reports/02_exterior_detail/render_exterior.py after --extras
  python3 docs/agent_reports/02_exterior_detail/render_exterior.py before --reframe mirror --only mirror

--extras adds AFTER-only steel-wheel/grille variants, lamp normal views and
furniture closeups. --reframe locks requested cameras to the union of the two
captured meshes; rerender both stages afterwards to retain matched comparisons.

BEFORE never overwrites/rebuilds an existing snapshot. AFTER builds fresh unless
--reuse is supplied. Both styles are captured BEFORE any PNG rendering begins.
--capture-only saves meshes and exits, allowing production editing immediately.
Camera positions, framing, dimensions and lighting are locked in cameras.json
from BEFORE; AFTER uses those exact settings. Defaults: canonical sedan/sports,
seed 0, all default components, the same blue paint, 960x720, 2x supersampling.
Snapshots are trusted, locally generated pickle files: never load untrusted ones.
All script-controlled writes (including the body cache) stay beside this script.
Lens-free views remove transparent lens faces only; geometry is never moved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
os.environ["MAKECAR_CACHE"] = str(REPORT / ".cache")

import numpy as np
from makecar.export.render import Camera, Renderer
from makecar.geometry.mesh import Mesh

STYLES = ("sedan", "sports")


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "makecar").rglob("*.py"))}


def capture(stage):
    # Import/build everything before returning; rendering never rebuilds parts.
    from makecar.body import CarBody
    from makecar.assembly import assemble
    from makecar.components.base import Palette

    source = source_hashes()
    body, palette = CarBody(), Palette(paint="#426b88")
    data = {"stage": stage,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_sha256": source, "styles": {}}
    for style in STYLES:
        asm = assemble(body.build(style=style, paint=palette.paint), palette=palette, seed=0)
        parts = {}
        for inst in asm.instances:
            c = inst.connector
            parts[c.name] = {"mesh": inst.result.mesh.copy(), "component": inst.component,
                             "tags": sorted(c.tags), "origin": c.origin.copy(),
                             "frame": c.frame.matrix.copy(), "options": dict(inst.options)}
        data["styles"][style] = {"full": asm.mesh(), "body": asm.body.mesh.copy(),
                                 "parts": parts, "measurements": dict(asm.body.measurements),
                                 "hints": dict(asm.body.hints)}
        print(f"{style}: captured {data['styles'][style]['full'].n_faces:,} faces", flush=True)
    if source_hashes() != source:
        raise RuntimeError("Production source changed during capture; refusing a mixed snapshot")
    path = REPORT / f"{stage}_meshes.pkl"
    if stage == "before":
        with path.open("xb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        temp = path.with_suffix(".pkl.tmp")
        with temp.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(path)
    print(f"BOTH {stage.upper()} BASELINE/ASSEMBLY MESHES CAPTURED: {path}", flush=True)
    return data


def load_snapshot(path):
    with path.open("rb") as f:
        return pickle.load(f)  # Only the script's own local snapshots are trusted.


def selected(car, key):
    if key == "full":
        return car["full"]
    if key == "underside_furniture":
        mesh = Mesh(name=key)
        for name, part in car["parts"].items():
            tags = set(part["tags"])
            if tags & {"exhaust", "exhaust_system", "diffuser", "mud_flap", "mudflap", "tow_eye"} or any(
                    word in name for word in ("exhaust", "diffuser", "mudflap", "mud_flap", "tow_eye")):
                mesh.merge(part["mesh"])
        return mesh
    return car["parts"][key]["mesh"]


def view_definitions():
    # name, rendered selection, framing selection, azimuth, elevation, margin, lens-free
    return [
        ("whole_front_three_quarter", "full", "full", 35, 18, 1.14, False),
        ("whole_rear_three_quarter", "full", "full", 145, 20, 1.14, False),
        ("wheel_01_outboard", "wheel_front_L", "wheel_front_L", 90, 0, 1.28, False),
        ("wheel_02_oblique", "wheel_front_L", "wheel_front_L", 48, 18, 1.28, False),
        ("wheel_03_tread", "wheel_front_L", "wheel_front_L", 12, 12, 1.28, False),
        ("wheel_context", "full", "wheel_front_L", 65, 10, 1.7, False),
        ("headlight_isolated", "headlight_L", "headlight_L", 35, 22, 1.35, False),
        ("headlight_lens_removed", "headlight_L", "headlight_L", 35, 22, 1.35, True),
        ("headlight_context", "full", "headlight_L", 35, 22, 1.8, False),
        ("taillight_isolated", "taillight_L", "taillight_L", 145, 18, 1.35, False),
        ("taillight_lens_removed", "taillight_L", "taillight_L", 145, 18, 1.35, True),
        ("taillight_context", "full", "taillight_L", 145, 18, 1.8, False),
        ("mirror_housing_isolated", "mirror_L", "mirror_L", 45, 20, 1.45, False),
        ("mirror_glass_isolated", "mirror_L", "mirror_L", 145, 15, 1.45, False),
        ("mirror_context", "full", "mirror_L", 65, 15, 3.0, False),
        ("grille_isolated", "grille", "grille", 12, 12, 1.35, False),
        ("grille_context", "full", "grille", 8, 5, 1.8, False),
        ("underside_oblique", "full", "full", 145, -38, 1.14, False),
        ("underside_bottom", "full", "full", 90, -89.9, 1.14, False),
        ("underside_furniture_isolated", "underside_furniture", "full", 145, -38, 1.14, False),
        ("exhaust_tip_isolated", "exhaust_L", "exhaust_L", 160, -18, 1.7, False),
    ]


def fitted_camera(mesh, azimuth, elevation, margin, width, height):
    target = np.mean(mesh.bounds(), axis=0)
    cam = Camera.orbit(target, 1.0, azimuth, elevation, fov_deg=30)
    rotation = cam.view_matrix()[:3, :3]
    local = (mesh.vertices - target) @ rotation.T
    tangent = np.tan(np.radians(cam.fov_deg / 2))
    demand = np.maximum(np.abs(local[:, 0]) / (tangent * width / height),
                        np.abs(local[:, 1]) / tangent)
    distance = max(float(np.max(local[:, 2] + demand * margin)), 0.4)
    cam.eye = target + (cam.eye - target) * distance
    return {"eye": cam.eye.tolist(), "target": target.tolist(), "up": cam.up.tolist(),
            "fov_deg": cam.fov_deg}


def make_cameras(baseline, width, height, supersample):
    settings = {"width": width, "height": height, "supersample": supersample,
                "background": [[0.93, 0.95, 0.98], [0.78, 0.82, 0.87]], "styles": {}}
    for style in STYLES:
        settings["styles"][style] = {}
        car = baseline["styles"][style]
        for name, key, focus, az, el, margin, lens_free in view_definitions():
            settings["styles"][style][name] = {
                "selection": key, "focus": focus, "lens_removed": lens_free,
                "camera": fitted_camera(selected(car, focus), az, el, margin, width, height),
                "ground": name.startswith("whole_"), "lines": name.startswith("whole_"),
            }
    return settings


def render(data, settings, only, supplemental=False):
    out = REPORT / data["stage"]
    if supplemental:
        out = out / "extras"
    out.mkdir(exist_ok=True)
    manifest = {"stage": data["stage"], "captured_at": data["captured_at"],
                "source_sha256": data["source_sha256"], "styles": {}, "images": []}
    for style in STYLES:
        car = data["styles"][style]
        wheel_faces = sum(p["mesh"].n_faces for p in car["parts"].values() if "wheel" in p["tags"])
        manifest["styles"][style] = {"full": car["full"].stats(), "wheel_faces": wheel_faces,
                                      "components": {n: {"component": p["component"], "faces": p["mesh"].n_faces}
                                                     for n, p in car["parts"].items()}}
        folder = out / style
        folder.mkdir(exist_ok=True)
        for name, spec in settings["styles"][style].items():
            if only and not any(token in name for token in only):
                continue
            mesh = selected(car, spec["selection"])
            removed = 0
            if spec["lens_removed"]:
                keep = [i for i, material in enumerate(mesh.face_materials)
                        if not ("lens" in material and mesh.materials[material].alpha < 0.999)]
                removed = mesh.n_faces - len(keep)
                mesh = mesh.subset(keep)
            if not mesh.n_faces:
                raise RuntimeError(f"Empty selection: {style}/{name}")
            cam = Camera(**spec["camera"])
            renderer = Renderer(settings["width"], settings["height"], settings["supersample"],
                                background=settings["background"])
            # Keep lighting identical for before/after; illuminate downward faces
            # for inspection rather than leaving the entire underside in ambient.
            if name.startswith("underside") or name.startswith("exhaust"):
                renderer.light_dir = renderer._unit(np.array([-0.5, 0.6, -1.0]))
                renderer.fill_dir = renderer._unit(np.array([0.7, -0.4, -0.4]))
            begin = time.monotonic()
            image = renderer.render(mesh, cam, ground=spec["ground"], lines=spec["lines"])
            if not np.isfinite(image).all():
                raise RuntimeError(f"Non-finite pixels: {style}/{name}")
            path = folder / f"{name}.png"
            renderer.save(image, path)
            manifest["images"].append({"path": str(path.relative_to(REPORT)), **spec,
                                       "faces": mesh.n_faces, "lens_faces_removed": removed,
                                       "seconds": round(time.monotonic() - begin, 3)})
            print(f"{data['stage']}/{style}/{name}.png ({mesh.n_faces:,} faces)", flush=True)
    suffix = "_" + "_".join(only) if only else ""
    # Filename tokens come from CLI, so sanitize to keep all writes in REPORT.
    suffix = "".join(c if c.isalnum() or c in "_-" else "_" for c in suffix)
    (out / f"manifest{suffix}.json").write_text(json.dumps(manifest, indent=2) + "\n")


def supplemental_views(data, settings):
    """Optional components and inspection angles, independently framed/labeled.

    These are AFTER-only supplements, not a replacement for matched main views.
    The source must match the frozen main assembly to prevent stale variants.
    """
    if source_hashes() != data["source_sha256"]:
        raise RuntimeError("Source changed since main capture; recapture AFTER before --extras")
    from makecar.body import CarBody
    from makecar.components.base import Palette, BuildContext, get_component
    palette, body = Palette(paint="#426b88"), CarBody()
    extra = {**data, "styles": {}}
    extra_settings = {**settings, "styles": {}}
    for style in STYLES:
        old = data["styles"][style]
        car = {**old, "parts": dict(old["parts"])}
        res = body.build(style=style, paint=palette.paint)
        ctx = BuildContext(res.measurements, res.hints, palette, np.random.default_rng(0))
        for key, component, connector in (("steel_wheel", "wheel.steel", "wheel_front_L"),
                                           ("honeycomb_grille", "grille.honeycomb", "grille"),
                                           ("mesh_grille", "grille.mesh", "grille")):
            conn = res.connector(connector)
            mesh = get_component(component).build(conn, {}, ctx).mesh
            car["parts"][key] = {"mesh": mesh, "component": component, "tags": [],
                                  "origin": conn.origin.copy(), "frame": conn.frame.matrix.copy(), "options": {}}
        extra["styles"][style] = car
        definitions = [
            ("steel_wheel_outboard", "steel_wheel", "steel_wheel", 90, 0, 1.25, False),
            ("steel_wheel_oblique", "steel_wheel", "steel_wheel", 48, 18, 1.25, False),
            ("steel_wheel_tread", "steel_wheel", "steel_wheel", 12, 12, 1.25, False),
            ("grille_honeycomb", "honeycomb_grille", "honeycomb_grille", 12, 18, 1.25, False),
            ("grille_mesh", "mesh_grille", "mesh_grille", 12, 18, 1.25, False),
            ("mirror_signal_front", "mirror_L", "mirror_L", 0, 0, 1.25, False),
            ("mirror_signal_side", "mirror_L", "mirror_L", 90, 0, 1.25, False),
            ("mirror_signal_below", "mirror_L", "mirror_L", 40, -18, 1.25, False),
            ("fog_isolated", "fog_front_L", "fog_front_L", 10, 10, 1.3, False),
            ("fog_context", "full", "fog_front_L", 15, 10, 3.0, False),
            ("wipers_isolated", "wipers", "wipers", 25, 48, 1.3, False),
            ("wipers_context", "full", "wipers", 25, 48, 1.5, False),
            ("rear_furniture_context", "full", "plate_rear", 170, 5, 3.5, False),
            ("plate_lamp_isolated", "plate_lamp_L", "plate_lamp_L", 170, -15, 1.35, False),
            ("handle_isolated", "handle_front_L", "handle_front_L", 65, 12, 1.3, False),
            ("rear_reflector_isolated", "rear_reflector_L", "rear_reflector_L", 170, 8, 1.3, False),
        ]
        if "diffuser" in car["parts"]:
            definitions.append(("underside_diffuser_isolated", "diffuser", "diffuser", 145, -35, 1.25, False))
        for key in ("headlight_L", "taillight_L"):
            n = car["parts"][key]["frame"][:3, 2]
            az, el = np.degrees(np.arctan2(n[1], n[0])), np.degrees(np.arcsin(n[2]))
            for lens_removed in (False, True):
                name = key.split("_")[0] + "_normal" + ("_lens_removed" if lens_removed else "")
                definitions.append((name, key, key, az, el, 1.3, lens_removed))
        extra_settings["styles"][style] = {}
        for name, key, focus, az, el, margin, lens_free in definitions:
            extra_settings["styles"][style][name] = {
                "selection": key, "focus": focus, "lens_removed": lens_free,
                "camera": fitted_camera(selected(car, focus), az, el, margin, settings["width"], settings["height"]),
                "ground": False, "lines": False,
            }
    if source_hashes() != data["source_sha256"]:
        raise RuntimeError("Source changed during supplemental capture; refusing mixed variants")
    with (REPORT / "after_extras_meshes.pkl").open("wb") as f:
        pickle.dump(extra, f, protocol=pickle.HIGHEST_PROTOCOL)
    render(extra, extra_settings, None, supplemental=True)


def reframe_matches(settings, terms):
    before = load_snapshot(REPORT / "before_meshes.pkl")
    after = load_snapshot(REPORT / "after_meshes.pkl")
    definitions = {entry[0]: entry for entry in view_definitions()}
    for style in STYLES:
        for name, spec in settings["styles"][style].items():
            if not any(term in name for term in terms):
                continue
            focus = spec["focus"]
            combined = selected(before["styles"][style], focus).copy()
            combined.merge(selected(after["styles"][style], focus))
            _, _, _, az, el, margin, _ = definitions[name]
            spec["camera"] = fitted_camera(combined, az, el, margin, settings["width"], settings["height"])
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("before", "after"))
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--reuse", action="store_true", help="render an already captured AFTER snapshot")
    parser.add_argument("--only", nargs="+", help="render view names containing any of these strings")
    parser.add_argument("--reframe", nargs="+", help="reframe matching cameras on BEFORE/AFTER union; rerender BOTH stages afterwards")
    parser.add_argument("--extras", action="store_true", help="also capture/render AFTER-only variants and furniture inspection views")
    parser.add_argument("--width", type=int, default=960, help="used only when first creating cameras.json")
    parser.add_argument("--height", type=int, default=720, help="used only when first creating cameras.json")
    parser.add_argument("--supersample", type=int, default=2, help="used only when first creating cameras.json")
    args = parser.parse_args()
    if min(args.width, args.height, args.supersample) <= 0:
        parser.error("Image dimensions and supersample must be positive")
    baseline_path = REPORT / "before_meshes.pkl"
    if args.stage == "after" and not baseline_path.exists():
        parser.error("Capture BEFORE before building/rendering AFTER")
    path = REPORT / f"{args.stage}_meshes.pkl"
    if path.exists() and (args.stage == "before" or args.reuse):
        data = load_snapshot(path)
        print(f"Using frozen snapshot: {path}", flush=True)
    elif args.reuse:
        parser.error(f"Snapshot not found: {path}")
    else:
        data = capture(args.stage)
    if args.capture_only:
        return
    cameras_path = REPORT / "cameras.json"
    if cameras_path.exists():
        settings = json.loads(cameras_path.read_text())
    else:
        baseline = data if args.stage == "before" else load_snapshot(baseline_path)
        settings = make_cameras(baseline, args.width, args.height, args.supersample)
        cameras_path.write_text(json.dumps(settings, indent=2) + "\n")
    if args.reframe:
        if not (REPORT / "after_meshes.pkl").exists():
            parser.error("--reframe requires both frozen snapshots")
        settings = reframe_matches(settings, args.reframe)
        cameras_path.write_text(json.dumps(settings, indent=2) + "\n")
    render(data, settings, args.only)
    if args.extras:
        if args.stage != "after":
            parser.error("--extras is for AFTER-only optional components")
        supplemental_views(data, settings)


if __name__ == "__main__":
    main()
