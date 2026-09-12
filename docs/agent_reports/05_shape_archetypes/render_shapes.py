#!/usr/bin/env python3
"""Verification-only car renders, never runtime geometry or data charts.

  python3 docs/agent_reports/05_shape_archetypes/render_shapes.py before
  python3 docs/agent_reports/05_shape_archetypes/render_shapes.py after
  python3 docs/agent_reports/05_shape_archetypes/render_shapes.py sweeps
  python3 docs/agent_reports/05_shape_archetypes/render_shapes.py after \
      --styles sedan sports --views side front plan nose --quick --output /tmp/shape-check
  python3 docs/agent_reports/05_shape_archetypes/render_shapes.py sweeps \
      --targets face/wedge plan/pointed --variants clay --quick --output /tmp/shape-check

Before automatically imports an archive of BASELINE_COMMIT, NOT the edited tree.
--source can instead point to an already archived source tree. After/sweeps use
current worktree code. Every car is built by build_car, with all default parts.
Full renders contain the entire assembly. Clay renders contain the actual shell
plus assembled glass ONLY, recolored with opaque glass: no grille, lamps, plates,
wipers, interior or mirrors hiding the body. Holes in the shell are not filled.

All cameras are orthographic, frozen in baseline.json before rasterization and
reused for after and sedan sweeps (never refitted to the changed geometry). Side
and plan put the nose at image right. Per-style framing preserves baseline scale
within a before/after pair; different styles are NOT shown at a common scale.
--quick halves cell resolution and disables supersampling; it defaults to a
separate quick/ folder so it cannot replace final baseline or comparison images.
Pillow is used ONLY here for PNG contact sheets and labels. No runtime dependency.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
BASELINE_COMMIT = "d7122f722651333e7a4e9cfbb9fafb82a867b2a0"
STYLES = ("sedan", "hatchback", "wagon", "suv", "pickup", "coupe", "sports", "van")
VIEWS = ("side", "front", "plan", "three_quarter_front", "nose")
TARGETS = tuple("face/" + x for x in ("wedge", "upright", "shark", "snub", "cabforward", "longhood")) + tuple(
    "plan/" + x for x in ("pointed", "square", "cokebottle")) + tuple(
    "section/" + x for x in ("tumblehome", "slabside", "domed"))
BLENDS = (("face/wedge", "face/upright"), ("face/longhood", "face/shark"))
SIZE = (480, 320)
PAINT = "#91acc1"
BACKGROUND = ((0.94, 0.95, 0.96), (0.87, 0.89, 0.91))


def source_fingerprint(source):
    digest = hashlib.sha256()
    for path in sorted((source / "makecar").rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_pipeline(args):
    # Delay makecar imports until the baseline/current implementation is selected.
    temporary = None
    if args.source:
        source = args.source.resolve()
    elif args.phase == "before":
        temporary = tempfile.TemporaryDirectory(prefix="shape-baseline-")
        source = Path(temporary.name)
        archive = subprocess.check_output(["git", "-C", str(ROOT), "archive", BASELINE_COMMIT])
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter="data")
    else:
        source = ROOT
    if not (source / "makecar" / "pipeline.py").is_file():
        raise ValueError(f"No makecar pipeline in {source}")
    fingerprint = source_fingerprint(source)
    # Isolate target caches by complete source hash, including uncommitted edits.
    os.environ["MAKECAR_CACHE"] = str(Path(tempfile.gettempdir()) / "shape-render-cache" / fingerprint[:16])
    sys.path.insert(0, str(source))
    global CarConfig, Camera, Renderer, Material, build_car, car_body_for
    from makecar.config import CarConfig
    from makecar.export.render import Camera, Renderer
    from makecar.geometry.mesh import Material
    from makecar.pipeline import build_car, car_body_for
    return source, fingerprint, temporary


def config_for(style, modifiers=None):
    return {"name": style, "seed": 0,
            "body": {"style": style, "modifiers": modifiers or {}},
            "palette": {"paint": PAINT}, "components": {"defaults": True}}


def capture(config):
    cfg = CarConfig.from_dict(config)
    requested = cfg.body.get("modifiers") or {}
    unknown = set(requested) - set(car_body_for(cfg).library.modifiers)
    if unknown:
        raise ValueError(f"Targets not implemented in selected source: {sorted(unknown)}")
    body, assembly, _ = build_car(cfg)
    full = assembly.mesh()
    clay = body.mesh.copy()
    clay.set_material("diagnostic_shell")
    clay.materials = {"diagnostic_shell": Material("diagnostic_shell", (0.69, 0.72, 0.74), shininess=0.15)}
    for instance in assembly.instances:
        if "glass" in instance.connector.tags:
            glass = instance.result.mesh.copy()
            glass.set_material("diagnostic_glass")
            glass.materials = {"diagnostic_glass": Material("diagnostic_glass", (0.22, 0.29, 0.34), shininess=0.12)}
            clay.merge(glass)
    metadata = {"config": config, "measurements": body.measurements,
                "effective_modifiers": body.modifier_values,
                "bounds": [x.tolist() for x in full.bounds()],
                "body_faces": body.mesh.n_faces, "assembly_faces": full.n_faces,
                "clay_faces": clay.n_faces, "components": len(assembly.instances)}
    return {"full": full, "clay": clay, "metadata": metadata}


def camera_dict(camera):
    result = asdict(camera)
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}


def cameras_for(metadata):
    lo, hi = np.asarray(metadata["bounds"])
    center = (lo + hi) / 2
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    cameras = {}
    for view, azimuth, elevation in (("side", -90, 0), ("front", 0, 0), ("three_quarter_front", -35, 20)):
        camera = Camera.orbit(center, 25, azimuth, elevation, ortho=True)
        projected = (corners - center) @ camera.view_matrix()[:3, :3].T
        extent = np.ptp(projected, axis=0)
        camera.ortho_height = max(extent[1], extent[0] / (SIZE[0] / SIZE[1])) * 1.24
        cameras[view] = camera_dict(camera)
    # Exact top view (not 89.9 degrees); world +X points right on both side/plan.
    camera = Camera(center + [0, 0, 25], center, up=[0, 1, 0], ortho=True,
                    ortho_height=max(hi[1] - lo[1], (hi[0] - lo[0]) / (SIZE[0] / SIZE[1])) * 1.24)
    cameras["plan"] = camera_dict(camera)
    m = metadata["measurements"]
    nose = [m["x_front"] - 0.30, 0, (m["nose_z_top"] + m["nose_z_bottom"]) / 2 + 0.17]
    camera = Camera.orbit(nose, 25, -28, 20, ortho=True,
                          ortho_height=max(1.65, (hi[1] - lo[1]) * 0.82))
    cameras["nose"] = camera_dict(camera)
    return cameras


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n")


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def sheet(path, title, subtitle, cells, columns, size):
    """Contact sheet of labelled car renders; None pads a partial last row."""
    width, height = size
    gap, label, header = 8, 28, 70
    rows = (len(cells) + columns - 1) // columns
    result = Image.new("RGB", (columns * (width + gap) + gap,
                               rows * (height + label + gap) + header + gap), "#f5f6f7")
    draw = ImageDraw.Draw(result)
    draw.text((12, 9), title, fill="#17212a", font=font(21))
    draw.text((12, 39), subtitle, fill="#44525f", font=font(13))
    for index, cell in enumerate(cells):
        if cell is None:
            continue
        image, text = cell
        x = gap + (index % columns) * (width + gap)
        y = header + (index // columns) * (height + label + gap)
        draw.text((x + 5, y + 4), text, fill="#17212a", font=font(15))
        if isinstance(image, Path):
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        if image.size != size:
            raise ValueError(f"Contact-sheet cell size mismatch: {image.size} != {size}")
        result.paste(image, (x, y + label))
    result.save(path, optimize=True)


def render(data, variant, camera, size, supersample):
    renderer = Renderer(*size, supersample=supersample, background=BACKGROUND)
    # Light the nose itself; identical light directions in all phases/variants.
    renderer.light_dir = renderer._unit(np.array([0.6, -0.7, 1.0]))
    renderer.fill_dir = renderer._unit(np.array([-0.4, 0.5, 0.6]))
    pixels = renderer.render(data[variant], Camera(**camera), ground=False, lines=True)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def cell_path(output, phase, style, variant, view):
    return output / "cells" / f"{phase}_{style}_{variant}_{view}.png"


def styles_run(args, state, fingerprint, size, supersample):
    metadata = {}
    # Build every requested style and persist exact cameras before rasterization.
    cars = {}
    for style in args.styles:
        if args.phase == "after" and style not in state["styles"]:
            raise ValueError(f"Capture the {style} baseline before rendering after")
        config = state["styles"].get(style, {}).get("config", config_for(style))
        cars[style] = capture(config)
        metadata[style] = cars[style]["metadata"]
        if args.phase == "before":
            old = state["styles"].get(style)
            item = {**metadata[style], "cameras": cameras_for(metadata[style])}
            if old and old["cameras"] != item["cameras"]:
                raise ValueError("Refusing to change existing baseline cameras")
            state["styles"][style] = item
        print(f"Built {style}: {metadata[style]['assembly_faces']} full-assembly faces", flush=True)
    if args.phase == "before":
        state["source_sha256"] = fingerprint
        write_json(args.state, state)
        print(f"Baseline cameras/configs saved safely: {args.state}", flush=True)
    write_json(args.output / f"{args.phase}_manifest.json", {
        "phase": args.phase, "source_sha256": fingerprint, "size": size,
        "supersample": supersample, "views": args.views, "variants": args.variants,
        "styles": metadata, "cameras": {s: state["styles"][s]["cameras"] for s in args.styles}})
    for style, data in cars.items():
        for variant in args.variants:
            for view in args.views:
                path = cell_path(args.output, args.phase, style, variant, view)
                render(data, variant, state["styles"][style]["cameras"][view], size, supersample).save(path)
                print(path.name, flush=True)
    for variant in args.variants:
        subtitle = ("Actual complete build_car assembly | orthographic | matched before/after cameras" if variant == "full" else
                    "Actual shell + opaque assembled glass | no grille/lamps/accessories | matched cameras")
        for view in args.views:
            cells = [(cell_path(args.output, args.phase, style, variant, view), style) for style in args.styles]
            sheet(args.output / f"{args.phase}_{variant}_{view}.png", f"{args.phase.upper()} | {variant} | {view}", subtitle,
                  cells, min(4, len(cells)), size)
            if args.phase == "after":
                pairs = []
                for style in args.styles:
                    before = cell_path(args.output, "before", style, variant, view)
                    if not before.exists():
                        before = cell_path(REPORT, "before", style, variant, view)
                    if not before.exists():
                        continue
                    with Image.open(before) as image:
                        if image.size != size:
                            continue  # never upscale quick cells into baseline comparisons
                    pairs += [(before, f"{style} | BEFORE"),
                              (cell_path(args.output, "after", style, variant, view), f"{style} | AFTER")]
                if pairs:
                    sheet(args.output / f"compare_{variant}_{view}.png", f"BEFORE / AFTER | {variant} | {view}", subtitle,
                          pairs, min(4, len(pairs)), size)


def sweep_cases(target):
    return [(f"{weight:g}", {target: weight}) for weight in (0.0, 0.5, 1.0)]


def blend_cases(first, second):
    # Include over-sum input explicitly, proving family normalization at 1 + 1.
    return [("1 + 0", {first: 1.0, second: 0.0}),
            ("0.5 + 0.5", {first: 0.5, second: 0.5}),
            ("0 + 1", {first: 0.0, second: 1.0}),
            ("1 + 1 (normalized)", {first: 1.0, second: 1.0})]


def sweeps_run(args, state, fingerprint, size, supersample):
    cameras = state["styles"]["sedan"]["cameras"]
    groups = [(target.replace("/", "_"), target, sweep_cases(target)) for target in args.targets]
    if not args.no_blends:
        groups += [("blend_" + a.split("/")[1] + "_" + b.split("/")[1], f"{a} + {b}", blend_cases(a, b)) for a, b in BLENDS]
    manifest = {"source_sha256": fingerprint, "size": size, "supersample": supersample,
                "cameras": cameras, "views": args.views, "variants": args.variants, "groups": {}}
    for name, title, cases in groups:
        cars = [(label, capture(config_for("sedan", modifiers))) for label, modifiers in cases]
        manifest["groups"][name] = [{"label": label, **data["metadata"]} for label, data in cars]
        for variant in args.variants:
            cells = []
            for view in args.views:
                for label, data in cars:
                    image = render(data, variant, cameras[view], size, supersample)
                    cells.append((image, f"{view.replace('three_quarter_front', 'front 3/4')} | {label}"))
            path = args.output / f"sweep_{name}_{variant}.png"
            sheet(path, f"SEDAN | {title} | {variant}",
                  "Same sedan baseline camera across each row | inputs shown; effective values in sweeps_manifest.json",
                  cells, len(cases), size)
            print(path.name, flush=True)
        write_json(args.output / "sweeps_manifest.json", manifest)


def self_test():
    assert len(STYLES) == 8 and len(TARGETS) == len(set(TARGETS)) == 12
    assert len(BLENDS) == 2
    for target in TARGETS:
        assert [m[target] for _, m in sweep_cases(target)] == [0, 0.5, 1]
    for a, b in BLENDS:
        assert [sum(m.values()) for _, m in blend_cases(a, b)] == [1, 1, 1, 2]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sheet.png"
        sheet(path, "test", "test", [(Image.new("RGB", SIZE), str(i)) for i in range(8)], 4, SIZE)
        with Image.open(path) as image:
            assert image.size == (1960, 790)
    print("Verification-script checks passed: 8 styles, 12 three-point sweeps, 2 normalized blends, sheet dimensions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("before", "after", "sweeps", "self-test"))
    parser.add_argument("--source", type=Path, help="Import makecar from this source tree (default before: archived baseline commit)")
    parser.add_argument("--output", type=Path, help="PNG/manifest directory; default this report (quick: report/quick)")
    parser.add_argument("--state", type=Path, default=REPORT / "baseline.json", help="Persistent baseline cameras and configs")
    parser.add_argument("--styles", nargs="+", choices=STYLES, default=list(STYLES))
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS[:-1]))
    parser.add_argument("--nose", action="store_true", help="Also render a close-up of the front body")
    parser.add_argument("--variants", nargs="+", choices=("full", "clay"), default=["full", "clay"])
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--no-blends", action="store_true", help="Only individual archetype strips in sweeps mode")
    parser.add_argument("--quick", action="store_true", help="240x160 cells, 1x sampling, separate output directory")
    args = parser.parse_args()
    if args.phase == "self-test":
        self_test()
        return
    args.output = args.output or (REPORT / "quick" if args.quick else REPORT)
    if args.nose and "nose" not in args.views:
        args.views.append("nose")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cells").mkdir(exist_ok=True)
    state = json.loads(args.state.read_text()) if args.state.exists() else {
        "version": 1, "baseline_commit": BASELINE_COMMIT, "size": list(SIZE),
        "camera_policy": "Per-style baseline framing, immutable across phases; sedan cameras for every sweep",
        "paint": PAINT, "background": BACKGROUND, "styles": {}}
    if state.get("version") != 1 or state.get("size") != list(SIZE):
        parser.error("Unsupported baseline state format")
    if args.phase != "before" and "sedan" not in state["styles"]:
        parser.error("Capture baseline first")
    source, fingerprint, temporary = load_pipeline(args)
    try:
        if args.phase == "before" and state.get("source_sha256", fingerprint) != fingerprint:
            parser.error("Refusing to overwrite baseline using different source geometry")
        print(f"Source: {source}; SHA256: {fingerprint}", flush=True)
        size = tuple(x // 2 for x in SIZE) if args.quick else SIZE
        supersample = 1 if args.quick else 2
        if args.phase == "sweeps":
            sweeps_run(args, state, fingerprint, size, supersample)
        else:
            styles_run(args, state, fingerprint, size, supersample)
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
