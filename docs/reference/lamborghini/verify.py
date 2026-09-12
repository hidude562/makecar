#!/usr/bin/env python3
"""Render and measure the Lamborghini config against its side-profile reference.

Run from the repository root with ``python3 docs/reference/lamborghini/verify.py``.
The side camera is orthographic.  Both rows use 6.373 mm per image pixel, so
4.780 m occupies 750 pixels in the rendered row and in the cropped reference.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from makecar.config import CarConfig
from makecar.export.render import Camera, Renderer
from makecar.pipeline import build_car, render_views


HERE = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "lamborghini.yaml"
PUBLISHED_M = {
    "length": 4.780,
    "width": 2.030,
    "height": 1.136,
    "wheelbase": 2.700,
}
PX_PER_METRE = 750 / PUBLISHED_M["length"]
SIDE_SIZE = (900, 250)
THREE_QUARTER_SIZE = (900, 620)


def as_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8), "RGB")


def label(image: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + 28), "#f8f7f3")
    out.paste(image, (0, 28))
    draw = ImageDraw.Draw(out)
    draw.text((12, 7), text, fill="#24211d", font=ImageFont.load_default())
    return out


def side_camera(mesh, measurements, assembly) -> Camera:
    lo, hi = mesh.bounds()
    centre = (lo + hi) / 2
    centre[2] = 0.59
    # Orthographic height fixes the physical scale independently of the camera distance.
    ortho_height = SIDE_SIZE[1] / PX_PER_METRE
    return Camera.orbit(centre, 4 * measurements["length"], 90, 0, ortho=True, ortho_height=ortho_height)


def main() -> None:
    cfg = CarConfig.load(CONFIG)
    body, assembly, _ = build_car(cfg)
    measurements = body.measurements

    print("2012 Lamborghini Aventador LP 700-4 dimensions")
    print(f"{'dimension':<12} {'published m':>12} {'config m':>12} {'difference':>12}")
    for key, published in PUBLISHED_M.items():
        measured = float(measurements[key])
        diff = (measured / published - 1) * 100
        print(f"{key:<12} {published:>12.3f} {measured:>12.3f} {diff:>+11.2f}%")
    clearance = float(measurements["z_floor"])
    print(f"{'clearance':<12} {0.125:>12.3f} {clearance:>12.3f} {(clearance / 0.125 - 1) * 100:>+11.2f}%")

    full = assembly.mesh()
    side = as_pil(Renderer(*SIDE_SIZE, supersample=2).render(full, side_camera(full, measurements, assembly)))
    side.save(HERE / "side_render.png", optimize=True)

    reference = Image.open(HERE / "reference_side.jpg").convert("RGB")
    # The vehicle itself spans 690 px in the hand-checked crop (not its white margins).
    # Resampling it to 750 px makes the reference and render share a physical scale.
    reference = reference.resize((round(reference.width * 750 / 690), round(reference.height * 750 / 690)), Image.Resampling.LANCZOS)
    comparison = Image.new("RGB", (SIDE_SIZE[0], 28 + SIDE_SIZE[1] + 28 + reference.height), "#f8f7f3")
    comparison.paste(label(side, "CONFIG — orthographic side elevation; 4.780 m = 750 px"), (0, 0))
    comparison.paste(label(reference, "REFERENCE — 2012 Aventador LP 700-4 side profile; 4.780 m = 750 px"), (0, 28 + SIDE_SIZE[1]))
    comparison.save(HERE / "comparison.png", optimize=True)

    three_quarter = as_pil(render_views(assembly, ["three_quarter_front"], THREE_QUARTER_SIZE)["three_quarter_front"])
    label(three_quarter, "CONFIG — front three-quarter view").save(HERE / "three_quarter.png", optimize=True)
    print(f"wrote {HERE / 'comparison.png'}")
    print(f"wrote {HERE / 'three_quarter.png'}")


if __name__ == "__main__":
    main()
