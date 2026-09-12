#!/usr/bin/env python3
"""Render a dimensioned 2024 Tesla Model 3 configuration comparison."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from makecar.config import CarConfig
from makecar.export.png import write_png
from makecar.export.render import Camera, Renderer
from makecar.pipeline import build_car

HERE = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "tesla.yaml"
REFERENCE = HERE / "model3_2024_side_reference.jpg"

# 2024 US-market Model 3 RWD / Long Range; Tesla owner's manual.
PUBLISHED = {
    "length": 4.720,
    "width": 1.849,
    "height": 1.440,
    "wheelbase": 2.875,
    "ground_clearance": 0.138,
}
# The fixed CC0 reference's vehicle outline, inspected in its original 1280x768 image.
REFERENCE_VEHICLE_X = (60, 1207)
REFERENCE_CROP = (40, 175, 1240, 625)
PIXELS_PER_METRE = 270
SIDE_SIZE = (1500, 600)
THREE_QUARTER_SIZE = (900, 620)


def label(image: Image.Image, text: str) -> Image.Image:
    """Add a concise title band without altering the calibrated image scale."""
    band = Image.new("RGB", (image.width, 38), "#ffffff")
    draw = ImageDraw.Draw(band)
    draw.text((16, 11), text, fill="#20252c", font=ImageFont.load_default())
    out = Image.new("RGB", (image.width, image.height + band.height), "#ffffff")
    out.paste(band, (0, 0))
    out.paste(image, (0, band.height))
    return out


def reference_at_model_scale() -> Image.Image:
    """Crop the source photo and size its measured vehicle to the model scale."""
    source = Image.open(REFERENCE).convert("RGB")
    crop = source.crop(REFERENCE_CROP)
    source_vehicle_width = REFERENCE_VEHICLE_X[1] - REFERENCE_VEHICLE_X[0]
    scale = (PUBLISHED["length"] * PIXELS_PER_METRE) / source_vehicle_width
    scaled = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", SIDE_SIZE, "#ffffff")
    canvas.paste(scaled, ((canvas.width - scaled.width) // 2, (canvas.height - scaled.height) // 2))
    return canvas


def rendered_image(array: np.ndarray) -> Image.Image:
    """Convert the renderer's floating RGB buffer for Pillow composition."""
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def main() -> None:
    if not REFERENCE.exists():
        raise FileNotFoundError(f"missing reference image: {REFERENCE}")

    cfg = CarConfig.load(CONFIG)
    body, assembly, _ = build_car(cfg)
    measured = body.measurements

    print("2024 Tesla Model 3 RWD / Long Range — metres")
    print(f"{'metric':<18} {'published':>10} {'measured':>10} {'delta':>9}")
    for key in ("length", "width", "height", "wheelbase", "ground_clearance"):
        value = measured["z_floor"] if key == "ground_clearance" else measured[key]
        delta = (value / PUBLISHED[key] - 1.0) * 100
        print(f"{key:<18} {PUBLISHED[key]:10.3f} {value:10.3f} {delta:8.2f}%")

    exterior = assembly.exterior_mesh()
    lo, hi = exterior.bounds()
    target = (lo + hi) / 2
    target[2] = PUBLISHED["height"] / 2

    # With this orthographic height, 270 px represents one metre in both rows.
    side_camera = Camera.orbit(
        target, 20.0, 90, 0, ortho=True,
        ortho_height=SIDE_SIZE[1] / PIXELS_PER_METRE,
    )
    side = Renderer(*SIDE_SIZE, supersample=2, background=((1, 1, 1), (1, 1, 1))).render(
        exterior, side_camera, ground=False,
    )
    write_png(HERE / "rendered_side.png", side)
    side_image = rendered_image(side)

    three_quarter_camera = Camera.orbit(target, 2.05 * measured["length"], 35, 16, fov_deg=30)
    three_quarter = Renderer(*THREE_QUARTER_SIZE, supersample=2).render(exterior, three_quarter_camera)
    write_png(HERE / "rendered_three_quarter.png", three_quarter)

    front_camera = Camera.orbit(target, 20.0, 0, 0, ortho=True, ortho_height=2.2)
    front = Renderer(700, 620, supersample=2).render(exterior, front_camera)
    write_png(HERE / "rendered_front.png", front)

    nose_target = np.array([measured["x_front"] - 0.25, 0.0, 0.60])
    nose_camera = Camera.orbit(nose_target, 3.2, 35, 10, fov_deg=28)
    nose = Renderer(900, 620, supersample=2).render(exterior, nose_camera)
    write_png(HERE / "rendered_nose.png", nose)

    reference = reference_at_model_scale()
    reference.save(HERE / "reference_cropped_scaled.png", optimize=True)
    comparison = Image.new("RGB", (SIDE_SIZE[0], 2 * (SIDE_SIZE[1] + 38) + THREE_QUARTER_SIZE[1] + 38), "#ffffff")
    comparison.paste(label(reference, "CC0 side-profile reference — 2023 refresh sold as 2024 Model 3"), (0, 0))
    comparison.paste(label(side_image, "makecar config approximation — same 270 px/m scale"), (0, SIDE_SIZE[1] + 38))
    quarter = rendered_image(three_quarter)
    comparison.paste(label(quarter, "makecar three-quarter exterior view"), ((SIDE_SIZE[0] - quarter.width) // 2, 2 * (SIDE_SIZE[1] + 38)))
    comparison.save(HERE / "comparison.png", optimize=True)
    print(f"wrote {HERE / 'comparison.png'}")


if __name__ == "__main__":
    main()
