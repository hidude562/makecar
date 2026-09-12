#!/usr/bin/env python3
"""Build the Civic approximation and render scale-matched reference views.

The saved reference is intentionally a private modelling aid.  Its crop spans the
car almost edge to edge, so its pixel width divided by the published overall
length establishes the side-view scale used for the generated car.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from makecar.config import CarConfig
from makecar.export.render import Camera, Renderer
from makecar.pipeline import build_car


CONFIG = ROOT / "configs" / "civic.yaml"
REFERENCE = Path(__file__).with_name("reference_side.jpg")
PUBLISHED_METRES = {
    "length": 4.6736,
    "width": 1.8009,
    "height": 1.4097,
    "wheelbase": 2.7356,
}


def render_views(body, assembly, reference: Image.Image):
    """Render a true orthographic side view at the reference's pixel scale."""
    mesh = assembly.mesh()
    lo, hi = mesh.bounds()
    center = (lo + hi) / 2
    center[2] = PUBLISHED_METRES["height"] / 2
    length = float(hi[0] - lo[0])

    pixels_per_metre = reference.width / PUBLISHED_METRES["length"]
    side_width, side_height = 1200, 520
    side_camera = Camera.orbit(
        center,
        4 * length,
        90,
        0,
        ortho=True,
        ortho_height=side_height / pixels_per_metre,
    )
    side_renderer = Renderer(side_width, side_height, supersample=2)
    side = Image.fromarray(np.clip(side_renderer.render(mesh, side_camera, ground=False), 0, 255).astype(np.uint8))

    quarter_camera = Camera.orbit(center, 2.05 * length, 35, 16, fov_deg=30)
    quarter_renderer = Renderer(1200, 800, supersample=2)
    quarter = Image.fromarray(np.clip(quarter_renderer.render(mesh, quarter_camera), 0, 255).astype(np.uint8))
    return side, quarter, pixels_per_metre


def write_comparison(output: Path, reference: Image.Image, side: Image.Image):
    """Place the web reference directly above the generated side elevation."""
    canvas = Image.new("RGB", (1200, 940), "#f5f1e9")
    draw = ImageDraw.Draw(canvas)
    reference_x = (canvas.width - reference.width) // 2
    canvas.paste(reference, (reference_x, 42))
    draw.text((24, 16), "WEB REFERENCE — 2022 Honda Civic Si Sedan (11th generation)", fill="#252525")
    draw.line((24, 370, 1176, 370), fill="#b6afa2", width=1)
    draw.text((24, 388), "CONFIG APPROXIMATION — orthographic side elevation, same overall-length scale", fill="#252525")
    canvas.paste(side, (0, 412))
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = CarConfig.load(CONFIG)
    body, assembly, _ = build_car(config)
    reference = Image.open(REFERENCE).convert("RGB")
    side, quarter, pixels_per_metre = render_views(body, assembly, reference)
    write_comparison(args.output / "comparison.png", reference, side)
    quarter.save(args.output / "three_quarter.png")

    print("Published versus measured body dimensions (metres):")
    for name, published in PUBLISHED_METRES.items():
        measured = body.measurements[name]
        error = 100 * (measured - published) / published
        print(f"{name:10s} published={published:.4f} measured={measured:.4f} error={error:+.2f}%")
    print(f"side-view scale: {pixels_per_metre:.2f} px/m")
    print(f"wrote {args.output / 'comparison.png'}")
    print(f"wrote {args.output / 'three_quarter.png'}")


if __name__ == "__main__":
    main()
