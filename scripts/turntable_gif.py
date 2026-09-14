#!/usr/bin/env python3
"""Render a looping GIF that morphs through every body style while the camera orbits.
Usage (from the repo root): python -m scripts.turntable_gif [-o output/turntable.gif] [--frames-per-style 12]

Needs Pillow (pip install pillow) on top of the normal makecar dependencies.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from makecar.body import CarBody
from makecar.body.styles import style_names
from makecar.assembly import assemble
from makecar.components.base import Palette
from makecar.export.render import Renderer, Camera

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--out", default="output/turntable.gif")
ap.add_argument("--frames-per-style", type=int, default=12)
ap.add_argument("--size", type=int, nargs=2, default=[480, 300])
ap.add_argument("--elevation", type=float, default=16.0)
args = ap.parse_args()

styles = style_names()
body = CarBody()
palette = Palette()
renderer = Renderer(args.size[0], args.size[1], supersample=2)

n = len(styles)
total = n * args.frames_per_style
frames = []
for k in range(total):
    pos = k / args.frames_per_style
    i = int(pos) % n
    t = pos - int(pos)
    a, b = styles[i], styles[(i + 1) % n]
    style = {a: 1.0 - t, b: t} if t > 1e-9 else a
    res = body.build(style, {}, {}, {}, paint=palette.paint)
    asm = assemble(res, {"defaults": True}, palette, seed=1)
    mesh = asm.mesh()
    lo, hi = mesh.bounds()
    c = (lo + hi) / 2
    c[2] = 0.72
    length = float(hi[0] - lo[0])
    cam = Camera.orbit(c, 2.05 * length, 360.0 * k / total, args.elevation, fov_deg=30)
    img = renderer.render(mesh, cam)
    frames.append(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)))
    print(f"frame {k + 1}/{total}  {a} -> {b}  t={t:.2f}")

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
frames[0].save(out, save_all=True, append_images=frames[1:], duration=90, loop=0)
print(f"wrote {out} ({len(frames)} frames)")
