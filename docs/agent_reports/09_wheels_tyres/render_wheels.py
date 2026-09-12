#!/usr/bin/env python3
"""Reproduce wheel inspection PNGs with makecar's numpy rasterizer.

From the worktree root: python3 docs/agent_reports/09_wheels_tyres/render_wheels.py before|after
The before stage must be run at the baseline commit; it refuses to overwrite images.
No extra runtime dependencies or changes to the renderer are needed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
sys.path.insert(0, str(ROOT))
from makecar.components import get_component
from makecar.components.base import BuildContext, Palette
from makecar.connectors import CircleConnector
from makecar.geometry.frame import Frame
from makecar.export.render import Camera, Renderer


def wheel(options=None, radius=.33, width=.225):
    frame = Frame.from_normal([0, 0, radius], [0, 1, 0], [1, 0, 0])
    conn = CircleConnector('wheel_front_L', frame, radius, meta={'tire_width': width})
    ctx = BuildContext({}, {}, Palette(), np.random.default_rng(0))
    return get_component('wheel.alloy').build(conn, options or {}, ctx), conn


def render(mesh, path, az=90, el=0, target=None, distance=1.65):
    target = np.array([0., 0., .33]) if target is None else np.asarray(target)
    cam = Camera.orbit(target, distance, az, el, fov_deg=30)
    renderer = Renderer(960, 960, supersample=2,
                        background=((.92, .94, .96), (.76, .80, .85)))
    image = renderer.render(mesh, cam, ground=False, lines=False)
    renderer.save(image, path)
    print(path.relative_to(REPORT), mesh.n_faces, flush=True)
    return {'path': str(path.relative_to(REPORT)), 'faces': mesh.n_faces,
            'camera': {'eye': cam.eye.tolist(), 'target': cam.target.tolist(),
                       'up': cam.up.tolist(), 'fov_deg': cam.fov_deg}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['before', 'after'])
    args = parser.parse_args()
    folder = REPORT / args.stage
    folder.mkdir(exist_ok=True)
    if args.stage == 'before' and any(folder.glob('*.png')):
        raise SystemExit('Baseline images already exist; refusing to overwrite')
    manifest = {'source_sha256': hashlib.sha256((ROOT / 'makecar/components/exterior.py').read_bytes()).hexdigest(),
                'images': []}
    result, conn = wheel()
    manifest['default_fit'] = result.info
    manifest['family_options'] = {'aspect_ratio': 40, 'disc_pattern': 'drilled', 'lettering': True}
    for name, az, el in [('head_on', 90, 0), ('three_quarter', 48, 18), ('tread', 12, 12)]:
        manifest['images'].append(render(result.mesh, folder / f'default_{name}.png', az, el))
    if args.stage == 'after':
        for family in ('mesh', 'five_spoke', 'twin_five', 'multi_spoke', 'turbine', 'dish', 'steel_cap'):
            result, _ = wheel({'spoke_family': family, 'aspect_ratio': 40, 'disc_pattern': 'drilled', 'lettering': True})
            for name, az, el in [('head_on', 90, 0), ('three_quarter', 48, 18)]:
                manifest['images'].append(render(result.mesh, folder / f'{family}_{name}.png', az, el))
        for pattern in ('directional_v', 'asymmetric_block', 'all_terrain', 'slick'):
            result, _ = wheel({'tread_pattern': pattern})
            manifest['images'].append(render(result.mesh, folder / f'tread_{pattern}.png',
                                              12, 24, target=[.12, 0, .53], distance=.86))
        for label, width, aspect, inches in [('245_35_R20', .245, 35, 20), ('205_65_R15', .205, 65, 15)]:
            radius = width * aspect / 100 + inches * .0254 / 2
            result, _ = wheel({'aspect_ratio': aspect}, radius, width)
            manifest['images'].append(render(result.mesh, folder / f'profile_{label}.png', 65, 10,
                                              target=[0, 0, radius]))
        for pattern in ('plain', 'drilled', 'slotted'):
            result, conn = wheel({'disc_pattern': pattern})
            ids = [i for i, mat in enumerate(result.mesh.face_materials)
                   if mat in ('brake_disc', 'brake_hat', 'caliper', 'brake_pad', 'brake_shield')]
            brake = result.mesh.subset(ids)
            for name, az, el in [('face', 68, 12), ('edge', 12, 12)]:
                manifest['images'].append(render(brake, folder / f'brake_{pattern}_{name}.png', az, el,
                                                  target=conn.frame.to_world([0, 0, -.055])[0], distance=.94))
            manifest['images'].append(render(brake, folder / f'caliper_{pattern}_close.png', 165, 12,
                                              target=conn.frame.to_world([-.15, 0, .0025])[0], distance=.36))
        result, conn = wheel({'aspect_ratio': 35, 'lettering': True, 'disc_pattern': 'drilled', 'spoke_family': 'twin_five'},
                             radius=.245 * .35 + .254, width=.245)
        manifest['images'].append(render(result.mesh, folder / 'sidewall_lettering_close.png', 90, 0,
                                          target=[0, .11, .63], distance=.61))
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
