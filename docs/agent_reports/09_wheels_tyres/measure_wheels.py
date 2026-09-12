#!/usr/bin/env python3
"""Reproducible wheel/15-car assembly measurements (no PNG/export timing)."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
sys.path.insert(0, str(ROOT))
from makecar.body import CarBody
from makecar.assembly import assemble
from makecar.components import get_component
from makecar.components.base import BuildContext, Palette
from makecar.components.exterior import Wheel
from makecar.connectors import CircleConnector
from makecar.geometry.frame import Frame


def main():
    ctx = BuildContext({}, {}, Palette(), np.random.default_rng(0))
    conn = CircleConnector('hub', Frame.identity(), .33, meta={'tire_width': .225})
    data = {'source_sha256': hashlib.sha256((ROOT / 'makecar/components/exterior.py').read_bytes()).hexdigest(),
            'timing_scope': 'Wall time on this machine; body target cache warm, no rendering or file export.',
            'families': {}, 'legacy_max': {}, 'assembly_runs': {}}
    for family in Wheel.SPOKE_FAMILIES:
        res = Wheel().build(conn, {'spoke_family': family}, ctx)
        mesh = res.mesh
        data['families'][family] = {'faces': mesh.n_faces, 'triangles': len(mesh.triangulated()[0]),
                                    'four_wheel_faces': mesh.n_faces * 4,
                                    'spoke_solids': sum(k.startswith('spoke_') for k in mesh.zones)}
        assert mesh.n_faces * 4 < 40000
    for name in ('wheel.alloy', 'wheel.steel'):
        mesh = get_component(name).build(conn, {'spokes': 12, 'tread_blocks': 64, 'tread_grooves': 4}, ctx).mesh
        data['legacy_max'][name] = mesh.n_faces * 4
        assert mesh.n_faces * 4 < 24000
    body = CarBody()
    styles = ('sedan', 'hatchback', 'wagon', 'suv', 'pickup', 'coupe', 'sports', 'van')
    for name, options in (('default', {}), ('low_detail', {'tread_blocks': 0, 'shoulder_blocks': 0,
                                                         'disc_vanes': 0, 'lettering': False})):
        cars = []
        start = time.perf_counter()
        for i in range(15):
            style = styles[i % len(styles)]
            built = body.build(style=style, modifiers={'width': (i % 3 - 1) * .1})
            assembly = assemble(built, {'assign': {'wheel': {'options': options}}}, seed=i)
            mesh = assembly.mesh()
            wheels = [inst for inst in assembly.instances if 'wheel' in inst.connector.tags]
            assert len(wheels) == 4
            assert mesh.n_faces < 120000
            for inst in wheels:
                local = inst.connector.frame.to_local(inst.result.mesh.vertices)
                error = abs(np.linalg.norm(local[:, :2], axis=1).max() - inst.connector.radius)
                assert error < 1e-10
                assert inst.result.mesh.vertices[:, 2].min() >= -.005
            cars.append({'style': style, 'faces': mesh.n_faces,
                         'wheel_faces': sum(inst.result.mesh.n_faces for inst in wheels)})
        data['assembly_runs'][name] = {'cars': cars, 'seconds': round(time.perf_counter() - start, 4),
                                       'wheel_options': options}
    (REPORT / 'measurements.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
