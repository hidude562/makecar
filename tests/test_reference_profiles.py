"""Measured reference curves can replace the analytic body profiles."""
import dataclasses
import numpy as np
import pytest

from makecar.body.params import BodyParams
from makecar.body.generator import BodyGenerator
from makecar.body import CarBody
from makecar.reference.trace import Calibration, tyre_radius, trace_edge, clean_trace, resample_xz


def sample(gen, fn, xs):
    return [[float(x), float(fn(x))] for x in xs]


@pytest.fixture(scope="module")
def base():
    return BodyGenerator(BodyParams())


def test_dense_resample_of_analytic_profiles_reproduces_the_body(base):
    L = base.L
    xs = np.linspace(L["x_rear"], L["x_front"], 400)
    ref = {"profiles": {"z_center": sample(base, base.center_height, xs), "floor": sample(base, base.floor, xs),
                        "belt": sample(base, base.belt, xs), "sill": sample(base, base.sill, xs),
                        "crown": sample(base, base.crown, xs), "roof_edge": sample(base, base.roof_edge, xs),
                        "half_width": sample(base, base.plan_half_width, xs)}}
    a = base.build()
    b = BodyGenerator(dataclasses.replace(BodyParams(), reference=ref)).build()
    assert a.n_vertices == b.n_vertices and a.faces == b.faces
    assert np.abs(a.vertices - b.vertices).max() < 0.012  # spline re-fit of sharp keypoints only


def test_measured_roofline_moves_the_roof_and_keeps_topology(base):
    L = base.L
    xs = np.linspace(L["x_rear"], L["x_front"], 120)
    bump = lambda x: base.center_height(x) + 0.08 * np.exp(-((x - (L["u_roof_front"] + L["u_roof_rear"]) / 2) / 0.4) ** 2)
    m0 = base.build()
    m1 = BodyGenerator(dataclasses.replace(BodyParams(), reference={"profiles": {"z_center": sample(base, bump, xs)}})).build()
    assert m1.faces == m0.faces
    assert m1.vertices[:, 2].max() == pytest.approx(m0.vertices[:, 2].max() + 0.08, abs=0.01)


def test_reference_validation():
    with pytest.raises(KeyError):
        BodyGenerator(dataclasses.replace(BodyParams(), reference={"profiles": {"roofline": [[0, 1], [1, 1]]}}))
    with pytest.raises(ValueError):
        BodyGenerator(dataclasses.replace(BodyParams(), reference={"profiles": {"floor": [[0, 0.1]]}}))


def test_library_caches_params_with_reference():
    L = BodyGenerator(BodyParams()).L
    ref = {"profiles": {"floor": [[L["x_rear"], 0.2], [L["x_front"], 0.2]]}}
    cb = CarBody(dataclasses.replace(BodyParams(), reference=ref))
    assert len(cb.library.modifiers) > 50
    res = cb.build(style="sedan")
    assert res.full_mesh.vertices[:, 2].min() == pytest.approx(0.2, abs=0.02)


def test_calibration_round_trip_with_roll_and_mirroring():
    wb, r = 2.7356, tyre_radius(235, 40, 18)
    for front_right in (True, False):
        s, roll = 1 / 900.0, np.radians(-2.0)

        def px(x, z):
            u, v = x / s, -z / s
            c, sn = np.cos(roll), np.sin(roll)
            u, v = c * u - sn * v, sn * u + c * v
            return (2000 + (u if front_right else -u), 1600 + v)

        cal = Calibration.from_hubs(px(wb / 2, r), px(-wb / 2, r), wb, r, r)
        assert cal.residual() < 1e-9
        pts = np.array([[1.5, 1.2], [-2.2, 0.3]])
        assert np.allclose(cal.to_world([px(*p) for p in pts]), pts, atol=1e-9)


def test_tyre_radius_and_edge_tracer():
    assert tyre_radius(235, 40, 18) == pytest.approx(0.3226, abs=1e-4)
    img = np.zeros((300, 500, 3), np.float32)
    v = 150 + 20 * np.sin(np.arange(500) / 40)
    for u in range(500):
        img[int(v[u]):, u] = 200
    uv = clean_trace(trace_edge(img, 5, 495, (90, 240), polarity="rise", step=5))
    assert len(uv) > 90 and np.abs(uv[:, 1] - (150 + 20 * np.sin(uv[:, 0] / 40))).max() < 2.5  # steep slope + 3-column averaging
    xz = resample_xz(np.column_stack([uv[:, 0], uv[:, 1]]), n=10)
    assert len(xz) == 10 and xz[0][0] < xz[-1][0]
