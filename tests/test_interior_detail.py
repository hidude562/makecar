"""Physical packaging and detail contracts; distances are metres, not pixels."""
import dataclasses

import numpy as np
import pytest

from makecar.assembly import assemble
from makecar.body.body import CarBody
from makecar.body.connectors import aperture_loop
from makecar.components.base import BuildContext, Palette, get_component
from makecar.components.interior import BucketSeat, SportSeat, SeatParams


@pytest.fixture(scope="module", params=CarBody.styles())
def cabin(request):
    return assemble(CarBody().build(request.param))


def instances(assembly):
    return {i.connector.name: i for i in assembly.instances}


def test_adult_front_seat_packaging_all_styles(cabin):
    parts = instances(cabin)
    wheel = cabin.body.connector("steering_wheel")
    pedal = parts["pedals"]
    heel = pedal.connector.frame.to_world(pedal.result.info["heel_local"])[0]
    for name in ("seat_front_driver", "seat_front_passenger"):
        seat = parts[name]
        conn, mesh = seat.connector, seat.result.mesh
        hp = conn.frame.to_world(seat.result.info["h_point_local"])[0]
        # The mount rectangle's centre is not the H-point.  Use the fitted
        # seat's declared hip datum, and independently measure its geometry.
        local = conn.frame.to_local(mesh.vertices)
        cushion = local[mesh.groups["cushion"]]
        assert np.ptp(cushion[:, 0]) == pytest.approx(0.50)
        assert 0.48 <= np.ptp(cushion[:, 1]) <= 0.52
        axis = np.array([-np.sin(np.radians(22)), 0, np.cos(np.radians(22))])
        assert 0.60 <= np.ptp(local[mesh.groups["backrest"]] @ axis) <= 0.65
        assert mesh.vertices[mesh.groups["headrest"], 2].max() - hp[2] >= 0.80
        for rail in ("rail_0", "rail_1"):
            assert local[mesh.groups[rail], 2].min() == pytest.approx(0.0)
        if name.endswith("driver"):
            assert 0.8 <= heel[0] - hp[0] <= 0.9
            assert 0.55 <= wheel.origin[0] - hp[0] <= 0.65
            assert 0.30 <= wheel.origin[2] - hp[2] <= 0.35
    assert 20 <= np.degrees(np.arctan2(wheel.normal[2], -wheel.normal[0])) <= 25
    rim = wheel.frame.to_local(parts["steering_wheel"].result.mesh.vertices)
    rim = rim[parts["steering_wheel"].result.mesh.groups["rim"]]
    assert np.ptp(rim[:, 0]) == pytest.approx(0.37, abs=0.002)
    assert np.ptp(rim[:, 2]) == pytest.approx(0.032, abs=0.002)


def test_pillar_and_belt_connectors_follow_body(cabin):
    body = cabin.body
    _, _, _, glass = aperture_loop(body.full_mesh, "aperture/windshield")
    parts = instances(cabin)
    for side, sign in (("L", 1), ("R", -1)):
        a = body.connector(f"pillar_trim_A_{side}")
        col = np.argmax(glass[:, :, 1].mean(axis=0) * sign)
        assert np.allclose(a.meta["path"], glass[:, col] + [-0.02, -sign * 0.025, -0.025])
        assert a.normal[1] * sign < 0
        belt = body.connector(f"belt_anchor_{side}")
        seat = parts[belt.meta["seat"]]
        buckle = seat.connector.frame.to_world(seat.result.info["buckle_local"])[0]
        assert np.allclose(belt.meta["buckle"], buckle)
        assert belt.origin[2] > buckle[2] + 0.35
        assert "shoulder_webbing" in parts[belt.name].result.mesh.groups
        assert a.name in parts
        assert f"pillar_trim_B_{side}" in parts
    assert "cowl_trim" in parts


def test_sport_seat_and_headrest_option():
    body = CarBody().build("sports")
    conn = body.connector("seat_front_driver")
    ctx = BuildContext(body.measurements, body.hints, Palette(), np.random.default_rng(0))
    sport = get_component("seat.sport").build(conn, None, ctx)
    standard = get_component("seat.bucket").build(conn, None, ctx)
    assert sport.info["modifier_values"]["bolster"] > standard.info["modifier_values"]["bolster"]
    assert not any("post" in key for key in sport.mesh.zones)
    assert "headrest" in sport.mesh.zones
    hp = conn.frame.to_world(sport.info["h_point_local"])[0]
    assert sport.mesh.vertices[sport.mesh.groups["headrest"], 2].max() - hp[2] >= 0.80
    no_head = get_component("seat.bucket").build(conn, {"headrest": False}, ctx)
    assert "headrest" not in no_head.mesh.zones
    assert "backrest" in no_head.mesh.zones
    assert no_head.mesh.n_faces < standard.mesh.n_faces


@pytest.mark.parametrize("cls", [BucketSeat, SportSeat])
def test_seat_targets_keep_exact_topology(cls):
    component = cls()
    base = component.generate(SeatParams())
    for spec in component.modifier_specs:
        for delta in (-spec.delta_minus, spec.delta_plus):
            params = dataclasses.replace(SeatParams(), **{spec.param: getattr(SeatParams(), spec.param) + delta})
            mesh = component.generate(params)
            assert mesh.n_vertices == base.n_vertices
            assert mesh.faces == base.faces
            assert np.isfinite(mesh.vertices).all()


def test_interior_budget_and_materials(cabin):
    assert cabin.interior_mesh().n_faces < 50000
    assert cabin.mesh().n_faces < 120000
    for part in cabin.instances:
        mesh = part.result.mesh
        assert len(mesh.face_materials) == mesh.n_faces
        assert set(mesh.face_materials) <= set(mesh.materials)
        assert np.isfinite(mesh.vertices).all()
