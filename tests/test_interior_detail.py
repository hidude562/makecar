"""Physical packaging and detail contracts; distances are metres, not pixels."""
import dataclasses

import numpy as np
import pytest

from makecar.assembly import assemble, INTERIOR_TAGS
from makecar.connectors import PointConnector
from makecar.geometry.frame import Frame
from makecar.geometry.primitives import signed_volume
from makecar.body.body import CarBody
from makecar.body.connectors import aperture_loop
from makecar.components.base import BuildContext, Palette, get_component
from makecar.components.interior import BucketSeat, SportSeat, SeatParams, _cushion_section, _backrest_section
from makecar.components import interior_helpers as H


@pytest.fixture(scope="module", params=CarBody.styles())
def cabin(request):
    return assemble(CarBody().build(request.param))


def instances(assembly):
    return {i.connector.name: i for i in assembly.instances}


def complete_interior(assembly):
    # The shared assembler's hardcoded INTERIOR_TAGS is outside task ownership.
    # Include neutral-tagged extensions explicitly rather than undercount them.
    mesh = assembly.interior_mesh()
    for part in assembly.instances:
        if "interior_detail" in part.connector.tags and not part.connector.tags & INTERIOR_TAGS:
            mesh.merge(part.result.mesh)
    return mesh


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
        assert seat.result.info["headroom_clearance"] > 0
        assert "fit_warnings" not in seat.result.info
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
    assert np.allclose(rim[:10], rim[-10:])
    rings = rim[:-10].reshape(-1, 10, 3)
    centres = rings.mean(axis=1)
    assert np.allclose(np.linalg.norm(rings - centres[:, None], axis=2), 0.016)
    if cabin.body.hints["style"] == "sports":
        assert centres[:, 1].max() == pytest.approx(0.185 * 0.74)
        assert np.count_nonzero(np.isclose(centres[:, 1], centres[:, 1].max())) > 5
        mesh = parts["steering_wheel"].result.mesh
        spoke = wheel.frame.to_local(mesh.vertices[mesh.groups["spoke_2"]])
        assert spoke[:, 1].max() <= centres[:, 1].max() + 1e-9


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
        assert np.allclose(parts[belt.name].result.info["buckle_world"], buckle)
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
    interior = complete_interior(cabin)
    assert interior.n_faces < 50000
    assert len(interior.triangulated()[0]) < 50000
    assert cabin.mesh().n_faces < 120000
    for part in cabin.instances:
        mesh = part.result.mesh
        assert len(mesh.face_materials) == mesh.n_faces
        assert set(mesh.face_materials) <= set(mesh.materials)
        assert np.isfinite(mesh.vertices).all()


def test_dashboard_controls_and_closure_all_styles(cabin):
    parts = instances(cabin)
    assert "passenger_airbag_seam" in parts["dashboard"].result.mesh.groups
    assert "upper_firewall" in parts["firewall"].result.mesh.groups
    cluster = parts["dashboard/cluster"]
    assert {"tachometer_face", "speedometer_face", "needle_0", "needle_1", "binnacle_hood"} <= set(cluster.result.mesh.groups)
    assert cluster.connector.origin[2] >= cabin.body.connector("steering_wheel").origin[2] + 0.074
    hvac = parts["dashboard/hvac"]
    console = parts["console"].result.mesh
    assert hvac.result.mesh.bounds()[0][2] > console.vertices[console.groups["console_body"], 2].max()
    screen = parts["dashboard/screen"].result.mesh
    assert screen.bounds()[0][2] > hvac.result.mesh.bounds()[1][2]
    assert parts["dashboard/vent_center_L"].result.mesh.bounds()[0][2] > screen.bounds()[1][2]
    assert "binnacle_back" in cluster.result.mesh.groups
    local = hvac.connector.frame.to_local(hvac.result.mesh.vertices)
    for i in range(6):
        button = local[hvac.result.mesh.groups[f"hvac_button_{i}"]]
        assert np.ptp(button[:, 0]) == pytest.approx(0.012)
        assert np.ptp(button[:, 1]) == pytest.approx(0.010)
    for side in ("L", "R"):
        for prefix in ("pillar_trim_A", "pillar_trim_B", "pillar_trim_C", "pillar_trim_rear", "cabin_side_liner"):
            mesh = parts[f"{prefix}_{side}"].result.mesh
            assert "pillar_skin" in mesh.groups
            assert np.min(np.linalg.norm(mesh.face_normals(), axis=1)) > 0.99
    mirror = parts["rearview_mirror"]
    assert mirror.result.info["header_mount_world"][0] == pytest.approx(cabin.body.measurements["x_roof_front"] + 0.015)
    assert mirror.result.mesh.vertices[:, 0].mean() > cabin.body.measurements["x_roof_front"] - 0.20


def test_console_floor_doors_and_ceiling_all_styles(cabin):
    parts = instances(cabin)
    assert {"cup_sliding_cover", "ebrake_switch", "armrest_seam"} <= set(parts["console"].result.mesh.groups)
    floor = parts["floor"].result.mesh
    mats = [g for g in floor.groups if g.startswith("floor_mat_")]
    assert len(mats) == 4
    for g in mats:
        assert np.ptp(floor.vertices[floor.groups[g], 1]) == pytest.approx(0.36)
    for name, part in parts.items():
        if "door_card" in part.connector.tags:
            assert {"map_pocket", "sill_scuff_plate", "sill_return"} <= set(part.result.mesh.groups)
            assert "speaker_grille_mesh" in parts[name + "/speaker"].result.mesh.groups
        if "seat_bench" in part.connector.tags:
            assert part.result.info["headrests"] == 3
            assert all(f"headrest_{i}/headrest" in part.result.mesh.groups for i in range(3))
    assert {"vanity_mirror_L", "vanity_mirror_R"} <= set(parts["headliner"].result.mesh.groups)
    for side in ("L", "R"):
        assert {"grab_base_0", "grab_base_1"} <= set(parts[f"headliner/grab_handle_{side}"].result.mesh.groups)


def test_cargo_wheelhouses_track_rear_axle(cabin):
    if cabin.body.measurements["has_bed"]:
        return
    parts = instances(cabin)
    for side in ("L", "R"):
        part = parts[f"cargo_wheelhouse_{side}"]
        assert part.connector.origin[0] == pytest.approx(cabin.body.measurements["wheel_rear_x"])
        lo, hi = part.result.mesh.bounds()
        assert hi[2] > part.connector.origin[2] + 0.18
        assert (lo[0] + hi[0]) / 2 == pytest.approx(part.connector.origin[0])
        assert np.ptp(part.result.mesh.vertices[:, 1]) == pytest.approx(0.25)


def test_stitch_grooves_are_inset_geometry():
    for section in (_cushion_section(0.50, 0.11, 0.035), _backrest_section(0.50, 0.05, 0.055, 0.035)):
        for left, bottom, right in ((7, 8, 9), (11, 12, 13)):
            assert section[bottom, 1] < min(section[left, 1], section[right, 1]) - 0.003
            assert abs(section[left, 0] - section[right, 0]) == pytest.approx(0.006)


def test_speaker_grille_has_real_open_cells():
    mesh = H.grille(0.08)
    ribs = mesh.vertices[mesh.groups["grille_mesh"]].reshape(-1, 8, 3)
    # The origin lies between crossing ribs, not on a disguised solid disc.
    lo, hi = ribs.min(axis=1), ribs.max(axis=1)
    assert not np.any(np.all((lo[:, :2] <= 0) & (hi[:, :2] >= 0), axis=1))
    assert len(ribs) > 40
    assert np.allclose(np.sort(hi[:, :2] - lo[:, :2], axis=1)[:, 0], 0.001)


def test_cluster_hood_end_caps_do_not_bridge_opening(cabin):
    mesh = instances(cabin)["dashboard/cluster"].result.mesh
    faces = mesh.zone_faces("binnacle_hood")
    # End caps must be strips, not fan triangles bridging the gauge opening.
    assert all(len(mesh.faces[i]) == 4 for i in faces)
    assert len(faces) == 52


@pytest.mark.parametrize("style", ["sedan", "suv", "sports"])
def test_right_hand_drive_manual_and_component_options(style):
    body = CarBody().build(style, hints={"drive": "right", "transmission": "manual"})
    cabin = assemble(body, {"assign": {"seat_front": "seat.sport", "console": {"options": {"cup_cover": 1.0}}}})
    parts = instances(cabin)
    assert parts["seat_front_driver"].connector.origin[1] < 0
    assert parts["dashboard/cluster"].connector.origin[1] < 0
    assert "footrest" in parts["pedals"].result.mesh.groups
    for side in ("L", "R"):
        belt = body.connector(f"belt_anchor_{side}")
        seat = parts[belt.meta["seat"]]
        assert np.allclose(parts[belt.name].result.info["buckle_world"], seat.connector.frame.to_world(seat.result.info["buckle_local"])[0])
    ctx = BuildContext(body.measurements, body.hints, Palette(), np.random.default_rng(0))
    opened = get_component("console.center").build(body.connector("console"), {"cup_cover": 0, "armrest": False}, ctx)
    assert "cup_sliding_cover" not in opened.mesh.groups
    assert "armrest_lid" not in opened.mesh.groups
    round_wheel = get_component("steering.wheel").build(body.connector("steering_wheel"), {"flat_bottom": False}, ctx)
    assert not round_wheel.info["flat_bottom"]
    bench = next((p for p in parts.values() if "seat_bench" in p.connector.tags), None)
    if bench:
        single = get_component("seat.bench").build(bench.connector, {"headrests": 1}, ctx)
        loc = bench.connector.frame.to_local(single.mesh.vertices[single.mesh.groups["headrest_0/headrest"]])
        assert loc[:, 1].mean() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("tag,component", [("dashboard", "dashboard.sculpted"), ("bulkhead", "bulkhead.trim"),
                                         ("seat", "seat.bucket"), ("cargo_floor", "floor.cargo")])
def test_existing_tag_assignments_do_not_select_extension_points(tag, component):
    cabin = assemble(CarBody().build("suv"), {"assign": {tag: component}})
    assert not cabin.unattached or set(cabin.unattached) <= {"roof_rail_L", "roof_rail_R"}
    for part in cabin.instances:
        if "interior_detail" in part.connector.tags:
            assert tag not in part.connector.tags


@pytest.mark.parametrize("hp", [0.10, 0.50])
def test_belt_endpoint_uses_clamped_seat_fit(hp):
    cabin = assemble(CarBody().build("van", hints={"h_point_height": hp}))
    parts = instances(cabin)
    for side in ("L", "R"):
        belt = parts[f"belt_anchor_{side}"]
        seat = parts[belt.connector.meta["seat"]]
        target = seat.connector.frame.to_world(seat.result.info["buckle_local"])[0]
        assert np.allclose(belt.result.info["buckle_world"], target)


@pytest.mark.parametrize("end", [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0.4, 0.6, 0.8]])
def test_ribbon_winding_faces_outward(end):
    mesh = H.ribbon([[0, 0, 0], end], 0.04, 0.002, "int_belt")
    assert signed_volume(mesh) == pytest.approx(np.linalg.norm(end) * 0.04 * 0.002)


def test_explicit_mirror_mounts_follow_connector():
    ctx = BuildContext({}, {}, Palette(), np.random.default_rng(0))
    component = get_component("mirror.rearview")
    a = PointConnector("custom_mirror", Frame.from_normal([0.3, 0, 1.2], [-1, 0, 0], x_hint=(0, 1, 0)))
    b = PointConnector("custom_mirror", Frame.from_normal([0.5, 0.1, 1.3], [-1, 0, 0], x_hint=(0, 1, 0)))
    first, second = component.build(a, None, ctx), component.build(b, None, ctx)
    assert not first.info["fitted_header"]
    assert np.allclose(second.mesh.vertices - first.mesh.vertices, [0.2, 0.1, 0.1])
    body = CarBody().build()
    ctx = BuildContext(body.measurements, body.hints, Palette(), np.random.default_rng(0))
    old = component.build(body.connector("rearview_mirror"), {"fit_header": False}, ctx)
    assert np.allclose(old.info["header_mount_world"], body.connector("rearview_mirror").origin)


def test_impossible_headroom_is_reported_not_silently_miniaturized():
    body = CarBody().build()
    seat = body.connector("seat_front_driver")
    seat.meta["headroom"] = 1.0
    ctx = BuildContext(body.measurements, body.hints, Palette(), np.random.default_rng(0))
    result = get_component("seat.bucket").build(seat, None, ctx)
    assert result.info["headroom_clearance"] < 0
    assert result.info["fit_warnings"]
    assert 0.60 <= result.info["backrest_length"] <= 0.65

