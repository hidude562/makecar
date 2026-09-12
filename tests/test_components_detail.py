"""Exterior detail geometry, not just material-presence smoke tests."""
import itertools

import numpy as np
import pytest

from makecar.components import get_component
from makecar.components.exterior import Wheel, WheelParams
from makecar.connectors import CircleConnector
from makecar.geometry.frame import Frame


def wheel(ctx, **options):
    conn = CircleConnector("hub", Frame.identity(), .33, meta={"tire_width": .225})
    return get_component(options.pop("component", "wheel.alloy")).build(conn, options, ctx).mesh


def test_wheel_target_topology_at_combined_extremes():
    base = Wheel().generate(WheelParams())
    for radius, width, ratio, dish in itertools.product((.23, .47), (.145, .345), (.47, .82), (.01, .09)):
        mesh = Wheel().generate(WheelParams(radius, width, ratio, dish))
        assert mesh.faces == base.faces
        assert mesh.n_vertices == base.n_vertices
        assert np.isfinite(mesh.vertices).all()


def test_tread_is_geometry_and_count_is_an_option(ctx):
    a, b = wheel(ctx, tread_blocks=20), wheel(ctx, tread_blocks=40)
    assert b.n_faces - a.n_faces == 20 * 4 * 6
    assert len([k for k in b.zones if k.startswith("tread_block_")]) == 160
    block = b.vertices[b.groups["tread_block_0_0"]]
    assert np.ptp(np.linalg.norm(block[:, :2], axis=1)) == pytest.approx(.004)
    assert np.ptp(block[:, 2]) > .025
    assert np.linalg.norm(b.vertices[b.groups["tread_rib_0"]][:, :2], axis=1).max() == pytest.approx(.327)


def test_spokes_are_separate_solids_with_open_gaps(ctx):
    a, b = wheel(ctx, spokes=5), wheel(ctx, spokes=10)
    assert len([k for k in a.zones if k.startswith("spoke_")]) == 5
    assert b.n_faces - a.n_faces == 5 * len(a.zones["spoke_0"])
    spoke = a.vertices[a.groups["spoke_0"]]
    assert np.ptp(spoke[:, 2]) > .03
    # There is no longer a full circular face painted dark between spokes.
    core = Wheel().generate(WheelParams())
    assert not any(.08 < np.hypot(*c[:2]) < .17 and c[2] > 0
                   for c, mat in zip(core.face_centroids(), core.face_materials) if mat == "rim")
    assert a.vertices[a.groups["brake_disc"]][:, 2].max() < spoke[:, 2].min()
    assert all(f"lug_{k}" in a.groups for k in range(5))
    assert {"centre_cap", "valve_stem", "caliper"} <= a.groups.keys()


def test_wheel_options_and_budget(ctx):
    for name in ("wheel.alloy", "wheel.steel"):
        m = wheel(ctx, component=name, spokes=12, tread_blocks=64, tread_grooves=4, caliper_color="#123456")
        assert m.n_faces * 4 < 24000
        assert m.materials["caliper"].color == pytest.approx((0x12 / 255, 0x34 / 255, 0x56 / 255))
        assert np.linalg.norm(m.vertices[:, :2], axis=1).max() <= .33000001
    steel = wheel(ctx, component="wheel.steel")
    assert "steel_hubcap" in steel.groups and "steel_web_0" in steel.groups
    assert "steel_hubcap" not in wheel(ctx, component="wheel.steel", hubcap=False).groups
    shallow, deep = wheel(ctx, dish=.01), wheel(ctx, dish=.09)
    assert deep.vertices[deep.groups["spoke_0"]][0, 2] < shallow.vertices[shallow.groups["spoke_0"]][0, 2] - .07


STYLES = ("sedan", "hatchback", "wagon", "suv", "pickup", "coupe", "sports", "van")


@pytest.mark.parametrize("style", STYLES)
def test_lamp_panes_fit_apertures_without_inverted_faces(car_body, ctx, style):
    body = car_body.build(style=style)
    for kind in ("headlight", "taillight"):
        for side in ("L", "R"):
            conn = body.connector(f"{kind}_{side}")
            m = get_component(f"light.{kind}").build(conn, None, ctx).mesh
            pane = m.subset(m.zones["lens"])
            if kind == "headlight":
                assert np.all(pane.face_normals() @ conn.normal > 0), (style, kind, side)
            else:
                # Wagon/SUV/van tail grids already fold in the unmodified body.
                # Preserve their exact geometry; never "fix" it by flipping
                # isolated cells. Bulging must not introduce any further folds.
                from makecar.components.fills import fill_connector
                reference = fill_connector(conn, "reference", upsample=3)
                assert np.all(np.einsum("ij,ij->i", pane.face_normals(), reference.face_normals()) > 0)
            assert np.isfinite(m.vertices).all()
            assert np.all(pane.vertices >= conn.points.min(axis=0) - .01)
            assert np.all(pane.vertices <= conn.points.max(axis=0) + .01)
            # A bulged pane still meets the original aperture's corners.
            for corner in conn.meta["grid_points"][[0, -1]][:, [0, -1]].reshape(-1, 3):
                assert np.linalg.norm(pane.vertices - corner - conn.normal * .001, axis=1).min() < 1e-8
            if kind == "headlight":
                assert {"projector_0", "projector_1", "drl", "indicator"} <= m.groups.keys()
                drl = m.vertices[m.groups["drl"]]
                indicator = m.vertices[m.groups["indicator"]]
                assert drl[:, 2].mean() > m.vertices[m.groups["projector_0"]][:, 2].mean()
                assert np.abs(indicator[:, 1]).mean() > abs(conn.points[:, 1].mean())
            else:
                assert {"light_guide", "reverse", "reflex_strip", "bezel"} <= m.groups.keys()
                assert m.materials["tail_bezel"].color == ctx.material("paint").color


def test_grille_variants_and_badge_clearance(sedan, ctx):
    conn = sedan.connector("grille")
    slats = get_component("grille.slats").build(conn, None, ctx).mesh
    for name, ids in slats.groups.items():
        if name.startswith("slat_"):
            pts = conn.frame.to_local(slats.vertices[ids])
            assert np.ptp(pts[:, 2]) == pytest.approx(.030)
            assert np.linalg.norm(pts[:, :2], axis=1).min() >= .046 - 1e-9
    for name, prefix in (("grille.honeycomb", "cell_"), ("grille.mesh", "wire_")):
        mesh = get_component(name).build(conn, None, ctx).mesh
        assert any(k.startswith(prefix) for k in mesh.groups)
        pts = conn.frame.to_local(mesh.vertices)
        assert np.max(np.abs(pts[:, 0])) <= conn.width / 2 + 1e-9
        assert np.max(np.abs(pts[:, 1])) <= conn.height / 2 + 1e-9


def test_mirror_and_handle_are_world_symmetric(sedan, ctx):
    for connector, component in (("mirror", "mirror.side"), ("handle_front", "handle.pull")):
        left = get_component(component).build(sedan.connector(connector + "_L"), None, ctx).mesh
        right = get_component(component).build(sedan.connector(connector + "_R"), None, ctx).mesh
        assert np.allclose(left.vertices * [1, -1, 1], right.vertices)
    mirror = get_component("mirror.side").build(sedan.connector("mirror_L"), None, ctx).mesh
    assert {"base_plinth", "housing", "mirror_pane", "turn_signal"} <= mirror.groups.keys()
    assert np.all(mirror.face_normals()[mirror.zones["mirror_pane"], 0] < 0)


def test_visual_regressions_have_geometric_guards(sedan, ctx):
    from makecar.components.exterior import _LampSurface
    from makecar.geometry import primitives as P
    m = wheel(ctx)
    disc_top = m.vertices[m.groups["brake_disc"]][:, 2].max()
    hat_top = m.vertices[m.groups["disc_hat"]][:, 2].max()
    assert hat_top - disc_top == pytest.approx(.005)  # no coplanar annulus
    for zone in ("spoke_0", "caliper", "tread_block_0_0"):
        assert P.signed_volume(m.subset(m.zones[zone])) > 0
    steel = wheel(ctx, component="wheel.steel")
    web = steel.subset(steel.zones["steel_web_0"])
    assert web.face_normals()[-2, 1] < -.99
    end_angle = 2 * np.pi / 12 * .55
    assert web.face_normals()[-1] @ [-np.sin(end_angle), np.cos(end_angle), 0] > .99
    conn = sedan.connector("grille")
    grille = get_component("grille.slats").build(conn, None, ctx).mesh
    assert conn.frame.to_local(grille.vertices[grille.groups["back"]])[:, 2].max() > 0
    lamp = sedan.connector("headlight_L")
    surf = _LampSurface(lamp)
    centre = lamp.frame.to_world(surf.at([.5, .49]))[0]
    assert centre[2] < lamp.points[:, 2].max() - .055  # no hood-edge crowding
    guide = surf.ribbon([[.1, .3], [.9, .3], [.9, .8], [.1, .8]], .008, "guide", closed=True)
    assert guide.n_vertices > 6 * 80  # curved grid edges must not become four chords


def _front_depth(mesh, frame, xy):
    """Frontmost triangle hit along a connector-normal ray."""
    triangles, _ = mesh.triangulated()
    a, b, c = frame.to_local(mesh.vertices)[triangles].transpose(1, 0, 2)
    ab, ac, q = b - a, c - a, np.r_[xy, 0.] - a
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    safe = np.where(np.abs(det) > 1e-10, det, 1.)
    u = (q[:, 0] * ac[:, 1] - q[:, 1] * ac[:, 0]) / safe
    v = (ab[:, 0] * q[:, 1] - ab[:, 1] * q[:, 0]) / safe
    hit = (np.abs(det) > 1e-10) & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
    return (a[:, 2] + u * ab[:, 2] + v * ac[:, 2])[hit].max()


@pytest.mark.parametrize("style", STYLES)
def test_intake_backing_clears_uncut_fascia(car_body, ctx, style):
    body = car_body.build(style=style)
    conn = body.connector("intake")
    intake = get_component("grille.intake").build(conn, None, ctx).mesh
    back = intake.subset(intake.zones["back"])
    for x, y in itertools.product((-.35 * conn.width, 0, .35 * conn.width), (-.06, 0, .06)):
        assert _front_depth(back, conn.frame, [x, y]) > _front_depth(body.mesh, conn.frame, [x, y]) + .002
    plate_conn = body.connector("plate_front")
    plate = get_component("plate.standard").build(plate_conn, None, ctx).mesh
    fascia = body.mesh.copy().merge(intake)
    for x, y in itertools.product((-.24, 0, .24), (-.050, 0, .050)):
        # Neither the printed face nor its lower border may hit the insert.
        assert _front_depth(plate, plate_conn.frame, [x, y]) > _front_depth(fascia, plate_conn.frame, [x, y])
