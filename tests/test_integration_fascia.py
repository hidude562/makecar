"""Whole-footprint fit against the actual stepped shell, not its old plane."""
import itertools

import numpy as np
import pytest

from makecar.body import style_names
from makecar.components import BuildContext, Palette, get_component


@pytest.fixture(scope="module", params=style_names())
def fitted(request, car_body):
    body = car_body.build(style=request.param)
    ctx = BuildContext(body.measurements, body.hints, Palette(), np.random.default_rng(0))
    return body, ctx


def front_depths(mesh, frame, xy):
    triangles, _ = mesh.triangulated()
    a, b, c = frame.to_local(mesh.vertices)[triangles].transpose(1, 0, 2)
    ab, ac = b - a, c - a
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    valid = abs(det) > 1e-12
    den = np.where(valid, det, 1.)
    out = []
    for x, y in xy:
        q = np.array([x, y]) - a[:, :2]
        u = (q[:, 0] * ac[:, 1] - q[:, 1] * ac[:, 0]) / den
        v = (ab[:, 0] * q[:, 1] - ab[:, 1] * q[:, 0]) / den
        hit = valid & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
        assert hit.any(), (mesh.name, x, y)
        out.append((a[:, 2] + u * ab[:, 2] + v * ac[:, 2])[hit].max())
    return np.asarray(out)


def footprint(width, height):
    return list(itertools.product(np.linspace(-.43, .43, 9) * width,
                                  np.linspace(-.40, .40, 7) * height))


def test_all_grille_backing_vertices_seat_ahead_of_mount(fitted):
    body, ctx = fitted
    for name in ("grille", "intake"):
        conn = body.connector(name)
        for component in ("grille.slats", "grille.honeycomb", "grille.mesh"):
            mesh = get_component(component).build(conn, None, ctx).mesh
            back = mesh.subset(mesh.zones["back"])
            local = conn.frame.to_local(back.vertices)
            assert local[:, 2].min() >= -1e-10  # not just the front face
            xy = footprint(conn.width, conn.height)
            assert np.all(front_depths(back, conn.frame, xy)
                          > front_depths(body.mesh, conn.frame, xy) + .002)


def test_front_plate_regions_clear_skin_and_intake(fitted):
    body, ctx = fitted
    conn = body.connector("plate_front")
    intake = get_component("grille.intake").build(body.connector("intake"), None, ctx).mesh
    fascia = body.mesh.copy().merge(intake)
    for region, width, height in (("eu", .52, .11), ("us", .305, .152)):
        mesh = get_component("plate.standard").build(conn, {"region": region}, ctx).mesh
        xy = list(itertools.product(np.linspace(-.46, .46, 9) * width,
                                    np.linspace(-.46, .46, 9) * height))
        assert np.all(front_depths(mesh, conn.frame, xy)
                      > front_depths(fascia, conn.frame, xy) + .002)
        plinth = conn.frame.to_local(mesh.vertices[mesh.groups["mounting_plinth"]])
        assert plinth[:, 2].min() == pytest.approx(-.012)
        if region == "eu":
            assert np.ptp(plinth[:, 2]) == pytest.approx(.018)


def test_single_default_nose_badge_and_optional_grille_badge(fitted):
    body, ctx = fitted
    conn = body.connector("grille")
    plain = get_component("grille.slats").build(conn, None, ctx).mesh
    assert "badge" not in plain.groups
    assert body.connector("badge_front").tags == {"badge"}
    for name, ids in plain.groups.items():
        if name.startswith("slat_"):
            points = conn.frame.to_local(plain.vertices[ids])
            assert points[:, 0].min() < 0 < points[:, 0].max()  # no phantom badge hole
            front = points[np.isclose(points[:, 2], points[:, 2].max())]
            assert np.ptp(front[:, 1]) >= .5 * np.ptp(points[:, 1])  # not a knife-edge bevel
    decorated = get_component("grille.slats").build(conn, {"badge": True}, ctx).mesh
    rim = conn.frame.to_local(decorated.vertices[decorated.groups["badge_rim"]])
    radius = np.linalg.norm(rim[:, :2], axis=1).max()
    for name, ids in decorated.groups.items():
        if name.startswith("slat_"):
            points = conn.frame.to_local(decorated.vertices[ids])
            assert np.linalg.norm(points[:, :2], axis=1).min() >= radius + .006 - 1e-9


def test_rear_furniture_is_visible_on_the_new_bumper(fitted):
    body, ctx = fitted
    for side in ("L", "R"):
        for stem, component in (("rear_reflector", "light.rear_reflector"),
                                ("plate_lamp", "light.plate")):
            conn = body.connector(stem + "_" + side)
            mesh = get_component(component).build(conn, None, ctx).mesh
            assert conn.normal[0] < -.8
            xy = footprint(conn.meta["width"] * .75, conn.meta["height"] * .60)
            assert np.all(front_depths(mesh, conn.frame, xy)
                          > front_depths(body.mesh, conn.frame, xy) + .002)


def test_narrow_grille_default_spacing_preserves_explicit_counts(car_body, ctx):
    conn = car_body.build("sports").connector("grille")
    for options, expected in ((None, 2), ({"slats": 5}, 5), ({"slats": 14}, 14)):
        mesh = get_component("grille.slats").build(conn, options, ctx).mesh
        assert sum(name.startswith("slat_") for name in mesh.groups) == expected


def test_side_mirror_geometry_stays_symmetric_on_new_shoulders(fitted):
    body, ctx = fitted
    left = get_component("mirror.side").build(body.connector("mirror_L"), None, ctx).mesh
    right = get_component("mirror.side").build(body.connector("mirror_R"), None, ctx).mesh
    np.testing.assert_allclose(left.vertices * [1, -1, 1], right.vertices, atol=1e-10)
    for mesh in (left, right):
        assert np.all(mesh.face_normals()[mesh.zones["mirror_pane"], 0] < 0)
