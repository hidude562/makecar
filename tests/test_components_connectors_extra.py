"""Exterior detail connector contracts, including world-space routed parts."""
import json
from itertools import product

import numpy as np
import pytest

from makecar.assembly import assemble
from makecar.body import style_names
from makecar.body.connectors import aperture_loop, measure
from makecar.body.connectors_extra import extra_exterior_connectors
from makecar.body.shapes import SHAPE_DELTAS
from makecar.components import BuildContext, Palette, default_component_for, get_component
from makecar.connectors import PointConnector, RectangleConnector


MIRROR = np.array([1.0, -1.0, 1.0])
POINT_TAGS = {
    "fog_front_L": "fog", "fog_front_R": "fog",
    "tow_hook_cover_front": "tow_hook_cover", "tow_hook_cover_rear": "tow_hook_cover",
    "rear_reflector_L": "rear_reflector", "rear_reflector_R": "rear_reflector",
    "rear_fog_L": "rear_aux", "rear_reverse_R": "rear_aux",
    "plate_lamp_L": "plate_lamp", "plate_lamp_R": "plate_lamp",
    "side_marker_L": "side_marker", "side_marker_R": "side_marker",
    "exhaust_system": "exhaust_system", "tow_eye": "tow_eye",
}


@pytest.fixture(scope="module", params=style_names())
def detailed_body(request, car_body):
    return car_body.build(style=request.param, hints={"mud_flaps": True})


def details(body):
    return {c.name: c for c in extra_exterior_connectors(body.full_mesh, body.measurements, body.hints)}


def assert_frame(connector):
    rotation = connector.frame.rotation
    assert np.isfinite(connector.origin).all()
    assert np.isfinite(rotation).all()
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    if isinstance(connector, RectangleConnector):
        assert connector.width > 0 and connector.height > 0 and connector.area > 0
        assert np.isfinite(connector.points).all()
    for key in ("width", "height", "depth", "pipe_radius"):
        if key in connector.meta:
            assert np.isfinite(connector.meta[key]) and connector.meta[key] > 0


def test_extra_names_types_frames_and_serialization(detailed_body):
    connectors = details(detailed_body)
    for name, tag in POINT_TAGS.items():
        assert type(connectors[name]) is PointConnector
        assert tag in connectors[name].tags
    assert connectors["rear_fog_L"].meta["kind"] == "fog"
    assert connectors["rear_reverse_R"].meta["kind"] == "reverse"
    for name in ("wipers", "mud_flap_front_L", "mud_flap_front_R", "mud_flap_rear_L", "mud_flap_rear_R"):
        assert isinstance(connectors[name], RectangleConnector)
    assert ("diffuser" in connectors) == (detailed_body.hints["style"] in ("sports", "coupe"))
    for connector in connectors.values():
        assert_frame(connector)
    # Paths must survive connector JSON export, including nested lists.
    json.dumps([c.to_dict() for c in connectors.values()], allow_nan=False)
    all_names = [c.name for c in detailed_body.connectors]
    assert len(all_names) == len(set(all_names))


def test_extra_mirror_symmetry_and_mount_directions(detailed_body):
    connectors = details(detailed_body)
    for stem in ("fog_front", "rear_reflector", "plate_lamp", "side_marker", "mud_flap_front", "mud_flap_rear"):
        left, right = connectors[stem + "_L"], connectors[stem + "_R"]
        np.testing.assert_allclose(left.origin * MIRROR, right.origin, atol=1e-8)
        np.testing.assert_allclose(left.normal * MIRROR, right.normal, atol=1e-8)
        assert left.side() == "left" and right.side() == "right"
    for name in ("fog_front_L", "fog_front_R", "tow_hook_cover_front"):
        assert connectors[name].normal[0] > 0.8
        assert connectors[name].frame.y_axis[2] > 0.8
    for name in ("rear_reflector_L", "rear_fog_L", "rear_reverse_R", "plate_lamp_L", "tow_eye"):
        assert connectors[name].normal[0] < -0.8
        assert connectors[name].frame.y_axis[2] > 0.8
    for side, sign in (("L", 1), ("R", -1)):
        marker = connectors["side_marker_" + side]
        assert sign * marker.normal[1] > 0.7
        lip = detailed_body.full_mesh.vertices[detailed_body.full_mesh.groups["arch_front_" + side]]
        assert marker.origin[0] > lip[:, 0].max()
        lamp = connectors["plate_lamp_" + side]
        plate = detailed_body.connector("plate_rear")
        assert lamp.origin[2] > plate.points[:, 2].max() + 0.015
        for axle in ("front", "rear"):
            flap = connectors[f"mud_flap_{axle}_{side}"]
            assert flap.normal[0] < -0.99
            assert flap.points[:, 2].min() >= 0.07
            assert flap.width == pytest.approx(detailed_body.hints["tire_width"] + 0.025)
            lip = detailed_body.full_mesh.vertices[detailed_body.full_mesh.groups[f"arch_{axle}_{side}"]]
            assert flap.origin[0] < lip[:, 0].min()


def assert_wiper_grid_fit(body):
    wipers = body.connector("wipers")
    paths = np.asarray(wipers.meta["paths"])
    normals = np.asarray(wipers.meta["blade_normals"])
    uv_paths = np.asarray(wipers.meta["path_uv"])
    assert paths.shape == normals.shape == (2, 13, 3)
    assert wipers.meta["paths_space"] == "world"
    lengths = np.linalg.norm(np.diff(paths, axis=1), axis=2).sum(axis=1)
    assert np.all((lengths > 0.4) & (lengths < 0.6))
    np.testing.assert_allclose(paths[0] * MIRROR, paths[1], atol=1e-8)
    np.testing.assert_allclose(normals[0] * MIRROR, normals[1], atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=2), 1.0, atol=1e-10)
    assert np.all(normals[:, :, 2] > 0.3)
    assert wipers.normal[0] > 0 and wipers.normal[2] > 0.3
    assert abs(wipers.normal[1]) < 1e-8
    _, _, _, grid = aperture_loop(body.full_mesh, "aperture/windshield")
    for path, uv, ns in zip(paths, uv_paths, normals):
        assert np.all((uv > 0) & (uv < 1))
        r, c = (uv * (np.array(grid.shape[:2]) - 1)).T
        i, j = r.astype(int), c.astype(int)
        a, b = (r - i)[:, None], (c - j)[:, None]
        expected = ((1-a)*(1-b)*grid[i, j] + a*(1-b)*grid[i+1, j]
                    + (1-a)*b*grid[i, j+1] + a*b*grid[i+1, j+1])
        np.testing.assert_allclose(path - ns * wipers.meta["blade_clearance"], expected, atol=1e-10)
        du = (1-b)*(grid[i+1, j]-grid[i, j]) + b*(grid[i+1, j+1]-grid[i, j+1])
        dv = (1-a)*(grid[i, j+1]-grid[i, j]) + a*(grid[i+1, j+1]-grid[i+1, j])
        np.testing.assert_allclose(np.sum(ns * du, axis=1), 0, atol=1e-10)
        np.testing.assert_allclose(np.sum(ns * dv, axis=1), 0, atol=1e-10)
    pivots = np.asarray(wipers.meta["pivots"])
    np.testing.assert_allclose(pivots[0] * MIRROR, pivots[1], atol=1e-8)
    assert np.all(pivots[:, 0] > paths[:, :, 0].max(axis=1))


def test_wipers_use_exact_windshield_grid(detailed_body):
    assert_wiper_grid_fit(detailed_body)


def test_exhaust_routes_to_existing_tips(detailed_body):
    exhaust = detailed_body.connector("exhaust_system")
    paths = [np.asarray(p) for p in exhaust.meta["paths"]]
    assert len(paths) == 3 and exhaust.meta["paths_space"] == "world"
    assert exhaust.origin[0] == pytest.approx(0.5 * (detailed_body.measurements["wheel_front_x"]
                                                   + detailed_body.measurements["wheel_rear_x"]))
    assert exhaust.origin[1] == 0 and exhaust.normal[2] == -1
    for path in paths:
        assert np.isfinite(path).all() and path[:, 2].min() > 0.04
        assert np.all(np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-5)
    assert paths[0][0, 0] > detailed_body.measurements["wheel_front_x"]
    floor = detailed_body.full_mesh.vertices[detailed_body.full_mesh.groups["line/floor"]]
    np.testing.assert_allclose(paths[0][:, 2], np.interp(paths[0][:, 0], floor[:, 0], floor[:, 2]) - 0.033)
    np.testing.assert_allclose(paths[1] * MIRROR, paths[2], atol=1e-8)
    for branch, side, tip_meta in zip(paths[1:], ("L", "R"), exhaust.meta["tips"]):
        tip = detailed_body.connector("exhaust_" + side)
        np.testing.assert_allclose(branch[0], paths[0][-1], atol=1e-10)
        np.testing.assert_allclose(branch[-1], tip.origin, atol=1e-10)
        np.testing.assert_allclose(tip_meta, tip.origin, atol=1e-10)
        tangent = branch[-1] - branch[-2]
        np.testing.assert_allclose(tangent / np.linalg.norm(tangent), tip.normal, atol=1e-10)
    assert np.linalg.norm(np.vstack(paths).mean(axis=0) - exhaust.origin) < 0.5
    if detailed_body.hints["style"] in ("sports", "coupe"):
        diffuser = detailed_body.connector("diffuser")
        assert diffuser.normal[2] < -0.5
        assert diffuser.points[:, 2].min() > 0.06


def test_mud_flaps_are_opt_in(sedan):
    assert not any("mud_flap" in c.tags for c in sedan.connectors)
    enabled = extra_exterior_connectors(sedan.full_mesh, sedan.measurements, {**sedan.hints, "mud_flaps": True})
    assert len([c for c in enabled if "mud_flap" in c.tags]) == 4


@pytest.mark.parametrize("style,expected", [({"sedan": 0.6, "sports": 0.4}, True),
                                           ({"coupe": 0.3, "wagon": 0.7}, True),
                                           ({"sedan": 1, "sports": 0}, False),
                                           ({"suv": 0.5, "van": 0.5}, False)])
def test_diffuser_accepts_blended_hints(sedan, style, expected):
    connectors = extra_exterior_connectors(sedan.full_mesh, sedan.measurements, {"style": style})
    assert any(c.name == "diffuser" for c in connectors) == expected


def test_connectors_follow_morphed_vertices(car_body, sedan):
    body = car_body.build(style={"coupe": 0.6, "sedan": 0.4}, modifiers={
        "width": 0.7, "wheelbase": 0.6, "ground_clearance": 0.5, "windshield_length": -0.5,
        "front_fascia_rake": 0.8, "rear_fascia_rake": -0.7}, sculpt={"wedge": 0.5})
    assert_wiper_grid_fit(body)
    for c in details(body).values():
        assert_frame(c)
    assert not np.allclose(body.connector("wipers").meta["paths"], sedan.connector("wipers").meta["paths"])
    # Translation of the *mesh*, without parameters, moves every emitted detail.
    shifted = body.full_mesh.copy()
    delta = np.array([0.27, 0, 0.12])
    shifted.vertices += delta
    moved = extra_exterior_connectors(shifted, measure(shifted), body.hints)
    originals = details(body)
    for c in moved:
        np.testing.assert_allclose(c.origin, originals[c.name].origin + delta, atol=1e-8)
        if "paths" in c.meta:
            for path, original in zip(c.meta["paths"], originals[c.name].meta["paths"]):
                np.testing.assert_allclose(path, np.asarray(original) + delta, atol=1e-8)
        if "floor_grid" in c.meta:
            np.testing.assert_allclose(c.meta["floor_grid"],
                                       np.asarray(originals[c.name].meta["floor_grid"]) + delta, atol=1e-8)


# Component acceptance uses each style's actual measurements/hints, not the
# shared sedan BuildContext fixture. Existing wheel/lamp/mirror/grille detail
# tests live in test_components_detail.py; only integrated budgets touch them.
NEW_DEFAULTS = {
    "fog": "light.fog", "tow_hook_cover": "bumper.tow_cover",
    "rear_reflector": "light.rear_reflector", "rear_aux": "light.rear_aux",
    "plate_lamp": "light.plate", "side_marker": "light.side_marker",
    "wipers": "wipers.parked", "exhaust_system": "exhaust.system",
    "diffuser": "bumper.diffuser", "mud_flap": "mud_flap.standard",
    "tow_eye": "bumper.tow_eye",
}


@pytest.fixture(scope="module")
def detailed_assembly(detailed_body):
    return assemble(detailed_body)


def instance(assembly, name):
    return next(i for i in assembly.instances if i.connector.name == name)


def assert_closed_solid(mesh, label):
    """Check winding of a solid, including welded primitive seam/cap vertices.

    A positive mount-normal dot is NOT meaningful for a solid's back/sides.
    Instead require closed consistently oriented edges and positive volume.
    """
    triangles, _ = mesh.triangulated()
    assert len(triangles), label
    a, b, c = (mesh.vertices[triangles[:, k]] for k in range(3))
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    assert np.all(area2 > 2e-12), label
    _, welded = np.unique(np.round(mesh.vertices, 10), axis=0, return_inverse=True)
    faces = welded[triangles]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, edge_ids, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    assert np.all(counts == 2), (label, "open/nonmanifold solid")
    orientation = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    assert np.all(np.bincount(edge_ids, weights=orientation) == 0), (label, "inconsistent winding")
    # Recenter for numerical stability when checking tiny prisms far from zero.
    q = mesh.vertices - mesh.vertices.mean(axis=0)
    a, b, c = (q[triangles[:, k]] for k in range(3))
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    assert volume > 1e-12, (label, "inward or collapsed", volume)


def test_new_default_components_have_complete_finite_solids(detailed_assembly):
    found = set()
    for inst in detailed_assembly.instances:
        tags = inst.connector.tags & NEW_DEFAULTS.keys()
        if not tags:
            continue
        tag, = tags
        found.add(tag)
        assert default_component_for(inst.connector) == inst.component == NEW_DEFAULTS[tag]
        mesh = inst.result.mesh
        label = (detailed_assembly.body.hints["style"], inst.connector.name, inst.component)
        assert mesh.n_faces > 0 and mesh.n_vertices > 0, label
        assert np.isfinite(mesh.vertices).all(), label
        assert mesh.vertices[:, 2].min() > 0.01, (label, "ground intersection")
        assert set(mesh.face_materials) <= mesh.materials.keys(), label
        assert len(mesh.face_materials) == mesh.n_faces, label
        for mat in mesh.materials.values():
            assert np.isfinite(mat.color).all() and np.isfinite(mat.alpha), label
        covered = set()
        for name, ids in mesh.zones.items():
            assert_closed_solid(mesh.subset(ids), (label, name))
            covered.update(map(int, ids))
        # Some components start with an unlabelled plinth/undertray before
        # adding labelled subparts. Check that solid too, not only its screws.
        remainder = sorted(set(range(mesh.n_faces)) - covered)
        if remainder:
            assert_closed_solid(mesh.subset(remainder), (label, "base"))
    expected = set(NEW_DEFAULTS)
    if detailed_assembly.body.hints["style"] not in ("sports", "coupe"):
        expected.remove("diffuser")
    assert found == expected


def test_all_style_face_budgets_and_hint_opt_in_assembly(detailed_assembly):
    wheels = [i for i in detailed_assembly.instances if "wheel" in i.connector.tags]
    assert len(wheels) == 4
    assert sum(i.result.mesh.n_faces for i in wheels) < 24000
    total_faces = detailed_assembly.body.mesh.n_faces + sum(i.result.mesh.n_faces for i in detailed_assembly.instances)
    assert total_faces < 120000, (detailed_assembly.body.hints["style"], total_faces)
    flaps = [i for i in detailed_assembly.instances if "mud_flap" in i.connector.tags]
    assert detailed_assembly.body.hints["mud_flaps"] is True
    assert len(flaps) == 4 and all(i.component == "mud_flap.standard" for i in flaps)
    assert detailed_assembly.missing() == []
    for flap in flaps:
        assert {"rivet_0", "rivet_1"} <= flap.result.mesh.groups.keys()
        assert flap.result.mesh.vertices[:, 2].min() > 0.07


def swept_rings(mesh, group, path, sides):
    """Recover sweep cross-sections, excluding the two cap-centre vertices."""
    points = mesh.vertices[mesh.groups[group]]
    assert len(points) == len(path) * sides + 2
    rings = points[:-2].reshape(len(path), sides, 3)
    np.testing.assert_allclose(rings.mean(axis=1), path, atol=1e-9)
    np.testing.assert_allclose(points[-2:], np.asarray(path)[[0, -1]], atol=1e-9)
    return rings


def test_two_wiper_blades_fit_exact_glass_and_arms(detailed_assembly):
    inst = instance(detailed_assembly, "wipers")
    conn, mesh = inst.connector, inst.result.mesh
    names = {k for k in mesh.groups if k.startswith("blade_") and k[6:].isdigit()}
    assert names == {"blade_0", "blade_1"}
    assert {"arm_0", "arm_1", "pivot_0", "pivot_1", "blade_spine_0", "blade_spine_1"} <= mesh.groups.keys()
    for k, (path, normals, pivot) in enumerate(zip(conn.meta["paths"], conn.meta["blade_normals"], conn.meta["pivots"])):
        path, normals = np.asarray(path), np.asarray(normals)
        rings = swept_rings(mesh, f"blade_{k}", path, 6)
        np.testing.assert_allclose(np.linalg.norm(rings - path[:, None], axis=2), .0035, atol=1e-9)
        # Connector tests above independently recover the exact windshield grid.
        # The rubber is only 2.5–9.5mm above it, not a floating straight chord.
        surface = path - conn.meta["blade_clearance"] * normals
        distance = np.einsum("nki,ni->nk", rings - surface[:, None], normals)
        assert distance.min() >= .0025 - 1e-9 and distance.max() <= .0095 + 1e-9
        assert .4 < np.linalg.norm(np.diff(rings.mean(axis=1), axis=0), axis=1).sum() < .6
        swept_rings(mesh, f"blade_spine_{k}", path + .005 * normals, 6)
        arm_pts = mesh.vertices[mesh.groups[f"arm_{k}"]]
        np.testing.assert_allclose(arm_pts[-2], pivot, atol=1e-9)
        middle = len(path) // 2
        np.testing.assert_allclose(arm_pts[-1], path[middle] + .011 * normals[middle], atol=1e-9)
        pivot_pts = mesh.vertices[mesh.groups[f"pivot_{k}"]]
        np.testing.assert_allclose(pivot_pts.mean(axis=0), pivot, atol=1e-9)


def test_exhaust_geometry_is_continuous_and_meets_tips(detailed_assembly):
    inst = instance(detailed_assembly, "exhaust_system")
    conn, mesh = inst.connector, inst.result.mesh
    paths = [np.asarray(p) for p in conn.meta["paths"]]
    assert {k for k in mesh.groups if k.startswith("pipe_run_")} == {"pipe_run_0", "pipe_run_1", "pipe_run_2"}
    assert {"catalyst", "silencer"} <= mesh.groups.keys()
    for name in ("catalyst", "silencer"):
        clearance = mesh.vertices[mesh.groups[name], 2].min()
        assert clearance >= .05 - 1e-8, (detailed_assembly.body.hints["style"], name, clearance)
    for k, path in enumerate(paths):
        rings = swept_rings(mesh, f"pipe_run_{k}", path, 10)
        np.testing.assert_allclose(np.linalg.norm(rings - path[:, None], axis=2), conn.meta["pipe_radius"], atol=1e-9)
        # Every consecutive sample is joined by a full section of actual faces;
        # a disconnected set of short cylinders would fail this adjacency test.
        ids = mesh.groups[f"pipe_run_{k}"]
        quads = {frozenset(mesh.faces[fi]) for fi in mesh.zones[f"pipe_run_{k}"] if len(mesh.faces[fi]) == 4}
        assert len(quads) == (len(path) - 1) * 10
        for sample in range(len(path) - 1):
            for j in range(10):
                next_j = (j + 1) % 10
                expected = frozenset((ids[sample*10+j], ids[sample*10+next_j],
                                      ids[(sample+1)*10+j], ids[(sample+1)*10+next_j]))
                assert expected in quads, ("discontinuous exhaust", k, sample, j)
    for k, side in enumerate(("L", "R"), 1):
        np.testing.assert_allclose(paths[k][0], paths[0][-1], atol=1e-9)
        tip = instance(detailed_assembly, "exhaust_" + side).connector
        np.testing.assert_allclose(paths[k][-1], tip.origin, atol=1e-9)
        ring = mesh.vertices[mesh.groups[f"pipe_run_{k}"]][(len(paths[k])-1)*10:len(paths[k])*10]
        np.testing.assert_allclose((ring - tip.origin) @ tip.normal, 0, atol=1e-9)
        assert np.max(np.linalg.norm(ring - tip.origin, axis=1)) < tip.radius
    assert np.linalg.norm(mesh.vertices.mean(axis=0) - conn.origin) < .5


def underbody_clearance(body, points, *, whole_shell=False):
    """Vertical clearance to actual pan triangles, independent of route metadata."""
    shell = body.full_mesh if whole_shell else body.full_mesh.subset(body.full_mesh.zones["underbody"])
    triangles, _ = shell.triangulated()
    a, b, c = shell.vertices[triangles].transpose(1, 0, 2)
    ab, ac = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    den = np.where(abs(det) > 1e-12, det, 1.)
    clearances = []
    for p in points:
        q = p[:2] - a[:, :2]
        u = (q[:, 0] * ac[:, 1] - q[:, 1] * ac[:, 0]) / den
        v = (ab[:, 0] * q[:, 1] - ab[:, 1] * q[:, 0]) / den
        valid = (abs(det) > 1e-12) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        if whole_shell and not valid.any():
            # Exterior tip mouths extend aft of the shell's XY footprint.
            clearances.append(np.inf)
            continue
        assert valid.any(), ("no floor above exhaust", p)
        heights = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
        clearances.append(heights[valid].min() - p[2])
    return np.asarray(clearances)


def assert_exhaust_pan_fit(body):
    conn = body.connector("exhaust_system")
    ctx = BuildContext(body.measurements, body.hints, Palette(), np.random.default_rng(0))
    mesh = get_component("exhaust.system").build(conn, {}, ctx).mesh
    # Both full-length can skins and the tunnel pipe must remain outside the
    # actual pan; merely checking their ground clearance hid crown penetration.
    for name in ("catalyst", "silencer", "pipe_run_0"):
        part = mesh.subset(mesh.zones[name])
        edges = np.array([(a, b) for face in part.faces for a, b in zip(face, face[1:] + face[:1])])
        samples = np.vstack([part.vertices, part.vertices[edges].mean(axis=1)])
        assert underbody_clearance(body, samples).min() >= .004 - 1e-8, name
        if name != "pipe_run_0":
            assert part.vertices[:, 2].min() >= .05 - 1e-8, name
            assert np.ptp(part.vertices[:, 2]) > .02, "can must not collapse"
            assert_closed_solid(part, name)
    for k in (1, 2):
        path = np.asarray(conn.meta["paths"][k])
        rings = swept_rings(mesh, f"pipe_run_{k}", path, 10)
        # The whole branch stays below the pan until its short rise inside the
        # rear bumper: checking only the turn missed an early floor penetration.
        last = len(path) - 3
        samples = np.vstack([rings[:last + 1].reshape(-1, 3),
                             ((rings[:last] + rings[1:last + 1]) / 2).reshape(-1, 3)])
        assert underbody_clearance(body, samples).min() >= .004 - 1e-8, k
        # Below-valance outlets no longer need to punch through the rear floor:
        # extend the same clearance requirement through the complete terminal.
        whole_branch = np.vstack([rings.reshape(-1, 3),
                                  ((rings[:-1] + rings[1:]) / 2).reshape(-1, 3)])
        assert underbody_clearance(body, whole_branch, whole_shell=True).min() >= .004 - 1e-8, k


@pytest.mark.parametrize("dual,length", [(False, .08), (True, .08), (True, .14)])
def test_exhaust_outlets_fit_below_actual_valance(detailed_body, dual, length):
    body = detailed_body
    ctx = BuildContext(body.measurements, body.hints, Palette(), np.random.default_rng(0))
    for side in ("L", "R"):
        conn = body.connector("exhaust_" + side)
        mesh = get_component("exhaust.tip").build(conn, {"dual": dual, "length": length}, ctx).mesh
        # A full-shell downward-ray envelope catches buried upper rims and
        # wrongly stacked twins, including the inlet portions inside the skirt.
        clearance = underbody_clearance(body, mesh.vertices, whole_shell=True)
        assert np.isfinite(clearance).any(), "tip inlet must remain beneath the car"
        assert clearance.min() >= .001 - 1e-8
        assert mesh.vertices[:, 2].min() >= .05
        local = conn.frame.to_local(mesh.vertices)
        assert np.ptp(local[:, 1]) == pytest.approx(2 * conn.radius)
        assert np.ptp(local[:, 0]) == pytest.approx(2 * conn.radius * (2.2 if dual else 1))
        assert local[:, 2].max() == pytest.approx(length - .03)
        inlet = mesh.subset(mesh.zones["collector" if dual else "inlet"])
        assert_closed_solid(inlet, "outlet inlet connection")
        if dual:
            # Its wider shared inlet spans the original central system endpoint.
            lp = conn.frame.to_local(inlet.vertices)
            assert lp[:, 2].min() < -.03 < 0 < lp[:, 2].max()
            assert lp[:, 1].min() < -.025 and lp[:, 1].max() > .025


def test_exhaust_cans_and_branch_turn_clear_current_pan(detailed_body):
    assert_exhaust_pan_fit(detailed_body)


@pytest.mark.parametrize("style,modifiers", [
    ("sports", {"ground_clearance": -1}),
    ("sedan", {"tunnel_height": 1, "tunnel_width": -1}),
    ({"coupe": .6, "wagon": .4}, {"wheelbase": .7, "rear_fascia_rake": -.8,
                                  "ground_clearance": -.4, "tunnel_height": -.7}),
])
def test_exhaust_pan_and_tip_fit_follow_morphs(car_body, style, modifiers):
    body = car_body.build(style=style, modifiers=modifiers)
    assert_exhaust_pan_fit(body)
    conn = body.connector("exhaust_system")
    for path, tip_meta, side in zip(conn.meta["paths"][1:], conn.meta["tips"], ("L", "R")):
        tip = body.connector("exhaust_" + side)
        np.testing.assert_allclose(path[-1], tip.origin, atol=1e-9)
        np.testing.assert_allclose(tip_meta, tip.origin, atol=1e-9)
        tangent = np.subtract(path[-1], path[-2])
        np.testing.assert_allclose(tangent / np.linalg.norm(tangent), tip.normal, atol=1e-9)
    ctx = BuildContext(body.measurements, body.hints, Palette(), np.random.default_rng(0))
    pipe_only = get_component("exhaust.system").build(conn, {"muffler": False}, ctx).mesh
    assert not {"catalyst", "silencer"} & pipe_only.groups.keys()
    assert {"pipe_run_0", "pipe_run_1", "pipe_run_2"} <= pipe_only.groups.keys()


def test_exhaust_bends_do_not_fold_between_close_rings(detailed_body):
    conn = detailed_body.connector("exhaust_system")
    ctx = BuildContext(detailed_body.measurements, detailed_body.hints, Palette(), np.random.default_rng(0))
    mesh = get_component("exhaust.system").build(conn, {"muffler": False}, ctx).mesh
    for k, path in enumerate(conn.meta["paths"]):
        path = np.asarray(path)
        rings = swept_rings(mesh, f"pipe_run_{k}", path, 10)
        advance = np.einsum("nki,ni->nk", rings[1:] - rings[:-1], np.diff(path, axis=0))
        assert advance.min() > 0, (detailed_body.hints["style"], k, "folded sweep faces")


@pytest.mark.parametrize("style", style_names())
@pytest.mark.parametrize("shape,weight", product(SHAPE_DELTAS, (0., 1.)))
def test_exhaust_fits_inherited_archetype_endpoints(car_body, style, shape, weight):
    assert_exhaust_pan_fit(car_body.build(style=style, modifiers={shape: weight}))


_SHAPE_CORNERS = list(product(*[[name for name in SHAPE_DELTAS if name.startswith(family + "/")]
                               for family in ("face", "plan", "section")]))


@pytest.mark.parametrize("style", style_names())
@pytest.mark.parametrize("shapes", _SHAPE_CORNERS, ids=lambda names: "+".join(names))
def test_exhaust_fits_full_strength_archetype_corners(car_body, style, shapes):
    # Explicitly remove inherited weights: setting a single new member to one
    # would normalize against a style's defaults and miss the true endpoints.
    modifiers = {name: float(name in shapes) for name in SHAPE_DELTAS}
    body = car_body.build(style=style, modifiers=modifiers)
    assert_exhaust_pan_fit(body)
    conn = body.connector("exhaust_system")
    ctx = BuildContext(body.measurements, body.hints, Palette(), np.random.default_rng(0))
    mesh = get_component("exhaust.system").build(conn, {"muffler": False}, ctx).mesh
    for k, path in enumerate(conn.meta["paths"]):
        path = np.asarray(path)
        rings = swept_rings(mesh, f"pipe_run_{k}", path, 10)
        np.testing.assert_allclose(np.linalg.norm(rings - path[:, None], axis=2), .025, atol=1e-9)
        advance = np.einsum("nki,ni->nk", rings[1:] - rings[:-1], np.diff(path, axis=0))
        assert advance.min() > 0, (style, shapes, k, "folded sweep faces")
        assert_closed_solid(mesh.subset(mesh.zones[f"pipe_run_{k}"]), (style, shapes, k))
        assert rings[:, :, 2].min() > 0
        if k:
            tip = body.connector("exhaust_" + ("L" if k == 1 else "R"))
            np.testing.assert_allclose(path[0], conn.meta["paths"][0][-1], atol=1e-9)
            np.testing.assert_allclose(path[-1], tip.origin, atol=1e-9)
            np.testing.assert_allclose((rings[-1] - tip.origin) @ tip.normal, 0, atol=1e-9)


def test_exhaust_floor_envelope_catches_dips_between_rings():
    from makecar.body.connectors_extra import _ExhaustFloor

    # Neither endpoint nor the midpoint sees this narrow low floor station.
    grid = np.array([[[x, y, z] for y in (0., .4)] for x, z in
                     ((0., .2), (.21, .2), (.23, .15), (.25, .2), (1., .2))])
    floor = _ExhaustFloor(grid)
    path = np.array([[.9, .1, .167], [.1, .1, .167]])
    assert floor.clearance(*path, .025) == pytest.approx(-.042)
    fitted, = floor.fit_paths([path], .025)
    assert floor.clearance(*np.asarray(fitted), .025) >= .004
    np.testing.assert_allclose(np.asarray(fitted)[:, :2], path[:, :2])


def test_exhaust_floor_envelope_checks_lateral_radius_and_slopes():
    from makecar.body.connectors_extra import _ExhaustFloor

    # The centreline fits; the outboard edge clips the tunnel shoulder.
    grid = np.array([[[x, y, z + .1 * x] for y, z in
                      ((0., .2), (.08, .2), (.12, .14), (.4, .14))] for x in (0., 1.)])
    floor = _ExhaustFloor(grid)
    path = np.array([[.9, .075, .257], [.1, .075, .177]])
    assert floor.clearance(*path, .025) < 0
    fitted, = floor.fit_paths([path], .025)
    assert floor.clearance(*np.asarray(fitted), .025) >= .004
    mirrored = path * MIRROR
    np.testing.assert_allclose(floor.fit_paths([mirrored], .025)[0], np.asarray(fitted) * MIRROR)


def test_exhaust_floor_checks_lateral_segments_and_reuses_xy_clipping():
    from makecar.body.connectors_extra import _ExhaustFloor

    grid = np.array([[[x, y, .2] for y in (0., .4)] for x in (0., 1.)])
    floor = _ExhaustFloor(grid)
    path = np.array([[.5, .1, .195], [.5, .2, .195]])
    assert floor.clearance(*path, .025) == pytest.approx(-.02)
    fitted, = floor.fit_paths([path], .025)
    assert floor.clearance(*np.asarray(fitted), .025) >= .004
    assert len(floor._footprints) == 1, "height fitting must reuse its invariant XY footprint"


def test_exhaust_floor_does_not_accept_a_blocked_fixed_collar():
    from makecar.body.connectors_extra import _ExhaustFloor

    grid = np.array([[[x, y, .2] for y in (0., .4)] for x in (0., 1.)])
    paths = [[[.9, 0., .167], [.6, 0., .167]],
             [[.6, 0., .167], [.3, .1, .195], [.2, .1, .195]]]
    with pytest.raises(ValueError, match="fixed tip collar"):
        _ExhaustFloor(grid).fit_paths(paths, .025)


def test_diffuser_has_real_tapered_fins_only_on_sporty_styles(detailed_assembly):
    sporty = detailed_assembly.body.hints["style"] in ("sports", "coupe")
    matches = [i for i in detailed_assembly.instances if "diffuser" in i.connector.tags]
    assert len(matches) == int(sporty)
    if not sporty:
        return
    inst = matches[0]
    conn, mesh = inst.connector, inst.result.mesh
    assert {k for k in mesh.groups if k.startswith("fin_")} == {f"fin_{k}" for k in range(5)}
    centres = []
    for k in range(5):
        fin = conn.frame.to_local(mesh.vertices[mesh.groups[f"fin_{k}"]])
        assert np.ptp(fin[:, 0]) == pytest.approx(.008)
        assert np.ptp(fin[:, 1]) == pytest.approx(conn.height)
        assert fin[:, 2].max() == pytest.approx(conn.meta["depth"])
        rear, front = fin[np.isclose(fin[:, 1], -conn.height / 2)], fin[np.isclose(fin[:, 1], conn.height / 2)]
        assert rear[:, 2].max() > front[:, 2].max() + .02
        centres.append(fin[:, 0].mean())
    assert np.all(np.diff(centres) > .10)
    body = detailed_assembly.body
    ctx = BuildContext(body.measurements, body.hints, Palette(), np.random.default_rng(0))
    seven = get_component("bumper.diffuser").build(conn, {"fins": 7}, ctx).mesh
    assert len([k for k in seven.groups if k.startswith("fin_")]) == 7
    assert seven.n_faces > mesh.n_faces


def test_small_lamp_lenses_colors_and_plate_aim(detailed_assembly):
    for name in ("rear_fog_L", "rear_reverse_R", "rear_reflector_L", "side_marker_L"):
        inst = instance(detailed_assembly, name)
        mesh = inst.result.mesh
        assert {"plinth", "lens"} <= mesh.groups.keys()
        assert len([k for k in mesh.groups if k.startswith("prism_")]) >= 3
        material = mesh.face_materials[mesh.zones["lens"][0]]
        r, g, b = mesh.materials[material].color
        if name == "rear_reverse_R":
            assert min(r, g, b) > .7
        elif name == "side_marker_L":
            assert r > g > b
        else:
            assert r > 2 * max(g, b)
    for side in ("L", "R"):
        inst = instance(detailed_assembly, "plate_lamp_" + side)
        lens = inst.result.mesh.subset(inst.result.mesh.zones["lens"])
        normals = lens.face_normals()
        front = normals[normals @ inst.connector.normal > .75]
        assert len(front) > 0
        assert np.all(front[:, 2] < -.4), "plate-lamp lens must aim DOWN at the plate"


def test_mud_flap_hint_roundtrips_through_config():
    from makecar.config import CarConfig
    from makecar.pipeline import build_car

    cfg = CarConfig.from_dict({"body": {"style": "suv", "hints": {"mud_flaps": True}}})
    _, assembly, _ = build_car(cfg)
    assert len([i for i in assembly.instances if i.component == "mud_flap.standard"]) == 4
    cfg = CarConfig.from_dict({"body": {"style": "suv", "hints": {"mud_flaps": False}}})
    _, assembly, _ = build_car(cfg)
    assert not any(i.component == "mud_flap.standard" for i in assembly.instances)
