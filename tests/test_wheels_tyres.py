"""Physical wheel dimensions, open topology and inspectable detail contracts."""
import itertools

import numpy as np
import pytest

from makecar.components.exterior import Wheel, WheelParams
from makecar.connectors import CircleConnector
from makecar.geometry.frame import Frame


def build(ctx, radius=.33, width=.225, frame=None, **options):
    conn = CircleConnector("hub", frame or Frame.identity(), radius, meta={"tire_width": width})
    return Wheel().build(conn, options, ctx)


@pytest.mark.parametrize("width,aspect,inches", [(.245, 35, 20), (.205, 65, 15), (.305, 30, 21)])
def test_standard_tyre_sizes_fit_the_actual_bead_seat(ctx, width, aspect, inches):
    rr = inches * .0254 / 2
    radius = rr + width * aspect / 100
    result = build(ctx, radius, width, aspect_ratio=aspect)
    m = result.mesh
    assert result.info["sidewall_height"] == pytest.approx(width * aspect / 100)
    assert result.info["rim_diameter_inches"] == pytest.approx(inches)
    carcass = m.vertices[m.groups["carcass"]]
    assert np.linalg.norm(carcass[:, :2], axis=1).min() == pytest.approx(rr)
    assert np.ptp(carcass[:, 2]) == pytest.approx(width)
    bead = carcass[np.isclose(np.linalg.norm(carcass[:, :2], axis=1), rr)]
    assert np.abs(bead[:, 2]) == pytest.approx(np.full(len(bead), width * .4))
    assert np.linalg.norm(m.vertices[:, :2], axis=1).max() == pytest.approx(radius)


def test_aspect_ratio_changes_sidewall_not_connector_envelope(ctx):
    low = build(ctx, aspect_ratio=35)
    high = build(ctx, aspect_ratio=65)
    assert high.info["sidewall_height"] / low.info["sidewall_height"] == pytest.approx(65 / 35)
    assert high.info["rim_radius"] < low.info["rim_radius"]
    for res in (low, high):
        assert np.ptp(res.mesh.vertices[:, 2]) == pytest.approx(.225)
        assert np.linalg.norm(res.mesh.vertices[:, :2], axis=1).max() == pytest.approx(.33)


@pytest.mark.parametrize("side", [-1, 1])
def test_staggered_axles_fit_their_own_connectors_without_slider_clipping(ctx, side):
    for x, track, radius, width in [(1.4, 1.58, .32, .205), (-1.4, 1.72, .49, .365)]:
        frame = Frame.from_normal([x, side * track / 2, radius], [0, side, 0], [1, 0, 0])
        res = build(ctx, radius, width, frame, aspect_ratio=40)
        v = frame.to_local(res.mesh.vertices)
        assert np.linalg.norm(v[:, :2], axis=1).max() == pytest.approx(radius)
        assert np.ptp(v[:, 2]) == pytest.approx(width)
        assert res.info["rim_radius"] == pytest.approx(radius - width * .4)
        assert np.mean([v[:, 2].min(), v[:, 2].max()]) == pytest.approx(0, abs=1e-12)
        assert res.mesh.vertices[:, 2].min() >= -.005


@pytest.mark.parametrize("pattern", Wheel.TREAD_PATTERNS)
def test_tread_family_has_real_radial_relief_or_a_smooth_slick(ctx, pattern):
    result = build(ctx, tread_pattern=pattern, groove_depth=.009)
    mesh = result.mesh
    blocks = [k for k in mesh.groups if k.startswith("tread_block_")]
    if pattern == "slick":
        assert not blocks
        assert not any(k.startswith("shoulder_") for k in mesh.groups)
        assert result.info["groove_depth"] == 0
        assert np.linalg.norm(mesh.vertices[mesh.groups["carcass"]][:, :2], axis=1).max() == pytest.approx(.33)
    else:
        assert blocks
        block = mesh.vertices[mesh.groups[blocks[0]]]
        assert np.ptp(np.linalg.norm(block[:, :2], axis=1)) == pytest.approx(.009)
        assert any(k.startswith("shoulder_") for k in mesh.groups)


def test_zero_depth_does_not_leave_floating_blocks(ctx):
    res = build(ctx, groove_depth=0)
    assert not any(k.startswith("tread_block_") for k in res.mesh.groups)
    assert np.linalg.norm(res.mesh.vertices[:, :2], axis=1).max() == pytest.approx(.33)


def test_spoke_families_have_distinct_counts_and_connectivity(ctx):
    meshes = [build(ctx, spoke_family=family).mesh for family in Wheel.SPOKE_FAMILIES]
    assert len({m.n_faces for m in meshes}) == len(Wheel.SPOKE_FAMILIES)
    assert len({tuple(m.faces) for m in meshes}) == len(Wheel.SPOKE_FAMILIES)
    for family, m in zip(Wheel.SPOKE_FAMILIES, meshes):
        assert np.isfinite(m.vertices).all()
        assert m.n_faces * 4 < 40000
        if family not in ("steel_cap", "dish"):
            spoke = m.vertices[m.groups["spoke_0"]][:-2].reshape(5, 8, 3)
            assert np.ptp(spoke[:, :, 2]) > .025
            # Octagonal sections have a narrower cast front face and edge breaks.
            assert len(m.zones["spoke_0"]) == 48


def test_convex_and_concave_faces_move_the_hub_not_the_rim(ctx):
    convex, concave = [build(ctx, dish=d).mesh for d in (-.02, .08)]
    for mesh in (convex, concave):
        assert np.linalg.norm(mesh.vertices[:, :2], axis=1).max() == pytest.approx(.33)
    assert convex.vertices[convex.groups["spoke_0"]][0, 2] > concave.vertices[concave.groups["spoke_0"]][0, 2] + .09
    assert np.allclose(convex.vertices[convex.groups["barrel"]], concave.vertices[concave.groups["barrel"]])


def _depths(mesh, xy):
    triangles, _ = mesh.triangulated()
    a, b, c = mesh.vertices[triangles].transpose(1, 0, 2)
    ab, ac, q = b - a, c - a, np.r_[xy, 0.] - a
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    safe = np.where(np.abs(det) > 1e-12, det, 1.)
    u = (q[:, 0] * ac[:, 1] - q[:, 1] * ac[:, 0]) / safe
    v = (ab[:, 0] * q[:, 1] - ab[:, 1] * q[:, 0]) / safe
    hit = (np.abs(det) > 1e-12) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    return np.unique(np.round((a[:, 2] + u * ab[:, 2] + v * ac[:, 2])[hit], 10))


@pytest.mark.parametrize("dish", [-.02, .04, .09])
@pytest.mark.parametrize("radius,width,aspect", [(.33, .225, 55), (.34, .245, 35), (.4, .305, 65)])
def test_caliper_straddles_the_disc_with_actual_triangle_clearance(ctx, dish, radius, width, aspect):
    m = build(ctx, radius, width, aspect_ratio=aspect, dish=dish).mesh
    rotor = m.vertices[m.groups["brake_disc"]]
    lo, hi = rotor[:, 2].min(), rotor[:, 2].max()
    rd = np.linalg.norm(rotor[:, :2], axis=1).max()
    assert m.vertices[m.groups["caliper_cheek_1"]][:, 2].min() > hi + .004
    assert m.vertices[m.groups["caliper_cheek_-1"]][:, 2].max() < lo - .004
    bridge = m.subset(m.zones["caliper_bridge"])
    assert bridge.vertices[:, 2].min() < lo and bridge.vertices[:, 2].max() > hi
    # The closest point of every projected triangle edge also clears the
    # rotor, guarding against a coarse chord cutting through the swept disc.
    triangles, _ = bridge.triangulated()
    pts = bridge.vertices[triangles][:, :, :2]
    for j in range(3):
        a, b = pts[:, j], pts[:, (j + 1) % 3]
        ab = b - a
        t = np.clip(-np.sum(a * ab, axis=1) / np.maximum(np.sum(ab * ab, axis=1), 1e-20), 0, 1)
        assert np.linalg.norm(a + t[:, None] * ab, axis=1).min() > rd + .003
    assert m.vertices[m.groups["brake_pad_1"]][:, 2].min() > hi + .0005
    assert m.vertices[m.groups["brake_pad_-1"]][:, 2].max() < lo - .0005
    caliper = m.vertices[m.groups["caliper"]]
    spokes = np.concatenate([m.vertices[v] for k, v in m.groups.items() if k.startswith("spoke_")])
    assert caliper[:, 2].max() < spokes[:, 2].min() - .006
    assert m.vertices[m.groups["dust_shield"]][:, 2].max() < lo


@pytest.mark.parametrize("pattern", ["plain", "drilled", "slotted"])
def test_drilling_is_through_geometry_and_slots_have_recessed_floors(ctx, pattern):
    m = build(ctx, disc_pattern=pattern).mesh
    disc = m.subset(m.zones["brake_disc"])
    rd = np.linalg.norm(disc.vertices[:, :2], axis=1).max()
    inner = .066
    center = (inner + rd) / 2
    if pattern == "drilled":
        for row in range(2):
            radius = inner + (rd - inner) * (row + .5) / 2
            for k in range(12):
                a = 2 * np.pi * k / 12 + row * np.pi / 12
                assert len(_depths(disc, radius * np.array([np.cos(a), np.sin(a)]))) == 0
    else:
        hits = _depths(disc, [center, 0])
        assert len(hits) == 4  # two genuine friction plates, not one thick disc
        if pattern == "slotted":
            assert hits[-1] == pytest.approx(disc.vertices[:, 2].max() - .0015)
            assert hits[0] == pytest.approx(disc.vertices[:, 2].min() + .0015)
        else:
            assert hits[-1] == pytest.approx(disc.vertices[:, 2].max())
    # The vent vanes span just the open space between the plates.
    vane = m.vertices[m.groups["disc_vane_0"]]
    front = m.vertices[m.groups["disc_face_1"]]
    back = m.vertices[m.groups["disc_face_-1"]]
    assert vane[:, 2].min() == pytest.approx(back[:, 2].max())
    assert vane[:, 2].max() == pytest.approx(front[:, 2].min())


def test_lug_heads_are_recessed_inside_open_conical_seats(ctx):
    m = build(ctx).mesh
    seats = m.subset(m.zones["lug_seats"])
    for k in range(5):
        a = 2 * np.pi * k / 5
        xy = .0465 * np.array([np.cos(a), np.sin(a)])
        assert len(_depths(seats, xy)) == 0
        lug = m.vertices[m.groups[f"lug_{k}"]]
        assert lug[:, 2].max() < seats.vertices[:, 2].max() - .003
        assert np.linalg.norm(lug[:, :2] - xy, axis=1).max() < .006


def test_expensive_detail_can_be_disabled_without_changing_fit(ctx):
    full = build(ctx)
    cheap = build(ctx, tread_blocks=0, shoulder_blocks=0, disc_vanes=0)
    assert cheap.mesh.n_faces < full.mesh.n_faces - 1000
    assert not any(k.startswith(("tread_block_", "shoulder_", "disc_vane_")) for k in cheap.mesh.groups)
    for key in ("rim_radius", "radius", "section_width", "sidewall_height"):
        assert full.info[key] == cheap.info[key]


def _assert_closed_solid(mesh):
    from makecar.geometry import primitives as P
    triangles, owners = mesh.triangulated()
    xyz = mesh.vertices[triangles]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    assert np.linalg.norm(cross, axis=1).min() > 1e-12
    assert np.all(np.sum(cross * mesh.face_normals()[owners], axis=1) > 0)
    _, inverse = np.unique(np.round(mesh.vertices, 10), axis=0, return_inverse=True)
    welded = inverse[triangles]
    edges = np.concatenate([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert np.all(counts == 2)
    # Each directed use must have one opposite use, not just incidence two.
    directed = set(map(tuple, edges))
    assert all((b, a) in directed for a, b in directed)
    assert P.signed_volume(mesh) > 0


@pytest.mark.parametrize("pattern", ["plain", "drilled", "slotted"])
def test_brake_parts_are_closed_outward_solids_without_internal_band_walls(ctx, pattern):
    m = build(ctx, disc_pattern=pattern).mesh
    for name in ("disc_face_1", "disc_face_-1", "caliper_cheek_1", "caliper_cheek_-1",
                 "caliper_bridge", "caliper_boss_0", "caliper_boss_1", "pad_backing_1", "brake_pad_-1",
                 "dust_shield", "lug_seats", "barrel"):
        _assert_closed_solid(m.subset(m.zones[name]))
    shield = m.vertices[m.groups["dust_shield"]]
    assert np.abs(np.arctan2(shield[:, 1], -shield[:, 0])).min() >= .47 - 1e-10


@pytest.mark.parametrize("pattern", Wheel.TREAD_PATTERNS[:-1])
def test_both_shoulder_sides_and_spokes_have_outward_winding(ctx, pattern):
    m = build(ctx, tread_pattern=pattern).mesh
    for name in ("shoulder_-1_0", "shoulder_1_0", "tread_block_0_0", "spoke_0"):
        _assert_closed_solid(m.subset(m.zones[name]))


@pytest.mark.parametrize("dish,radius,width", itertools.product([-.025, .04, .09], [.23, .49], [.145, .365]))
def test_aero_dish_is_a_constant_thickness_graph_at_all_offsets(ctx, dish, radius, width):
    m = build(ctx, radius, width, dish=dish, spoke_family="dish").mesh
    profile = m.vertices[m.groups["aero_dish"]][:9]
    assert np.all(np.diff(np.linalg.norm(profile[:4, :2], axis=1)) > 0)
    assert profile[:4, 2] - profile[4:8, 2][::-1] == pytest.approx(np.full(4, .012))
    _assert_closed_solid(m.subset(m.zones["aero_dish"]))


def test_rib_crest_matches_carcass_sampling_and_remains_visible(ctx):
    m = build(ctx).mesh
    carcass = m.vertices[m.groups["carcass"]]
    rib = m.vertices[m.groups["tread_rib_0"]]
    crest = rib[np.isclose(np.linalg.norm(rib[:, :2], axis=1), .327)]
    assert len(np.unique(np.round(np.arctan2(crest[:, 1], crest[:, 0]), 9))) >= Wheel.N
    assert np.linalg.norm(crest[:, :2], axis=1).min() > .326
    mesh = m.subset(m.zones["tread_rib_0"])
    triangles, owner = mesh.triangulated()
    a, b, c = mesh.vertices[triangles].transpose(1, 0, 2)
    assert np.all(np.sum(np.cross(b - a, c - a) * mesh.face_normals()[owner], axis=1) > 1e-12)
    # Both surfaces use the same angular stations, so every intervening crest
    # chord stays outside the tread bed instead of vanishing at its midpoint.
    angles = np.unique(np.round(np.mod(np.arctan2(crest[:, 1], crest[:, 0]), 2 * np.pi), 9))
    bed = carcass[np.isclose(np.linalg.norm(carcass[:, :2], axis=1), .326)]
    bed_angles = np.unique(np.round(np.mod(np.arctan2(bed[:, 1], bed[:, 0]), 2 * np.pi), 9))
    assert np.array_equal(angles, bed_angles)


@pytest.mark.parametrize("pattern", Wheel.TREAD_PATTERNS)
def test_low_detail_preserves_actual_outer_radius(ctx, pattern):
    m = build(ctx, tread_pattern=pattern, tread_blocks=0, shoulder_blocks=0, disc_vanes=0).mesh
    assert np.linalg.norm(m.vertices[:, :2], axis=1).max() == pytest.approx(.33)


def test_optional_lettering_is_shallow_rubber_geometry(ctx):
    base, relief = build(ctx), build(ctx, lettering=True)
    assert "sidewall_brand" not in base.mesh.groups
    for name in ("sidewall_brand", "sidewall_size"):
        m = relief.mesh.subset(relief.mesh.zones[name])
        assert set(m.face_materials) == {"tyre"}
        assert m.n_faces > 100
        assert m.vertices[:, 2].max() <= .225 / 2 + .0008
    assert relief.mesh.n_faces * 4 < 40000


@pytest.mark.parametrize("options", [{"aspect_ratio": .35}, {"aspect_ratio": 100}, {"aspect_ratio": float("nan")},
                                      {"rim_ratio": 1.}, {"tread_pattern": "unknown"},
                                      {"spoke_family": "unknown"}, {"disc_pattern": "unknown"}])
def test_invalid_sizes_and_families_fail_clearly(ctx, options):
    with pytest.raises(ValueError):
        build(ctx, **options)
