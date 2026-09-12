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


@pytest.mark.parametrize("options", [{"aspect_ratio": .35}, {"aspect_ratio": 100}, {"aspect_ratio": float("nan")},
                                      {"rim_ratio": 1.}, {"tread_pattern": "unknown"},
                                      {"spoke_family": "unknown"}, {"disc_pattern": "unknown"}])
def test_invalid_sizes_and_families_fail_clearly(ctx, options):
    with pytest.raises(ValueError):
        build(ctx, **options)
