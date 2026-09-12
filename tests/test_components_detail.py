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
