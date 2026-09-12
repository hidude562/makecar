"""Tests for makecar.body: parameters, styles, the fixed-topology generator,
the morphable CarBody and the connectors it emits."""
from __future__ import annotations

import json

import numpy as np
import pytest

from makecar.body import BodyParams, BodyGenerator, style_params, style_names, CarBody, BodyResult
from makecar.body.generator import RING, RING_N, HALF_N, N_STATIONS, mirror_index, station_index
from makecar.body.targets import BODY_MODIFIER_SPECS, SCULPTS
from makecar.body.styles import STYLE_OVERRIDES
from makecar.connectors import PolygonConnector, RectangleConnector, CircleConnector, PointConnector

from conftest import signed_volume

STYLES = style_names()
EXPECTED_STYLES = {"sedan", "hatchback", "wagon", "suv", "pickup", "coupe", "sports", "van"}


@pytest.fixture(scope="module")
def style_meshes():
    return {s: BodyGenerator(style_params(s)).build() for s in STYLES}


def _spec(param):
    return next(s for s in BODY_MODIFIER_SPECS if (s.name or s.param) == param)


# ------------------------------------------------------------- parameters
class TestParamsAndStyles:
    def test_style_registry(self):
        assert set(STYLES) == EXPECTED_STYLES
        assert STYLES[0] == "sedan"
        assert style_params("sedan") == BodyParams()
        assert style_params("suv").wheelbase == STYLE_OVERRIDES["suv"]["wheelbase"]
        assert style_params("coupe").door_count == 2 and style_params("sedan").door_count == 4
        assert CarBody.styles() == STYLES

    def test_unknown_style_raises(self):
        with pytest.raises(KeyError):
            style_params("limousine")

    def test_style_params_on_custom_base(self):
        base = BodyParams(width=2.1)
        assert style_params("wagon", base).width == 2.1   # wagon does not override width
        assert style_params("suv", base).width == STYLE_OVERRIDES["suv"]["width"]

    def test_dict_roundtrip_and_validation(self):
        p = BodyParams(wheelbase=3.0)
        assert BodyParams.from_dict(p.to_dict()) == p
        with pytest.raises(KeyError, match="unknown body parameter"):
            BodyParams.from_dict({"wheelbase": 3.0, "spoiler": 1})
        nf = BodyParams.numeric_fields()
        assert "wheelbase" in nf and "door_count" not in nf

    def test_derived_quantities(self):
        p = BodyParams()
        assert p.length == pytest.approx(p.wheelbase + p.front_overhang + p.rear_overhang)
        assert p.arch_radius == pytest.approx(p.wheel_diameter / 2 + p.arch_gap)
        assert p.axle_height == pytest.approx(p.wheel_diameter / 2)

    @pytest.mark.parametrize("style", STYLES)
    def test_layout_chains_are_strictly_ordered(self, style):
        p = style_params(style)
        L = p.layout()
        upper, lower = np.asarray(L["upper"]), np.asarray(L["lower"])
        assert len(upper) == 10 and len(lower) == 8
        assert np.all(np.diff(upper) > 0)
        assert np.all(np.diff(lower) > 0)
        assert upper[0] == lower[0] == L["x_rear"]
        assert upper[-1] == lower[-1] == L["x_front"]
        assert L["x_fa"] == pytest.approx(p.wheelbase / 2) and L["x_ra"] == pytest.approx(-p.wheelbase / 2)
        assert L["u_cowl"] < L["u_front"] and L["u_deck"] > L["u_rear"]
        assert 0 < L["arch_half"] <= L["arch_radius"] + 1e-9

    def test_layout_enforces_gap_for_degenerate_input(self):
        # a huge windshield would push roof_front behind the B pillar; ordering must survive
        p = BodyParams(windshield_length=3.0, rear_window_length=3.0)
        upper = np.asarray(p.layout()["upper"])
        assert np.all(np.diff(upper) >= 0.02 - 1e-9)


# -------------------------------------------------------------- generator
class TestGenerator:
    def test_ring_layout_constants(self):
        assert RING["A"] == 0 and RING["H"] == HALF_N and RING_N == 2 * HALF_N
        assert mirror_index(0) == 0 and mirror_index(HALF_N) == HALF_N
        assert mirror_index(RING["E"]) == RING_N - RING["E"]
        assert station_index("rear") == 0 and station_index("front") == N_STATIONS - 1
        assert station_index("roof_rear") < station_index("roof_front") < station_index("cowl")

    def test_topology_is_identical_across_styles(self, style_meshes):
        ref = style_meshes["sedan"]
        for s, m in style_meshes.items():
            assert m.n_vertices == ref.n_vertices, s
            assert m.n_faces == ref.n_faces, s
            assert m.faces == ref.faces, s
            assert set(m.groups) == set(ref.groups), s
            for g in ref.groups:
                assert np.array_equal(m.groups[g], ref.groups[g]), (s, g)
        assert ref.n_vertices == ref.meta["n_rings"] * RING_N + 2
        assert ref.n_faces == ref.meta["n_loft_faces"] + 2 * RING_N

    @pytest.mark.parametrize("style", STYLES)
    def test_every_style_is_sane(self, style, style_meshes):
        m = style_meshes[style]
        assert not np.isnan(m.vertices).any()
        lo, hi = m.bounds()
        assert lo[2] >= 0.1
        assert 3.5 <= hi[0] - lo[0] <= 6.0
        assert 1.6 <= hi[1] - lo[1] <= 2.2
        assert 1.1 <= hi[2] <= 2.0
        assert abs(lo[1] + hi[1]) < 1e-6  # symmetric about the centre plane
        assert signed_volume(m) > 0

    def test_mesh_is_mirror_symmetric(self, style_meshes):
        m = style_meshes["sedan"]
        n_rings = m.meta["n_rings"]
        V = m.vertices[: n_rings * RING_N].reshape(n_rings, RING_N, 3)
        mirror = np.array([mirror_index(j) for j in range(RING_N)])
        assert np.allclose(V[:, :, 0], V[:, mirror, 0])
        assert np.allclose(V[:, :, 2], V[:, mirror, 2])
        assert np.allclose(V[:, :, 1], -V[:, mirror, 1])
        assert np.allclose(V[:, [0, HALF_N], 1], 0.0)
        assert np.allclose(m.vertices[-2:, 1], 0.0)  # apexes on the centre line

    def test_zones_and_materials(self, style_meshes):
        m = style_meshes["sedan"]
        apertures = {z for z in m.zones if z.startswith("aperture/")}
        assert apertures == {"aperture/windshield", "aperture/rear_window", "aperture/glass_front_L", "aperture/glass_front_R",
                             "aperture/glass_rear_L", "aperture/glass_rear_R", "aperture/headlight_L", "aperture/headlight_R",
                             "aperture/taillight_L", "aperture/taillight_R"}
        assert {"underbody", "wheel_well", "trim"} <= set(m.zones)
        for z, idx in m.zones.items():
            assert idx.min() >= 0 and idx.max() < m.n_faces
            assert len(np.unique(idx)) == len(idx)
        for z in apertures:
            assert all(m.face_materials[i] == "aperture" for i in m.zones[z])
        assert set(m.meta["aperture_rects"]) == apertures
        # wagons/vans/suvs have quarter windows, sedans do not
        assert "aperture/glass_quarter_L" in style_meshes["wagon"].zones
        assert "aperture/glass_quarter_L" not in m.zones
        assert "bed" in style_meshes["pickup"].zones and "bed" not in m.zones

    def test_face_normals_point_outward(self, style_meshes):
        m = style_meshes["sedan"]
        fn, fc = m.face_normals(), m.face_centroids()
        c = m.vertices.mean(axis=0)
        assert np.mean(np.sum(fn * (fc - c), axis=1) > 0) > 0.9

    def test_groups_reference_valid_vertices(self, style_meshes):
        m = style_meshes["sedan"]
        for g, idx in m.groups.items():
            assert idx.min() >= 0 and idx.max() < m.n_vertices, g
        assert len(m.groups["ring/cowl"]) == RING_N
        assert len(m.groups["arch_front_L"]) == station_index("fa_end", "lower") - station_index("fa_start", "lower") + 1
        assert len(m.groups["apex_front"]) == 1
        # arch lip vertices sit on a circle around the front axle
        pts = m.vertices[m.groups["arch_front_L"]]
        p = BodyParams()
        r = np.hypot(pts[:, 0] - p.wheelbase / 2, pts[:, 2] - p.axle_height)
        assert np.allclose(r, p.arch_radius, atol=1e-6)

    def test_generator_is_deterministic(self):
        a = BodyGenerator(BodyParams()).build()
        b = BodyGenerator(BodyParams()).build()
        assert np.array_equal(a.vertices, b.vertices) and a.faces == b.faces


# ---------------------------------------------------------------- CarBody
class TestCarBody:
    def test_library_modifiers(self, car_body):
        mods = set(car_body.library.modifiers)
        assert {s.name or s.param for s in BODY_MODIFIER_SPECS} <= mods
        assert {f"style/{s}" for s in STYLES if s != "sedan"} <= mods
        assert {f"sculpt/{s}" for s in SCULPTS} <= mods
        assert "style/sedan" not in mods
        names = [d["name"] for d in car_body.modifiers()]
        assert len(names) == len(set(names)) == len(mods)
        assert car_body.library.base.n_vertices == BodyGenerator(BodyParams()).build().n_vertices

    def test_build_result_structure(self, sedan):
        assert isinstance(sedan, BodyResult)
        assert sedan.mesh is not sedan.full_mesh
        assert sedan.modifier_values == {}
        assert sedan.base_params == BodyParams()
        assert sedan.hints["style"] == "sedan" and sedan.hints["door_count"] == 4
        assert sedan.hints["tire_width"] == pytest.approx(0.225) and sedan.hints["arch_gap"] == pytest.approx(0.06)
        assert {"paint", "aperture", "underbody", "trim", "bed"} <= set(sedan.mesh.materials)
        assert sedan.mesh.materials["paint"].color == pytest.approx((0x8a / 255, 0x1c / 255, 0x1c / 255))

    def test_measurements(self, sedan):
        m = sedan.measurements
        p = BodyParams()
        assert m["wheelbase"] == pytest.approx(p.wheelbase, abs=0.01)
        assert m["arch_front_r"] == pytest.approx(p.arch_radius, abs=0.005)
        assert m["arch_rear_r"] == pytest.approx(p.arch_radius, abs=0.005)
        assert m["wheel_front_z"] == pytest.approx(p.axle_height, abs=0.01)
        assert m["height"] == pytest.approx(p.roof_height, abs=0.01)
        assert m["width"] == pytest.approx(p.width, abs=0.05)
        assert m["length"] == pytest.approx(p.length, abs=0.1)
        assert m["track"] < m["width"]
        assert m["z_sill"] < m["z_belt"] < m["z_cowl"] < m["z_roof_front"]
        assert m["x_rear"] < m["x_deck"] < m["x_roof_rear"] < m["x_bpillar"] < m["x_roof_front"] < m["x_cowl"] < m["x_front"]
        assert m["has_bed"] == 0.0

    @pytest.mark.parametrize("modifier, value, key", [
        ("wheelbase", 1.0, "wheelbase"),
        ("wheelbase", -1.0, "wheelbase"),
        ("roof_height", -1.0, "height"),
        ("roof_height", 1.0, "height"),
        ("width", 1.0, "width"),
        ("width", -1.0, "width"),
    ])
    def test_modifiers_move_measurements_by_their_unit(self, car_body, sedan, modifier, value, key):
        spec = _spec(modifier)
        expected = value * (spec.delta_plus if value > 0 else spec.delta_minus)
        out = car_body.build(modifiers={modifier: value})
        assert out.measurements[key] - sedan.measurements[key] == pytest.approx(expected, abs=0.02)
        assert out.modifier_values == {modifier: value}

    def test_wheelbase_unit_is_the_documented_value(self, car_body, sedan):
        out = car_body.build(modifiers={"wheelbase": 1.0})
        assert out.measurements["wheelbase"] - sedan.measurements["wheelbase"] == pytest.approx(0.55, abs=0.01)
        low = car_body.build(modifiers={"roof_height": -1.0})
        assert low.measurements["height"] - sedan.measurements["height"] == pytest.approx(-0.15, abs=0.01)

    def test_half_slider_is_half_displacement(self, car_body, sedan):
        base = sedan.full_mesh.vertices
        full = car_body.build(modifiers={"wheelbase": 1.0}).full_mesh.vertices - base
        half = car_body.build(modifiers={"wheelbase": 0.5}).full_mesh.vertices - base
        assert np.allclose(half, 0.5 * full, atol=1e-9)

    @pytest.mark.parametrize("style", STYLES)
    def test_style_macros_reproduce_generator_geometry(self, car_body, style):
        out = car_body.build(style=style)
        gen = BodyGenerator(style_params(style)).build()
        assert out.full_mesh.n_vertices == gen.n_vertices
        assert np.allclose(out.full_mesh.vertices, gen.vertices, atol=1e-6)
        assert out.hints["style"] == style
        assert out.hints["door_count"] == style_params(style).door_count

    def test_style_blend_is_linear(self, car_body, sedan):
        base = sedan.full_mesh.vertices
        suv = car_body.build(style="suv").full_mesh.vertices
        half = car_body.build(style={"suv": 0.5}).full_mesh.vertices
        assert np.allclose(half, 0.5 * (base + suv), atol=1e-9)
        blend = car_body.build(style={"suv": 0.5, "coupe": 0.25})
        assert blend.hints["style"] == "suv"  # dominant style drives the hints
        assert set(blend.modifier_values) == {"style/suv", "style/coupe"}
        assert car_body.build(style="sedan").modifier_values == {}

    def test_sculpt_targets(self, car_body, sedan):
        out = car_body.build(sculpt={"haunches": 1.0})
        d = out.full_mesh.vertices - sedan.full_mesh.vertices
        assert np.abs(d[:, 1]).max() > 0.01
        assert np.allclose(d[:, [0, 2]], 0.0)          # haunches only push sideways
        assert out.modifier_values == {"sculpt/haunches": 1.0}
        assert car_body.build(sculpt={"sculpt/wedge": -0.5}).modifier_values == {"sculpt/wedge": -0.5}

    def test_unknown_names_raise_keyerror(self, car_body):
        with pytest.raises(KeyError):
            car_body.build(modifiers={"turbo": 1.0})
        with pytest.raises(KeyError):
            car_body.build(style="limousine")
        with pytest.raises(KeyError):
            car_body.build(style={"suv": 0.5, "boat": 0.5})
        with pytest.raises(KeyError):
            car_body.build(sculpt={"spoiler": 1.0})
        with pytest.raises(KeyError):
            car_body.build().connector("nonexistent")

    def test_build_is_deterministic_and_leaves_library_untouched(self, car_body, sedan):
        before = car_body.library.base.vertices.copy()
        again = car_body.build()
        assert np.array_equal(again.full_mesh.vertices, sedan.full_mesh.vertices)
        assert np.array_equal(car_body.library.base.vertices, before)
        assert [c.name for c in again.connectors] == [c.name for c in sedan.connectors]

    def test_paint_option_and_hints_passthrough(self, car_body):
        out = car_body.build(paint="#00ff00", hints={"drive": "right", "fuel_side": "right"})
        assert out.mesh.materials["paint"].color == pytest.approx((0, 1, 0))
        assert out.connector("fuel_cap").origin[1] < 0
        assert out.connector("steering_wheel").origin[1] < 0
        assert out.connector("seat_front_driver").origin[1] < 0


# ------------------------------------------------------- apertures / seams
class TestShellAndSeams:
    def test_apertures_removed_from_shell(self, sedan):
        full, shell = sedan.full_mesh, sedan.mesh
        n_ap = sum(len(full.zones[z]) for z in full.zones if z.startswith("aperture/"))
        assert n_ap > 0
        assert shell.n_faces == full.n_faces - n_ap
        assert not any(z.startswith("aperture/") for z in shell.zones)
        assert "aperture" not in set(shell.face_materials)
        assert "aperture" in set(full.face_materials)
        for z, idx in shell.zones.items():
            assert idx.max() < shell.n_faces, z
        assert shell.n_vertices == full.n_vertices  # vertices are kept for the connectors
        # the aperture outlines show up as open boundary loops in the shell (adjacent
        # apertures that share a vertex chain merge into one hole, so count <= 10)
        loops = shell.boundary_loops()
        assert 1 <= len(loops) <= 10
        boundary_pts = shell.vertices[sorted({v for loop in loops for v in loop})]
        for c in sedan.connectors:
            if c.meta.get("aperture"):
                d = np.linalg.norm(boundary_pts[None, :, :] - c.points[:, None, :], axis=2).min(axis=1)
                assert np.all(d < 1e-9), c.name

    def test_seams_are_present(self, sedan):
        for m in (sedan.mesh, sedan.full_mesh):
            assert len(m.lines) > 0
            assert len(m.lines) == len(m.line_names)
            for pts, name in zip(m.lines, m.line_names):
                assert pts.ndim == 2 and pts.shape[1] == 3 and len(pts) >= 2, name
                assert not np.isnan(pts).any()
        names = set(sedan.mesh.line_names)
        assert {"hood_seam_L", "hood_seam_R", "deck_seam_L", "front_door_seam_L", "rear_door_seam_R", "hood_rear_seam"} <= names

    def test_coupe_has_no_rear_door_seams_or_handles(self, car_body):
        coupe = car_body.build(style="coupe")
        assert coupe.hints["door_count"] == 2
        assert not any(n.startswith("rear_door_seam") for n in coupe.mesh.line_names)
        names = {c.name for c in coupe.connectors}
        assert "handle_front_L" in names and "handle_rear_L" not in names
        assert "door_card_front_L" in names and "door_card_rear_L" not in names


# --------------------------------------------------------------- connectors
class TestConnectors:
    def test_wheels(self, sedan):
        wheels = [c for c in sedan.connectors if "wheel" in c.tags]
        assert len(wheels) == 4
        assert {c.name for c in wheels} == {"wheel_front_L", "wheel_front_R", "wheel_rear_L", "wheel_rear_R"}
        gap = sedan.hints["arch_gap"]
        for c in wheels:
            assert isinstance(c, CircleConnector)
            assert c.radius == pytest.approx(c.meta["arch_radius"] - gap, abs=1e-6)
            assert c.radius == pytest.approx(BodyParams().wheel_diameter / 2, abs=0.01)
            ys = 1.0 if c.name.endswith("_L") else -1.0
            assert np.allclose(c.normal, [0, ys, 0], atol=1e-9)
            assert c.meta["side"] == ("left" if ys > 0 else "right")
            assert c.meta["tire_width"] == pytest.approx(sedan.hints["tire_width"])
            assert c.origin[2] == pytest.approx(sedan.measurements[f"wheel_{c.meta['axle']}_z"])
        L = sedan.connector("wheel_front_L")
        R = sedan.connector("wheel_front_R")
        assert np.allclose(L.origin * [1, -1, 1], R.origin)
        assert sedan.connector("wheel_front_L").origin[0] - sedan.connector("wheel_rear_L").origin[0] == pytest.approx(
            sedan.measurements["wheelbase"])

    def test_apertures(self, sedan):
        aps = [c for c in sedan.connectors if c.meta.get("aperture")]
        assert len(aps) == 10
        lo, hi = sedan.full_mesh.bounds()
        centre = (lo + hi) / 2
        for c in aps:
            assert type(c) is PolygonConnector
            rows, cols = c.meta["grid"]
            assert rows >= 2 and cols >= 2
            assert len(c.points) == 2 * (rows - 1) + 2 * (cols - 1), c.name
            assert len(np.unique(np.round(c.points, 9), axis=0)) == len(c.points), c.name
            assert ("glass" in c.tags) != ("light" in c.tags)
            assert np.dot(c.normal, c.origin - centre) > 0, c.name  # outward
            assert c.area > 0.02
            assert c.meta["side"] in ("left", "right", "center")
        assert "windshield" in sedan.connector("windshield").tags
        assert "headlight" in sedan.connector("headlight_L").tags
        assert sedan.connector("windshield").meta["side"] == "center"
        assert sedan.connector("glass_front_L").meta["side"] == "left"

    def test_aperture_points_lie_on_the_body_surface(self, sedan):
        verts = sedan.full_mesh.vertices
        for name in ("windshield", "glass_front_L", "headlight_R"):
            pts = sedan.connector(name).points
            d = np.linalg.norm(verts[None, :, :] - pts[:, None, :], axis=2).min(axis=1)
            assert np.all(d < 1e-9), name

    def test_interior_connectors_present_for_sedan(self, sedan):
        names = {c.name for c in sedan.connectors}
        expected = {"seat_front_driver", "seat_front_passenger", "seat_row2", "steering_wheel", "dashboard", "pedals",
                    "console", "floor", "headliner", "door_card_front_L", "door_card_front_R", "parcel_shelf",
                    "trunk_floor", "firewall", "rear_bulkhead", "rearview_mirror"}
        assert expected <= names
        assert "bed_floor" not in names
        assert isinstance(sedan.connector("steering_wheel"), CircleConnector)
        assert isinstance(sedan.connector("dashboard"), PolygonConnector)
        assert isinstance(sedan.connector("seat_front_driver"), RectangleConnector)
        driver = sedan.connector("seat_front_driver")
        assert driver.origin[1] > 0 and sedan.connector("seat_front_passenger").origin[1] < 0  # left-hand drive
        assert np.allclose(driver.normal, [0, 0, 1])
        assert np.allclose(sedan.connector("floor").normal, [0, 0, 1])
        wheel = sedan.connector("steering_wheel")
        assert wheel.normal[0] < 0 and wheel.normal[2] > 0  # tilted towards the driver

    def test_exterior_connectors_present(self, sedan):
        names = {c.name for c in sedan.connectors}
        assert {"grille", "intake", "plate_front", "plate_rear", "badge_front", "badge_rear", "exhaust_L", "exhaust_R",
                "mirror_L", "mirror_R", "handle_front_L", "handle_rear_R", "fuel_cap", "antenna"} <= names
        assert len(names) == len(sedan.connectors)  # unique names
        assert sedan.connector("grille").normal[0] > 0.9
        assert sedan.connector("plate_rear").normal[0] < -0.9
        assert isinstance(sedan.connector("badge_front"), PointConnector)
        assert sedan.connector("antenna").origin[2] > 1.3

    def test_pickup_has_bed_floor(self, pickup):
        names = {c.name for c in pickup.connectors}
        assert "bed_floor" in names
        assert "parcel_shelf" not in names and "trunk_floor" not in names
        bed = pickup.connector("bed_floor")
        assert isinstance(bed, RectangleConnector)
        assert "bed" in bed.tags and np.allclose(bed.normal, [0, 0, 1])
        assert pickup.measurements["has_bed"] == 1.0
        assert bed.origin[2] < pickup.measurements["z_deck"] - 0.3

    def test_connectors_inside_body_bounds(self, sedan, pickup):
        for res in (sedan, pickup):
            lo, hi = res.full_mesh.bounds()
            for c in res.connectors:
                assert np.all(c.origin >= lo - 0.3) and np.all(c.origin <= hi + 0.3), c.name
                if isinstance(c, PolygonConnector):
                    assert np.all(c.points >= lo - 0.3) and np.all(c.points <= hi + 0.3), c.name

    def test_connectors_are_json_serialisable_and_well_formed(self, sedan):
        for c in sedan.connectors:
            d = c.to_dict()
            json.dumps(d)
            assert d["owner"] == "body" and d["name"] == c.name
            assert np.allclose(np.linalg.norm(c.frame.rotation, axis=0), 1.0)
            assert np.linalg.det(c.frame.rotation) > 0

    def test_left_right_pairs_are_mirror_images(self, sedan):
        for base in ("wheel_front", "headlight", "mirror", "handle_front", "door_card_front", "exhaust"):
            L, R = sedan.connector(f"{base}_L"), sedan.connector(f"{base}_R")
            assert np.allclose(L.origin * [1, -1, 1], R.origin, atol=1e-6), base
            assert np.allclose(L.normal * [1, -1, 1], R.normal, atol=1e-6), base
            assert type(L) is type(R)


# ------------------------------------------------------------- known bugs
class TestKnownBugs:
    """Regression tests for connector defects found during development: the upper
    chain is numbered rear -> front, which the range code originally inverted."""

    def test_roof_rail_connectors_are_emitted(self, sedan):
        names = {c.name for c in sedan.connectors}
        assert {"roof_rail_L", "roof_rail_R"} <= names

    def test_headliner_grid_matches_points(self, sedan):
        hl = sedan.connector("headliner")
        rows, cols = hl.meta["grid"]
        assert rows > 1 and cols > 1
        assert len(hl.points) == 2 * (rows - 1) + 2 * (cols - 1)

    def test_door_card_points_match_grid(self, sedan):
        dc = sedan.connector("door_card_front_L")
        rows, cols = dc.meta["grid"]
        assert len(dc.points) == 2 * (rows - 1) + 2 * (cols - 1)
        assert len(np.unique(np.round(dc.points, 9), axis=0)) == len(dc.points)
