"""Exterior archetype geometry and additive-target semantics.

Directions are measured on generated vertices, not merely on parameter deltas.
Target interpolation is tested separately: interpolating parameter values is not
in general equivalent to interpolating the resulting lofts. Reuse the session
CarBody; only a handful of integration checks need connector emission.
"""
from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from makecar.body import BodyGenerator, BodyParams, style_names, style_params
from makecar.body.generator import RING, RING_N, mirror_index, station_index
from makecar.body.shapes import (
    SHAPE_DELTAS,
    SHAPE_FAMILIES,
    normalize_shape_values,
    resolve_shape_values,
    shape_params,
)
from makecar.body.styles import STYLE_SHAPE_DEFAULTS
from makecar.body import targets
from makecar.morph.morphable import MorphableMesh


EXPECTED_SHAPES = {
    "face/wedge", "face/upright", "face/shark", "face/snub",
    "face/cabforward", "face/longhood", "plan/pointed", "plan/square",
    "plan/cokebottle", "section/tumblehome", "section/slabside", "section/domed",
}
SHAPES = sorted(EXPECTED_SHAPES)
STYLES = style_names()


@pytest.fixture(scope="module")
def archetype_meshes():
    base = BodyParams()
    return {name: BodyGenerator(shape_params(base, {name: 1})).build()
            for name in ["neutral", *SHAPES]}


def group(mesh, name):
    return mesh.vertices[mesh.groups[name]]


def hood_metrics(mesh):
    """Actual cowl-to-leading-edge centreline, including the fascia transition."""
    hood = group(mesh, "line/center_top")[station_index("cowl"):]
    dx = hood[-1, 0] - hood[0, 0]
    t = (hood[:, 0] - hood[0, 0]) / dx
    chord = hood[0, 2] + t * (hood[-1, 2] - hood[0, 2])
    return (hood[-1, 2] - hood[0, 2]) / dx, np.max(np.abs(hood[:, 2] - chord))


def hood_run(mesh):
    return (group(mesh, "fascia/front_face")[RING["H"], 0]
            - group(mesh, "ring/cowl")[RING["H"], 0])


def glass_lean(mesh):
    ring = group(mesh, "ring/bp_f")
    base, rail = ring[[RING["F"], RING["G"]]]
    return (base[1] - rail[1]) / (rail[2] - base[2])


class TestArchetypeDirections:
    def test_wedge_lowers_tip_and_makes_hood_steeper_and_straighter(self, archetype_meshes):
        base, wedge = (archetype_meshes[n] for n in ("neutral", "face/wedge"))
        before, after = (group(m, "fascia/front_face") for m in (base, wedge))
        assert after[RING["H"], 2] < before[RING["H"], 2] - 0.10
        slope0, error0 = hood_metrics(base)
        slope1, error1 = hood_metrics(wedge)
        hood = group(wedge, "line/center_top")[station_index("cowl"):]
        assert np.all(np.diff(hood[:, 0]) > 0), "hood folds backward immediately behind its leading edge"
        assert slope1 < 1.5 * slope0 < 0
        assert error1 < error0 * 0.5, "wedge hood centreline bends at the last loft stations"
        assert error1 < 0.008
        # The low upper face must retreat behind the forward splitter.
        assert after[RING["A"], 0] - after[RING["H"], 0] > 0.10

    def test_upright_has_high_flat_hood_and_vertical_tall_face(self, archetype_meshes):
        base, upright = (archetype_meshes[n] for n in ("neutral", "face/upright"))
        before, after = (group(m, "fascia/front_face") for m in (base, upright))
        assert after[RING["H"], 2] > before[RING["H"], 2] + 0.10
        assert np.ptp(after[:, 2]) > np.ptp(before[:, 2]) + 0.10
        rake0 = abs(before[RING["D"], 0] - before[RING["H"], 0])
        rake1 = abs(after[RING["D"], 0] - after[RING["H"], 0])
        assert rake1 < min(0.015, rake0 * 0.5)
        slope, error = hood_metrics(upright)
        assert abs(slope) < 0.05
        assert error < hood_metrics(base)[1] * 0.5

    def test_shark_projects_and_drops_center_below_proud_fenders(self, archetype_meshes):
        base, shark = (archetype_meshes[n] for n in ("neutral", "face/shark"))
        before, after = (group(m, "fascia/front_face") for m in (base, shark))
        tip0, tip1 = before[RING["H"]], after[RING["H"]]
        fender0, fender1 = before[RING["G"]], after[RING["G"]]
        assert tip1[0] > tip0[0] + 0.005, "centre extension must exceed the upper fascia setback"
        assert tip1[2] < tip0[2] - 0.07
        assert fender1[2] > fender0[2] + 0.02
        assert fender1[2] - tip1[2] > 0.08
        assert tip1[0] > fender1[0] + 0.02
        # Lower grille projects ahead of its upper edge, independently of the beak.
        assert after[RING["D"], 0] > after[RING["F"], 0]

    def test_snub_shortens_and_raises_hood_with_larger_corners(self, archetype_meshes):
        base, snub = (archetype_meshes[n] for n in ("neutral", "face/snub"))
        before, after = (group(m, "fascia/front_face") for m in (base, snub))
        assert hood_run(snub) < hood_run(base) * 0.8
        assert after[RING["H"], 2] > before[RING["H"], 2] + 0.10
        # The top-to-side return radius is measured on the front elevation.
        radius0 = np.max(before[:, 1]) - before[RING["G"], 1]
        radius1 = np.max(after[:, 1]) - after[RING["G"], 1]
        assert radius1 > radius0 * 2
        # In plan the rounded corner starts farther behind the bumper.
        g0 = BodyGenerator(BodyParams())
        g1 = BodyGenerator(shape_params(BodyParams(), {"face/snub": 1}))
        x = g0.L["x_front"] - 0.25
        assert g1.plan_half_width(x) < g0.plan_half_width(x) - 0.01

    def test_cabforward_moves_cowl_forward_and_nearly_eliminates_hood(self, archetype_meshes):
        base, cab = (archetype_meshes[n] for n in ("neutral", "face/cabforward"))
        cowl0, cowl1 = (group(m, "ring/cowl")[RING["H"]] for m in (base, cab))
        assert cowl1[0] > cowl0[0] + 0.5
        assert hood_run(cab) < hood_run(base) * 0.55
        windscreen0 = cowl0[0] - group(base, "ring/roof_front")[RING["H"], 0]
        windscreen1 = cowl1[0] - group(cab, "ring/roof_front")[RING["H"], 0]
        assert windscreen1 > windscreen0 + 0.3

    def test_longhood_moves_cowl_back_and_extends_high_flat_plane(self, archetype_meshes):
        base, longhood = (archetype_meshes[n] for n in ("neutral", "face/longhood"))
        cowl0, cowl1 = (group(m, "ring/cowl")[RING["H"]] for m in (base, longhood))
        tip0, tip1 = (group(m, "fascia/front_face")[RING["H"]] for m in (base, longhood))
        assert cowl1[0] < cowl0[0] - 0.2
        assert hood_run(longhood) > hood_run(base) + 0.2
        assert tip1[0] > tip0[0]
        assert tip1[2] > tip0[2] + 0.10
        slope0, error0 = hood_metrics(base)
        slope1, error1 = hood_metrics(longhood)
        assert abs(slope1) < abs(slope0) * 0.5
        assert error1 < error0 * 0.5

    @pytest.mark.parametrize("name, direction", [("plan/pointed", -1), ("plan/square", 1)])
    def test_plan_targets_change_both_end_widths(self, archetype_meshes, name, direction):
        base, shaped = archetype_meshes["neutral"], archetype_meshes[name]
        for end in ("nose_ring", "tail_ring"):
            before, after = (np.ptp(group(m, end)[:, 1]) for m in (base, shaped))
            assert direction * (after - before) > 0.20, (name, end)
        generator = BodyGenerator(shape_params(BodyParams(), {name: 1}))
        neutral = BodyGenerator(BodyParams())
        # Do not test only width-ratio endpoints: taper must affect nearby panels.
        xs = [neutral.L["x_rear"] + 0.15, neutral.L["x_front"] - 0.15]
        assert np.all(direction * (generator.plan_half_width(xs) - neutral.plan_half_width(xs)) > 0.01)

    def test_cokebottle_waists_doors_but_preserves_both_axle_shoulders(self):
        base = BodyGenerator(BodyParams())
        shaped = BodyGenerator(shape_params(BodyParams(), {"plan/cokebottle": 1}))
        xs = np.array([base.L["x_ra"], 0.0, base.L["x_fa"]])
        before, after = base.plan_half_width(xs), shaped.plan_half_width(xs)
        assert after[1] < before[1] - 0.10
        np.testing.assert_allclose(after[[0, 2]], before[[0, 2]], atol=1e-9, rtol=0)
        assert np.all(after[[0, 2]] > after[1] + 0.10)
        # Verify the generated door belt, not just the analytic plan profile.
        ring0, ring1 = (g.ring(station_index("bp_f")) for g in (base, shaped))
        assert ring1[RING["E"], 1] < ring0[RING["E"], 1] - 0.10

    def test_tumblehome_narrows_roof_relative_to_hips(self, archetype_meshes):
        base, shaped = (archetype_meshes[n] for n in ("neutral", "section/tumblehome"))
        ring0, ring1 = (group(m, "ring/bp_f") for m in (base, shaped))
        ratios = [r[RING["G"], 1] / r[RING["E"], 1] for r in (ring0, ring1)]
        assert ratios[1] < ratios[0] - 0.08
        assert ring1[RING["E"], 1] == pytest.approx(ring0[RING["E"], 1])
        assert glass_lean(shaped) > glass_lean(base) + 0.10

    def test_slabside_stands_glass_upright_and_squares_shoulders(self, archetype_meshes):
        base, shaped = (archetype_meshes[n] for n in ("neutral", "section/slabside"))
        assert 0 <= glass_lean(shaped) < glass_lean(base) * 0.65
        ring0, ring1 = (group(m, "ring/bp_f") for m in (base, shaped))
        assert ring1[RING["G"], 1] > ring0[RING["G"], 1] + 0.10
        radius0 = ring0[RING["E"], 1] - ring0[RING["E"] + 3, 1]
        radius1 = ring1[RING["E"], 1] - ring1[RING["E"] + 3, 1]
        assert radius1 < radius0 * 0.7

    def test_domed_increases_transverse_crown_and_rounds_shoulders(self, archetype_meshes):
        base, shaped = (archetype_meshes[n] for n in ("neutral", "section/domed"))
        ring0, ring1 = (group(m, "ring/bp_f") for m in (base, shaped))
        crowns = [r[RING["H"], 2] - r[RING["G"], 2] for r in (ring0, ring1)]
        assert crowns[1] > crowns[0] + 0.07
        radius0 = ring0[RING["E"], 1] - ring0[RING["E"] + 3, 1]
        radius1 = ring1[RING["E"], 1] - ring1[RING["E"] + 3, 1]
        assert radius1 > radius0 + 0.02


class TestShapeTargetSemantics:
    def test_registry_is_complete_unipolar_and_grouped(self, car_body):
        assert set(SHAPE_DELTAS) == EXPECTED_SHAPES
        assert set(SHAPE_FAMILIES) == {"face", "plan", "section"}
        registered = {n for n, m in car_body.library.modifiers.items() if m.group in SHAPE_FAMILIES}
        assert registered == EXPECTED_SHAPES
        for name in SHAPES:
            modifier = car_body.library.modifiers[name]
            assert (modifier.min_value, modifier.max_value, modifier.default) == (0, 1, 0)
            assert modifier.group == name.split("/")[0]
            assert modifier.decr is None and modifier.incr is not None
            assert modifier.incr.magnitude > 0.02, name

    @pytest.mark.parametrize("name", SHAPES)
    def test_zero_half_full_strength_is_linear_and_endpoint_matches_generator(self, car_body, archetype_meshes, name):
        library = car_body.library
        base = library.base.vertices
        full = library.morph({name: 1}).vertices
        zero = library.morph({name: 0}).vertices
        half = library.morph({name: 0.5}).vertices
        np.testing.assert_array_equal(zero, base)
        np.testing.assert_allclose(full, archetype_meshes[name].vertices, atol=1e-6, rtol=0)
        np.testing.assert_allclose(half, (base + full) / 2, atol=1e-12, rtol=0)
        assert np.max(np.linalg.norm(full - base, axis=1)) > 0.02

    @pytest.mark.parametrize("name", SHAPES)
    def test_archetype_clamps_at_both_ends(self, car_body, name):
        library = car_body.library
        np.testing.assert_array_equal(library.displacement({name: -0.8}), np.zeros_like(library.base.vertices))
        np.testing.assert_array_equal(library.displacement({name: 3}), library.displacement({name: 1}))

    @pytest.mark.parametrize("family", SHAPE_FAMILIES)
    @pytest.mark.parametrize("weights", [(0.2, 0.4), (0.25, 0.75), (1.0, 1.0), (0.8, 0.6)])
    def test_family_normalizes_only_above_one(self, car_body, family, weights):
        names = [n for n in SHAPES if n.startswith(family + "/")][:2]
        values = dict(zip(names, weights))
        divisor = max(1.0, sum(weights))
        expected = {name: weight / divisor for name, weight in values.items()}
        assert normalize_shape_values(values) == pytest.approx(expected)
        displacement = sum(car_body.library.modifiers[name].incr.offsets * weight
                           for name, weight in expected.items())
        np.testing.assert_allclose(car_body.library.displacement(values), displacement, atol=1e-12, rtol=0)

    def test_families_clamp_then_normalize_independently_and_do_not_mutate_input(self, car_body):
        values = {"face/wedge": 5.0, "face/upright": 0.5, "face/shark": -2.0,
                  "plan/pointed": 0.2, "plan/square": 0.3,
                  "section/tumblehome": 1, "section/domed": 1,
                  "width": -0.4, "sculpt/haunches": 0.7}
        original = values.copy()
        expected = {**values, "face/wedge": 2 / 3, "face/upright": 1 / 3,
                    "face/shark": 0, "section/tumblehome": 0.5, "section/domed": 0.5}
        resolved = normalize_shape_values(values)
        assert resolved == pytest.approx(expected)
        assert values == original and resolved is not values
        assert normalize_shape_values(resolved) == pytest.approx(resolved)
        np.testing.assert_allclose(car_body.library.displacement(values),
                                   MorphableMesh.displacement(car_body.library, expected), atol=1e-12, rtol=0)
        assert values == original

    def test_parameter_helper_copies_custom_base_and_combines_normalized_deltas(self):
        base = BodyParams(width=2.01, hood_crown=0.047)
        before = base.to_dict()
        values = {"face/wedge": 1, "face/upright": 1, "plan/square": 0.3}
        result = shape_params(base, values)
        expected = before.copy()
        for name, weight in {"face/wedge": 0.5, "face/upright": 0.5, "plan/square": 0.3}.items():
            for field, delta in SHAPE_DELTAS[name].items():
                expected[field] += delta * weight
        assert result.to_dict() == pytest.approx(expected)
        assert base.to_dict() == before and result is not base
        assert shape_params(base, {"face/wedge": 0}) == base


class TestStyleDefaultsAndOverrides:
    def test_style_footprints_differ_without_greenhouse_or_scale_cues(self):
        # Intersect the actual projected mesh edges, not plan_half_width: the
        # fascia and the independent arch/roof chains also affect the outline.
        # Normalize away both length and width so dimension changes alone pass
        # neither this regression nor the accompanying silhouette contact sheet.
        profiles = {}
        for style in STYLES:
            mesh = BodyGenerator(style_params(style)).build()
            edges = np.array(sorted({tuple(sorted((a, b))) for face in mesh.faces
                                     for a, b in zip(face, face[1:] + face[:1])}))
            a, b = mesh.vertices[edges].transpose(1, 0, 2)
            lo, hi = mesh.bounds()
            dx = b[:, 0] - a[:, 0]
            widths = []
            for fraction in np.linspace(0.01, 0.99, 41):
                x = lo[0] + fraction * (hi[0] - lo[0])
                hit = ((np.minimum(a[:, 0], b[:, 0]) <= x)
                       & (np.maximum(a[:, 0], b[:, 0]) >= x) & (abs(dx) > 1e-9))
                t = (x - a[hit, 0]) / dx[hit]
                y = a[hit, 1] + t * (b[hit, 1] - a[hit, 1])
                widths.append(np.max(abs(y)) * 2 / (hi[1] - lo[1]))
            profiles[style] = np.array(widths)
        for first, second in combinations(STYLES, 2):
            difference = abs(profiles[first] - profiles[second])
            assert np.sqrt(np.mean(difference ** 2)) > 0.012, (first, second)
            assert np.max(difference) > 0.055, (first, second)

    def test_every_style_has_a_valid_default_mix(self):
        assert set(STYLE_SHAPE_DEFAULTS) == set(STYLES)
        assert STYLE_SHAPE_DEFAULTS["sedan"] == {}
        for style, defaults in STYLE_SHAPE_DEFAULTS.items():
            assert set(defaults) <= EXPECTED_SHAPES
            assert all(0 < value <= 1 for value in defaults.values())
            assert normalize_shape_values(defaults) == pytest.approx(defaults)
            if style != "sedan":
                assert defaults, style

    @pytest.mark.parametrize("style", STYLES)
    def test_style_inherits_defaults_and_matches_generated_preset(self, car_body, style):
        requested = car_body.resolve_values(style=style)
        effective = resolve_shape_values(requested)
        for name, weight in STYLE_SHAPE_DEFAULTS[style].items():
            assert effective[name] == pytest.approx(weight)
        actual = car_body.library.morph(requested)
        expected = BodyGenerator(style_params(style)).build()
        np.testing.assert_allclose(actual.vertices, expected.vertices, atol=1e-6, rtol=0)

    @pytest.mark.parametrize("style", [s for s in STYLES if s != "sedan"])
    def test_explicit_zero_removes_each_inherited_target_only(self, car_body, style):
        library = car_body.library
        requested = {"style/" + style: 1}
        inherited = library.displacement(requested)
        for name, weight in STYLE_SHAPE_DEFAULTS[style].items():
            values = {**requested, name: 0}
            resolved = resolve_shape_values(values)
            assert resolved[name] == 0
            for other, other_weight in STYLE_SHAPE_DEFAULTS[style].items():
                if other != name:
                    assert resolved[other] == pytest.approx(other_weight)
            expected = inherited - weight * library.modifiers[name].incr.offsets
            np.testing.assert_allclose(library.displacement(values), expected, atol=1e-12, rtol=0)

    def test_explicit_nonzero_overrides_instead_of_adding_to_style_default(self, car_body):
        name = "face/wedge"
        default = STYLE_SHAPE_DEFAULTS["sports"][name]
        values = {"style/sports": 1, name: 0.2}
        assert resolve_shape_values(values)[name] == pytest.approx(0.2)
        expected = (car_body.library.displacement({"style/sports": 1})
                    + (0.2 - default) * car_body.library.modifiers[name].incr.offsets)
        np.testing.assert_allclose(car_body.library.displacement(values), expected, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("style, weight", [("sports", 0.5), ("suv", 0.25), ("van", 0.6)])
    def test_partial_style_is_linear_including_inherited_shapes(self, car_body, style, weight):
        library = car_body.library
        np.testing.assert_allclose(library.displacement({"style/" + style: weight}),
                                   weight * library.displacement({"style/" + style: 1}), atol=1e-12, rtol=0)

    def test_two_style_blend_inherits_weighted_defaults(self, car_body):
        weights = {"sports": 0.4, "suv": 0.35}
        values = {"style/" + name: weight for name, weight in weights.items()}
        effective = resolve_shape_values(values)
        inherited = {}
        for style, weight in weights.items():
            for name, default in STYLE_SHAPE_DEFAULTS[style].items():
                inherited[name] = inherited.get(name, 0) + weight * default
        assert effective == pytest.approx({**values, **inherited})
        expected = sum(weight * car_body.library.displacement({"style/" + style: 1})
                       for style, weight in weights.items())
        np.testing.assert_allclose(car_body.library.displacement(values), expected, atol=1e-12, rtol=0)

    def test_overfull_style_defaults_normalize_shapes_not_style_macros(self, car_body):
        values = {"style/sports": 1, "style/suv": 1, "width": 0.2}
        inherited = {}
        for style in ("sports", "suv"):
            for name, weight in STYLE_SHAPE_DEFAULTS[style].items():
                inherited[name] = inherited.get(name, 0) + weight
        expected = dict(values)
        for family in SHAPE_FAMILIES:
            family_values = {n: min(1, w) for n, w in inherited.items() if n.startswith(family + "/")}
            total = max(1, sum(family_values.values()))
            expected.update({n: w / total for n, w in family_values.items()})
        assert resolve_shape_values(values) == pytest.approx(expected)
        np.testing.assert_allclose(car_body.library.displacement(values),
                                   MorphableMesh.displacement(car_body.library, expected), atol=1e-12, rtol=0)

    @pytest.mark.parametrize("weight, inherited_weight", [(-0.5, 0), (2.0, 1)])
    def test_style_weights_are_clamped_before_default_inheritance(self, weight, inherited_weight):
        effective = resolve_shape_values({"style/sports": weight})
        for name, default in STYLE_SHAPE_DEFAULTS["sports"].items():
            assert effective.get(name, 0) == pytest.approx(inherited_weight * default)

    @pytest.mark.parametrize("style, modifiers", [
        ("sedan", {"face/wedge": 1, "face/upright": 1, "plan/square": 0.3}),
        ("sports", {"face/wedge": 0, "section/tumblehome": 0, "width": 0.2}),
        ({"coupe": 0.5, "suv": 0.25}, {"face/shark": 0.6, "sculpt/haunches": 0.2}),
    ])
    def test_carbody_build_uses_the_same_effective_displacement(self, car_body, style, modifiers):
        before = modifiers.copy()
        requested = car_body.resolve_values(style=style, modifiers=modifiers)
        effective = resolve_shape_values(requested)
        expected = car_body.library.base.vertices + MorphableMesh.displacement(car_body.library, effective)
        built = car_body.build(style=style, modifiers=modifiers)
        np.testing.assert_allclose(built.full_mesh.vertices, expected, atol=1e-12, rtol=0)
        assert modifiers == before


class TestArchetypeTopology:
    @pytest.mark.parametrize("style", STYLES)
    @pytest.mark.parametrize("name", SHAPES)
    def test_every_archetype_style_and_sweep_weight_keeps_exact_topology(self, car_body, style, name):
        library = car_body.library
        base = library.base
        for weight in (0, 0.5, 1):
            values = car_body.resolve_values(style=style, modifiers={name: weight})
            mesh = library.morph(values)
            assert mesh.n_vertices == base.n_vertices
            assert mesh.n_faces == base.n_faces
            assert mesh.faces == base.faces, (style, name, weight)
            assert np.isfinite(mesh.vertices).all()
            assert set(mesh.groups) == set(base.groups)
            for group_name in base.groups:
                np.testing.assert_array_equal(mesh.groups[group_name], base.groups[group_name])
            rings = mesh.vertices[:mesh.meta["n_rings"] * RING_N].reshape(-1, RING_N, 3)
            reflected = rings[:, [mirror_index(j) for j in range(RING_N)]] * [1, -1, 1]
            np.testing.assert_allclose(rings, reflected, atol=1e-9, rtol=0)

    @pytest.mark.parametrize("name", SHAPES)
    def test_generated_endpoints_keep_indices_and_wheel_circle_landmarks(self, archetype_meshes, name):
        base, mesh = archetype_meshes["neutral"], archetype_meshes[name]
        assert mesh.n_vertices == base.n_vertices
        assert mesh.n_faces == base.n_faces
        assert mesh.faces == base.faces
        for group_name in base.groups:
            np.testing.assert_array_equal(mesh.groups[group_name], base.groups[group_name])
        for axle in ("front", "rear"):
            for side in ("L", "R"):
                group_name = f"arch_{axle}_{side}"
                np.testing.assert_allclose(group(mesh, group_name)[:, [0, 2]],
                                           group(base, group_name)[:, [0, 2]], atol=1e-9, rtol=0)


class TestArchetypeCachingAndCustomBase:
    def test_disk_roundtrip_retains_shapes_style_overrides_and_normalization(self, car_body, tmp_path, monkeypatch):
        params = BodyParams()
        path = tmp_path / f"body_targets_{targets._cache_key(params)}.npz"
        targets._save_library(car_body.library, path)
        monkeypatch.setattr(targets, "_cache_dir", lambda: tmp_path)

        def unexpected_rebuild(*args, **kwargs):
            pytest.fail("valid disk cache was ignored and differential targets were rebuilt")

        monkeypatch.setattr(targets.DifferentialTargetBuilder, "build", unexpected_rebuild)
        # Bypass only the process LRU so this really exercises the public library's
        # disk-load path, without flushing or mutating shared session fixtures.
        loaded = targets._cached_library.__wrapped__(tuple(sorted(params.to_dict().items())))
        assert isinstance(loaded, targets.BodyMorphableMesh)
        assert set(loaded.modifiers) == set(car_body.library.modifiers)
        for values in ({name: 1 for name in SHAPES},
                       {"style/sports": 1, "face/wedge": 0, "face/shark": 0.6},
                       {"style/suv": 0.4, "style/coupe": 0.5, "width": -0.3}):
            np.testing.assert_allclose(loaded.displacement(values), car_body.library.displacement(values),
                                       atol=1e-7, rtol=0)
        for name in SHAPES:
            assert loaded.modifiers[name].to_dict() == car_body.library.modifiers[name].to_dict()

    def test_cache_key_tracks_shape_geometry_and_style_defaults(self, monkeypatch):
        params = BodyParams()
        original = targets._cache_key(params)
        assert targets.LIBRARY_VERSION > 11
        assert targets._cache_key(replace(params, tumblehome=0.03)) != original
        assert targets._cache_key(replace(params, hood_straightness=0.2)) != original
        with monkeypatch.context() as scoped:
            scoped.setitem(SHAPE_DELTAS, "face/wedge", {**SHAPE_DELTAS["face/wedge"], "hood_front_height": -0.123})
            assert targets._cache_key(params) != original
        with monkeypatch.context() as scoped:
            scoped.setitem(STYLE_SHAPE_DEFAULTS, "sports", {**STYLE_SHAPE_DEFAULTS["sports"], "face/wedge": 0.123})
            assert targets._cache_key(params) != original
        assert targets._cache_key(params) == original

    def test_custom_base_archetypes_are_relative_to_that_base(self, car_body, tmp_path, monkeypatch):
        params = BodyParams(width=1.98, hood_crown=0.047, roof_height=1.50, cowl_offset=0.45)
        monkeypatch.setattr(targets, "_cache_dir", lambda: tmp_path)
        # One full custom library, not one expensive rebuild per archetype.
        library = targets._cached_library.__wrapped__(tuple(sorted(params.to_dict().items())))
        assert library is not car_body.library
        np.testing.assert_allclose(library.base.vertices, BodyGenerator(params).build().vertices, atol=1e-12, rtol=0)
        for name in SHAPES:
            expected = BodyGenerator(shape_params(params, {name: 1})).build()
            np.testing.assert_allclose(library.morph({name: 1}).vertices, expected.vertices, atol=1e-9, rtol=0)
        expected_style = BodyGenerator(style_params("sports", params)).build()
        np.testing.assert_allclose(library.morph({"style/sports": 1}).vertices, expected_style.vertices, atol=1e-9, rtol=0)
        wedge = STYLE_SHAPE_DEFAULTS["sports"]["face/wedge"]
        expected_override = expected_style.vertices - wedge * library.modifiers["face/wedge"].incr.offsets
        np.testing.assert_allclose(library.morph({"style/sports": 1, "face/wedge": 0}).vertices,
                                   expected_override, atol=1e-9, rtol=0)
        np.testing.assert_array_equal(car_body.library.base.vertices, BodyGenerator(BodyParams()).build().vertices)
