"""Tests for makecar.morph: targets, modifiers, morphable meshes and the
differential target builder."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from makecar.geometry import primitives as P
from makecar.geometry.mesh import Mesh
from makecar.morph import Target, Modifier, MorphableMesh, DifferentialTargetBuilder
from makecar.morph.morphable import ModifierSpec


# ------------------------------------------------------------------ helpers
@dataclass
class BoxParams:
    sx: float = 1.0
    sy: float = 2.0
    sz: float = 3.0


def make_box(p: BoxParams) -> Mesh:
    return P.box(p.sx, p.sy, p.sz)


def _targets(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return Target("a", rng.normal(size=(n, 3))), Target("b", rng.normal(size=(n, 3)))


# ------------------------------------------------------------------- Target
class TestTarget:
    def test_from_difference(self):
        base = np.zeros((5, 3))
        var = np.arange(15, dtype=float).reshape(5, 3)
        var[0] = 0.0
        t = Target.from_difference("t", base, var, "desc")
        assert t.offsets.shape == (5, 3)
        assert np.array_equal(t.offsets, var)
        assert t.n_vertices == 5 and t.description == "desc"
        assert t.magnitude == pytest.approx(np.linalg.norm(var[-1]))
        assert t.nonzero_count() == 4  # first row is (0,0,0)
        assert np.allclose(t.scaled(0.5), var * 0.5)

    def test_from_difference_rejects_topology_mismatch(self):
        with pytest.raises(ValueError, match="topology mismatch"):
            Target.from_difference("bad", np.zeros((5, 3)), np.zeros((6, 3)))
        with pytest.raises(ValueError):
            Target.from_difference("bad", np.zeros((5, 3)), np.zeros((5, 2)))

    def test_offsets_are_cast_to_float(self):
        t = Target("i", [[1, 2, 3]])
        assert t.offsets.dtype == float
        assert Target("e", np.zeros((0, 3))).magnitude == 0.0

    def test_save_load_roundtrip(self, tmp_path):
        rng = np.random.default_rng(7)
        off = rng.normal(size=(40, 3))
        off[::3] = 0.0  # sparse rows are omitted from the file
        t = Target("roof-incr", off, "raise the roof")
        path = tmp_path / "roof-incr.target"
        t.save(path)
        text = path.read_text().splitlines()
        assert text[0].startswith("# makecar target roof-incr: raise the roof")
        data_lines = [l for l in text if l and not l.startswith("#")]
        assert len(data_lines) == t.nonzero_count()
        assert all(len(l.split()) == 4 for l in data_lines)
        loaded = Target.load(path, n_vertices=40)
        assert loaded.name == "roof-incr"  # from the file stem
        assert loaded.offsets.shape == (40, 3)
        assert np.allclose(loaded.offsets, off, atol=1e-6)
        assert np.all(loaded.offsets[::3] == 0.0)
        named = Target.load(path, n_vertices=40, name="other")
        assert named.name == "other"

    def test_load_ignores_blank_and_comment_lines(self, tmp_path):
        p = tmp_path / "t.target"
        p.write_text("# comment\n\n2 0.5 -0.25 1.0\n\n")
        t = Target.load(p, 4)
        assert np.allclose(t.offsets[2], [0.5, -0.25, 1.0])
        assert t.nonzero_count() == 1


# ----------------------------------------------------------------- Modifier
class TestModifier:
    def test_bipolar_weights(self):
        incr, decr = _targets()
        m = Modifier("m", incr, decr)
        assert m.weights(0.5) == [(incr, 0.5)]
        assert m.weights(-0.25) == [(decr, 0.25)]
        assert m.weights(0.0) == []
        assert m.weights(3.0) == [(incr, 1.0)]     # clamped
        assert m.weights(-9.0) == [(decr, 1.0)]
        assert not m.is_unipolar
        assert m.clamp(1.7) == 1.0 and m.clamp(-1.7) == -1.0

    def test_unipolar_weights_ignore_negative_values(self):
        incr, _ = _targets()
        m = Modifier("macro", incr, None, min_value=0.0, max_value=1.0)
        assert m.is_unipolar
        assert m.weights(0.7) == [(incr, 0.7)]
        assert m.weights(-0.5) == []
        assert m.weights(2.0) == [(incr, 1.0)]

    def test_bipolar_with_only_incr_mirrors(self):
        incr, _ = _targets()
        m = Modifier("m", incr, None, min_value=-1.0)
        assert m.weights(-0.4) == [(incr, -0.4)]
        assert m.weights(0.4) == [(incr, 0.4)]

    def test_no_targets_gives_no_weights(self):
        m = Modifier("empty")
        assert m.weights(1.0) == [] and m.weights(-1.0) == []

    def test_to_dict(self):
        incr, decr = _targets()
        d = Modifier("m", incr, decr, group="g", description="d").to_dict()
        assert d["name"] == "m" and d["group"] == "g" and d["targets"] == ["a", "b"]
        assert d["min"] == -1.0 and d["max"] == 1.0 and d["default"] == 0.0
        assert Modifier("u", incr, None, 0.0, 1.0).to_dict()["targets"] == ["a"]


# ------------------------------------------------------------ MorphableMesh
class TestMorphableMesh:
    def _mm(self):
        base = P.box(1, 1, 1)
        mm = MorphableMesh(base, "mm")
        ta, tb = _targets()
        tc = Target("c", np.random.default_rng(5).normal(size=(8, 3)))
        mm.add_modifier(Modifier("a", ta, tb))                   # bipolar with both sides
        mm.add_modifier(Modifier("b", tc, None, 0.0, 1.0))       # unipolar
        return mm, ta, tb, tc

    def test_add_target_rejects_wrong_size(self):
        mm = MorphableMesh(P.box(1, 1, 1))
        with pytest.raises(ValueError):
            mm.add_target(Target("x", np.zeros((7, 3))))
        mm.add_target(Target("ok", np.zeros((8, 3))))
        assert "ok" in mm.targets

    def test_add_modifier_registers_targets(self):
        mm, ta, tb, tc = self._mm()
        assert set(mm.targets) == {"a", "b", "c"}
        assert mm.modifier_names() == ["a", "b"]
        assert mm.modifier_names(group="misc") == ["a", "b"]
        assert mm.modifier_names(group="zzz") == []
        assert [d["name"] for d in mm.describe()] == ["a", "b"]

    def test_displacement_semantics(self):
        mm, ta, tb, tc = self._mm()
        assert np.allclose(mm.displacement({"a": 0.5}), 0.5 * ta.offsets)
        assert np.allclose(mm.displacement({"a": -0.5}), 0.5 * tb.offsets)
        assert np.allclose(mm.displacement({"b": -1.0}), 0.0)  # unipolar clamps negatives away
        assert np.allclose(mm.displacement({"a": 0.0, "b": None}), 0.0)

    def test_morph_is_linear_in_modifiers(self):
        mm, ta, tb, tc = self._mm()
        base = mm.base.vertices
        da = mm.morph({"a": 0.3}).vertices - base
        db = mm.morph({"b": 0.8}).vertices - base
        dab = mm.morph({"a": 0.3, "b": 0.8}).vertices - base
        assert np.allclose(da + db, dab)
        dneg = mm.morph({"a": -0.6}).vertices - base
        assert np.allclose(dneg, 0.6 * tb.offsets)

    def test_morph_does_not_mutate_base_and_drops_lines(self):
        mm, *_ = self._mm()
        mm.base.add_line([[0, 0, 0], [1, 1, 1]], "seam")
        before = mm.base.vertices.copy()
        out = mm.morph({"a": 1.0})
        assert np.array_equal(mm.base.vertices, before)
        assert out.lines == [] and out.line_names == []
        assert out.n_faces == mm.base.n_faces
        assert np.array_equal(mm.morph().vertices, before)

    def test_morph_extra_offsets(self):
        mm, *_ = self._mm()
        extra = np.ones_like(mm.base.vertices)
        out = mm.morph(None, extra_offsets=extra)
        assert np.allclose(out.vertices, mm.base.vertices + 1)

    def test_unknown_modifier_raises(self):
        mm, *_ = self._mm()
        with pytest.raises(KeyError, match="unknown modifier"):
            mm.morph({"nope": 1.0})


# ------------------------------------------------- DifferentialTargetBuilder
class TestDifferentialTargetBuilder:
    SPECS = [
        ModifierSpec("sx", 0.2, 0.5, "size", "box length"),
        ModifierSpec("sz", 0.0, 1.0, "size", "box height"),  # incr only -> unipolar
    ]

    def _build(self, macros=None):
        builder = DifferentialTargetBuilder(make_box, BoxParams(), name="toybox")
        return builder.build(self.SPECS, macros=macros, macro_group="style")

    def test_base_and_modifier_structure(self):
        mm = self._build()
        assert mm.name == "toybox"
        assert np.allclose(mm.base.vertices, make_box(BoxParams()).vertices)
        assert set(mm.modifiers) == {"sx", "sz"}
        sx = mm.modifiers["sx"]
        assert sx.incr.name == "sx-incr" and sx.decr.name == "sx-decr"
        assert not sx.is_unipolar and sx.unit_scale == 0.5
        assert sx.group == "size" and sx.description == "box length"
        sz = mm.modifiers["sz"]
        assert sz.decr is None and sz.is_unipolar
        assert "sx-incr" in mm.targets and "sz-incr" in mm.targets and "sz-decr" not in mm.targets

    def test_value_one_reproduces_variant_exactly(self):
        mm = self._build()
        assert np.allclose(mm.morph({"sx": 1.0}).vertices, make_box(BoxParams(sx=1.5)).vertices)
        assert np.allclose(mm.morph({"sx": -1.0}).vertices, make_box(BoxParams(sx=0.8)).vertices)
        assert np.allclose(mm.morph({"sz": 1.0}).vertices, make_box(BoxParams(sz=4.0)).vertices)
        # box vertices are linear in the size, so half the slider is half the change
        assert np.allclose(mm.morph({"sx": 0.5}).vertices, make_box(BoxParams(sx=1.25)).vertices)
        # combined modifiers change independent parameters
        assert np.allclose(mm.morph({"sx": 1.0, "sz": 1.0}).vertices, make_box(BoxParams(sx=1.5, sz=4.0)).vertices)

    def test_macros_become_unipolar_style_modifiers(self):
        tall = BoxParams(1.0, 2.0, 6.0)
        mm = self._build(macros={"tall": tall})
        assert "style/tall" in mm.modifiers
        mod = mm.modifiers["style/tall"]
        assert mod.is_unipolar and mod.group == "style"
        assert mod.incr.name == "style-tall"
        assert np.allclose(mm.morph({"style/tall": 1.0}).vertices, make_box(tall).vertices)
        half = mm.morph({"style/tall": 0.5}).vertices
        assert np.allclose(half, 0.5 * (mm.base.vertices + make_box(tall).vertices))

    def test_bounds_clamp_the_variant(self):
        specs = [ModifierSpec("sx", 0.9, 0.5, upper_bound=1.2, lower_bound=0.5)]
        mm = DifferentialTargetBuilder(make_box, BoxParams()).build(specs)
        assert np.allclose(mm.morph({"sx": 1.0}).vertices, make_box(BoxParams(sx=1.2)).vertices)
        assert np.allclose(mm.morph({"sx": -1.0}).vertices, make_box(BoxParams(sx=0.5)).vertices)
        assert "1.2" in mm.modifiers["sx"].incr.description

    def test_custom_modifier_name(self):
        specs = [ModifierSpec("sy", 0.1, 0.1, name="depth")]
        mm = DifferentialTargetBuilder(make_box, BoxParams()).build(specs)
        assert set(mm.modifiers) == {"depth"}
        assert mm.modifiers["depth"].incr.name == "depth-incr"

    def test_generator_topology_mismatch_is_detected(self):
        def bad(p):
            return P.box(p.sx, p.sy, p.sz, bevel=0.1 if p.sx > 1.0 else 0.0)

        with pytest.raises(ValueError, match="topology mismatch"):
            DifferentialTargetBuilder(bad, BoxParams()).build([ModifierSpec("sx", 0.0, 0.5)])
