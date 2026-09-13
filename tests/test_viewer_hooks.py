"""Engine hooks the interactive viewer relies on: connector overrides and custom targets."""
import numpy as np
import pytest

from makecar.connectors import apply_override, RectangleConnector, CircleConnector, PolygonConnector, PointConnector
from makecar.geometry.frame import Frame
from makecar.config import CarConfig
from makecar.pipeline import build_car
from makecar.body import CarBody
from makecar.morph.target import Target


class TestApplyOverride:
    def test_translate_and_rotate_keep_identity(self):
        c = RectangleConnector("seat", Frame.identity([1, 2, 0.3]), 0.5, 0.6, tags=["seat"], meta={"row": 0})
        o = apply_override(c, {"translate": [0.1, 0, 0], "rotate_deg": [0, 0, 90]})
        assert isinstance(o, RectangleConnector) and o.name == "seat" and "seat" in o.tags and o.meta["row"] == 0
        assert np.allclose(o.origin, [1.1, 2, 0.3])
        assert np.allclose(o.frame.x_axis, [0, 1, 0], atol=1e-9)
        assert o.meta["override"]["translate"] == [0.1, 0, 0]

    def test_sizes(self):
        r = apply_override(RectangleConnector("r", Frame.identity(), 0.5, 0.6), {"scale": 2.0})
        assert (r.width, r.height) == (1.0, 1.2)
        r2 = apply_override(RectangleConnector("r", Frame.identity(), 0.5, 0.6), {"width": 0.7})
        assert (r2.width, r2.height) == (0.7, 0.6)
        c = apply_override(CircleConnector("c", Frame.identity(), 0.3), {"radius": 0.4})
        assert c.radius == 0.4 and len(c.points) == 32
        p = apply_override(PointConnector("p", Frame.identity()), {"scale": 3.0, "translate": [0, 0, 1]})
        assert isinstance(p, PointConnector) and np.allclose(p.origin, [0, 0, 1])

    def test_polygon_scales_about_its_centroid(self):
        pts = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0.0]])
        c = PolygonConnector.from_points("poly", pts, meta={"grid": [2, 2]})
        o = apply_override(c, {"scale": 0.5})
        assert np.allclose(o.centroid, c.centroid, atol=1e-9)
        assert o.width == pytest.approx(c.width * 0.5) and o.height == pytest.approx(c.height * 0.5)

    def test_empty_override_is_identity(self):
        c = CircleConnector("c", Frame.identity(), 0.3)
        assert apply_override(c, {}) is c


class TestConfigOverrides:
    def test_override_moves_component(self):
        base = CarConfig.from_dict({"name": "a"})
        moved = CarConfig.from_dict({"name": "b", "connectors": {"overrides": {"steering_wheel": {"translate": [0, 0, 0.1], "scale": 1.2}}}})
        _, a, _ = build_car(base)
        _, b, _ = build_car(moved)
        ca = next(i for i in a.instances if i.connector.name == "steering_wheel")
        cb = next(i for i in b.instances if i.connector.name == "steering_wheel")
        assert cb.connector.radius == pytest.approx(ca.connector.radius * 1.2)
        za, zb = ca.result.mesh.bounds()[1][2], cb.result.mesh.bounds()[1][2]
        assert zb > za + 0.05

    def test_override_on_sub_connector(self):
        cfg = CarConfig.from_dict({"name": "c", "connectors": {"overrides": {"dashboard/screen": {"scale": 1.5}}}})
        _, asm, _ = build_car(cfg)
        scr = next(c for c in asm.connectors if c.name == "dashboard/screen")
        assert scr.meta.get("override", {}).get("scale") == 1.5

    def test_unknown_override_key_rejected(self):
        with pytest.raises(ValueError):
            CarConfig.from_dict({"name": "x", "connectors": {"overrides": {"seat": {"colour": "red"}}}})

    def test_save_roundtrip(self, tmp_path):
        cfg = CarConfig.from_dict({"name": "rt", "body": {"style": "coupe", "modifiers": {"wheelbase": 0.2}},
                                   "connectors": {"overrides": {"mirror_L": {"translate": [0, 0.01, 0]}}}})
        p = cfg.save(tmp_path / "rt.yaml")
        again = CarConfig.load(p)
        assert again.body["style"] == "coupe" and again.body["modifiers"] == {"wheelbase": 0.2}
        assert again.connector_overrides() == {"mirror_L": {"translate": [0, 0.01, 0]}}
        text = p.read_text()
        assert "views:" not in text  # untouched defaults are not written out


class TestCustomTargets:
    def test_save_load_and_apply(self, tmp_path):
        cb = CarBody(custom_targets_dir=tmp_path / "targets")
        n = cb.library.base.n_vertices
        off = np.zeros((n, 3))
        off[:100, 2] = 0.05
        key = cb.save_custom_target("roof_lift", off)
        assert key == "custom/roof_lift" and (tmp_path / "targets" / "roof_lift.target").exists()
        res = cb.build(style="sedan", modifiers={"custom/roof_lift": 1.0})
        base = cb.build(style="sedan")
        d = res.full_mesh.vertices - base.full_mesh.vertices
        assert np.allclose(d[:100, 2], 0.05) and np.allclose(d[100:], 0)
        # a fresh body loads it from disk, and the shared cache is untouched
        cb2 = CarBody(custom_targets_dir=tmp_path / "targets")
        assert "custom/roof_lift" in cb2.library.modifiers
        assert "custom/roof_lift" not in CarBody().library.modifiers

    def test_config_wires_custom_dir(self, tmp_path):
        (tmp_path / "t").mkdir()
        Target("custom-x", np.zeros((CarBody().library.base.n_vertices, 3))).save(tmp_path / "t" / "x.target")
        cfgp = tmp_path / "car.yaml"
        cfgp.write_text("name: cc\nbody:\n  custom_targets: t\n  modifiers: {custom/x: 0.5}\n")
        cfg = CarConfig.load(cfgp)
        assert cfg.custom_targets_dir() == tmp_path / "t"
        res, asm, _ = build_car(cfg)
        assert res.modifier_values.get("custom/x") == 0.5
