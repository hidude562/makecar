"""Integration tests: config -> assembly -> outputs, and the CLI."""
import json
from pathlib import Path

import numpy as np
import pytest

from makecar.config import CarConfig, deep_merge, DEFAULT_CONFIG
from makecar.assembly import assemble, INTERIOR_TAGS
from makecar.body import CarBody
from makecar.body.styles import style_names
from makecar.components import REGISTRY, default_component_for
from makecar.pipeline import build_car, write_outputs, render_views, cutaway_mesh, car_body_for
from makecar import cli

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestConfig:
    def test_defaults_are_merged(self):
        cfg = CarConfig.from_dict({"name": "x"})
        assert cfg.body["style"] == "sedan"
        assert cfg.output["formats"] == DEFAULT_CONFIG["output"]["formats"]

    def test_deep_merge_does_not_mutate(self):
        base = {"a": {"b": 1, "c": 2}}
        out = deep_merge(base, {"a": {"b": 5}})
        assert out == {"a": {"b": 5, "c": 2}} and base["a"]["b"] == 1

    @pytest.mark.parametrize("bad", [
        {"body": {"style": "spaceship"}},
        {"body": {"style": {"sedan": 2.0}}},
        {"body": {"modifiers": {"wheelbase": 3}}},
        {"body": {"modifiers": {"wheelbase": "wide"}}},
        {"output": {"formats": ["stl"]}},
    ])
    def test_validation_errors(self, bad):
        with pytest.raises(ValueError):
            CarConfig.from_dict({"name": "x", **bad})

    def test_variants_expand_and_rename(self):
        cfg = CarConfig.from_dict({"name": "base", "body": {"modifiers": {"width": 0.1}},
                                   "variants": [{"body": {"modifiers": {"wheelbase": 0.5}}}, {"name": "second"}]})
        vs = cfg.variants()
        assert [v.name for v in vs] == ["base_v1", "second"]
        assert vs[0].body["modifiers"] == {"width": 0.1, "wheelbase": 0.5}
        assert "variants" not in vs[0].raw

    def test_variant_style_replaces_instead_of_merging(self):
        cfg = CarConfig.from_dict({"name": "b", "body": {"style": {"sedan": 0.5, "wagon": 0.5}},
                                   "variants": [{"name": "v", "body": {"style": {"suv": 0.7, "coupe": 0.3}}},
                                                {"name": "w", "body": {"style": "van"}},
                                                {"name": "u", "body": {"modifiers": {"width": 0.2}}}]})
        v, w, u = cfg.variants()
        assert v.body["style"] == {"suv": 0.7, "coupe": 0.3}
        assert w.body["style"] == "van"
        assert u.body["style"] == {"sedan": 0.5, "wagon": 0.5}

    def test_palette_and_hints(self):
        cfg = CarConfig.from_dict({"name": "x", "palette": {"paint": "#123456", "custom": "#abcdef"},
                                   "body": {"hints": {"drive": "right", "door_count": None}}})
        p = cfg.palette()
        assert p.paint == "#123456" and p.color("custom") == "#abcdef"
        assert cfg.hints()["drive"] == "right" and "door_count" not in cfg.hints()

    def test_shipped_configs_load(self):
        paths = sorted(CONFIG_DIR.glob("*.yaml"))
        assert len(paths) >= 8
        for p in paths:
            CarConfig.load(p)


class TestAssembly:
    def test_every_interior_tag_has_a_default(self):
        from makecar.connectors import RectangleConnector, CircleConnector, PointConnector, PolygonConnector
        from makecar.geometry.frame import Frame
        needed = {"seat_front": RectangleConnector, "seat_bench": RectangleConnector, "steering_wheel": CircleConnector,
                  "dashboard": PolygonConnector, "pedals": RectangleConnector, "console": RectangleConnector,
                  "floor": RectangleConnector, "bulkhead": RectangleConnector, "shelf": RectangleConnector,
                  "cargo_floor": RectangleConnector, "headliner": PolygonConnector, "door_card": PolygonConnector,
                  "rearview_mirror": PointConnector, "wheel": CircleConnector, "glass": PolygonConnector,
                  "headlight": PolygonConnector, "grille": RectangleConnector, "plate": RectangleConnector}
        fr = Frame.identity()
        for tag, cls in needed.items():
            if cls is RectangleConnector:
                c = cls("c", fr, 0.5, 0.5, tags=[tag])
            elif cls is CircleConnector:
                c = cls("c", fr, 0.3, tags=[tag])
            elif cls is PolygonConnector:
                c = cls("c", fr, np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]]), tags=[tag])
            else:
                c = cls("c", fr, tags=[tag])
            assert default_component_for(c) is not None, tag

    @pytest.mark.parametrize("style", style_names())
    def test_all_styles_assemble_completely(self, style):
        cfg = CarConfig.from_dict({"name": style, "body": {"style": style}})
        res, asm, _ = build_car(cfg)
        # only the opt-in roof rails may stay unattached by default
        assert set(asm.unattached) <= {"roof_rail_L", "roof_rail_R"}, asm.unattached
        assert asm.disabled == []
        full = asm.mesh()
        assert np.isfinite(full.vertices).all()
        assert full.n_faces > 15000
        # sub-connectors were resolved recursively (dashboard -> cluster etc.)
        assert any("/" in c.name for c in asm.connectors)
        assert any(i.parent is not None for i in asm.instances)

    def test_rules_disable_and_override(self, sedan):
        asm = assemble(sedan, {"disable": ["antenna", "handle_*"],
                                    "assign": {"wheel": {"component": "wheel.alloy", "options": {"spokes": 7}},
                                               "roof_rail": "roof.rails"}})
        assert "antenna" in asm.disabled and "handle_front_L" in asm.disabled
        wheel = next(i for i in asm.instances if i.connector.name == "wheel_front_L")
        assert wheel.options == {"spokes": 7}
        assert asm.connector_component("roof_rail_L") == "roof.rails"

    def test_bad_component_type_raises(self, sedan):
        # a wheel (circle-only component) cannot attach to the grille rectangle ...
        with pytest.raises(TypeError):
            assemble(sedan, {"assign": {"grille": "wheel.alloy"}})
        # ... but glass (accepts any polygon) attaches to a circle by inheritance
        asm = assemble(sedan, {"assign": {"fuel_cap": "glass.tinted"}})
        assert asm.connector_component("fuel_cap") == "glass.tinted"

    def test_unknown_component_raises(self, sedan):
        with pytest.raises(KeyError):
            assemble(sedan, {"assign": {"wheel": "wheel.nonexistent"}})

    def test_serialisation(self, sedan):
        asm = assemble(sedan, {})
        json.dumps(asm.to_dict())
        d = asm.connectors_dict()
        json.dumps(d)
        assert all("component" in c for c in d)
        assert asm.interior_mesh().n_faces > 1000

    def test_object_groups_cover_every_face(self, sedan):
        asm = assemble(sedan, {})
        groups = asm.object_groups()
        assert len(groups) == asm.mesh().n_faces
        assert set(groups[i] for i in range(sedan.mesh.n_faces)) == {"body"}


class TestPipeline:
    def test_random_modifiers_are_seeded(self):
        a = CarConfig.from_dict({"name": "a", "seed": 5, "body": {"random": {"amount": 0.5}}})
        b = CarConfig.from_dict({"name": "b", "seed": 5, "body": {"random": {"amount": 0.5}}})
        c = CarConfig.from_dict({"name": "c", "seed": 6, "body": {"random": {"amount": 0.5}}})
        ra, _, _ = build_car(a)
        rb, _, _ = build_car(b)
        rc, _, _ = build_car(c)
        assert ra.modifier_values == rb.modifier_values
        assert ra.modifier_values != rc.modifier_values
        assert all(abs(v) <= 0.5 for v in ra.modifier_values.values())

    def test_cutaway_mesh_has_no_roof(self):
        res, asm, _ = build_car(CarConfig.from_dict({"name": "x"}))
        cut = cutaway_mesh(asm)
        zmax_body = max(res.mesh.vertices[list(f)][:, 2].max() for f in cut.faces[: 100])
        assert zmax_body < res.measurements["z_belt"] + 0.07
        assert cut.n_faces < asm.mesh().n_faces

    def test_write_outputs(self, tmp_path):
        cfg = CarConfig.from_dict({"name": "tiny", "output": {"formats": ["obj", "json", "png", "targets"],
                                                               "views": ["side", "driver"], "image_size": [160, 100]},
                                   "body": {"modifiers": {"wheelbase": 0.3}}})
        res, asm, timings = build_car(cfg)
        files = write_outputs(cfg, res, asm, tmp_path / "tiny", timings)
        names = {f.name for f in files}
        assert {"car.obj", "car.mtl", "body.obj", "connectors.json", "assembly.json", "spec.txt", "preview.png", "interior.png"} <= names
        info = json.loads((tmp_path / "tiny" / "assembly.json").read_text())
        assert info["stats"]["components"] == len(asm.instances)
        assert info["modifier_values"] == {"wheelbase": 0.3}
        conns = json.loads((tmp_path / "tiny" / "connectors.json").read_text())
        assert any(c["name"] == "dashboard/cluster" for c in conns)
        assert any(p.suffix == ".target" for p in (tmp_path / "tiny" / "targets").iterdir())
        obj = (tmp_path / "tiny" / "car.obj").read_text()
        assert obj.count("\ng ") > 20 and "usemtl" in obj

    def test_render_views_shapes(self):
        res, asm, _ = build_car(CarConfig.from_dict({"name": "x"}))
        imgs = render_views(asm, ["front", "interior_cutaway"], [120, 80])
        assert imgs["front"].shape == (56, 60, 3)
        assert imgs["interior_cutaway"].shape == (80, 120, 3)


class TestCLI:
    def test_list_commands(self, capsys):
        cli.main(["list-styles"])
        cli.main(["list-modifiers"])
        cli.main(["list-components"])
        out = capsys.readouterr().out
        assert "hatchback" in out and "wheelbase" in out and "seat.bucket" in out and "wheel.alloy" in out

    def test_build_and_connectors(self, tmp_path, capsys):
        cfgp = tmp_path / "c.json"
        cfgp.write_text(json.dumps({"name": "clitest", "body": {"style": "coupe"},
                                    "output": {"formats": ["json"], "views": []}}))
        cli.main(["build", str(cfgp), "-o", str(tmp_path / "out"), "--set", "roof_height=-0.2"])
        info = json.loads((tmp_path / "out" / "clitest" / "assembly.json").read_text())
        assert info["modifier_values"]["roof_height"] == -0.2
        cli.main(["connectors", str(cfgp)])
        out = capsys.readouterr().out
        assert "steering_wheel" in out and "wheel.alloy" in out
