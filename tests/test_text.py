"""The text engine (fonts, layout, filling) and the text.box component."""
from __future__ import annotations

import json

import numpy as np
import pytest

from makecar.assembly import assemble
from makecar.components import Palette, get_component
from makecar.components.text import FONT_DIR, fill, font_names, layout, load_font, planar_mapper, text_mesh
from makecar.geometry import primitives as P
from makecar.geometry.frame import Frame


def _area(c):
    x, y = c[:, 0], c[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class TestFonts:
    def test_shipped_fonts_load_with_metrics_and_kerning(self):
        assert {"sans", "sans-bold", "sans-condensed-bold", "serif-bold", "serif-condensed-bold", "mono-bold"} <= set(font_names())
        assert (FONT_DIR / "LICENSE.txt").exists()
        for name in font_names():
            f = load_font(name)
            assert f.units_per_em > 0 and 0 < f.cap_height < f.ascender and f.descender < 0
            assert {"A", "Z", "0", "9", " ", "-"} <= set(f.glyphs)
            assert f.glyph("O").contours and not f.glyph(" ").contours
            assert f.kerning.get("AV", 0) < 0 or name == "mono-bold"
            assert f.glyph("é") is not None
        assert load_font("sans-bold") is load_font("sans-bold")
        with pytest.raises(KeyError):
            load_font("comic-sans")

    def test_a_font_json_can_be_loaded_by_path(self, tmp_path):
        data = json.loads((FONT_DIR / "sans.json").read_text())
        p = tmp_path / "mine.json"
        p.write_text(json.dumps(data))
        assert load_font(str(p)).name == "sans"


class TestLayoutAndFill:
    def test_fill_matches_the_outline_area_including_holes(self):
        f = load_font("sans-bold")
        for ch in "OAB8gQ":
            g = f.glyph(ch)
            outline = abs(sum(_area(c) for c in g.contours))
            quads = fill(g.contours, f.cap_height / 16)
            filled = sum(abs(_area(q)) for q in quads)
            assert filled == pytest.approx(outline, rel=0.02), ch   # curves are chords across each band

    def test_layout_size_alignment_and_kerning(self):
        f = load_font("sans-bold")
        lay = layout("H", f, 0.14)
        pts = np.vstack(lay.glyphs[0].contours)
        assert pts[:, 1].max() - pts[:, 1].min() == pytest.approx(0.14, abs=1e-6)   # size is the cap height
        assert lay.height == pytest.approx(0.14)
        two = layout("AB\nC", f, 0.1, align="left")
        assert two.height > 0.2 and len(two.line_widths) == 2
        first_x = np.vstack(two.glyphs[0].contours)[:, 0].min()
        assert -two.width / 2 <= first_x < -two.width / 2 + 0.02          # left edge, up to the A's side bearing
        right = layout("AB\nC", f, 0.1, align="right")
        last_x = np.vstack(right.glyphs[-1].contours)[:, 0].max()
        assert right.width / 2 - 0.02 < last_x <= right.width / 2 + 1e-9
        assert layout("AV", f, 0.1).width < layout("AV", f, 0.1, kerning=False).width
        assert layout("AV", f, 0.1, letter_spacing=0.02).width == pytest.approx(layout("AV", f, 0.1).width + 0.02)
        slanted = np.vstack(layout("I", f, 0.1, slant_deg=15).glyphs[0].contours)
        upright = np.vstack(layout("I", f, 0.1).glyphs[0].contours)
        assert slanted[:, 0].max() - slanted[:, 0].min() > upright[:, 0].max() - upright[:, 0].min() + 0.02
        assert layout("", f, 0.1).glyphs == [] and layout("☃", f, 0.1).glyphs == []   # unknown glyphs are skipped

    def test_text_mesh_is_a_solid_with_one_group_per_glyph(self):
        f = load_font("sans-bold")
        fr = Frame.from_normal([0, 0, 0], [0, 0, 1], [1, 0, 0])
        mp = planar_mapper(fr, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.zeros(3))
        lay = layout("PO 8", f, 0.14)
        m = text_mesh(lay, mp, 0.002, "ink")
        assert set(m.groups) == {"glyph_0", "glyph_1", "glyph_3"}     # the space has nothing to draw
        assert set(m.face_materials) == {"ink"}
        lo, hi = m.bounds()
        assert hi[2] == pytest.approx(0.002) and lo[2] == pytest.approx(0.0)
        assert hi[1] - lo[1] == pytest.approx(0.14, abs=0.01)   # round letters overshoot the cap line a little
        assert P.signed_volume(m) > 0
        # the O is a ring: its top faces cover the outline area minus the counter
        o = m.zones["glyph_1"]
        cent = m.face_centroids()[o]
        assert (cent[:, 2] > 0.0019).sum() > 20


class TestTextBox:
    def test_text_wraps_onto_a_panel_and_reads_each_side(self, sedan):
        asm = assemble(sedan, {"defaults": False, "assign": {
            "panel_door_front": {"component": "text.box", "options": {"text": "TAXI", "size": 0.12, "font": "serif-bold"}}}}, Palette(), 1)
        by = {i.connector.name: i for i in asm.instances}
        for side, sign in (("L", -1), ("R", 1)):
            m = by[f"panel_door_front_{side}"].result.mesh
            first, last = m.vertices[m.groups["glyph_0"]][:, 0].mean(), m.vertices[m.groups["glyph_3"]][:, 0].mean()
            assert sign * (last - first) > 0.2, side
            assert by[f"panel_door_front_{side}"].result.info["font"] == "serif-bold"
            grid = sedan.connector(f"panel_door_front_{side}").meta["grid_points"]
            # letters hug the skin: every vertex is within the raised thickness of the nearest grid point, sideways
            assert abs(abs(m.vertices[:, 1]).max() - abs(grid[..., 1]).max()) < 0.01

    def test_flat_text_on_a_rectangle_and_options(self, sedan, ctx):
        plate = sedan.connector("plate_front")
        comp = get_component("text.box")
        r = comp.build(plate, {"text": "ABC 123", "size": 0.12, "fit": True}, ctx)
        lo, hi = r.mesh.bounds()
        assert hi[0] - lo[0] < 0.02 and hi[1] - lo[1] <= 0.85 * plate.width + 1e-6   # flat on the (x-facing) plate, shrunk to fit
        assert 0.05 < hi[2] - lo[2] < 0.10
        # reading a fascia rectangle: facing the nose, the driver's side (+y) is on your right, so text runs toward +y
        m = r.mesh
        assert m.vertices[m.groups["glyph_0"]][:, 1].mean() < m.vertices[m.groups["glyph_6"]][:, 1].mean()
        big = comp.build(plate, {"text": "ABC 123", "size": 0.12, "fit": False}, ctx).mesh
        assert big.bounds()[1][1] - big.bounds()[0][1] > plate.width
        assert big.bounds()[1][2] - big.bounds()[0][2] == pytest.approx(0.12, abs=0.01)
        two = comp.build(sedan.connector("panel_roof"), {"text": "UNIT\n42", "size": 0.2, "align": "left", "case": "upper"}, ctx)
        assert two.info["glyphs"] == 6 and two.mesh.n_faces > 100
        with pytest.raises(ValueError):
            comp.build(plate, {"text": "  "}, ctx)

    def test_decal_panel_uses_the_same_engine(self, sedan, ctx):
        r = get_component("decal.panel").build(sedan.connector("panel_hood"), {"text": "POLICE", "font": "sans-condensed-bold", "text_height": 0.16}, ctx)
        assert r.info["font"] == "sans-condensed-bold" and r.info["reading"] == "front" and "glyph_5" in r.mesh.groups

    def test_outline_and_text_blocks(self, sedan, ctx):
        comp = get_component("text.box")
        plain = comp.build(sedan.connector("panel_door_front_L"), {"text": "POLICE", "size": 0.14}, ctx).mesh
        out = comp.build(sedan.connector("panel_door_front_L"), {"text": "POLICE", "size": 0.14, "outline": 0.006, "outline_color": "#ffffff"}, ctx).mesh
        assert "outline/glyph_0" in out.groups and "glyph_0" in out.groups
        assert {"text_f4f4f0", "text_ffffff"} <= set(out.materials)
        lo_p, hi_p = plain.bounds()
        lo_o, hi_o = out.bounds()
        assert hi_o[2] - lo_o[2] > hi_p[2] - lo_p[2] + 0.010     # 6 mm on each side, a little chord loss
        # the outline sits under the letters, the letters on top
        assert out.vertices[out.groups["glyph_0"]][:, 1].max() > out.vertices[out.groups["outline/glyph_0"]][:, 1].max()
        r = get_component("decal.panel").build(sedan.connector("panel_door_front_L"), {"texts": [
            {"text": "CITY OF SPRINGFIELD", "size": 0.05, "y": 0.55}, {"text": "POLICE", "size": 0.13, "y": 0.34, "slant_deg": 12}]}, ctx)
        assert len(r.info["texts"]) == 2 and "text0/glyph_0" in r.mesh.groups and "text1/glyph_5" in r.mesh.groups
        with pytest.raises(ValueError):
            get_component("decal.panel").build(sedan.connector("panel_door_front_L"), {"texts": [{"text": "X", "font_size": 1}]}, ctx)

    def test_plates_carry_real_characters(self, sedan, ctx):
        plate = get_component("plate.standard")
        us = plate.build(sedan.connector("plate_front"), {"region": "us", "text": "ABC 1234"}, ctx).mesh
        assert {"number/glyph_0", "number/glyph_7"} <= set(us.groups) and "number/glyph_3" not in us.groups   # the space
        gov = plate.build(sedan.connector("plate_rear"), {"style": "government", "text": "PD 400"}, ctx).mesh
        assert "band/glyph_0" in gov.groups and "govband" in [z for z in gov.zones] or "plate_band_text" in gov.materials
        # no text option: a number is drawn per car and shared by both plates through the hints
        a = plate.build(sedan.connector("plate_front"), {"region": "eu"}, ctx).mesh
        b = plate.build(sedan.connector("plate_rear"), {"region": "eu"}, ctx).mesh
        glyphs = lambda m: {g for g in m.groups if g.startswith("number/")}  # noqa: E731
        assert ctx.hints["plate_number"].count(" ") == 2 and glyphs(a) == glyphs(b) and len(glyphs(a)) == 7

    def test_decals_in_two_colours_survive_the_car_merge(self, sedan):
        asm = assemble(sedan, {"defaults": False, "assign": {
            "panel_door_front_L": {"component": "text.box", "options": {"text": "A", "color": "#112233"}},
            "panel_deck": {"component": "text.box", "options": {"text": "B", "color": "#ffffff"}}}}, Palette(), 1)
        full = asm.mesh()
        assert np.allclose(full.materials["text_112233"].color, (0x11 / 255, 0x22 / 255, 0x33 / 255))
        assert np.allclose(full.materials["text_ffffff"].color, (1, 1, 1))
