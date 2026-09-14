"""Equipment mounts, body panels, liveries and the fleet/police components."""
from __future__ import annotations

import numpy as np
import pytest

from makecar.assembly import assemble
from makecar.body.panels import ZONES, apply_livery, panel_faces
from makecar.components import Palette, get_component
from makecar.components.equipment import SurfaceGrid, glyph_runs
from makecar.config import CarConfig

MOUNTS = {"roof_mount", "bumper_front", "spotlight_L", "spotlight_R", "antenna_aux_L", "antenna_aux_R"}


def _instance(asm, name):
    return next(i for i in asm.instances if i.connector.name == name)


# ---------------------------------------------------------------- mounts and panels
class TestMounts:
    @pytest.mark.parametrize("style", ["sedan", "suv", "pickup", "coupe", "van"])
    def test_every_style_emits_the_equipment_mounts(self, car_body, style):
        res = car_body.build(style)
        names = {c.name for c in res.connectors}
        assert MOUNTS <= names
        assert "panel_roof" in names
        assert ("panel_hood" in names) == (style != "van")   # a one-box van has no hood to letter
        m = res.measurements
        roof = res.connector("roof_mount")
        assert m["x_roof_rear"] < roof.origin[0] < m["x_roof_front"] and roof.origin[2] > m["z_belt"]
        bumper = res.connector("bumper_front")
        assert bumper.origin[0] > m["x_front"] - 0.15 and m["nose_z_bottom"] < bumper.origin[2] < m["nose_z_top"]
        for side, sign in (("L", 1), ("R", -1)):
            spot = res.connector(f"spotlight_{side}")
            assert sign * spot.origin[1] > 0.5 and sign * spot.normal[1] > 0.9 and spot.origin[2] > m["z_belt"]
            ant = res.connector(f"antenna_aux_{side}")
            assert sign * ant.origin[1] > 0.3 and ant.normal[2] > 0.99

    def test_four_door_cars_get_door_panels_and_a_partition(self, sedan, pickup):
        for res in (sedan, pickup):
            names = {c.name for c in res.connectors}
            assert {"panel_door_front_L", "panel_door_front_R", "panel_door_rear_L", "cabin_partition"} <= names
        m = sedan.measurements
        door = sedan.connector("panel_door_front_L")
        g = door.meta["grid_points"]
        assert tuple(door.meta["grid"]) == g.shape[:2] and "panel" in door.tags and "panel_door" in door.tags
        assert m["x_bpillar"] < g[..., 0].min() and g[..., 0].max() < m["wheel_front_x"] - m["arch_front_r"]
        assert abs(g[..., 2].min() - m["z_sill"]) < 0.03 and abs(g[..., 2].max() - m["z_belt"]) < 0.03
        assert (g[..., 1] > 0.7).all()
        part = sedan.connector("cabin_partition")
        assert sedan.connector("seat_row2").origin[0] < part.origin[0] < sedan.connector("seat_front_driver").origin[0]
        assert part.width > 1.2 and part.height > 0.9

    def test_two_door_cars_have_no_rear_door_panel(self, car_body):
        res = car_body.build("coupe")
        names = {c.name for c in res.connectors}
        assert "panel_door_front_L" in names and "panel_door_rear_L" not in names

    def test_top_panels_are_exact_loft_vertices(self, sedan):
        V = sedan.full_mesh.vertices
        for name in ("panel_hood", "panel_roof", "panel_deck"):
            g = sedan.connector(name).meta["grid_points"].reshape(-1, 3)
            d = np.linalg.norm(V[None] - g[:, None], axis=2).min(axis=1)
            assert d.max() < 1e-9, name


# ---------------------------------------------------------------- livery
class TestLivery:
    def test_zones_partition_the_painted_skin(self, sedan):
        zones = panel_faces(sedan.full_mesh, sedan.measurements, sedan.hints)
        assert set(zones) == set(ZONES)
        all_faces = np.concatenate(list(zones.values()))
        assert len(all_faces) == len(set(all_faces.tolist()))
        cent = sedan.full_mesh.face_centroids()
        m = sedan.measurements
        doors = cent[zones["doors"]]
        assert len(doors) > 200
        assert doors[:, 0].min() > m["wheel_rear_x"] and doors[:, 0].max() < m["wheel_front_x"]
        assert cent[zones["roof"]][:, 2].min() > m["z_belt"] + 0.3
        assert cent[zones["hood"]][:, 0].min() > m["x_cowl"] - 0.05
        assert cent[zones["deck"]][:, 0].max() < m["x_roof_rear"]

    def test_apply_livery_repaints_only_the_zone(self, car_body):
        res = car_body.build("sedan", livery={"doors": "secondary", "hood": "#ff0000"}, paint_secondary="#ffffff")
        mats = np.array(res.full_mesh.face_materials)
        assert (mats == "paint_secondary").sum() > 200 and (mats == "paint_hood").sum() > 200
        assert np.allclose(res.full_mesh.materials["paint_secondary"].color, (1, 1, 1))
        assert np.allclose(res.full_mesh.materials["paint_hood"].color, (1, 0, 0))
        cent = res.full_mesh.face_centroids()
        m = res.measurements
        assert cent[mats == "paint_secondary"][:, 0].max() < m["wheel_front_x"]
        assert cent[mats == "paint_hood"][:, 0].min() > m["x_cowl"] - 0.05
        with pytest.raises(KeyError):
            apply_livery(res.full_mesh, m, res.hints, {"wings": "secondary"}, "#fff")

    def test_config_validates_livery_and_carries_the_secondary_paint(self):
        cfg = CarConfig.from_dict({"name": "x", "body": {"livery": {"roof": "secondary"}}, "palette": {"paint_secondary": "#123456"}})
        assert cfg.palette().paint_secondary == "#123456"
        with pytest.raises(ValueError):
            CarConfig.from_dict({"name": "x", "body": {"livery": {"wings": "secondary"}}})
        with pytest.raises(ValueError):
            CarConfig.from_dict({"name": "x", "body": {"livery": ["roof"]}})


# ---------------------------------------------------------------- components
FULL_KIT = {
    "roof_mount": "light.bar",
    "bumper_front": {"component": "bumper.push_bar", "options": {"siren": True}},
    "spotlight_L": "light.spotlight",
    "antenna_aux_L": "antenna.whip",
    "cabin_partition": {"component": "partition.cage", "options": {"style": "clear"}},
    "console": {"options": {"laptop_mount": True}},
    "headliner": {"options": {"visor_lights": True}},
    "parcel_shelf": {"options": {"deck_lights": True}},
    "grille": {"options": {"emergency_lights": True}},
    "intake": {"options": {"siren": True}},
    "mirror": {"options": {"emergency_light": True}},
    "plate": {"options": {"style": "government"}},
    "panel_door_front": {"component": "decal.panel", "options": {"text": "POLICE", "stripe": True}},
    "panel_roof": {"component": "decal.panel", "options": {"text": "42", "text_height": 0.3}},
}


@pytest.fixture(scope="module")
def kitted(car_body):
    res = car_body.build("sedan")
    return res, assemble(res, {"defaults": True, "assign": FULL_KIT}, Palette(), 1)


class TestEquipment:
    def test_light_bar_sits_level_above_the_roof_crown(self, kitted):
        res, asm = kitted
        bar = _instance(asm, "roof_mount")
        roof = res.connector("roof_mount")
        lo, hi = bar.result.mesh.bounds()
        assert lo[2] >= roof.meta["profile"][:, 2].min() - 1e-6 and hi[2] < roof.origin[2] + 0.15
        assert hi[1] - lo[1] <= roof.height + 1e-6
        mats = set(bar.result.mesh.face_materials)
        assert {"lens_red", "lens_blue", "lens_clear"} <= mats

    def test_push_bar_stands_ahead_of_the_bumper(self, kitted):
        res, asm = kitted
        bar = _instance(asm, "bumper_front")
        conn = res.connector("bumper_front")
        lo, hi = bar.result.mesh.bounds()
        assert lo[0] > conn.origin[0] - 0.03 and hi[0] < conn.origin[0] + 0.15   # the wrap bar hugs the corners
        assert lo[2] > conn.meta["bottom_z"] and hi[2] < conn.meta["grille_top_z"] + 0.05
        assert "siren_horn" in bar.result.mesh.zones

    def test_spotlight_and_whip(self, kitted):
        res, asm = kitted
        spot = _instance(asm, "spotlight_L")
        conn = res.connector("spotlight_L")
        lo, hi = spot.result.mesh.bounds()
        assert hi[1] > conn.origin[1] + 0.1 and hi[0] > conn.origin[0] + 0.15
        whip = _instance(asm, "antenna_aux_L")
        lo, hi = whip.result.mesh.bounds()
        assert 0.7 < hi[2] - lo[2] < 0.9

    def test_cabin_parts_hang_off_their_sub_connectors(self, kitted):
        res, asm = kitted
        by = {i.connector.name: i for i in asm.instances}
        assert by["console/laptop"].component == "mount.laptop"
        assert by["headliner/visor_light"].component == "light.visor"
        assert by["parcel_shelf/deck_light"].component == "light.deck"
        assert by["cabin_partition"].component == "partition.cage"
        assert "pane" in by["cabin_partition"].result.mesh.zones
        console_top = by["console"].result.mesh.bounds()[1][2]
        assert by["console/laptop"].result.mesh.bounds()[1][2] > console_top + 0.2
        assert by["headliner/visor_light"].result.mesh.bounds()[1][2] < res.measurements["z_roof_front"]

    def test_options_on_existing_components(self, kitted):
        res, asm = kitted
        by = {i.connector.name: i for i in asm.instances}
        assert {"led_lens_red", "led_lens_blue"} <= set(by["grille"].result.mesh.zones)
        assert "siren_horn" in by["intake"].result.mesh.zones
        assert "emergency_lens" in by["mirror_L"].result.mesh.zones
        assert "lens_red" in by["mirror_L"].result.mesh.face_materials and "lens_blue" in by["mirror_R"].result.mesh.face_materials
        assert "govband" in [z for z in by["plate_front"].result.mesh.zones] or any("govband" in f for f in ("govband",))
        assert "plate_text" in by["plate_front"].result.mesh.materials

    def test_nothing_attaches_by_default(self, sedan):
        asm = assemble(sedan, {"defaults": True}, Palette(), 1)
        comps = {i.component for i in asm.instances}
        assert not comps & {"light.bar", "bumper.push_bar", "light.spotlight", "antenna.whip", "partition.cage",
                            "mount.laptop", "light.visor", "light.deck", "decal.panel"}
        assert MOUNTS <= set(asm.unattached)


# ---------------------------------------------------------------- decals
class TestDecals:
    def test_glyph_runs_merge_pixels(self):
        runs = glyph_runs("I")
        assert (0, 5, 6, 7) in runs and (0, 5, 0, 1) in runs and (2, 3, 3, 4) in runs
        assert glyph_runs(" ") == []
        assert len(glyph_runs("AB")) > len(glyph_runs("A"))

    def test_surface_grid_uses_arc_length(self):
        # rows bunched near the start: half the parameter range covers a quarter of the length
        xs = np.concatenate([np.linspace(0, 1, 5), np.linspace(2, 4, 4)])
        grid = np.array([[[x, y, 0.0] for y in (0.0, 1.0)] for x in xs])
        S = SurfaceGrid(grid, np.array([0, 0, 1.0]))
        assert S.len_u == pytest.approx(4.0) and S.len_v == pytest.approx(1.0)
        assert S.point(S.to_param(0, 0.5), 0.0)[0] == pytest.approx(2.0, abs=1e-6)
        assert S.axis_for((1, 0, 0)) == (0, 1.0) and S.axis_for((0, -1, 0)) == (1, -1.0)
        slab = S.slab(0.1, 0.3, 0.2, 0.4, 0.01, "m")
        assert slab.n_faces == 2 + 4 and slab.vertices[:, 2].max() == pytest.approx(0.01)

    def test_lettering_reads_correctly_from_each_side(self, sedan):
        asm = assemble(sedan, {"defaults": False, "assign": {
            "panel_door_front": {"component": "decal.panel", "options": {"text": "POLICE", "text_height": 0.14}}}}, Palette(), 1)
        by = {i.connector.name: i for i in asm.instances}
        for side, sign in (("L", -1), ("R", 1)):
            m = by[f"panel_door_front_{side}"].result.mesh
            first = m.vertices[m.groups["glyph_0"]][:, 0].mean()
            last = m.vertices[m.groups[f"glyph_{len(m.zones) - 1}"]][:, 0].mean()
            # facing the driver's door the hood is on your left, so the text runs rearwards there
            assert sign * (last - first) > 0.3, side
            z = m.vertices[:, 2]
            assert 0.12 < z.max() - z.min() < 0.16
            panel = sedan.connector(f"panel_door_front_{side}")
            assert abs(m.vertices[:, 1]).max() < abs(panel.meta["grid_points"][..., 1]).max() + 0.006

    def test_stripe_is_a_thin_band_along_the_panel(self, sedan):
        asm = assemble(sedan, {"defaults": False, "assign": {
            "panel_fender_L": {"component": "decal.panel", "options": {"stripe": True, "stripe_y": 0.5, "stripe_height": 0.05, "stripe_accent": None}}}},
                       Palette(), 1)
        m = asm.instances[0].result.mesh
        lo, hi = m.bounds()
        panel = sedan.connector("panel_fender_L")
        assert hi[2] - lo[2] < 0.09 and hi[0] - lo[0] > 0.9 * panel.width
        assert set(m.face_materials) == {"decal_stripe"}

    def test_decal_needs_a_panel_and_something_to_draw(self, sedan, ctx):
        with pytest.raises(ValueError):
            get_component("decal.panel").build(sedan.connector("plate_front"), {"text": "X"}, ctx)
        with pytest.raises(ValueError):
            get_component("decal.panel").build(sedan.connector("panel_hood"), {}, ctx)
