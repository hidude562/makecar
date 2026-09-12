"""Metric regression tests for production-scale shell details (metres)."""
from dataclasses import replace

import numpy as np
import pytest

from makecar.body import BodyGenerator, BodyParams, CarBody, style_params, style_names
from makecar.body.connectors import fascia_point
from makecar.body.generator import RING, RING_N, N_FASCIA, mirror_index, station_index
from makecar.body.targets import BODY_MODIFIER_SPECS


def mesh(params=None):
    return BodyGenerator(params or BodyParams()).build()


def group(m, name):
    return m.vertices[m.groups[name]]


class TestFascia:
    def test_bumper_is_upright_and_proud(self):
        p = BodyParams()
        m = mesh(p)
        face = group(m, "fascia/front_face")
        nose = group(m, "nose_ring")
        # D is below the crease; it lies on the vertical bumper face.
        assert face[RING["D"], 0] - nose[RING["D"], 0] >= p.bumper_projection - 1e-9
        upright = face[(face[:, 2] > p.front_bumper_bottom + 0.045) &
                       (face[:, 2] <= p.bumper_crease_height)]
        assert len(upright) >= 4
        assert np.ptp(upright[:, 0]) < 1e-9
        assert N_FASCIA == 6

    def test_hood_overhang_moves_upper_grille_not_hood_tip(self):
        p = BodyParams()
        a, b = mesh(p), mesh(replace(p, hood_overhang=p.hood_overhang + 0.02))
        ga, gb = group(a, "fascia/front_face"), group(b, "fascia/front_face")
        upper = (ga[:, 2] >= p.bumper_crease_height + 0.025) & (ga[:, 2] <= p.hood_front_height - 0.035)
        assert upper.any()
        assert np.allclose(gb[upper, 0] - ga[upper, 0], -0.02)
        assert gb[RING["H"], 0] == pytest.approx(ga[RING["H"], 0])

    def test_splitter_extends_beyond_face(self):
        p = BodyParams(front_splitter=0.055)
        face = group(mesh(p), "fascia/front_face")
        assert face[RING["A"], 0] - face[RING["D"], 0] == pytest.approx(p.front_splitter)

    def test_rear_pocket_has_flat_recessed_floor(self):
        p = BodyParams(plate_recess=0.035)
        m = mesh(p)
        rim, floor = group(m, "fascia/rear_pocket_rim"), group(m, "fascia/rear_pocket_floor")
        assert np.allclose(floor[:, 0] - rim[:, 0], p.plate_recess)
        assert np.ptp(floor[:, 0]) < 1e-9
        assert group(m, "apex_rear")[0, 0] == pytest.approx(floor[0, 0])
        point = fascia_point(m, "rear", group(m, "apex_rear")[0, 2])
        assert point[0] == pytest.approx(floor[0, 0])

    def test_rear_diffuser_and_tailgate_panel_steps(self):
        p = BodyParams(diffuser_step=0.06, tailgate_panel=0.04, rear_fascia_rake=0)
        face = group(mesh(p), "fascia/rear_face")
        assert face[RING["A"], 0] - p.layout()["x_rear"] == pytest.approx(p.diffuser_step)
        assert face[RING["H"], 0] - p.layout()["x_rear"] == pytest.approx(p.tailgate_panel)

    @pytest.mark.parametrize("name,group_name", [("bumper_crease_height", "fascia/front_face"),
                                                ("rear_bumper_crease_height", "fascia/rear_face")])
    def test_crease_height_is_a_mesh_landmark(self, name, group_name):
        p = BodyParams()
        before = group(mesh(p), group_name)[RING["E"], 2]
        after = group(mesh(replace(p, **{name: getattr(p, name) + 0.02})), group_name)[RING["E"], 2]
        assert after - before == pytest.approx(0.02)

    def test_rear_plate_connector_follows_recess_target(self):
        car = CarBody()
        a = car.build()
        b = car.build(modifiers={"plate_recess": 1})
        assert b.connector("plate_rear").origin[0] - a.connector("plate_rear").origin[0] == pytest.approx(0.025, abs=1e-6)
        assert a.connector("plate_rear").normal.tolist() == [-1, 0, 0]


class TestSheetMetal:
    def test_rocker_is_vertical_with_a_measured_door_hem(self):
        p = BodyParams()
        gen = BodyGenerator(p)
        r = gen.ring(station_index("bp_f"))
        sill = r[RING["C"]:RING["D"] + 1]
        assert np.ptp(sill[:, 1]) < 1e-9
        assert np.ptp(sill[:, 2]) == pytest.approx(p.rocker_height)
        assert r[RING["D"] + 1, 1] - r[RING["D"], 1] == pytest.approx(p.door_step)
        assert r[RING["E"], 1] > sill[0, 1] + 0.03
        assert r[RING["B"], 1] < sill[0, 1]  # lower flare-in

    @pytest.mark.parametrize("height", [0.11, 0.15, 0.19])
    def test_rocker_height_changes_the_flat_face(self, height):
        r = BodyGenerator(BodyParams(rocker_height=height)).ring(station_index("bp_f"))
        assert r[RING["D"], 2] - r[RING["C"], 2] == pytest.approx(height)

    @pytest.mark.parametrize("width", [0.008, 0.020, 0.035])
    def test_arch_lip_is_a_uniform_radial_flange(self, width):
        p = BodyParams(arch_lip_width=width)
        g = BodyGenerator(p)
        for i in range(station_index("fa_start", "lower") + 1, station_index("fa_end", "lower")):
            r = g.ring(i)
            inner, outer, root = r[RING["D"]:RING["D"] + 3]
            assert np.linalg.norm(outer[[0, 2]] - inner[[0, 2]]) == pytest.approx(width)
            assert outer[1] - root[1] == pytest.approx(width * 0.6)
            assert np.hypot(inner[0] - p.wheelbase / 2, inner[2] - p.axle_height) == pytest.approx(p.arch_radius)

    @pytest.mark.parametrize("radius", [0.01, 0.025, 0.04])
    def test_shoulder_is_a_circular_quarter_radius(self, radius):
        r = BodyGenerator(BodyParams(shoulder_radius=radius)).ring(station_index("bp_f"))
        e = r[RING["E"], 1:]
        center = e + [-radius, 0]
        arc = r[RING["E"]:RING["E"] + 4, 1:]
        assert np.allclose(np.linalg.norm(arc - center, axis=1), radius)

    @pytest.mark.parametrize("depth", [0.002, 0.012, 0.024])
    def test_fender_crease_has_a_real_fold_not_a_seam_line(self, depth):
        r = BodyGenerator(BodyParams(fender_crease=depth)).ring(station_index("cowl") + 3)
        assert r[RING["F"], 2] - r[RING["F"] + 1, 2] == pytest.approx(depth)
        assert r[RING["F"], 1] - r[RING["F"] + 1, 1] == pytest.approx(0.006)


class TestGreenhouse:
    @pytest.mark.parametrize("depth", [0.002, 0.008, 0.016])
    def test_glass_and_boundaries_sink_by_normal_distance(self, depth):
        flush = mesh(BodyParams(glass_recess=0))
        recessed = mesh(BodyParams(glass_recess=depth))
        vertices = set()
        for name, (r0, r1, j0, j1) in flush.meta["aperture_rects"].items():
            if "light" in name:
                continue
            for row in range(r0, r1 + 2):
                vertices.update(row * RING_N + j for j in range(j0, j1 + 2))
        ids = sorted(vertices)
        delta = recessed.vertices - flush.vertices
        assert np.allclose(np.linalg.norm(delta[ids], axis=1), depth)
        other = np.ones(flush.n_vertices, bool)
        other[ids] = False
        assert np.allclose(delta[other], 0)  # frame surface is not sunk with the glass

    def test_apertures_have_unbroken_painted_frame_bands(self):
        m = mesh()
        for name, (r0, r1, j0, j1) in m.meta["aperture_rects"].items():
            if "light" in name:
                continue
            for row in range(r0, r1 + 1):
                assert m.face_materials[row * RING_N + j0 - 1] == "paint"
                assert m.face_materials[row * RING_N + j1 + 1] == "paint"
            for j in range(j0, j1 + 1):
                assert m.face_materials[(r0 - 1) * RING_N + j] == "paint"
                assert m.face_materials[(r1 + 1) * RING_N + j] == "paint"

    def test_side_glass_top_follows_roof_curve(self):
        m = mesh()
        r0, r1, _, j1 = m.meta["aperture_rects"]["aperture/glass_front_L"]
        top = m.vertices[np.arange(r0, r1 + 2) * RING_N + j1 + 1]
        t = (top[:, 0] - top[0, 0]) / (top[-1, 0] - top[0, 0])
        chord = top[0, 2] + (top[-1, 2] - top[0, 2]) * t
        assert len(top) >= 6
        assert np.max(np.abs(top[:, 2] - chord)) > 0.0005

    def test_a_and_c_pillar_widths_move_mesh_boundaries(self):
        p = BodyParams()
        a = mesh(p)
        b = mesh(replace(p, a_pillar_width=p.a_pillar_width + 0.025))
        assert group(b, "ring/roof_front")[RING["F"], 0] - group(a, "ring/roof_front")[RING["F"], 0] == pytest.approx(-0.025)
        c = mesh(replace(p, c_pillar_width=p.c_pillar_width + 0.04))
        assert group(c, "ring/cp_f")[RING["G"], 0] - group(a, "ring/cp_f")[RING["G"], 0] == pytest.approx(0.04)

    def test_roof_radius_and_drip_bead_do_not_inflate_roof(self):
        p = BodyParams()
        g = BodyGenerator(p)
        r = g.ring(station_index("bp_f"))
        radius = BodyGenerator(replace(p, roof_edge_radius=p.roof_edge_radius + 0.01)).ring(station_index("bp_f"))
        bead = BodyGenerator(replace(p, drip_rail=p.drip_rail + 0.004)).ring(station_index("bp_f"))
        assert radius[RING["G"], 2] - r[RING["G"], 2] == pytest.approx(-0.005)
        assert bead[RING["G"] + 1, 2] - r[RING["G"] + 1, 2] == pytest.approx(0.004)
        assert np.allclose(radius[RING["H"]], r[RING["H"]])
        assert np.allclose(bead[RING["H"]], r[RING["H"]])
