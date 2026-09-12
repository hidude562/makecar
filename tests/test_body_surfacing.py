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
