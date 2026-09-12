"""Tests for makecar.connectors."""
from __future__ import annotations

import json

import numpy as np
import pytest

from makecar.connectors import (
    Connector, PointConnector, PolygonConnector, RectangleConnector, CircleConnector, CONNECTOR_TYPES,
)
from makecar.geometry.frame import Frame

from conftest import is_right_handed


def _frame():
    return Frame.from_normal([1.0, 2.0, 0.5], [0.0, 1.0, 0.0], x_hint=[1.0, 0.0, 0.0])


# --------------------------------------------------------------- hierarchy
class TestHierarchy:
    def test_accepts_follows_class_hierarchy(self):
        assert PolygonConnector.accepts(RectangleConnector)
        assert PolygonConnector.accepts(CircleConnector)
        assert PolygonConnector.accepts(PolygonConnector)
        assert not CircleConnector.accepts(RectangleConnector)
        assert not RectangleConnector.accepts(CircleConnector)
        assert not CircleConnector.accepts(PolygonConnector)
        assert Connector.accepts(PointConnector) and Connector.accepts(CircleConnector)
        assert not PointConnector.accepts(PolygonConnector)
        assert not PolygonConnector.accepts(PointConnector)

    def test_kinds_and_registry(self):
        assert PointConnector.kind == "point" and PolygonConnector.kind == "polygon"
        assert RectangleConnector.kind == "rectangle" and CircleConnector.kind == "circle"
        assert CONNECTOR_TYPES == {"point": PointConnector, "polygon": PolygonConnector,
                                   "rectangle": RectangleConnector, "circle": CircleConnector}

    def test_basic_properties(self):
        c = PointConnector("p", _frame(), tags=["a", "b"], meta={"k": 1}, owner="me")
        assert np.allclose(c.origin, [1, 2, 0.5])
        assert np.allclose(c.normal, [0, 1, 0])
        assert c.tags == {"a", "b"} and c.meta == {"k": 1} and c.owner == "me"
        assert c.side() == "left"
        assert PointConnector("q", Frame.identity((0, -1, 0))).side() == "right"
        assert PointConnector("r", Frame.identity()).side() == "center"
        assert PointConnector("s", Frame.identity(), meta={"side": "right"}).side() == "right"


# ------------------------------------------------------------- rectangles
class TestRectangleConnector:
    def test_area_and_local_points(self):
        r = RectangleConnector("r", _frame(), 0.4, 0.2)
        assert r.width == 0.4 and r.height == 0.2
        assert r.area == pytest.approx(0.08)
        lp = r.local_points()
        assert lp.shape == (4, 2)
        assert np.allclose(lp, [[-0.2, -0.1], [0.2, -0.1], [0.2, 0.1], [-0.2, 0.1]])
        assert r.extents() == pytest.approx((0.4, 0.2))
        assert np.allclose(r.centroid, r.origin)

    def test_points_are_in_world_space(self):
        f = _frame()
        r = RectangleConnector("r", f, 0.4, 0.2)
        assert r.points.shape == (4, 3)
        # all corners lie in the frame's XY plane through the origin
        assert np.allclose((r.points - f.origin) @ f.z_axis, 0)
        assert np.allclose(f.to_local(r.points)[:, 2], 0)
        assert np.allclose(r.points.mean(axis=0), f.origin)

    def test_to_dict(self):
        d = RectangleConnector("r", _frame(), 0.4, 0.2, tags=["t"]).to_dict()
        assert d["kind"] == "rectangle" and d["width"] == 0.4 and d["height"] == 0.2
        assert d["area"] == pytest.approx(0.08) and len(d["points"]) == 4
        json.dumps(d)


# ---------------------------------------------------------------- circles
class TestCircleConnector:
    def test_area_and_geometry(self):
        c = CircleConnector("c", _frame(), 0.33)
        assert c.radius == 0.33 and c.diameter == pytest.approx(0.66)
        assert c.area == pytest.approx(np.pi * 0.33 ** 2, rel=1e-2)
        assert len(c.points) == 32
        assert np.allclose(np.linalg.norm(c.points - c.origin, axis=1), 0.33)
        assert np.allclose(c.centroid, c.origin, atol=1e-12)
        assert c.width == pytest.approx(0.66) and c.height == pytest.approx(0.66)

    def test_area_converges_with_segments(self):
        c = CircleConnector("c", Frame.identity(), 1.0, segments=720)
        assert c.area == pytest.approx(np.pi, rel=1e-4)  # inscribed 720-gon: relative error ~ (2pi/n)^2 / 6
        assert len(c.points) == 720

    def test_to_dict(self):
        d = CircleConnector("c", _frame(), 0.25).to_dict()
        assert d["kind"] == "circle" and d["radius"] == 0.25
        json.dumps(d)


# --------------------------------------------------------------- polygons
class TestPolygonConnector:
    SQUARE = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=float) + [5, 5, 1]

    def test_from_points_newell_normal(self):
        p = PolygonConnector.from_points("p", self.SQUARE)
        assert np.allclose(p.normal, [0, 0, 1])
        assert np.allclose(p.origin, self.SQUARE.mean(axis=0))
        assert p.area == pytest.approx(2.0)
        assert p.width == pytest.approx(2.0) and p.height == pytest.approx(1.0)  # PCA picks the long side as X
        assert abs(abs(p.frame.x_axis[0]) - 1.0) < 1e-9
        assert np.allclose(p.centroid, self.SQUARE.mean(axis=0))
        assert is_right_handed(p.frame)

    def test_from_points_normal_flips_to_hint_and_reverses_points(self):
        p = PolygonConnector.from_points("p", self.SQUARE, normal_hint=[0, 0, -1], x_hint=[1, 0, 0])
        assert np.allclose(p.normal, [0, 0, -1])
        # winding is reversed (restarting at the same first corner) so the Newell normal agrees with the hint
        assert np.allclose(p.points[0], self.SQUARE[0])
        assert np.allclose(np.sort(p.points, axis=0), np.sort(self.SQUARE, axis=0))
        newell = np.zeros(3)
        for a, b in zip(p.points, np.roll(p.points, -1, axis=0)):
            newell += np.cross(a, b)
        assert newell @ np.array([0, 0, -1.0]) > 0
        q = PolygonConnector.from_points("q", self.SQUARE, normal_hint=[0.1, 0.2, 0.9], x_hint=[1, 0, 0])
        assert np.allclose(q.normal, [0, 0, 1])
        assert np.allclose(q.points, self.SQUARE)
        assert np.allclose(q.frame.x_axis, [1, 0, 0])
        assert p.area == pytest.approx(q.area)

    def test_from_points_x_hint_is_projected(self):
        p = PolygonConnector.from_points("p", self.SQUARE, x_hint=[0, 1, 1])
        assert np.allclose(p.frame.x_axis, [0, 1, 0])
        assert np.allclose(p.local_points()[:, 0], (self.SQUARE - p.origin)[:, 1])

    def test_from_points_tilted_polygon(self):
        f = Frame.from_normal([0, 0, 2], [1, 1, 1], [0, 0, 1])
        pts = f.to_world(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float))
        p = PolygonConnector.from_points("p", pts, normal_hint=f.z_axis, x_hint=f.x_axis)
        assert np.allclose(p.normal, f.z_axis)
        assert np.allclose(p.frame.x_axis, f.x_axis)
        assert p.area == pytest.approx(4.0)
        assert np.allclose(np.abs(p.local_points()), 1.0)
        # without an x hint PCA picks *some* in-plane axis; the area is invariant
        q = PolygonConnector.from_points("q", pts, normal_hint=f.z_axis)
        assert q.area == pytest.approx(4.0) and abs(np.dot(q.frame.x_axis, f.z_axis)) < 1e-9

    def test_degenerate_points_default_to_z(self):
        p = PolygonConnector.from_points("line", np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float))
        assert np.allclose(p.normal, [0, 0, 1])

    def test_to_dict_is_json_serialisable_with_numpy_meta(self):
        p = PolygonConnector.from_points("p", self.SQUARE, tags=["glass", "side"],
                                         meta={"grid": [3, 4], "path": np.arange(6.0).reshape(2, 3),
                                               "f": np.float64(1.5), "i": np.int64(3), "flag": True})
        d = p.to_dict()
        s = json.dumps(d)
        back = json.loads(s)
        assert back["tags"] == ["glass", "side"] and back["kind"] == "polygon"
        assert back["meta"]["path"] == [[0, 1, 2], [3, 4, 5]]
        assert back["meta"]["f"] == 1.5 and back["meta"]["i"] == 3 and back["meta"]["flag"] is True
        assert len(back["points"]) == 4 and back["area"] == pytest.approx(2.0, abs=1e-4)
        assert set(back["frame"]) == {"origin", "x_axis", "y_axis", "z_axis"}


# --------------------------------------------------------------- matching
class TestMatching:
    def test_matches_name_tag_and_glob(self):
        c = PointConnector("seat_front_driver", Frame.identity(), tags=["seat", "seat_front"])
        assert c.matches("seat_front_driver")
        assert c.matches("seat")
        assert c.matches("seat_front")
        assert c.matches("seat_*")
        assert c.matches("*driver")
        assert c.matches("seat_front_?river")
        assert not c.matches("seat_row2")
        assert not c.matches("wheel*")
        assert not c.matches("Seat_front_driver")  # case-sensitive
        assert not c.matches("seat_")


# --------------------------------------------------------------- mirroring
class TestMirrored:
    def test_point_mirrored(self):
        c = PointConnector("mirror_L", _frame(), tags=["mirror"], meta={"side": "left", "door": "front"})
        m = c.mirrored("mirror_R")
        assert isinstance(m, PointConnector) and m.name == "mirror_R"
        assert np.allclose(m.origin, [1, -2, 0.5])
        assert np.allclose(m.normal, [0, -1, 0])
        assert is_right_handed(m.frame)
        assert m.meta["side"] == "right" and m.meta["door"] == "front"
        assert m.tags == {"mirror"} and m.tags is not c.tags
        assert m.meta is not c.meta and m.owner == c.owner
        # original untouched
        assert c.name == "mirror_L" and c.meta["side"] == "left" and np.allclose(c.origin, [1, 2, 0.5])
        assert m.mirrored("back").meta["side"] == "left"
        assert PointConnector("c", Frame.identity()).mirrored("d").meta.get("side") is None

    def test_polygon_mirrored_flips_points_and_reverses_order(self):
        pts = np.array([[0, 0.5, 0], [1, 0.5, 0], [1, 1.5, 0.3], [0, 1.5, 0.3]])
        p = PolygonConnector.from_points("p", pts, normal_hint=[0, 1, 0], meta={"side": "left"})
        stored = p.points.copy()  # from_points may have reversed the input to honour the normal hint
        m = p.mirrored("m")
        assert isinstance(m, PolygonConnector)
        assert np.allclose(m.points, (stored * [1, -1, 1])[::-1])
        assert np.allclose(m.normal, p.normal * [1, -1, 1])
        assert m.area == pytest.approx(p.area)
        assert np.allclose(m.centroid, p.centroid * [1, -1, 1])
        assert m.meta["side"] == "right"
        assert np.allclose(p.points, stored)  # original untouched

    def test_rectangle_and_circle_mirrored_keep_dimensions(self):
        r = RectangleConnector("r", _frame(), 0.4, 0.2, meta={"side": "left"})
        rm = r.mirrored("rm")
        assert isinstance(rm, RectangleConnector)
        assert rm.width == 0.4 and rm.height == 0.2 and rm.area == pytest.approx(0.08)
        assert np.allclose(rm.origin, [1, -2, 0.5])
        c = CircleConnector("c", _frame(), 0.3, meta={"side": "right"})
        cm = c.mirrored("cm")
        assert isinstance(cm, CircleConnector) and cm.radius == 0.3
        assert cm.meta["side"] == "left"
        assert np.allclose(np.linalg.norm(cm.points - cm.origin, axis=1), 0.3)
