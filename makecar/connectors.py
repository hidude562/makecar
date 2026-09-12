"""Connectors: typed attachment sites emitted by the body (and by components).

Hierarchy (a component that accepts a base type accepts all subtypes):

    Connector
     ├── PointConnector                 a location + orientation
     └── PolygonConnector               a closed 3-D outline (roughly planar)
          ├── RectangleConnector        width x height in the frame's XY plane
          └── CircleConnector           radius in the frame's XY plane

Every connector carries a `Frame` whose Z axis is the mount normal (things are
built along +Z in local space) and whose X axis is the connector's major
direction (car-forward for most interior parts, lateral for windows).
`tags` are used by the config to select which component goes where, and `meta`
carries free-form hints (e.g. tire width, seat row, driver side).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from .geometry.frame import Frame
from .geometry.curves import circle_points, polygon_area_2d, polygon_centroid_2d


class Connector:
    kind = "connector"

    def __init__(self, name: str, frame: Frame, tags: Optional[Sequence[str]] = None, meta: Optional[dict] = None,
                 owner: str = "body"):
        self.name = name
        self.frame = frame
        self.tags: Set[str] = set(tags or [])
        self.meta: Dict[str, Any] = dict(meta or {})
        self.owner = owner  # who emitted it ("body" or a component instance name)

    # ------------------------------------------------------------ geometry
    @property
    def origin(self) -> np.ndarray:
        return self.frame.origin

    @property
    def normal(self) -> np.ndarray:
        return self.frame.z_axis

    def side(self) -> str:
        return self.meta.get("side", "left" if self.origin[1] > 0.01 else "right" if self.origin[1] < -0.01 else "center")

    # ----------------------------------------------------------- matching
    def matches(self, selector: str) -> bool:
        """Selector grammar: exact name, tag, or glob on the name ('seat.*')."""
        import fnmatch

        if selector == self.name or selector in self.tags:
            return True
        return fnmatch.fnmatchcase(self.name, selector)

    @classmethod
    def accepts(cls, other_cls: type) -> bool:
        return issubclass(other_cls, cls)

    # ---------------------------------------------------------- serialise
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "owner": self.owner,
            "tags": sorted(self.tags),
            "frame": self.frame.to_dict(),
            "meta": {k: _jsonable(v) for k, v in self.meta.items() if k != "grid_points"},
        }

    def mirrored(self, new_name: str) -> "Connector":
        c = self.__class__.__new__(self.__class__)
        c.__dict__.update(self.__dict__)
        c.name = new_name
        c.frame = self.frame.mirrored_y()
        c.tags = set(self.tags)
        c.meta = dict(self.meta)
        c.meta["side"] = "right" if self.meta.get("side") == "left" else "left" if self.meta.get("side") == "right" else self.meta.get("side")
        return c

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.name!r}, origin={np.round(self.origin, 3).tolist()})"


class PointConnector(Connector):
    kind = "point"


class PolygonConnector(Connector):
    """Closed outline in 3-D.  Points are stored in world space; `local_points`
    projects them onto the frame's XY plane."""

    kind = "polygon"

    def __init__(self, name: str, frame: Frame, points_world: np.ndarray, tags=None, meta=None, owner="body"):
        super().__init__(name, frame, tags, meta, owner)
        self._points = np.asarray(points_world, dtype=float).reshape(-1, 3)

    @classmethod
    def from_points(cls, name: str, points_world, normal_hint=None, x_hint=None, tags=None, meta=None, owner="body"):
        """Build a polygon connector with a best-fit plane frame from 3-D points."""
        pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
        c = pts.mean(axis=0)
        # Newell normal
        nrm = np.zeros(3)
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            nrm += np.cross(a, b)
        if np.linalg.norm(nrm) < 1e-9:
            nrm = np.array([0.0, 0.0, 1.0])
        nrm /= np.linalg.norm(nrm)
        meta = dict(meta or {})
        if normal_hint is not None:
            hint = np.asarray(normal_hint, dtype=float)
            hint /= np.linalg.norm(hint)
            if np.dot(nrm, hint) < 0:
                # reverse the winding; a grid loop (side0=rows, side1=cols, ...) stays a valid grid loop
                # if we restart at the same corner, which swaps the roles of rows and cols
                nrm = -nrm
                pts = np.roll(pts[::-1], 1, axis=0)
                if meta.get("grid"):
                    rows, cols = meta["grid"]
                    meta["grid"] = [int(cols), int(rows)]
                if meta.get("grid_points") is not None:
                    meta["grid_points"] = np.asarray(meta["grid_points"]).transpose(1, 0, 2).copy()
            # for strongly non-planar outlines (wrap-around lamps) the hint, e.g. the mean
            # surface normal of the aperture, is the better mount normal than the Newell fit
            if np.dot(nrm, hint) < np.cos(np.radians(25.0)):
                nrm = hint
        if x_hint is None:
            # major axis via PCA in the plane
            q = pts - c
            q = q - np.outer(q @ nrm, nrm)
            u, s, vt = np.linalg.svd(q, full_matrices=False)
            x_hint = vt[0]
        frame = Frame.from_normal(c, nrm, x_hint)
        return cls(name, frame, pts, tags, meta, owner)

    @property
    def points(self) -> np.ndarray:
        return self._points

    def local_points(self) -> np.ndarray:
        return self.frame.to_local(self._points)[:, :2]

    @property
    def area(self) -> float:
        return abs(polygon_area_2d(self.local_points()))

    @property
    def centroid(self) -> np.ndarray:
        c2 = polygon_centroid_2d(self.local_points())
        return self.frame.to_world([[c2[0], c2[1], 0.0]])[0]

    def extents(self) -> Tuple[float, float]:
        lp = self.local_points()
        return float(np.ptp(lp[:, 0])), float(np.ptp(lp[:, 1]))

    @property
    def width(self) -> float:
        return self.extents()[0]

    @property
    def height(self) -> float:
        return self.extents()[1]

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["points"] = np.round(self._points, 4).tolist()
        d["width"] = round(self.width, 4)
        d["height"] = round(self.height, 4)
        d["area"] = round(self.area, 4)
        return d

    def mirrored(self, new_name: str) -> "PolygonConnector":
        c = super().mirrored(new_name)
        pts = self._points * np.array([1.0, -1.0, 1.0])
        c._points = pts[::-1].copy()
        return c  # type: ignore


class RectangleConnector(PolygonConnector):
    kind = "rectangle"

    def __init__(self, name: str, frame: Frame, width: float, height: float, tags=None, meta=None, owner="body"):
        hw, hh = width / 2.0, height / 2.0
        local = np.array([[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]], dtype=float)
        super().__init__(name, frame, frame.to_world(local), tags, meta, owner)
        self._width = float(width)
        self._height = float(height)

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["width"], d["height"] = round(self._width, 4), round(self._height, 4)
        return d


class CircleConnector(PolygonConnector):
    kind = "circle"

    def __init__(self, name: str, frame: Frame, radius: float, tags=None, meta=None, owner="body", segments: int = 32):
        c2 = circle_points(radius, segments)
        local = np.hstack([c2, np.zeros((segments, 1))])
        super().__init__(name, frame, frame.to_world(local), tags, meta, owner)
        self.radius = float(radius)

    @property
    def diameter(self) -> float:
        return 2 * self.radius

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["radius"] = round(self.radius, 4)
        return d


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return np.round(v, 5).tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


CONNECTOR_TYPES = {
    "point": PointConnector,
    "polygon": PolygonConnector,
    "rectangle": RectangleConnector,
    "circle": CircleConnector,
}
