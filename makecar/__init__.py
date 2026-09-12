"""makecar — a MakeHuman-style procedural car modeller.

The car *body* is a morphable mesh: a fixed-topology base mesh plus a library of
blendable *targets* (per-vertex displacement fields), exactly like MakeHuman.
The body emits *connectors* (points, polygons, rectangles, circles) that describe
where and how big things should be.  *Components* (wheels, seats, dashboards,
glass ...) are themselves morphable meshes whose target weights are fitted to the
connector they are attached to.  Everything is driven from a config file.
"""

__version__ = "0.1.0"

from .geometry.mesh import Mesh  # noqa: F401
from .geometry.frame import Frame  # noqa: F401
from .connectors import (  # noqa: F401
    Connector,
    PointConnector,
    PolygonConnector,
    RectangleConnector,
    CircleConnector,
)
from .morph.target import Target, Modifier  # noqa: F401
from .morph.morphable import MorphableMesh  # noqa: F401
