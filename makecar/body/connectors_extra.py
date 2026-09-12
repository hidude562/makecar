"""Extension point for additional body connectors.

`extra_exterior_connectors` and `extra_interior_connectors` are called by
`emit_connectors` / `interior_connectors` after the built-in set.  Detail
work (fog lamps, wipers, seat belts, ...) adds connectors here so several
work streams can extend the body without editing `connectors.py`.
Both receive the morphed mesh, the measurements dict and the hints dict and
return a list of Connector objects (world frame).
"""
from __future__ import annotations

from typing import Dict, List

from ..connectors import Connector
from ..geometry.mesh import Mesh


def extra_exterior_connectors(mesh: Mesh, meas: Dict[str, float], hints: Dict) -> List[Connector]:
    return []


def extra_interior_connectors(mesh: Mesh, meas: Dict[str, float], hints: Dict) -> List[Connector]:
    return []
