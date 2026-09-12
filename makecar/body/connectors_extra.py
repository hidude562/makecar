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
    """Cabin-side trim paths and shoulder-belt mounts, all following the loft.

    Path metadata is world-space.  Pillar +Z faces into the cabin; belt mounts
    carry the matching seat name, leaving buckle placement to the fitted seat.
    Existing body connector names and frames are deliberately left untouched.
    """
    import numpy as np
    from ..connectors import PointConnector
    from ..geometry.frame import Frame
    from .connectors import aperture_loop, ring_pts, J, station_index, mirror_index

    out = []
    _, _, _, windshield = aperture_loop(mesh, "aperture/windshield")
    base = windshield[np.argmax(windshield[:, :, 0].mean(axis=1))].copy()
    base += np.array([-0.025, 0.0, -0.025])
    out.append(PointConnector("cowl_trim", Frame.from_normal(base.mean(axis=0), [0, 0, 1]),
                              tags=["cowl_trim", "dashboard"], meta={"path": base}))
    floor = meas["z_floor"] + 0.05
    hp = float(hints.get("h_point_height", np.clip(0.17 + (meas["z_roof_front"] - 1.2) * 0.4, 0.17, 0.38)))
    for side, sign in (("L", 1.0), ("R", -1.0)):
        col = int(np.argmax(windshield[:, :, 1].mean(axis=0) * sign))
        path = windshield[:, col].copy()
        path += np.array([-0.02, -sign * 0.025, -0.025])
        normal = np.array([0.0, -sign, 0.0])
        out.append(PointConnector(f"pillar_trim_A_{side}", Frame.from_normal(path.mean(axis=0), normal),
                                  tags=["pillar_trim", "bulkhead"], meta={"pillar": "A", "side": side, "path": path, "width": 0.065}))
        js = list(range(J["D"], J["G"] + 1))
        if sign < 0:
            js = [mirror_index(j) for j in js]
        path = (ring_pts(mesh, station_index("bp_r"))[js] + ring_pts(mesh, station_index("bp_f"))[js]) / 2
        path += np.array([0.0, -sign * 0.08, -0.035])
        foot = path[0].copy()
        foot[2] = floor
        path = np.vstack([foot, path])
        out.append(PointConnector(f"pillar_trim_B_{side}", Frame.from_normal(path.mean(axis=0), normal),
                                  tags=["pillar_trim", "bulkhead"], meta={"pillar": "B", "side": side, "path": path, "width": 0.10}))
        height = float(np.clip(floor + hp + 0.57, path[0, 2] + 0.2, path[-1, 2] - 0.06))
        anchor = np.array([np.interp(height, path[:, 2], path[:, k]) for k in range(3)])
        anchor[1] -= sign * 0.016
        role = "driver" if (sign > 0) == (hints.get("drive", "left") == "left") else "passenger"
        xh = meas["x_cowl"] - float(hints.get("front_h_point_offset", 0.95))
        seat_y = min(0.42, meas["y_shoulder"] - 0.44)
        buckle = [xh + 0.025, sign * (seat_y - 0.275), floor + hp - 0.005]
        out.append(PointConnector(f"belt_anchor_{side}", Frame.from_normal(anchor, normal, x_hint=(1, 0, 0)),
                                  tags=["belt_anchor", "seat"], meta={"side": side, "seat": f"seat_front_{role}",
                                                            "belt_width": 0.047, "path": path, "buckle": buckle}))
    return out
