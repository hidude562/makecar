"""Equipment mounts and body panels on the morphed shell.

Every car emits these; they have no default component, so a plain car ignores
them.  Fleet equipment (light bars, push bars, spotlights, whip antennas, a
cabin partition) and decals (lettering, stripes) attach to them from the config.

Panels are polygon connectors with a `grid_points` (rows, cols, 3) surface grid,
sampled from the actual loft, so a decal can sit on the curved skin.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from ..geometry.mesh import Mesh
from ..geometry.frame import Frame
from ..connectors import Connector, PointConnector, PolygonConnector, RectangleConnector
from .generator import RING, N_STATIONS, mirror_index, station_index
from .connectors import vidx, surface_line_at_x, fascia_mount, fascia_triangles

J = RING


def _grid_loop(grid: np.ndarray) -> np.ndarray:
    """Boundary of a (rows, cols, 3) grid, laid out as fills.coons_patch expects."""
    rows, cols = grid.shape[:2]
    loop = [grid[i, 0] for i in range(rows)]
    loop += [grid[rows - 1, j] for j in range(1, cols)]
    loop += [grid[i, cols - 1] for i in range(rows - 2, -1, -1)]
    loop += [grid[0, j] for j in range(cols - 2, 0, -1)]
    return np.asarray(loop)


def _side_panel(mesh: Mesh, name: str, x0: float, x1: float, ys: float, tags: List[str],
                j_lo: int = J["D"], j_hi: int = J["E"], **meta) -> Optional[PolygonConnector]:
    """Grid on the outer skin between x0 < x1, from ring index j_lo (sill top) up to j_hi (belt)."""
    length = x1 - x0
    if length < 0.25:
        return None
    mir = (lambda j: j) if ys > 0 else mirror_index
    js = [mir(j) for j in range(j_lo, j_hi + 1)]
    # rows are the loft's own stations between the edges (exact vertices), plus the two edges on
    # mesh edges, so the grid is the skin itself and a decal never sinks into a curve between samples
    lines = [surface_line_at_x(mesh, x0, mir(j_lo), mir(j_hi))]
    for i in range(N_STATIONS):
        pts = mesh.vertices[[vidx(mesh, i, j) for j in js]]
        if x0 + 0.015 < pts[:, 0].mean() < x1 - 0.015:
            lines.append(pts)
    lines.append(surface_line_at_x(mesh, x1, mir(j_lo), mir(j_hi)))
    grid = np.array(lines)
    loop = _grid_loop(grid)
    meta = {"grid": [len(lines), grid.shape[1]], "grid_points": grid, "side": "left" if ys > 0 else "right",
            "panel": True, **meta}
    return PolygonConnector.from_points(name, loop, normal_hint=[0.0, ys, 0.0], x_hint=[1.0, 0.0, 0.0],
                                        tags=["panel"] + tags, meta=meta)


def _top_panel(mesh: Mesh, name: str, i0: int, i1: int, j_lo: int, tags: List[str], **meta) -> Optional[PolygonConnector]:
    """Grid on the top surface between stations i0 < i1 (upper chain), across j_lo .. mirror(j_lo)."""
    if i1 - i0 < 2:
        return None
    stations = list(range(i0, i1 + 1))
    js = list(range(j_lo, mirror_index(j_lo) + 1))
    grid = np.array([[mesh.vertices[vidx(mesh, i, j)] for j in js] for i in stations])
    if grid[-1, :, 0].mean() - grid[0, :, 0].mean() < 0.25:
        return None
    loop = _grid_loop(grid)
    meta = {"grid": [len(stations), len(js)], "grid_points": grid, "side": "center", "panel": True, **meta}
    return PolygonConnector.from_points(name, loop, normal_hint=[0.0, 0.0, 1.0], x_hint=[1.0, 0.0, 0.0],
                                        tags=["panel"] + tags, meta=meta)


def _top_surface_point(mesh: Mesh, x: float, y: float, stations) -> np.ndarray:
    """Point on the upper skin nearest (x, y): every top ring index is sampled at x."""
    js = range(J["F"], mirror_index(J["F"]) + 1)
    pts = np.array([surface_line_at_x(mesh, x, j, j, stations)[0] for j in js])
    k = int(np.argmin(np.abs(pts[:, 1] - y)))
    return pts[k]


def equipment_connectors(mesh: Mesh, meas: Dict[str, float], hints: Dict, existing: List[Connector]) -> List[Connector]:
    V = mesh.vertices
    X, Y, Z = np.eye(3)
    st = station_index
    out: List[Connector] = []
    door_count = int(hints.get("door_count", 4))
    has_bed = meas.get("has_bed", 0.0) > 0.5
    by_name = {c.name: c for c in existing}

    # ------------------------------------------------------------ roof mount
    # Light bars sit on the front half of the roof, behind the windshield header.
    x_rr, x_rf = meas["x_roof_rear"], meas["x_roof_front"]
    roof_len = x_rf - x_rr
    if roof_len > 0.5:
        x_mount = x_rf - min(0.45, 0.30 * roof_len)
        roof_st = range(st("roof_rear"), st("roof_front") + 1)
        js = range(J["G"], mirror_index(J["G"]) + 1)
        profile = np.array([surface_line_at_x(mesh, x_mount, j, j, roof_st)[0] for j in js])
        centre = profile[len(profile) // 2].copy()
        width_y = float(profile[0, 1] - profile[-1, 1])
        out.append(RectangleConnector("roof_mount", Frame.from_normal(centre, Z, x_hint=X), 0.40, abs(width_y),
                                      tags=["roof_mount"], meta={"profile": profile, "roof_length": roof_len}))
    # ------------------------------------------------------------ bumper front
    tris, _ = mesh.triangulated()
    front = fascia_triangles(mesh, "front", tris)
    crease_z = V[mesh.groups["fascia/front_face"]][J["E"], 2]
    z_bot = meas["nose_z_bottom"]
    bumper_h = max(0.16, min(0.34, crease_z - z_bot - 0.03))
    bumper_z = z_bot + 0.03 + bumper_h / 2
    point = fascia_mount(mesh, "front", bumper_z, 0.6, bumper_h, 0.004, front)
    out.append(RectangleConnector("bumper_front", Frame.from_normal(point, X, x_hint=-Y),
                                  min(2 * meas["nose_half_width"] * 0.9, 1.5), bumper_h,
                                  tags=["bumper", "bumper_front"],
                                  meta={"grille_top_z": float(meas["nose_z_top"] - 0.045), "bottom_z": float(z_bot)}))
    # ------------------------------------------------------------ A-pillar spotlights
    i_cowl, i_rf = st("cowl"), st("roof_front")
    for side, ys in (("L", 1.0), ("R", -1.0)):
        mir = (lambda j: j) if ys > 0 else mirror_index
        i = int(round(i_cowl - 0.42 * (i_cowl - i_rf)))
        p = V[vidx(mesh, i, mir(J["G"]))].copy()
        below = V[vidx(mesh, i, mir(J["G"] - 1))]
        n = np.array([0.0, ys, 0.0]) + 0.35 * X
        n /= np.linalg.norm(n)
        out.append(PointConnector(f"spotlight_{side}", Frame.from_normal(p + 0.006 * n, n, x_hint=X),
                                  tags=["spotlight"], meta={"side": "left" if ys > 0 else "right",
                                                            "pillar_z": float(p[2]), "belt_z": float(below[2])}))
    # ------------------------------------------------------------ aux antennas
    # Trunk lid when there is one, otherwise the rear corners of the roof.
    deck_len = meas["x_roof_rear"] - meas["x_deck"]
    if deck_len > 0.45 and not has_bed:
        x_a = meas["x_deck"] - 0.30
        stations = range(0, st("roof_rear") + 1)
    else:
        x_a = meas["x_roof_rear"] + 0.16
        stations = range(st("roof_rear"), st("roof_front") + 1)
    half = meas["roof_half_width"] - 0.16
    for side, ys in (("L", 1.0), ("R", -1.0)):
        p = _top_surface_point(mesh, x_a, ys * half, stations)
        out.append(PointConnector(f"antenna_aux_{side}", Frame.from_normal(p, Z, x_hint=X), tags=["antenna_aux"],
                                  meta={"side": "left" if ys > 0 else "right", "on_deck": bool(deck_len > 0.45 and not has_bed)}))
    # ------------------------------------------------------------ cabin partition
    seat = by_name.get("seat_front_driver")
    if seat is not None and "seat_row2" in by_name:
        floor = float(seat.origin[2])
        x_p = float(seat.origin[0]) - 0.55
        inner_half = meas["y_shoulder"] - 0.08
        z_top = float(meas["z_roof_front"] - 0.06)
        out.append(RectangleConnector("cabin_partition", Frame.from_normal([x_p, 0.0, (floor + z_top) / 2], -X, x_hint=Y),
                                      2 * inner_half + 0.04, z_top - floor, tags=["partition", "interior_detail"],
                                      meta={"floor": floor, "belt_z": float(meas["z_belt"]), "tunnel_height": 0.16}))
    # ------------------------------------------------------------ panels
    x_fd0 = meas["wheel_front_x"] - meas["arch_front_r"] - 0.06
    x_fd1 = meas["x_bpillar"]
    x_rd1 = meas["wheel_rear_x"] + meas["arch_rear_r"] + 0.02
    x_ff0 = meas["wheel_front_x"] + meas["arch_front_r"] + 0.03
    x_ff1 = meas["x_front"] - 0.30
    x_q0 = meas["x_rear"] + 0.30
    x_q1 = meas["wheel_rear_x"] - meas["arch_rear_r"] - 0.03
    for side, ys in (("L", 1.0), ("R", -1.0)):
        # the car points +x, so a door runs from its rear edge (smaller x) to its front edge
        panels = [(f"panel_door_front_{side}", x_fd1 + 0.03, x_fd0 - 0.04, ["panel_door", "panel_door_front"])]
        if door_count >= 4:
            panels.append((f"panel_door_rear_{side}", x_rd1 + 0.04, x_fd1 - 0.03, ["panel_door", "panel_door_rear"]))
        panels.append((f"panel_fender_{side}", x_ff0, x_ff1, ["panel_fender"]))
        panels.append((f"panel_quarter_{side}", x_q0, x_q1, ["panel_quarter"]))
        for name, x0, x1, tags in panels:
            c = _side_panel(mesh, name, x0, x1, ys, tags, belt_z=float(meas["z_belt"]), sill_z=float(meas["z_sill"]))
            if c is not None:
                out.append(c)
    hood = _top_panel(mesh, "panel_hood", st("cowl") + 1, N_STATIONS - 3, J["F"] + 1, ["panel_hood"])
    if hood is not None:
        out.append(hood)
    roof = _top_panel(mesh, "panel_roof", st("roof_rear") + 1, st("roof_front") - 1, J["G"] + 1, ["panel_roof"])
    if roof is not None:
        out.append(roof)
    if not has_bed and deck_len > 0.45:
        deck = _top_panel(mesh, "panel_deck", 2, st("deck") - 1, J["F"] + 1, ["panel_deck"])
        if deck is not None:
            out.append(deck)
    return out
