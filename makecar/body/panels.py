"""Body panel zones (doors, roof, hood, deck, fenders) as face sets, for two-tone liveries.

Faces of the loft are addressed by (station, ring index), exactly like the
vertices, so the zones follow the sliders the same way connectors do.
"""
from __future__ import annotations

from typing import Dict
import numpy as np

from ..geometry.mesh import Mesh, Material
from .generator import RING, RING_N, N_STATIONS, mirror_index, station_index

J = RING
ZONES = ("doors", "roof", "hood", "deck", "fenders", "pillars")


def _face(mesh: Mesh, station: int, j: int) -> int:
    return (station + mesh.meta["ring_offset"]) * RING_N + j


def panel_faces(mesh: Mesh, meas: Dict[str, float], hints: Dict) -> Dict[str, np.ndarray]:
    """Loft face indices per named panel zone (only faces currently painted)."""
    st = station_index
    door_count = int(hints.get("door_count", 4))
    xs = np.array([mesh.vertices[_face_vertex(mesh, i)][0] for i in range(N_STATIONS)])
    x_hi = meas["wheel_front_x"] - meas["arch_front_r"] - 0.06          # front edge of the front door
    x_lo = meas["wheel_rear_x"] + meas["arch_rear_r"] + 0.02 if door_count >= 4 else meas["x_bpillar"]
    lower_js = list(range(J["D"], J["F"])) + [mirror_index(k) - 1 for k in range(J["D"], J["F"])]
    upper_js = list(range(J["F"], J["G"])) + [mirror_index(k) - 1 for k in range(J["F"], J["G"])]
    top_g = range(J["G"], mirror_index(J["G"]))
    top_f = range(J["F"], mirror_index(J["F"]))
    zones: Dict[str, list] = {z: [] for z in ZONES}
    for i in range(N_STATIONS - 1):
        xm = 0.5 * (xs[i] + xs[i + 1])
        in_doors = x_lo <= xm <= x_hi
        in_cabin = st("roof_rear") <= i < st("roof_front")
        zones["doors" if in_doors else "fenders"] += [_face(mesh, i, j) for j in lower_js]
        if in_cabin:
            # belt -> roof rail is the door frame on a door, otherwise the B pillar; the top is the roof
            zones["doors" if in_doors else "pillars"] += [_face(mesh, i, j) for j in upper_js]
            zones["roof"] += [_face(mesh, i, j) for j in top_g]
        else:
            # ahead of / behind the cabin the whole top (shoulder to shoulder) is one panel
            top = "hood" if i >= st("cowl") else "deck" if i < st("deck") else "pillars"
            zones[top] += [_face(mesh, i, j) for j in top_f]
    painted = np.array([m == "paint" for m in mesh.face_materials])
    return {z: np.array([f for f in idx if painted[f]], dtype=int) for z, idx in zones.items()}


def _face_vertex(mesh: Mesh, station: int) -> int:
    return (station + mesh.meta["ring_offset"]) * RING_N + J["H"]


def apply_livery(mesh: Mesh, meas: Dict[str, float], hints: Dict, livery: Dict[str, str], secondary: str) -> Dict[str, int]:
    """Repaint zones.  Values are 'primary', 'secondary' or a colour; returns faces repainted per zone."""
    zones = panel_faces(mesh, meas, hints)
    done: Dict[str, int] = {}
    for zone, value in (livery or {}).items():
        if zone not in zones:
            raise KeyError(f"unknown livery zone {zone!r}; choose from {list(ZONES)}")
        if value in (None, "primary"):
            continue
        color = secondary if value == "secondary" else str(value)
        name = "paint_secondary" if value == "secondary" else f"paint_{zone}"
        if name not in mesh.materials:
            mesh.materials[name] = Material(name, Material.parse_color(color), 1.0, 0.85)
        mesh.set_material(name, zones[zone])
        done[zone] = int(len(zones[zone]))
    return done
