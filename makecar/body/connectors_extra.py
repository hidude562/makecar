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
    """Detail mounts on the morphed shell; all path metadata is in world metres.

    Furniture uses local Z outward and local Y up (side markers use X forward).
    Wiper ``paths`` / ``blade_normals`` are paired L/R samples on the exact
    windshield grid; ``path_uv`` records their normalized grid coordinates.
    Exhaust ``paths`` are [tunnel, left branch, right branch], sharing a junction
    and terminating at ``tips``, the existing exhaust connectors' exact origins.
    Diffuser X is lateral, Y forward along the floor and Z down. Mud flaps are
    opt-in via ``hints['mud_flaps']`` and use X lateral, Y up, Z rearward, with
    their origin at the rectangle centre.
    """
    import numpy as np

    from ..connectors import PointConnector, RectangleConnector
    from ..geometry.frame import Frame
    from .connectors import _face_normal_from_ring, aperture_loop, surface_line_at_x
    from .generator import RING, RING_N, N_STATIONS, mirror_index

    V = mesh.vertices
    X, Y, Z = np.eye(3)
    out = []
    sides = (("L", 1.0, "left"), ("R", -1.0, "right"))
    nose, tail = (V[mesh.groups[name]] for name in ("nose_ring", "tail_ring"))
    apex_r = V[mesh.groups["apex_rear"]][0]
    tail_n = _face_normal_from_ring(tail, -1)

    def on_plane(apex, normal, z, ahead=0.0):
        # Same raked plane used by the built-in plate/exhaust connectors.
        p = np.array([apex[0], 0.0, z])
        p[0] += np.dot(apex - p, normal) / normal[0]
        return p + ahead * normal

    # Project onto the actual rounded fascia, rather than an estimated box or
    # the apex plane (which misses the bumper corners on strongly raked noses).
    off = mesh.meta["ring_offset"]
    fascia = {}
    for position, sign in (("front", 1), ("rear", -1)):
        limit = (off + N_STATIONS - 1) * RING_N
        tris = []
        for face in mesh.faces:
            indices = np.asarray(face)
            belongs = np.all(indices >= limit) if sign > 0 else np.all(
                (indices < (off + 1) * RING_N) | (indices == mesh.groups["apex_rear"][0]))
            if not belongs:
                continue
            pts = V[indices]
            # A centre fan, unlike an arbitrary quad diagonal, preserves L/R
            # symmetry of the rounded skin and its interpolated normals.
            if len(face) == 3:
                tris.append(pts)
            else:
                centre = pts.mean(axis=0)
                tris.extend([centre, a, b] for a, b in zip(pts, np.roll(pts, -1, axis=0)))
        fascia[position] = np.asarray(tris)

    def fascia_frame(position, y, z, clearance=0.006):
        tri = fascia[position]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, ac = b[:, 1:] - a[:, 1:], c[:, 1:] - a[:, 1:]
        q = np.array([y, z]) - a[:, 1:]
        det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        safe = np.where(np.abs(det) > 1e-12, det, 1.0)
        u = (q[:, 0] * ac[:, 1] - q[:, 1] * ac[:, 0]) / safe
        v = (ab[:, 0] * q[:, 1] - ab[:, 1] * q[:, 0]) / safe
        valid = (np.abs(det) > 1e-12) & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
        sign = 1 if position == "front" else -1
        if valid.any():
            pts = a + u[:, None] * (b - a) + v[:, None] * (c - a)
            k = int(np.argmax(np.where(valid, sign * pts[:, 0], -np.inf)))
            p = pts[k]
            n = np.cross(b[k] - a[k], c[k] - a[k])
            n /= np.linalg.norm(n)
            if n[0] * sign < 0:
                n = -n
        else:
            # A deeply sculpted / narrow fascia may no longer cover a nominal
            # detail position: use the nearest actual triangle centroid.
            cent = tri.mean(axis=1)
            k = int(np.argmin(np.linalg.norm(cent[:, 1:] - [y, z], axis=1)))
            p = cent[k]
            n = np.cross(b[k] - a[k], c[k] - a[k])
            n /= max(np.linalg.norm(n), 1e-12)
            if n[0] * sign < 0:
                n = -n
        return Frame.from_normal(p + n * clearance, n, sign * Y)

    def furniture(name, tag, position, y, z, width, height, depth, **meta):
        frame = fascia_frame(position, y, z)
        out.append(PointConnector(name, frame, tags=[tag], meta={
            "position": position, "width": float(width), "height": float(height),
            "depth": float(depth), **meta}))

    for side, sign, side_name in sides:
        furniture(f"fog_front_{side}", "fog", "front", sign * meas["nose_half_width"] * 0.80,
                  meas["nose_z_bottom"] + 0.15, 0.14, 0.09, 0.035, side=side_name)
        furniture(f"rear_reflector_{side}", "rear_reflector", "rear", sign * meas["tail_half_width"] * 0.82,
                  meas["tail_z_bottom"] + 0.12, 0.14, 0.035, 0.008, side=side_name)
        kind = "fog" if side == "L" else "reverse"
        furniture(f"rear_{kind}_{side}", "rear_aux", "rear", sign * meas["tail_half_width"] * 0.76,
                  meas["tail_z_bottom"] + 0.235, 0.13, 0.055, 0.018, side=side_name, kind=kind)

        plate_z = 0.5 * (meas["tail_z_bottom"] + meas["tail_z_top"])
        plate = on_plane(apex_r, tail_n, plate_z, 0.012)
        plate_up = Frame.from_normal(plate, tail_n, -Y).y_axis
        p = plate + plate_up * 0.085 + sign * Y * 0.16
        out.append(PointConnector(f"plate_lamp_{side}", Frame.from_normal(p, tail_n, -Y),
                                 tags=["plate_lamp"], meta={"side": side_name, "position": "rear",
                                 "width": 0.055, "height": 0.022, "depth": 0.018}))

    for position, half_width, bottom in (("front", meas["nose_half_width"], meas["nose_z_bottom"]),
                                         ("rear", meas["tail_half_width"], meas["tail_z_bottom"])):
        furniture(f"tow_hook_cover_{position}", "tow_hook_cover", position, half_width * 0.46,
                  bottom + 0.18, 0.065, 0.055, 0.005, side="left")
    furniture("tow_eye", "tow_eye", "rear", -meas["tail_half_width"] * 0.44,
              meas["tail_z_bottom"] + 0.055, 0.075, 0.065, 0.075, side="right")

    for side, sign, side_name in sides:
        mir = (lambda j: j) if sign > 0 else mirror_index
        arch = V[mesh.groups[f"arch_front_{side}"]]
        ahead = float(arch[:, 0].max())
        room = float(nose[RING["D"], 0] - ahead)
        if room > 0.10:
            x = ahead + min(0.14, 0.45 * room)
            line = surface_line_at_x(mesh, x, mir(RING["D"]), mir(RING["E"]))
            p = 0.5 * (line[1] + line[2])
            next_line = surface_line_at_x(mesh, x + 0.005, mir(RING["D"]), mir(RING["E"]))
            along = 0.5 * (next_line[1] + next_line[2]) - p
            n = np.cross(line[2] - line[1], along)
            n /= max(np.linalg.norm(n), 1e-12)
            if sign * n[1] < 0:
                n = -n
            out.append(PointConnector(f"side_marker_{side}", Frame.from_normal(p + 0.004 * n, n, X),
                                     tags=["side_marker"], meta={"side": side_name, "position": "front",
                                     "width": 0.09, "height": 0.027, "depth": 0.008}))

        for axle in (("front", "rear") if hints.get("mud_flaps", False) else ()):
            lip = V[mesh.groups[f"arch_{axle}_{side}"]]
            trailing = lip[int(np.argmin(lip[:, 0]))].copy()
            tire_width = max(0.08, float(hints.get("tire_width", 0.225)))
            height = float(np.clip(trailing[2] - 0.10, 0.10, 0.25))
            top = trailing + np.array([-0.025, -sign * (tire_width / 2 + 0.01), -0.01])
            frame = Frame.from_normal(top - Z * height / 2, -X, -Y)
            out.append(RectangleConnector(f"mud_flap_{axle}_{side}", frame, tire_width + 0.025, height,
                                         tags=["mud_flap"], meta={"side": side_name, "axle": axle,
                                         "optional": True, "mount_top": top.tolist(), "depth": 0.006}))

    # Bilinear sampling of the untransposed aperture grid, including its local
    # derivatives: using only an average normal makes blades cut into the crown.
    _, _, _, glass = aperture_loop(mesh, "aperture/windshield")
    rows, cols = glass.shape[:2]

    def windshield(u, v):
        r, c = float(u) * (rows - 1), float(v) * (cols - 1)
        i, j = min(int(r), rows - 2), min(int(c), cols - 2)
        a, b = r - i, c - j
        g00, g01, g10, g11 = glass[i, j], glass[i, j + 1], glass[i + 1, j], glass[i + 1, j + 1]
        p = (1-a) * ((1-b)*g00 + b*g01) + a * ((1-b)*g10 + b*g11)
        du, dv = (1-b)*(g10-g00) + b*(g11-g01), (1-a)*(g01-g00) + a*(g11-g10)
        n = np.cross(du, dv)
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        return p, n

    def glass_v(u, y):
        r = u * (rows - 1)
        i = min(int(r), rows - 2)
        row = glass[i] * (1 - (r-i)) + glass[i + 1] * (r-i)
        return float(np.interp(y, row[::-1, 1], np.linspace(1, 0, cols)))

    blade_length = float(np.clip(np.ptp(glass[-1, :, 1]) * 0.34, 0.44, 0.56))
    paths, normals, uv_paths, pivots = [], [], [], []
    clearance = 0.006
    for _, sign, _ in sides:
        path, path_normals, uv = [], [], []
        for t in np.linspace(0, 1, 13):
            u = 0.88 + 0.04 * t
            v = glass_v(u, sign * (0.065 + t * blade_length))
            p, n = windshield(u, v)
            path.append((p + clearance * n).tolist())
            path_normals.append(n.tolist())
            uv.append([float(u), v])
        p, n = windshield(1.0, glass_v(1.0, sign * 0.055))
        pivots.append((p + clearance * n).tolist())
        paths.append(path)
        normals.append(path_normals)
        uv_paths.append(uv)
    cowl, _ = windshield(0.96, 0.5)
    normal = windshield(0.96, 0.49)[1] + windshield(0.96, 0.51)[1]
    normal /= np.linalg.norm(normal)
    out.append(RectangleConnector("wipers", Frame.from_normal(cowl + clearance * normal, normal, Y),
                                 float(np.ptp(glass[-1, :, 1]) * 0.92), 0.16, tags=["wipers"],
                                 meta={"paths": paths, "blade_normals": normals, "path_uv": uv_paths,
                                       "pivots": pivots, "blade_clearance": clearance, "paths_space": "world"}))

    floor = V[mesh.groups["line/floor"]]

    def underfloor(x, clearance=0.033):
        return np.array([x, 0.0, np.interp(x, floor[:, 0], floor[:, 2]) - clearance])

    midpoint = 0.5 * (meas["wheel_front_x"] + meas["wheel_rear_x"])
    junction = underfloor(meas["wheel_rear_x"] + 0.24)
    # Dense tunnel samples keep the system's mesh centroid near its underfloor
    # origin; do not anchor this long component at either rear exhaust tip.
    main_path = [underfloor(x).tolist() for x in np.linspace(meas["wheel_front_x"] + 0.16, junction[0], 41)]
    exhaust_paths, tips = [main_path], []
    for _, sign, _ in sides:
        tip = on_plane(apex_r, tail_n, meas["tail_z_bottom"] + 0.10)
        tip[1] = sign * (meas["tail_half_width"] - 0.32)
        approach = tip - tail_n * 0.14
        bend = underfloor(meas["wheel_rear_x"] - 0.15)
        bend[1] = tip[1] * 0.65
        exhaust_paths.append([junction.tolist(), ((junction+bend)/2).tolist(), bend.tolist(),
                              ((bend+approach)/2).tolist(), approach.tolist(), tip.tolist()])
        tips.append(tip.tolist())
    out.append(PointConnector("exhaust_system", Frame.from_normal(underfloor(midpoint), -Z, X),
                              tags=["exhaust_system"], meta={"paths": exhaust_paths, "tips": tips,
                              "pipe_radius": 0.025, "paths_space": "world"}))

    style = hints.get("style", "sedan")
    sporting = style in ("sports", "coupe") if isinstance(style, str) else any(
        float(style.get(name, 0)) > 0 for name in ("sports", "coupe")) if isinstance(style, dict) else False
    if sporting:
        rear = underfloor(float(tail[RING["A"], 0]) + 0.045, 0.012)
        front = underfloor(rear[0] + min(0.48, meas["wheelbase"] * 0.18), 0.012)
        normal = np.cross(Y, front - rear)
        out.append(RectangleConnector("diffuser", Frame.from_normal((front+rear)/2, normal, Y),
                                     min(1.35, 2 * meas["tail_half_width"] - 0.25),
                                     float(np.linalg.norm(front-rear)), tags=["diffuser"],
                                     meta={"depth": 0.045, "style_hint": style}))
    return out


def extra_interior_connectors(mesh: Mesh, meas: Dict[str, float], hints: Dict) -> List[Connector]:
    return []
