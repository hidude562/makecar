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
    from .connectors import aperture_loop, surface_line_at_x, fascia_point, fascia_triangles
    from .generator import RING, RING_N, N_STATIONS, mirror_index

    V = mesh.vertices
    X, Y, Z = np.eye(3)
    out = []
    sides = (("L", 1.0, "left"), ("R", -1.0, "right"))
    nose, tail = (V[mesh.groups[name]] for name in ("nose_ring", "tail_ring"))
    rear_triangles = fascia_triangles(mesh, "rear")

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
        # Above the diffuser's sloping return: reflectors must face traffic,
        # not the road. Leave a separate strip below the rear auxiliary lamps.
        furniture(f"rear_reflector_{side}", "rear_reflector", "rear", sign * meas["tail_half_width"] * 0.82,
                  meas["tail_z_bottom"] + 0.17, 0.14, 0.035, 0.008, side=side_name)
        kind = "fog" if side == "L" else "reverse"
        furniture(f"rear_{kind}_{side}", "rear_aux", "rear", sign * meas["tail_half_width"] * 0.76,
                  meas["tail_z_bottom"] + 0.235, 0.13, 0.055, 0.018, side=side_name, kind=kind)

        plate_z = 0.5 * (meas["tail_z_bottom"] + meas["tail_z_top"])
        # The fan centre is now a recessed pocket, not a raked fascia plane.
        # Sample the pocket above the plate at each fixture's actual position.
        p = fascia_point(mesh, "rear", plate_z + 0.085, sign * 0.16, rear_triangles) - X * 0.008
        out.append(PointConnector(f"plate_lamp_{side}", Frame.from_normal(p, -X, -Y),
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

    # A is now the raised tunnel crown, not the flat pan at B. Retain the
    # measured cross-sections so cans also fit the tunnel's width and ceiling.
    floor_grid = np.array([V[(off + i) * RING_N:(off + i) * RING_N + RING["B"] + 1]
                           for i in range(N_STATIONS)])

    def underfloor(x, clearance=0.033, y=0.0):
        row = np.array([[np.interp(x, chain[:, 0], chain[:, k]) for k in range(3)]
                        for chain in floor_grid.transpose(1, 0, 2)])
        return np.array([x, y, np.interp(abs(y), row[:, 1], row[:, 2]) - clearance])

    midpoint = 0.5 * (meas["wheel_front_x"] + meas["wheel_rear_x"])
    junction = underfloor(meas["wheel_rear_x"] + 0.24)
    # Dense tunnel samples keep the system's mesh centroid near its underfloor
    # origin; do not anchor this long component at either rear exhaust tip.
    main_path = [underfloor(x).tolist() for x in np.linspace(meas["wheel_front_x"] + 0.16, junction[0], 61)]
    exhaust_paths, tips = [main_path], []
    for _, sign, _ in sides:
        # Use the built-in mounts' actual lateral fascia ray and rearward axis,
        # not the obsolete centre-apex plane (the plate pocket is recessed).
        tip = fascia_point(mesh, "rear", meas["tail_z_bottom"] + 0.10,
                           sign * (meas["tail_half_width"] - 0.32), rear_triangles)
        approach = tip + X * 0.14
        bend = underfloor(meas["wheel_rear_x"] - 0.15, clearance=0.04, y=tip[1] * 0.65)
        # Drop under the flat pan *before* turning out of the raised tunnel.
        # A diagonal directly from the crown-height junction cuts its shoulder.
        drop = underfloor(junction[0] - 0.16, clearance=0.04, y=bend[1])
        drop[1] = tip[1] * 0.10
        # Follow every pan slope change until the short entry into the rear
        # bumper. A straight bend-to-tip diagonal passes through most of the
        # rear floor, making the visible pipe stop just behind the axle.
        rear_x = approach[0] + 0.08
        xs = floor_grid[:, 0, 0]
        samples = np.unique(np.r_[rear_x, xs[(xs > rear_x) & (xs < bend[0])], bend[0]])[::-1]
        rear_path = []
        for x in samples:
            t = (bend[0] - x) / (bend[0] - rear_x)
            # Extra vertical room clears the radius around the rising rear pan,
            # where a tilted sweep needs more than its nominal radius in Z.
            p = underfloor(x, clearance=0.04, y=bend[1] * (1-t) + tip[1] * t)
            # Drop redundant stations along straight stretches of pan, but
            # retain every actual change of slope in the measured profile.
            while len(rear_path) > 1 and np.linalg.norm(np.cross(
                    rear_path[-1] - rear_path[-2], p - rear_path[-1])) < 1e-10:
                rear_path.pop()
            rear_path.append(p)
        exhaust_paths.append([junction.tolist(), drop.tolist(), *[p.tolist() for p in rear_path],
                              approach.tolist(), tip.tolist()])
        tips.append(tip.tolist())
    out.append(PointConnector("exhaust_system", Frame.from_normal(underfloor(midpoint), -Z, X),
                              tags=["exhaust_system"], meta={"paths": exhaust_paths, "tips": tips,
                              "pipe_radius": 0.025, "paths_space": "world", "floor_grid": floor_grid.tolist()}))

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
    """Cabin-side trim paths and shoulder-belt mounts, all following the loft.

    Path metadata is world-space.  Pillar +Z faces into the cabin; belt mounts
    carry the matching seat name, leaving buckle placement to the fitted seat.
    Existing body connector names and frames are deliberately left untouched.
    The neutral interior_detail tag must not alias old component-selection tags
    (e.g. dashboard), whose assignments would otherwise target these points.
    """
    import numpy as np
    from ..connectors import PointConnector
    from ..geometry.frame import Frame
    from .connectors import aperture_loop, ring_pts, J, station_index, mirror_index

    out = []

    def skin_grid(first, last, j0, j1, sign):
        js = range(J[j0], J[j1] + 1)
        js = list(js) if sign > 0 else [mirror_index(j) for j in js]
        grid = np.array([ring_pts(mesh, i)[js] for i in range(station_index(first), station_index(last) + 1)])
        dx = -0.015 if first == "roof_front" else 0.015 if last in ("roof_rear", "cp_r", "cp_f") else 0.0
        return grid + np.array([dx, -sign * 0.012, -0.008])

    _, _, _, windshield = aperture_loop(mesh, "aperture/windshield")
    base = windshield[np.argmax(windshield[:, :, 0].mean(axis=1))].copy()
    base += np.array([-0.025, 0.0, -0.025])
    out.append(PointConnector("cowl_trim", Frame.from_normal(base.mean(axis=0), [0, 0, 1]),
                              tags=["cowl_trim", "interior_detail"], meta={"path": base}))
    floor = meas["z_floor"] + 0.05
    hp = float(hints.get("h_point_height", np.clip(0.17 + (meas["z_roof_front"] - 1.2) * 0.4, 0.17, 0.38)))
    for side, sign in (("L", 1.0), ("R", -1.0)):
        col = int(np.argmax(windshield[:, :, 1].mean(axis=0) * sign))
        path = windshield[:, col].copy()
        path += np.array([-0.02, -sign * 0.025, -0.025])
        normal = np.array([0.0, -sign, 0.0])
        out.append(PointConnector(f"pillar_trim_A_{side}", Frame.from_normal(path.mean(axis=0), normal),
                                  tags=["pillar_trim", "interior_detail"], meta={"pillar": "A", "side": side, "path": path, "width": 0.065,
                                                                        "grid_points": skin_grid("roof_front", "cowl", "E", "G", sign)}))
        js = list(range(J["D"], J["G"] + 1))
        if sign < 0:
            js = [mirror_index(j) for j in js]
        path = (ring_pts(mesh, station_index("bp_r"))[js] + ring_pts(mesh, station_index("bp_f"))[js]) / 2
        path += np.array([0.0, -sign * 0.08, -0.035])
        foot = path[0].copy()
        foot[2] = floor
        path = np.vstack([foot, path])
        out.append(PointConnector(f"pillar_trim_B_{side}", Frame.from_normal(path.mean(axis=0), normal),
                                  tags=["pillar_trim", "interior_detail"], meta={"pillar": "B", "side": side, "path": path, "width": 0.10,
                                                                        "grid_points": skin_grid("bp_r", "bp_f", "E", "G", sign)}))
        lower = skin_grid("deck", "front", "B", "E", sign)
        end = int(np.searchsorted(lower[:, :, 0].mean(axis=1), meas["x_cowl"] + 0.20))
        lower = lower[:max(2, end + 1)]
        lower[:, :, 1] -= sign * 0.06
        lower[:, :, 0] -= 0.015
        lower[:, :, 2] += 0.02
        out.append(PointConnector(f"cabin_side_liner_{side}", Frame.from_normal(lower.reshape(-1, 3).mean(axis=0), normal),
                                  tags=["pillar_trim", "interior_detail"], meta={"pillar": "lower", "side": side, "grid_points": lower}))
        for pillar, first, last in (("C", "cp_r", "cp_f"), ("rear", "deck", "roof_rear"), ("belt", "roof_rear", "roof_front")):
            grid = skin_grid(first, last, "E", "F" if pillar == "belt" else "G", sign)
            out.append(PointConnector(f"pillar_trim_{pillar}_{side}", Frame.from_normal(grid.reshape(-1, 3).mean(axis=0), normal),
                                      tags=["pillar_trim", "interior_detail"], meta={"pillar": pillar, "side": side, "grid_points": grid}))
        if f"aperture/glass_quarter_{side}" not in mesh.meta["aperture_rects"]:
            grid = skin_grid("roof_rear", "cp_r", "F", "G", sign)
            out.append(PointConnector(f"pillar_trim_quarter_{side}", Frame.from_normal(grid.reshape(-1, 3).mean(axis=0), normal),
                                      tags=["pillar_trim", "interior_detail"], meta={"pillar": "quarter", "side": side, "grid_points": grid}))
        height = float(np.clip(floor + hp + 0.57, path[0, 2] + 0.2, path[-1, 2] - 0.06))
        anchor = np.array([np.interp(height, path[:, 2], path[:, k]) for k in range(3)])
        anchor[1] -= sign * 0.016
        role = "driver" if (sign > 0) == (hints.get("drive", "left") == "left") else "passenger"
        xh = meas["x_cowl"] - float(hints.get("front_h_point_offset", 0.95))
        seat_y = min(0.42, meas["y_shoulder"] - 0.44)
        out.append(PointConnector(f"belt_anchor_{side}", Frame.from_normal(anchor, normal, x_hint=(1, 0, 0)),
                                  tags=["belt_anchor", "interior_detail"], meta={"side": side, "seat": f"seat_front_{role}",
                                                            "belt_width": 0.047, "path": path,
                                                            "seat_origin": [xh + 0.12, sign * seat_y, floor],
                                                            "seat_size": [0.55, 0.52], "h_point_height": hp}))
    if meas.get("has_bed", 0.0) < 0.5:
        for side, sign in (("L", 1.0), ("R", -1.0)):
            origin = [meas["wheel_rear_x"], sign * (meas["y_shoulder"] - 0.12), floor + 0.03]
            radius = meas["arch_rear_r"] + 0.035
            height = max(0.18, meas["wheel_rear_z"] + radius - origin[2])
            out.append(PointConnector(f"cargo_wheelhouse_{side}", Frame.from_normal(origin, [0, 0, 1], x_hint=(1, 0, 0)),
                                      tags=["cargo_wheelhouse", "interior_detail"],
                                      meta={"side": side, "radius": radius, "height": height, "depth": 0.25}))
    return out
