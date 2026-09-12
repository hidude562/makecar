"""Derive measurements, feature lines (seams) and connectors from a morphed body.

Everything here reads *vertex groups* and the aperture grid recorded by the
generator, never the generator parameters, so connectors follow the mesh
wherever the targets move it (the MakeHuman "joints follow the mesh" idea).
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from ..geometry.mesh import Mesh
from ..geometry.frame import Frame
from ..connectors import Connector, PointConnector, PolygonConnector, RectangleConnector, CircleConnector
from .generator import RING, RING_N, HALF_N, N_STATIONS, mirror_index, station_index

J = RING  # shorthand: J["F"] etc.


# ----------------------------------------------------------------- helpers
def vidx(mesh: Mesh, station: int, j: int) -> int:
    return (station + mesh.meta["ring_offset"]) * RING_N + j


def ring_pts(mesh: Mesh, station: int) -> np.ndarray:
    a = vidx(mesh, station, 0)
    return mesh.vertices[a : a + RING_N]


def fit_circle(pts_xz: np.ndarray) -> Tuple[float, float, float]:
    """Algebraic least-squares circle fit in 2-D -> (cx, cz, r)."""
    x, y = pts_xz[:, 0], pts_xz[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    return float(cx), float(cy), float(r)


def surface_line_at_x(mesh: Mesh, x_target: float, j_from: int, j_to: int, stations=None) -> np.ndarray:
    """Polyline on the loft surface at constant x, for ring indices j_from..j_to.
    Each ring index j is a longitudinal vertex chain; we interpolate along it."""
    stations = range(N_STATIONS) if stations is None else stations
    out = []
    step = 1 if j_to >= j_from else -1
    for j in range(j_from, j_to + step, step):
        chain = np.array([mesh.vertices[vidx(mesh, i, j)] for i in stations])
        xs = chain[:, 0]
        k = np.clip(np.searchsorted(xs, x_target) - 1, 0, len(xs) - 2)
        t = np.clip((x_target - xs[k]) / max(xs[k + 1] - xs[k], 1e-9), 0, 1)
        out.append(chain[k] + (chain[k + 1] - chain[k]) * t)
    return np.asarray(out)


def aperture_loop(mesh: Mesh, name: str) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray, np.ndarray]:
    """Ordered vertex loop around a grid aperture + (rows, cols) + outward normal
    + the full vertex grid (rows, cols, 3) the aperture was cut from."""
    r0, r1, j0, j1 = mesh.meta["aperture_rects"][name]
    off = mesh.meta["ring_offset"]
    # rows are loft ring rows (already include the fascia offset); convert to station space
    r0s, r1s = r0 - off, r1 - off + 1
    j1e = j1 + 1
    loop = []
    loop += [vidx(mesh, r, j0) for r in range(r0s, r1s + 1)]
    loop += [vidx(mesh, r1s, j) for j in range(j0 + 1, j1e + 1)]
    loop += [vidx(mesh, r, j1e) for r in range(r1s - 1, r0s - 1, -1)]
    loop += [vidx(mesh, r0s, j) for j in range(j1e - 1, j0, -1)]
    pts = mesh.vertices[loop]
    grid = np.array([[mesh.vertices[vidx(mesh, r, j)] for j in range(j0, j1e + 1)] for r in range(r0s, r1s + 1)])
    faces = mesh.zones[name]
    if "_face_normals" not in mesh.meta:
        mesh.meta["_face_normals"] = mesh.face_normals()
    fn = mesh.meta["_face_normals"][faces].mean(axis=0)
    fn /= max(np.linalg.norm(fn), 1e-9)
    return pts, (r1s - r0s + 1, j1e - j0 + 1), fn, grid


def fascia_triangles(mesh: Mesh, end: str, tris=None) -> np.ndarray:
    """Triangulated end surface; callers fitting several mounts can reuse it."""
    ids = np.append(mesh.groups[f"fascia/{end}"], mesh.groups[f"apex_{end}"])
    mask = np.zeros(mesh.n_vertices, dtype=bool)
    mask[ids] = True
    if tris is None:
        tris, _ = mesh.triangulated()
    return mesh.vertices[tris[mask[tris].all(axis=1)]]


def fascia_point(mesh: Mesh, end: str, z: float, y: float = 0.0, triangles=None) -> np.ndarray:
    """Mount on the actual stepped fascia, including the rear plate pocket.

    Intersect an X ray with the fascia triangles instead of treating a bumper,
    upper panel and recess as one raked plane. All inputs come from the morphed
    mesh, so differential targets and style blends move the mounts with it.
    """
    if triangles is None:
        triangles = fascia_triangles(mesh, end)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    u, v = b[:, 1:] - a[:, 1:], c[:, 1:] - a[:, 1:]
    q = np.array([y, z]) - a[:, 1:]
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    valid = np.abs(det) > 1e-12
    den = np.where(valid, det, 1.0)
    s = (q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0]) / den
    t = (u[:, 0] * q[:, 1] - u[:, 1] * q[:, 0]) / den
    valid &= (s >= -1e-8) & (t >= -1e-8) & (s + t <= 1 + 1e-8)
    x = a[:, 0] + s * (b[:, 0] - a[:, 0]) + t * (c[:, 0] - a[:, 0])
    if valid.any():
        return np.array([x[valid].max() if end == "front" else x[valid].min(), y, z])
    # Out-of-range custom hints still get a finite nearest-surface mount.
    pts = triangles.reshape(-1, 3)
    nearest = pts[np.argmin(np.linalg.norm(pts[:, 1:] - [y, z], axis=1))]
    return np.array([nearest[0], y, z])


def fascia_mount(mesh: Mesh, end: str, z: float, width: float, height: float,
                 ahead=0.004, triangles=None) -> np.ndarray:
    """Clear a rigid rectangle's whole footprint, not just its centre ray.

    A linear triangle reaches its X extremum at a clipped polygon vertex:
    original vertices, edge/rectangle intersections, or rectangle corners.
    Checking those points avoids both buried plates and coarse sampling misses.
    """
    if triangles is None:
        triangles = fascia_triangles(mesh, end)
    lo, hi = np.array([-width / 2, z - height / 2]), np.array([width / 2, z + height / 2])
    candidates = [triangles.reshape(-1, 3)]
    edges = triangles.reshape(-1, 3)
    ends = np.roll(triangles, -1, axis=1).reshape(-1, 3)
    delta = ends - edges
    for axis in (1, 2):
        for bound in (lo[axis - 1], hi[axis - 1]):
            den = delta[:, axis]
            t = (bound - edges[:, axis]) / np.where(abs(den) > 1e-12, den, 1)
            valid = (abs(den) > 1e-12) & (t >= 0) & (t <= 1)
            candidates.append(edges[valid] + delta[valid] * t[valid, None])
    candidates.append(np.array([fascia_point(mesh, end, zz, yy, triangles)
                                for yy in (lo[0], hi[0]) for zz in (lo[1], hi[1])]))
    pts = np.vstack(candidates)
    inside = ((pts[:, 1:] >= lo - 1e-9) & (pts[:, 1:] <= hi + 1e-9)).all(axis=1)
    sign = 1 if end == "front" else -1
    x = sign * np.max(sign * pts[inside, 0])
    return np.array([x + sign * ahead, 0, z])


# ------------------------------------------------------------ measurements
def measure(mesh: Mesh) -> Dict[str, float]:
    V = mesh.vertices
    G = mesh.groups

    def ring(name):
        return V[G[f"ring/{name}"]]

    def lring(name):
        return V[G[f"lring/{name}"]]

    H, E, F, Gj, D, A = J["H"], J["E"], J["F"], J["G"], J["D"], J["A"]
    cowl, rf, rr, deck = ring("cowl"), ring("roof_front"), ring("roof_rear"), ring("deck")
    bp = ring("bp_r")
    m: Dict[str, float] = {}
    m["x_front"] = float(V[G["fascia/front"]][:, 0].max())
    m["x_rear"] = float(V[G["fascia/rear"]][:, 0].min())
    m["length"] = m["x_front"] - m["x_rear"]
    m["width"] = float(2 * np.abs(V[:, 1]).max())
    m["height"] = float(V[:, 2].max())
    m["x_cowl"], m["z_cowl"] = float(cowl[H, 0]), float(cowl[H, 2])
    m["x_roof_front"], m["z_roof_front"] = float(rf[H, 0]), float(rf[H, 2])
    m["x_roof_rear"], m["z_roof_rear"] = float(rr[H, 0]), float(rr[H, 2])
    m["x_deck"], m["z_deck"] = float(deck[H, 0]), float(deck[H, 2])
    m["x_bpillar"] = float(0.5 * (ring("bp_r")[Gj, 0] + ring("bp_f")[Gj, 0]))
    m["roof_half_width"] = float(0.5 * (rf[Gj, 1] + rr[Gj, 1]))
    # cabin section at the B pillar
    m["z_belt"] = float(bp[E, 2])
    m["y_belt"] = float(bp[E, 1])
    m["y_shoulder"] = float(bp[F, 1])
    m["z_sill"] = float(np.mean([lring("ra_end")[D, 2], lring("fa_start")[D, 2]]))
    m["y_sill"] = float(lring("fa_start")[D, 1])
    cabin_st = range(station_index("bp_r") - 4, station_index("bp_f") + 4)
    # A is the raised tunnel crown; the flat pan is sampled at B.
    m["z_floor"] = float(np.mean([V[vidx(mesh, i, J["B"]), 2] for i in cabin_st]))
    m["y_floor_edge"] = float(np.mean([V[vidx(mesh, i, J["B"]), 1] for i in cabin_st]))
    # wheels
    for tag in ("front", "rear"):
        pts = V[G[f"arch_{tag}_L"]][:, [0, 2]]
        cx, cz, r = fit_circle(pts)
        m[f"wheel_{tag}_x"], m[f"wheel_{tag}_z"], m[f"arch_{tag}_r"] = cx, cz, r
        m[f"wheel_{tag}_y"] = float(np.mean(V[G[f"arch_{tag}_L"]][:, 1]))
    m["wheelbase"] = m["wheel_front_x"] - m["wheel_rear_x"]
    m["track"] = 2 * m["wheel_front_y"]
    # nose / tail faces
    nose, tail = V[G["nose_ring"]], V[G["tail_ring"]]
    m["nose_z_top"], m["nose_z_bottom"] = float(nose[H, 2]), float(nose[A, 2])
    m["nose_half_width"] = float(nose[E, 1])
    m["nose_top_half_width"] = float(nose[Gj, 1])
    m["tail_z_top"], m["tail_z_bottom"] = float(tail[H, 2]), float(tail[A, 2])
    m["tail_half_width"] = float(tail[E, 1])
    m["tail_top_half_width"] = float(tail[Gj, 1])
    m["bed"] = float(max(0.0, V[G["ring/deck"]][H, 2] - V[vidx(mesh, station_index("deck") // 2, H), 2]))
    m["has_bed"] = float(m["bed"] > 0.2)
    return m


# ------------------------------------------------------------ feature lines
def feature_lines(mesh: Mesh, meas: Dict[str, float], door_count: int = 4) -> List[Tuple[str, np.ndarray]]:
    V = mesh.vertices
    lines: List[Tuple[str, np.ndarray]] = []
    D, E, F, Gj, H = J["D"], J["E"], J["F"], J["G"], J["H"]
    st = station_index
    off = mesh.meta["ring_offset"]

    def chain(j, i0, i1):
        return np.array([V[vidx(mesh, i, j)] for i in range(i0, i1 + 1)])

    def ring_seg(i, j0, j1):
        r = ring_pts(mesh, i)
        idx = list(range(j0, j1 + 1)) if j1 >= j0 else list(range(j0, j1 - 1, -1))
        return r[idx]

    for side, mir in (("L", lambda j: j), ("R", mirror_index)):
        # hood: side seams along the fender/hood edge, rear seam across the cowl
        lines.append((f"hood_seam_{side}", chain(mir(F), st("cowl") + 1, st("front") - 3)))
        # trunk lid side seams
        lines.append((f"deck_seam_{side}", chain(mir(F), 3, st("deck"))))
        # B pillar seams sill -> roof rail
        for nm in ("bp_r", "bp_f"):
            lines.append((f"{nm}_seam_{side}", ring_seg(st(nm), mir(D), mir(Gj))))
        # front door leading edge: vertical on the door, then along the A pillar aperture edge
        x_fd = meas["wheel_front_x"] - meas["arch_front_r"] - 0.06
        vert = surface_line_at_x(mesh, x_fd, mir(D), mir(E))
        pil = ring_seg(st("roof_front"), mir(F), mir(Gj))
        lines.append((f"front_door_seam_{side}", np.vstack([vert, pil])))
        if door_count >= 4:
            x_rd = meas["wheel_rear_x"] + meas["arch_rear_r"] + 0.02
            vert = surface_line_at_x(mesh, x_rd, mir(D), mir(E))
            pil = ring_seg(st("cp_f"), mir(F), mir(Gj))
            lines.append((f"rear_door_seam_{side}", np.vstack([vert, pil])))
    # transverse seams (over the top)
    lines.append(("hood_rear_seam", ring_seg(st("cowl") + 1, F, mirror_index(F))))
    lines.append(("deck_front_seam", ring_seg(st("deck"), F, mirror_index(F))))
    # nose & tail hood/trunk front edges
    lines.append(("hood_front_seam", ring_seg(st("front") - 3, F, mirror_index(F))))
    return lines


def _sync_optional_zones(mesh: Mesh):
    """Style targets move vertices but retain base metadata; follow the mesh.

    Quarter glass and pickup liners must activate after morphing, before the
    shell apertures are removed. Copy nested metadata, never mutate the cached
    base library's rectangle table.
    """
    V, G = mesh.vertices, mesh.groups
    width = V[G["ring/cp_r"]][J["G"], 0] - V[G["ring/roof_rear"]][J["G"], 0]
    rects = dict(mesh.meta["aperture_rects"])
    for name, (r0, r1, j0, j1) in mesh.meta.get("optional_aperture_rects", {}).items():
        faces = [r * RING_N + j for r in range(r0, r1 + 1) for j in range(j0, j1 + 1)]
        if width > 0.15:
            rects[name] = (r0, r1, j0, j1)
            mesh.add_zone(name, faces)
            mesh.set_material("aperture", faces)
        else:
            rects.pop(name, None)
            mesh.zones.pop(name, None)
            mesh.set_material("paint", faces)
    mesh.meta["aperture_rects"] = rects
    previous = mesh.zones.pop("bed", [])
    mesh.set_material("paint", previous)
    bed = []
    for i in range(1, station_index("deck")):
        ring = ring_pts(mesh, i)
        if ring[J["G2"], 2] - ring[J["H"], 2] > 0.05:
            bed.extend((i + mesh.meta["ring_offset"]) * RING_N + j
                       for j in range(J["G2"], RING_N - J["G2"]))
    if bed:
        mesh.add_zone("bed", bed)
        mesh.set_material("bed", bed)


# --------------------------------------------------------------- connectors
def emit_connectors(mesh: Mesh, meas: Dict[str, float], hints: Dict) -> List[Connector]:
    """All connectors the body emits.  `hints` carries non-geometric options:
    door_count, seat_rows ('auto'|int), drive ('left'|'right'), fuel_side."""
    _sync_optional_zones(mesh)
    V = mesh.vertices
    out: List[Connector] = []
    door_count = int(hints.get("door_count", 4))
    drive = hints.get("drive", "left")
    ysign_driver = 1.0 if drive == "left" else -1.0
    X, Y, Z = np.eye(3)

    # ---------------------------------------------------------------- wheels
    tire_w = float(hints.get("tire_width", 0.225))
    for tag in ("front", "rear"):
        cx, cz, r = meas[f"wheel_{tag}_x"], meas[f"wheel_{tag}_z"], meas[f"arch_{tag}_r"]
        tire_r = r - float(hints.get("arch_gap", 0.06))
        y = meas[f"wheel_{tag}_y"] - tire_w / 2 - 0.01
        for side, ys in (("L", 1.0), ("R", -1.0)):
            fr = Frame.from_normal([cx, ys * y, cz], [0, ys, 0], x_hint=X)
            out.append(CircleConnector(f"wheel_{tag}_{side}", fr, tire_r, tags=["wheel", f"wheel_{tag}"],
                                       meta={"side": "left" if ys > 0 else "right", "axle": tag,
                                             "tire_width": tire_w, "arch_radius": r, "steer": tag == "front"}))

    # ------------------------------------------------------------- apertures
    for name in mesh.meta["aperture_rects"]:
        pts, grid, nrm, grid_pts = aperture_loop(mesh, name)
        short = name.split("/", 1)[1]
        kind = "glass" if short.startswith(("glass", "windshield", "rear_window")) else "light"
        tags = [kind, short.rstrip("_LR").rstrip("_")]
        if short.startswith("headlight"):
            tags.append("headlight")
        if short.startswith("taillight"):
            tags.append("taillight")
        side = "left" if short.endswith("_L") else "right" if short.endswith("_R") else "center"
        x_hint = Y if kind == "glass" else X
        conn = PolygonConnector.from_points(name.replace("aperture/", ""), pts, normal_hint=nrm, x_hint=x_hint,
                                            tags=tags, meta={"grid": list(grid), "grid_points": grid_pts, "side": side, "aperture": True})
        out.append(conn)

    # --------------------------------------------------------- exterior bits
    nose_n, tail_n = X, -X
    tris, _ = mesh.triangulated()
    fascia = {end: fascia_triangles(mesh, end, tris) for end in ("front", "rear")}
    z_nose_top, z_nose_bot = meas["nose_z_top"], meas["nose_z_bottom"]
    z_tail_top, z_tail_bot = meas["tail_z_top"], meas["tail_z_bottom"]
    nose_w = 2 * meas["nose_top_half_width"]
    tail_w = 2 * meas["tail_top_half_width"]

    def rectangle(name, end, z, width, height, tags, meta, ahead=0.004):
        normal = nose_n if end == "front" else tail_n
        point = fascia_mount(mesh, end, z, width, height, ahead, fascia[end])
        out.append(RectangleConnector(name, Frame.from_normal(point, normal, x_hint=-Y if end == "front" else Y),
                                      width, height, tags=tags, meta=meta))

    # Fit the upper grille between the bumper crease and the hood lip. The
    # front plate lives on the upright bumper, not across the setback.
    crease_z = V[mesh.groups["fascia/front_face"]][J["E"], 2]
    grille_lo, grille_hi = crease_z + 0.030, z_nose_top - 0.045
    grille_z = (grille_lo + grille_hi) / 2
    grille_h = max(0.045, min(0.17, grille_hi - grille_lo))
    rectangle("grille", "front", grille_z, min(nose_w * 0.62, 1.1), grille_h,
              ["grille"], {"style_hint": hints.get("style", "sedan")})
    intake_z = z_nose_bot + 0.105
    intake_h = max(0.05, min(0.14, 2 * (crease_z - 0.145 - intake_z)))
    rectangle("intake", "front", intake_z, min(nose_w * 0.8, 1.2), intake_h,
              ["grille", "intake"], {"lower": True})
    rectangle("plate_front", "front", crease_z - 0.075, 0.52, 0.11,
              ["plate"], {"position": "front"}, 0.012)
    rectangle("plate_rear", "rear", (z_tail_bot + z_tail_top) / 2, 0.52, 0.11,
              ["plate"], {"position": "rear"}, 0.012)
    for end, z, normal in (("front", z_nose_top - 0.035, nose_n), ("rear", z_tail_top - 0.16, tail_n)):
        point = fascia_mount(mesh, end, z, 0.06, 0.05, 0.012, fascia[end])
        out.append(PointConnector(f"badge_{end}", Frame.from_normal(point, normal, x_hint=-Y if end == "front" else Y), tags=["badge"]))
    # Exhaust mounts sample their actual lateral position, outside the plate pocket.
    for side, ys in (("L", 1.0), ("R", -1.0)):
        p = fascia_point(mesh, "rear", z_tail_bot + 0.10,
                         ys * (meas["tail_half_width"] - 0.32), fascia["rear"])
        out.append(CircleConnector(f"exhaust_{side}", Frame.from_normal(p, tail_n, x_hint=Y), 0.038, tags=["exhaust"],
                                   meta={"side": "left" if ys > 0 else "right"}))
    # side mirrors at the front-bottom corner of the front door glass
    st = station_index
    for side, mir, ys in (("L", lambda j: j, 1.0), ("R", mirror_index, -1.0)):
        p = V[vidx(mesh, st("roof_front"), mir(J["F"]))].copy()
        p[0] += 0.02
        p[2] += 0.02
        out.append(PointConnector(f"mirror_{side}", Frame.from_normal(p, [0.0, ys, 0.0], x_hint=X), tags=["mirror"],
                                  meta={"side": "left" if ys > 0 else "right"}))
        # door handles: on the door skin just below the belt, mid-door
        x_fd0 = meas["wheel_front_x"] - meas["arch_front_r"] - 0.06
        x_fd1 = meas["x_bpillar"]
        for door, x0, x1, ok in (("front", x_fd0, x_fd1, True),
                                 ("rear", x_fd1, meas["wheel_rear_x"] + meas["arch_rear_r"] + 0.02, door_count >= 4)):
            if not ok:
                continue
            xh = x1 + 0.62 * (x0 - x1)
            zh = meas["z_belt"] - 0.11
            line = surface_line_at_x(mesh, xh, mir(J["D"]), mir(J["E"]))
            k = int(np.argmin(np.abs(line[:, 2] - zh)))
            p = line[k].copy()
            nrm = np.array([0.0, ys, 0.0])
            out.append(PointConnector(f"handle_{door}_{side}", Frame.from_normal(p, nrm, x_hint=X), tags=["handle"],
                                      meta={"side": "left" if ys > 0 else "right", "door": door}))
        # roof rails along the roof edge (upper chain is numbered rear -> front)
        i0, i1 = st("roof_rear") + 1, st("roof_front") - 1
        if i1 > i0 + 1:
            pts = np.array([V[vidx(mesh, i, mir(J["G"]))] for i in range(i0, i1 + 1)])
            length = float(pts[-1, 0] - pts[0, 0])
            c = pts.mean(axis=0)
            fr = Frame.from_normal(c, Z, x_hint=X)
            out.append(RectangleConnector(f"roof_rail_{side}", fr, abs(length), 0.05, tags=["roof_rail"],
                                          meta={"side": "left" if ys > 0 else "right", "path": pts}))
    # fuel cap on the rear quarter (fuel side from hints)
    fs = hints.get("fuel_side", "left")
    ys = 1.0 if fs == "left" else -1.0
    mir = (lambda j: j) if ys > 0 else mirror_index
    xf = meas["wheel_rear_x"] + 0.08
    line = surface_line_at_x(mesh, xf, mir(J["D"]), mir(J["E"]))
    zf = meas["z_belt"] - 0.20
    k = int(np.argmin(np.abs(line[:, 2] - zf)))
    out.append(CircleConnector("fuel_cap", Frame.from_normal(line[k], [0, ys, 0], x_hint=X), 0.065, tags=["fuel_cap"],
                               meta={"side": fs}))
    # antenna / shark fin near the rear of the roof
    # Frame stations are only 25 mm from a glass edge; use physical roof
    # clearance so the shark-fin footprint stays off the rear window.
    x_ant = min(meas["x_roof_rear"] + 0.18, (meas["x_roof_rear"] + meas["x_roof_front"]) / 2)
    p = surface_line_at_x(mesh, x_ant, J["H"], J["H"], range(st("roof_rear"), st("roof_front") + 1))[0]
    out.append(PointConnector("antenna", Frame.from_normal(p, Z, x_hint=X), tags=["antenna"]))

    # -------------------------------------------------------------- interior
    out += interior_connectors(mesh, meas, hints)
    from .connectors_extra import extra_exterior_connectors
    out += extra_exterior_connectors(mesh, meas, hints)
    return out



def _face_normal_from_ring(ring: np.ndarray, sign: float) -> np.ndarray:
    """Outward normal of an end fascia, derived from the rake of its end ring.

    Kept as a shared helper because `connectors_extra` mounts rear furniture
    (reflectors, plate lamps, tow eye) against the same plane the body uses.
    `sign` is +1 for the nose and -1 for the tail.
    """
    top = ring[J["H"]]
    bot = ring[J["A"]]
    d = top - bot  # face direction (mostly vertical, tilted by the rake)
    n = np.array([d[2], 0.0, -d[0]])
    norm = np.linalg.norm(n)
    if norm < 1e-9:  # a perfectly vertical fascia
        return np.array([float(np.sign(sign)), 0.0, 0.0])
    n /= norm
    if np.sign(n[0]) != np.sign(sign):
        n = -n
    return n


def interior_connectors(mesh: Mesh, meas: Dict[str, float], hints: Dict) -> List[Connector]:
    V = mesh.vertices
    out: List[Connector] = []
    X, Y, Z = np.eye(3)
    door_count = int(hints.get("door_count", 4))
    drive = hints.get("drive", "left")
    ysd = 1.0 if drive == "left" else -1.0
    st = station_index

    floor = meas["z_floor"] + 0.05             # cabin floor (carpet top)
    inner_half = meas["y_shoulder"] - 0.08     # cabin inner half width at the belt
    roof_h = meas["z_roof_front"]
    h_point = float(hints.get("h_point_height", np.clip(0.17 + (roof_h - 1.2) * 0.4, 0.17, 0.38)))
    x_h1 = meas["x_cowl"] - float(hints.get("front_h_point_offset", 0.95))
    z_h1 = floor + h_point
    y_seat = min(0.42, inner_half - 0.36)
    x_firewall = meas["x_cowl"] + 0.12
    has_bed = meas.get("has_bed", 0.0) > 0.5
    x_cab_rear = min(meas["x_roof_rear"] - 0.05, meas["x_deck"] + 0.55) if not has_bed else meas["x_deck"] - 0.05
    x_cab_rear = min(x_cab_rear, meas["x_roof_rear"] + 0.15)
    # seat rows -----------------------------------------------------------------
    rows_cfg = hints.get("seat_rows", "auto")
    pitch = 0.86
    row_x = [x_h1]
    while len(row_x) < 3 and row_x[-1] - pitch - 0.30 > x_cab_rear - 0.12:
        row_x.append(row_x[-1] - pitch)
    if rows_cfg != "auto":
        rows_cfg = int(rows_cfg)
        row_x = [x_h1 - k * pitch for k in range(max(1, rows_cfg))]
    n_rows = len(row_x)
    if has_bed and n_rows > 1:
        # The pickup's rear wall is a genuine cab/bed boundary. The default
        # reclined bench (including its raised headrests) reaches 551 mm behind
        # its floor mount, not the 300 mm cushion allowance in auto allocation.
        # Reserve 560 mm, the 20 mm bulkhead trim and a 20 mm free gap. Preserve
        # all requested/emitted rows and the front mounts; compress rear pitch
        # only when the last complete seat would otherwise enter the bed.
        last_mount_min = x_cab_rear + 0.56 + 0.02 + 0.02
        if row_x[-1] + 0.10 < last_mount_min:
            pitch = (x_h1 + 0.10 - last_mount_min) / (n_rows - 1)
            row_x = [x_h1 - k * pitch for k in range(n_rows)]
    seat_w, seat_d = 0.52, 0.55
    for r, xh in enumerate(row_x):
        if r == 0:
            for name, ys in (("driver", ysd), ("passenger", -ysd)):
                fr = Frame.from_normal([xh + 0.12, ys * y_seat, floor], Z, x_hint=X)
                out.append(RectangleConnector(f"seat_front_{name}", fr, seat_d, seat_w, tags=["seat", "seat_front"],
                                              meta={"row": 0, "role": name, "h_point_height": h_point,
                                                    "headroom": roof_h - 0.03 - floor, "side": "left" if ys > 0 else "right"}))
        else:
            bench_w = 2 * inner_half - 0.12
            room = row_x[r - 1] - xh
            fr = Frame.from_normal([xh + 0.10, 0.0, floor + (0.02 if r == 1 else 0.0)], Z, x_hint=X)
            out.append(RectangleConnector(f"seat_row{r + 1}", fr, seat_d - 0.05, bench_w, tags=["seat", "seat_bench", f"seat_row{r + 1}"],
                                          meta={"row": r, "role": "bench", "h_point_height": h_point - 0.02 * r,
                                                "headroom": meas["z_roof_rear"] - 0.03 - floor if r == n_rows - 1 else roof_h - floor,
                                                "legroom": room}))
    # steering wheel / column ---------------------------------------------------
    col_angle = np.radians(float(hints.get("column_angle", 24.0)))
    wheel_c = np.array([x_h1 + 0.52, ysd * y_seat, z_h1 + 0.32])
    n = np.array([-np.cos(col_angle), 0.0, np.sin(col_angle)])
    out.append(CircleConnector("steering_wheel", Frame.from_normal(wheel_c, n, x_hint=Y), 0.185, tags=["steering_wheel"],
                               meta={"column_angle_deg": float(np.degrees(col_angle)), "column_length": 0.35}))
    # dashboard: interior section just behind the windshield base ----------------
    cowl = ring_pts(mesh, st("cowl"))
    yE = cowl[J["E"], 1] - 0.09
    z_top = meas["z_cowl"] - 0.03
    z_lo = floor + 0.32
    x_dash = meas["x_cowl"] - 0.02
    dash_pts = np.array([
        [x_dash, -yE, z_lo], [x_dash, yE, z_lo], [x_dash, yE, z_top - 0.12], [x_dash, yE - 0.10, z_top],
        [x_dash, -yE + 0.10, z_top], [x_dash, -yE, z_top - 0.12],
    ])
    dash = PolygonConnector.from_points("dashboard", dash_pts, normal_hint=-X, x_hint=-Y, tags=["dashboard"],
                                        meta={"driver_y": ysd * y_seat, "depth": 0.55, "floor": floor,
                                              "windshield_x": meas["x_cowl"], "steering_center": wheel_c})
    out.append(dash)
    # pedals ------------------------------------------------------------------------
    out.append(RectangleConnector("pedals", Frame.from_normal([x_h1 + 0.86, ysd * y_seat, floor + 0.02], Z, x_hint=X),
                                  0.20, 0.34, tags=["pedals"], meta={"transmission": hints.get("transmission", "automatic")}))
    # centre console between the front seats --------------------------------------
    console_len = 0.95
    out.append(RectangleConnector("console", Frame.from_normal([x_h1 + 0.18, 0.0, floor], Z, x_hint=X), console_len, 0.26,
                                  tags=["console"], meta={"height": z_h1 - floor + 0.06, "driver_y": ysd * y_seat}))
    # cabin floor --------------------------------------------------------------------
    fl_len = x_firewall - x_cab_rear
    out.append(RectangleConnector("floor", Frame.from_normal([(x_firewall + x_cab_rear) / 2, 0.0, floor - 0.01], Z, x_hint=X),
                                  fl_len, 2 * inner_half + 0.10, tags=["floor"], meta={"tunnel_height": 0.16, "seat_rows": n_rows}))
    # firewall / bulkheads (simple rectangles so the cabin reads as enclosed)
    out.append(RectangleConnector("firewall", Frame.from_normal([x_firewall, 0.0, (floor + z_lo) / 2], -X, x_hint=-Y),
                                  2 * inner_half + 0.10, z_lo - floor + 0.02, tags=["bulkhead"], meta={"position": "front"}))
    out.append(RectangleConnector("rear_bulkhead", Frame.from_normal([x_cab_rear, 0.0, (floor + meas["z_belt"]) / 2], X, x_hint=Y),
                                  2 * inner_half + 0.10, meas["z_belt"] - floor, tags=["bulkhead"], meta={"position": "rear"}))
    # parcel shelf (sedan) or cargo floor (hatch/wagon)
    if meas["x_deck"] < x_cab_rear - 0.2 and not has_bed:
        length = x_cab_rear - meas["x_deck"]
        out.append(RectangleConnector("parcel_shelf", Frame.from_normal([x_cab_rear - length / 2, 0.0, meas["z_deck"] - 0.10], Z, x_hint=X),
                                      length, 2 * inner_half + 0.05, tags=["shelf"]))
        # trunk floor: from the rear bulkhead back to just ahead of the rear bumper
        # structure (never past the tail, whatever the deck position of the style)
        x_boot_rear = meas["x_rear"] + 0.28
        tl = max(0.35, x_cab_rear - x_boot_rear)
        out.append(RectangleConnector("trunk_floor", Frame.from_normal([x_cab_rear - tl / 2, 0.0, floor + 0.10], Z, x_hint=X),
                                      tl, 2 * inner_half - 0.1, tags=["cargo_floor"], meta={"enclosed": True}))
    elif not has_bed:
        length = x_cab_rear - (meas["x_rear"] + 0.25)
        if length > 0.3:
            out.append(RectangleConnector("cargo_floor", Frame.from_normal([x_cab_rear - length / 2, 0.0, floor + 0.12], Z, x_hint=X),
                                          length, 2 * inner_half - 0.05, tags=["cargo_floor"], meta={"enclosed": False}))
    # headliner: interior side of the roof (stations run rear -> front) --------------------
    i0, i1 = st("roof_rear"), st("roof_front")
    jG, jH = J["G"], J["H"]
    loop = []
    loop += [vidx(mesh, i, jG) for i in range(i0, i1 + 1)]
    loop += [vidx(mesh, i1, j) for j in range(jG + 1, mirror_index(jG) + 1)]
    loop += [vidx(mesh, i, mirror_index(jG)) for i in range(i1 - 1, i0 - 1, -1)]
    loop += [vidx(mesh, i0, j) for j in range(mirror_index(jG) - 1, jG, -1)]
    pts = V[loop].copy()
    pts[:, 2] -= 0.035
    hl_grid = np.array([[V[vidx(mesh, i, j)] for j in range(jG, mirror_index(jG) + 1)] for i in range(i0, i1 + 1)])
    hl_grid[:, :, 2] -= 0.035
    out.append(PolygonConnector.from_points("headliner", pts, normal_hint=-Z, x_hint=X, tags=["headliner"],
                                            meta={"grid": [i1 - i0 + 1, mirror_index(jG) - jG + 1], "grid_points": hl_grid,
                                                  "roof_center_x": (meas["x_roof_front"] + meas["x_roof_rear"]) / 2}))
    # rear-view mirror
    p = V[vidx(mesh, i0 + 1, jH)].copy()
    p[2] -= 0.12
    out.append(PointConnector("rearview_mirror", Frame.from_normal(p, -X, x_hint=Y), tags=["rearview_mirror"]))
    # door cards: interior side of each door ---------------------------------------------
    x_fd0 = meas["wheel_front_x"] - meas["arch_front_r"] - 0.06
    x_fd1 = meas["x_bpillar"]
    x_rd1 = meas["wheel_rear_x"] + meas["arch_rear_r"] + 0.02
    doors = [("front", x_fd0, x_fd1)]
    if door_count >= 4:
        doors.append(("rear", x_fd1, x_rd1))
    for side, mir, ys in (("L", lambda j: j, 1.0), ("R", mirror_index, -1.0)):
        for door, xa, xb in doors:
            ja, jb = mir(J["D"]), mir(J["E"])
            front = surface_line_at_x(mesh, xa, ja, jb)
            rear = surface_line_at_x(mesh, xb, ja, jb)
            n_len = 6
            poly = []
            # bottom edge (along the sill) from front to rear, rear edge up, top edge back, front edge down
            for t in np.linspace(0, 1, n_len, endpoint=False):
                poly.append(front[0] + (rear[0] - front[0]) * t)
            poly += list(rear)
            for t in np.linspace(0, 1, n_len, endpoint=False)[1:]:
                poly.append(rear[-1] + (front[-1] - rear[-1]) * t)
            poly += list(front[::-1][:-1])
            poly = np.asarray(poly)
            poly[:, 1] -= ys * 0.09
            poly[:, 2] += 0.02
            out.append(PolygonConnector.from_points(f"door_card_{door}_{side}", poly, normal_hint=[0, -ys, 0], x_hint=X,
                                                    tags=["door_card"], meta={"side": "left" if ys > 0 else "right", "door": door,
                                                                              "grid": [n_len + 1, len(front)], "belt_z": float(meas["z_belt"])}))
    # pickup bed floor connector
    if has_bed:
        deck_r = ring_pts(mesh, st("deck"))
        length = meas["x_deck"] - meas["x_rear"] - 0.1
        out.append(RectangleConnector("bed_floor", Frame.from_normal([meas["x_rear"] + 0.05 + length / 2, 0.0, meas["z_deck"] - meas["bed"] + 0.01], Z, x_hint=X),
                                      length, 2 * (meas["roof_half_width"] - 0.05), tags=["cargo_floor", "bed"], meta={"enclosed": False}))
    from .connectors_extra import extra_interior_connectors
    out += extra_interior_connectors(mesh, meas, hints)
    return out
