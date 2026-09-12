"""Helpers shared by the interior component library.

* `loop_patch` / `connector_patch`: Coons patches for *any* polygon connector
  loop.  `fills.fill_connector` needs the loop length to match the `grid` meta;
  the body currently emits headliner / door-card loops that do not, so we find
  the four corners geometrically and split the loop into four boundary curves.
* `heightfield_slab`: a closed slab whose top surface is a height field
  (carpet with a transmission tunnel, ribbed bed liner, ...).
* small utilities: rotation matrices, raised-cosine blends, face painting and
  reconstruction of physical parameters from morph-slider values.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Sequence, Tuple
import numpy as np

from ..geometry.mesh import Mesh
from ..geometry.frame import Frame, rotation_matrix
from ..geometry.curves import resample_polyline
from ..geometry import primitives as P
from ..connectors import PolygonConnector
from .fills import coons_patch, grid_mesh, resample_grid


# ------------------------------------------------------------------ patches
def _corner_indices(lp: np.ndarray) -> Tuple[int, int, int, int]:
    """Indices (in loop order) of the points closest to the four bounding-box
    corners of a 2-D loop: (min,min) (max,min) (max,max) (min,max)."""
    lo, hi = lp.min(axis=0), lp.max(axis=0)
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    idx = [int(np.argmin(np.linalg.norm(lp - c, axis=1))) for c in corners]
    if len(set(idx)) < 4:  # degenerate (very thin) loop: fall back to quarter splits
        n = len(lp)
        idx = [0, n // 4, n // 2, (3 * n) // 4]
    return tuple(idx)  # type: ignore


def _arc(points: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """Points of a closed loop from index i0 to i1 inclusive (cyclic)."""
    n = len(points)
    if i1 >= i0:
        return points[i0 : i1 + 1]
    return np.vstack([points[i0:], points[: i1 + 1]])


def loop_patch(points_world: np.ndarray, frame: Frame, res_u: int, res_v: int) -> np.ndarray:
    """Coons patch (res_u, res_v, 3) spanning an arbitrary closed loop.  The
    loop is split at its four bounding-box corners (in the frame's XY plane);
    u runs along the first side (from the (min,min) corner towards (max,min)),
    v along the second."""
    pts = np.asarray(points_world, dtype=float)
    lp = frame.to_local(pts)[:, :2]
    c = _corner_indices(lp)
    order = np.argsort(c)                 # walk corners in loop order
    cs = [c[i] for i in order]
    # first side should start at the (min,min) corner
    k0 = list(order).index(0)
    cs = cs[k0:] + cs[:k0]
    a, b, cc, d = cs
    s0 = _arc(pts, a, b)      # u at v=0
    s1 = _arc(pts, b, cc)     # v at u=1
    s2 = _arc(pts, cc, d)     # u (reversed) at v=1
    s3 = _arc(pts, d, a)      # v (reversed) at u=0
    c0 = resample_polyline(s0, res_u)
    c1 = resample_polyline(s2[::-1], res_u)
    d1 = resample_polyline(s1, res_v)
    d0 = resample_polyline(s3[::-1], res_v)
    u = np.linspace(0, 1, res_u)[:, None, None]
    v = np.linspace(0, 1, res_v)[None, :, None]
    P00, P10, P01, P11 = c0[0], c0[-1], c1[0], c1[-1]
    S = (1 - v) * c0[:, None, :] + v * c1[:, None, :] + (1 - u) * d0[None, :, :] + u * d1[None, :, :]
    S -= (1 - u) * (1 - v) * P00 + u * (1 - v) * P10 + (1 - u) * v * P01 + u * v * P11
    return S


def connector_patch(conn: PolygonConnector, res_u: int, res_v: int, clip_x_max: Optional[float] = None) -> np.ndarray:
    """World-space surface grid (res_u, res_v, 3) for a polygon connector.
    Prefers the exact vertex grid the body recorded (`meta['grid_points']`),
    then a Coons patch of the loop when it matches `meta['grid']`, otherwise
    the geometric corner split.  `clip_x_max` clamps world x (used to stop a
    door card at the firewall)."""
    gp = conn.meta.get("grid_points")
    if gp is not None:
        gp = np.asarray(gp, dtype=float)
        if clip_x_max is not None:
            gp = gp.copy()
            gp[..., 0] = np.minimum(gp[..., 0], clip_x_max)
        return resample_grid(gp, res_u, res_v)
    pts = conn.points.copy()
    if clip_x_max is not None:
        pts[:, 0] = np.minimum(pts[:, 0], clip_x_max)
    grid = conn.meta.get("grid")
    if grid and len(pts) == 2 * (grid[0] - 1) + 2 * (grid[1] - 1) and min(grid) >= 2:
        return coons_patch(pts, grid[0], grid[1], res_u, res_v)
    return loop_patch(pts, conn.frame, res_u, res_v)


def patch_surface(grid: np.ndarray, normal: np.ndarray, material: str, offset: float = 0.0,
                  thickness: float = 0.0, name: str = "panel") -> Mesh:
    """Mesh a (u, v, 3) grid, oriented along `normal`, optionally with a back
    face `thickness` behind it (closed by side strips)."""
    nrm = np.asarray(normal, dtype=float)
    S = grid + offset * nrm
    m = grid_mesh(S, material, name)
    if np.mean(m.face_normals() @ nrm) < 0:
        m.flip_normals()
    if thickness > 0:
        back = grid_mesh(S - thickness * nrm, material, name + "_back")
        if np.mean(back.face_normals() @ nrm) > 0:
            back.flip_normals()
        ru, rv, _ = S.shape
        # side strips around the border
        border = ([(i, 0) for i in range(ru)] + [(ru - 1, j) for j in range(1, rv)]
                  + [(i, rv - 1) for i in range(ru - 2, -1, -1)] + [(0, j) for j in range(rv - 2, 0, -1)])
        top_idx = [i * rv + j for i, j in border]
        off = m.n_vertices
        m.merge(back)
        faces = []
        for k in range(len(top_idx)):
            a, b = top_idx[k], top_idx[(k + 1) % len(top_idx)]
            faces.append((a, b, b + off, a + off))
        m.faces.extend(faces)
        m.face_materials.extend([material] * len(faces))
        # orient side faces outward: compare with the direction away from the centroid
        fn = m.face_normals()
        cent = m.face_centroids()
        centre = S.reshape(-1, 3).mean(axis=0)
        start = m.n_faces - len(faces)
        for fi in range(start, m.n_faces):
            if np.dot(fn[fi], cent[fi] - centre) < 0:
                m.faces[fi] = tuple(reversed(m.faces[fi]))
    return m


def grid_local_z(grid: np.ndarray, frame: Frame, x: float, y: float) -> float:
    """Local z of the patch surface nearest to local (x, y) (for placing parts on it)."""
    loc = frame.to_local(grid.reshape(-1, 3))
    k = int(np.argmin((loc[:, 0] - x) ** 2 + (loc[:, 1] - y) ** 2))
    return float(loc[k, 2])


# --------------------------------------------------------------- height fields
def heightfield_slab(xs: np.ndarray, ys: np.ndarray, z_top: Callable[[np.ndarray, np.ndarray], np.ndarray],
                     z_bottom: float, material: str, name: str = "slab") -> Mesh:
    """Closed slab: top surface z = z_top(X, Y) over the grid xs x ys, flat
    bottom at z_bottom, side walls in between.  Normals point outward."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Zt = np.asarray(z_top(X, Y), dtype=float)
    nx, ny = X.shape
    top = np.stack([X, Y, Zt], axis=-1)
    bot = np.stack([X, Y, np.full_like(Zt, z_bottom)], axis=-1)
    m = grid_mesh(top, material, name)
    if np.mean(m.face_normals()[:, 2]) < 0:
        m.flip_normals()
    b = grid_mesh(bot, material, name + "_bottom")
    if np.mean(b.face_normals()[:, 2]) > 0:
        b.flip_normals()
    off = m.n_vertices
    m.merge(b)
    border = ([(i, 0) for i in range(nx)] + [(nx - 1, j) for j in range(1, ny)]
              + [(i, ny - 1) for i in range(nx - 2, -1, -1)] + [(0, j) for j in range(ny - 2, 0, -1)])
    idx = [i * ny + j for i, j in border]
    faces = []
    for k in range(len(idx)):
        a, bb = idx[k], idx[(k + 1) % len(idx)]
        faces.append((a, bb, bb + off, a + off))
    start = m.n_faces
    m.faces.extend(faces)
    m.face_materials.extend([material] * len(faces))
    fn = m.face_normals()
    cent = m.face_centroids()
    cx, cy = float(xs.mean()), float(ys.mean())
    for fi in range(start, m.n_faces):
        d = cent[fi] - np.array([cx, cy, cent[fi, 2]])
        if np.dot(fn[fi], d) < 0:
            m.faces[fi] = tuple(reversed(m.faces[fi]))
    return m


# ------------------------------------------------------------------- utilities
def rot_x(angle: float) -> np.ndarray:
    return rotation_matrix((1, 0, 0), angle)


def rot_y(angle: float) -> np.ndarray:
    return rotation_matrix((0, 1, 0), angle)


def rot_z(angle: float) -> np.ndarray:
    return rotation_matrix((0, 0, 1), angle)


def place(mesh: Mesh, at, rot: Optional[np.ndarray] = None) -> Mesh:
    """Rotate (4x4, about the origin) then translate a mesh in place."""
    if rot is not None:
        mesh.transform(rot)
    mesh.translate(np.asarray(at, dtype=float))
    return mesh


def bump(x, centre: float, half_width: float, falloff: float):
    """Raised-cosine plateau: 1 inside |x-centre|<half_width, smooth 0 beyond half_width+falloff."""
    d = np.abs(np.asarray(x, dtype=float) - centre) - half_width
    t = np.clip(d / max(falloff, 1e-9), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def smoothstep(e0: float, e1: float, x):
    t = np.clip((np.asarray(x, dtype=float) - e0) / max(e1 - e0, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def paint(mesh: Mesh, predicate: Callable[[np.ndarray], np.ndarray], material: str,
          only: Optional[Sequence[str]] = None) -> None:
    """Assign `material` to faces whose centroid satisfies predicate(centroids)->mask."""
    cent = mesh.face_centroids()
    mask = np.asarray(predicate(cent), dtype=bool)
    for i in np.where(mask)[0]:
        if only is None or mesh.face_materials[i] in only:
            mesh.face_materials[i] = material


def params_from_values(component, values: Dict[str, float]):
    """Reconstruct the physical generator parameters implied by slider values
    (inverse of `MorphableComponent.value_for`), so build-time extras can be
    placed on the morphed geometry."""
    p = component.canonical()
    over = {}
    for spec in component.modifier_specs:
        name = spec.name or spec.param
        v = float(values.get(name, 0.0))
        canon = getattr(p, spec.param)
        over[spec.param] = canon + (v * spec.delta_plus if v >= 0 else v * spec.delta_minus)
    return dataclasses.replace(p, **over)


def section_loft(sections: Sequence[np.ndarray], origins: Sequence[np.ndarray], u_axis: np.ndarray,
                 v_axis: np.ndarray, material: str, cap_start=True, cap_end=True, name="loft") -> Mesh:
    """Loft closed 2-D sections (K,2) placed at `origins[i]` in the plane
    spanned by (u_axis, v_axis).  Sections must all have K points."""
    u = np.asarray(u_axis, dtype=float)
    v = np.asarray(v_axis, dtype=float)
    rings = [np.asarray(o, dtype=float) + s[:, 0:1] * u + s[:, 1:2] * v for s, o in zip(sections, origins)]
    return P.loft(rings, closed_rings=True, cap_start=cap_start, cap_end=cap_end, material=material, name=name)


def orient_outward(m: Mesh) -> Mesh:
    """Flip the whole mesh if most faces point towards its centroid (helper for lofts
    whose winding depends on the section orientation)."""
    fn = m.face_normals()
    cent = m.face_centroids()
    c = m.vertices.mean(axis=0)
    if np.mean(np.einsum("ij,ij->i", fn, cent - c)) < 0:
        m.flip_normals()
    return m


def knob(radius: float, height: float, material_body: str, material_cap: str, n: int = 20,
         center=(0.0, 0.0, 0.0), pointer: Optional[str] = None) -> Mesh:
    """Rotary knob: body cylinder, chrome-ish cap ring and an optional pointer mark."""
    cx, cy, cz = center
    m = P.cylinder(radius, height, n, material=material_body, center=(cx, cy, cz + height / 2), name="knob")
    m.merge(P.cylinder(radius * 0.72, 0.004, n, material=material_cap, center=(cx, cy, cz + height + 0.002), name="knob_cap"))
    if pointer:
        m.merge(P.box(radius * 0.18, radius * 0.9, 0.002, material=pointer, center=(cx, cy + radius * 0.5, cz + height + 0.005), name="pointer"))
    return m


def button_row(n: int, pitch: float, size: Tuple[float, float], height: float, material: str,
               center=(0.0, 0.0, 0.0), along="x") -> Mesh:
    """A row of n small rounded buttons."""
    cx, cy, cz = center
    out = None
    for k in range(n):
        d = (k - (n - 1) / 2) * pitch
        c = (cx + d, cy, cz) if along == "x" else (cx, cy + d, cz)
        b = P.box(size[0], size[1], height, material=material, center=(c[0], c[1], c[2] + height / 2), name="button")
        out = b if out is None else out.merge(b)
    return out
