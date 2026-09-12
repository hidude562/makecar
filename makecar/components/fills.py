"""Filling polygon connectors with surfaces (glass, lenses, trim panels).

`coons_fill` reconstructs the four boundary curves of a grid aperture (the
body records the grid shape in the connector's meta) and evaluates a Coons
patch, so the fill follows the curvature of the surrounding body surface.
`concentric_fill` is the fallback for arbitrary outlines.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from ..geometry.mesh import Mesh
from ..geometry.curves import resample_polyline


def _resample(curve: np.ndarray, n: int) -> np.ndarray:
    return resample_polyline(curve, n)


def coons_patch(loop: np.ndarray, rows: int, cols: int, res_u: int, res_v: int) -> np.ndarray:
    """Evaluate a Coons patch from an ordered loop laid out as
    side0 (rows pts) + side1 (cols pts) + side2 (rows pts, reversed) + side3 (cols pts, reversed).
    Returns grid (res_u, res_v, 3)."""
    k = len(loop)
    assert k == 2 * (rows - 1) + 2 * (cols - 1), f"loop of {k} points does not match grid {rows}x{cols}"
    s0 = loop[0:rows]                                  # u: 0..1 at v=0
    s1 = loop[rows - 1 : rows - 1 + cols]              # v: 0..1 at u=1
    s2 = loop[rows - 1 + cols - 1 : rows - 1 + cols - 1 + rows][::-1]   # u at v=1
    s3 = np.vstack([loop[rows - 1 + cols - 1 + rows - 1 :], loop[:1]])[::-1]  # v at u=0
    c0, c1 = _resample(s0, res_u), _resample(s2, res_u)
    d0, d1 = _resample(s3, res_v), _resample(s1, res_v)
    u = np.linspace(0, 1, res_u)[:, None, None]
    v = np.linspace(0, 1, res_v)[None, :, None]
    P00, P10, P01, P11 = c0[0], c0[-1], c1[0], c1[-1]
    S = (1 - v) * c0[:, None, :] + v * c1[:, None, :] + (1 - u) * d0[None, :, :] + u * d1[None, :, :]
    S -= (1 - u) * (1 - v) * P00 + u * (1 - v) * P10 + (1 - u) * v * P01 + u * v * P11
    return S


def grid_mesh(grid: np.ndarray, material: str, name="patch", flip=False) -> Mesh:
    ru, rv, _ = grid.shape
    verts = grid.reshape(-1, 3)
    faces = []
    for i in range(ru - 1):
        for j in range(rv - 1):
            a = i * rv + j
            f = (a, a + rv, a + rv + 1, a + 1)
            faces.append(f[::-1] if flip else f)
    return Mesh(verts, faces, [material] * len(faces), name=name)


def resample_grid(grid: np.ndarray, res_u: int, res_v: int) -> np.ndarray:
    """Bilinearly resample a (rows, cols, 3) point grid to (res_u, res_v, 3)."""
    rows, cols, _ = grid.shape
    u = np.linspace(0, rows - 1, res_u)
    v = np.linspace(0, cols - 1, res_v)
    i0 = np.clip(np.floor(u).astype(int), 0, rows - 2)
    j0 = np.clip(np.floor(v).astype(int), 0, cols - 2)
    fu = (u - i0)[:, None, None]
    fv = (v - j0)[None, :, None]
    g00 = grid[i0][:, j0]
    g10 = grid[i0 + 1][:, j0]
    g01 = grid[i0][:, j0 + 1]
    g11 = grid[i0 + 1][:, j0 + 1]
    return (1 - fu) * (1 - fv) * g00 + fu * (1 - fv) * g10 + (1 - fu) * fv * g01 + fu * fv * g11


def coons_fill(loop: np.ndarray, rows: int, cols: int, material: str, normal: np.ndarray, offset: float = 0.0,
               border: float = 0.0, border_material: Optional[str] = None, upsample: int = 2, bulge: float = 0.0,
               thickness: float = 0.0, name="fill", grid_points: Optional[np.ndarray] = None) -> Mesh:
    """Fill a grid aperture.  `offset` moves the surface along `normal`; `border`
    (metres) paints the outermost band with `border_material`; `bulge` domes the
    interior along the normal; `thickness` adds a back face (solid pane).
    When the exact vertex grid the aperture was cut from is known
    (`grid_points`, shape (rows, cols, 3)) it is used directly instead of a
    Coons interpolation of the boundary."""
    res_u, res_v = max(2, (rows - 1) * upsample + 1), max(2, (cols - 1) * upsample + 1)
    if grid_points is not None and np.shape(grid_points)[:2] == (rows, cols):
        S = resample_grid(np.asarray(grid_points, dtype=float), res_u, res_v)
    else:
        S = coons_patch(loop, rows, cols, res_u, res_v)
    nrm = np.asarray(normal, dtype=float)
    if bulge:
        u = np.linspace(0, 1, res_u)[:, None]
        v = np.linspace(0, 1, res_v)[None, :]
        dome = np.sin(np.pi * u) * np.sin(np.pi * v)
        S = S + (dome[:, :, None] * bulge) * nrm
    S = S + offset * nrm
    m = grid_mesh(S, material, name)
    # orient faces along the normal
    fn = m.face_normals()
    if np.mean(fn @ nrm) < 0:
        m.flip_normals()
    if border > 0 and border_material:
        # approximate band width using average cell sizes
        du = np.linalg.norm(np.diff(S[:, res_v // 2, :], axis=0), axis=1).mean()
        dv = np.linalg.norm(np.diff(S[res_u // 2, :, :], axis=0), axis=1).mean()
        bu, bv = int(np.ceil(border / max(du, 1e-6))), int(np.ceil(border / max(dv, 1e-6)))
        k = 0
        for i in range(res_u - 1):
            for j in range(res_v - 1):
                if i < bu or i >= res_u - 1 - bu or j < bv or j >= res_v - 1 - bv:
                    m.face_materials[k] = border_material
                k += 1
    if thickness > 0:
        back = grid_mesh(S - thickness * nrm, material, name + "_back", flip=True)
        fnb = back.face_normals()
        if np.mean(fnb @ nrm) > 0:
            back.flip_normals()
        m.merge(back)
    return m


def concentric_fill(loop: np.ndarray, material: str, normal: np.ndarray, offset: float = 0.0, n_rings: int = 3,
                    bulge: float = 0.0, name="fill") -> Mesh:
    """Fallback fill for arbitrary loops: rings shrinking to the centroid."""
    pts = np.asarray(loop, dtype=float)
    nrm = np.asarray(normal, dtype=float)
    c = pts.mean(axis=0)
    k = len(pts)
    rings = []
    for r in range(n_rings):
        t = 1 - r / n_rings
        rings.append(c + (pts - c) * t + nrm * (offset + bulge * (1 - t * t)))
    verts = np.vstack(rings + [c + nrm * (offset + bulge)])
    faces = []
    for r in range(n_rings - 1):
        a, b = r * k, (r + 1) * k
        for j in range(k):
            j2 = (j + 1) % k
            faces.append((a + j, a + j2, b + j2, b + j))
    ci = n_rings * k
    a = (n_rings - 1) * k
    for j in range(k):
        faces.append((a + j, a + (j + 1) % k, ci))
    m = Mesh(verts, faces, [material] * len(faces), name=name)
    if np.mean(m.face_normals() @ nrm) < 0:
        m.flip_normals()
    return m


def fill_connector(conn, material: str, offset=0.0, border=0.0, border_material=None, bulge=0.0, thickness=0.0,
                   upsample=2, name="fill") -> Mesh:
    """Fill a PolygonConnector in *world* space using its grid meta if present."""
    grid = conn.meta.get("grid")
    nrm = conn.frame.z_axis
    if grid and len(conn.points) == 2 * (grid[0] - 1) + 2 * (grid[1] - 1):
        return coons_fill(conn.points, grid[0], grid[1], material, nrm, offset, border, border_material, upsample, bulge,
                          thickness, name, grid_points=conn.meta.get("grid_points"))
    return concentric_fill(conn.points, material, nrm, offset, 3, bulge, name)
