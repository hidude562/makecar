"""Procedural mesh primitives used by components: lofts, extrusions, revolves,
boxes, cylinders, tori and polygon fills.  All return `Mesh` objects in a local
frame where +Z is "up"/"out" unless stated otherwise."""
from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np

from .mesh import Mesh, Material
from .curves import circle_points, resample_polyline, polygon_centroid_2d, polygon_area_2d


def loft(rings: Sequence[np.ndarray], closed_rings: bool = True, cap_start=False, cap_end=False,
         material: str = "default", name="loft") -> Mesh:
    """Connect successive rings (each (K,3), same K) with quads."""
    rings = [np.asarray(r, dtype=float) for r in rings]
    k = len(rings[0])
    verts = np.vstack(rings)
    faces = []
    n = len(rings)
    for i in range(n - 1):
        a, b = i * k, (i + 1) * k
        rng = range(k) if closed_rings else range(k - 1)
        for j in rng:
            j2 = (j + 1) % k
            faces.append((a + j, a + j2, b + j2, b + j))
    m = Mesh(verts, faces, [material] * len(faces), name=name)
    if cap_start:
        _cap(m, list(range(0, k)), material, flip=True)
    if cap_end:
        _cap(m, list(range((n - 1) * k, n * k)), material, flip=False)
    return m


def _cap(m: Mesh, ring: List[int], material: str, flip: bool):
    center = m.vertices[ring].mean(axis=0)
    ci = m.n_vertices
    m.vertices = np.vstack([m.vertices, center[None, :]])
    k = len(ring)
    for j in range(k):
        a, b = ring[j], ring[(j + 1) % k]
        f = (ci, b, a) if flip else (ci, a, b)
        m.faces.append(f)
        m.face_materials.append(material)


def extrude_polygon(outline2d: np.ndarray, depth: float, material="default", name="extrude",
                    z0: float = 0.0, cap=True, taper: float = 0.0) -> Mesh:
    """Extrude a CCW 2-D outline (in the XY plane) from z0 to z0+depth (+Z)."""
    p = np.asarray(outline2d, dtype=float)
    if polygon_area_2d(p) < 0:
        p = p[::-1]
    k = len(p)
    c = polygon_centroid_2d(p)
    p_top = c + (p - c) * (1.0 - taper)
    bottom = np.hstack([p, np.full((k, 1), z0)])
    top = np.hstack([p_top, np.full((k, 1), z0 + depth)])
    m = loft([bottom, top], closed_rings=True, material=material, name=name)
    if cap:
        fill_bottom = fill_polygon(p, material=material, z=z0, flip=True)
        fill_top = fill_polygon(p_top, material=material, z=z0 + depth, flip=False)
        m.merge(fill_bottom).merge(fill_top)
    return m


def fill_polygon(outline2d: np.ndarray, material="default", z: float = 0.0, flip=False, name="fill") -> Mesh:
    """Fan-triangulate a (convex-ish) 2-D outline into a flat mesh at height z."""
    p = np.asarray(outline2d, dtype=float)
    k = len(p)
    c = polygon_centroid_2d(p)
    verts = np.vstack([np.hstack([p, np.full((k, 1), z)]), np.array([[c[0], c[1], z]])])
    faces = []
    for j in range(k):
        a, b = j, (j + 1) % k
        faces.append((k, b, a) if flip else (k, a, b))
    return Mesh(verts, faces, [material] * len(faces), name=name)


def grid_fill_polygon(outline2d: np.ndarray, n_radial: int = 3, material="default", z=0.0, flip=False,
                      bulge: float = 0.0, name="gridfill") -> Mesh:
    """Fill a 2-D outline with concentric rings towards the centroid (fixed topology).
    `bulge` displaces interior rings along +Z with a dome profile."""
    p = np.asarray(outline2d, dtype=float)
    k = len(p)
    c = polygon_centroid_2d(p)
    rings = []
    for r in range(n_radial):
        t = 1.0 - r / n_radial  # 1 at outline, ->0 inside
        ring = c + (p - c) * t
        zz = z + bulge * (1.0 - t * t)
        rings.append(np.hstack([ring, np.full((k, 1), zz)]))
    m = loft(rings, closed_rings=True, material=material, name=name)
    _cap(m, list(range((n_radial - 1) * k, n_radial * k)), material, flip=False)
    m.vertices[-1, 2] = z + bulge
    if flip:
        m.flip_normals()
    return m


def box(sx: float, sy: float, sz: float, material="default", center=(0.0, 0.0, 0.0), name="box",
        bevel: float = 0.0) -> Mesh:
    """Axis-aligned box centred at `center`; optional edge bevel (chamfer)."""
    cx, cy, cz = center
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    if bevel <= 0:
        v = np.array(
            [
                [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
            ]
        ) + np.array(center)
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        return Mesh(v, f, [material] * 6, name=name)
    b = min(bevel, hx, hy, hz) * 0.999
    # build as a loft of 4 rings (bevelled box)
    def ring(h, inset):
        w, d = hx - inset, hy - inset
        return np.array([[-w, -d, h], [w, -d, h], [w, d, h], [-w, d, h]])
    rings = [ring(-hz, b), ring(-hz + b, 0), ring(hz - b, 0), ring(hz, b)]
    m = loft(rings, closed_rings=True, cap_start=True, cap_end=True, material=material, name=name)
    m.translate(center)
    return m


def cylinder(radius: float, height: float, n: int = 24, material="default", center=(0.0, 0.0, 0.0),
             radius_top: Optional[float] = None, name="cylinder", cap=True) -> Mesh:
    """Cylinder along +Z centred at `center` (height split evenly)."""
    rt = radius if radius_top is None else radius_top
    c0 = circle_points(radius, n)
    c1 = circle_points(rt, n)
    r0 = np.hstack([c0, np.full((n, 1), -height / 2)])
    r1 = np.hstack([c1, np.full((n, 1), height / 2)])
    m = loft([r0, r1], closed_rings=True, cap_start=cap, cap_end=cap, material=material, name=name)
    m.translate(center)
    return m


def tube(radius_outer: float, radius_inner: float, height: float, n: int = 24, material="default",
         center=(0.0, 0.0, 0.0), name="tube") -> Mesh:
    co, ci = circle_points(radius_outer, n), circle_points(radius_inner, n)
    zlo, zhi = -height / 2, height / 2
    rings = [
        np.hstack([ci, np.full((n, 1), zlo)]),
        np.hstack([co, np.full((n, 1), zlo)]),
        np.hstack([co, np.full((n, 1), zhi)]),
        np.hstack([ci, np.full((n, 1), zhi)]),
        np.hstack([ci, np.full((n, 1), zlo)]),
    ]
    m = loft(rings, closed_rings=True, material=material, name=name)
    m.translate(center)
    return m


def torus(radius: float, tube_radius: float, n_major: int = 32, n_minor: int = 12, material="default",
          center=(0.0, 0.0, 0.0), name="torus", squash: float = 1.0) -> Mesh:
    """Torus in the XY plane (axis +Z).  `squash` scales the tube's Z thickness."""
    rings = []
    for i in range(n_major):
        a = 2 * np.pi * i / n_major
        ca, sa = np.cos(a), np.sin(a)
        b = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)
        r = radius + tube_radius * np.cos(b)
        z = tube_radius * np.sin(b) * squash
        rings.append(np.stack([r * ca, r * sa, z], axis=1))
    # close the major loop by lofting ring i -> i+1 including last->first
    rings.append(rings[0])
    m = loft(rings, closed_rings=True, material=material, name=name)
    m.flip_normals()  # loft winding faces inward for this ring orientation
    m.translate(center)
    return m


def revolve(profile_rz: np.ndarray, n: int = 32, material="default", name="revolve", angle=2 * np.pi,
            center=(0.0, 0.0, 0.0), outward: bool = True) -> Mesh:
    """Revolve a (r, z) profile about +Z.

    Face winding follows the profile direction (normal = profile tangent x
    revolution tangent).  With `outward=True` (default) the result is flipped
    if its signed volume about the axis origin is negative, so closed-ish
    profiles always face away from the enclosed volume regardless of the
    direction they were written in."""
    prof = np.asarray(profile_rz, dtype=float)
    full = abs(angle - 2 * np.pi) < 1e-9
    angles = np.linspace(0, angle, n, endpoint=not full)
    rings = []
    for a in angles:
        rings.append(np.stack([prof[:, 0] * np.cos(a), prof[:, 0] * np.sin(a), prof[:, 1]], axis=1))
    if full:
        rings.append(rings[0])
    m = loft(rings, closed_rings=False, material=material, name=name)
    if outward and signed_volume(m) < 0:
        m.flip_normals()
    m.translate(center)
    return m


def signed_volume(m: Mesh) -> float:
    """Signed volume via the divergence theorem (positive for outward-facing closed meshes)."""
    tris, _ = m.triangulated()
    if len(tris) == 0:
        return 0.0
    v = m.vertices
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def sweep_profile(path: np.ndarray, profile2d: np.ndarray, material="default", name="sweep",
                  closed_path=False, up_hint=(0, 0, 1)) -> Mesh:
    """Sweep a 2-D profile along a 3-D path using a rotation-minimising frame."""
    path = np.asarray(path, dtype=float)
    prof = np.asarray(profile2d, dtype=float)
    if closed_path:
        path_ext = np.vstack([path, path[:1]])
    else:
        path_ext = path
    n = len(path_ext)
    tangents = np.zeros_like(path_ext)
    tangents[1:-1] = path_ext[2:] - path_ext[:-2]
    tangents[0] = path_ext[1] - path_ext[0]
    tangents[-1] = path_ext[-1] - path_ext[-2]
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    up = np.asarray(up_hint, dtype=float)
    rings = []
    prev_n = None
    for i in range(n):
        t = tangents[i]
        if prev_n is None:
            nrm = up - np.dot(up, t) * t
            if np.linalg.norm(nrm) < 1e-6:
                nrm = np.array([1.0, 0, 0]) - t[0] * t
        else:
            nrm = prev_n - np.dot(prev_n, t) * t
        nrm /= max(np.linalg.norm(nrm), 1e-12)
        bnrm = np.cross(t, nrm)
        prev_n = nrm
        rings.append(path_ext[i] + prof[:, 0:1] * bnrm + prof[:, 1:2] * nrm)
    return loft(rings, closed_rings=True, cap_start=not closed_path, cap_end=not closed_path, material=material, name=name)


def plate(width: float, height: float, thickness: float, material="default", center=(0, 0, 0), name="plate",
          bevel: float = 0.0) -> Mesh:
    return box(width, height, thickness, material=material, center=center, name=name, bevel=bevel)


def rounded_box(sx, sy, sz, radius, n_corner=3, material="default", center=(0, 0, 0), name="rbox"):
    from .curves import rounded_rect_points
    outline = rounded_rect_points(sx, sy, radius, n_corner)
    m = extrude_polygon(outline, sz, material=material, name=name, z0=-sz / 2)
    m.translate(center)
    return m


def mirror_merge(m: Mesh) -> Mesh:
    """Return m merged with its Y-mirror (car-symmetric parts)."""
    out = m.copy()
    out.merge(m.mirrored(1))
    return out
