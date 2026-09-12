"""Polygon mesh container with materials, vertex groups and face zones.

Faces are stored as a list of index tuples (quads and triangles mixed) so that
the loft topology stays readable; `triangulated()` gives an (M,3) array for
rendering.  Vertex *groups* (named vertex index sets) are the MakeHuman analogue
of landmark/joint vertex groups: connectors are derived from them so that they
follow the mesh when targets are applied.  Face *zones* are named face index
sets used for materials and for apertures (windows, lights) that are removed
from the body and handed to components as polygon connectors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class Material:
    name: str
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    alpha: float = 1.0
    shininess: float = 0.3
    emissive: float = 0.0
    metallic: float = 0.0

    @staticmethod
    def parse_color(c) -> Tuple[float, float, float]:
        if isinstance(c, str):
            s = c.lstrip("#")
            if len(s) == 3:
                s = "".join(ch * 2 for ch in s)
            return tuple(int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore
        c = tuple(float(v) for v in c)
        if max(c) > 1.0:
            c = tuple(v / 255.0 for v in c)
        return c  # type: ignore

    def with_color(self, c, name: Optional[str] = None) -> "Material":
        m = Material(name or self.name, Material.parse_color(c), self.alpha, self.shininess, self.emissive, self.metallic)
        return m

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "color": [round(v, 4) for v in self.color],
            "alpha": self.alpha,
            "shininess": self.shininess,
            "emissive": self.emissive,
            "metallic": self.metallic,
        }


DEFAULT_MATERIAL = Material("default")


class Mesh:
    def __init__(
        self,
        vertices=None,
        faces: Optional[Sequence[Sequence[int]]] = None,
        face_materials: Optional[Sequence[str]] = None,
        materials: Optional[Dict[str, Material]] = None,
        name: str = "mesh",
    ):
        self.name = name
        self.vertices = np.zeros((0, 3)) if vertices is None else np.asarray(vertices, dtype=float).reshape(-1, 3)
        self.faces: List[Tuple[int, ...]] = [tuple(int(i) for i in f) for f in (faces or [])]
        if face_materials is None:
            face_materials = [DEFAULT_MATERIAL.name] * len(self.faces)
        self.face_materials: List[str] = list(face_materials)
        self.materials: Dict[str, Material] = dict(materials or {})
        if DEFAULT_MATERIAL.name not in self.materials and DEFAULT_MATERIAL.name in self.face_materials:
            self.materials[DEFAULT_MATERIAL.name] = DEFAULT_MATERIAL
        self.groups: Dict[str, np.ndarray] = {}
        self.zones: Dict[str, np.ndarray] = {}
        self.lines: List[np.ndarray] = []  # decorative polylines (seams etc.), world coords (K,3)
        self.line_names: List[str] = []
        self.meta: Dict[str, Any] = {}     # generator bookkeeping (copied, not exported)

    # ------------------------------------------------------------- basics
    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def copy(self) -> "Mesh":
        m = Mesh(self.vertices.copy(), list(self.faces), list(self.face_materials), dict(self.materials), self.name)
        m.groups = {k: v.copy() for k, v in self.groups.items()}
        m.zones = {k: v.copy() for k, v in self.zones.items()}
        m.lines = [l.copy() for l in self.lines]
        m.line_names = list(self.line_names)
        m.meta = dict(self.meta)
        return m

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n_vertices == 0:
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def add_material(self, mat: Material) -> Material:
        self.materials[mat.name] = mat
        return mat

    def set_material(self, name: str, face_indices=None):
        if face_indices is None:
            self.face_materials = [name] * self.n_faces
        else:
            for i in np.asarray(face_indices).ravel():
                self.face_materials[int(i)] = name

    def add_group(self, name: str, indices: Iterable[int]):
        self.groups[name] = np.asarray(list(indices), dtype=int)

    def add_zone(self, name: str, face_indices: Iterable[int]):
        self.zones[name] = np.asarray(list(face_indices), dtype=int)

    def add_line(self, pts, name: str = "line"):
        self.lines.append(np.asarray(pts, dtype=float).reshape(-1, 3))
        self.line_names.append(name)

    # ---------------------------------------------------------- transforms
    def transform(self, m4: np.ndarray) -> "Mesh":
        r, t = m4[:3, :3], m4[:3, 3]
        self.vertices = self.vertices @ r.T + t
        self.lines = [l @ r.T + t for l in self.lines]
        if np.linalg.det(r) < 0:
            self.faces = [tuple(reversed(f)) for f in self.faces]
        return self

    def translate(self, d) -> "Mesh":
        d = np.asarray(d, dtype=float)
        self.vertices = self.vertices + d
        self.lines = [l + d for l in self.lines]
        return self

    def scale(self, s) -> "Mesh":
        s = np.asarray(s, dtype=float)
        self.vertices = self.vertices * s
        self.lines = [l * s for l in self.lines]
        if np.ndim(s) and np.prod(np.sign(s)) < 0:
            self.faces = [tuple(reversed(f)) for f in self.faces]
        return self

    def apply_frame(self, frame) -> "Mesh":
        """Interpret current coordinates as local to `frame` and map to world."""
        return self.transform(frame.matrix)

    def mirrored(self, axis: int = 1) -> "Mesh":
        s = np.ones(3)
        s[axis] = -1.0
        m = self.copy()
        m.scale(s)
        return m

    # ---------------------------------------------------------- topology
    def merge(self, other: "Mesh", group_prefix: Optional[str] = None) -> "Mesh":
        off = self.n_vertices
        foff = self.n_faces
        self.vertices = np.vstack([self.vertices, other.vertices]) if self.n_vertices else other.vertices.copy()
        self.faces.extend(tuple(i + off for i in f) for f in other.faces)
        self.face_materials.extend(other.face_materials)
        for k, v in other.materials.items():
            self.materials.setdefault(k, v)
        for k, v in other.groups.items():
            key = f"{group_prefix}/{k}" if group_prefix else k
            self.groups[key] = v + off
        for k, v in other.zones.items():
            key = f"{group_prefix}/{k}" if group_prefix else k
            self.zones[key] = v + foff
        self.lines.extend(l.copy() for l in other.lines)
        self.line_names.extend(other.line_names)
        return self

    def remove_faces(self, face_indices) -> "Mesh":
        drop = set(int(i) for i in np.asarray(face_indices).ravel())
        if not drop:
            return self
        keep = [i for i in range(self.n_faces) if i not in drop]
        remap = {old: new for new, old in enumerate(keep)}
        self.faces = [self.faces[i] for i in keep]
        self.face_materials = [self.face_materials[i] for i in keep]
        new_zones = {}
        for k, v in self.zones.items():
            nv = np.asarray([remap[i] for i in v if i in remap], dtype=int)
            if len(nv):
                new_zones[k] = nv
        self.zones = new_zones
        return self

    def prune_unused_vertices(self) -> "Mesh":
        used = np.zeros(self.n_vertices, dtype=bool)
        for f in self.faces:
            used[list(f)] = True
        if used.all():
            return self
        remap = -np.ones(self.n_vertices, dtype=int)
        remap[used] = np.arange(used.sum())
        self.vertices = self.vertices[used]
        self.faces = [tuple(int(remap[i]) for i in f) for f in self.faces]
        self.groups = {k: remap[v][remap[v] >= 0] for k, v in self.groups.items()}
        return self

    def triangulated(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (tris (M,3) int, tri_face_index (M,) int) using fan triangulation."""
        tris, owner = [], []
        for fi, f in enumerate(self.faces):
            for k in range(1, len(f) - 1):
                tris.append((f[0], f[k], f[k + 1]))
                owner.append(fi)
        if not tris:
            return np.zeros((0, 3), dtype=int), np.zeros(0, dtype=int)
        return np.asarray(tris, dtype=int), np.asarray(owner, dtype=int)

    def _faces_by_size(self):
        groups: Dict[int, List[int]] = {}
        for i, f in enumerate(self.faces):
            groups.setdefault(len(f), []).append(i)
        return groups

    def face_normals(self) -> np.ndarray:
        """Unit normals per face (Newell's method, vectorised per polygon size)."""
        n = np.zeros((self.n_faces, 3))
        v = self.vertices
        for size, idx in self._faces_by_size().items():
            idx = np.asarray(idx)
            F = np.asarray([self.faces[i] for i in idx], dtype=int)  # (m, size)
            P = v[F]                                                 # (m, size, 3)
            Q = np.roll(P, -1, axis=1)
            nx = np.sum((P[:, :, 1] - Q[:, :, 1]) * (P[:, :, 2] + Q[:, :, 2]), axis=1)
            ny = np.sum((P[:, :, 2] - Q[:, :, 2]) * (P[:, :, 0] + Q[:, :, 0]), axis=1)
            nz = np.sum((P[:, :, 0] - Q[:, :, 0]) * (P[:, :, 1] + Q[:, :, 1]), axis=1)
            nn = np.column_stack([nx, ny, nz])
            l = np.linalg.norm(nn, axis=1, keepdims=True)
            n[idx] = nn / np.maximum(l, 1e-12)
        return n

    def face_centroids(self) -> np.ndarray:
        c = np.zeros((self.n_faces, 3))
        v = self.vertices
        for size, idx in self._faces_by_size().items():
            idx = np.asarray(idx)
            F = np.asarray([self.faces[i] for i in idx], dtype=int)
            c[idx] = v[F].mean(axis=1)
        return c

    def vertex_normals(self) -> np.ndarray:
        fn = self.face_normals()
        vn = np.zeros_like(self.vertices)
        for f, n in zip(self.faces, fn):
            vn[list(f)] += n
        l = np.linalg.norm(vn, axis=1, keepdims=True)
        return vn / np.maximum(l, 1e-12)

    def flip_normals(self) -> "Mesh":
        self.faces = [tuple(reversed(f)) for f in self.faces]
        return self

    def boundary_loops(self, face_indices=None) -> List[List[int]]:
        """Ordered vertex loops bounding the given faces (default: whole mesh)."""
        faces = self.faces if face_indices is None else [self.faces[i] for i in np.asarray(face_indices).ravel()]
        edge_count: Dict[Tuple[int, int], int] = {}
        directed: Dict[Tuple[int, int], int] = {}
        for f in faces:
            for a, b in zip(f, f[1:] + f[:1]):
                key = (min(a, b), max(a, b))
                edge_count[key] = edge_count.get(key, 0) + 1
                directed[(a, b)] = directed.get((a, b), 0) + 1
        boundary = [(a, b) for (a, b) in directed if edge_count[(min(a, b), max(a, b))] == 1]
        nxt: Dict[int, List[int]] = {}
        for a, b in boundary:
            nxt.setdefault(a, []).append(b)
        loops = []
        used = set()
        for a, b in boundary:
            if (a, b) in used:
                continue
            loop = [a]
            cur, prev = b, a
            used.add((a, b))
            guard = 0
            while cur != a and guard < 100000:
                loop.append(cur)
                cands = [c for c in nxt.get(cur, []) if (cur, c) not in used]
                if not cands:
                    break
                c = cands[0]
                used.add((cur, c))
                prev, cur = cur, c
                guard += 1
            loops.append(loop)
        return loops

    def zone_faces(self, name: str) -> np.ndarray:
        return self.zones.get(name, np.zeros(0, dtype=int))

    def subset(self, face_indices) -> "Mesh":
        idx = np.asarray(face_indices, dtype=int).ravel()
        m = Mesh(self.vertices.copy(), [self.faces[i] for i in idx], [self.face_materials[i] for i in idx], dict(self.materials), self.name)
        m.prune_unused_vertices()
        return m

    def stats(self) -> dict:
        lo, hi = self.bounds()
        return {
            "vertices": int(self.n_vertices),
            "faces": int(self.n_faces),
            "bounds_min": [round(float(v), 4) for v in lo],
            "bounds_max": [round(float(v), 4) for v in hi],
            "materials": sorted(self.materials.keys()),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"Mesh({self.name!r}, {self.n_vertices} verts, {self.n_faces} faces)"
