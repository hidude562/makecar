"""Binary mesh packing for the viewer.

Format "MCB1": uint32 magic length prefix + JSON header, then raw buffers.
Header: {"parts": [{"name", "component", "material", "color": [r,g,b], "alpha",
"pos": [offset, n_floats], "idx": [offset, n_ints]}], "extra": {...}}.
Positions are float32 xyz, indices uint32 triangle lists.  Each part is one
(component, material) draw group so the browser can colour, hide and pick
parts individually.
"""
from __future__ import annotations

import json
import struct
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..geometry.mesh import Mesh, Material


def _part_buffers(mesh: Mesh, face_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compact vertex/triangle buffers for a subset of faces (fan triangulated)."""
    faces = [mesh.faces[i] for i in face_idx]
    used = sorted({v for f in faces for v in f})
    remap = {v: k for k, v in enumerate(used)}
    pos = mesh.vertices[used].astype(np.float32)
    tris: List[Tuple[int, int, int]] = []
    for f in faces:
        r = [remap[v] for v in f]
        for k in range(1, len(r) - 1):
            tris.append((r[0], r[k], r[k + 1]))
    return pos, np.asarray(tris, dtype=np.uint32).reshape(-1, 3), np.asarray(used, dtype=np.uint32)


def pack_parts(parts: List[dict], extra: Optional[dict] = None) -> bytes:
    """parts: [{"name","component","material","color","alpha","pos": ndarray(N,3) f32, "idx": ndarray(M,3) u32}]."""
    buffers: List[bytes] = []
    header_parts = []
    offset = 0
    for p in parts:
        pos = np.ascontiguousarray(p["pos"], dtype=np.float32).tobytes()
        idx = np.ascontiguousarray(p["idx"], dtype=np.uint32).tobytes()
        mp = np.ascontiguousarray(p["map"], dtype=np.uint32).tobytes() if p.get("map") is not None else b""
        header_parts.append({
            "name": p["name"], "component": p.get("component", ""), "material": p["material"],
            "color": [round(float(c), 4) for c in p["color"]], "alpha": float(p.get("alpha", 1.0)),
            "shininess": float(p.get("shininess", 0.3)), "emissive": float(p.get("emissive", 0.0)),
            "pos": [offset, len(pos) // 4], "idx": [offset + len(pos), len(idx) // 4],
            "map": [offset + len(pos) + len(idx), len(mp) // 4],
        })
        buffers += [pos, idx, mp]
        offset += len(pos) + len(idx) + len(mp)
    header = json.dumps({"parts": header_parts, "extra": extra or {}}).encode("utf-8")
    # typed-array views need 4-byte aligned offsets: pad the header (JSON tolerates trailing spaces)
    header += b" " * (-len(header) % 4)
    return b"MCB1" + struct.pack("<I", len(header)) + header + b"".join(buffers)


def mesh_parts(mesh: Mesh, component: str = "body", name_prefix: str = "") -> List[dict]:
    """Split a mesh into one part per material."""
    by_mat: Dict[str, List[int]] = {}
    for i, m in enumerate(mesh.face_materials):
        by_mat.setdefault(m, []).append(i)
    out = []
    for mat_name, idx in by_mat.items():
        mat = mesh.materials.get(mat_name, Material(mat_name))
        pos, tris, used = _part_buffers(mesh, np.asarray(idx))
        out.append({"name": f"{name_prefix}{mat_name}", "component": component, "material": mat_name,
                    "color": mat.color, "alpha": mat.alpha, "shininess": mat.shininess, "emissive": mat.emissive,
                    "pos": pos, "idx": tris, "map": used if component == "body" else None})
    return out


def body_extra(mesh: Mesh) -> dict:
    """Per-vertex data the sculpt brush needs: full-mesh positions (float list is too big -> handled as a
    separate 'body_vertices' part) and the Y-mirror index for symmetric sculpting."""
    from ..body.generator import RING_N, mirror_index

    n = mesh.n_vertices
    ring_n = mesh.meta.get("ring_n", RING_N)
    n_rings = mesh.meta.get("n_rings", 0)
    mirror = np.arange(n, dtype=np.int64)
    ring_verts = n_rings * ring_n
    for i in range(min(ring_verts, n)):
        r, j = divmod(i, ring_n)
        mirror[i] = r * ring_n + mirror_index(j)
    return {"n_vertices": int(n), "ring_n": int(ring_n), "n_rings": int(n_rings), "mirror": mirror.tolist()}
