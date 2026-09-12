"""Wavefront OBJ + MTL export (polygons kept as-is, materials by name)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

from ..geometry.mesh import Mesh, Material


def write_obj(mesh: Mesh, path, mtl_path: Optional[str] = None, write_lines: bool = True,
              object_groups: Optional[dict] = None) -> Path:
    path = Path(path)
    mtl_name = mtl_path or (path.stem + ".mtl")
    m = mesh
    with path.open("w") as fh:
        fh.write(f"# makecar export: {m.name}\n")
        fh.write(f"mtllib {Path(mtl_name).name}\n")
        fh.write(f"o {m.name}\n")
        for v in m.vertices:
            fh.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        # object groups: face index -> group name, emitted as 'g' statements
        current_mat = None
        current_grp = None
        order = range(m.n_faces)
        if object_groups:
            order = sorted(order, key=lambda i: (object_groups.get(i, ""), m.face_materials[i]))
        for i in order:
            grp = object_groups.get(i, "body") if object_groups else None
            if grp is not None and grp != current_grp:
                fh.write(f"g {grp}\n")
                current_grp = grp
            mat = m.face_materials[i]
            if mat != current_mat:
                fh.write(f"usemtl {mat}\n")
                current_mat = mat
            fh.write("f " + " ".join(str(idx + 1) for idx in m.faces[i]) + "\n")
        if write_lines and m.lines:
            fh.write("g feature_lines\n")
            base = m.n_vertices
            for pts, name in zip(m.lines, m.line_names):
                for v in pts:
                    fh.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
                fh.write("l " + " ".join(str(base + k + 1) for k in range(len(pts))) + "\n")
                base += len(pts)
    write_mtl(m, path.with_name(Path(mtl_name).name))
    return path


def write_mtl(mesh: Mesh, path) -> Path:
    path = Path(path)
    with path.open("w") as fh:
        for name in sorted(set(mesh.face_materials)):
            mat = mesh.materials.get(name, Material(name))
            r, g, b = mat.color
            fh.write(f"newmtl {name}\n")
            fh.write(f"Ka {r*0.2:.4f} {g*0.2:.4f} {b*0.2:.4f}\n")
            fh.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
            ks = 0.9 if mat.shininess > 0.5 else 0.3
            fh.write(f"Ks {ks:.3f} {ks:.3f} {ks:.3f}\n")
            fh.write(f"Ns {max(1.0, mat.shininess * 400):.1f}\n")
            fh.write(f"d {mat.alpha:.3f}\n")
            if mat.emissive > 0:
                fh.write(f"Ke {r*mat.emissive:.3f} {g*mat.emissive:.3f} {b*mat.emissive:.3f}\n")
            fh.write("illum 2\n\n")
    return path
