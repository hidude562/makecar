"""Tests for makecar.export: OBJ/MTL writer, PNG writer and the software renderer."""
from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from makecar.export import write_obj, Renderer, Camera
from makecar.export.obj import write_mtl
from makecar.export.png import write_png
from makecar.export.render import default_materials
from makecar.geometry import primitives as P
from makecar.geometry.mesh import Mesh, Material


def _lines(path, prefix):
    return [l for l in path.read_text().splitlines() if l.startswith(prefix)]


# --------------------------------------------------------------------- OBJ
class TestObj:
    def test_box_obj_and_mtl(self, tmp_path):
        b = P.box(1, 2, 3, material="paint", name="crate")
        b.add_material(Material("paint", (1.0, 0.0, 0.0), 1.0, 0.9))
        out = write_obj(b, tmp_path / "crate.obj")
        assert out == tmp_path / "crate.obj" and out.exists()
        text = out.read_text()
        assert text.startswith("# makecar export: crate\n")
        assert "mtllib crate.mtl\n" in text and "o crate\n" in text
        assert len(_lines(out, "v ")) == 8
        assert len(_lines(out, "f ")) == 6
        assert _lines(out, "usemtl") == ["usemtl paint"]
        # faces are 1-based and reference valid vertices
        for l in _lines(out, "f "):
            idx = [int(t) for t in l.split()[1:]]
            assert len(idx) == 4 and min(idx) >= 1 and max(idx) <= 8
        mtl = tmp_path / "crate.mtl"
        assert mtl.exists()
        mtxt = mtl.read_text()
        assert "newmtl paint\n" in mtxt
        assert "Kd 1.0000 0.0000 0.0000" in mtxt
        assert "Ns 360.0" in mtxt and "d 1.000" in mtxt and "illum 2" in mtxt

    def test_obj_vertices_match_mesh(self, tmp_path):
        b = P.box(1, 2, 3, center=(0.12345, -1, 2))
        out = write_obj(b, tmp_path / "b.obj", write_lines=False)
        v = np.array([[float(t) for t in l.split()[1:]] for l in _lines(out, "v ")])
        assert np.allclose(v, b.vertices, atol=1e-5)

    def test_feature_lines_and_groups(self, tmp_path):
        b = P.box(1, 1, 1)
        b.add_line([[0, 0, 0], [1, 0, 0], [1, 1, 0]], "seam")
        out = write_obj(b, tmp_path / "l.obj")
        assert len(_lines(out, "v ")) == 8 + 3
        ls = _lines(out, "l ")
        assert ls == ["l 9 10 11"]
        assert "g feature_lines" in out.read_text()
        no_lines = write_obj(b, tmp_path / "nl.obj", write_lines=False)
        assert len(_lines(no_lines, "v ")) == 8 and not _lines(no_lines, "l ")
        grouped = write_obj(b, tmp_path / "g.obj", object_groups={0: "bottom", 1: "top"}, write_lines=False)
        gs = _lines(grouped, "g ")
        assert gs == ["g body", "g bottom", "g top"]

    def test_custom_mtl_name(self, tmp_path):
        b = P.box(1, 1, 1)
        write_obj(b, tmp_path / "a.obj", mtl_path="shared.mtl")
        assert (tmp_path / "shared.mtl").exists()
        assert "mtllib shared.mtl" in (tmp_path / "a.obj").read_text()

    def test_mtl_uses_placeholder_for_unknown_materials(self, tmp_path):
        m = Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0]], [(0, 1, 2)], ["mystery"])
        p = write_mtl(m, tmp_path / "x.mtl")
        txt = p.read_text()
        assert "newmtl mystery" in txt and "Kd 0.7000 0.7000 0.7000" in txt
        em = Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0]], [(0, 1, 2)], ["led"],
                  {"led": Material("led", (1, 1, 1), 0.5, 0.2, emissive=0.8)})
        txt = write_mtl(em, tmp_path / "e.mtl").read_text()
        assert "Ke 0.800 0.800 0.800" in txt and "d 0.500" in txt

    def test_sedan_shell_exports(self, sedan, tmp_path):
        m = sedan.mesh
        out = write_obj(m, tmp_path / "sedan.obj")
        n_line_pts = sum(len(l) for l in m.lines)
        assert len(_lines(out, "v ")) == m.n_vertices + n_line_pts
        assert len(_lines(out, "f ")) == m.n_faces
        assert len(_lines(out, "l ")) == len(m.lines)
        used = {l.split()[1] for l in _lines(out, "usemtl")}
        assert used == set(m.face_materials)
        mtl = (tmp_path / "sedan.mtl").read_text()
        for name in used:
            assert f"newmtl {name}\n" in mtl
        for l in _lines(out, "f "):
            idx = [int(t) for t in l.split()[1:]]
            assert 1 <= min(idx) and max(idx) <= m.n_vertices


# --------------------------------------------------------------------- PNG
def _parse_png(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, []
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        crc = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0]
        assert crc == (zlib.crc32(tag + body) & 0xFFFFFFFF), tag
        chunks.append((tag, body))
        pos += 12 + length
    return chunks


class TestPng:
    def test_header_size_and_payload(self, tmp_path):
        h, w = 5, 7
        img = np.zeros((h, w, 3))
        img[2, 3] = (255, 128, 1)
        p = write_png(tmp_path / "a.png", img)
        chunks = _parse_png(p.read_bytes())
        assert [t for t, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
        W, H, depth, ctype = struct.unpack(">IIBB", chunks[0][1][:10])
        assert (W, H, depth, ctype) == (w, h, 8, 2)
        raw = zlib.decompress(chunks[1][1])
        assert len(raw) == h * (1 + 3 * w)
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(h, 1 + 3 * w)
        assert np.all(rows[:, 0] == 0)  # filter byte per scanline
        pix = rows[:, 1:].reshape(h, w, 3)
        assert tuple(pix[2, 3]) == (255, 128, 1) and pix.sum() == 255 + 128 + 1

    def test_clipping_and_grayscale(self, tmp_path):
        img = np.array([[-10.0, 300.0], [12.6, 0.0]])
        p = write_png(tmp_path / "g.png", img)
        chunks = _parse_png(p.read_bytes())
        raw = zlib.decompress(chunks[1][1])
        pix = np.frombuffer(raw, dtype=np.uint8).reshape(2, 1 + 6)[:, 1:].reshape(2, 2, 3)
        assert tuple(pix[0, 0]) == (0, 0, 0) and tuple(pix[0, 1]) == (255, 255, 255)
        assert tuple(pix[1, 0]) == (12, 12, 12)


# ---------------------------------------------------------------- renderer
class TestRenderer:
    def _box(self):
        b = P.box(1, 1, 1, material="paint", center=(0, 0, 0.5))
        b.materials.update(default_materials())
        return b

    def test_renders_small_image_with_object_in_centre(self):
        ren = Renderer(120, 80, supersample=1)
        cam = Camera.orbit([0, 0, 0.5], 6.0, 35, 20)
        img = ren.render(self._box(), cam)
        assert img.shape == (80, 120, 3)
        assert img.min() >= 0 and img.max() <= 255
        assert not np.isnan(img).any()
        bg = ren.render(Mesh(), cam)
        assert bg.shape == (80, 120, 3)
        assert not np.allclose(img[40, 60], bg[40, 60])
        assert np.allclose(img[0, 0], bg[0, 0])  # corner is still background

    def test_background_gradient_and_ground_toggle(self):
        ren = Renderer(40, 30, supersample=1, background=((1, 1, 1), (0, 0, 0)))
        bg = ren.render(Mesh(), Camera.orbit([0, 0, 0], 5, 0, 10))
        assert np.allclose(bg[0], 255) and np.allclose(bg[-1], 0)
        floating = P.box(1, 1, 1, material="paint", center=(0, 0, 1.5))  # hovering: the blob shadow is visible
        floating.materials.update(default_materials())
        cam = Camera.orbit([0, 0, 1.0], 7.0, 35, 25)
        with_shadow = ren.render(floating, cam, ground=True)
        without = ren.render(floating, cam, ground=False)
        assert not np.allclose(with_shadow, without)

    def test_supersample_and_ortho(self):
        cam = Camera.orbit([0, 0, 0.5], 6.0, 35, 20)
        img = Renderer(60, 40, supersample=2).render(self._box(), cam)
        assert img.shape == (40, 60, 3)
        ocam = Camera.orbit([0, 0, 0.5], 6.0, 90, 0, ortho=True, ortho_height=3.0)
        oimg = Renderer(60, 40, supersample=1).render(self._box(), ocam)
        assert oimg.shape == (40, 60, 3)
        bg = Renderer(60, 40, supersample=1).render(Mesh(), ocam)
        assert not np.allclose(oimg[20, 30], bg[20, 30])

    def test_transparent_pass_blends(self):
        cam = Camera.orbit([0, 0, 0.5], 6.0, 35, 20)
        ren = Renderer(60, 40, supersample=1)
        opaque = self._box()
        glassy = self._box()
        glassy.materials["paint"] = Material("paint", (0.72, 0.12, 0.12), alpha=0.3)
        a = ren.render(opaque, cam, ground=False)
        b = ren.render(glassy, cam, ground=False)
        bg = ren.render(Mesh(), cam, ground=False)
        # the transparent box lets background through: its centre is closer to the background
        assert np.linalg.norm(b[20, 30] - bg[20, 30]) < np.linalg.norm(a[20, 30] - bg[20, 30])

    def test_clipping_and_lines(self):
        cam = Camera.orbit([0, 0, 0.5], 6.0, 35, 20)
        ren = Renderer(60, 40, supersample=1)
        b = self._box()
        full = ren.render(b, cam, ground=False)
        clipped = ren.render(b, cam, ground=False, clip_z=0.25)
        assert not np.allclose(full, clipped)
        b.add_line([[-0.5, -0.5, 1.001], [0.5, 0.5, 1.001]], "diag")
        with_lines = ren.render(b, cam, ground=False, lines=True)
        no_lines = ren.render(b, cam, ground=False, lines=False)
        assert not np.allclose(with_lines, no_lines)

    def test_save_and_stacks(self, tmp_path):
        ren = Renderer(30, 20, supersample=1)
        cam = Camera.orbit([0, 0, 0.5], 6.0, 35, 20)
        img = ren.render(self._box(), cam)
        p = ren.save(img, tmp_path / "r.png")
        assert p.exists() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        h = Renderer.hstack([img, img[:10]], pad=4)
        assert h.shape == (20, 30 + 4 + 30, 3)
        v = Renderer.vstack([img, img[:, :10]], pad=2)
        assert v.shape == (20 + 2 + 20, 30, 3)

    def test_camera(self):
        cam = Camera.orbit([1, 2, 3], 10.0, 0, 0)
        assert np.allclose(cam.eye, [11, 2, 3]) and np.allclose(cam.target, [1, 2, 3])
        assert np.allclose(cam.up, [0, 0, 1])
        V = cam.view_matrix()
        assert np.allclose(V[:3, :3] @ V[:3, :3].T, np.eye(3))
        # the target lies straight ahead (negative z) in view space
        t = V[:3, :3] @ cam.target + V[:3, 3]
        assert np.allclose(t, [0, 0, -10])
        top = Camera([0, 0, 10], [0, 0, 0])  # looking straight down: up-vector fallback
        assert np.isfinite(top.view_matrix()).all()

    def test_sedan_renders(self, sedan):
        ren = Renderer(96, 64, supersample=1)
        cam = Camera.orbit([0, 0, 0.7], 9.0, 35, 18)
        img = ren.render(sedan.mesh, cam)
        assert img.shape == (64, 96, 3) and not np.isnan(img).any()
        bg = ren.render(Mesh(), cam)
        assert not np.allclose(img[32, 48], bg[32, 48])
