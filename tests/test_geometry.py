"""Tests for makecar.geometry: frames, curves, primitives and the Mesh container."""
from __future__ import annotations

import numpy as np
import pytest

from makecar.geometry.frame import Frame, rotation_matrix, translation_matrix, scale_matrix, unit
from makecar.geometry.curves import (
    Profile, catmull_rom_closed, catmull_rom_open, resample_polyline, polyline_length,
    rounded_rect_points, circle_points, polygon_area_2d, polygon_centroid_2d, point_in_polygon_2d, smoothstep,
)
from makecar.geometry import primitives as P
from makecar.geometry.mesh import Mesh, Material
from makecar.components.fills import grid_mesh

from conftest import naive_face_normals, signed_volume, is_right_handed


# ======================================================================= Frame
class TestFrame:
    def test_identity(self):
        f = Frame.identity((1, 2, 3))
        assert np.allclose(f.origin, [1, 2, 3])
        assert np.allclose(f.rotation, np.eye(3))
        assert np.allclose(f.to_world([[0, 0, 0]]), [[1, 2, 3]])

    def test_axes_are_normalised_on_construction(self):
        f = Frame([0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4])
        assert np.allclose([np.linalg.norm(a) for a in (f.x_axis, f.y_axis, f.z_axis)], 1.0)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_to_world_to_local_roundtrip(self, seed):
        rng = np.random.default_rng(seed)
        f = Frame.from_normal(rng.normal(size=3), rng.normal(size=3), rng.normal(size=3))
        pts = rng.normal(size=(20, 3)) * 3
        assert np.allclose(f.to_local(f.to_world(pts)), pts, atol=1e-12)
        assert np.allclose(f.to_world(f.to_local(pts)), pts, atol=1e-12)

    def test_to_world_single_point_is_2d(self):
        f = Frame.identity((1, 0, 0))
        out = f.to_world([1, 1, 1])
        assert out.shape == (1, 3)
        assert np.allclose(out, [[2, 1, 1]])

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_from_normal_is_orthonormal_right_handed(self, seed):
        rng = np.random.default_rng(seed)
        normal, hint = rng.normal(size=3), rng.normal(size=3)
        f = Frame.from_normal([0, 0, 0], normal, hint)
        assert is_right_handed(f)
        assert np.allclose(f.z_axis, unit(normal))
        assert abs(np.dot(f.x_axis, f.z_axis)) < 1e-12
        # x is the hint projected onto the plane (same direction)
        proj = hint - np.dot(hint, f.z_axis) * f.z_axis
        assert np.allclose(f.x_axis, unit(proj))
        assert np.allclose(np.cross(f.x_axis, f.y_axis), f.z_axis)

    def test_from_normal_parallel_hint_falls_back(self):
        f = Frame.from_normal([0, 0, 0], [1, 0, 0], x_hint=[1, 0, 0])
        assert is_right_handed(f)
        assert np.allclose(f.z_axis, [1, 0, 0])
        f2 = Frame.from_normal([0, 0, 0], [0, 0, 1], x_hint=[0, 0, -1])
        assert is_right_handed(f2)
        assert np.allclose(f2.z_axis, [0, 0, 1])

    def test_from_axes(self):
        f = Frame.from_axes([0, 0, 0], [1, 0, 1], [0, 0, 1])  # x not perpendicular: gets projected
        assert is_right_handed(f)
        assert np.allclose(f.x_axis, [1, 0, 0])
        assert np.allclose(f.y_axis, [0, 1, 0])

    def test_unit_rejects_zero_vector(self):
        with pytest.raises(ValueError):
            unit([0, 0, 0])

    def test_matrix_matches_to_world(self):
        f = Frame.from_normal([1, 2, 3], [0, 1, 1], [1, 0, 0])
        p = np.array([0.3, -0.7, 1.1])
        hom = f.matrix @ np.append(p, 1.0)
        assert np.allclose(hom[:3], f.to_world(p)[0])
        assert np.allclose(f.dir_to_world(p)[0], f.rotation @ p)

    def test_mirrored_y_stays_right_handed(self):
        f = Frame.from_normal([1.0, 0.5, 0.2], [0.2, 0.9, 0.3], [1, 0.1, 0])
        m = f.mirrored_y()
        assert is_right_handed(m)
        M = np.diag([1.0, -1.0, 1.0])
        assert np.allclose(m.origin, M @ f.origin)
        assert np.allclose(m.z_axis, M @ f.z_axis)
        assert np.allclose(m.x_axis, M @ f.x_axis)
        # y axis is the *negated* mirror image so the frame keeps det = +1
        assert np.allclose(m.y_axis, -(M @ f.y_axis))
        # points in the local XZ plane map onto the mirror image
        loc = np.array([[0.4, 0.0, -0.2], [1.0, 0.0, 2.0]])
        assert np.allclose(m.to_world(loc), f.to_world(loc) @ M.T)
        assert np.allclose(f.mirrored_y().mirrored_y().rotation, f.rotation)

    def test_rotations_keep_orthonormality_and_invert(self):
        f = Frame.from_normal([0, 0, 0], [1, 2, 3], [0, 1, 0])
        for rot in ("rotated_about_x", "rotated_about_y", "rotated_about_z"):
            g = getattr(f, rot)(0.7)
            assert is_right_handed(g)
            back = getattr(g, rot)(-0.7)
            assert np.allclose(back.rotation, f.rotation)
            assert np.allclose(back.origin, f.origin)

    def test_rotation_quarter_turns(self):
        f = Frame.identity()
        assert np.allclose(f.rotated_about_z(np.pi / 2).x_axis, [0, 1, 0])
        assert np.allclose(f.rotated_about_z(np.pi / 2).y_axis, [-1, 0, 0])
        assert np.allclose(f.rotated_about_x(np.pi / 2).z_axis, [0, 1, 0])
        assert np.allclose(f.rotated_about_y(np.pi / 2).z_axis, [-1, 0, 0])
        assert np.allclose(f.rotated_about_y(np.pi / 2).x_axis, [0, 0, 1])

    def test_rotated_about_z_matches_rotation_matrix(self):
        f = Frame.identity()
        for ang in (0.3, -1.2, 2.5):
            r = rotation_matrix([0, 0, 1], ang)[:3, :3]
            assert np.allclose(f.rotated_about_z(ang).rotation, r)
            assert np.allclose(f.transformed(rotation_matrix([0, 0, 1], ang)).rotation, r)

    def test_translated_offset_local_transformed(self):
        f = Frame.from_normal([1, 1, 1], [0, 0, 1], [0, 1, 0])  # x = +Y, z = +Z, y = -X
        assert np.allclose(f.translated([1, 0, 0]).origin, [2, 1, 1])
        assert np.allclose(f.offset_local(dx=1).origin, [1, 2, 1])
        assert np.allclose(f.offset_local(dz=2).origin, [1, 1, 3])
        g = f.transformed(translation_matrix([0, 0, 5]))
        assert np.allclose(g.origin, [1, 1, 6])
        assert np.allclose(g.rotation, f.rotation)

    def test_to_dict_rounds(self):
        d = Frame.identity((1 / 3, 0, 0)).to_dict()
        assert set(d) == {"origin", "x_axis", "y_axis", "z_axis"}
        assert d["origin"][0] == round(1 / 3, 5)

    def test_helper_matrices(self):
        assert np.allclose(scale_matrix(2), np.diag([2, 2, 2, 1]))
        assert np.allclose(scale_matrix(1, 2, 3), np.diag([1, 2, 3, 1]))
        assert np.allclose(translation_matrix([1, 2, 3])[:3, 3], [1, 2, 3])
        r = rotation_matrix([0, 0, 1], np.pi / 2)
        assert np.allclose(r[:3, :3] @ [1, 0, 0], [0, 1, 0])


# ===================================================================== Profile
class TestProfile:
    def test_passes_through_keypoints(self):
        kp = [(0.0, 0.3), (1.0, 1.2), (2.5, 1.5), (4.0, 0.9)]
        pr = Profile(kp)
        for x, y in kp:
            assert pr(x) == pytest.approx(y)

    def test_scalar_in_float_out_array_in_array_out(self):
        pr = Profile([(0, 0), (1, 1)])
        assert isinstance(pr(0.5), float)
        out = pr(np.array([0.25, 0.5]))
        assert out.shape == (2,)
        assert np.allclose(out, [0.25, 0.5])

    def test_two_keypoints_are_linear(self):
        pr = Profile([(0, 1), (2, 3)])
        xs = np.linspace(0, 2, 11)
        assert np.allclose(pr(xs), 1 + xs)

    def test_single_keypoint_is_constant(self):
        pr = Profile([(1.0, 4.0)])
        assert pr(-3) == 4.0 and pr(7) == 4.0
        assert np.allclose(pr(np.array([0, 1, 2])), 4.0)

    def test_clamps_outside_range(self):
        pr = Profile([(0.0, 0.5), (1.0, 2.0), (3.0, 1.0)])
        assert pr(-10) == pytest.approx(0.5)
        assert pr(-1e-9) == pytest.approx(0.5)
        assert pr(3.0) == pytest.approx(1.0)
        assert pr(100) == pytest.approx(1.0)
        assert np.allclose(pr(np.array([-5, 50])), [0.5, 1.0])

    @pytest.mark.parametrize("seed", range(6))
    def test_no_overshoot_between_keypoints(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 9))
        xs = np.cumsum(rng.uniform(0.05, 1.0, n))
        ys = rng.uniform(-1, 1, n)
        pr = Profile(list(zip(xs, ys)))
        for k in range(n - 1):
            xq = np.linspace(pr.x[k], pr.x[k + 1], 200)
            v = pr(xq)
            lo, hi = min(pr.y[k], pr.y[k + 1]), max(pr.y[k], pr.y[k + 1])
            assert v.min() >= lo - 1e-9
            assert v.max() <= hi + 1e-9

    def test_monotone_data_gives_monotone_curve(self):
        pr = Profile([(0, 0), (0.5, 0.1), (0.6, 0.9), (2, 1.0), (5, 1.05)])
        v = pr(np.linspace(0, 5, 500))
        assert np.all(np.diff(v) >= -1e-12)

    def test_extremum_has_zero_tangent_without_sharp(self):
        pr = Profile([(0, 0), (1, 1), (2, 0)])
        assert pr.m[1] == pytest.approx(0.0)
        eps = 1e-6
        assert abs((pr(1) - pr(1 - eps)) / eps) < 1e-4
        assert abs((pr(1 + eps) - pr(1)) / eps) < 1e-4

    def test_sharp_keypoint_makes_a_corner(self):
        pr = Profile([(0, 0), (1, 1), (2, 0)], sharp=[False, True, False])
        eps = 1e-6
        left = (pr(1) - pr(1 - eps)) / eps
        right = (pr(1 + eps) - pr(1)) / eps
        assert left == pytest.approx(1.0, abs=1e-4)    # chord slope on the left interval
        assert right == pytest.approx(-1.0, abs=1e-4)  # chord slope on the right interval
        assert pr(1) == pytest.approx(1.0)

    def test_interval_between_two_sharp_points_is_linear(self):
        pr = Profile([(0, 0), (1, 1), (2, 0)], sharp=[True, True, True])
        xs = np.linspace(0, 1, 7)
        assert np.allclose(pr(xs), xs)
        assert np.allclose(pr(xs + 1), 1 - xs)

    def test_duplicate_x_is_nudged_not_rejected(self):
        pr = Profile([(0, 0), (0, 1), (1, 2)])
        assert np.all(np.diff(pr.x) > 0)
        assert pr(1) == pytest.approx(2)

    def test_unsorted_keypoints_are_sorted(self):
        pr = Profile([(2, 0), (0, 0), (1, 1)])
        assert np.allclose(pr.x, [0, 1, 2])
        assert pr(1) == pytest.approx(1)


# ====================================================================== curves
class TestCurves:
    SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    def test_smoothstep(self):
        assert smoothstep(0, 1, -1) == 0
        assert smoothstep(0, 1, 2) == 1
        assert smoothstep(0, 1, 0.5) == pytest.approx(0.5)
        assert np.all(np.diff(smoothstep(0, 1, np.linspace(0, 1, 50))) >= 0)

    def test_catmull_rom_closed_sample_counts(self):
        out = catmull_rom_closed(self.SQUARE, 5)
        assert out.shape == (20, 2)
        out2 = catmull_rom_closed(self.SQUARE, [1, 2, 3, 4])
        assert out2.shape == (10, 2)
        # segment start points are the control points themselves
        assert np.allclose(out[::5], self.SQUARE)
        assert np.allclose(out2[[0, 1, 3, 6]], self.SQUARE)

    def test_catmull_rom_closed_sharp_corners_are_polygon_edges(self):
        out = catmull_rom_closed(self.SQUARE, 6, sharpness=np.ones(4))
        for i in range(4):
            a, b = self.SQUARE[i], self.SQUARE[(i + 1) % 4]
            seg = out[i * 6:(i + 1) * 6]
            d = b - a
            cross = (seg[:, 0] - a[0]) * d[1] - (seg[:, 1] - a[1]) * d[0]
            assert np.allclose(cross, 0)
            t = (seg - a) @ d / (d @ d)
            assert np.all((t >= -1e-12) & (t <= 1 + 1e-12))

    def test_catmull_rom_closed_smooth_ring_is_symmetric(self):
        circle_ctrl = circle_points(1.0, 8)
        out = catmull_rom_closed(circle_ctrl, 4)
        r = np.linalg.norm(out, axis=1)
        # radial distances repeat with the control-point period
        assert np.allclose(r.reshape(8, 4), r.reshape(8, 4)[0])

    def test_catmull_rom_open_counts_and_endpoints(self):
        pts = np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=float)
        out = catmull_rom_open(pts, 4)
        assert out.shape == (13, 2)
        assert np.allclose(out[0], pts[0]) and np.allclose(out[-1], pts[-1])
        out2 = catmull_rom_open(pts, [1, 2, 3])
        assert out2.shape == (7, 2)

    def test_polyline_length(self):
        assert polyline_length(self.SQUARE) == pytest.approx(3.0)
        assert polyline_length(self.SQUARE, closed=True) == pytest.approx(4.0)

    def test_resample_straight_line_is_uniform(self):
        line = np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0]], dtype=float)
        out = resample_polyline(line, 9)
        assert out.shape == (9, 3)
        assert np.allclose(out[0], line[0]) and np.allclose(out[-1], line[-1])
        assert np.allclose(np.linalg.norm(np.diff(out, axis=0), axis=1), 0.5)

    def test_resample_preserves_length_and_stays_on_polyline(self):
        path = np.array([[0, 0], [2, 0], [2, 1], [3, 1]], dtype=float)
        out = resample_polyline(path, 400)
        # resampling cuts the corners slightly, so the length is (barely) shorter
        assert polyline_length(path) - 0.01 < polyline_length(out) <= polyline_length(path) + 1e-9
        assert out[:, 0].min() >= -1e-9 and out[:, 0].max() <= 3 + 1e-9
        assert out[:, 1].min() >= -1e-9 and out[:, 1].max() <= 1 + 1e-9

    def test_resample_closed_excludes_endpoint(self):
        out = resample_polyline(self.SQUARE, 8, closed=True)
        assert out.shape == (8, 2)
        assert not np.allclose(out[0], out[-1])
        assert np.allclose(np.linalg.norm(np.diff(np.vstack([out, out[:1]]), axis=0), axis=1), 0.5)

    def test_resample_degenerate_polyline(self):
        out = resample_polyline(np.zeros((3, 2)), 5)
        assert out.shape == (5, 2) and np.allclose(out, 0)

    def test_circle_points(self):
        c = circle_points(2.0, 16)
        assert c.shape == (16, 2)
        assert np.allclose(np.linalg.norm(c, axis=1), 2.0)
        assert polygon_area_2d(c) > 0  # counter-clockwise
        assert np.allclose(c[0], [2, 0])
        c2 = circle_points(1.0, 4, start_angle=np.pi / 2)
        assert np.allclose(c2[0], [0, 1])

    @pytest.mark.parametrize("n_corner", [2, 3, 5])
    def test_rounded_rect_points(self, n_corner):
        w, h, r = 2.0, 1.0, 0.2
        pts = rounded_rect_points(w, h, r, n_corner)
        assert pts.shape == (4 * n_corner, 2)
        assert polygon_area_2d(pts) > 0  # CCW
        assert pts[:, 0].max() == pytest.approx(w / 2) and pts[:, 0].min() == pytest.approx(-w / 2)
        assert pts[:, 1].max() == pytest.approx(h / 2) and pts[:, 1].min() == pytest.approx(-h / 2)
        assert np.allclose(polygon_centroid_2d(pts), [0, 0], atol=1e-12)
        # area lies between the rectangle with its corner squares cut off and the exact rounded rectangle
        assert w * h - 4 * r * r < polygon_area_2d(pts) < w * h - (4 - np.pi) * r * r + 1e-9

    def test_rounded_rect_zero_radius_is_plain_rectangle(self):
        pts = rounded_rect_points(2.0, 1.0, 0.0)
        assert pts.shape == (4, 2)
        assert polygon_area_2d(pts) == pytest.approx(2.0)

    def test_rounded_rect_radius_is_clamped(self):
        pts = rounded_rect_points(1.0, 1.0, 10.0, 8)  # radius clamps to 0.5 -> a circle-ish shape
        assert pts.shape == (32, 2)
        assert np.allclose(np.linalg.norm(pts, axis=1), 0.5)

    def test_polygon_area_centroid_and_point_in_polygon(self):
        assert polygon_area_2d(self.SQUARE) == pytest.approx(1.0)
        assert polygon_area_2d(self.SQUARE[::-1]) == pytest.approx(-1.0)
        assert np.allclose(polygon_centroid_2d(self.SQUARE + [3, 4]), [3.5, 4.5])
        assert point_in_polygon_2d((0.5, 0.5), self.SQUARE)
        assert not point_in_polygon_2d((1.5, 0.5), self.SQUARE)


# ================================================================== primitives
class TestPrimitives:
    def test_box_is_closed_and_outward(self):
        b = P.box(1, 2, 3, center=(1, 1, 1))
        assert b.n_vertices == 8 and b.n_faces == 6
        assert all(len(f) == 4 for f in b.faces)
        fn, fc = b.face_normals(), b.face_centroids()
        assert np.all(np.sum(fn * (fc - np.array([1, 1, 1])), axis=1) > 0)
        # exactly one face per axis direction
        dirs = sorted(tuple(int(round(v)) for v in n) for n in fn)
        assert dirs == sorted([(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)])
        assert b.boundary_loops() == []
        assert signed_volume(b) == pytest.approx(6.0)
        lo, hi = b.bounds()
        assert np.allclose(lo, [0.5, 0, -0.5]) and np.allclose(hi, [1.5, 2, 2.5])

    def test_bevelled_box(self):
        b = P.box(1, 1, 1, bevel=0.1)
        assert b.n_vertices == 4 * 4 + 2
        assert b.n_faces == 3 * 4 + 2 * 4
        fn, fc = b.face_normals(), b.face_centroids()
        assert np.all(np.sum(fn * fc, axis=1) > 0)
        assert b.boundary_loops() == []
        assert 0 < signed_volume(b) < 1.0

    def test_cylinder_counts_and_orientation(self):
        n = 24
        c = P.cylinder(0.5, 1.0, n)
        assert c.n_vertices == 2 * n + 2
        assert c.n_faces == 3 * n
        fn, fc = c.face_normals(), c.face_centroids()
        assert np.all(np.sum(fn * fc, axis=1) > 0)
        assert c.boundary_loops() == []
        expected = (n / 2) * 0.25 * np.sin(2 * np.pi / n) * 1.0  # polygonal cross-section area x height
        assert signed_volume(c) == pytest.approx(expected, rel=1e-9)
        lo, hi = c.bounds()
        assert lo[2] == pytest.approx(-0.5) and hi[2] == pytest.approx(0.5)

    def test_cylinder_without_caps_and_with_taper(self):
        n = 12
        c = P.cylinder(0.5, 1.0, n, cap=False, radius_top=0.25)
        assert c.n_vertices == 2 * n and c.n_faces == n
        top = c.vertices[c.vertices[:, 2] > 0]
        assert np.allclose(np.hypot(top[:, 0], top[:, 1]), 0.25)
        assert len(c.boundary_loops()) == 2

    def test_tube_counts(self):
        n = 16
        t = P.tube(0.5, 0.4, 1.0, n)
        assert t.n_vertices == 5 * n
        assert t.n_faces == 4 * n
        assert signed_volume(t) > 0

    def test_torus_counts_and_bounds(self):
        nmaj, nmin = 16, 8
        t = P.torus(1.0, 0.2, nmaj, nmin, squash=0.5)
        assert t.n_vertices == (nmaj + 1) * nmin
        assert t.n_faces == nmaj * nmin
        assert all(len(f) == 4 for f in t.faces)
        lo, hi = t.bounds()
        assert hi[0] == pytest.approx(1.2) and lo[0] == pytest.approx(-1.2)
        assert hi[2] == pytest.approx(0.1) and lo[2] == pytest.approx(-0.1)

    def test_torus_faces_point_outward(self):
        t = P.torus(1.0, 0.2, 16, 8)
        fn, fc = t.face_normals(), t.face_centroids()
        ring = fc.copy()
        ring[:, 2] = 0
        ring /= np.linalg.norm(ring, axis=1, keepdims=True)  # nearest point on the major circle (R = 1)
        assert np.all(np.sum(fn * (fc - ring), axis=1) > 0)

    def test_revolve_counts(self):
        prof = np.array([[0.2, 0.0], [0.5, 0.5], [0.3, 1.0]])
        n = 10
        full = P.revolve(prof, n)
        assert full.n_vertices == (n + 1) * len(prof)
        assert full.n_faces == n * (len(prof) - 1)
        part = P.revolve(prof, n, angle=np.pi)
        assert part.n_vertices == n * len(prof)
        assert part.n_faces == (n - 1) * (len(prof) - 1)
        assert np.allclose(np.hypot(full.vertices[:, 0], full.vertices[:, 1]).max(), 0.5)

    def test_revolve_orientation_follows_profile_direction(self):
        # profile ordered top->bottom gives outward faces (as the wheel's spoke disc relies on)
        down = P.revolve(np.array([[1.0, 1.0], [1.0, 0.0]]), 16)
        fn, fc = down.face_normals(), down.face_centroids()
        radial = fc.copy()
        radial[:, 2] = 0
        assert np.all(np.sum(fn * radial, axis=1) > 0)

    def test_loft_quad_counts_and_caps(self):
        k, nr = 8, 3
        ring = np.column_stack([np.cos(np.linspace(0, 2 * np.pi, k, endpoint=False)),
                                np.sin(np.linspace(0, 2 * np.pi, k, endpoint=False)), np.zeros(k)])
        rings = [ring + [0, 0, z] for z in range(nr)]
        m = P.loft(rings)
        assert m.n_vertices == k * nr and m.n_faces == (nr - 1) * k
        assert all(len(f) == 4 for f in m.faces)
        m_open = P.loft(rings, closed_rings=False)
        assert m_open.n_faces == (nr - 1) * (k - 1)
        capped = P.loft(rings, cap_start=True, cap_end=True)
        assert capped.n_vertices == k * nr + 2
        assert capped.n_faces == (nr - 1) * k + 2 * k
        assert capped.boundary_loops() == []
        assert signed_volume(capped) > 0

    def test_extrude_polygon_and_fill(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        m = P.extrude_polygon(sq, 2.0)
        assert m.n_vertices == 8 + 5 + 5
        assert m.n_faces == 4 + 4 + 4
        assert signed_volume(m) == pytest.approx(2.0)  # side loft and both caps face outward
        # clockwise input is reversed, not broken
        m2 = P.extrude_polygon(sq[::-1], 2.0)
        assert signed_volume(m2) == pytest.approx(2.0)
        tapered = P.extrude_polygon(sq, 1.0, taper=0.5)
        top = tapered.vertices[np.isclose(tapered.vertices[:, 2], 1.0)]
        assert top[:, 0].max() - top[:, 0].min() == pytest.approx(0.5)

    def test_fill_polygon_and_grid_fill(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        f = P.fill_polygon(sq, z=0.5)
        assert f.n_vertices == 5 and f.n_faces == 4
        assert np.allclose(f.face_normals(), [0, 0, 1])
        assert np.allclose(P.fill_polygon(sq, flip=True).face_normals(), [0, 0, -1])
        g = P.grid_fill_polygon(sq, n_radial=3, bulge=0.2)
        assert g.n_vertices == 3 * 4 + 1
        assert g.n_faces == 2 * 4 + 4
        assert g.vertices[-1, 2] == pytest.approx(0.2)
        assert np.allclose(g.vertices[:4, 2], 0.0)  # outline stays flat

    def test_rounded_box_and_plate(self):
        m = P.rounded_box(1.0, 0.5, 0.2, 0.1, center=(1, 2, 3))
        lo, hi = m.bounds()
        assert np.allclose(hi - lo, [1.0, 0.5, 0.2])
        assert np.allclose((hi + lo) / 2, [1, 2, 3])
        assert 0 < signed_volume(m) < 0.1
        pl = P.plate(1, 1, 0.1)
        assert pl.n_faces == 6

    def test_sweep_profile_counts(self):
        path = np.array([[0, 0, 0], [1, 0, 0], [2, 0.5, 0], [3, 0.5, 0.5]], dtype=float)
        prof = circle_points(0.1, 6)
        m = P.sweep_profile(path, prof)
        assert m.n_vertices == len(path) * 6 + 2
        assert m.n_faces == (len(path) - 1) * 6 + 2 * 6
        assert m.boundary_loops() == []

    def test_mirror_merge_is_symmetric(self):
        m = P.box(1, 1, 1, center=(0, 2, 0))
        mm = P.mirror_merge(m)
        assert mm.n_vertices == 16 and mm.n_faces == 12
        lo, hi = mm.bounds()
        assert lo[1] == pytest.approx(-2.5) and hi[1] == pytest.approx(2.5)
        assert signed_volume(mm) == pytest.approx(2.0)  # mirrored half keeps outward winding


# ======================================================================== Mesh
def _two_meshes():
    a = P.box(1, 1, 1, material="ma", name="a")
    a.add_group("ga", [0, 1])
    a.add_zone("za", [0])
    a.add_material(Material("ma", (1, 0, 0)))
    a.add_line([[0, 0, 0], [1, 0, 0]], "la")
    b = P.box(1, 1, 1, material="mb", center=(3, 0, 0), name="b")
    b.add_group("gb", [2, 3])
    b.add_zone("zb", [1, 5])
    b.add_material(Material("mb", (0, 1, 0)))
    b.add_line([[3, 0, 0], [4, 0, 0]], "lb")
    return a, b


class TestMesh:
    def test_constructor_and_default_material(self):
        m = Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0]], [(0, 1, 2)])
        assert m.n_vertices == 3 and m.n_faces == 1
        assert m.face_materials == ["default"]
        assert "default" in m.materials
        assert Mesh().n_vertices == 0 and np.allclose(Mesh().bounds(), 0)

    def test_material_parse_color(self):
        assert Material.parse_color("#f00") == pytest.approx((1.0, 0.0, 0.0))
        assert Material.parse_color("#00ff00") == pytest.approx((0.0, 1.0, 0.0))
        assert Material.parse_color((255, 0, 128)) == pytest.approx((1.0, 0.0, 128 / 255))
        assert Material.parse_color((0.5, 0.5, 0.5)) == pytest.approx((0.5, 0.5, 0.5))
        m = Material("x").with_color("#fff", name="y")
        assert m.name == "y" and m.color == pytest.approx((1, 1, 1))

    def test_merge_offsets_faces_groups_and_zones(self):
        a, b = _two_meshes()
        a.merge(b)
        assert a.n_vertices == 16 and a.n_faces == 12
        assert a.faces[6] == tuple(i + 8 for i in P.box(1, 1, 1).faces[0])
        assert np.array_equal(a.groups["ga"], [0, 1])
        assert np.array_equal(a.groups["gb"], [10, 11])
        assert np.array_equal(a.zones["za"], [0])
        assert np.array_equal(a.zones["zb"], [7, 11])
        assert a.face_materials == ["ma"] * 6 + ["mb"] * 6
        assert set(a.materials) >= {"ma", "mb"}
        assert a.line_names == ["la", "lb"] and len(a.lines) == 2

    def test_merge_with_group_prefix(self):
        a, b = _two_meshes()
        a.merge(b, group_prefix="wheel")
        assert "wheel/gb" in a.groups and "wheel/zb" in a.zones
        assert "gb" not in a.groups

    def test_merge_into_empty_mesh(self):
        m = Mesh()
        m.merge(P.box(1, 1, 1))
        assert m.n_vertices == 8 and m.n_faces == 6

    def test_remove_faces_remaps_zones_and_materials(self):
        m = P.box(1, 1, 1)
        m.add_zone("bottom", [0])
        m.add_zone("top", [1])
        m.add_zone("sides", [2, 3, 4, 5])
        m.set_material("special", [1])
        m.remove_faces([0, 2])
        assert m.n_faces == 4
        assert "bottom" not in m.zones          # fully removed zone disappears
        assert np.array_equal(m.zones["top"], [0])
        assert np.array_equal(m.zones["sides"], [1, 2, 3])
        assert m.face_materials[0] == "special"
        assert m.n_vertices == 8                  # vertices are not pruned implicitly
        assert m.remove_faces([]) is m

    def test_prune_unused_vertices(self):
        m = Mesh([[0, 0, 0], [9, 9, 9], [1, 0, 0], [1, 1, 0], [5, 5, 5]], [(0, 2, 3)])
        m.add_group("g", [0, 1, 3, 4])
        m.prune_unused_vertices()
        assert m.n_vertices == 3
        assert m.faces == [(0, 1, 2)]
        assert np.allclose(m.vertices[1], [1, 0, 0])
        assert np.array_equal(m.groups["g"], [0, 2])
        n = m.n_vertices
        assert m.prune_unused_vertices().n_vertices == n

    def test_subset_prunes(self):
        b = P.box(1, 1, 1)
        s = b.subset([0, 1])
        assert s.n_faces == 2 and s.n_vertices == 8
        s2 = b.subset([0])
        assert s2.n_faces == 1 and s2.n_vertices == 4

    def test_boundary_loops_grid_patch(self):
        ru, rv = 4, 6
        gx, gy = np.meshgrid(np.linspace(0, 1, ru), np.linspace(0, 1, rv), indexing="ij")
        grid = np.stack([gx, gy, np.zeros_like(gx)], axis=-1)
        m = grid_mesh(grid, "m")
        loops = m.boundary_loops()
        assert len(loops) == 1
        assert len(loops[0]) == 2 * ru + 2 * rv - 4
        assert len(set(loops[0])) == len(loops[0])
        # a subset of faces has its own boundary
        sub = m.boundary_loops([0])
        assert len(sub) == 1 and len(sub[0]) == 4

    def test_boundary_loops_single_quad_and_closed_mesh(self):
        q = Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [(0, 1, 2, 3)])
        assert q.boundary_loops() == [[0, 1, 2, 3]]
        assert P.box(1, 1, 1).boundary_loops() == []

    def test_face_normals_match_naive_newell_on_body(self, sedan):
        m = sedan.full_mesh
        assert len(set(len(f) for f in m.faces)) > 1  # mixed quads and triangles
        assert np.allclose(m.face_normals(), naive_face_normals(m), atol=1e-9)

    def test_face_normals_match_naive_newell_random(self):
        rng = np.random.default_rng(3)
        m = P.torus(1.0, 0.3, 12, 6)
        m.vertices += rng.normal(scale=0.05, size=m.vertices.shape)  # non-planar quads
        m.merge(P.cylinder(0.3, 1.0, 9))
        assert np.allclose(m.face_normals(), naive_face_normals(m), atol=1e-9)
        assert np.allclose(np.linalg.norm(m.face_normals(), axis=1), 1.0)

    def test_face_centroids_and_vertex_normals(self):
        b = P.box(2, 2, 2)
        c = b.face_centroids()
        assert np.allclose(np.sort(np.abs(c).sum(axis=1)), 1.0)
        vn = b.vertex_normals()
        assert np.allclose(np.linalg.norm(vn, axis=1), 1.0)
        assert np.allclose(np.abs(vn), 1 / np.sqrt(3))

    def test_triangulated(self):
        b = P.box(1, 1, 1)
        tris, owner = b.triangulated()
        assert tris.shape == (12, 3) and owner.shape == (12,)
        assert list(owner) == [i // 2 for i in range(12)]
        e_tris, e_owner = Mesh().triangulated()
        assert e_tris.shape == (0, 3) and e_owner.shape == (0,)

    def test_transform_with_reflection_keeps_outward_winding(self):
        b = P.box(1, 2, 3)
        b.transform(np.diag([1.0, -1.0, 1.0, 1.0]))
        assert signed_volume(b) == pytest.approx(6.0)
        m = P.box(1, 2, 3).mirrored(1)
        assert signed_volume(m) == pytest.approx(6.0)
        s = P.box(1, 2, 3).scale([1, -1, 1])
        assert signed_volume(s) == pytest.approx(6.0)
        s2 = P.box(1, 2, 3).scale([2, 1, 1])
        assert signed_volume(s2) == pytest.approx(12.0)

    def test_translate_apply_frame_and_flip(self):
        b = P.box(1, 1, 1)
        b.add_line([[0, 0, 0], [1, 0, 0]])
        b.translate([1, 0, 0])
        assert np.allclose(b.vertices.mean(axis=0), [1, 0, 0])
        assert np.allclose(b.lines[0][0], [1, 0, 0])
        f = Frame.from_normal([0, 0, 5], [1, 0, 0], [0, 1, 0])
        c = P.box(1, 1, 1).apply_frame(f)
        assert np.allclose(c.vertices.mean(axis=0), [0, 0, 5])
        assert signed_volume(c) == pytest.approx(1.0)
        assert signed_volume(P.box(1, 1, 1).flip_normals()) == pytest.approx(-1.0)

    def test_copy_is_independent(self):
        a, _ = _two_meshes()
        c = a.copy()
        c.vertices[0] += 10
        c.groups["ga"][0] = 99
        c.faces.append((0, 1, 2))
        c.meta["x"] = 1
        assert not np.allclose(a.vertices[0], c.vertices[0])
        assert a.groups["ga"][0] == 0
        assert a.n_faces == 6 and "x" not in a.meta

    def test_set_material_zone_faces_and_stats(self):
        b = P.box(1, 1, 1)
        b.set_material("all")
        assert b.face_materials == ["all"] * 6
        assert b.zone_faces("nope").shape == (0,)
        st = b.stats()
        assert st["vertices"] == 8 and st["faces"] == 6
        assert st["bounds_min"] == [-0.5, -0.5, -0.5]
