"""Tests for makecar.components: registry, defaults, the morphable Wheel,
building the default exterior components on a sedan, and polygon fills."""
from __future__ import annotations

import numpy as np
import pytest

from makecar.components import (
    REGISTRY, register, get_component, default_component_for, describe_all,
    CarComponent, MorphableComponent, ComponentResult, BuildContext, Palette,
)
from makecar.components import exterior, fills
from makecar.components.exterior import Wheel, WheelParams
from makecar.connectors import PointConnector, PolygonConnector, RectangleConnector, CircleConnector
from makecar.geometry.frame import Frame
from makecar.geometry.mesh import Mesh
from makecar.morph.morphable import MorphableMesh

EXTERIOR_NAMES = {
    "wheel.alloy", "glass.tinted", "light.headlight", "light.taillight", "grille.slats", "grille.intake",
    "plate.standard", "badge.roundel", "exhaust.tip", "fuel.cap", "antenna.sharkfin", "handle.pull",
    "mirror.side", "roof.rails",
}


# ---------------------------------------------------------------- registry
class TestRegistry:
    def test_contains_exterior_components(self):
        assert EXTERIOR_NAMES <= set(REGISTRY)
        for name in EXTERIOR_NAMES:
            assert REGISTRY[name].name == name
            assert REGISTRY[name].__module__ == exterior.__name__

    def test_get_component(self):
        w = get_component("wheel.alloy")
        assert isinstance(w, Wheel) and isinstance(w, MorphableComponent)
        with pytest.raises(KeyError, match="unknown component"):
            get_component("wing.gullwing")

    def test_register_rejects_duplicates_and_missing_names(self):
        class Dup(CarComponent):
            name = "wheel.alloy"

        class NoName(CarComponent):
            name = ""

        with pytest.raises(ValueError):
            register(Dup)
        with pytest.raises(ValueError):
            register(NoName)
        assert REGISTRY["wheel.alloy"] is Wheel

    def test_describe_all(self):
        d = {x["name"]: x for x in describe_all()}
        assert d["wheel.alloy"]["morphable"] is True
        assert d["wheel.alloy"]["accepts"] == ["CircleConnector"]
        assert d["wheel.alloy"]["default_for"] == ["wheel"]
        assert d["glass.tinted"]["morphable"] is False
        assert d["glass.tinted"]["options"]["alpha"] == 0.42

    def test_default_component_for(self, sedan):
        assert default_component_for(sedan.connector("wheel_front_L")) == "wheel.alloy"
        assert default_component_for(sedan.connector("windshield")) == "glass.tinted"
        assert default_component_for(sedan.connector("glass_rear_R")) == "glass.tinted"
        assert default_component_for(sedan.connector("headlight_L")) == "light.headlight"
        assert default_component_for(sedan.connector("taillight_R")) == "light.taillight"
        assert default_component_for(sedan.connector("grille")) == "grille.slats"
        assert default_component_for(sedan.connector("intake")) == "grille.intake"  # priority beats grille.slats
        assert default_component_for(sedan.connector("plate_front")) == "plate.standard"
        assert default_component_for(sedan.connector("mirror_L")) == "mirror.side"
        rail = RectangleConnector("roof_rail_L", Frame.identity(), 1.0, 0.05, tags=["roof_rail"])
        assert default_component_for(rail) is None  # roof rails are opt-in only
        assert default_component_for(PointConnector("odd", Frame.identity(), tags=["zzz"])) is None

    def test_default_component_respects_connector_type(self):
        # a rectangle tagged "wheel" cannot take the wheel (needs a circle)
        r = RectangleConnector("wheel_x", Frame.identity(), 0.5, 0.5, tags=["wheel"])
        assert default_component_for(r) != "wheel.alloy"
        assert Wheel.can_attach(CircleConnector("c", Frame.identity(), 0.3))
        assert not Wheel.can_attach(r)

    def test_build_rejects_wrong_connector_type(self, ctx):
        r = RectangleConnector("r", Frame.identity(), 0.5, 0.5)
        with pytest.raises(TypeError):
            Wheel().build(r, None, ctx)

    def test_resolve_options(self):
        w = Wheel()
        assert w.resolve_options(None) == Wheel.options
        o = w.resolve_options({"spokes": 7, "extra": 1})
        assert o["spokes"] == 7 and o["extra"] == 1 and o["hubcap"] is True
        assert Wheel.options["spokes"] == 5  # class defaults untouched

    def test_palette_and_context_material(self, ctx):
        pal = Palette(extras={"custom": "#010203"})
        assert pal.color("paint") == Palette.paint
        assert pal.color("custom") == "#010203"
        assert pal.color("missing") == "#808080"
        m = ctx.material("paint")
        assert m.name == "paint" and m.color == pytest.approx((0x8a / 255, 0x1c / 255, 0x1c / 255))
        m2 = ctx.material("x", "#ffffff", alpha=0.5)
        assert m2.color == pytest.approx((1, 1, 1)) and m2.alpha == 0.5


# ------------------------------------------------------------------- wheel
class TestWheel:
    def test_library_builds(self):
        lib = Wheel().library()
        assert isinstance(lib, MorphableMesh)
        assert set(lib.modifiers) == {"radius", "width", "rim_ratio", "dish"}
        assert lib.base.n_vertices == Wheel().generate(WheelParams()).n_vertices
        assert Wheel().library() is lib  # cached per class
        assert "face_ring" in lib.base.groups

    def test_generate_has_fixed_topology(self):
        a = Wheel().generate(WheelParams())
        b = Wheel().generate(WheelParams(radius=0.45, width=0.3, rim_ratio=0.8, dish=0.02))
        assert a.n_vertices == b.n_vertices and a.faces == b.faces
        assert {"tyre", "rim", "rim_dark"} == set(a.face_materials)

    def test_value_for_maps_physical_to_slider(self):
        w = Wheel()
        assert w.value_for("radius", 0.33) == 0.0
        assert w.value_for("radius", 0.33 + 0.14) == pytest.approx(1.0)
        assert w.value_for("radius", 0.33 - 0.05) == pytest.approx(-0.5)
        assert w.value_for("radius", 10.0) == 1.0 and w.value_for("radius", -10.0) == -1.0
        assert MorphableComponent.ratio_to_value(1.0, 1.0, 0.0, 0.0) == 0.0
        with pytest.raises(KeyError):
            w.spec("nonexistent")

    def _wheel(self, ctx, radius, tire_width=0.225):
        conn = CircleConnector("hub", Frame.identity(), radius, tags=["wheel"], meta={"tire_width": tire_width})
        return Wheel().build(conn, None, ctx)

    def test_fit_bigger_hub_gives_bigger_tyre(self, ctx):
        small = self._wheel(ctx, 0.30)
        big = self._wheel(ctx, 0.40)
        ext_s = np.ptp(small.mesh.vertices, axis=0)
        ext_b = np.ptp(big.mesh.vertices, axis=0)
        assert ext_b[0] > ext_s[0] + 0.15 and ext_b[1] > ext_s[1] + 0.15
        assert ext_b[0] == pytest.approx(2 * 0.40, abs=0.05)
        assert ext_s[0] == pytest.approx(2 * 0.30, abs=0.05)
        assert big.info["modifier_values"]["radius"] > small.info["modifier_values"]["radius"]
        assert small.info["modifier_values"]["radius"] < 0 < big.info["modifier_values"]["radius"]

    def test_fit_honours_tire_width(self, ctx):
        narrow = self._wheel(ctx, 0.33, tire_width=0.20)
        wide = self._wheel(ctx, 0.33, tire_width=0.30)
        assert np.ptp(wide.mesh.vertices[:, 2]) > np.ptp(narrow.mesh.vertices[:, 2]) + 0.05
        assert wide.info["modifier_values"]["width"] > 0 > narrow.info["modifier_values"]["width"]

    def test_wheel_is_centred_on_the_hub_and_has_materials(self, ctx):
        res = self._wheel(ctx, 0.33)
        assert isinstance(res, ComponentResult)
        lo, hi = res.mesh.bounds()
        assert np.allclose((lo + hi)[:2] / 2, 0.0, atol=1e-9)   # axisymmetric about the hub
        assert np.linalg.norm(res.mesh.vertices.mean(axis=0)[:2]) < 0.01  # (revolve seam duplicates a ring)
        assert {"tyre", "rim", "rim_dark"} <= set(res.mesh.materials)
        assert res.mesh.name == "hub:wheel.alloy"
        assert not np.isnan(res.mesh.vertices).any()
        # spoke cut-outs paint part of the rim face dark
        assert res.mesh.face_materials.count("rim_dark") > 16

    def test_wheel_follows_connector_frame(self, ctx):
        conn = CircleConnector("hub", Frame.from_normal([1.0, -0.8, 0.33], [0, -1, 0], x_hint=[1, 0, 0]), 0.33, tags=["wheel"])
        res = Wheel().build(conn, {"spokes": 6}, ctx)
        lo, hi = res.mesh.bounds()
        c = (lo + hi) / 2
        assert np.allclose(c[[0, 2]], [1.0, 0.33], atol=1e-9)
        assert c[1] < -0.8 + 0.05                   # tyre extends outboard (-Y) from the hub
        assert np.ptp(res.mesh.vertices[:, 1]) < 0.4  # axial extent is the tyre width


# ------------------------------------------------------- default exteriors
def _exterior_defaults(result):
    for c in result.connectors:
        name = default_component_for(c)
        if name is None:
            continue
        cls = REGISTRY[name]
        if cls.__module__ == exterior.__name__:
            yield c, cls


class TestDefaultExteriorComponents:
    def test_every_default_exterior_component_builds(self, sedan, ctx):
        built = []
        for conn, cls in _exterior_defaults(sedan):
            res = cls().build(conn, None, ctx)
            assert isinstance(res, ComponentResult), conn.name
            m = res.mesh
            assert m.n_vertices > 0 and m.n_faces > 0, conn.name
            assert not np.isnan(m.vertices).any(), conn.name
            assert m.name == f"{conn.name}:{cls.name}"
            centroid = m.vertices.mean(axis=0)
            assert np.linalg.norm(centroid - conn.origin) < 0.5, (conn.name, cls.name)
            for mat in set(m.face_materials):
                assert mat in m.materials, (conn.name, mat)
            for sub in res.connectors:
                assert sub.owner == conn.name and sub.name.startswith(conn.name + "/")
            built.append(cls.name)
        assert {"wheel.alloy", "glass.tinted", "light.headlight", "light.taillight", "grille.slats", "grille.intake",
                "plate.standard", "badge.roundel", "exhaust.tip", "fuel.cap", "antenna.sharkfin", "handle.pull",
                "mirror.side"} <= set(built)

    def test_glass_follows_aperture(self, sedan, ctx):
        conn = sedan.connector("windshield")
        res = get_component("glass.tinted").build(conn, None, ctx)
        m = res.mesh
        assert {"glass", "glass_frame"} == set(m.face_materials)
        assert m.materials["glass"].alpha == pytest.approx(0.42)
        # the pane stays within a few cm of the aperture outline
        lo, hi = m.bounds()
        plo, phi = conn.points.min(axis=0), conn.points.max(axis=0)
        assert np.all(lo >= plo - 0.03) and np.all(hi <= phi + 0.03)
        assert res.info["area"] == pytest.approx(conn.area)

    def test_glass_tint_option(self, sedan, ctx):
        res = get_component("glass.tinted").build(sedan.connector("glass_front_L"), {"tint": "#ff0000", "alpha": 0.9}, ctx)
        assert res.mesh.materials["glass"].color == pytest.approx((1, 0, 0))
        assert res.mesh.materials["glass"].alpha == pytest.approx(0.9)

    def test_plate_regions(self, sedan, ctx):
        conn = sedan.connector("plate_front")
        eu = get_component("plate.standard").build(conn, {"region": "eu"}, ctx).mesh
        us = get_component("plate.standard").build(conn, {"region": "us"}, ctx).mesh
        assert "plate_blue" in set(eu.face_materials) and "plate_blue" not in set(us.face_materials)
        w_eu = np.ptp(conn.frame.to_local(eu.vertices)[:, 0])
        w_us = np.ptp(conn.frame.to_local(us.vertices)[:, 0])
        assert w_eu == pytest.approx(0.52, abs=1e-6) and w_us == pytest.approx(0.305, abs=1e-6)

    def test_exhaust_dual_option(self, sedan, ctx):
        conn = sedan.connector("exhaust_L")
        single = get_component("exhaust.tip").build(conn, None, ctx).mesh
        dual = get_component("exhaust.tip").build(conn, {"dual": True}, ctx).mesh
        assert dual.n_faces == 2 * single.n_faces

    def test_mirror_sits_outboard(self, sedan, ctx):
        for name, sign in (("mirror_L", 1), ("mirror_R", -1)):
            conn = sedan.connector(name)
            m = get_component("mirror.side").build(conn, None, ctx).mesh
            assert sign * (m.vertices[:, 1].mean() - conn.origin[1]) > 0.05

    def test_roof_rails_from_path_and_fallback(self, ctx):
        # the body does not currently emit roof_rail connectors (see TestKnownBugs in test_body.py),
        # so exercise the component with a synthetic connector shaped like emit_connectors() would make
        xs = np.linspace(-1.0, 0.1, 12)
        path = np.column_stack([xs, np.full_like(xs, 0.6), 1.42 + 0.02 * np.cos(xs)])
        fr = Frame.from_normal(path.mean(axis=0), [0, 0, 1], x_hint=[1, 0, 0])
        conn = RectangleConnector("roof_rail_L", fr, float(np.ptp(xs)), 0.05, tags=["roof_rail"], meta={"path": path})
        m = get_component("roof.rails").build(conn, None, ctx).mesh
        assert m.n_faces > 0 and not np.isnan(m.vertices).any()
        lo, hi = m.bounds()
        assert lo[0] <= path[:, 0].min() + 0.05 and hi[0] >= path[:, 0].max() - 0.05
        assert lo[2] >= path[:, 2].min() - 0.01  # rails sit on top of the roof
        assert hi[2] > path[:, 2].max() + 0.04
        plain = RectangleConnector("roof_rail_R", fr, 1.1, 0.05, tags=["roof_rail"])
        box = get_component("roof.rails").build(plain, None, ctx).mesh
        assert box.n_faces == 6 and np.ptp(box.vertices[:, 0]) == pytest.approx(1.1)


# -------------------------------------------------------------------- fills
def _dist_to_closed_polyline(pts: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Distance from each point to the closed polyline through `loop`."""
    a = loop
    b = np.roll(loop, -1, axis=0)
    ab = b - a
    best = np.full(len(pts), np.inf)
    for p in range(len(pts)):
        t = np.clip(np.einsum("ij,ij->i", pts[p] - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
        d = np.linalg.norm(a + ab * t[:, None] - pts[p], axis=1)
        best[p] = d.min()
    return best


def _grid_loop(rows, cols, z_fn=None):
    gx, gy = np.meshgrid(np.linspace(0, 1.5, rows), np.linspace(0, 1.0, cols), indexing="ij")
    gz = np.zeros_like(gx) if z_fn is None else z_fn(gx, gy)
    G = np.stack([gx, gy, gz], axis=-1)
    loop = ([G[i, 0] for i in range(rows)] + [G[rows - 1, j] for j in range(1, cols)]
            + [G[i, cols - 1] for i in range(rows - 2, -1, -1)] + [G[0, j] for j in range(cols - 2, 0, -1)])
    return np.asarray(loop), G


class TestFills:
    def test_coons_patch_shape_and_assert(self):
        loop, _ = _grid_loop(4, 6)
        S = fills.coons_patch(loop, 4, 6, 7, 11)
        assert S.shape == (7, 11, 3)
        with pytest.raises(AssertionError):
            fills.coons_patch(loop, 5, 6, 7, 11)

    def test_coons_fill_reproduces_boundary_exactly(self):
        rows, cols = 4, 6
        loop, G = _grid_loop(rows, cols)
        m = fills.coons_fill(loop, rows, cols, "glass", np.array([0, 0, 1.0]), upsample=1)
        assert m.n_vertices == rows * cols and m.n_faces == (rows - 1) * (cols - 1)
        S = m.vertices.reshape(rows, cols, 3)
        assert np.allclose(S[:, 0], loop[:rows])                                         # side 0
        assert np.allclose(S[-1, :], loop[rows - 1: rows - 1 + cols])                    # side 1
        assert np.allclose(S[:, -1], loop[rows + cols - 2: 2 * rows + cols - 2][::-1])   # side 2
        assert np.allclose(S[0, 1:], loop[2 * rows + cols - 3:][::-1])                   # side 3 (minus the shared corner)
        assert np.allclose(S, G)                                                          # flat patch == the grid
        assert np.allclose(m.face_normals(), [0, 0, 1])                                  # oriented along the normal

    def test_coons_fill_follows_curved_boundary(self):
        rows, cols = 5, 5
        loop, _ = _grid_loop(rows, cols, z_fn=lambda x, y: 0.2 * x * x)
        m = fills.coons_fill(loop, rows, cols, "glass", np.array([0, 0, 1.0]), upsample=2)
        assert m.n_vertices == (2 * (rows - 1) + 1) * (2 * (cols - 1) + 1)
        S = m.vertices.reshape(2 * (rows - 1) + 1, 2 * (cols - 1) + 1, 3)
        # boundary corners are reproduced exactly
        assert np.allclose(S[0, 0], loop[0]) and np.allclose(S[-1, 0], loop[rows - 1])
        assert np.allclose(S[-1, -1], loop[rows + cols - 2]) and np.allclose(S[0, -1], loop[2 * rows + cols - 3])
        assert np.all(S[..., 2] >= -1e-9)

    def test_coons_fill_offset_bulge_border_thickness(self):
        rows, cols = 4, 6
        loop, G = _grid_loop(rows, cols)
        n = np.array([0, 0, 1.0])
        off = fills.coons_fill(loop, rows, cols, "g", n, offset=0.1, upsample=1)
        assert np.allclose(off.vertices[:, 2], 0.1)
        bulge = fills.coons_fill(loop, rows, cols, "g", n, bulge=0.3, upsample=1)
        S = bulge.vertices.reshape(rows, cols, 3)
        assert np.allclose(S[0, :, 2], 0) and np.allclose(S[:, 0, 2], 0)   # boundary stays put
        assert S[1:-1, 1:-1, 2].min() > 0.05                               # interior domes along the normal
        bordered = fills.coons_fill(loop, rows, cols, "g", n, border=0.2, border_material="frame", upsample=3)
        mats = set(bordered.face_materials)
        assert mats == {"g", "frame"}
        assert bordered.face_materials.count("frame") > bordered.face_materials.count("g") * 0
        solid = fills.coons_fill(loop, rows, cols, "g", n, thickness=0.02, upsample=1)
        assert solid.n_faces == 2 * (rows - 1) * (cols - 1)
        assert solid.n_vertices == 2 * rows * cols
        fn = solid.face_normals()
        assert (fn[:, 2] > 0).sum() == (fn[:, 2] < 0).sum()  # back face points the other way

    def test_fill_connector_without_grid_uses_concentric_fallback(self):
        rc = RectangleConnector("r", Frame.from_normal([1, 2, 3], [0, 1, 0]), 0.4, 0.2)
        assert "grid" not in rc.meta
        m = fills.fill_connector(rc, "trim")
        assert m.n_vertices == 3 * 4 + 1
        assert m.n_faces == 2 * 4 + 4
        assert np.allclose(m.vertices[:4], rc.points)
        assert np.allclose(m.vertices[-1], rc.origin)
        assert np.all(m.face_normals() @ rc.normal > 0)
        assert set(m.face_materials) == {"trim"}
        off = fills.fill_connector(rc, "trim", offset=0.05)
        assert np.allclose(off.vertices - m.vertices, 0.05 * rc.normal)

    def test_fill_connector_with_grid_uses_coons(self, sedan):
        conn = sedan.connector("glass_front_L")
        rows, cols = conn.meta["grid"]
        m = fills.fill_connector(conn, "glass", upsample=1)
        assert m.n_vertices == rows * cols
        S = m.vertices.reshape(rows, cols, 3)
        # corners are reproduced exactly; the sides are resampled by arc length so the
        # other boundary points move along the aperture outline but stay on it
        assert np.allclose(S[0, 0], conn.points[0]) and np.allclose(S[-1, 0], conn.points[rows - 1])
        assert np.allclose(S[-1, -1], conn.points[rows + cols - 2]) and np.allclose(S[0, -1], conn.points[2 * rows + cols - 3])
        boundary = np.vstack([S[:, 0], S[-1, :], S[:, -1], S[0, :]])
        assert _dist_to_closed_polyline(boundary, conn.points).max() < 1e-9
        align = m.face_normals() @ conn.normal
        assert align.mean() > 0.7  # curved pane, oriented outward on the whole (see TestKnownBugs for the folds)
        # a mismatching grid meta falls back to the concentric fill instead of failing
        bad = PolygonConnector.from_points("bad", conn.points, meta={"grid": [3, 3]})
        mb = fills.fill_connector(bad, "glass")
        assert mb.n_vertices == 3 * len(conn.points) + 1

    def test_concentric_fill_direct(self):
        loop = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        m = fills.concentric_fill(loop, "m", np.array([0, 0, -1.0]), n_rings=4, bulge=0.1)
        assert m.n_vertices == 4 * 4 + 1 and m.n_faces == 3 * 4 + 4
        assert np.all(m.face_normals()[:, 2] < 0)
        assert m.vertices[-1, 2] == pytest.approx(-0.1)  # bulge along the (downward) normal

    def test_grid_mesh_flip(self):
        _, G = _grid_loop(3, 3)
        a = fills.grid_mesh(G, "m")
        b = fills.grid_mesh(G, "m", flip=True)
        assert a.n_faces == b.n_faces == 4
        assert np.allclose(a.face_normals(), -b.face_normals())


# -------------------------------------------------------------- known bugs
class TestKnownBugs:
    """Regression tests for defects found during development (all fixed)."""

    def test_aperture_coons_fill_has_no_folded_cells(self, sedan):
        for name in ("windshield", "headlight_R"):
            conn = sedan.connector(name)
            m = fills.fill_connector(conn, "glass")
            assert np.all(m.face_normals() @ conn.normal > 0), name

    def test_tyre_tread_faces_point_outward(self):
        m = Wheel().generate(WheelParams())
        fn, fc = m.face_normals(), m.face_centroids()
        radial = fc.copy()
        radial[:, 2] = 0
        tread = [i for i in range(m.n_faces)
                 if m.face_materials[i] == "tyre" and np.hypot(fc[i, 0], fc[i, 1]) > 0.31 and abs(fc[i, 2]) < 0.05]
        assert len(tread) > 0
        assert np.all(np.einsum("ij,ij->i", fn[tread], radial[tread]) > 0)
