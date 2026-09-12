"""End-to-end: config -> body -> assembly -> output folder."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .config import CarConfig
from .body import CarBody, BodyResult
from .body.styles import STYLE_DESCRIPTIONS
from .assembly import assemble, CarAssembly
from .export.obj import write_obj
from .export.render import Renderer, Camera
from .export.png import write_png
from .geometry.mesh import Mesh


@dataclass
class BuildOutput:
    config: CarConfig
    body: BodyResult
    assembly: CarAssembly
    out_dir: Optional[Path]
    files: List[Path]
    timings: Dict[str, float]


_BODY_CACHE: Dict[tuple, CarBody] = {}


def car_body_for(cfg: CarConfig) -> CarBody:
    params = cfg.base_params()
    key = tuple(sorted(params.to_dict().items()))
    if key not in _BODY_CACHE:
        _BODY_CACHE[key] = CarBody(params)
    return _BODY_CACHE[key]


def randomised_modifiers(cfg: CarConfig, body: CarBody) -> Dict[str, float]:
    """Apply body.random jitter on top of the configured modifiers."""
    mods = dict(cfg.body.get("modifiers") or {})
    rnd = cfg.body.get("random") or {}
    amount = float(rnd.get("amount", 0.0) or 0.0)
    if amount <= 0:
        return mods
    rng = np.random.default_rng(cfg.seed)
    groups = set(rnd.get("groups") or [])
    exclude = set(rnd.get("exclude") or [])
    for name, mod in body.library.modifiers.items():
        if mod.group in groups and name not in exclude and name not in mods and not mod.is_unipolar:
            mods[name] = float(np.clip(rng.normal(0.0, amount * 0.5), -amount, amount))
    return mods


def build_car(cfg: CarConfig) -> tuple[BodyResult, CarAssembly, Dict[str, float]]:
    timings: Dict[str, float] = {}
    t = time.time()
    body = car_body_for(cfg)
    timings["library"] = time.time() - t
    t = time.time()
    mods = randomised_modifiers(cfg, body)
    res = body.build(cfg.body.get("style", "sedan"), mods, cfg.body.get("sculpt") or {}, cfg.hints(),
                     paint=cfg.palette().paint)
    timings["body"] = time.time() - t
    t = time.time()
    asm = assemble(res, cfg.components, cfg.palette(), cfg.seed)
    timings["assembly"] = time.time() - t
    return res, asm, timings


# ------------------------------------------------------------------ views
def _camera(view: str, mesh_bounds, meas: Dict[str, float], asm: CarAssembly) -> tuple[Camera, dict]:
    lo, hi = mesh_bounds
    c = (lo + hi) / 2
    c[2] = 0.72
    L = float(hi[0] - lo[0])
    d = 2.05 * L
    if view == "three_quarter_front":
        return Camera.orbit(c, d, 35, 16, fov_deg=30), {}
    if view == "three_quarter_rear":
        return Camera.orbit(c, d, -140, 18, fov_deg=30), {}
    if view == "side":
        return Camera.orbit(c, 4 * L, 90, 0, ortho=True, ortho_height=L * 0.52), {}
    if view == "top":
        return Camera.orbit(c, 4 * L, 90, 89.9, ortho=True, ortho_height=L * 0.52), {}
    if view == "front":
        return Camera.orbit(c, 4 * L, 0, 0, ortho=True, ortho_height=2.4), {}
    if view == "rear":
        return Camera.orbit(c, 4 * L, 180, 0, ortho=True, ortho_height=2.4), {}
    if view == "interior_cutaway":
        cc = c.copy()
        cc[2] = 0.75
        return Camera.orbit(cc, 1.45 * L, 42, 42, fov_deg=32), {"cutaway": True}
    if view == "interior_side":
        cc = c.copy()
        cc[2] = 0.8
        return Camera.orbit(cc, 0.95 * L, 96, 12, fov_deg=40), {"clip_y": -0.04}
    if view == "driver":
        try:
            seat = asm.body.connector("seat_front_driver")
            hp = seat.meta.get("h_point_height", 0.27)
            o = seat.origin
            eye = np.array([o[0] - 0.15, o[1], o[2] + hp + 0.64])
            target = np.array([o[0] + 2.5, o[1] * 0.35, o[2] + hp + 0.42])
        except KeyError:
            eye = np.array([c[0], 0.4, 1.1])
            target = eye + np.array([3, 0, -0.2])
        return Camera(eye, target, fov_deg=78), {}
    raise KeyError(f"unknown view {view!r}")


CUTAWAY_HIDE_TAGS = {"glass", "headliner", "antenna", "roof_rail", "rearview_mirror", "dome_light", "grab_handle", "mirror",
                     "pillar_trim"}


def cutaway_mesh(asm: CarAssembly) -> Mesh:
    """Body below the belt line + every component except the roof-related ones:
    a clean 'roof off' view of the interior without cut-face shards."""
    body = asm.body.mesh
    zc = asm.body.measurements["z_belt"] + 0.06
    vz = body.vertices[:, 2]
    keep = [i for i, f in enumerate(body.faces) if vz[list(f)].max() < zc]
    m = body.subset(keep)
    m.lines = [l for l in body.lines if l[:, 2].max() < zc]
    m.line_names = list(body.line_names[: len(m.lines)])
    m.materials.update(body.materials)
    for inst in asm.instances:
        top = inst.connector
        root = asm.body.connector(top.name.split("/")[0]) if "/" in top.name else top
        if root.tags & CUTAWAY_HIDE_TAGS or top.tags & CUTAWAY_HIDE_TAGS:
            continue
        # a part that is mostly above the cut (upper pillar skins) would leave
        # fragments floating over the open cabin
        plo, phi = inst.result.mesh.bounds()
        if plo[2] > zc - 0.02:
            continue
        m.merge(inst.result.mesh)
    return m


def render_views(asm: CarAssembly, views: List[str], size=(1200, 800)) -> Dict[str, np.ndarray]:
    full = asm.mesh()
    b = full.bounds()
    out = {}
    W, H = int(size[0]), int(size[1])
    cut = None
    for v in views:
        cam, kw = _camera(v, b, asm.body.measurements, asm)
        w, h = (W, H)
        if v in ("side", "top"):
            w, h = W, int(H * 0.55)
        if v in ("front", "rear"):
            w, h = int(W * 0.5), int(H * 0.7)
        target = full
        if kw.pop("cutaway", False):
            cut = cutaway_mesh(asm) if cut is None else cut
            target = cut
        img = Renderer(w, h, 2).render(target, cam, **kw)
        out[v] = img
    return out


def compose_sheets(images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    sheets = {}
    ext_rows = []
    if "three_quarter_front" in images or "three_quarter_rear" in images:
        ext_rows.append(Renderer.hstack([images[k] for k in ("three_quarter_front", "three_quarter_rear") if k in images]))
    if "side" in images or "top" in images:
        ext_rows.append(Renderer.hstack([images[k] for k in ("side", "top") if k in images]))
    if "front" in images or "rear" in images:
        ext_rows.append(Renderer.hstack([images[k] for k in ("front", "rear") if k in images]))
    if ext_rows:
        sheets["preview"] = Renderer.vstack(ext_rows)
    int_imgs = [images[k] for k in ("interior_cutaway", "interior_side", "driver") if k in images]
    if int_imgs:
        sheets["interior"] = Renderer.hstack(int_imgs)
    return sheets


# ------------------------------------------------------------------ output
def write_outputs(cfg: CarConfig, res: BodyResult, asm: CarAssembly, out_dir: Path, timings: Dict[str, float]) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    formats = cfg.output.get("formats", [])
    full = asm.mesh()
    if "obj" in formats:
        t = time.time()
        files.append(write_obj(full, out_dir / "car.obj", object_groups=asm.object_groups()))
        files.append(out_dir / "car.mtl")
        if cfg.output.get("write_body", True):
            files.append(write_obj(res.mesh, out_dir / "body.obj"))
            files.append(out_dir / "body.mtl")
        timings["obj"] = time.time() - t
    if "json" in formats:
        (out_dir / "connectors.json").write_text(json.dumps(asm.connectors_dict(), indent=1))
        files.append(out_dir / "connectors.json")
        info = {
            "name": cfg.name,
            "description": cfg.raw.get("description", ""),
            "config": cfg.raw,
            "style_description": STYLE_DESCRIPTIONS.get(str(cfg.body.get("style")), ""),
            **asm.to_dict(),
            "mesh": full.stats(),
            "timings_s": {k: round(v, 3) for k, v in timings.items()},
        }
        (out_dir / "assembly.json").write_text(json.dumps(info, indent=1, default=str))
        files.append(out_dir / "assembly.json")
        (out_dir / "spec.txt").write_text(spec_text(cfg, res, asm))
        files.append(out_dir / "spec.txt")
    if "targets" in formats:
        tdir = out_dir / "targets"
        tdir.mkdir(exist_ok=True)
        body = car_body_for(cfg)
        for name, value in res.modifier_values.items():
            mod = body.library.modifiers[name]
            for tgt, w in mod.weights(value):
                p = tdir / (tgt.name + ".target")
                tgt.save(p)
                files.append(p)
    if "svg" in formats:
        t = time.time()
        try:
            from .export.svg import write_blueprint, write_connector_map
        except ImportError as e:  # pragma: no cover
            print(f"  [svg] exporter unavailable: {e}")
        else:
            theme = cfg.output.get("blueprint_theme", "blueprint")
            desc = str(cfg.raw.get("description", "") or "")
            subtitle = desc if len(desc) <= 64 else desc[:61].rstrip() + "..."
            files.append(write_blueprint(out_dir / "blueprint.svg", asm.exterior_mesh(), asm.connectors, res.measurements,
                                         title=cfg.name, subtitle=subtitle,
                                         show_connectors=bool(cfg.output.get("blueprint_connectors", False)),
                                         interior_mesh=asm.interior_mesh(), theme=theme))
            if cfg.output.get("connector_map", True):
                files.append(write_connector_map(out_dir / "connector_map.svg", asm.exterior_mesh(), asm.connectors, title=cfg.name))
        timings["svg"] = time.time() - t
    if "png" in formats:
        t = time.time()
        images = render_views(asm, cfg.output.get("views", []), cfg.output.get("image_size", [1200, 800]))
        for name, img in compose_sheets(images).items():
            files.append(write_png(out_dir / f"{name}.png", img))
        timings["png"] = time.time() - t
    return files


def spec_text(cfg: CarConfig, res: BodyResult, asm: CarAssembly) -> str:
    m = res.measurements
    lines = [f"{cfg.name}", "=" * len(cfg.name), ""]
    if cfg.raw.get("description"):
        lines += [cfg.raw["description"], ""]
    lines += [
        f"style:          {cfg.body.get('style')}",
        f"modifiers:      {json.dumps(res.modifier_values)}",
        "",
        "Dimensions (m)",
        f"  length        {m['length']:.3f}",
        f"  width         {m['width']:.3f}",
        f"  height        {m['height']:.3f}",
        f"  wheelbase     {m['wheelbase']:.3f}",
        f"  track         {m['track']:.3f}",
        f"  ground clr.   {m['z_floor']:.3f}",
        f"  tyre radius   {m['arch_front_r'] - res.hints.get('arch_gap', 0.06):.3f}",
        f"  belt line     {m['z_belt']:.3f}",
        f"  cowl          x={m['x_cowl']:.3f} z={m['z_cowl']:.3f}",
        f"  roof          x={m['x_roof_front']:.3f}..{m['x_roof_rear']:.3f} z={m['z_roof_front']:.3f}",
        "",
        f"Connectors: {len(asm.connectors)}  Components: {len(asm.instances)}  Faces: {asm.to_dict()['stats']['body_faces'] + asm.to_dict()['stats']['component_faces']}",
        "",
        "Component tree",
    ]
    by_parent: Dict[Optional[str], List] = {}
    for inst in asm.instances:
        by_parent.setdefault(inst.parent, []).append(inst)

    def walk(parent, indent):
        for inst in by_parent.get(parent, []):
            lines.append(f"{'  ' * indent}- {inst.connector.name} [{inst.connector.kind}] -> {inst.component}")
            walk(inst.id, indent + 1)

    walk(None, 1)
    if asm.unattached:
        lines += ["", "Unattached connectors: " + ", ".join(asm.unattached)]
    if asm.disabled:
        lines += ["Disabled connectors: " + ", ".join(asm.disabled)]
    return "\n".join(lines) + "\n"


def run_config(cfg: CarConfig, out_root: Path, with_variants: bool = True, quiet=False) -> List[BuildOutput]:
    outputs = []
    cfgs = [cfg] + (cfg.variants() if with_variants else [])
    for c in cfgs:
        t0 = time.time()
        res, asm, timings = build_car(c)
        out_dir = Path(out_root) / c.name
        files = write_outputs(c, res, asm, out_dir, timings)
        timings["total"] = time.time() - t0
        if not quiet:
            st = asm.to_dict()["stats"]
            print(f"  {c.name:28s} {st['connectors']:3d} connectors  {st['components']:3d} components  "
                  f"{st['body_faces'] + st['component_faces']:6d} faces  {timings['total']:.1f}s  -> {out_dir}")
        outputs.append(BuildOutput(c, res, asm, out_dir, files, timings))
    return outputs
