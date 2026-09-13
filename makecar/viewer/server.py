"""HTTP server for the interactive viewer (stdlib only)."""
from __future__ import annotations

import base64
import copy
import json
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs
import numpy as np

from ..config import CarConfig, DEFAULT_CONFIG
from ..pipeline import car_body_for, randomised_modifiers, write_outputs
from ..assembly import assemble, CarAssembly
from ..body.body import BodyResult
from ..body.styles import style_names, STYLE_DESCRIPTIONS
from ..components import describe_all, REGISTRY
from ..connectors import apply_override
from .packing import pack_parts, mesh_parts, body_extra

STATIC = Path(__file__).parent / "static"


class ViewerSession:
    """Owns the live config and the last build; all mutation goes through here."""

    def __init__(self, config: CarConfig, output_dir: Optional[Path] = None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.lock = threading.Lock()
        self.body_result: Optional[BodyResult] = None
        self.assembly: Optional[CarAssembly] = None
        self.dirty = False
        self.rebuild_body()

    # --------------------------------------------------------------- engine
    @property
    def car_body(self):
        return car_body_for(self.config)

    def rebuild_body(self) -> BodyResult:
        cfg = self.config
        body = self.car_body
        mods = randomised_modifiers(cfg, body)
        self.body_result = body.build(cfg.body.get("style", "sedan"), mods, cfg.body.get("sculpt") or {},
                                      cfg.hints(), paint=cfg.palette().paint)
        self.assembly = None
        return self.body_result

    def rebuild_assembly(self) -> CarAssembly:
        if self.body_result is None:
            self.rebuild_body()
        self.assembly = assemble(self.body_result, self.config.components, self.config.palette(), self.config.seed,
                                 connectors_cfg=self.config.raw.get("connectors"))
        return self.assembly

    # ------------------------------------------------------------ state json
    def modifier_state(self) -> List[dict]:
        cfg = self.config
        values: Dict[str, float] = {}
        style = cfg.body.get("style", "sedan")
        if isinstance(style, dict):
            values.update({f"style/{k}": v for k, v in style.items()})
        elif style and style != "sedan":
            values[f"style/{style}"] = 1.0
        values.update({(k if k in self.car_body.library.modifiers else k): v for k, v in (cfg.body.get("modifiers") or {}).items()})
        values.update({(k if k.startswith("sculpt/") else f"sculpt/{k}"): v for k, v in (cfg.body.get("sculpt") or {}).items()})
        out = []
        for m in self.car_body.library.modifiers.values():
            d = m.to_dict()
            d["value"] = float(values.get(m.name, m.default))
            out.append(d)
        return out

    def connectors_json(self) -> List[dict]:
        conns = self.assembly.connectors if self.assembly else self.body_result.connectors
        overrides = self.config.connector_overrides()
        by_conn = {i.connector.name: i for i in (self.assembly.instances if self.assembly else [])}
        out = []
        for c in conns:
            if self.assembly is None and c.name in overrides:
                c = apply_override(c, overrides[c.name])
            d = c.to_dict()
            inst = by_conn.get(c.name)
            d["component"] = inst.component if inst else None
            d["options"] = inst.options if inst else {}
            d["override"] = overrides.get(c.name, {})
            d["disabled"] = c.name in (self.assembly.disabled if self.assembly else [])
            if c.kind == "circle":
                d["radius"] = c.radius
            if c.kind in ("rectangle", "polygon", "circle"):
                d["width"], d["height"] = c.width, c.height
            out.append(d)
        return out

    def state(self) -> dict:
        m = self.body_result.measurements
        return {
            "config": self.config.raw,
            "config_path": str(self.config.path) if self.config.path else None,
            "dirty": self.dirty,
            "styles": [{"name": s, "description": STYLE_DESCRIPTIONS.get(s, "")} for s in style_names()],
            "modifiers": self.modifier_state(),
            "components": describe_all(),
            "connectors": self.connectors_json(),
            "measurements": {k: round(float(v), 4) for k, v in m.items()},
            "assembled": self.assembly is not None,
            "custom_targets_dir": str(self.config.custom_targets_dir()) if self.config.custom_targets_dir() else None,
        }

    # ------------------------------------------------------------ binaries
    def body_binary(self) -> bytes:
        mesh = self.body_result.mesh
        parts = mesh_parts(mesh, component="body")
        # full aperture-less shell for sculpting/picking: use full_mesh vertices (same indexing as targets)
        full = self.body_result.full_mesh
        extra = body_extra(full)
        extra["full_positions_b64"] = base64.b64encode(full.vertices.astype(np.float32).tobytes()).decode()
        extra["full_normals_b64"] = base64.b64encode(full.vertex_normals().astype(np.float32).tobytes()).decode()
        extra["seams"] = [np.round(l, 4).tolist() for l in mesh.lines]
        extra["measurements"] = {k: round(float(v), 4) for k, v in self.body_result.measurements.items()}
        extra["connectors"] = self.connectors_json()
        return pack_parts(parts, extra)

    def assembly_binary(self) -> bytes:
        asm = self.assembly or self.rebuild_assembly()
        parts = mesh_parts(self.body_result.mesh, component="body")
        for inst in asm.instances:
            parts += mesh_parts(inst.result.mesh, component=inst.connector.name, name_prefix=f"{inst.connector.name}/")
        extra = {"connectors": self.connectors_json(), "stats": asm.to_dict()["stats"],
                 "unattached": asm.unattached, "disabled": asm.disabled}
        return pack_parts(parts, extra)

    # ------------------------------------------------------------- mutation
    def set_modifiers(self, values: Dict[str, float]) -> None:
        body = self.config.body
        mods = dict(body.get("modifiers") or {})
        sculpt = dict(body.get("sculpt") or {})
        style = body.get("style", "sedan")
        style_d = dict(style) if isinstance(style, dict) else ({style: 1.0} if style != "sedan" else {})
        for name, v in values.items():
            v = float(v)
            if name.startswith("style/"):
                key = name.split("/", 1)[1]
                if abs(v) < 1e-9:
                    style_d.pop(key, None)
                else:
                    style_d[key] = v
            elif name.startswith("sculpt/"):
                key = name.split("/", 1)[1]
                if abs(v) < 1e-9:
                    sculpt.pop(key, None)
                else:
                    sculpt[key] = v
            else:
                if abs(v) < 1e-9:
                    mods.pop(name, None)
                else:
                    mods[name] = v
        body["modifiers"], body["sculpt"] = mods, sculpt
        if not style_d:
            body["style"] = "sedan"
        elif len(style_d) == 1 and abs(list(style_d.values())[0] - 1.0) < 1e-9:
            body["style"] = list(style_d)[0]
        else:
            body["style"] = style_d
        self.dirty = True
        self.rebuild_body()

    def set_override(self, name: str, override: Optional[dict]) -> None:
        conns = self.config.raw.setdefault("connectors", {}).setdefault("overrides", {})
        if not override:
            conns.pop(name, None)
        else:
            clean = {k: v for k, v in override.items() if k in ("translate", "rotate_deg", "scale", "radius", "width", "height")
                     and v is not None}
            if clean:
                conns[name] = clean
            else:
                conns.pop(name, None)
        self.dirty = True
        self.assembly = None

    def set_assignment(self, selector: str, component: Optional[str], options: Optional[dict]) -> None:
        assign = self.config.components.setdefault("assign", {})
        if not isinstance(assign, dict):
            raise ValueError("components.assign is a list in this config; edit it in the Config tab")
        if component is None and not options:
            assign.pop(selector, None)
        else:
            entry = {}
            if component:
                entry["component"] = component
            if options:
                entry["options"] = options
            assign[selector] = entry if len(entry) > 1 or options is not None else component
        self.dirty = True
        self.assembly = None

    def set_disabled(self, selector: str, disabled: bool) -> None:
        dis = self.config.components.setdefault("disable", [])
        if disabled and selector not in dis:
            dis.append(selector)
        if not disabled and selector in dis:
            dis.remove(selector)
        self.dirty = True
        self.assembly = None

    def set_palette(self, key: str, color: str) -> None:
        self.config.raw.setdefault("palette", {})[key] = color
        self.dirty = True
        self.rebuild_body()

    def replace_config(self, raw: dict) -> None:
        new = CarConfig.from_dict(raw, path=self.config.path)
        self.config = new
        self.dirty = True
        self.rebuild_body()

    def save(self, path: Optional[str] = None) -> str:
        p = self.config.save(path)
        self.dirty = False
        return str(p)

    def save_target(self, name: str, offsets: np.ndarray) -> dict:
        import re

        slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_") or "sculpt"
        if self.config.custom_targets_dir() is None:
            self.config.body["custom_targets"] = "targets"
            # rebuild the cached body with the new directory
        body = self.car_body
        if body.custom_targets_dir is None:
            body.load_custom_targets(self.config.custom_targets_dir())
        key = body.save_custom_target(slug, offsets)
        self.set_modifiers({key: 1.0})
        self.dirty = True
        return {"modifier": key, "path": str(Path(body.custom_targets_dir) / f"{slug}.target")}

    def export(self, formats: List[str]) -> List[str]:
        cfg = copy.deepcopy(self.config)
        cfg.raw["output"]["formats"] = formats
        asm = self.assembly or self.rebuild_assembly()
        files = write_outputs(cfg, self.body_result, asm, self.output_dir / cfg.name, {})
        return [str(f) for f in files]


# ------------------------------------------------------------------ HTTP
class _Handler(BaseHTTPRequestHandler):
    session: ViewerSession = None  # type: ignore
    quiet = True

    def log_message(self, fmt, *args):  # pragma: no cover
        if not self.quiet:
            super().log_message(fmt, *args)

    # helpers -------------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(status, json.dumps(obj, default=_default).encode("utf-8"), "application/json")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw or b"{}")

    # routes ---------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        s = self.session
        try:
            if path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path.startswith("/static/"):
                f = (STATIC / path[len("/static/"):]).resolve()
                if not str(f).startswith(str(STATIC.resolve())) or not f.is_file():
                    self._send(404, b"not found", "text/plain")
                    return
                ctype = {"js": "text/javascript", "css": "text/css", "html": "text/html", "png": "image/png",
                         "svg": "image/svg+xml", "json": "application/json"}.get(f.suffix.lstrip("."), "application/octet-stream")
                self._send(200, f.read_bytes(), ctype)
            elif path == "/api/state":
                with s.lock:
                    self._json(s.state())
            elif path == "/api/body.bin":
                with s.lock:
                    self._send(200, s.body_binary(), "application/octet-stream")
            elif path == "/api/assembly.bin":
                with s.lock:
                    self._send(200, s.assembly_binary(), "application/octet-stream")
            elif path == "/api/config.yaml":
                with s.lock:
                    self._send(200, s.config.to_yaml().encode("utf-8"), "text/yaml; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            self._json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        s = self.session
        try:
            data = self._read_json()
            with s.lock:
                if path == "/api/modifiers":
                    s.set_modifiers(data.get("values") or {})
                    self._send(200, s.body_binary(), "application/octet-stream")
                elif path == "/api/connector":
                    s.set_override(data["name"], data.get("override"))
                    s.rebuild_assembly() if data.get("assemble") else None
                    self._json({"connectors": s.connectors_json(), "dirty": s.dirty})
                elif path == "/api/assign":
                    s.set_assignment(data["selector"], data.get("component"), data.get("options"))
                    self._json({"ok": True, "config": s.config.raw})
                elif path == "/api/disable":
                    s.set_disabled(data["selector"], bool(data.get("disabled", True)))
                    self._json({"ok": True, "config": s.config.raw})
                elif path == "/api/palette":
                    s.set_palette(data["key"], data["color"])
                    self._send(200, s.body_binary(), "application/octet-stream")
                elif path == "/api/config":
                    if "yaml" in data:
                        import yaml  # type: ignore
                        raw = yaml.safe_load(data["yaml"]) or {}
                    else:
                        raw = data.get("raw") or {}
                    s.replace_config(raw)
                    self._json(s.state())
                elif path == "/api/save":
                    p = s.save(data.get("path"))
                    self._json({"saved": p})
                elif path == "/api/target":
                    off = np.frombuffer(base64.b64decode(data["offsets_b64"]), dtype=np.float32).reshape(-1, 3)
                    info = s.save_target(data.get("name", "sculpt"), off.astype(float))
                    info["modifiers"] = s.modifier_state()
                    self._json(info)
                elif path == "/api/export":
                    files = s.export(data.get("formats") or ["obj", "json"])
                    self._json({"files": files})
                elif path == "/api/assemble":
                    s.rebuild_assembly()
                    self._send(200, s.assembly_binary(), "application/octet-stream")
                else:
                    self._json({"error": "unknown endpoint"}, 404)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": str(e), "trace": traceback.format_exc()}, 500)


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def make_server(session: ViewerSession, host="127.0.0.1", port=8765, quiet=True) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"session": session, "quiet": quiet})
    return ThreadingHTTPServer((host, port), handler)


def serve(config_path=None, host="127.0.0.1", port=8765, open_browser=True, output_dir=None):
    cfg = CarConfig.load(config_path) if config_path else CarConfig.from_dict({"name": "untitled"})
    session = ViewerSession(cfg, output_dir=output_dir)
    httpd = make_server(session, host, port, quiet=False)
    url = f"http://{host}:{port}/"
    print(f"makecar viewer: {url}   (config: {cfg.path or 'unsaved'})   Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
