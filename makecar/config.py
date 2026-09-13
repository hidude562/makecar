"""Config file schema, loading and normalisation.

A config describes one car (and optionally variants of it):

    name: family_sedan
    seed: 7
    body:
      style: sedan                       # or a blend: {sedan: 0.5, wagon: 0.5}
      modifiers: {wheelbase: 0.2, roof_height: -0.1}     # sliders in [-1, 1]
      sculpt: {haunches: 0.6}
      base_params: {}                    # advanced: regenerate the base with other canonical params
      hints: {door_count: 4, drive: left, seat_rows: auto, transmission: automatic, fuel_side: left}
      random: {amount: 0.0, groups: [proportions, greenhouse]}   # jitter modifiers by up to `amount`
    palette: {paint: "#8a1c1c", interior: "#2a2724", ...}
    components:
      defaults: true
      disable: [antenna]
      assign:
        wheel: {component: wheel.alloy, options: {spokes: 7}}
        seat_front: seat.bucket
        roof_rail: roof.rails
    output:
      formats: [obj, svg, png, json]
      views: [three_quarter_front, three_quarter_rear, side, top, front, rear, interior_cutaway, interior_side, driver]
      image_size: [1200, 800]
    variants:
      - name: family_sedan_long
        body: {modifiers: {wheelbase: 0.8}}
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .components.base import Palette
from .body.params import BodyParams

DEFAULT_VIEWS = ["three_quarter_front", "three_quarter_rear", "side", "top", "front", "rear",
                 "interior_cutaway", "interior_side", "driver"]
ALL_FORMATS = ["obj", "svg", "png", "json", "targets"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "name": "car",
    "description": "",
    "seed": 0,
    "body": {
        "style": "sedan",
        "modifiers": {},
        "sculpt": {},
        "base_params": {},
        "hints": {"door_count": None, "drive": "left", "seat_rows": "auto", "transmission": "automatic",
                  "fuel_side": "left", "column_angle": 24.0},
        "random": {"amount": 0.0, "groups": ["proportions", "greenhouse", "front", "rear", "stance", "lower_body"],
                   "exclude": ["bed_depth", "quarter_window_length"]},
        "custom_targets": None,   # directory of .target files -> custom/<name> modifiers (relative to the config)
    },
    "palette": {},
    "components": {"defaults": True, "disable": [], "assign": {}},
    "connectors": {"overrides": {}},   # per-connector edits made in the viewer (see connectors.apply_override)
    "output": {"formats": ["obj", "svg", "png", "json"], "views": DEFAULT_VIEWS, "image_size": [1200, 800],
               "connector_map": True, "write_body": True, "blueprint_theme": "blueprint", "blueprint_connectors": False},
    "variants": [],
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config_file(path) -> dict:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("PyYAML is required for YAML configs (pip install pyyaml) or use JSON") from e
        data = yaml.safe_load(text) or {}
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported config format: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    data.setdefault("name", path.stem)
    return data


@dataclass
class CarConfig:
    raw: Dict[str, Any]
    path: Optional[Path] = None

    @classmethod
    def from_dict(cls, d: dict, path=None) -> "CarConfig":
        cfg = deep_merge(DEFAULT_CONFIG, d)
        validate(cfg)
        return cls(cfg, Path(path) if path else None)

    @classmethod
    def load(cls, path) -> "CarConfig":
        return cls.from_dict(load_config_file(path), path=path)

    def custom_targets_dir(self) -> Optional[Path]:
        d = self.body.get("custom_targets")
        if not d:
            return None
        d = Path(d)
        if not d.is_absolute() and self.path is not None:
            d = self.path.parent / d
        return d

    def connector_overrides(self) -> Dict[str, dict]:
        return dict((self.raw.get("connectors") or {}).get("overrides") or {})

    def to_yaml(self) -> str:
        import yaml  # type: ignore

        clean = _strip_defaults(self.raw, DEFAULT_CONFIG)
        return yaml.safe_dump(clean, sort_keys=False, default_flow_style=None, allow_unicode=True)

    def save(self, path=None) -> Path:
        path = Path(path or self.path)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(_strip_defaults(self.raw, DEFAULT_CONFIG), indent=2))
        else:
            path.write_text(self.to_yaml())
        self.path = path
        return path

    # accessors ---------------------------------------------------------------
    @property
    def name(self) -> str:
        return str(self.raw["name"])

    @property
    def body(self) -> dict:
        return self.raw["body"]

    @property
    def components(self) -> dict:
        return self.raw["components"]

    @property
    def output(self) -> dict:
        return self.raw["output"]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 0))

    def palette(self) -> Palette:
        p = Palette()
        for k, v in (self.raw.get("palette") or {}).items():
            if hasattr(p, k) and k != "extras":
                setattr(p, k, v)
            else:
                p.extras[k] = v
        return p

    def base_params(self) -> BodyParams:
        return BodyParams.from_dict({**BodyParams().to_dict(), **(self.body.get("base_params") or {})})

    def hints(self) -> dict:
        h = {k: v for k, v in (self.body.get("hints") or {}).items() if v is not None}
        return h

    def variants(self) -> List["CarConfig"]:
        """Expand `variants` into full configs (each deep-merged over this one)."""
        out = []
        for i, v in enumerate(self.raw.get("variants") or []):
            base = copy.deepcopy(self.raw)
            base.pop("variants", None)
            merged = deep_merge(base, v)
            # a variant's style replaces the base style (a style blend is one value, not a set of keys to merge)
            if isinstance(v, dict) and "style" in (v.get("body") or {}):
                merged["body"]["style"] = copy.deepcopy(v["body"]["style"])
            merged.setdefault("name", f"{self.name}_v{i + 1}")
            if merged["name"] == self.name:
                merged["name"] = f"{self.name}_v{i + 1}"
            out.append(CarConfig(merged))
        return out

    def to_json(self) -> str:
        return json.dumps(self.raw, indent=2, default=str)


def _strip_defaults(value, default):
    """Drop keys that still equal DEFAULT_CONFIG so saved files stay readable."""
    if isinstance(value, dict) and isinstance(default, dict):
        out = {}
        for k, v in value.items():
            if k in default:
                if v == default[k]:
                    continue
                out[k] = _strip_defaults(v, default[k])
            else:
                out[k] = v
        return out
    return value


def validate(cfg: dict) -> None:
    from .body.styles import style_names

    for name, ov in ((cfg.get("connectors") or {}).get("overrides") or {}).items():
        if not isinstance(ov, dict):
            raise ValueError(f"connectors.overrides.{name} must be a mapping")
        for key in ov:
            if key not in ("translate", "rotate_deg", "scale", "radius", "width", "height"):
                raise ValueError(f"connectors.overrides.{name}: unknown key {key!r}")

    body = cfg["body"]
    style = body.get("style", "sedan")
    names = style_names()
    if isinstance(style, str):
        if style not in names:
            raise ValueError(f"body.style {style!r} unknown; choose from {names}")
    elif isinstance(style, dict):
        for k, w in style.items():
            if k not in names:
                raise ValueError(f"body.style key {k!r} unknown; choose from {names}")
            if not (0.0 <= float(w) <= 1.5):
                raise ValueError(f"body.style weight for {k!r} must be in [0, 1.5]")
    else:
        raise ValueError("body.style must be a string or a mapping of style -> weight")
    for group in ("modifiers", "sculpt"):
        for k, v in (body.get(group) or {}).items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"body.{group}.{k} must be a number") from None
            if not (-1.0 <= fv <= 1.0):
                raise ValueError(f"body.{group}.{k} must be in [-1, 1] (MakeHuman-style slider)")
    fmts = cfg["output"].get("formats", [])
    bad = [f for f in fmts if f not in ALL_FORMATS]
    if bad:
        raise ValueError(f"output.formats contains unknown formats {bad}; allowed: {ALL_FORMATS}")
    comps = cfg["components"]
    if not isinstance(comps.get("assign", {}), (dict, list)):
        raise ValueError("components.assign must be a mapping or a list")
