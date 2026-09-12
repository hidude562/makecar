"""CarBody: the MakeHuman-style morphable car body + connector emission."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from ..geometry.mesh import Mesh, Material
from ..connectors import Connector
from .params import BodyParams
from .styles import STYLE_OVERRIDES, style_params, style_names
from .targets import body_library
from .connectors import measure, feature_lines, emit_connectors


@dataclass
class BodyResult:
    mesh: Mesh                      # shell with apertures removed (paint/underbody/trim)
    full_mesh: Mesh                 # shell including aperture faces (for blueprints)
    connectors: List[Connector]
    measurements: Dict[str, float]
    modifier_values: Dict[str, float]
    base_params: BodyParams
    hints: Dict

    def connector(self, name: str) -> Connector:
        for c in self.connectors:
            if c.name == name:
                return c
        raise KeyError(name)


class CarBody:
    """Build bodies from style macros + modifier sliders (+ optional base overrides)."""

    def __init__(self, base_params: Optional[BodyParams] = None):
        self.base_params = base_params or BodyParams()
        self.library = body_library(self.base_params)

    # ------------------------------------------------------------ queries
    def modifiers(self) -> List[dict]:
        return self.library.describe()

    @staticmethod
    def styles() -> List[str]:
        return style_names()

    # -------------------------------------------------------------- build
    def resolve_values(self, style=None, modifiers: Optional[Dict[str, float]] = None,
                       sculpt: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        values: Dict[str, float] = {}
        if style:
            if isinstance(style, str):
                style = {style: 1.0}
            for name, w in style.items():
                if name == "sedan":
                    continue
                if f"style/{name}" not in self.library.modifiers:
                    raise KeyError(f"unknown style {name!r}; choose from {style_names()}")
                values[f"style/{name}"] = float(w)
        for name, v in (modifiers or {}).items():
            key = name if name in self.library.modifiers else f"shape/{name}"
            if key not in self.library.modifiers:
                raise KeyError(f"unknown modifier {name!r}. Available: {sorted(self.library.modifiers)}")
            values[key] = float(v)
        for name, v in (sculpt or {}).items():
            key = name if name.startswith("sculpt/") else f"sculpt/{name}"
            if key not in self.library.modifiers:
                raise KeyError(f"unknown sculpt target {name!r}")
            values[key] = float(v)
        return values

    def build(self, style=None, modifiers=None, sculpt=None, hints: Optional[Dict] = None,
              paint="#8a1c1c") -> BodyResult:
        values = self.resolve_values(style, modifiers, sculpt)
        mesh = self.library.morph(values)
        hints = dict(hints or {})
        # style hints: dominant style name, door count, tyre width from the effective params
        dominant = "sedan"
        if style:
            sd = style if isinstance(style, dict) else {style: 1.0}
            dominant = max(sd.items(), key=lambda kv: kv[1])[0]
        eff = style_params(dominant, self.base_params) if dominant in STYLE_OVERRIDES else self.base_params
        hints.setdefault("style", dominant)
        hints.setdefault("door_count", eff.door_count)
        hints.setdefault("tire_width", eff.tire_width + 0.06 * values.get("tire_width", 0.0))
        hints.setdefault("arch_gap", eff.arch_gap)
        meas = measure(mesh)
        for name, pts in feature_lines(mesh, meas, hints["door_count"]):
            mesh.add_line(pts, name)
        connectors = emit_connectors(mesh, meas, hints)
        # materials
        paint_mat = Material("paint", Material.parse_color(paint), 1.0, 0.85)
        mesh.materials.update({
            "paint": paint_mat,
            "aperture": Material("aperture", (0.4, 0.55, 0.65), 0.45, 0.9),
            "underbody": Material("underbody", (0.10, 0.10, 0.11), 1.0, 0.05),
            "trim": Material("trim", (0.09, 0.09, 0.10), 1.0, 0.25),
            "bed": Material("bed", (0.14, 0.14, 0.15), 1.0, 0.1),
        })
        full = mesh.copy()
        shell = mesh.copy()
        drop = np.concatenate([shell.zones[z] for z in shell.zones if z.startswith("aperture/")]) if shell.zones else []
        shell.remove_faces(drop)
        return BodyResult(shell, full, connectors, meas, values, self.base_params, hints)
