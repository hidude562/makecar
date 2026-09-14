"""Component framework.

A `CarComponent` attaches to a `Connector` and produces a mesh (plus optional
sub-connectors of its own, so that e.g. a dashboard can offer gauge-cluster and
vent connectors to further components).

Type matching uses the connector class hierarchy: a component that `accepts`
`PolygonConnector` also accepts `RectangleConnector` and `CircleConnector`.

`MorphableComponent` is the target-based flavour and uses *exactly* the same
machinery as the body: a fixed-topology parametric generator, a set of
`ModifierSpec`s turned into differential targets, and a `fit()` step that maps
the connector's geometry (radius, width, height, meta hints) onto modifier
values in [-1, 1].

Local frame convention for all components: build in the connector's local
frame where +Z is the mount normal (things grow along +Z), +X the connector's
major direction, origin at the connector origin.  `build()` transforms the
result into world space.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
import numpy as np

from ..geometry.mesh import Mesh, Material
from ..geometry.frame import Frame
from ..connectors import Connector, PointConnector, PolygonConnector, RectangleConnector, CircleConnector
from ..morph.morphable import MorphableMesh, DifferentialTargetBuilder, ModifierSpec
from ..morph.target import Target, Modifier


@dataclass
class Palette:
    """Colours shared across components (from the config)."""
    paint: str = "#8a1c1c"
    paint_secondary: str = "#f4f4f0"
    trim: str = "#141416"
    chrome: str = "#c9ccd1"
    glass: str = "#7fa3b8"
    interior: str = "#2a2724"
    interior_accent: str = "#6b5a48"
    seat: str = "#3a332d"
    rim: str = "#b8bcc2"
    light: str = "#e8ecef"
    taillight: str = "#b3121b"
    wood: str = "#5a3b22"
    carpet: str = "#1c1a18"
    extras: Dict[str, str] = field(default_factory=dict)

    def color(self, key: str) -> str:
        return self.extras.get(key, getattr(self, key, "#808080"))


@dataclass
class BuildContext:
    measurements: Dict[str, float]
    hints: Dict[str, Any]
    palette: Palette
    rng: np.random.Generator
    depth: int = 0

    def material(self, name: str, color=None, alpha=1.0, shininess=0.3, emissive=0.0, metallic=0.0) -> Material:
        c = self.palette.color(name) if color is None else color
        return Material(name, Material.parse_color(c), alpha, shininess, emissive, metallic)


@dataclass
class ComponentResult:
    mesh: Mesh
    connectors: List[Connector] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


class CarComponent:
    """Base class: procedural component bound to a connector type."""

    name: str = "component"
    accepts: Tuple[Type[Connector], ...] = (Connector,)
    default_for: Tuple[str, ...] = ()       # connector tags this is the default for
    priority: int = 0                       # higher wins among defaults for the same tag
    options: Dict[str, Any] = {}            # documented defaults
    description: str = ""

    @classmethod
    def can_attach(cls, connector: Connector) -> bool:
        return isinstance(connector, cls.accepts)

    def resolve_options(self, options: Optional[dict]) -> dict:
        o = dict(self.options)
        o.update(options or {})
        return o

    def build(self, connector: Connector, options: Optional[dict], ctx: BuildContext) -> ComponentResult:
        if not self.can_attach(connector):
            raise TypeError(f"{self.name} accepts {[t.__name__ for t in self.accepts]}, got {type(connector).__name__} ({connector.name})")
        opts = self.resolve_options(options)
        result = self.build_local(connector, opts, ctx)
        result.mesh.apply_frame(connector.frame)
        result.mesh.name = f"{connector.name}:{self.name}"
        for c in result.connectors:
            c.frame = c.frame.transformed(connector.frame.matrix)
            if isinstance(c, PolygonConnector):
                c._points = connector.frame.to_world(c._points)
            c.owner = connector.name
            c.name = f"{connector.name}/{c.name}"
        return result

    # subclasses implement this in the connector's local frame
    def build_local(self, connector: Connector, opts: dict, ctx: BuildContext) -> ComponentResult:
        raise NotImplementedError

    @classmethod
    def describe(cls) -> dict:
        return {
            "name": cls.name,
            "accepts": [t.__name__ for t in cls.accepts],
            "default_for": list(cls.default_for),
            "options": dict(cls.options),
            "description": cls.description,
            "morphable": issubclass(cls, MorphableComponent),
        }


class MorphableComponent(CarComponent):
    """Target-based component (same logic as the body).

    Subclasses define:
      params_cls        dataclass of generator parameters (canonical values)
      modifier_specs    list[ModifierSpec] -> differential targets
      generate(params)  fixed-topology mesh in the local frame
      fit(connector, opts, ctx) -> {modifier: value in [-1,1]}
      sub_connectors(connector, opts, ctx, values) -> local-frame connectors
    Variant generators (e.g. bench vs bucket seat) can be exposed as *macro*
    targets by overriding `macros()`.
    """

    params_cls: Type = None  # type: ignore
    modifier_specs: List[ModifierSpec] = []
    _libraries: Dict[str, MorphableMesh] = {}

    def canonical(self):
        return self.params_cls()

    def macros(self) -> Dict[str, Any]:
        return {}

    def generate(self, params) -> Mesh:
        raise NotImplementedError

    def fit(self, connector: Connector, opts: dict, ctx: BuildContext) -> Dict[str, float]:
        return {}

    def sub_connectors(self, connector: Connector, opts: dict, ctx: BuildContext, values: Dict[str, float]) -> List[Connector]:
        return []

    def library(self) -> MorphableMesh:
        key = f"{type(self).__module__}.{type(self).__name__}"
        lib = MorphableComponent._libraries.get(key)
        if lib is None:
            builder = DifferentialTargetBuilder(self.generate, self.canonical(), name=self.name)
            lib = builder.build(self.modifier_specs, macros=self.macros(), macro_group="variant")
            MorphableComponent._libraries[key] = lib
        return lib

    @staticmethod
    def ratio_to_value(actual: float, canonical: float, delta_minus: float, delta_plus: float) -> float:
        """Map a desired physical value onto a bipolar slider value (clamped)."""
        d = actual - canonical
        if d >= 0:
            return float(np.clip(d / delta_plus, 0, 1)) if delta_plus else 0.0
        return float(np.clip(d / delta_minus, -1, 0)) if delta_minus else 0.0

    def spec(self, name: str) -> ModifierSpec:
        for s in self.modifier_specs:
            if (s.name or s.param) == name:
                return s
        raise KeyError(name)

    def value_for(self, name: str, actual: float) -> float:
        s = self.spec(name)
        canon = getattr(self.canonical(), s.param)
        return self.ratio_to_value(actual, canon, s.delta_minus, s.delta_plus)

    def materials(self, ctx: BuildContext, opts: dict) -> Dict[str, Material]:
        return {}

    def build_local(self, connector: Connector, opts: dict, ctx: BuildContext) -> ComponentResult:
        values = self.fit(connector, opts, ctx)
        lib = self.library()
        for k in list(values):
            if k not in lib.modifiers:
                raise KeyError(f"{self.name}: fit() produced unknown modifier {k!r}")
        mesh = lib.morph(values)
        mesh.materials.update(self.materials(ctx, opts))
        subs = self.sub_connectors(connector, opts, ctx, values)
        return ComponentResult(mesh, subs, {"modifier_values": values})


# ------------------------------------------------------------------- registry
REGISTRY: Dict[str, Type[CarComponent]] = {}


def register(cls: Type[CarComponent]) -> Type[CarComponent]:
    if not cls.name or cls.name in REGISTRY:
        raise ValueError(f"component name missing or duplicate: {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


def get_component(name: str) -> CarComponent:
    if name not in REGISTRY:
        raise KeyError(f"unknown component {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def default_component_for(connector: Connector) -> Optional[str]:
    """Pick the default component for a connector by its tags (then by type)."""
    best: Tuple[int, str] | None = None
    for name, cls in REGISTRY.items():
        if not cls.can_attach(connector):
            continue
        for tag in cls.default_for:
            if tag in connector.tags or tag == connector.name:
                score = cls.priority * 10 + (5 if tag == connector.name else 0)
                if best is None or score > best[0]:
                    best = (score, name)
    return best[1] if best else None


def describe_all() -> List[dict]:
    return [cls.describe() for cls in REGISTRY.values()]
