"""Assembly: attach components to connectors according to config rules,
recursively (components may emit connectors of their own)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from .geometry.mesh import Mesh
from .connectors import Connector
from .body.body import BodyResult
from .components import (REGISTRY, get_component, default_component_for, BuildContext, Palette, ComponentResult)


@dataclass
class AssignRule:
    match: str                      # connector selector: exact name, tag, or glob
    component: Optional[str] = None  # None -> use default component
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, key: str, value) -> "AssignRule":
        if value is None or value is True:
            return cls(key)
        if isinstance(value, str):
            return cls(key, value)
        if isinstance(value, dict):
            return cls(value.get("match", key), value.get("component"), dict(value.get("options", {})))
        raise ValueError(f"bad component assignment for {key!r}: {value!r}")


@dataclass
class ComponentInstance:
    id: str
    connector: Connector
    component: str
    options: Dict[str, Any]
    result: ComponentResult
    parent: Optional[str]
    children: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "connector": self.connector.name,
            "connector_kind": self.connector.kind,
            "component": self.component,
            "options": self.options,
            "parent": self.parent,
            "children": list(self.children),
            "faces": self.result.mesh.n_faces,
            "info": {k: _jsonable(v) for k, v in self.result.info.items()},
        }


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return np.round(v, 5).tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


@dataclass
class CarAssembly:
    body: BodyResult
    instances: List[ComponentInstance]
    connectors: List[Connector]          # every connector, including sub-connectors
    unattached: List[str]
    disabled: List[str]
    palette: Palette

    @property
    def body_mesh(self) -> Mesh:
        return self.body.mesh

    def components_mesh(self) -> Mesh:
        m = Mesh(name="components")
        for inst in self.instances:
            m.merge(inst.result.mesh, group_prefix=inst.id)
        return m

    def mesh(self) -> Mesh:
        """Full car: body shell + every component (world coordinates)."""
        m = self.body.mesh.copy()
        m.name = "car"
        for inst in self.instances:
            m.merge(inst.result.mesh, group_prefix=inst.id)
        return m

    def object_groups(self) -> Dict[int, str]:
        """face index -> object group name for OBJ export (body faces first)."""
        groups: Dict[int, str] = {}
        n = self.body.mesh.n_faces
        for i in range(n):
            groups[i] = "body"
        for inst in self.instances:
            for k in range(inst.result.mesh.n_faces):
                groups[n + k] = inst.id
            n += inst.result.mesh.n_faces
        return groups

    def interior_mesh(self) -> Mesh:
        m = Mesh(name="interior")
        for inst in self.instances:
            if _is_interior(inst.connector):
                m.merge(inst.result.mesh, group_prefix=inst.id)
        return m

    def exterior_mesh(self) -> Mesh:
        """Body shell (with glass apertures as faces) + exterior components only."""
        m = self.body.full_mesh.copy()
        m.name = "exterior"
        for inst in self.instances:
            if not _is_interior(inst.connector) and "glass" not in inst.connector.tags:
                m.merge(inst.result.mesh, group_prefix=inst.id)
        return m

    def connector_component(self, conn_name: str) -> Optional[str]:
        for inst in self.instances:
            if inst.connector.name == conn_name:
                return inst.component
        return None

    def to_dict(self) -> dict:
        return {
            "measurements": {k: round(float(v), 5) for k, v in self.body.measurements.items()},
            "modifier_values": self.body.modifier_values,
            "hints": _jsonable(self.body.hints),
            "components": [i.to_dict() for i in self.instances],
            "unattached_connectors": self.unattached,
            "disabled_connectors": self.disabled,
            "stats": {
                "body_faces": self.body.mesh.n_faces,
                "component_faces": sum(i.result.mesh.n_faces for i in self.instances),
                "connectors": len(self.connectors),
                "components": len(self.instances),
            },
        }

    def connectors_dict(self) -> list:
        out = []
        by_name = {i.connector.name: i for i in self.instances}
        for c in self.connectors:
            d = c.to_dict()
            inst = by_name.get(c.name)
            d["component"] = inst.component if inst else None
            out.append(d)
        return out


INTERIOR_TAGS = {"seat", "steering_wheel", "dashboard", "pedals", "console", "floor", "bulkhead", "shelf", "cargo_floor",
                 "headliner", "door_card", "rearview_mirror", "cluster", "screen", "vent", "glovebox", "hvac", "shifter",
                 "cupholder", "dome_light", "grab_handle", "speaker"}


def _is_interior(c: Connector) -> bool:
    return bool(c.tags & INTERIOR_TAGS) or "/" in c.name and any(t in c.tags for t in INTERIOR_TAGS)


def assemble(body: BodyResult, components_cfg: Optional[dict] = None, palette: Optional[Palette] = None,
             seed: int = 0, max_depth: int = 4) -> CarAssembly:
    cfg = dict(components_cfg or {})
    use_defaults = bool(cfg.get("defaults", True))
    disable = list(cfg.get("disable", []))
    rules: List[AssignRule] = []
    assign = cfg.get("assign", {})
    if isinstance(assign, dict):
        rules = [AssignRule.parse(k, v) for k, v in assign.items()]
    else:
        for item in assign:
            if isinstance(item, dict) and "match" in item:
                rules.append(AssignRule(item["match"], item.get("component"), dict(item.get("options", {}))))
            else:
                raise ValueError(f"bad assign entry: {item!r}")
    palette = palette or Palette()
    rng = np.random.default_rng(seed)
    ctx = BuildContext(body.measurements, body.hints, palette, rng)

    instances: List[ComponentInstance] = []
    all_connectors: List[Connector] = []
    unattached: List[str] = []
    disabled: List[str] = []
    queue: List[tuple] = [(c, None, 0) for c in body.connectors]
    while queue:
        conn, parent, depth = queue.pop(0)
        all_connectors.append(conn)
        if any(conn.matches(sel) for sel in disable):
            disabled.append(conn.name)
            continue
        comp_name, options = None, {}
        for rule in rules:  # later rules override earlier ones
            if conn.matches(rule.match):
                comp_name = rule.component or comp_name
                options = {**options, **rule.options}
        if comp_name is None and use_defaults:
            comp_name = default_component_for(conn)
        if comp_name is None:
            unattached.append(conn.name)
            continue
        comp = get_component(comp_name)
        if not comp.can_attach(conn):
            raise TypeError(f"component {comp_name!r} cannot attach to {conn.kind} connector {conn.name!r} "
                            f"(accepts {[t.__name__ for t in comp.accepts]})")
        ctx.depth = depth
        result = comp.build(conn, options, ctx)
        inst = ComponentInstance(f"{conn.name}:{comp_name}", conn, comp_name, options, result, parent)
        instances.append(inst)
        if parent is not None:
            for p in instances:
                if p.id == parent:
                    p.children.append(inst.id)
        if depth < max_depth:
            for sub in result.connectors:
                queue.append((sub, inst.id, depth + 1))
    return CarAssembly(body, instances, all_connectors, unattached, disabled, palette)
