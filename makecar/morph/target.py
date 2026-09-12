"""MakeHuman-style targets and modifiers.

A `Target` is a per-vertex displacement field for a specific base-mesh
topology.  A `Modifier` is the user-facing slider: a value in [-1, 1] (or
[0, 1]) that maps onto one or two targets ("decr"/"incr"), exactly like
MakeHuman's `something-decr|incr` modifiers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Target:
    name: str
    offsets: np.ndarray  # (N,3)
    description: str = ""

    def __post_init__(self):
        self.offsets = np.asarray(self.offsets, dtype=float)

    @classmethod
    def from_difference(cls, name: str, base_vertices: np.ndarray, variant_vertices: np.ndarray, description="") -> "Target":
        base_vertices = np.asarray(base_vertices)
        variant_vertices = np.asarray(variant_vertices)
        if base_vertices.shape != variant_vertices.shape:
            raise ValueError(f"target {name}: topology mismatch {base_vertices.shape} vs {variant_vertices.shape}")
        return cls(name, variant_vertices - base_vertices, description)

    @property
    def n_vertices(self) -> int:
        return len(self.offsets)

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.offsets, axis=1).max()) if len(self.offsets) else 0.0

    def scaled(self, w: float) -> np.ndarray:
        return self.offsets * w

    def nonzero_count(self, eps=1e-9) -> int:
        return int((np.linalg.norm(self.offsets, axis=1) > eps).sum())

    # MakeHuman-compatible sparse text format: "index dx dy dz" per line
    def save(self, path) -> None:
        path = Path(path)
        idx = np.where(np.linalg.norm(self.offsets, axis=1) > 1e-9)[0]
        with path.open("w") as fh:
            fh.write(f"# makecar target {self.name}: {self.description}\n")
            for i in idx:
                dx, dy, dz = self.offsets[i]
                fh.write(f"{i} {dx:.6f} {dy:.6f} {dz:.6f}\n")

    @classmethod
    def load(cls, path, n_vertices: int, name: Optional[str] = None) -> "Target":
        path = Path(path)
        off = np.zeros((n_vertices, 3))
        with path.open() as fh:
            for line in fh:
                if not line.strip() or line.startswith("#"):
                    continue
                i, dx, dy, dz = line.split()
                off[int(i)] = (float(dx), float(dy), float(dz))
        return cls(name or path.stem, off)


@dataclass
class Modifier:
    """A slider mapping onto targets.

    * bipolar (min=-1, max=1): value>0 blends `incr`, value<0 blends `decr`.
    * unipolar (min=0, max=1): value blends `incr` only (macro/style targets).
    """

    name: str
    incr: Optional[Target] = None
    decr: Optional[Target] = None
    min_value: float = -1.0
    max_value: float = 1.0
    default: float = 0.0
    group: str = "misc"
    description: str = ""
    unit_scale: Optional[float] = None  # physical change (m) of incr at value 1, if known

    def clamp(self, value: float) -> float:
        return float(min(self.max_value, max(self.min_value, value)))

    def weights(self, value: float) -> List[Tuple[Target, float]]:
        v = self.clamp(value)
        out: List[Tuple[Target, float]] = []
        if v > 0 and self.incr is not None:
            out.append((self.incr, v))
        elif v < 0:
            if self.decr is not None:
                out.append((self.decr, -v))
            elif self.incr is not None:
                out.append((self.incr, v))  # mirror when only one side exists
        return out

    @property
    def is_unipolar(self) -> bool:
        return self.min_value >= 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "group": self.group,
            "min": self.min_value,
            "max": self.max_value,
            "default": self.default,
            "description": self.description,
            "targets": [t.name for t in (self.incr, self.decr) if t is not None],
        }
