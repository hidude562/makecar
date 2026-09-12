"""Right-handed coordinate frames used to place connectors and components.

Car convention (world): +X forward, +Y left, +Z up, ground at Z = 0.

A Frame has an origin and three orthonormal axes.  By convention the frame's
Z axis is the *mount normal* of a connector (the direction things stick out
along) and its X axis is the connector's "major" direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

_EPS = 1e-9


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < _EPS:
        raise ValueError("cannot normalise a zero vector")
    return v / n


@dataclass
class Frame:
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    x_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    y_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    z_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        self.x_axis = unit(self.x_axis)
        self.y_axis = unit(self.y_axis)
        self.z_axis = unit(self.z_axis)

    # ------------------------------------------------------------------ builders
    @classmethod
    def identity(cls, origin=(0, 0, 0)) -> "Frame":
        return cls(np.asarray(origin, dtype=float))

    @classmethod
    def from_normal(cls, origin, normal, x_hint=(1.0, 0.0, 0.0)) -> "Frame":
        """Frame whose Z axis is `normal`; X is `x_hint` projected onto the plane."""
        z = unit(normal)
        hint = np.asarray(x_hint, dtype=float)
        x = hint - np.dot(hint, z) * z
        if np.linalg.norm(x) < 1e-6:
            # hint parallel to the normal: pick any perpendicular direction
            alt = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            x = alt - np.dot(alt, z) * z
        x = unit(x)
        y = np.cross(z, x)
        return cls(np.asarray(origin, dtype=float), x, y, z)

    @classmethod
    def from_axes(cls, origin, x_axis, z_axis) -> "Frame":
        z = unit(z_axis)
        x = np.asarray(x_axis, dtype=float)
        x = unit(x - np.dot(x, z) * z)
        return cls(np.asarray(origin, dtype=float), x, np.cross(z, x), z)

    # -------------------------------------------------------------- conversions
    @property
    def rotation(self) -> np.ndarray:
        """3x3 matrix whose columns are the axes (local -> world rotation)."""
        return np.stack([self.x_axis, self.y_axis, self.z_axis], axis=1)

    @property
    def matrix(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.origin
        return m

    def to_world(self, pts) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        return p @ self.rotation.T + self.origin

    def to_local(self, pts) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        return (p - self.origin) @ self.rotation

    def dir_to_world(self, v) -> np.ndarray:
        return np.atleast_2d(np.asarray(v, dtype=float)) @ self.rotation.T

    def transformed(self, m4: np.ndarray) -> "Frame":
        r = m4[:3, :3]
        return Frame(r @ self.origin + m4[:3, 3], r @ self.x_axis, r @ self.y_axis, r @ self.z_axis)

    def translated(self, d) -> "Frame":
        return Frame(self.origin + np.asarray(d, dtype=float), self.x_axis, self.y_axis, self.z_axis)

    def offset_local(self, dx=0.0, dy=0.0, dz=0.0) -> "Frame":
        return self.translated(dx * self.x_axis + dy * self.y_axis + dz * self.z_axis)

    def rotated_about_z(self, angle: float) -> "Frame":
        c, s = np.cos(angle), np.sin(angle)
        x = c * self.x_axis + s * self.y_axis
        return Frame(self.origin.copy(), x, np.cross(self.z_axis, x), self.z_axis.copy())

    def rotated_about_x(self, angle: float) -> "Frame":
        c, s = np.cos(angle), np.sin(angle)
        z = c * self.z_axis + s * self.y_axis
        return Frame(self.origin.copy(), self.x_axis.copy(), np.cross(z, self.x_axis), z)

    def rotated_about_y(self, angle: float) -> "Frame":
        c, s = np.cos(angle), np.sin(angle)
        z = c * self.z_axis - s * self.x_axis
        x = np.cross(self.y_axis, z)
        return Frame(self.origin.copy(), x, self.y_axis.copy(), z)

    def mirrored_y(self) -> "Frame":
        """Mirror across the car's centre plane (Y -> -Y) keeping right-handedness."""
        m = np.diag([1.0, -1.0, 1.0])
        o = m @ self.origin
        x = m @ self.x_axis
        z = m @ self.z_axis
        return Frame(o, x, np.cross(z, x), z)

    def to_dict(self) -> dict:
        return {
            "origin": [round(float(v), 5) for v in self.origin],
            "x_axis": [round(float(v), 5) for v in self.x_axis],
            "y_axis": [round(float(v), 5) for v in self.y_axis],
            "z_axis": [round(float(v), 5) for v in self.z_axis],
        }

    def __repr__(self) -> str:  # pragma: no cover
        o = ", ".join(f"{v:.3f}" for v in self.origin)
        z = ", ".join(f"{v:.2f}" for v in self.z_axis)
        return f"Frame(origin=({o}), normal=({z}))"


def rotation_matrix(axis, angle: float) -> np.ndarray:
    """4x4 rotation about an arbitrary axis through the origin."""
    a = unit(axis)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = a
    r = np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )
    m = np.eye(4)
    m[:3, :3] = r
    return m


def translation_matrix(d) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = np.asarray(d, dtype=float)
    return m


def scale_matrix(sx, sy=None, sz=None) -> np.ndarray:
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    return np.diag([sx, sy, sz, 1.0])
