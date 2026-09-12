"""Shared fixtures for the makecar test-suite.

The body target library is expensive on a cold cache (~4 s) and cheap after
that, so the sedan `CarBody` and the two `BodyResult`s used across modules are
built once per session.  Tests must treat these fixtures as read-only: copy a
mesh before mutating it.
"""
from __future__ import annotations

import numpy as np
import pytest

from makecar.body import CarBody
from makecar.components import BuildContext, Palette


# ----------------------------------------------------------------- helpers
def naive_face_normals(mesh) -> np.ndarray:
    """Per-face Newell normals computed one face at a time (reference impl)."""
    out = np.zeros((mesh.n_faces, 3))
    for i, f in enumerate(mesh.faces):
        n = np.zeros(3)
        for k in range(len(f)):
            a = mesh.vertices[f[k]]
            b = mesh.vertices[f[(k + 1) % len(f)]]
            n += np.cross(a, b)
        length = np.linalg.norm(n)
        out[i] = n / length if length > 1e-12 else 0.0
    return out


def signed_volume(mesh) -> float:
    """Signed volume via the divergence theorem; positive for a closed,
    outward-facing mesh."""
    tris, _ = mesh.triangulated()
    a, b, c = (mesh.vertices[tris[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def is_right_handed(frame) -> bool:
    r = frame.rotation
    return bool(np.allclose(r.T @ r, np.eye(3), atol=1e-9) and np.linalg.det(r) > 0)


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def car_body() -> CarBody:
    return CarBody()


@pytest.fixture(scope="session")
def sedan(car_body):
    return car_body.build()


@pytest.fixture(scope="session")
def pickup(car_body):
    return car_body.build(style="pickup")


@pytest.fixture
def ctx(sedan) -> BuildContext:
    return BuildContext(dict(sedan.measurements), dict(sedan.hints), Palette(), np.random.default_rng(0))
