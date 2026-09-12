#!/usr/bin/env python3
"""Body + exhaust geometry audit; never assembles or renders a complete car.

Examples (run from any working directory):
  python3 audit_exhaust.py --baseline 2a4a5e7 --output audit_before.json
  python3 audit_exhaust.py --source-root /path/to/checkout --output audit_after.json
  python3 audit_exhaust.py --source-root /path/to/snapshot --cases 135 522

The source package is frozen in a temporary directory before importing it.
Outputs contain compact case configurations/minima, source hashes, and worst
sample locations, not meshes or full per-vertex dumps. All dimensions are mm
unless explicitly suffixed otherwise. Negative clearance means penetration.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np


class Envelope:
    """Actual exported triangle lower-Z envelope, accelerated only by XY bounds."""

    def __init__(self, mesh):
        triangles, _ = mesh.triangulated()
        tri = mesh.vertices[triangles]
        a, b, c = tri.transpose(1, 0, 2)
        ab, ac = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        good = abs(det) > 1e-12
        self.a, self.ab, self.ac, self.det = a[good], ab[good], ac[good], det[good]
        self.dz1, self.dz2 = (b[:, 2] - a[:, 2])[good], (c[:, 2] - a[:, 2])[good]
        self.lo, self.hi = tri.min(axis=1)[good, :2], tri.max(axis=1)[good, :2]

    def heights(self, points):
        p = np.asarray(points)
        result = np.full(len(p), np.inf)
        bins = np.floor(p[:, 0] / .08).astype(int)
        for key in np.unique(bins):
            indices = np.flatnonzero(bins == key)
            ps = p[indices]
            mask = ((self.hi[:, 0] >= ps[:, 0].min() - 1e-9)
                    & (self.lo[:, 0] <= ps[:, 0].max() + 1e-9)
                    & (self.hi[:, 1] >= ps[:, 1].min() - 1e-9)
                    & (self.lo[:, 1] <= ps[:, 1].max() + 1e-9))
            if not mask.any():
                continue
            a, ab, ac, det = self.a[mask], self.ab[mask], self.ac[mask], self.det[mask]
            for start in range(0, len(indices), 128):
                q = ps[start:start + 128, None, :2] - a[None, :, :2]
                u = (q[:, :, 0] * ac[None, :, 1] - q[:, :, 1] * ac[None, :, 0]) / det
                v = (ab[None, :, 0] * q[:, :, 1] - ab[None, :, 1] * q[:, :, 0]) / det
                valid = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
                h = a[None, :, 2] + u * self.dz1[mask] + v * self.dz2[mask]
                result[indices[start:start + 128]] = np.where(valid, h, np.inf).min(axis=1)
        return result


def slow_heights(mesh, points):
    """Unaccelerated test oracle, matching tests' underbody_clearance equations."""
    triangles, _ = mesh.triangulated()
    a, b, c = mesh.vertices[triangles].transpose(1, 0, 2)
    ab, ac = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    den = np.where(abs(det) > 1e-12, det, 1.)
    result = []
    for p in points:
        q = p[:2] - a[:, :2]
        u = (q[:, 0] * ac[:, 1] - q[:, 1] * ac[:, 0]) / den
        v = (ab[:, 0] * q[:, 1] - ab[:, 1] * q[:, 0]) / den
        valid = (abs(det) > 1e-12) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        h = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
        result.append(h[valid].min() if valid.any() else np.inf)
    return np.asarray(result)


def metric(envelope, points):
    points = np.asarray(points)
    h = envelope.heights(points)
    gap = h - points[:, 2]
    i = int(np.argmin(gap))
    return {"clearance": float(gap[i] * 1000), "point_m": points[i].tolist(),
            "road": float(points[:, 2].min() * 1000), "missing": int(np.isinf(h).sum())}


def sweep_metrics(mesh, name, path):
    """Check actual longitudinal edges and outward exported side triangles.

    Forward progress is dot(next_vertex - vertex, unit centreline segment).
    Outward cosine compares the exported triangle normal with its vertices'
    mean radial vector from their own ring centres. Caps are excluded. These
    catch local folds; they are not a general nonadjacent self-intersection test.
    """
    ids = np.asarray(mesh.groups[name])
    rings = mesh.vertices[ids[:-2]].reshape(len(path), 10, 3)
    segments = np.diff(path, axis=0)
    tangent = segments / np.maximum(np.linalg.norm(segments, axis=1, keepdims=True), 1e-15)
    progress = np.einsum("nsi,ni->ns", np.diff(rings, axis=0), tangent)
    worst_edge = tuple(map(int, np.unravel_index(np.argmin(progress), progress.shape)))
    quads = [mesh.faces[i] for i in mesh.zones[name] if len(mesh.faces[i]) == 4]
    triangles = np.asarray([(f[0], f[k], f[k + 1]) for f in quads for k in (1, 2)])
    ring_id = np.full(mesh.n_vertices, -1, dtype=int)
    ring_id[ids[:-2]] = np.repeat(np.arange(len(path)), 10)
    tri = mesh.vertices[triangles]
    radial = (tri - path[ring_id[triangles]]).mean(axis=1)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(normal, axis=1)
    cosine = np.einsum("ni,ni->n", normal, radial) / np.maximum(
        area2 * np.linalg.norm(radial, axis=1), 1e-30)
    i = int(np.argmin(cosine))
    return {"forward_mm": float(progress.min() * 1000), "forward_edge": list(worst_edge),
            "outward_cosine": float(cosine[i]), "side_triangle": i,
            "nonforward_edges": int((progress <= 0).sum()),
            "nonoutward_triangles": int((cosine <= 0).sum()),
            "degenerate_triangles": int((area2 <= 2e-12).sum())}


def case_matrix(styles, shape_names):
    zero = dict.fromkeys(shape_names, 0)
    mixes = {
        "wedge_pointed_tumblehome": {"face/wedge": 1, "plan/pointed": 1, "section/tumblehome": 1},
        "shark_cokebottle_domed": {"face/shark": 1, "plan/cokebottle": 1, "section/domed": 1},
        "upright_square_slabside": {"face/upright": 1, "plan/square": 1, "section/slabside": 1},
        "cabforward_pointed_slabside": {"face/cabforward": 1, "plan/pointed": 1, "section/slabside": 1},
        "longhood_cokebottle_tumblehome": {"face/longhood": 1, "plan/cokebottle": 1, "section/tumblehome": 1},
        "snub_square_domed": {"face/snub": 1, "plan/square": 1, "section/domed": 1},
        "opposed_faces": {"face/wedge": 1, "face/upright": 1, "plan/pointed": 1,
                          "plan/square": 1, "section/domed": 1},
        "longhood_shark": {"face/longhood": 1, "face/shark": 1, "plan/cokebottle": 1, "section/tumblehome": 1},
    }
    for style in styles:
        yield "default", style, {}
        for name in shape_names:
            for weight in (0, 1):
                yield "inherited_override", style, {name: weight}
        yield "all_shapes_zero", style, zero
        for name in shape_names:
            yield "isolated_endpoint", style, {**zero, name: 1}
        for name, mix in mixes.items():
            yield "mix_" + name, style, {**zero, **mix}
        for gc, th, tw in itertools.product((-1, 1), repeat=3):
            packaging = {"ground_clearance": gc, "tunnel_height": th, "tunnel_width": tw}
            yield "packaging_default", style, packaging
            for name in ("wedge_pointed_tumblehome", "shark_cokebottle_domed", "cabforward_pointed_slabside"):
                yield "packaging_" + name, style, {**zero, **mixes[name], **packaging}
    yield "legacy_mixed_style", {"coupe": .6, "wagon": .4}, {
        "wheelbase": .7, "rear_fascia_rake": -.8, "ground_clearance": -.4, "tunnel_height": -.7}
    yield "legacy_low_sports", "sports", {"ground_clearance": -1}
    yield "legacy_tall_narrow_sedan", "sedan", {"tunnel_height": 1, "tunnel_width": -1}


def audit(body, api, validate=False):
    BuildContext, Palette, get_component = api
    conn = body.connector("exhaust_system")
    ctx = BuildContext(body.measurements, body.hints, Palette(), np.random.default_rng(0))
    mesh = get_component("exhaust.system").build(conn, {}, ctx).mesh
    pan_mesh = body.full_mesh.subset(body.full_mesh.zones["underbody"])
    pan, shell = Envelope(pan_mesh), Envelope(body.full_mesh)
    if validate:
        for source, envelope in ((pan_mesh, pan), (body.full_mesh, shell)):
            points = mesh.vertices[::7]
            np.testing.assert_allclose(envelope.heights(points), slow_heights(source, points), atol=1e-12, rtol=0)
    metrics, sweeps = {}, {}
    paths = [np.asarray(p) for p in conn.meta["paths"]]
    for name in ("pipe_run_0", "catalyst", "silencer"):
        part = mesh.subset(mesh.zones[name])
        edges = np.array([(a, b) for face in part.faces for a, b in zip(face, face[1:] + face[:1])])
        points = np.vstack([part.vertices, part.vertices[edges].mean(axis=1)])
        metrics[name] = metric(pan, points)
        metrics[name]["height"] = float(np.ptp(part.vertices[:, 2]) * 1000)
    for k, path in enumerate(paths):
        name = f"pipe_run_{k}"
        sweeps[name] = sweep_metrics(mesh, name, path)
        if not k:
            continue
        rings = mesh.vertices[mesh.groups[name]][:-2].reshape(len(path), 10, 3)
        last = len(path) - 3
        points = np.vstack([rings[:last + 1].reshape(-1, 3),
                            ((rings[:last] + rings[1:last + 1]) / 2).reshape(-1, 3)])
        metrics[f"branch_{k}"] = metric(pan, points)
        points = np.vstack([rings[last:].reshape(-1, 3),
                            ((rings[last:-1] + rings[last + 1:]) / 2).reshape(-1, 3)])
        metrics[f"terminal_{k}"] = metric(shell, points)
        points = np.vstack([rings.reshape(-1, 3), ((rings[:-1] + rings[1:]) / 2).reshape(-1, 3)])
        metrics[f"whole_branch_{k}"] = metric(shell, points)
    for dual, length in ((False, .08), (True, .08), (True, .14)):
        for side in ("L", "R"):
            tip = get_component("exhaust.tip").build(body.connector("exhaust_" + side),
                    {"dual": dual, "length": length}, ctx).mesh
            metrics[f"tip_{side}_{dual}_{length}"] = metric(shell, tip.vertices)
    floor = body.full_mesh.vertices[body.full_mesh.groups["line/floor"]]
    crown_error = paths[0][:, 2] - (np.interp(paths[0][:, 0], floor[:, 0], floor[:, 2]) - .033)
    grid = np.asarray(conn.meta["floor_grid"])
    package = {"pipe_road": float(mesh.vertices[:, 2].min() * 1000),
               "can_road": min(metrics[n]["road"] for n in ("catalyst", "silencer")),
               "can_height": min(metrics[n]["height"] for n in ("catalyst", "silencer")),
               "flat_pan": float(grid[:, -1, 2].min() * 1000),
               "grid_crown": float(grid[:, 0, 2].min() * 1000),
               "crown_offset_deviation": float(abs(crown_error).max() * 1000)}
    return metrics, sweeps, package


GROUPS = {"main": ("pipe_run_0",), "cans": ("catalyst", "silencer"),
          "branches": ("branch_1", "branch_2"), "terminal": ("terminal_1", "terminal_2"),
          "whole_branch": ("whole_branch_1", "whole_branch_2")}


def compact_record(index, category, style, modifiers, shape_names, metrics, sweeps, package):
    isolated = all(name in modifiers for name in shape_names)
    mods = {k: v for k, v in modifiers.items() if not (isolated and k in shape_names and v == 0)}
    record = {"id": index, "category": category, "style": style, "modifiers": mods}
    if isolated:
        record["zero_other_shapes"] = True
    record["clearance_mm"] = {g: min(metrics[n]["clearance"] for n in ns) for g, ns in GROUPS.items()}
    record["clearance_mm"]["tips"] = min(v["clearance"] for n, v in metrics.items() if n.startswith("tip_"))
    record["packaging_mm"] = package
    record["sweep"] = {"forward_mm": min(v["forward_mm"] for v in sweeps.values()),
                       "outward_cosine": min(v["outward_cosine"] for v in sweeps.values()),
                       "nonforward_edges": sum(v["nonforward_edges"] for v in sweeps.values()),
                       "nonoutward_triangles": sum(v["nonoutward_triangles"] for v in sweeps.values()),
                       "degenerate_triangles": sum(v["degenerate_triangles"] for v in sweeps.values())}
    record["missing_pan_samples"] = sum(metrics[n]["missing"] for g in ("main", "cans", "branches") for n in GROUPS[g])
    return record


def update_worst(worst, case_id, metrics, sweeps):
    groups = {**GROUPS, "tips": tuple(n for n in metrics if n.startswith("tip_"))}
    for group, names in groups.items():
        name = min(names, key=lambda n: metrics[n]["clearance"])
        if group not in worst or metrics[name]["clearance"] < worst[group]["clearance"]:
            worst[group] = {"case": case_id, "part": name, **metrics[name]}
    for key in ("forward_mm", "outward_cosine"):
        name = min(sweeps, key=lambda n: sweeps[n][key])
        if key not in worst or sweeps[name][key] < worst[key][key]:
            worst[key] = {"case": case_id, "part": name, **sweeps[name]}


def summary(records):
    good = [r for r in records if "error" not in r]
    result = {"completed": len(records), "build_errors": [r["id"] for r in records if "error" in r]}
    result["clearance_failures"] = {group: [r["id"] for r in good if r["clearance_mm"][group] < threshold - 1e-5]
            for group, threshold in [(g, 4.) for g in GROUPS] + [("tips", 1.)]}
    result["other_failures"] = {
        "missing_pan": [r["id"] for r in good if r["missing_pan_samples"]],
        "can_road": [r["id"] for r in good if r["packaging_mm"]["can_road"] < 50 - 1e-5],
        "can_collapse": [r["id"] for r in good if r["packaging_mm"]["can_height"] <= 20],
        "below_road": [r["id"] for r in good if r["packaging_mm"]["pipe_road"] < 0],
        "nonforward_edges": [r["id"] for r in good if r["sweep"]["nonforward_edges"]],
        "nonoutward_triangles": [r["id"] for r in good if r["sweep"]["nonoutward_triangles"]],
        "degenerate_triangles": [r["id"] for r in good if r["sweep"]["degenerate_triangles"]],
    }
    result["packaging_minima"] = {key: {"case": (r := min(good, key=lambda r: r["packaging_mm"][key]))["id"],
                                        "mm": r["packaging_mm"][key]}
                                  for key in ("pipe_road", "can_road", "can_height", "flat_pan", "grid_crown")} if good else {}
    return result


def clean(value):
    if isinstance(value, float):
        return round(value, 8) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def save(path, metadata, records, worst, start):
    payload = {**metadata, "elapsed_seconds": time.time() - start,
               "summary": summary(records), "worst": worst}
    # One compact row per case keeps evidence readable without a multi-MB dump.
    header = json.dumps(clean(payload), indent=2, allow_nan=False)
    rows = ",\n".join("    " + json.dumps(clean(r), separators=(",", ":"), allow_nan=False) for r in records)
    path.write_text(header[:-2] + ',\n  "cases": [\n' + rows + '\n  ]\n}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--baseline", help="git revision to archive from source-root instead of its current package")
    parser.add_argument("--output", type=Path, default=Path("audit_exhaust.json"))
    parser.add_argument("--cases", type=int, nargs="+", help="reproduce selected stable case IDs")
    parser.add_argument("--limit", type=int, help="smoke-test the first N cases")
    args = parser.parse_args()
    source = args.source_root.resolve()
    start = time.time()
    with tempfile.TemporaryDirectory(prefix="makecar-exhaust-audit-") as directory:
        frozen = Path(directory)
        revision = None
        if args.baseline:
            revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", args.baseline], text=True).strip()
            archive = subprocess.check_output(["git", "-C", str(source), "archive", revision, "makecar"])
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(frozen, filter="data")
        else:
            shutil.copytree(source / "makecar", frozen / "makecar", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        hashes = {str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((frozen / "makecar").rglob("*.py"))}
        package_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        os.environ["MAKECAR_CACHE"] = str(frozen / "cache")
        sys.path.insert(0, str(frozen))
        from makecar.body import CarBody, style_names
        from makecar.body.shapes import SHAPE_DELTAS
        from makecar.components import BuildContext, Palette, get_component
        cb = CarBody()
        shapes = list(SHAPE_DELTAS)
        cases = list(case_matrix(style_names(), shapes))
        metadata = {"source_root": str(source), "baseline_commit": revision,
                    "source_sha256": hashes, "package_sha256": package_hash,
                    "matrix_cases": len(cases), "shape_names": shapes,
                    "method": "Actual exported triangle lower-Z envelope; original exhaust-test vertex/edge-midpoint samples; full shell for terminals/tips. Both sides and all three tip options. Accelerated oracle checked against unaccelerated equations at 1e-12m on first case.",
                    "sweep_method": "Corresponding-ring edge progress along unit centreline segment; actual side-triangle normal versus mean ring radial vector. Strictly positive expected; caps excluded. Not a general self-intersection proof.",
                    "configuration_encoding": "zero_other_shapes=true expands to zero for every shape_names entry before overlaying modifiers; otherwise inherited defaults remain. Family normalization is performed by the audited package.",
                    "thresholds_mm": {"pan": 4, "tips": 1, "can_road": 50, "can_height_strictly_greater": 20},
                    "packaging_note": "50mm road assertion applies to cans, not pipes. A 50mm-high pipe requires roof>=104mm for 50mm road plus4mm pan gap. Global floor minima do not prove every possible reroute impossible."}
        records, worst = [], {}
        for index, (category, style, modifiers) in enumerate(cases):
            if args.cases and index not in args.cases:
                continue
            if args.limit is not None and len(records) >= args.limit:
                break
            try:
                body = cb.build(style=style, modifiers=modifiers)
                metrics, sweeps, package = audit(body, (BuildContext, Palette, get_component), validate=not records)
                records.append(compact_record(index, category, style, modifiers, shapes, metrics, sweeps, package))
                update_worst(worst, index, metrics, sweeps)
            except Exception as error:
                records.append({"id": index, "category": category, "style": style, "modifiers": modifiers,
                                "error": repr(error)})
            if len(records) % 20 == 0:
                save(args.output, metadata, records, worst, start)
                print(json.dumps({"completed": len(records), "seconds": round(time.time() - start, 1),
                                  "pan_failures": len(summary(records)["clearance_failures"]["whole_branch"]),
                                  "fold_cases": len(summary(records)["other_failures"]["nonoutward_triangles"])}), flush=True)
        save(args.output, metadata, records, worst, start)
        print(json.dumps(clean({"output": str(args.output.resolve()), "summary": summary(records), "worst": worst}), indent=2), flush=True)


if __name__ == "__main__":
    main()
