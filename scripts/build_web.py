#!/usr/bin/env python3
"""Build the in-browser editor as a static site (for GitHub Pages).
Usage (from the repo root): python -m scripts.build_web [-o site]

The page runs makecar in Pyodide. Target libraries are built here and shipped,
so the browser does not have to build them on first load.
"""
import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "makecar" / "viewer" / "static"
APP_TAG = '<script type="module" src="static/app.js"></script>'

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--out", default="site")
ap.add_argument("--default", default="sedan", help="config opened when the URL has no ?config=")
args = ap.parse_args()

out = Path(args.out).resolve()
web = out / "web"
if out.exists():
    shutil.rmtree(out)
shutil.copytree(STATIC, out / "static", ignore=shutil.ignore_patterns("index.html"))
html = (STATIC / "index.html").read_text()
if APP_TAG not in html:
    raise SystemExit(f"index.html no longer loads the app with {APP_TAG!r}")
(out / "index.html").write_text(html.replace(APP_TAG, '<script type="module" src="web/backend.js"></script>\n' + APP_TAG))
(out / ".nojekyll").touch()
web.mkdir()
for name in ("backend.js", "worker.js"):
    shutil.copy(ROOT / "web" / name, web / name)

with zipfile.ZipFile(web / "makecar.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted((ROOT / "makecar").rglob("*.py")):
        if "__pycache__" not in p.parts:
            zf.write(p, p.relative_to(ROOT).as_posix())

os.environ["MAKECAR_CACHE"] = str(web / "cache")
from makecar.body.params import BodyParams  # noqa: E402
from makecar.body.targets import _cache_key  # noqa: E402
from makecar.config import CarConfig  # noqa: E402
from makecar.pipeline import car_body_for  # noqa: E402

(web / "configs").mkdir()
manifest = {"default": args.default, "configs": [], "cache": {}}
for path in sorted(ROOT.glob("configs/*.yaml")) + sorted(ROOT.glob("configs/measured/*.yaml")):
    cfg = CarConfig.load(path)
    car_body_for(cfg)
    params = BodyParams(**json.loads(json.dumps(cfg.base_params().to_dict(), sort_keys=True)))
    cache = f"cache/body_targets_{_cache_key(params)}.npz"
    if not (web / cache).exists():
        raise SystemExit(f"{path.name}: expected the target library at {cache}")
    shutil.copy(path, web / "configs" / path.name)
    manifest["configs"].append(path.stem)
    manifest["cache"][path.stem] = cache
    print(f"{path.stem:34s} {cache}")
if args.default not in manifest["configs"]:
    raise SystemExit(f"default config {args.default!r} is not in configs/")
(web / "manifest.json").write_text(json.dumps(manifest, indent=1))

size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
print(f"wrote {out} ({size / 1e6:.1f} MB)")
