"""Command line interface:  python -m makecar <command> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import CarConfig, load_config_file
from .pipeline import run_config, build_car


def cmd_build(args):
    out_root = Path(args.out)
    for path in args.config:
        cfg = CarConfig.load(path)
        if args.style:
            cfg.raw["body"]["style"] = args.style
        if args.seed is not None:
            cfg.raw["seed"] = args.seed
        if args.set:
            for kv in args.set:
                k, v = kv.split("=", 1)
                cfg.raw["body"].setdefault("modifiers", {})[k] = float(v)
        print(f"building {path}")
        run_config(cfg, out_root, with_variants=not args.no_variants)


def cmd_samples(args):
    cfg_dir = Path(args.configs)
    paths = sorted(cfg_dir.glob("*.yaml")) + sorted(cfg_dir.glob("*.yml")) + sorted(cfg_dir.glob("*.json"))
    if not paths:
        sys.exit(f"no configs found in {cfg_dir}")
    out_root = Path(args.out)
    print(f"generating {len(paths)} sample configs into {out_root}/")
    for p in paths:
        cfg = CarConfig.load(p)
        run_config(cfg, out_root)
    write_index(out_root)


def write_index(out_root: Path):
    rows = []
    for d in sorted(out_root.iterdir()):
        if (d / "assembly.json").exists():
            info = json.loads((d / "assembly.json").read_text())
            m = info["measurements"]
            rows.append((d.name, info.get("config", {}).get("body", {}).get("style"), m["length"], m["width"], m["height"], m["wheelbase"], info["stats"]["connectors"], info["stats"]["components"]))
    lines = ["# makecar samples", "", "| sample | style | length | width | height | wheelbase | connectors | components | files |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        name = r[0]
        style = r[1] if isinstance(r[1], str) else json.dumps(r[1])
        lines.append(f"| {name} | {style} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]} | {r[7]} | "
                     f"[preview]({name}/preview.png) [interior]({name}/interior.png) [blueprint]({name}/blueprint.svg) [obj]({name}/car.obj) |")
    (out_root / "README.md").write_text("\n".join(lines) + "\n")


def cmd_list_modifiers(args):
    from .body import CarBody
    cb = CarBody()
    rows = cb.modifiers()
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    for g, items in groups.items():
        print(f"[{g}]")
        for r in items:
            rng = f"[{r['min']:g}, {r['max']:g}]"
            print(f"  {r['name']:26s} {rng:10s} {r['description']}")


def cmd_list_components(args):
    from .components import describe_all
    for d in describe_all():
        kind = "morphable" if d["morphable"] else "procedural"
        print(f"{d['name']:22s} accepts={','.join(d['accepts']):20s} default_for={','.join(d['default_for']) or '-':18s} {kind}")
        if d["description"]:
            print(f"{'':22s} {d['description']}")
        if d["options"]:
            print(f"{'':22s} options: {json.dumps(d['options'])}")


def cmd_list_styles(args):
    from .body.styles import STYLE_DESCRIPTIONS, STYLE_OVERRIDES
    for k, v in STYLE_DESCRIPTIONS.items():
        print(f"{k:10s} {v}   ({len(STYLE_OVERRIDES[k])} parameter overrides)")


def cmd_connectors(args):
    cfg = CarConfig.load(args.config)
    res, asm, _ = build_car(cfg)
    for c in asm.connectors:
        comp = asm.connector_component(c.name) or "-"
        o = ", ".join(f"{v:.3f}" for v in c.origin)
        extra = ""
        if c.kind == "circle":
            extra = f"r={c.radius:.3f}"
        elif c.kind == "rectangle":
            extra = f"{c.width:.3f}x{c.height:.3f}"
        elif c.kind == "polygon":
            extra = f"{len(c.points)} pts {c.width:.2f}x{c.height:.2f}"
        print(f"{c.name:34s} {c.kind:9s} ({o}) {extra:22s} tags={','.join(sorted(c.tags)):24s} -> {comp}")


def cmd_random(args):
    import random
    from .body.styles import style_names
    out_root = Path(args.out)
    rng = random.Random(args.seed)
    for i in range(args.count):
        style = args.style or rng.choice(style_names())
        seed = rng.randrange(1 << 30)
        cfg = CarConfig.from_dict({
            "name": f"random_{style}_{i + 1:02d}",
            "description": f"random variation of the {style} style (seed {seed})",
            "seed": seed,
            "body": {"style": style, "random": {"amount": args.amount}},
            "palette": {"paint": "#%02x%02x%02x" % (rng.randrange(40, 220), rng.randrange(40, 200), rng.randrange(40, 220))},
        })
        run_config(cfg, out_root)


def cmd_viewer(args):
    from .viewer.server import serve

    serve(args.config, host=args.host, port=args.port, open_browser=not args.no_browser, output_dir=args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="makecar", description="MakeHuman-style procedural car generator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build car(s) from config file(s)")
    b.add_argument("config", nargs="+")
    b.add_argument("-o", "--out", default="output")
    b.add_argument("--style", help="override body.style")
    b.add_argument("--seed", type=int)
    b.add_argument("--set", action="append", metavar="MOD=VAL", help="override a modifier value")
    b.add_argument("--no-variants", action="store_true")
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("samples", help="build every config in a folder")
    s.add_argument("-c", "--configs", default="configs")
    s.add_argument("-o", "--out", default="output")
    s.set_defaults(fn=cmd_samples)
    r = sub.add_parser("random", help="generate random variations")
    r.add_argument("-o", "--out", default="output")
    r.add_argument("-n", "--count", type=int, default=3)
    r.add_argument("--style")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--amount", type=float, default=0.6)
    r.set_defaults(fn=cmd_random)
    c = sub.add_parser("connectors", help="list the connectors a config produces")
    c.add_argument("config")
    c.set_defaults(fn=cmd_connectors)
    v = sub.add_parser("viewer", help="interactive 3-D viewer/editor for targets and connectors")
    v.add_argument("config", nargs="?", help="config file to edit (default: a new untitled sedan)")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("-o", "--out", default="output", help="folder for exports from the viewer")
    v.add_argument("--no-browser", action="store_true")
    v.set_defaults(fn=cmd_viewer)
    sub.add_parser("list-modifiers", help="list body modifiers (sliders)").set_defaults(fn=cmd_list_modifiers)
    sub.add_parser("list-components", help="list registered components").set_defaults(fn=cmd_list_components)
    sub.add_parser("list-styles", help="list body styles (macro targets)").set_defaults(fn=cmd_list_styles)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
