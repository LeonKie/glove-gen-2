"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from . import demold, meshio, pipeline
from .config import MoldConfig


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="positive STL/OBJ/PLY to build a mold for")
    p.add_argument("-v", "--verbose", action="store_true", help="log every stage")
    p.add_argument(
        "--direction",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="pin the pull direction instead of searching for it",
    )
    p.add_argument(
        "--proxy-faces",
        type=int,
        default=None,
        help="face count of the decimated proxy used for ray queries",
    )


def _config_from_args(args) -> MoldConfig:
    cfg = MoldConfig()
    for name, value in (
        ("block_margin", getattr(args, "margin", None)),
        ("block_shape", getattr(args, "block", None)),
        ("cavity_offset", getattr(args, "cavity_offset", None)),
    ):
        if value is not None:
            setattr(cfg, name, value)
    if getattr(args, "grid", None):
        cfg.parting.grid = args.grid
    if getattr(args, "proxy_faces", None):
        cfg.demold.proxy_faces = args.proxy_faces
    if getattr(args, "no_keys", False):
        cfg.keys.enabled = False
    if getattr(args, "no_spout", False):
        cfg.spout.enabled = False
    if getattr(args, "no_vents", False):
        cfg.vents.enabled = False
    if getattr(args, "keys", None) is not None:
        cfg.keys.count = args.keys
    if getattr(args, "vents", None) is not None:
        cfg.vents.count = args.vents
    # Sizes. Placement stays automatic; these only change how big the knobs and
    # holes it picks come out.
    for target, attr, option in (
        (cfg.keys, "radius", "key_radius"),
        (cfg.keys, "height", "key_height"),
        (cfg.keys, "draft_deg", "key_draft"),
        (cfg.keys, "clearance", "key_clearance"),
        (cfg.spout, "inner_radius", "spout_inner"),
        (cfg.spout, "outer_radius", "spout_outer"),
        (cfg.vents, "radius", "vent_radius"),
    ):
        value = getattr(args, option, None)
        if value is not None:
            setattr(target, attr, value)
    return cfg


def _load_plan(path: Path | None):
    """Read a feature plan written by an earlier run, if one was asked for.

    ``report.json`` carries the plan a run used, so editing that and passing it
    back is how a command-line user overrides the automatic choice -- the same
    document the web app edits interactively.
    """
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    # Accept either a bare plan or a whole report containing one.
    return data.get("feature_plan", data) if isinstance(data, dict) else data


def _progress_printer(quiet: bool):
    state = {"last": -1.0}

    def report(fraction: float, message: str = "") -> None:
        if quiet:
            return
        pct = int(fraction * 100)
        if pct != state["last"] or message:
            state["last"] = pct
            print(f"  [{pct:3d}%] {message}", file=sys.stderr, flush=True)

    return report


def cmd_analyze(args) -> int:
    cfg = _config_from_args(args)
    part = meshio.load_mesh(args.input)

    if args.direction:
        d = np.asarray(args.direction, dtype=float)
        d /= np.linalg.norm(d)
        score = demold.score_direction(
            meshio.proxy(part, cfg.demold.proxy_faces),
            d,
            cfg.demold.fine_grid,
            block_margin=cfg.block_margin,
        )
        ranked = [score]
    else:
        t0 = time.time()
        d, score, ranked = demold.suggest_direction(
            part,
            cfg.demold,
            block_margin=cfg.block_margin,
            progress=_progress_printer(args.quiet),
        )
        print(f"# direction search took {time.time() - t0:.1f}s", file=sys.stderr)

    # One JSON document on stdout, so the output is pipeable into jq.
    out = {
        "part": meshio.mesh_stats(part),
        "pull_direction": [round(float(v), 6) for v in d],
        "best": score.as_dict(),
        "alternatives": [s.as_dict() for s in ranked[1 : 1 + args.top]],
    }

    if args.heatmap:
        mesh, stats = pipeline.heatmap_mesh(
            part, d, proxy_faces=cfg.demold.proxy_faces
        )
        meshio.export(mesh, args.heatmap)
        out["heatmap"] = {"path": str(args.heatmap), **stats}

    print(json.dumps(out, indent=2))
    return 0


def cmd_mold(args) -> int:
    cfg = _config_from_args(args)
    result = pipeline.run(
        args.input,
        cfg,
        direction=args.direction,
        verify=not args.no_verify,
        plan=_load_plan(args.plan),
        progress=_progress_printer(args.quiet),
    )
    written = result.write(args.out, extras=not args.no_extras)

    report = result.report()
    report["written"] = written
    out_dir = Path(args.out)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    if args.heatmap:
        mesh, stats = pipeline.heatmap_mesh(
            result.part, result.direction, proxy_faces=cfg.demold.proxy_faces
        )
        meshio.export(mesh, out_dir / "undercut_heatmap.ply")
        report["heatmap"] = stats

    print(json.dumps(report, indent=2))

    sep = result.separation
    if sep and not sep.get("opens", True):
        print("ERROR: mold halves do not separate", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="glovegen",
        description="Turn a positive scan into a printable two-part casting mold.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="demold analysis only, no geometry built")
    _add_common(p_an)
    p_an.add_argument("--margin", type=float, default=None, help="block wall margin, mm")
    p_an.add_argument("--top", type=int, default=5, help="how many runner-up directions to list")
    p_an.add_argument("--heatmap", type=Path, default=None, help="write a colour-coded PLY here")
    p_an.set_defaults(func=cmd_analyze)

    p_mo = sub.add_parser("mold", help="build the mold halves")
    _add_common(p_mo)
    p_mo.add_argument("-o", "--out", default="out", help="output directory")
    p_mo.add_argument("--margin", type=float, default=None, help="minimum mold wall, mm")
    p_mo.add_argument(
        "--block", choices=("box", "hull"), default=None,
        help="box = rectangular block; hull = dilated convex hull (much less material)",
    )
    p_mo.add_argument("--grid", type=int, default=None, help="parting surface grid resolution")
    p_mo.add_argument("--cavity-offset", type=float, default=None, help="grow the cavity, mm")
    p_mo.add_argument("--keys", type=int, default=None, help="number of alignment keys")
    p_mo.add_argument("--vents", type=int, default=None, help="maximum number of vents")
    p_mo.add_argument("--no-keys", action="store_true")
    p_mo.add_argument("--no-spout", action="store_true")
    p_mo.add_argument("--no-vents", action="store_true")

    sizes = p_mo.add_argument_group(
        "feature sizes",
        "Placement stays automatic; these change how big the knobs and holes are.",
    )
    sizes.add_argument("--key-radius", type=float, default=None, help="key radius, mm")
    sizes.add_argument("--key-height", type=float, default=None, help="key height, mm")
    sizes.add_argument("--key-draft", type=float, default=None, help="key draft angle, degrees")
    sizes.add_argument(
        "--key-clearance", type=float, default=None, help="key/socket gap per side, mm"
    )
    sizes.add_argument(
        "--spout-inner", type=float, default=None, help="spout radius at the cavity, mm"
    )
    sizes.add_argument(
        "--spout-outer", type=float, default=None, help="spout radius at the block face, mm"
    )
    sizes.add_argument("--vent-radius", type=float, default=None, help="vent radius, mm")
    p_mo.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="apply this feature plan instead of choosing one "
        "(a plan, or a report.json containing one)",
    )
    p_mo.add_argument("--no-verify", action="store_true", help="skip the separation check")
    p_mo.add_argument("--no-extras", action="store_true", help="only write the two halves")
    p_mo.add_argument("--heatmap", action="store_true", help="also write undercut_heatmap.ply")
    p_mo.set_defaults(func=cmd_mold)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
