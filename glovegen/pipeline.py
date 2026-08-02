"""End-to-end orchestration: scan in, printable mold halves out."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from . import core as core_mod
from . import demold, features, meshio, mold, validate
from .config import MoldConfig
from .features import FeatureReport
from .mold import MoldResult

log = logging.getLogger(__name__)

STAGES = (
    "load",
    "direction",
    "block",
    "parting",
    "cavity",
    "split",
    "features",
    "core",
    "verify",
)


@dataclass
class PipelineResult:
    part: trimesh.Trimesh
    direction: np.ndarray
    half_a: trimesh.Trimesh
    half_b: trimesh.Trimesh
    mold_result: MoldResult
    feature_report: FeatureReport
    direction_score: demold.DirectionScore | None
    separation: dict
    feature_plan: features.FeaturePlan | None = None
    # Only built when a core was asked for: the third printed body, plate and
    # all, plus what was measured about it.
    core: trimesh.Trimesh | None = None
    core_report: dict = field(default_factory=dict)
    # The loose dowel pins, if any were bored for.
    pins: trimesh.Trimesh | None = None
    # The eroded body before the plan touched it. Cached rather than rebuilt:
    # the erosion is the expensive part of a core run and depends on nothing the
    # plan decides, so an edited plan can be re-cut from it.
    core_body: trimesh.Trimesh | None = None
    timings: dict = field(default_factory=dict)

    def report(self) -> dict:
        """Everything worth knowing about this run, as plain JSON-able data."""
        out = {
            "part": meshio.mesh_stats(self.part),
            "pull_direction": [round(float(v), 6) for v in self.direction],
            "mold": {
                k: v for k, v in self.mold_result.stats.items() if k != "parting"
            },
            "parting_surface": self.mold_result.stats.get("parting", {}),
            "features": self.feature_report.as_dict(),
            # The plan is the editable form of the features: hand it back so a
            # caller can change a size or drop an item and re-run, instead of
            # reverse-engineering it from the report.
            "feature_plan": (
                self.feature_plan.as_dict() if self.feature_plan is not None else {}
            ),
            "separation": self.separation,
            "halves": {
                "half_a": meshio.mesh_stats(self.half_a),
                "half_b": meshio.mesh_stats(self.half_b),
            },
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
        }
        if self.direction_score is not None:
            out["demold"] = self.direction_score.as_dict()
        if self.core is not None:
            out["core"] = self.core_report
            out["halves"]["core"] = meshio.mesh_stats(self.core)
        return out

    def write(self, out_dir: str | Path, *, extras: bool = True) -> dict:
        """Write the printable parts (and diagnostics) to ``out_dir``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = {
            "half_a": str(meshio.export(self.half_a, out_dir / "mold_half_a.stl")),
            "half_b": str(meshio.export(self.half_b, out_dir / "mold_half_b.stl")),
        }
        if self.core is not None:
            written["core"] = str(meshio.export(self.core, out_dir / "core.stl"))
        if self.pins is not None:
            written["pins"] = str(meshio.export(self.pins, out_dir / "dowel_pins.stl"))
        if extras:
            written["parting_surface"] = str(
                meshio.export(
                    self.mold_result.surface.surface_mesh(),
                    out_dir / "parting_surface.stl",
                )
            )
            written["cavity_preview"] = str(
                meshio.export(self.mold_result.mold, out_dir / "mold_uncut.stl")
            )
        return written


def run(
    source: str | Path | trimesh.Trimesh,
    cfg: MoldConfig | None = None,
    *,
    direction=None,
    search_direction: bool = True,
    verify: bool = True,
    plan: features.FeaturePlan | dict | None = None,
    progress=None,
) -> PipelineResult:
    """Build a two-part mold.

    ``direction`` pins the pull direction; otherwise it is searched for (unless
    ``search_direction`` is False, in which case the part's shortest bounding-box
    axis is used as a cheap stand-in).

    ``plan`` pins the knobs and holes. Left alone, they are chosen
    automatically -- which is what a command-line run wants: one invocation,
    a finished mold, no questions.
    """
    cfg = cfg or MoldConfig()
    report = progress or (lambda *a, **k: None)
    timings: dict[str, float] = {}

    t0 = time.time()
    report(0.0, "loading mesh")
    part = source if isinstance(source, trimesh.Trimesh) else meshio.load_mesh(source)
    validate.assert_solid_enough(part, "input part")
    timings["load"] = time.time() - t0
    log.info("part: %s", meshio.mesh_stats(part))

    score = None
    t0 = time.time()
    if direction is not None:
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        score = demold.score_direction(
            meshio.proxy(part, cfg.demold.proxy_faces),
            d,
            cfg.demold.fine_grid,
            block_margin=cfg.block_margin,
        )
    elif cfg.demold.forced_direction is not None or search_direction:
        report(0.02, "searching for pull direction")
        d, score, _ = demold.suggest_direction(
            part,
            cfg.demold,
            block_margin=cfg.block_margin,
            progress=lambda f, msg: report(0.02 + 0.18 * f, msg),
        )
    else:
        # Cheap fallback: pull across the part's thinnest dimension.
        d = np.zeros(3)
        d[int(np.argmin(part.extents))] = 1.0
        score = demold.score_direction(
            meshio.proxy(part, cfg.demold.proxy_faces), d, cfg.demold.fine_grid
        )
    timings["direction"] = time.time() - t0
    log.info("pull direction %s", np.round(d, 4))

    report(0.22, "building mold")
    t0 = time.time()
    mold_result = mold.build_mold(
        part, d, cfg, progress=lambda f, msg: report(0.22 + 0.5 * f, msg)
    )
    timings.update(mold_result.timings)
    timings["mold_total"] = time.time() - t0

    # The core is built before the plan, not after it: the plate, the dowels and
    # the tabs are plan items like any other, and they need something to attach
    # to before they can be placed. The erosion is the expensive part and it
    # depends on nothing the plan decides, which is what lets an edited plan be
    # re-cut later without paying for it again.
    core_body = None
    if cfg.core.enabled:
        report(0.70, f"eroding the part by {cfg.core.wall} mm")
        t0 = time.time()
        core_body = core_mod.build_core_body(part, cfg)
        timings["core_erode"] = time.time() - t0
        log.info(
            "core body: %d faces, %.1f cm3 (part %.1f cm3)",
            len(core_body.faces),
            core_body.volume / 1000.0,
            part.volume / 1000.0,
        )

    t0 = time.time()
    ctx = features.FeatureContext.from_mold(mold_result, core=core_body)
    if plan is None:
        report(0.74, "planning features")
        plan = features.plan_features(ctx, cfg)
    plan = features.FeaturePlan.from_dict(plan).normalised(cfg)
    bodies, feat = features.apply_plan(
        mold_result.half_a,
        mold_result.half_b,
        ctx,
        plan,
        cfg=cfg,
        progress=lambda f, msg: report(0.76 + 0.16 * f, msg),
    )
    half_a, half_b, core_mesh = bodies
    pins = bodies.pins
    timings["features"] = time.time() - t0
    validate.assert_solid_enough(half_a, "half A (with features)")
    validate.assert_solid_enough(half_b, "half B (with features)")

    core_report = dict(feat.core)
    separation: dict = {}
    if verify:
        report(0.94, "verifying the mold opens")
        t0 = time.time()
        separation = validate.separation_report(half_a, half_b, part, d)
        if core_mesh is not None:
            core_report["release"] = core_mod.release_report(
                core_mesh, half_a, half_b, d
            )
        timings["verify"] = time.time() - t0
        if not separation.get("opens", False):
            log.error("mold halves do not separate: %s", separation)

    report(1.0, "done")
    return PipelineResult(
        part=part,
        direction=d,
        half_a=half_a,
        half_b=half_b,
        mold_result=mold_result,
        feature_report=feat,
        direction_score=score,
        separation=separation,
        feature_plan=plan,
        core=core_mesh,
        core_report=core_report,
        pins=pins,
        core_body=core_body,
        timings=timings,
    )


@dataclass
class FeatureResult:
    """Outcome of re-cutting features into halves that were already built."""

    half_a: trimesh.Trimesh
    half_b: trimesh.Trimesh
    direction: np.ndarray
    plan: features.FeaturePlan
    feature_report: FeatureReport
    separation: dict = field(default_factory=dict)
    core: trimesh.Trimesh | None = None
    core_report: dict = field(default_factory=dict)
    pins: trimesh.Trimesh | None = None
    timings: dict = field(default_factory=dict)

    def report(self) -> dict:
        a, b = float(self.half_a.volume), float(self.half_b.volume)
        out = {
            "pull_direction": [round(float(v), 6) for v in self.direction],
            "mold": {
                "half_a_volume_cm3": round(a / 1000.0, 2),
                "half_b_volume_cm3": round(b / 1000.0, 2),
                "mold_volume_cm3": round((a + b) / 1000.0, 2),
                # Meaningless here: the halves were not just cut from one solid,
                # they have had material added and removed. Reported as zero so
                # the shape of the document does not change between job kinds.
                "split_volume_error_cm3": 0.0,
            },
            "features": self.feature_report.as_dict(),
            "feature_plan": self.plan.as_dict(),
            "separation": self.separation,
            "halves": {
                "half_a": meshio.mesh_stats(self.half_a),
                "half_b": meshio.mesh_stats(self.half_b),
            },
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
        }
        if self.core is not None:
            out["core"] = self.core_report
            out["halves"]["core"] = meshio.mesh_stats(self.core)
        return out

    def write(self, out_dir: str | Path) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = {
            "half_a": str(meshio.export(self.half_a, out_dir / "mold_half_a.stl")),
            "half_b": str(meshio.export(self.half_b, out_dir / "mold_half_b.stl")),
        }
        if self.core is not None:
            written["core"] = str(meshio.export(self.core, out_dir / "core.stl"))
        if self.pins is not None:
            written["pins"] = str(meshio.export(self.pins, out_dir / "dowel_pins.stl"))
        return written


def apply_feature_plan(
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    ctx: features.FeatureContext,
    plan: features.FeaturePlan | dict,
    *,
    cfg: MoldConfig | None = None,
    cast: trimesh.Trimesh | None = None,
    verify: bool = True,
    progress=None,
) -> FeatureResult:
    """Cut a (typically edited) plan into base halves from an earlier run.

    This is the interactive half of the story. Everything up to the split --
    direction search, parting surface, three booleans on a multi-million-face
    scan -- is unchanged by moving a knob or widening a vent, so it is not
    redone: only the features are, against the halves as they came out of the
    split.
    """
    report = progress or (lambda *a, **k: None)
    cfg = cfg or MoldConfig()
    timings: dict[str, float] = {}

    report(0.05, "cutting features")
    t0 = time.time()
    bodies, feat = features.apply_plan(
        half_a, half_b, ctx, plan, cfg=cfg, progress=lambda f, msg: report(0.05 + 0.75 * f, msg)
    )
    half_a, half_b, core_mesh = bodies
    pins = bodies.pins
    timings["features"] = time.time() - t0
    validate.assert_solid_enough(half_a, "half A (with features)")
    validate.assert_solid_enough(half_b, "half B (with features)")

    core_report = dict(feat.core)
    separation: dict = {}
    if verify:
        report(0.85, "verifying the mold opens")
        t0 = time.time()
        separation = validate.separation_report(
            half_a, half_b, cast if cast is not None else ctx.cavity, ctx.direction
        )
        if core_mesh is not None:
            core_report["release"] = core_mod.release_report(
                core_mesh, half_a, half_b, ctx.direction
            )
        timings["verify"] = time.time() - t0
        if not separation.get("opens", False):
            log.error("mold halves do not separate: %s", separation)

    report(1.0, "done")
    return FeatureResult(
        half_a=half_a,
        half_b=half_b,
        direction=np.asarray(ctx.direction, dtype=np.float64),
        plan=features.FeaturePlan.from_dict(plan).normalised(cfg),
        feature_report=feat,
        separation=separation,
        core=core_mesh,
        core_report=core_report,
        pins=pins,
        timings=timings,
    )


def heatmap_mesh(
    part: trimesh.Trimesh,
    direction,
    *,
    proxy_faces: int = 150_000,
) -> tuple[trimesh.Trimesh, dict]:
    """A copy of the part coloured by undercut severity, for inspection."""
    small = meshio.proxy(part, proxy_faces)
    info = demold.face_undercut(small, direction)
    sev = info["severity"]

    colours = np.zeros((len(sev), 4), dtype=np.uint8)
    colours[:, 3] = 255
    clean = sev <= 0.0
    colours[clean] = [70, 170, 110, 255]  # releases along +/-d
    hot = ~clean
    # green -> amber -> red with severity
    colours[hot, 0] = np.clip(200 + 55 * sev[hot], 0, 255).astype(np.uint8)
    colours[hot, 1] = np.clip(170 * (1.0 - sev[hot]), 0, 255).astype(np.uint8)
    colours[hot, 2] = 40

    out = small.copy()
    out.visual.face_colors = colours
    stats = {
        k: v for k, v in info.items() if k not in ("severity", "undercut", "depth", "direction")
    }
    return out, stats
