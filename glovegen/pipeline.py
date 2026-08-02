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
from .core import CoreResult
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
    "core",
    "features",
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
    # Present only when cfg.core.enabled. `core` is the one-piece core; `core_a`
    # and `core_b` are set instead when cfg.core.split cut it on the parting
    # surface. `glove` is what will be cast, kept for inspection.
    core: trimesh.Trimesh | None = None
    core_a: trimesh.Trimesh | None = None
    core_b: trimesh.Trimesh | None = None
    glove: trimesh.Trimesh | None = None
    core_result: CoreResult | None = None
    core_stats: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def core_parts(self) -> dict[str, trimesh.Trimesh]:
        """The printable core pieces by output name, empty if there is no core."""
        if self.core_a is not None and self.core_b is not None:
            return {"core_a": self.core_a, "core_b": self.core_b}
        if self.core is not None:
            return {"core": self.core}
        return {}

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
        if self.core_stats:
            out["core"] = self.core_stats
        if self.direction_score is not None:
            out["demold"] = self.direction_score.as_dict()
        return out

    def write(self, out_dir: str | Path, *, extras: bool = True) -> dict:
        """Write the printable parts (and diagnostics) to ``out_dir``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = {
            "half_a": str(meshio.export(self.half_a, out_dir / "mold_half_a.stl")),
            "half_b": str(meshio.export(self.half_b, out_dir / "mold_half_b.stl")),
        }
        for name, mesh in self.core_parts().items():
            written[name] = str(meshio.export(mesh, out_dir / f"{name}.stl"))
        if extras and self.glove is not None:
            written["glove_preview"] = str(
                meshio.export(self.glove, out_dir / "glove_preview.stl")
            )
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

    ctx = features.FeatureContext.from_mold(mold_result)
    cavity = mold_result.cavity if mold_result.cavity is not None else part

    core_result: CoreResult | None = None
    pour_axis = None
    if cfg.core.enabled:
        report(0.72, "building the core")
        t0 = time.time()
        pour_axis = features.choose_pour_axis(cavity, cfg)
        core_result = core_mod.build_core(
            cavity,
            mold_result.frame,
            pour_axis,
            cfg,
            progress=lambda f, msg: report(0.72 + 0.06 * f, msg),
        )
        timings.update(core_result.timings)
        timings["core_total"] = time.time() - t0
        # Nothing is cut into the halves for it: the cuff cap is taken from the
        # cavity, so it already fits the void the cavity left.

    t0 = time.time()
    if plan is None:
        report(0.78, "planning features")
        plan = features.plan_features(ctx, cfg, pour_axis=pour_axis)
    else:
        plan = features.FeaturePlan.from_dict(plan)
    half_a, half_b, feat = features.apply_plan(
        mold_result.half_a,
        mold_result.half_b,
        ctx,
        plan,
        cfg=cfg,
        progress=lambda f, msg: report(0.80 + 0.12 * f, msg),
    )
    timings["features"] = time.time() - t0
    validate.assert_solid_enough(half_a, "half A (with features)")
    validate.assert_solid_enough(half_b, "half B (with features)")

    core_mesh = core_a = core_b = glove = None
    core_stats: dict = {}
    if core_result is not None:
        report(0.92, "cutting the glove preview")
        t0 = time.time()
        core_mesh = core_result.core
        glove = core_mod.glove_preview(cavity, core_result)
        if cfg.core.split:
            core_a, core_b = mold.split_mold(core_mesh, mold_result.parting_solid)
        timings["core_finish"] = time.time() - t0
        core_stats = _core_stats(cavity, core_mesh, glove, core_result, cfg)

    separation: dict = {}
    if verify:
        report(0.95, "verifying the mold opens")
        t0 = time.time()
        separation = validate.separation_report(half_a, half_b, part, d)
        if core_a is not None and core_b is not None:
            # A split core lifts away with its own mold half, on exactly the
            # argument that makes the mold open -- so check it the same way.
            # There is no equivalent check for a one-piece core: see
            # `release_note` below.
            core_stats["halves_open"] = validate.separation_report(
                core_a, core_b, glove, d
            )["opens"]
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
        feature_plan=features.FeaturePlan.from_dict(plan).normalised(cfg),
        core=core_mesh,
        core_a=core_a,
        core_b=core_b,
        glove=glove,
        core_result=core_result,
        core_stats=core_stats,
        timings=timings,
    )


def _core_stats(
    cavity: trimesh.Trimesh,
    core: trimesh.Trimesh,
    glove: trimesh.Trimesh,
    result: CoreResult,
    cfg: MoldConfig,
) -> dict:
    """The numbers worth checking before printing a core."""
    stats = dict(result.as_dict())
    stats.update(
        {
            "split": bool(cfg.core.split),
            # Deliberately not a measured number. A cast glove is a
            # zero-clearance fit on its former -- the glove's inner surface *is*
            # the core's surface -- so it comes off by stretching, never by
            # sliding, and every way of scoring "can the core be withdrawn"
            # bottoms out on that. Booleans across the coincident surfaces return
            # slivers; ray columns that skim along them dip in and out of the
            # extracted mesh and read as deep re-entrancy. Both scored a plain
            # rod, which slides out perfectly, at 8% of the cast. A number that
            # wrong is worse than no number, so what is reported instead is the
            # assumption itself -- and `--core-split` is the way to replace the
            # assumption with a guarantee.
            "release_note": (
                "one-piece core: peeled off a flexible cast, as the pipeline "
                "already assumes for the mold halves. Use core.split for a "
                "mechanically guaranteed release."
            )
            if not cfg.core.split
            else "split core: each piece lifts away with its own mold half",
            "volume_cm3": round(float(core.volume) / 1000.0, 2),
            "components": core_mod.components(core),
            "glove_volume_cm3": round(float(glove.volume) / 1000.0, 2),
            "glove_components": core_mod.components(glove),
            "wall": validate.core_wall_report(cavity, result.body, cfg.core.thickness),
        }
    )
    if stats["components"] > 1:
        log.error(
            "the core is in %d pieces: a %.2f mm wall erased a feature narrower "
            "than %.2f mm. Reduce core.thickness.",
            stats["components"],
            cfg.core.thickness,
            2.0 * cfg.core.thickness,
        )
    return stats


@dataclass
class FeatureResult:
    """Outcome of re-cutting features into halves that were already built."""

    half_a: trimesh.Trimesh
    half_b: trimesh.Trimesh
    direction: np.ndarray
    plan: features.FeaturePlan
    feature_report: FeatureReport
    separation: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def report(self) -> dict:
        a, b = float(self.half_a.volume), float(self.half_b.volume)
        return {
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

    def write(self, out_dir: str | Path) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "half_a": str(meshio.export(self.half_a, out_dir / "mold_half_a.stl")),
            "half_b": str(meshio.export(self.half_b, out_dir / "mold_half_b.stl")),
        }


def apply_feature_plan(
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    ctx: features.FeatureContext,
    plan: features.FeaturePlan | dict,
    *,
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
    timings: dict[str, float] = {}

    report(0.05, "cutting features")
    t0 = time.time()
    half_a, half_b, feat = features.apply_plan(
        half_a, half_b, ctx, plan, progress=lambda f, msg: report(0.05 + 0.75 * f, msg)
    )
    timings["features"] = time.time() - t0
    validate.assert_solid_enough(half_a, "half A (with features)")
    validate.assert_solid_enough(half_b, "half B (with features)")

    separation: dict = {}
    if verify:
        report(0.85, "verifying the mold opens")
        t0 = time.time()
        separation = validate.separation_report(
            half_a, half_b, cast if cast is not None else ctx.cavity, ctx.direction
        )
        timings["verify"] = time.time() - t0
        if not separation.get("opens", False):
            log.error("mold halves do not separate: %s", separation)

    report(1.0, "done")
    return FeatureResult(
        half_a=half_a,
        half_b=half_b,
        direction=np.asarray(ctx.direction, dtype=np.float64),
        plan=features.FeaturePlan.from_dict(plan).normalised(),
        feature_report=feat,
        separation=separation,
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
