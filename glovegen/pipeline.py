"""End-to-end orchestration: scan in, printable mold halves out."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

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
            "separation": self.separation,
            "halves": {
                "half_a": meshio.mesh_stats(self.half_a),
                "half_b": meshio.mesh_stats(self.half_b),
            },
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
        }
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
    progress=None,
) -> PipelineResult:
    """Build a two-part mold.

    ``direction`` pins the pull direction; otherwise it is searched for (unless
    ``search_direction`` is False, in which case the part's shortest bounding-box
    axis is used as a cheap stand-in).
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

    t0 = time.time()
    half_a, half_b, feat = features.apply_features(
        mold_result, part, cfg, progress=report
    )
    timings["features"] = time.time() - t0
    validate.assert_solid_enough(half_a, "half A (with features)")
    validate.assert_solid_enough(half_b, "half B (with features)")

    separation: dict = {}
    if verify:
        report(0.95, "verifying the mold opens")
        t0 = time.time()
        separation = validate.separation_report(half_a, half_b, part, d)
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
