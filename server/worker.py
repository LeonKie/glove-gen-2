"""Job bodies. These run in a separate process from the API.

Mesh work is CPU-bound and memory-hungry (a 2.6M-triangle scan peaks around
3.7 GB through the boolean stages), so it must not share a process with the
event loop. Progress is reported by rewriting the job's ``job.json``: the API
process just reads the file, which means there is no IPC to get wrong and
progress survives an API restart.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import trimesh

from glovegen import demold, features, meshio, pipeline, viewer_format
from glovegen.config import MoldConfig
from glovegen.parting import PartingSurface

from .store import Store

log = logging.getLogger(__name__)

# Face budget for the mesh the browser renders and the analysis runs against.
# Both must agree, because the heatmap is one byte per face in this mesh's
# face order.
PROXY_FACES = 80_000
# Mold halves are far too dense to hand to a browser as-is.
PREVIEW_FACES = 120_000

_PROGRESS_MIN_INTERVAL = 0.35


def _progress_writer(store: Store, job_id: str):
    """Throttled progress reporter that persists to the job file."""
    last = {"t": 0.0, "pct": -1}

    def report(fraction: float, message: str = "") -> None:
        now = time.time()
        pct = int(max(0.0, min(1.0, float(fraction))) * 100)
        if pct == last["pct"] and now - last["t"] < _PROGRESS_MIN_INTERVAL:
            return
        last["t"] = now
        last["pct"] = pct
        store.update_job(
            job_id, state="running", progress=round(float(fraction), 4), message=message
        )

    return report


def _write_preview(mesh: trimesh.Trimesh, path: Path, faces: int) -> dict:
    small = meshio.decimate(mesh, faces)
    path.write_bytes(viewer_format.encode(small))
    return {"faces": int(len(small.faces)), "bytes": path.stat().st_size}


def run_job(root: str, job_id: str) -> None:
    """Entry point invoked in the worker process."""
    store = Store(Path(root))
    job = store.get_job(job_id)
    if job is None:
        return
    import os

    store.update_job(
        job_id, state="running", pid=os.getpid(), started=time.time(), message="starting"
    )
    try:
        kind = job.get("kind")
        if kind == "prepare":
            result = _do_prepare(store, job)
        elif kind == "analyze":
            result = _do_analyze(store, job)
        elif kind == "mold":
            result = _do_mold(store, job)
        elif kind == "features":
            result = _do_features(store, job)
        else:
            raise ValueError(f"unknown job kind {kind!r}")
        store.update_job(
            job_id,
            state="done",
            progress=1.0,
            message="done",
            finished=time.time(),
            result=result,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        log.exception("job %s failed", job_id)
        store.update_job(
            job_id,
            state="failed",
            message=str(exc),
            error=traceback.format_exc(limit=8),
            finished=time.time(),
        )
        if job.get("kind") == "prepare":
            store.update_mesh(job["mesh_id"], state="failed", error=str(exc))


# --------------------------------------------------------------------------


def _do_prepare(store: Store, job: dict) -> dict:
    """Clean an upload and build the derived meshes everything else uses."""
    report = _progress_writer(store, job["id"])
    mesh_id = job["mesh_id"]
    meta = store.get_mesh(mesh_id)
    if meta is None:
        raise FileNotFoundError(f"mesh {mesh_id} is gone")
    d = store.mesh_dir(mesh_id)

    report(0.05, "loading mesh")
    part = meshio.load_mesh(d / meta["original"])

    report(0.5, "writing cleaned part")
    meshio.export(part, d / "part.stl")

    report(0.7, f"building {PROXY_FACES // 1000}k-face proxy")
    proxy = meshio.decimate(part, PROXY_FACES)
    meshio.export(proxy, d / "proxy.stl")

    report(0.9, "encoding viewer geometry")
    (d / "viewer.bin").write_bytes(viewer_format.encode(proxy))

    stats = meshio.mesh_stats(part)
    stats["proxy_faces"] = int(len(proxy.faces))
    store.update_mesh(mesh_id, state="ready", stats=stats, error=None)
    return {"stats": stats}


def _do_analyze(store: Store, job: dict) -> dict:
    """Search for the best pull direction, on the proxy."""
    report = _progress_writer(store, job["id"])
    d = store.mesh_dir(job["mesh_id"])
    cfg = MoldConfig.from_dict(job.get("config") or {})

    report(0.05, "loading proxy")
    proxy = trimesh.load(str(d / "proxy.stl"), process=False, force="mesh")
    proxy.merge_vertices()

    direction, score, ranked = demold.suggest_direction(
        proxy,
        cfg.demold,
        block_margin=cfg.block_margin,
        progress=lambda f, msg: report(0.05 + 0.9 * f, msg),
    )
    return {
        "direction": [float(v) for v in direction],
        "best": score.as_dict(),
        "alternatives": [s.as_dict() for s in ranked[1:6]],
    }


def _do_mold(store: Store, job: dict) -> dict:
    """Build the mold at full input resolution and write the printable parts."""
    report = _progress_writer(store, job["id"])
    mesh_dir = store.mesh_dir(job["mesh_id"])
    out_dir = store.job_dir(job["id"])
    config = dict(job.get("config") or {})
    direction = config.pop("direction", None)
    # The separation check is a handful of booleans over million-face meshes, so
    # it is the single most expensive stage on a full-resolution run. Worth it by
    # default; skippable when iterating.
    verify = bool(config.pop("verify", True))
    cfg = MoldConfig.from_dict(config)

    result = pipeline.run(
        mesh_dir / "part.stl",
        cfg,
        direction=direction,
        verify=verify,
        progress=lambda f, msg: report(0.02 + 0.88 * f, msg),
    )

    report(0.91, "writing parts")
    written = result.write(out_dir, extras=True)

    report(0.93, "caching the mold for feature edits")
    _write_base(store, job, result)

    report(0.97, "encoding previews")
    previews = {
        "half_a": _write_preview(result.half_a, out_dir / "half_a.bin", PREVIEW_FACES),
        "half_b": _write_preview(result.half_b, out_dir / "half_b.bin", PREVIEW_FACES),
    }
    if result.core is not None:
        previews["core"] = _write_preview(
            result.core, out_dir / "core.bin", PREVIEW_FACES
        )
    surface = result.mold_result.surface.surface_mesh()
    (out_dir / "parting.bin").write_bytes(
        viewer_format.encode(meshio.decimate(surface, PREVIEW_FACES))
    )

    report_data = result.report()
    (out_dir / "report.json").write_text(json.dumps(report_data, indent=2))

    store.update_job(job["id"], parts=_parts_manifest(out_dir, written))
    return {
        "report": report_data,
        "previews": previews,
        # The editable form of the features, and where to re-cut them from.
        "plan": result.feature_plan.as_dict(),
        "base_job": job["id"],
    }


# --------------------------------------------------------------------------
# feature edits
# --------------------------------------------------------------------------


def _write_base(store: Store, job: dict, result: pipeline.PipelineResult) -> None:
    """Cache what a later feature edit needs, and nothing else.

    The halves *before* features, the parting surface and the block's bounds.
    Together they are a few percent of a run's cost to reproduce and megabytes
    on disk, against tens of seconds of booleans -- and they are exactly the
    part of the mold that changing a knob's radius cannot invalidate.
    """
    out_dir = store.job_dir(job["id"])
    mold_result = result.mold_result
    meshio.save_npz(mold_result.half_a, out_dir / "base_half_a.npz")
    meshio.save_npz(mold_result.half_b, out_dir / "base_half_b.npz")
    mold_result.surface.save(out_dir / "parting.npz")

    # The cavity is usually the part itself, which the mesh directory already
    # holds; only an offset or decimated cavity is worth another copy.
    cavity = "mesh"
    if mold_result.cavity is not None and mold_result.cavity is not result.part:
        meshio.save_npz(mold_result.cavity, out_dir / "cavity.npz")
        cavity = "cavity.npz"

    # The core is not cached. Nothing about it depends on the feature plan -- no
    # feature is cut into it and it cuts nothing into the halves -- so a feature
    # edit copies the mold job's finished core across rather than rebuilding or
    # re-cutting it.
    (out_dir / "context.json").write_text(
        json.dumps(
            {
                "base_job": job["id"],
                "mesh_id": job["mesh_id"],
                "direction": [float(v) for v in result.direction],
                "local_bounds": [
                    [float(v) for v in row] for row in mold_result.local_bounds
                ],
                "cavity": cavity,
            },
            indent=2,
        )
    )


def _load_context(store: Store, base_job_id: str) -> features.FeatureContext:
    """Rebuild a feature context from a mold job's cached base state."""
    base_dir = store.job_dir(base_job_id)
    raw = base_dir / "context.json"
    if not raw.exists():
        raise FileNotFoundError(
            f"job {base_job_id} has no cached mold to re-cut features into"
        )
    ctx_data = json.loads(raw.read_text())

    if ctx_data["cavity"] == "mesh":
        cavity_path = store.mesh_dir(ctx_data["mesh_id"]) / "part.stl"
        if not cavity_path.exists():
            raise FileNotFoundError("the scan this mold was built from is gone")
        cavity = meshio.load_mesh(cavity_path)
    else:
        cavity = meshio.load_npz(base_dir / ctx_data["cavity"])

    return features.FeatureContext(
        surface=PartingSurface.load(base_dir / "parting.npz"),
        local_bounds=np.asarray(ctx_data["local_bounds"], dtype=float),
        cavity=cavity,
    )


def _do_features(store: Store, job: dict) -> dict:
    """Re-cut the knobs and holes into an already-built mold.

    Everything before the split is reused, so an edit costs the features and
    the separation check, not the whole pipeline.
    """
    report = _progress_writer(store, job["id"])
    out_dir = store.job_dir(job["id"])
    config = dict(job.get("config") or {})
    base_job = config["base_job"]
    verify = bool(config.get("verify", True))

    report(0.03, "loading the built mold")
    ctx = _load_context(store, base_job)
    base_dir = store.job_dir(base_job)
    half_a = meshio.load_npz(base_dir / "base_half_a.npz")
    half_b = meshio.load_npz(base_dir / "base_half_b.npz")

    result = pipeline.apply_feature_plan(
        half_a,
        half_b,
        ctx,
        config.get("plan") or {},
        verify=verify,
        progress=lambda f, msg: report(0.06 + 0.85 * f, msg),
    )

    report(0.93, "writing parts")
    written = result.write(out_dir)
    # The parting surface did not change, and neither did the core -- no feature
    # is cut into it and it cuts nothing into the halves. Carry the base job's
    # copies over rather than regenerating them; this job's directory then
    # stands alone.
    for name in (
        "parting_surface.stl",
        "parting.bin",
        "core.stl",
        "core_a.stl",
        "core_b.stl",
        "core.bin",
        "glove_preview.stl",
    ):
        src = base_dir / name
        if src.exists():
            shutil.copyfile(src, out_dir / name)
    # The manifest looks the core pieces up by key rather than off disk.
    for name in ("core", "core_a", "core_b"):
        if (out_dir / f"{name}.stl").exists():
            written[name] = str(out_dir / f"{name}.stl")

    report(0.97, "encoding previews")
    previews = {
        "half_a": _write_preview(result.half_a, out_dir / "half_a.bin", PREVIEW_FACES),
        "half_b": _write_preview(result.half_b, out_dir / "half_b.bin", PREVIEW_FACES),
    }
    if (out_dir / "core.bin").exists():
        previews["core"] = {"copied": True}

    report_data = result.report()
    report_data["base_job"] = base_job
    (out_dir / "report.json").write_text(json.dumps(report_data, indent=2))

    store.update_job(job["id"], parts=_parts_manifest(out_dir, written))
    return {
        "report": report_data,
        "previews": previews,
        "plan": result.plan.as_dict(),
        "base_job": base_job,
    }


def _parts_manifest(out_dir: Path, written: dict) -> dict:
    """The downloadable files a finished job offers."""
    parts = {
        "mold_half_a.stl": {
            "label": "Mold half A (pulls along +d)",
            "bytes": Path(written["half_a"]).stat().st_size,
        },
        "mold_half_b.stl": {
            "label": "Mold half B (pulls along -d)",
            "bytes": Path(written["half_b"]).stat().st_size,
        },
        "report.json": {
            "label": "Report",
            "bytes": (out_dir / "report.json").stat().st_size,
        },
    }
    for name, label in (
        ("core", "Core (the glove's inner form)"),
        ("core_a", "Core half A (pulls along +d)"),
        ("core_b", "Core half B (pulls along -d)"),
    ):
        if name in written:
            parts[f"{name}.stl"] = {
                "label": label,
                "bytes": Path(written[name]).stat().st_size,
            }
    for extra, label in (
        ("glove_preview.stl", "The glove that will be cast"),
        ("parting_surface.stl", "Parting surface"),
        ("mold_uncut.stl", "Mold before splitting"),
    ):
        p = out_dir / extra
        if p.exists():
            parts[extra] = {"label": label, "bytes": p.stat().st_size}
    return parts
