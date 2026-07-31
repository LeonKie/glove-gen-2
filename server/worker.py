"""Job bodies. These run in a separate process from the API.

Mesh work is CPU-bound and memory-hungry (a 2.6M-triangle scan peaks around
3.7 GB through the boolean stages), so it must not share a process with the
event loop. Progress is reported by rewriting the job's ``job.json``: the API
process just reads the file, which means there is no IPC to get wrong and
progress survives an API restart.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

import numpy as np
import trimesh

from glovegen import demold, meshio, pipeline, viewer_format
from glovegen.config import MoldConfig

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
        progress=lambda f, msg: report(0.02 + 0.9 * f, msg),
    )

    report(0.94, "writing parts")
    written = result.write(out_dir, extras=True)

    report(0.97, "encoding previews")
    previews = {
        "half_a": _write_preview(result.half_a, out_dir / "half_a.bin", PREVIEW_FACES),
        "half_b": _write_preview(result.half_b, out_dir / "half_b.bin", PREVIEW_FACES),
    }
    surface = result.mold_result.surface.surface_mesh()
    (out_dir / "parting.bin").write_bytes(
        viewer_format.encode(meshio.decimate(surface, PREVIEW_FACES))
    )

    report_data = result.report()
    (out_dir / "report.json").write_text(
        __import__("json").dumps(report_data, indent=2)
    )

    parts = {
        "mold_half_a.stl": {
            "label": "Mold half A (pulls along +d)",
            "bytes": Path(written["half_a"]).stat().st_size,
        },
        "mold_half_b.stl": {
            "label": "Mold half B (pulls along -d)",
            "bytes": Path(written["half_b"]).stat().st_size,
        },
        "report.json": {"label": "Report", "bytes": (out_dir / "report.json").stat().st_size},
    }
    for extra, label in (
        ("parting_surface.stl", "Parting surface"),
        ("mold_uncut.stl", "Mold before splitting"),
    ):
        p = out_dir / extra
        if p.exists():
            parts[extra] = {"label": label, "bytes": p.stat().st_size}

    store.update_job(job["id"], parts=parts)
    return {"report": report_data, "previews": previews}
