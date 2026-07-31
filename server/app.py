"""FastAPI backend: upload a scan, explore demoldability, build the mold."""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import trimesh
from fastapi import Body, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from glovegen import demold, viewer_format
from glovegen.config import MoldConfig

from .store import Store
from .worker import run_job

log = logging.getLogger(__name__)

STORE_ROOT = Path(os.environ.get("GLOVEGEN_STORE", "data/store")).resolve()
TTL_HOURS = float(os.environ.get("GLOVEGEN_TTL_HOURS", "24"))
MAX_UPLOAD_MB = float(os.environ.get("GLOVEGEN_MAX_UPLOAD_MB", "400"))
# One at a time: a full-resolution mold run peaks near 4 GB.
MAX_WORKERS = int(os.environ.get("GLOVEGEN_WORKERS", "1"))
SWEEP_INTERVAL_S = 900.0
STATIC_DIR = Path(__file__).parent / "static"

ALLOWED_SUFFIXES = {".stl", ".ply", ".obj", ".off", ".glb"}

store = Store(STORE_ROOT, ttl_seconds=TTL_HOURS * 3600.0)
_pool: ProcessPoolExecutor | None = None
_proxy_cache: "OrderedDict[str, trimesh.Trimesh]" = OrderedDict()
_PROXY_CACHE_MAX = 3


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    store.reap_orphans()
    store.purge_expired()
    # "spawn", not the default fork: a forked worker inherits the listening
    # socket, so a worker outliving a crashed API process would keep the port
    # bound and block the restart. Spawning also avoids inheriting embree and
    # threading state. Worker startup costs a fraction of a second, against jobs
    # that run for tens of seconds.
    _pool = ProcessPoolExecutor(
        max_workers=MAX_WORKERS, mp_context=multiprocessing.get_context("spawn")
    )
    sweeper = asyncio.create_task(_sweep_loop())
    log.info("store=%s ttl=%.1fh workers=%d", STORE_ROOT, TTL_HOURS, MAX_WORKERS)
    try:
        yield
    finally:
        sweeper.cancel()
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)


async def _sweep_loop() -> None:
    """Enforce the TTL for real, on a timer."""
    while True:
        try:
            await asyncio.sleep(SWEEP_INTERVAL_S)
            await asyncio.to_thread(store.purge_expired)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception:  # pragma: no cover
            log.exception("sweeper failed")


app = FastAPI(title="glovegen", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _submit(kind: str, mesh_id: str, config: dict | None = None) -> dict:
    if _pool is None:  # pragma: no cover
        raise HTTPException(503, "worker pool is not running")
    job = store.create_job(kind, mesh_id, config)
    _pool.submit(run_job, str(store.root), job["id"])
    return job


def _require_mesh(mesh_id: str) -> dict:
    meta = store.get_mesh(mesh_id)
    if meta is None:
        raise HTTPException(404, f"no mesh {mesh_id}")
    return meta


def _require_job(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return job


def _load_proxy(mesh_id: str) -> trimesh.Trimesh:
    """Load and cache the analysis proxy in this process.

    Heatmaps are interactive, so the mesh (and its embree BVH) has to stay
    resident between requests.
    """
    hit = _proxy_cache.get(mesh_id)
    if hit is not None:
        _proxy_cache.move_to_end(mesh_id)
        return hit
    path = store.mesh_dir(mesh_id) / "proxy.stl"
    if not path.exists():
        raise HTTPException(409, "mesh is still being prepared")
    mesh = trimesh.load(str(path), process=False, force="mesh")
    mesh.merge_vertices()
    _proxy_cache[mesh_id] = mesh
    while len(_proxy_cache) > _PROXY_CACHE_MAX:
        _proxy_cache.popitem(last=False)
    return mesh


def _safe_child(directory: Path, name: str) -> Path:
    """Resolve ``name`` inside ``directory``, refusing to escape it."""
    target = (directory / name).resolve()
    if directory.resolve() not in target.parents and target != directory.resolve():
        raise HTTPException(400, "invalid path")
    if not target.is_file():
        raise HTTPException(404, f"no file {name}")
    return target


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------


@app.post("/api/meshes")
async def upload_mesh(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            415, f"unsupported file type {suffix!r}; expected one of {sorted(ALLOWED_SUFFIXES)}"
        )
    payload = await file.read()
    if not payload:
        raise HTTPException(400, "empty upload")
    if len(payload) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            413, f"upload is {len(payload) / 1e6:.0f}MB, limit is {MAX_UPLOAD_MB:.0f}MB"
        )
    meta = store.create_mesh(file.filename or "upload.stl", payload)
    job = _submit("prepare", meta["id"])
    return {"mesh": meta, "prepare_job": job["id"]}


@app.get("/api/meshes")
def list_meshes() -> dict:
    return {"meshes": store.list_meshes()}


@app.get("/api/meshes/{mesh_id}")
def get_mesh(mesh_id: str) -> dict:
    return {"mesh": _require_mesh(mesh_id), "jobs": store.list_jobs(mesh_id)}


@app.delete("/api/meshes/{mesh_id}")
def delete_mesh(mesh_id: str) -> dict:
    _require_mesh(mesh_id)
    _proxy_cache.pop(mesh_id, None)
    return {"deleted": store.delete_mesh(mesh_id)}


@app.get("/api/meshes/{mesh_id}/viewer.bin")
def mesh_viewer(mesh_id: str):
    _require_mesh(mesh_id)
    path = store.mesh_dir(mesh_id) / "viewer.bin"
    if not path.exists():
        raise HTTPException(409, "mesh is still being prepared")
    return FileResponse(path, media_type="application/octet-stream")


@app.post("/api/meshes/{mesh_id}/heatmap")
def mesh_heatmap(mesh_id: str, payload: dict = Body(...)):
    """Per-face undercut severity for a candidate pull direction.

    Returns one byte per face, in the same order as ``viewer.bin``, so the
    client can drop it straight into a colour attribute. Summary numbers ride
    along in the ``X-Glovegen-Stats`` header.
    """
    _require_mesh(mesh_id)
    raw = payload.get("direction")
    if not raw or len(raw) != 3:
        raise HTTPException(422, "expected {'direction': [x, y, z]}")
    d = np.asarray(raw, dtype=float)
    if not np.isfinite(d).all() or np.linalg.norm(d) < 1e-9:
        raise HTTPException(422, "direction must be a non-zero finite vector")

    mesh = _load_proxy(mesh_id)
    info = demold.face_undercut(mesh, d / np.linalg.norm(d))
    body = viewer_format.encode_severity(info["severity"])

    grid = MoldConfig().demold.fine_grid
    score = demold.score_direction(mesh, d, grid)
    stats = {
        "undercut_area_fraction": round(float(info["undercut_area_fraction"]), 6),
        "undercut_face_count": int(info["undercut_face_count"]),
        "face_count": int(info["face_count"]),
        **score.as_dict(),
    }
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={"X-Glovegen-Stats": json.dumps(stats)},
    )


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


@app.post("/api/jobs")
def create_job(payload: dict = Body(...)) -> dict:
    mesh_id = payload.get("mesh_id")
    kind = payload.get("kind", "mold")
    if kind not in ("analyze", "mold"):
        raise HTTPException(422, "kind must be 'analyze' or 'mold'")
    meta = _require_mesh(mesh_id)
    if meta.get("state") != "ready":
        raise HTTPException(409, f"mesh is {meta.get('state')}, not ready")

    config = payload.get("config") or {}
    # Validate the config here so a typo fails the request, not the job.
    direction = config.pop("direction", None)
    try:
        MoldConfig.from_dict(config)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"bad config: {exc}") from exc
    if direction is not None:
        if len(direction) != 3 or not np.isfinite(np.asarray(direction, float)).all():
            raise HTTPException(422, "direction must be three finite numbers")
        config["direction"] = [float(v) for v in direction]

    return {"job": _submit(kind, mesh_id, config)}


@app.get("/api/jobs")
def list_jobs(mesh_id: str | None = None) -> dict:
    return {"jobs": store.list_jobs(mesh_id)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return {"job": _require_job(job_id)}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    _require_job(job_id)
    return {"deleted": store.delete_job(job_id)}


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str):
    _require_job(job_id)
    target = _safe_child(store.job_dir(job_id), name)
    return FileResponse(
        target, media_type="application/octet-stream", filename=target.name
    )


@app.get("/api/jobs/{job_id}/preview/{which}.bin")
def job_preview(job_id: str, which: str):
    if which not in ("half_a", "half_b", "parting"):
        raise HTTPException(404, "no such preview")
    _require_job(job_id)
    target = _safe_child(store.job_dir(job_id), f"{which}.bin")
    return FileResponse(target, media_type="application/octet-stream")


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------


@app.get("/api/status")
def status() -> dict:
    return {
        "version": app.version,
        "defaults": MoldConfig().to_dict(),
        "limits": {"max_upload_mb": MAX_UPLOAD_MB, "workers": MAX_WORKERS},
        "storage": store.usage(),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():  # pragma: no cover
        return HTMLResponse("<h1>glovegen</h1><p>static assets missing</p>", 500)
    return HTMLResponse(page.read_text())


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
