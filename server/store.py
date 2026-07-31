"""Disk-backed store for uploaded meshes and mold jobs.

v1 kept all mesh and job state in an in-process dict with no eviction, an
advertised-but-unenforced TTL, and no multi-process story, so restarting the
backend lost every upload and in-flight job. Here the filesystem *is* the
database:

* every mesh and job is a directory with an atomically-rewritten ``meta.json``;
* the API process holds no authoritative state, so it can restart freely and
  heavy geometry can run in a separate process that reports progress by writing
  to the job file;
* the TTL is enforced by a sweeper, not just documented;
* jobs left ``running`` by a crash are reaped into ``interrupted`` on startup
  rather than hanging forever.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

MESH_STATES = ("preparing", "ready", "failed")
JOB_STATES = ("queued", "running", "done", "failed", "interrupted")


def _now() -> float:
    return time.time()


def write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON so a reader never observes a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


@dataclass
class Store:
    root: Path
    ttl_seconds: float = 24 * 3600.0

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        (self.root / "meshes").mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)

    # -- paths -----------------------------------------------------------

    def mesh_dir(self, mesh_id: str) -> Path:
        return self.root / "meshes" / _safe_id(mesh_id)

    def job_dir(self, job_id: str) -> Path:
        return self.root / "jobs" / _safe_id(job_id)

    # -- meshes ----------------------------------------------------------

    def create_mesh(self, filename: str, payload: bytes) -> dict:
        mesh_id = uuid.uuid4().hex[:16]
        d = self.mesh_dir(mesh_id)
        d.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix.lower() or ".stl"
        original = d / f"original{suffix}"
        original.write_bytes(payload)
        meta = {
            "id": mesh_id,
            "name": Path(filename).name,
            "created": _now(),
            "state": "preparing",
            "bytes": len(payload),
            "original": original.name,
            "stats": None,
            "error": None,
        }
        self.save_mesh(meta)
        return meta

    def save_mesh(self, meta: dict) -> None:
        write_json_atomic(self.mesh_dir(meta["id"]) / "meta.json", meta)

    def get_mesh(self, mesh_id: str) -> dict | None:
        return read_json(self.mesh_dir(mesh_id) / "meta.json")

    def update_mesh(self, mesh_id: str, **fields: Any) -> dict | None:
        meta = self.get_mesh(mesh_id)
        if meta is None:
            return None
        meta.update(fields)
        self.save_mesh(meta)
        return meta

    def list_meshes(self) -> list[dict]:
        out = [
            m
            for p in sorted((self.root / "meshes").iterdir())
            if p.is_dir() and (m := read_json(p / "meta.json")) is not None
        ]
        return sorted(out, key=lambda m: m.get("created", 0), reverse=True)

    def delete_mesh(self, mesh_id: str) -> bool:
        d = self.mesh_dir(mesh_id)
        if not d.exists():
            return False
        shutil.rmtree(d, ignore_errors=True)
        for job in self.list_jobs():
            if job.get("mesh_id") == mesh_id:
                self.delete_job(job["id"])
        return True

    # -- jobs ------------------------------------------------------------

    def create_job(self, kind: str, mesh_id: str, config: dict | None = None) -> dict:
        job_id = uuid.uuid4().hex[:16]
        d = self.job_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        job = {
            "id": job_id,
            "kind": kind,
            "mesh_id": mesh_id,
            "config": config or {},
            "state": "queued",
            "progress": 0.0,
            "message": "queued",
            "created": _now(),
            "started": None,
            "finished": None,
            "pid": None,
            "result": None,
            "error": None,
            "parts": {},
        }
        self.save_job(job)
        return job

    def save_job(self, job: dict) -> None:
        write_json_atomic(self.job_dir(job["id"]) / "job.json", job)

    def get_job(self, job_id: str) -> dict | None:
        return read_json(self.job_dir(job_id) / "job.json")

    def update_job(self, job_id: str, **fields: Any) -> dict | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        job.update(fields)
        self.save_job(job)
        return job

    def list_jobs(self, mesh_id: str | None = None) -> list[dict]:
        out = [
            j
            for p in sorted((self.root / "jobs").iterdir())
            if p.is_dir() and (j := read_json(p / "job.json")) is not None
        ]
        if mesh_id:
            out = [j for j in out if j.get("mesh_id") == mesh_id]
        return sorted(out, key=lambda j: j.get("created", 0), reverse=True)

    def delete_job(self, job_id: str) -> bool:
        d = self.job_dir(job_id)
        if not d.exists():
            return False
        shutil.rmtree(d, ignore_errors=True)
        return True

    # -- housekeeping ----------------------------------------------------

    def reap_orphans(self) -> int:
        """Mark jobs whose worker process is gone as interrupted.

        Called on startup: a job recorded as running by a process that no longer
        exists will never progress, and saying so beats a spinner that never
        stops.
        """
        n = 0
        for job in self.list_jobs():
            if job.get("state") not in ("queued", "running"):
                continue
            if _pid_alive(job.get("pid")):
                continue
            self.update_job(
                job["id"],
                state="interrupted",
                message="worker process is gone (server restart?)",
                finished=_now(),
            )
            n += 1
        for mesh in self.list_meshes():
            if mesh.get("state") == "preparing":
                self.update_mesh(
                    mesh["id"], state="failed", error="preparation interrupted"
                )
                n += 1
        if n:
            log.warning("reaped %d interrupted job(s)/upload(s)", n)
        return n

    def purge_expired(self, ttl_seconds: float | None = None) -> int:
        """Delete meshes and jobs older than the TTL. Actually enforced."""
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        if ttl <= 0:
            return 0
        cutoff = _now() - ttl
        n = 0
        for job in self.list_jobs():
            if job.get("created", 0) < cutoff:
                n += int(self.delete_job(job["id"]))
        for mesh in self.list_meshes():
            if mesh.get("created", 0) < cutoff:
                n += int(self.delete_mesh(mesh["id"]))
        if n:
            log.info("purged %d expired record(s)", n)
        return n

    def usage(self) -> dict:
        def size(paths: Iterable[Path]) -> int:
            return sum(f.stat().st_size for f in paths if f.is_file())

        return {
            "root": str(self.root),
            "meshes": len(self.list_meshes()),
            "jobs": len(self.list_jobs()),
            "bytes": size(self.root.rglob("*")),
            "ttl_hours": round(self.ttl_seconds / 3600.0, 2),
        }


def _safe_id(value: str) -> str:
    """Reject anything that could escape the store directory."""
    v = str(value)
    if not v or not all(c.isalnum() or c in "-_" for c in v):
        raise ValueError(f"invalid id {value!r}")
    return v
