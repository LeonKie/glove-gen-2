"""Mesh loading, cleanup and decimation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

log = logging.getLogger(__name__)

_PROXY_CACHE: dict[tuple[int, int], trimesh.Trimesh] = {}


def load_mesh(path: str | Path, *, keep_largest: bool = True) -> trimesh.Trimesh:
    """Load a mesh and clean it up just enough to be a solid.

    High-detail scans are usually already closed manifolds; this does the
    cheap, safe repairs (vertex merge, degenerate/duplicate face removal,
    consistent winding) and drops stray shells.
    """
    loaded = trimesh.load(str(path), process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):  # pragma: no cover - force="mesh" handles it
        loaded = trimesh.util.concatenate(list(loaded.dump()))
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{path}: no triangles found")

    mesh = loaded
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    if keep_largest:
        mesh = _largest_component(mesh)

    if not mesh.is_winding_consistent:
        log.warning("inconsistent winding, repairing")
        trimesh.repair.fix_winding(mesh)
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()

    return mesh


def _largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Keep only the biggest connected shell, if there is more than one."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    parts = sorted(parts, key=lambda p: len(p.faces), reverse=True)
    dropped = sum(len(p.faces) for p in parts[1:])
    log.warning(
        "input has %d disconnected shells; keeping the largest "
        "(%d faces) and dropping %d faces",
        len(parts),
        len(parts[0].faces),
        dropped,
    )
    return parts[0]


def decimate(mesh: trimesh.Trimesh, face_count: int) -> trimesh.Trimesh:
    """Decimate to roughly ``face_count`` faces, preserving watertightness."""
    if face_count <= 0 or len(mesh.faces) <= face_count:
        return mesh
    out = mesh.simplify_quadric_decimation(face_count=int(face_count))
    out.merge_vertices()
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    return out


def proxy(mesh: trimesh.Trimesh, face_count: int) -> trimesh.Trimesh:
    """Cached decimated stand-in, for ray queries where exactness is not needed.

    Undercut analysis casts millions of rays; running them against a proxy
    instead of the full-resolution scan is the single biggest performance win
    in the pipeline, and the analysis answers are insensitive to it because
    they are integrals over large regions, not per-triangle facts.
    """
    if len(mesh.faces) <= face_count:
        return mesh
    key = (id(mesh), int(face_count))
    hit = _PROXY_CACHE.get(key)
    if hit is None:
        hit = decimate(mesh, face_count)
        log.info("built %d-face proxy from %d faces", len(hit.faces), len(mesh.faces))
        _PROXY_CACHE[key] = hit
    return hit


def convex_hull(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Convex hull, cached on the mesh object."""
    cached = getattr(mesh, "_gg_hull", None)
    if cached is None:
        cached = mesh.convex_hull
        try:
            mesh._gg_hull = cached
        except Exception:  # pragma: no cover - read-only mesh
            pass
    return cached


def export(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))
    return path


def mesh_stats(mesh: trimesh.Trimesh) -> dict:
    """Summary of a mesh, including *why* it might not be strictly watertight.

    ``watertight`` alone is misleading here. manifold3d legitimately emits
    solids with a few position-coincident vertices, and re-welding those on
    reload turns them into non-manifold edges -- so a perfectly closed,
    printable solid can report ``watertight: false``. ``boundary_edges`` is the
    number that actually matters: zero means no holes.
    """
    from .validate import solid_report

    rep = solid_report(mesh)
    ext = mesh.extents
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "extents_mm": [round(float(v), 3) for v in ext],
        "volume_cm3": round(float(mesh.volume) / 1000.0, 3),
        "closed": rep["boundary_edges"] == 0,
        "boundary_edges": rep["boundary_edges"],
        "nonmanifold_edges": rep["nonmanifold_edges"],
        "watertight": bool(mesh.is_watertight),
        "bounds": [[round(float(v), 3) for v in row] for row in mesh.bounds],
    }


def as_float32_arrays(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Flat vertex/index arrays, for shipping to a viewer."""
    return (
        np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        np.ascontiguousarray(mesh.faces, dtype=np.uint32),
    )
