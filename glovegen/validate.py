"""Solid-validity gating between pipeline stages.

Deliberately more lenient than trimesh's own ``is_volume`` / ``is_watertight``,
which reject a mesh on any single non-manifold edge. A handful of non-manifold
edges -- typically where two boolean results touch along exactly one edge --
are handled fine by manifold3d's boolean engine, so rejecting them would fail
good geometry. A genuinely *open* boundary is never tolerated: that means the
mesh has a hole and every downstream boolean will produce garbage.
"""

from __future__ import annotations

import numpy as np
import trimesh
from trimesh import grouping


class SolidError(RuntimeError):
    """A mesh is not a solid enough to continue with."""


def solid_report(mesh: trimesh.Trimesh) -> dict:
    """Cheap structural summary of a mesh."""
    edges = mesh.edges_sorted
    if len(edges) == 0:
        return {
            "faces": 0,
            "vertices": 0,
            "boundary_edges": 0,
            "nonmanifold_edges": 0,
            "components": 0,
            "volume": 0.0,
            "watertight": False,
            "winding_consistent": False,
        }
    groups = grouping.group_rows(edges, require_count=None)
    counts = np.array([len(g) for g in groups]) if len(groups) else np.zeros(0, int)
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "components": int(len(mesh.split(only_watertight=False))) if len(mesh.faces) else 0,
        "volume": float(mesh.volume),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }


def is_solid_enough(
    mesh: trimesh.Trimesh,
    *,
    max_nonmanifold_edges: int | None = None,
    min_volume: float = 1e-9,
) -> tuple[bool, str]:
    """Return ``(ok, reason)``. ``reason`` is empty when ok.

    The non-manifold budget scales with face count: these edges appear where two
    sheets of a boolean result touch, so a denser mesh has proportionally more
    of them without being any less printable. A fixed cap would pass a 10k-face
    test solid and spuriously fail a 1.4M-face mold half.
    """
    if mesh is None or len(mesh.faces) == 0:
        return False, "mesh is empty"
    if max_nonmanifold_edges is None:
        max_nonmanifold_edges = max(64, int(len(mesh.faces) * 2e-3))

    edges = mesh.edges_sorted
    groups = grouping.group_rows(edges, require_count=None)
    counts = np.array([len(g) for g in groups])
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())

    if boundary:
        return False, f"{boundary} open boundary edge(s): mesh has a hole"
    if nonmanifold > max_nonmanifold_edges:
        return False, (
            f"{nonmanifold} non-manifold edges (tolerating up to "
            f"{max_nonmanifold_edges})"
        )
    vol = float(mesh.volume)
    if not np.isfinite(vol):
        return False, "volume is not finite"
    if vol <= min_volume:
        return False, (
            f"volume is {vol:.6g}, expected > {min_volume:g} "
            "(inverted or degenerate solid?)"
        )
    return True, ""


def overlap_volume(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    """Volume shared by two solids, 0 if they are disjoint."""
    inter = trimesh.boolean.intersection([a, b], engine="manifold", check_volume=False)
    if inter is None or len(inter.faces) == 0:
        return 0.0
    return abs(float(inter.volume))


def separation_report(
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    cast: trimesh.Trimesh,
    direction: np.ndarray,
    *,
    travel: float | None = None,
    tol_fraction: float = 1.0e-6,
) -> dict:
    """Does the mold actually open? Measured, not assumed.

    Slides each half along its own pull direction and reports how much material
    it collides with. Two distinct questions:

    ``half_vs_half``
        Can the halves come apart from each other? Guaranteed by construction --
        each half lives on one side of a single-valued parting surface, and
        translating one side along the pull direction keeps it on that side --
        so a *significant* number here means a feature broke the invariant.

    ``half_vs_cast``
        How much mold has to be forced past the cast. Zero means a rigid cast
        would demold. Small-but-non-zero is the residual undercut a *flexible*
        cast deforms around, and the volume says how much deformation.

    Why the tolerance is not zero
    -----------------------------
    Half A and half B are cut by the same surface, so their parting faces are
    geometrically coincident. Intersecting million-face meshes across a
    coincident surface always leaves sub-epsilon slivers -- on a 390 mm hand
    scan, a scatter of fragments totalling ~1 mm^3, i.e. an equivalent uniform
    thickness of 2e-5 mm across the parting face. That is three orders of
    magnitude below any printer's resolution and well inside manifold3d's
    working epsilon.

    Genuine interlock is distinguished by *behaviour under travel*, not by
    magnitude: real interference persists or grows as the half moves further,
    while sliver noise falls away to nothing. So the test slides the halves a
    long way (default 10% of the part's size) where noise has decayed, and
    judges the residue relative to the mold's own volume.
    """
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    d = d / np.linalg.norm(d)
    if travel is None:
        travel = max(5.0, 0.1 * float(cast.scale))

    out: dict = {"travel_mm": round(float(travel), 3)}

    moved_a = half_a.copy()
    moved_a.apply_translation(d * travel)
    overlap = overlap_volume(moved_a, half_b)
    out["half_vs_half_mm3"] = round(overlap, 6)

    # Cast interference has to be sampled along the withdrawal path, not just at
    # its end: a half that fouls the cast for the first millimetre is clear of it
    # entirely by the time it has travelled 10% of the part's size, so a
    # single far-travel probe reports a spurious zero. Take the worst point on a
    # ladder of travels. (The authoritative swept measure is the ray-based
    # trapped-fin volume from demold analysis; this boolean probe is a cheap
    # independent cross-check of it.)
    scale = float(cast.scale)
    ladder = [0.5, 0.02 * scale, 0.06 * scale, float(travel)]
    ladder = sorted({round(t, 4) for t in ladder if t > 0})
    for name, half, sign in (("a", half_a, 1.0), ("b", half_b, -1.0)):
        worst = 0.0
        for t in ladder:
            moved = half.copy()
            moved.apply_translation(sign * d * t)
            worst = max(worst, overlap_volume(moved, cast))
        out[f"half_{name}_vs_cast_mm3"] = round(worst, 6)
    out["cast_probe_travels_mm"] = ladder

    mold_vol = abs(float(half_a.volume) + float(half_b.volume)) or 1.0
    cast_vol = abs(float(cast.volume)) or 1.0
    out["half_vs_half_fraction"] = round(overlap / mold_vol, 10)
    out["cast_interference_mm3"] = round(
        out["half_a_vs_cast_mm3"] + out["half_b_vs_cast_mm3"], 6
    )
    out["cast_interference_fraction"] = round(
        out["cast_interference_mm3"] / cast_vol, 8
    )
    out["opens"] = out["half_vs_half_fraction"] <= tol_fraction
    out["rigid_cast_demolds"] = out["cast_interference_fraction"] <= tol_fraction
    return out


def assert_solid_enough(mesh: trimesh.Trimesh, name: str, **kwargs) -> trimesh.Trimesh:
    """Raise :class:`SolidError` with an actionable message if ``mesh`` is bad.

    Called between stages so a bad intermediate state is reported where it was
    produced, instead of surfacing as an opaque boolean failure several
    expensive stages later.
    """
    ok, reason = is_solid_enough(mesh, **kwargs)
    if not ok:
        raise SolidError(f"{name}: {reason}")
    return mesh
