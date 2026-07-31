"""Demold analysis: pull-direction search and undercut heatmaps.

The physical model
-----------------
A 2-part mold pulls one half along ``+d`` and the other along ``-d``. A face is
*not* an undercut merely because its normal points against ``d`` -- it belongs
to the other half, which releases it fine. A true undercut is geometry that
neither half can reach without passing through the cast.

Both facts we need come from one cheap query: cast a single ray per grid column
parallel to ``d`` and look at the sorted crossings.

For a column with the expected 2 crossings the part occupies one interval, mold
material sits above and below it, and both blocks lift off freely.

For a column with 4+ crossings (a finger gap, say) the part occupies two
intervals and there is mold material *between* them -- a fin. That fin has cast
material both above and below it, so it cannot leave along either ``+d`` or
``-d``, no matter where the parting surface is placed. So:

    trapped fin volume = sum of interior gap lengths x cell area

is a direction-quality metric that is independent of the parting surface, has
units of volume (it is literally how much mold has to be forced past the cast),
and costs one ray per column. That is the objective the direction search
minimises.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import trimesh

from . import meshio
from .config import DemoldConfig
from .frame import Frame, fibonacci_hemisphere, unit

log = logging.getLogger(__name__)


@dataclass
class ColumnCast:
    """Result of casting one ray per grid column along a direction."""

    frame: Frame
    cell: float
    nx: int
    ny: int
    lo: np.ndarray  # local-frame min corner of the grid (x, y, z)
    z_start: float  # local z the rays start from

    # Ragged crossing data, flattened and grouped by column.
    t: np.ndarray  # (nhit,) distance from z_start, sorted within each column
    counts: np.ndarray  # (ncol,) crossings per column
    starts: np.ndarray  # (ncol,) offset of each column's block in `t`

    odd_columns: int = 0  # columns with an odd crossing count (grazing hits)

    @property
    def ncol(self) -> int:
        return self.nx * self.ny

    @property
    def cell_area(self) -> float:
        return self.cell * self.cell

    def segment_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-segment lengths between consecutive crossings.

        Returns ``(lengths, column, is_material)``. Segment ``k`` of a column
        spans crossings ``k -> k+1``; even ``k`` is inside the part, odd ``k``
        is an interior void (a mold fin).
        """
        if len(self.t) == 0:
            empty_f = np.zeros(0, dtype=np.float64)
            empty_i = np.zeros(0, dtype=np.int64)
            return empty_f, empty_i, np.zeros(0, dtype=bool)

        col = np.repeat(np.arange(self.ncol), self.counts)
        local = np.arange(len(self.t)) - np.repeat(self.starts, self.counts)
        # A segment exists where the next crossing belongs to the same column.
        has_next = local < (np.repeat(self.counts, self.counts) - 1)
        idx = np.flatnonzero(has_next)
        lengths = self.t[idx + 1] - self.t[idx]
        return lengths, col[idx], (local[idx] % 2 == 0)

    def local_z(self, t: np.ndarray) -> np.ndarray:
        """Convert a distance-along-ray into a local-frame z coordinate."""
        return self.z_start + t

    def thickest_material_interval(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The fattest slab of part material in each column.

        Returns ``(columns, z_lo, z_hi)`` in local-frame z, one entry per column
        that hit the part. Putting the parting surface in the *thickest* slab
        gives each mold half the widest, most robust mating face, and keeps the
        surface away from the cavity wall where it would leave slivers.
        """
        empty = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0))
        if len(self.t) == 0:
            return empty

        col = np.repeat(np.arange(self.ncol), self.counts)
        local = np.arange(len(self.t)) - np.repeat(self.starts, self.counts)
        cnt = np.repeat(self.counts, self.counts)
        is_material_seg = (local % 2 == 0) & (local < cnt - 1)
        idx = np.flatnonzero(is_material_seg)
        if len(idx) == 0:
            return empty

        length = self.t[idx + 1] - self.t[idx]
        c = col[idx]
        # Sort by (column, length) so the last row of each column group is its
        # thickest slab.
        order = np.lexsort((length, c))
        c_s = c[order]
        idx_s = idx[order]
        last = np.ones(len(c_s), dtype=bool)
        last[:-1] = c_s[:-1] != c_s[1:]
        sel = idx_s[last]
        return c_s[last], self.local_z(self.t[sel]), self.local_z(self.t[sel + 1])


@dataclass
class DirectionScore:
    """Quality of one candidate pull direction."""

    direction: np.ndarray
    trapped_volume: float  # mm^3 of mold that cannot leave along +/-d
    part_volume: float  # mm^3 of part seen by the ray grid
    undercut_fraction: float  # trapped / part
    multi_fraction: float  # fraction of hit columns with >2 crossings
    block_volume: float  # mm^3 of the bounding block for this orientation
    hit_columns: int = 0
    odd_columns: int = 0
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "direction": [round(float(v), 6) for v in self.direction],
            "trapped_volume_cm3": round(self.trapped_volume / 1000.0, 4),
            "part_volume_cm3": round(self.part_volume / 1000.0, 4),
            "undercut_fraction": round(float(self.undercut_fraction), 6),
            "undercut_percent": round(float(self.undercut_fraction) * 100.0, 4),
            "multi_fraction": round(float(self.multi_fraction), 6),
            "block_volume_cm3": round(self.block_volume / 1000.0, 3),
            "hit_columns": int(self.hit_columns),
            **self.extras,
        }


# --------------------------------------------------------------------------
# Column casting
# --------------------------------------------------------------------------


def cast_grid(
    mesh: trimesh.Trimesh,
    frame: Frame,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    z_start: float | None = None,
    cell: float | None = None,
    pad: float = 1.0,
) -> ColumnCast:
    """Cast one ray per (x, y) in the local-frame grid ``xs`` x ``ys``.

    Columns are indexed ``i * len(ys) + j``, i.e. C order over ``(xs, ys)``.
    """
    xs = np.asarray(xs, dtype=np.float64).reshape(-1)
    ys = np.asarray(ys, dtype=np.float64).reshape(-1)
    nx, ny = len(xs), len(ys)

    hull_local = frame.to_local(meshio.convex_hull(mesh).vertices)
    lo_hull = hull_local.min(axis=0)
    if z_start is None:
        z_start = float(lo_hull[2] - pad)
    if cell is None:
        cell = float(xs[1] - xs[0]) if nx > 1 else float(ys[1] - ys[0]) if ny > 1 else 1.0

    lo = np.array([xs[0], ys[0], lo_hull[2]], dtype=np.float64)

    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    origins_local = np.column_stack(
        [gx.ravel(), gy.ravel(), np.full(gx.size, float(z_start))]
    )
    origins = frame.to_world(origins_local)
    dirs = np.broadcast_to(frame.direction, origins.shape)

    loc, index_ray, _ = mesh.ray.intersects_location(
        origins, dirs, multiple_hits=True
    )

    ncol = nx * ny
    if len(loc) == 0:
        return ColumnCast(
            frame=frame,
            cell=cell,
            nx=nx,
            ny=ny,
            lo=lo,
            z_start=z_start,
            t=np.zeros(0),
            counts=np.zeros(ncol, dtype=np.int64),
            starts=np.zeros(ncol, dtype=np.int64),
        )

    t = (loc - origins[index_ray]) @ frame.direction
    order = np.lexsort((t, index_ray))
    t = np.ascontiguousarray(t[order])
    index_ray = np.ascontiguousarray(index_ray[order])

    counts = np.bincount(index_ray, minlength=ncol).astype(np.int64)

    # Columns with an odd number of crossings are numerically degenerate (the
    # ray grazed an edge or a coplanar facet). They cannot be interpreted as
    # material intervals, so drop them rather than let them corrupt the
    # volume sums.
    odd = counts % 2 == 1
    odd_columns = int(odd.sum())
    if odd_columns:
        keep = ~odd[index_ray]
        t = t[keep]
        index_ray = index_ray[keep]
        counts = np.bincount(index_ray, minlength=ncol).astype(np.int64)

    starts = np.zeros(ncol, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])

    return ColumnCast(
        frame=frame,
        cell=cell,
        nx=nx,
        ny=ny,
        lo=lo,
        z_start=z_start,
        t=t,
        counts=counts,
        starts=starts,
        odd_columns=odd_columns,
    )


def cast_columns(
    mesh: trimesh.Trimesh,
    direction,
    grid: int,
    *,
    pad: float = 1.0,
    frame: Frame | None = None,
) -> ColumnCast:
    """Cast one ray per grid column, parallel to ``direction``, over the mesh.

    ``grid`` is the number of cells along the longer footprint axis; the grid
    covers the mesh's footprint and samples at cell centres.
    """
    frame = frame or Frame.from_direction(direction)
    hull_local = frame.to_local(meshio.convex_hull(mesh).vertices)
    lo = hull_local.min(axis=0)
    hi = hull_local.max(axis=0)

    span_xy = np.maximum(hi[:2] - lo[:2], 1e-9)
    cell = float(span_xy.max()) / max(1, int(grid))
    nx = max(1, int(np.ceil(span_xy[0] / cell)))
    ny = max(1, int(np.ceil(span_xy[1] / cell)))

    xs = lo[0] + (np.arange(nx) + 0.5) * cell
    ys = lo[1] + (np.arange(ny) + 0.5) * cell
    return cast_grid(
        mesh, frame, xs, ys, z_start=float(lo[2] - pad), cell=cell, pad=pad
    )


def score_columns(cast: ColumnCast, *, block_margin: float = 0.0) -> DirectionScore:
    """Turn a column cast into a direction-quality score."""
    lengths, _, is_material = cast.segment_table()
    area = cast.cell_area

    part_volume = float(lengths[is_material].sum() * area) if len(lengths) else 0.0
    trapped = float(lengths[~is_material].sum() * area) if len(lengths) else 0.0

    hit = cast.counts > 0
    hit_columns = int(hit.sum())
    multi = int((cast.counts > 2).sum())
    multi_fraction = float(multi / hit_columns) if hit_columns else 0.0

    # Bounding block for this orientation, from the rotated hull extents.
    if len(cast.t):
        span_z = float(cast.t.max() - cast.t.min())
    else:
        span_z = 0.0
    ext = np.array(
        [
            cast.nx * cast.cell + 2 * block_margin,
            cast.ny * cast.cell + 2 * block_margin,
            span_z + 2 * block_margin,
        ]
    )
    block_volume = float(np.prod(ext))

    return DirectionScore(
        direction=cast.frame.direction.copy(),
        trapped_volume=trapped,
        part_volume=part_volume,
        undercut_fraction=(trapped / part_volume) if part_volume > 0 else 0.0,
        multi_fraction=multi_fraction,
        block_volume=block_volume,
        hit_columns=hit_columns,
        odd_columns=cast.odd_columns,
    )


def score_direction(
    mesh: trimesh.Trimesh,
    direction,
    grid: int,
    *,
    block_margin: float = 0.0,
) -> DirectionScore:
    return score_columns(
        cast_columns(mesh, direction, grid), block_margin=block_margin
    )


# --------------------------------------------------------------------------
# Direction search
# --------------------------------------------------------------------------


def _rank(scores: list[DirectionScore], tie_tol: float, size_weight: float) -> list[int]:
    """Order candidates best-first.

    Primary key is undercut fraction, quantised so that near-ties are treated
    as ties. A flattish organic shape can have dozens of directions genuinely
    tied at ~0% undercut, and picking blindly among them produces a needlessly
    convoluted parting surface -- so ties break on how complicated the parting
    surface will have to be (``multi_fraction``), then on how much material the
    resulting block costs.
    """
    if not scores:
        return []
    uc = np.array([s.undercut_fraction for s in scores])
    multi = np.array([s.multi_fraction for s in scores])
    block = np.array([s.block_volume for s in scores])
    block = block / max(block.min(), 1e-9)

    quant = np.round(uc / max(tie_tol, 1e-12)).astype(np.int64)
    secondary = multi + size_weight * (block - 1.0)
    return list(np.lexsort((secondary, quant)))


def suggest_direction(
    mesh: trimesh.Trimesh,
    cfg: DemoldConfig | None = None,
    *,
    block_margin: float = 0.0,
    progress=None,
) -> tuple[np.ndarray, DirectionScore, list[DirectionScore]]:
    """Search for the pull direction that minimises trapped mold volume.

    Two passes: a broad sweep on a coarse grid, then a finer sweep restricted to
    the neighbourhood of the best coarse candidates on a finer grid.
    """
    cfg = cfg or DemoldConfig()
    report = progress or (lambda *a, **k: None)

    if cfg.forced_direction is not None:
        d = unit(cfg.forced_direction)
        score = score_direction(mesh, d, cfg.fine_grid, block_margin=block_margin)
        return d, score, [score]

    small = meshio.proxy(mesh, cfg.proxy_faces)

    coarse_dirs = fibonacci_hemisphere(cfg.coarse_dirs)
    report(0.0, f"coarse sweep: {len(coarse_dirs)} directions")
    coarse: list[DirectionScore] = []
    for i, d in enumerate(coarse_dirs):
        coarse.append(
            score_direction(small, d, cfg.coarse_grid, block_margin=block_margin)
        )
        if i % 20 == 0:
            report(0.5 * (i + 1) / len(coarse_dirs), f"coarse sweep {i + 1}/{len(coarse_dirs)}")

    order = _rank(coarse, cfg.tie_tol, cfg.size_weight)
    best_coarse = [coarse[i] for i in order[:5]]
    log.info(
        "coarse best: undercut=%.4f%% dir=%s",
        best_coarse[0].undercut_fraction * 100,
        np.round(best_coarse[0].direction, 3),
    )

    # Fine sweep, restricted to the neighbourhood of the coarse leaders.
    anchors = np.array([s.direction for s in best_coarse])
    fine_dirs = fibonacci_hemisphere(cfg.fine_dirs)
    # |dot| because d and -d are the same split.
    near = (np.abs(fine_dirs @ anchors.T) >= cfg.fine_cos_window).any(axis=1)
    fine_dirs = fine_dirs[near]
    if len(fine_dirs) == 0:
        fine_dirs = anchors

    report(0.5, f"fine sweep: {len(fine_dirs)} directions")
    fine: list[DirectionScore] = []
    for i, d in enumerate(fine_dirs):
        fine.append(score_direction(small, d, cfg.fine_grid, block_margin=block_margin))
        if i % 10 == 0:
            report(
                0.5 + 0.5 * (i + 1) / len(fine_dirs),
                f"fine sweep {i + 1}/{len(fine_dirs)}",
            )

    all_scores = fine + best_coarse
    order = _rank(all_scores, cfg.tie_tol, cfg.size_weight)
    ranked = [all_scores[i] for i in order]
    best = ranked[0]
    log.info(
        "chosen pull direction %s: undercut=%.4f%% trapped=%.2fcm3 multi=%.3f",
        np.round(best.direction, 4),
        best.undercut_fraction * 100,
        best.trapped_volume / 1000,
        best.multi_fraction,
    )
    report(1.0, "direction chosen")
    return best.direction.copy(), best, ranked


# --------------------------------------------------------------------------
# Per-face heatmap
# --------------------------------------------------------------------------


def face_undercut(
    mesh: trimesh.Trimesh,
    direction,
    *,
    eps: float = 1e-3,
    flex_reference: float = 8.0,
) -> dict:
    """Per-face undercut severity for a candidate pull direction.

    A face is fine if it can be reached from ``+d`` or from ``-d``: the half
    that pulls that way releases it. A face reachable from neither is a true
    undercut -- with a rigid cast it would be un-demoldable; with a flexible
    cast it is where the cast has to deform on the way out.

    Severity is 0 for a face that releases, and otherwise scales with how much
    material is in the way (normalised by ``flex_reference`` mm), so the heatmap
    distinguishes "peel it out" from "genuinely stuck".
    """
    d = unit(direction)
    tri = mesh.triangles
    centroids = tri.mean(axis=1)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)

    # Nudge off the surface along the face normal so the face cannot hit itself.
    scale = max(float(mesh.scale), 1.0)
    origins = centroids + normals * (eps * scale)

    severity = np.zeros(len(tri), dtype=np.float64)
    blocked = np.ones(len(tri), dtype=bool)
    depth = np.full(len(tri), np.inf)

    for sign in (1.0, -1.0):
        dirs = np.broadcast_to(sign * d, origins.shape)
        loc, index_ray, _ = mesh.ray.intersects_location(
            origins, dirs, multiple_hits=False
        )
        hit = np.zeros(len(tri), dtype=bool)
        dist = np.full(len(tri), np.inf)
        if len(index_ray):
            hit[index_ray] = True
            dist[index_ray] = np.linalg.norm(loc - origins[index_ray], axis=1)
        blocked &= hit
        depth = np.minimum(depth, dist)

    stuck = blocked
    severity[stuck] = np.clip(depth[stuck] / max(flex_reference, 1e-6), 0.02, 1.0)

    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total = float(areas.sum())
    stuck_area = float(areas[stuck].sum())

    return {
        "severity": severity,
        "undercut": stuck,
        "depth": depth,
        "direction": d,
        "undercut_area_fraction": (stuck_area / total) if total else 0.0,
        "undercut_face_count": int(stuck.sum()),
        "face_count": int(len(tri)),
    }
