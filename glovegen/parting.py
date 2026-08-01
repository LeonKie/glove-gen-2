"""The curved parting surface.

Approach
--------
The parting surface is a *height field* ``z = h(x, y)`` in the frame where the
pull direction is +Z, defined over the whole block footprint.

That single choice buys a lot. A height field is single-valued, so it cannot
self-intersect; it triangulates into a guaranteed-manifold sheet; and extruding
it to the top of the block gives a watertight "everything above the parting
surface" solid that splits the mold with two boolean ops.

The surface is *constrained*, not free. In every column that passes through the
part, ``h`` is pinned inside the thickest slab of part material. So:

* through the palm, ``h`` runs through the middle of the hand -- the mold splits
  into a palm-side half and a back-of-hand half, which is the natural parting;
* approaching the silhouette the slab thins to nothing, so the band collapses
  and ``h`` is forced through the silhouette exactly. The surface follows the
  undercut boundary without any loop extraction, contouring or ear-clipping --
  which is where v1 spent its complexity and silently produced
  self-intersections;
* between the fingers no column hits the part, so ``h`` is unconstrained there
  and simply interpolates smoothly across the gap.

Solving it is a small quadratic program: minimise
``lambda * ||grad h||^2 + ||h - target||^2`` over constrained columns, subject to
``band_lo <= h <= band_hi``. That is one sparse linear solve plus a few
active-set passes to enforce the inequality.

Grid resolution affects seam quality, not correctness: coarsening it moves where
the seam line sits on the cast, but ``h`` stays inside the cavity, so the halves
still reproduce the shape exactly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh

from . import demold
from .config import PartingConfig
from .frame import Frame

log = logging.getLogger(__name__)

# Weight used to pin a node that smoothing dragged out of its band. A pinned
# node still drifts by roughly (neighbour height offset) / _HARD_WEIGHT, and at
# the silhouette a neighbour can be tens of mm away, so this needs to be large.
_HARD_WEIGHT = 1.0e6

# Tolerance for "h left its feasible band", in mm. Deliberately physical rather
# than epsilon-sized: near the silhouette the band legitimately narrows to a few
# microns, and asking a finite-weight least-squares solve to honour a 7-micron
# band to nanometre precision reports violations forever without any of them
# meaning anything. 0.1 micron is orders of magnitude below print resolution.
_BAND_TOL_MM = 1.0e-4


@dataclass
class PartingSurface:
    """A curved parting surface stored as a height field in the pull frame."""

    frame: Frame
    xs: np.ndarray  # (nx,) local x of grid nodes, spanning the block footprint
    ys: np.ndarray  # (ny,)
    h: np.ndarray  # (nx, ny) local z
    constrained: np.ndarray  # (nx, ny) bool: this column passes through the part
    band_lo: np.ndarray  # (nx, ny) lower feasible bound (only where constrained)
    band_hi: np.ndarray
    stats: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.xs), len(self.ys)

    # -- geometry ---------------------------------------------------------

    def _nodes_local(self, z: np.ndarray) -> np.ndarray:
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel(), np.asarray(z).ravel()])

    def _grid_faces(self, flip: bool) -> np.ndarray:
        """Triangulate the node grid; ``flip`` reverses the winding."""
        nx, ny = self.shape
        idx = np.arange(nx * ny).reshape(nx, ny)
        v00 = idx[:-1, :-1].ravel()
        v10 = idx[1:, :-1].ravel()
        v11 = idx[1:, 1:].ravel()
        v01 = idx[:-1, 1:].ravel()
        # Both triangles wound counter-clockwise in xy, i.e. normals along +Z.
        f = np.vstack(
            [np.column_stack([v00, v10, v11]), np.column_stack([v00, v11, v01])]
        )
        return f[:, ::-1] if flip else f

    def surface_mesh(self) -> trimesh.Trimesh:
        """The parting sheet on its own, as an open surface (for display)."""
        verts = self.frame.to_world(self._nodes_local(self.h))
        return trimesh.Trimesh(
            vertices=verts, faces=self._grid_faces(flip=False), process=False
        )

    def solid(self, z_top: float) -> trimesh.Trimesh:
        """Closed solid filling everything between the surface and ``z_top``.

        Intersecting the mold with this solid yields the half that pulls along
        ``+direction``; subtracting it yields the other half.
        """
        nx, ny = self.shape
        n = nx * ny
        if z_top <= float(self.h.max()):
            raise ValueError(
                f"z_top={z_top:.3f} is not above the parting surface "
                f"(max h={float(self.h.max()):.3f})"
            )

        bottom = self._nodes_local(self.h)
        top = self._nodes_local(np.full_like(self.h, float(z_top)))
        verts_local = np.vstack([bottom, top])

        faces = [
            self._grid_faces(flip=True),  # underside, normals along -Z
            self._grid_faces(flip=False) + n,  # cap, normals along +Z
        ]

        idx = np.arange(n).reshape(nx, ny)

        def wall(ring: np.ndarray, outward_flip: bool) -> np.ndarray:
            b0, b1 = ring[:-1], ring[1:]
            t0, t1 = b0 + n, b1 + n
            quad = np.vstack(
                [np.column_stack([b0, b1, t1]), np.column_stack([b0, t1, t0])]
            )
            return quad[:, ::-1] if outward_flip else quad

        # Winding chosen so every wall normal points away from the solid; the
        # volume-sign check below is the backstop.
        faces.append(wall(idx[0, :], True))  # x = xs[0],  outward -X
        faces.append(wall(idx[-1, :], False))  # x = xs[-1], outward +X
        faces.append(wall(idx[:, 0], False))  # y = ys[0],  outward -Y
        faces.append(wall(idx[:, -1], True))  # y = ys[-1], outward +Y

        mesh = trimesh.Trimesh(
            vertices=self.frame.to_world(verts_local),
            faces=np.vstack(faces),
            process=False,
        )
        mesh.merge_vertices()
        if mesh.volume < 0:
            mesh.invert()
        return mesh

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the surface to a ``.npz`` so a later run can place features.

        Everything a feature needs -- where the mating face sits, which columns
        pass through the cavity, which way the mold pulls -- lives here, and it
        is a few hundred kB against the minute it took to solve.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            rot=self.frame.rot,
            direction=self.frame.direction,
            xs=self.xs,
            ys=self.ys,
            h=self.h,
            constrained=self.constrained,
            band_lo=self.band_lo,
            band_hi=self.band_hi,
            stats=np.array(json.dumps(self.stats)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PartingSurface":
        """Inverse of :meth:`save`."""
        with np.load(str(path), allow_pickle=False) as d:
            return cls(
                frame=Frame(direction=d["direction"], rot=d["rot"]),
                xs=d["xs"],
                ys=d["ys"],
                h=d["h"],
                constrained=d["constrained"],
                band_lo=d["band_lo"],
                band_hi=d["band_hi"],
                stats=json.loads(str(d["stats"])),
            )

    # -- reporting --------------------------------------------------------

    def report(self) -> dict:
        nx, ny = self.shape
        thickness = np.where(
            self.constrained, self.band_hi - self.band_lo, np.nan
        )
        return {
            "grid": [nx, ny],
            "cell_mm": round(float(self.xs[1] - self.xs[0]), 4) if nx > 1 else 0.0,
            "constrained_columns": int(self.constrained.sum()),
            "total_columns": int(nx * ny),
            "median_band_mm": (
                round(float(np.nanmedian(thickness)), 3)
                if self.constrained.any()
                else 0.0
            ),
            **self.stats,
        }


# --------------------------------------------------------------------------


def _grid_laplacian(nx: int, ny: int) -> sp.csr_matrix:
    """5-point graph Laplacian on an ``nx`` x ``ny`` node grid, Neumann edges."""
    idx = np.arange(nx * ny).reshape(nx, ny)
    a = np.concatenate([idx[:-1, :].ravel(), idx[:, :-1].ravel()])
    b = np.concatenate([idx[1:, :].ravel(), idx[:, 1:].ravel()])
    n = nx * ny
    adj = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    adj = (adj + adj.T).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    return (sp.diags(deg) - adj).tocsr()


def _solve_height_field(
    target: np.ndarray,
    band_lo: np.ndarray,
    band_hi: np.ndarray,
    constrained: np.ndarray,
    *,
    lam: float,
    rounds: int,
) -> tuple[np.ndarray, dict]:
    """Smooth height field pinned into ``[band_lo, band_hi]`` where constrained.

    Minimises ``lam * ||grad h||^2 + sum_constrained (h - target)^2`` and then
    runs a few active-set passes: any constrained node that smoothing dragged
    out of its band gets re-pinned hard at the violated bound and the system is
    re-solved.
    """
    nx, ny = target.shape
    n = nx * ny
    lap = _grid_laplacian(nx, ny)

    mask = constrained.ravel()
    tgt = np.where(mask, target.ravel(), 0.0)
    lo = band_lo.ravel()
    hi = band_hi.ravel()

    if not mask.any():
        return np.zeros((nx, ny)), {"active_set_rounds_used": 0, "band_violations": 0}

    weight = mask.astype(np.float64)
    h = np.zeros(n)
    used = 0
    violations = 0

    for used in range(1, max(1, rounds) + 1):
        a = (lam * lap + sp.diags(weight + 1e-9)).tocsc()
        h = spla.spsolve(a, weight * tgt)
        if not np.all(np.isfinite(h)):  # pragma: no cover - singular fallback
            h = np.where(mask, tgt, float(np.nanmean(tgt[mask])))
            break

        below = mask & (h < lo - _BAND_TOL_MM)
        above = mask & (h > hi + _BAND_TOL_MM)
        violations = int(below.sum() + above.sum())
        if violations == 0:
            break
        weight[below] = _HARD_WEIGHT
        tgt[below] = lo[below]
        weight[above] = _HARD_WEIGHT
        tgt[above] = hi[above]

    # Final safety clamp: the active-set loop is capped, so guarantee the
    # invariant that matters -- the surface stays inside the cavity -- directly
    # rather than hoping the solve converged.
    clamped = np.where(mask, np.clip(h, lo, hi), h)
    correction = np.abs(clamped - h)
    h = clamped

    return h.reshape(nx, ny), {
        "active_set_rounds_used": int(used),
        "band_violations": int(violations),
        # The count alone is not meaningful; the magnitude is. A correction far
        # below a printer's layer height means the clamp changed nothing that
        # can be manufactured.
        "max_clamp_mm": round(float(correction.max()), 6),
    }


def build(
    mesh: trimesh.Trimesh,
    direction,
    footprint_lo,
    footprint_hi,
    cfg: PartingConfig | None = None,
    *,
    frame: Frame | None = None,
    progress=None,
) -> PartingSurface:
    """Build the parting surface for ``mesh`` pulled along ``direction``.

    ``footprint_lo`` / ``footprint_hi`` are the local-frame (x, y) bounds the
    surface must span -- i.e. the block's footprint, so the sheet reaches the
    block walls.
    """
    cfg = cfg or PartingConfig()
    report = progress or (lambda *a, **k: None)
    frame = frame or Frame.from_direction(direction)

    lo2 = np.asarray(footprint_lo, dtype=np.float64).reshape(2)
    hi2 = np.asarray(footprint_hi, dtype=np.float64).reshape(2)
    span = np.maximum(hi2 - lo2, 1e-6)
    cell = float(span.max()) / max(2, int(cfg.grid))
    nx = max(2, int(round(span[0] / cell)) + 1)
    ny = max(2, int(round(span[1] / cell)) + 1)
    xs = np.linspace(lo2[0], hi2[0], nx)
    ys = np.linspace(lo2[1], hi2[1], ny)

    report(0.15, f"casting parting grid {nx}x{ny}")
    cast = demold.cast_grid(mesh, frame, xs, ys)

    cols, z_lo, z_hi = cast.thickest_material_interval()
    if len(cols) == 0:
        raise ValueError(
            "no ray column passed through the part; cannot place a parting "
            "surface (is the pull direction sane?)"
        )

    constrained = np.zeros(nx * ny, dtype=bool)
    band_lo = np.zeros(nx * ny)
    band_hi = np.zeros(nx * ny)
    target = np.zeros(nx * ny)

    thickness = z_hi - z_lo
    # Never let the inset invert a thin band; near the silhouette the slab
    # vanishes and the band legitimately collapses to a point.
    inset = np.minimum(cfg.band_inset, 0.4 * thickness)
    constrained[cols] = True
    band_lo[cols] = z_lo + inset
    band_hi[cols] = z_hi - inset
    target[cols] = np.clip(
        z_lo + cfg.band_target * thickness, band_lo[cols], band_hi[cols]
    )

    report(0.55, "solving parting height field")
    h, solve_stats = _solve_height_field(
        target.reshape(nx, ny),
        band_lo.reshape(nx, ny),
        band_hi.reshape(nx, ny),
        constrained.reshape(nx, ny),
        lam=cfg.smooth_lambda,
        rounds=cfg.active_set_rounds,
    )

    surf = PartingSurface(
        frame=frame,
        xs=xs,
        ys=ys,
        h=h,
        constrained=constrained.reshape(nx, ny),
        band_lo=band_lo.reshape(nx, ny),
        band_hi=band_hi.reshape(nx, ny),
        stats={
            "odd_columns": int(cast.odd_columns),
            "multi_slab_columns": int((cast.counts > 2).sum()),
            **solve_stats,
        },
    )
    log.info("parting surface %dx%d: %s", nx, ny, surf.report())
    report(0.75, "parting surface ready")
    return surf
