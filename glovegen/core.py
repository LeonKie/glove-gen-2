"""The core: the inner form that makes the cast hollow.

The scan is the glove's *outside*. A glove of wall thickness ``t`` is the gap
between that surface and the same surface inset by ``t``, so casting one needs a
second printed part -- the core -- sitting inside the cavity with ``t`` of
clearance all round.

The core is two pieces unioned: the inset body, and a **cuff cap** taken from the
cavity's own material beyond the cuff plane. The cap is full size, so it seals
against the cavity wall and no cast gets past it -- which is what leaves the
glove open at the wrist -- and being cut from the cavity it is an exact fit in
the void the cavity left at that end of the block. Closing the mold traps it.
Nothing in this module touches the mold halves.

Why this one place offsets, when the mold deliberately does not
----------------------------------------------------------------
:mod:`glovegen.mold` avoids offsetting entirely: the mold is ``block - part``,
three booleans, no grids (see that module's docstring). The core cannot dodge it
the same way -- an inset surface is not expressible as a boolean of the input --
so the cost has to be paid somewhere. It is paid *here*, once, and only when a
core is asked for.

``manifold3d`` does expose ``minkowski_difference``, which is an exact erosion,
but it scales with the product of the two face counts; a 2.6M-triangle scan
against even a coarse ball is hopeless. So the inset is extracted from a distance
field:

1. **Occupancy** comes from :func:`glovegen.demold.cast_grid` -- one ray per grid
   column, material intervals read off the sorted crossings. That is the same
   model the pull-direction search is built on, and it reproduces ``mesh.volume``
   to 0.1%: an *exact* inside/outside classification, not a surface
   voxelisation with its scattered false negatives on thin walls.
2. **Signed distance** is two Euclidean distance transforms, ``edt(inside)``
   minus ``edt(outside)``, so the field is continuous across the boundary and the
   ``t``-level is well defined.
3. **The surface** is ``Manifold.level_set(..., level=t)``. Marching tetrahedra
   over a body-centred cubic grid, so the result is manifold by construction --
   which matters, because everything downstream is a boolean.

The cost is the level-set callback: it is Python, called once per grid point.
Roughly 0.3M evaluations/s, and the count scales as the bounding volume over
``edge_length`` cubed -- about 20 s on the hand scan at 1 mm. ``core.edge_length``
is the dial. If that ever becomes the bottleneck, ``skimage.measure.marching_cubes``
on the very same ``sdf`` array is ~1 s; it is not a dependency here because it
pulls in a stack this project otherwise has no use for, and because it does not
give the manifoldness guarantee.
"""

from __future__ import annotations

import logging
import time
from array import array
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy import ndimage

from . import demold, meshio, mold, validate
from .config import MoldConfig
from .frame import Frame, unit
from .mold import local_to_world

log = logging.getLogger(__name__)


class CoreError(validate.SolidError):
    """The core cannot be built as asked."""


# --------------------------------------------------------------------------
# the inward offset
# --------------------------------------------------------------------------

# Residual bias of the extracted level, as a fraction of the voxel pitch,
# after the centre-to-surface correction in `signed_distance` has been applied.
#
# Measured, not derived: insetting six shapes by 3 mm and measuring the wall that
# actually came out (median closest-point distance from the inset surface back to
# the input) gives, as a multiple of the pitch,
#
#     rotated cube  +0.23 / +0.31      capsule  +0.34 / +0.30      (pitch 1.0 / 0.5)
#     rotated slab  +0.25 / +0.32      torus    +0.32 / +0.35
#     sphere r20    +0.30 / +0.34      sphere r40  +0.28 / +0.33
#
# i.e. the wall comes out consistently that much *too thick*, independent of
# curvature -- a flat face and a 10 mm-radius sphere agree. Extracting that much
# shallower cancels it and leaves under 0.1 mm at a 1 mm pitch. `tests/test_core.py`
# pins this; if the field or the extractor changes, re-measure it there.
#
# One shape does not follow the rule: a box whose faces are axis-aligned *and*
# flush with the grid, where every boundary voxel centre lands exactly on the
# surface. Real scans have no such faces, and the grid is derived from the mesh's
# own bounding box, so this is only ever seen in a synthetic test.
_LEVEL_BIAS = 0.30


def _auto_pitch(thickness: float) -> float:
    """Voxel pitch for a given wall thickness.

    Half the wall, capped at 1 mm. The cap is what keeps a thick-walled glove
    from asking for a needlessly fine grid: accuracy of the inset is bounded by
    the pitch in absolute terms, not relative to ``t``.
    """
    return float(min(1.0, max(0.1, thickness / 2.0)))


def occupancy_grid(
    mesh: trimesh.Trimesh,
    frame: Frame,
    pitch: float,
    *,
    pad: float,
    max_voxels: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rasterise ``mesh`` into a boolean grid in ``frame``'s coordinates.

    Returns ``(occ, origin_local, pitch)`` where ``occ[i, j, k]`` says whether the
    centre of that voxel is inside the mesh, and ``origin_local`` is the centre of
    voxel ``(0, 0, 0)``. ``pitch`` comes back because it is coarsened if the
    requested one would blow the voxel budget.

    Filled directly from ray crossings rather than by voxelising the surface and
    flood-filling: a column's material intervals are exact, so a wall thinner
    than the pitch is still classified correctly instead of leaking.
    """
    hull_local = frame.to_local(meshio.convex_hull(mesh).vertices)
    lo = hull_local.min(axis=0) - pad
    hi = hull_local.max(axis=0) + pad

    extent = hi - lo

    def shape_for(p: float) -> np.ndarray:
        return np.maximum(np.ceil(extent / p).astype(np.int64) + 1, 2)

    want = shape_for(pitch)
    if int(np.prod(want, dtype=np.float64)) > max_voxels:
        asked = pitch
        # One rescale by the cube root of the overshoot is not quite enough --
        # the per-axis ceil rounds back up -- so nudge until it fits.
        while int(np.prod(want, dtype=np.float64)) > max_voxels:
            pitch *= max(
                1.02,
                (float(np.prod(want, dtype=np.float64)) / max_voxels) ** (1.0 / 3.0),
            )
            want = shape_for(pitch)
        log.warning(
            "core grid at %.2f mm would exceed the %.1fM voxel budget; "
            "coarsening to %.2f mm",
            asked,
            max_voxels / 1e6,
            pitch,
        )

    nx, ny, nz = (int(v) for v in want)
    xs = lo[0] + np.arange(nx) * pitch
    ys = lo[1] + np.arange(ny) * pitch
    zlo = float(lo[2])

    t0 = time.time()
    cast = demold.cast_grid(mesh, frame, xs, ys, z_start=zlo - pitch)
    log.info("core occupancy: %d columns in %.1fs", cast.ncol, time.time() - t0)

    occ = np.zeros((nx * ny, nz), dtype=bool)
    if len(cast.t):
        col = np.repeat(np.arange(cast.ncol), cast.counts)
        rank = np.arange(len(cast.t)) - np.repeat(cast.starts, cast.counts)
        # Segment k spans crossings k -> k+1; even k is inside the part. Same
        # convention as ColumnCast.segment_table.
        keep = (rank % 2 == 0) & (rank < np.repeat(cast.counts, cast.counts) - 1)
        idx = np.flatnonzero(keep)
        if len(idx):
            z0 = cast.local_z(cast.t[idx])
            z1 = cast.local_z(cast.t[idx + 1])
            k0 = np.clip(np.ceil((z0 - zlo) / pitch), 0, nz).astype(np.int64)
            k1 = np.clip(np.floor((z1 - zlo) / pitch) + 1, 0, nz).astype(np.int64)
            c = col[idx]
            live = k1 > k0
            c, k0, k1 = c[live], k0[live], k1[live]
            # Mark the ends of every interval and integrate: one pass, no python
            # loop over segments.
            stride = nz + 1
            size = cast.ncol * stride
            delta = np.bincount(c * stride + k0, minlength=size)
            delta -= np.bincount(c * stride + k1, minlength=size)
            occ = np.cumsum(delta.reshape(cast.ncol, stride)[:, :nz], axis=1) > 0

    origin = np.array([xs[0], ys[0], zlo], dtype=np.float64)
    return occ.reshape(nx, ny, nz), origin, pitch


def signed_distance(occ: np.ndarray, pitch: float) -> np.ndarray:
    """Signed distance to the surface, in mm. Positive inside.

    Both transforms are needed. ``edt(occ)`` alone is flat zero everywhere
    outside, so the field has no gradient there and the level-set extractor
    cannot interpolate a crossing near the boundary.

    The half-pitch term is not cosmetic. ``edt`` measures to the nearest voxel
    *centre* of the opposite class, but the surface runs somewhere between the
    two centres, so raw ``edt`` overstates every depth by about half a voxel.
    Left in, that offset lands directly on the wall: the ``t``-level comes out
    ``t - pitch/2`` deep and a 3 mm wall prints at 2.8.
    """
    inside = ndimage.distance_transform_edt(occ, sampling=pitch)
    outside = ndimage.distance_transform_edt(~occ, sampling=pitch)
    signed = inside - outside
    return (signed - 0.5 * pitch * np.where(occ, 1.0, -1.0)).astype(np.float32)


def _sampler(grid: np.ndarray, origin: np.ndarray, pitch: float):
    """A trilinear sampler of ``grid``, as a scalar callable for ``level_set``.

    Written against a flat ``array('f')`` rather than the numpy array on purpose:
    this runs millions of times, and indexing a numpy array returns a boxed numpy
    scalar, which costs several times what a plain float does.
    """
    nx, ny, nz = grid.shape
    flat = array("f", grid.ravel(order="C").tolist())
    step_y, step_x = nz, ny * nz
    ox, oy, oz = (float(v) for v in origin)
    inv = 1.0 / pitch
    lim_x, lim_y, lim_z = nx - 2, ny - 2, nz - 2
    far = float(grid.min())

    def sample(x: float, y: float, z: float) -> float:
        fx = (x - ox) * inv
        fy = (y - oy) * inv
        fz = (z - oz) * inv
        ix = int(fx)
        iy = int(fy)
        iz = int(fz)
        if ix < 0 or iy < 0 or iz < 0 or ix > lim_x or iy > lim_y or iz > lim_z:
            return far
        tx = fx - ix
        ty = fy - iy
        tz = fz - iz
        b = ix * step_x + iy * step_y + iz
        c00 = flat[b]
        c00 += (flat[b + 1] - c00) * tz
        c01 = flat[b + step_y]
        c01 += (flat[b + step_y + 1] - c01) * tz
        c10 = flat[b + step_x]
        c10 += (flat[b + step_x + 1] - c10) * tz
        c11 = flat[b + step_x + step_y]
        c11 += (flat[b + step_x + step_y + 1] - c11) * tz
        c0 = c00 + (c01 - c00) * ty
        c1 = c10 + (c11 - c10) * ty
        return c0 + (c1 - c0) * tx

    return sample


def deflate(
    mesh: trimesh.Trimesh,
    frame: Frame,
    thickness: float,
    *,
    pitch: float = 0.0,
    edge_length: float = 0.0,
    max_voxels: int = 40_000_000,
) -> tuple[trimesh.Trimesh, dict]:
    """``mesh`` inset by ``thickness`` mm. Returns ``(mesh, stats)``.

    Raises :class:`CoreError` if nothing survives -- an inset does not fail by
    getting thin, it fails by *vanishing*, and a wall thicker than half the
    part's narrowest feature erases that feature entirely.
    """
    from manifold3d import Manifold

    thickness = float(thickness)
    if thickness <= 0:
        raise CoreError(f"core thickness must be positive, got {thickness}")
    pitch = float(pitch) or _auto_pitch(thickness)
    stats: dict = {}

    t0 = time.time()
    occ, origin, pitch = occupancy_grid(
        mesh, frame, pitch, pad=thickness + 3.0 * pitch, max_voxels=max_voxels
    )
    sdf = signed_distance(occ, pitch)
    stats["grid_pitch_mm"] = round(pitch, 4)
    stats["grid_voxels"] = int(occ.size)
    stats["distance_field_s"] = round(time.time() - t0, 2)

    edge = float(edge_length) or max(1.0, pitch)
    level = max(0.25 * thickness, thickness - _LEVEL_BIAS * pitch)

    inside = sdf >= level
    if not inside.any():
        raise CoreError(
            f"a {thickness:g} mm wall leaves no core at all: the part's widest "
            f"point is only {float(sdf.max()):.2f} mm from its surface"
        )

    # Extract over the inset's own bounding box, not the part's: the level set
    # costs one callback per grid point of whatever box it is given.
    span = []
    for axis, n in ((0, occ.shape[0]), (1, occ.shape[1]), (2, occ.shape[2])):
        hit = np.flatnonzero(inside.any(axis=tuple(a for a in (0, 1, 2) if a != axis)))
        span.append((max(0, int(hit[0]) - 2), min(n - 1, int(hit[-1]) + 2)))
    lo = np.array([origin[a] + span[a][0] * pitch for a in range(3)])
    hi = np.array([origin[a] + span[a][1] * pitch for a in range(3)])

    t0 = time.time()
    solid = Manifold.level_set(
        _sampler(sdf, origin, pitch),
        [*(float(v) for v in lo), *(float(v) for v in hi)],
        edge,
        level,
    )
    stats["level_set_s"] = round(time.time() - t0, 2)
    stats["edge_length_mm"] = round(edge, 3)
    stats["level_mm"] = round(level, 4)

    if solid.is_empty() or solid.num_tri() == 0:
        raise CoreError(
            f"a {thickness:g} mm wall leaves no core the extractor can resolve; "
            f"try a finer core.edge_length than {edge:g} mm"
        )

    out = mold.from_manifold(solid)
    out.apply_transform(local_to_world(frame))
    log.info(
        "core inset by %.2f mm: %d faces, %.1f cm3 (grid %.2f mm in %.1fs, "
        "surface in %.1fs)",
        thickness,
        len(out.faces),
        out.volume / 1000.0,
        pitch,
        stats["distance_field_s"],
        stats["level_set_s"],
    )
    return out, stats


# --------------------------------------------------------------------------
# the core as a printable part
# --------------------------------------------------------------------------


@dataclass
class CoreResult:
    """The core, as a printable part."""

    core: trimesh.Trimesh
    # The inset body alone, before the cuff cap was unioned on. Kept because it,
    # not the finished core, is what the glove is the complement of.
    body: trimesh.Trimesh
    pour_axis: np.ndarray
    cuff_offset: float  # where the glove ends, as a projection onto the pour axis
    stats: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "pour_axis": [round(float(v), 4) for v in self.pour_axis],
            "cuff_offset_mm": round(float(self.cuff_offset), 3),
            **self.stats,
        }


def components(mesh: trimesh.Trimesh) -> int:
    """How many separate solids ``mesh`` is, asked of manifold3d rather than trimesh.

    ``Trimesh.split`` walks face adjacency over *merged* vertices, and a boolean
    result routinely carries position-coincident vertices that were never welded
    (the same quirk that makes these solids report ``watertight: false`` --
    see :func:`glovegen.meshio.mesh_stats`). Split reads each unwelded seam as a
    boundary and reports a single sphere as dozens of pieces. ``decompose``
    answers from the winding, so it is not fooled.
    """
    return len(mold.to_manifold(mesh).decompose()) or 1


def _halfspace(cavity: trimesh.Trimesh, axis: np.ndarray, offset: float) -> trimesh.Trimesh:
    """A box covering everything with ``p @ axis >= offset``."""
    size = 3.0 * float(cavity.scale)
    box = trimesh.creation.box(extents=(size, size, size))
    box.apply_transform(trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis))
    centre = np.asarray(cavity.centroid, dtype=np.float64)
    box.apply_translation(centre + axis * (offset - float(centre @ axis) + size / 2.0))
    return box


def _section_points(mesh: trimesh.Trimesh, axis: np.ndarray, offset: float) -> np.ndarray:
    """Points on the cross-section of ``mesh`` at ``p @ axis == offset``."""
    seg = trimesh.intersections.mesh_plane(
        mesh, plane_normal=axis, plane_origin=axis * offset
    )
    if seg is None or len(seg) == 0:
        return np.zeros((0, 3))
    return np.asarray(seg, dtype=np.float64).reshape(-1, 3)


def _inradius(points: np.ndarray, centre: np.ndarray, axis: np.ndarray) -> float:
    """Radius of the largest circle about ``centre`` fitting inside a section.

    The section's boundary is the set of points, so this is just the nearest one,
    measured perpendicular to the axis.
    """
    if len(points) == 0:
        return 0.0
    v = points - centre
    return float(np.linalg.norm(v - np.outer(v @ axis, axis), axis=1).min())


_CAP_OVERLAP = 1.0  # how far the inset body must poke past the cuff plane


def _cuff_plane(
    cavity: trimesh.Trimesh,
    body: trimesh.Trimesh,
    axis: np.ndarray,
    shift: float,
    thickness: float,
) -> float:
    """Where the glove ends along the pour axis.

    Two things have to be true here, and the part's extreme satisfies neither in
    general:

    - **The cap has to overlap the body**, or the core comes out in two pieces.
      The inset body stops ``thickness`` short of every surface, the end face
      included, so a cap taken from the very extreme floats free of it.
    - **The section has to be fat enough** that the inset body still exists
      there at all. True for a scan cut off flat at the wrist, where the section
      is full width right to the cut; false for anything tapering to a point,
      where the outermost sections are millimetres across.

    The second is a search: walk inward from the extreme until the section is
    wider than the wall. The first is not left to that search, because the
    section width is measured about the section's own centroid and a cut face
    that is slanted relative to the pour axis gives a thin crescent whose
    centroid sits near its edge -- on the hand scan that reads as 3.5 mm on a
    60 mm wrist. So the answer is clamped against the body's actual reach, which
    is exact and needs no geometry at all.
    """
    small = meshio.proxy(cavity, 80_000)
    proj = np.asarray(small.vertices, dtype=np.float64) @ axis
    hi = float(np.percentile(proj, 99.9))
    span = hi - float(proj.min())
    want = 1.5 * thickness

    cut = hi - thickness - 1.0
    for candidate in np.linspace(cut, hi - 0.35 * span, 24):
        pts = _section_points(small, axis, float(candidate))
        if len(pts) and _inradius(pts, pts.mean(axis=0), axis) >= want:
            cut = float(candidate)
            break
    else:
        log.warning(
            "no cross-section near the end of the part is a comfortable seat for "
            "the cuff cap; falling back on the body's reach"
        )

    reach = float((np.asarray(body.vertices, dtype=np.float64) @ axis).max())
    return min(cut, reach - _CAP_OVERLAP) + float(shift)


def build_core(
    cavity: trimesh.Trimesh,
    frame: Frame,
    pour_axis,
    cfg: MoldConfig | None = None,
    *,
    progress=None,
) -> CoreResult:
    """Build the core for ``cavity``: the inset body plus its cuff cap.

    Two pieces:

    - the **body** is the cavity inset by ``thickness``; the cast forms in the gap;
    - the **cuff cap** is the cavity's own material beyond the cuff plane, at full
      size. Because it matches the cavity exactly it seals against it, so no cast
      material gets past -- which is what leaves the glove *open* at the wrist.

    Nothing here touches the mold halves. The cap is cut from the cavity, so it
    is an exact fit in the void the cavity left at that end of the block: closing
    the mold traps it, which locates the core without any feature having to be
    added to the halves at all.
    """
    cfg = cfg or MoldConfig()
    core_cfg = cfg.core
    report = progress or (lambda *a, **k: None)
    timings: dict[str, float] = {}
    axis = unit(pour_axis)

    report(0.0, f"insetting the core by {core_cfg.thickness}mm")
    t0 = time.time()
    body, stats = deflate(
        cavity,
        frame,
        core_cfg.thickness,
        pitch=core_cfg.grid_pitch,
        edge_length=core_cfg.edge_length,
        max_voxels=core_cfg.max_voxels,
    )
    timings["core_inset"] = time.time() - t0
    validate.assert_solid_enough(body, "core (inset body)")

    report(0.7, "capping the cuff")
    t0 = time.time()
    cut = _cuff_plane(cavity, body, axis, core_cfg.cuff_offset, core_cfg.thickness)
    beyond = _halfspace(cavity, axis, cut)
    cap = mold._boolean(trimesh.boolean.intersection, [cavity, beyond], "cuff cap")
    if len(cap.faces) == 0 or cap.volume <= 0:
        raise CoreError(
            f"cuff_offset {core_cfg.cuff_offset:g} mm puts the glove's opening "
            "past the end of the part; there is nothing to cap"
        )
    # The body is deliberately *not* clipped at the cuff plane. It is inset from
    # the cavity, so whatever of it lies beyond the plane is already inside the
    # cap -- clipping would change nothing about the result while turning the
    # union into two solids meeting exactly on a plane, which manifold3d resolves
    # into a spray of coplanar slivers (the same degeneracy parting.py's
    # _SOLID_OVERHANG_MM exists to avoid).
    core = mold._boolean(trimesh.boolean.union, [body, cap], "core body + cuff cap")
    validate.assert_solid_enough(core, "core")
    timings["cuff_cap"] = time.time() - t0

    extreme = float((np.asarray(cavity.vertices, dtype=np.float64) @ axis).max())
    stats.update(
        {
            "thickness_mm": round(float(core_cfg.thickness), 3),
            # How far back from the part's end the glove's opening sits. Says
            # something a reader can check against the part; the cross-section's
            # width at that plane does not, on a slanted cut.
            "cuff_inset_mm": round(extreme - cut, 2),
            "volume_cm3": round(float(core.volume) / 1000.0, 2),
            "faces": int(len(core.faces)),
            "components": components(core),
        }
    )
    if stats["components"] > 1:
        log.warning(
            "the %.2f mm inset broke the core into %d pieces: a feature narrower "
            "than %.2f mm was erased",
            core_cfg.thickness,
            stats["components"],
            2.0 * core_cfg.thickness,
        )

    return CoreResult(
        core=core,
        body=body,
        pour_axis=axis,
        cuff_offset=cut,
        stats=stats,
        timings=timings,
    )


def glove_preview(cavity: trimesh.Trimesh, result: "CoreResult") -> trimesh.Trimesh:
    """What actually gets cast: the gap between the cavity and the core.

    Subtracts the inset *body* and clips at the cuff, rather than subtracting the
    finished core. Same solid, but it never asks the boolean engine about two
    coincident surfaces: the cuff cap's side wall is cut from the cavity and so
    lies exactly on it, and differencing across that produces a shell shot
    through with slivers -- twenty-odd disconnected fragments on a sphere.
    """
    beyond = _halfspace(cavity, result.pour_axis, result.cuff_offset)
    inner = mold._boolean(trimesh.boolean.difference, [cavity, beyond], "glove - cuff")
    return mold._boolean(trimesh.boolean.difference, [inner, result.body], "glove")
