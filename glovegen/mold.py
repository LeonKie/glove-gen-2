"""Mold block construction and the boolean split into two halves.

Why this is cheap
-----------------
v1 built each mold half by *offsetting the part surface outward* to make a
shell, which needs a signed-distance grid plus marching cubes. Voxel count grows
as ``O((extent / pitch)^3)``, so a hand-sized mold at a print-realistic wall
thickness never finished.

This pipeline never offsets anything. The mold is a block with the part
subtracted:

    mold   = block - part
    half_a = mold & above_parting_surface
    half_b = mold - above_parting_surface

Three boolean ops on the input mesh at full resolution, no grids. "Wall
thickness" is just the margin between the part and the block's outside, which
the block's construction guarantees by definition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from . import meshio, parting, validate
from .config import MoldConfig
from .frame import Frame, unit
from .parting import PartingSurface

log = logging.getLogger(__name__)


def local_to_world(frame: Frame) -> np.ndarray:
    """4x4 transform taking local-frame coordinates back to world."""
    t = np.eye(4)
    t[:3, :3] = frame.rot.T
    return t


@dataclass
class MoldResult:
    direction: np.ndarray
    frame: Frame
    block: trimesh.Trimesh
    mold: trimesh.Trimesh  # block with the cavity cut, before splitting
    half_a: trimesh.Trimesh  # pulls along +direction, before features
    half_b: trimesh.Trimesh  # pulls along -direction, before features
    surface: PartingSurface
    parting_solid: trimesh.Trimesh
    local_bounds: np.ndarray  # (2,3) block bounds in the pull frame
    # What was actually subtracted from the block: the input part, unless an
    # offset or a face budget changed it. Features are placed against *this*,
    # not the original scan, since it is the surface they have to clear.
    cavity: trimesh.Trimesh | None = None
    timings: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def snug_roll(mesh: trimesh.Trimesh, direction) -> np.ndarray:
    """The roll about ``direction`` that makes the smallest box, as a world vector.

    A box block is the hull's bounding box *in the pull frame*, so one of its
    axes is the pull direction and the other two are whatever roll the frame
    happens to carry. ``Frame.from_direction`` picks that roll from the
    direction alone -- the model never enters into it -- and on an
    axis-aligned pull it lands on the world axes. The block is then sized by
    how the scan happens to sit in world coordinates rather than by its own
    shape: on a hand-and-forearm, rotating the scan about the pull axis swings
    the block by a quarter of its volume without changing anything real.

    So pick the roll from the part instead. The minimum-area rectangle
    enclosing a convex polygon always has a side flush with one of its edges,
    so trying each edge in turn is exact rather than a search.
    """
    d = unit(direction)
    frame = Frame.from_direction(d)
    uv = frame.to_local(meshio.convex_hull(mesh).vertices)[:, :2]
    try:
        ring = uv[ConvexHull(uv).vertices]
    except Exception:  # pragma: no cover - degenerate (flat or tiny) footprint
        return frame.to_world([1.0, 0.0, 0.0])

    best = (np.inf, 0.0)
    edges = np.roll(ring, -1, axis=0) - ring
    for angle in np.arctan2(edges[:, 1], edges[:, 0]):
        c, s = np.cos(-angle), np.sin(-angle)
        turned = ring @ np.array([[c, -s], [s, c]]).T
        extent = turned.max(axis=0) - turned.min(axis=0)
        best = min(best, (float(extent[0] * extent[1]), float(angle)))

    return frame.to_world([np.cos(best[1]), np.sin(best[1]), 0.0])


def build_block(
    mesh: trimesh.Trimesh,
    frame: Frame,
    margin: float,
    shape: str = "box",
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Build the mold block around ``mesh``.

    Returns ``(block, local_bounds)`` where ``local_bounds`` is the block's
    axis-aligned extent in the pull frame -- used to size the parting grid and
    to place features.

    ``shape="box"`` gives a rectangular block. ``shape="hull"`` gives the
    part's convex hull dilated by ``margin``: the same minimum wall guarantee
    with far less material, at the cost of not being a tidy rectangle.
    """
    hull = meshio.convex_hull(mesh)
    hull_local = frame.to_local(hull.vertices)
    lo = hull_local.min(axis=0) - margin
    hi = hull_local.max(axis=0) + margin
    local_bounds = np.array([lo, hi])

    if shape == "box":
        transform = local_to_world(frame).copy()
        transform[:3, 3] = frame.to_world((lo + hi) / 2.0)
        block = trimesh.creation.box(extents=(hi - lo), transform=transform)
    elif shape == "hull":
        # Minkowski sum of a convex body with a ball is the convex hull of the
        # vertex-wise sum, so this is exact up to the sphere's tessellation.
        ball = trimesh.creation.icosphere(subdivisions=2, radius=float(margin))
        hv = np.asarray(hull.vertices, dtype=np.float64)
        if len(hv) > 3000:
            hv = hv[np.linspace(0, len(hv) - 1, 3000).astype(int)]
        pts = (hv[:, None, :] + ball.vertices[None, :, :]).reshape(-1, 3)
        block = trimesh.Trimesh(vertices=pts, process=False).convex_hull
    else:
        raise ValueError(f"unknown block_shape {shape!r}, expected 'box' or 'hull'")

    validate.assert_solid_enough(block, "mold block")
    return block, local_bounds


def to_manifold(mesh: trimesh.Trimesh):
    """trimesh -> manifold3d, for the ops trimesh's boolean wrapper does not expose."""
    from manifold3d import Manifold, Mesh

    return Manifold(
        mesh=Mesh(
            vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
            tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
        )
    )


def from_manifold(solid) -> trimesh.Trimesh:
    """Inverse of :func:`to_manifold`."""
    raw = solid.to_mesh()
    out = trimesh.Trimesh(
        vertices=np.asarray(raw.vert_properties[:, :3], dtype=np.float64),
        faces=np.asarray(raw.tri_verts),
        process=False,
    )
    out.merge_vertices()
    return out


def offset_cavity(mesh: trimesh.Trimesh, delta: float) -> trimesh.Trimesh:
    """Grow the cavity by ``delta`` mm, for cast shrinkage or fit clearance.

    Uses an exact Minkowski sum, which is expensive on a full-resolution scan;
    the caller is expected to decimate first if it matters. Shrinking is the
    same operation with the ball on the other side -- see
    :func:`glovegen.core.erode`.
    """
    if abs(delta) < 1e-9:
        return mesh
    if delta < 0:
        raise NotImplementedError(
            "negative cavity_offset (shrinking the cavity) is not supported; "
            "core.erode() does inward offsets"
        )
    from manifold3d import Manifold

    ball = trimesh.creation.icosphere(subdivisions=1, radius=float(delta))
    return from_manifold(
        Manifold.minkowski_sum(to_manifold(mesh), to_manifold(ball))
    )


def _boolean(op, meshes, label: str) -> trimesh.Trimesh:
    """Run a boolean, tolerating inputs trimesh's own checks would reject."""
    t0 = time.time()
    out = op(meshes, engine="manifold", check_volume=False)
    log.info("%s: %.1fs -> %d faces", label, time.time() - t0, len(out.faces))
    return out


def split_mold(
    mold: trimesh.Trimesh, parting_solid: trimesh.Trimesh
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Cut ``mold`` along the parting surface into its ``+d`` and ``-d`` halves."""
    half_a = _boolean(trimesh.boolean.intersection, [mold, parting_solid], "half A")
    half_b = _boolean(trimesh.boolean.difference, [mold, parting_solid], "half B")
    return half_a, half_b


def build_mold(
    mesh: trimesh.Trimesh,
    direction,
    cfg: MoldConfig | None = None,
    *,
    progress=None,
) -> MoldResult:
    """Build the two-part mold for ``mesh`` pulled along ``direction``."""
    cfg = cfg or MoldConfig()
    report = progress or (lambda *a, **k: None)
    timings: dict[str, float] = {}

    # The roll of the pull frame decides two of the block's three axes, and
    # left to `align_vectors` it is a function of the pull direction alone.
    # Two things want a say in it, and they disagree:
    #
    #  - a carrier plate is cut square to the pour axis, and slicing an
    #    arbitrarily-rolled box on an oblique plane gives a corner wedge rather
    #    than a plate. Squareness wins there, easily: a plate that is not a
    #    slab is not a plate, and a few percent of block is nothing beside it.
    #  - otherwise nothing cares about the roll except the block's own size, so
    #    it goes to whatever makes the block smallest.
    #
    # (features is imported here because it builds on this module.)
    if cfg.core.enabled and cfg.carrier.enabled:
        from .features import choose_pour_axis

        seed = choose_pour_axis(mesh, cfg)
    else:
        seed = snug_roll(mesh, direction)
    frame = Frame.from_direction(direction, seed=seed)

    cavity = mesh
    if cfg.cavity_faces:
        cavity = meshio.decimate(cavity, cfg.cavity_faces)
    if cfg.cavity_offset:
        report(0.05, f"offsetting cavity by {cfg.cavity_offset}mm")
        t0 = time.time()
        cavity = offset_cavity(cavity, cfg.cavity_offset)
        timings["cavity_offset"] = time.time() - t0
    validate.assert_solid_enough(cavity, "cavity")

    report(0.08, "building block")
    t0 = time.time()
    block, local_bounds = build_block(cavity, frame, cfg.block_margin, cfg.block_shape)
    timings["block"] = time.time() - t0

    report(0.12, "building parting surface")
    t0 = time.time()
    surface = parting.build(
        cavity,
        frame.direction,
        local_bounds[0, :2],
        local_bounds[1, :2],
        cfg.parting,
        frame=frame,
        progress=lambda f, msg: report(0.12 + 0.18 * f, msg),
    )
    timings["parting_surface"] = time.time() - t0

    z_top = float(local_bounds[1, 2]) + 1.0
    parting_solid = surface.solid(z_top)
    validate.assert_solid_enough(parting_solid, "parting solid")

    report(0.34, "cutting cavity out of block")
    t0 = time.time()
    mold = _boolean(trimesh.boolean.difference, [block, cavity], "block - part")
    timings["cavity_cut"] = time.time() - t0
    validate.assert_solid_enough(mold, "mold (block - part)")

    report(0.6, "splitting along parting surface")
    t0 = time.time()
    half_a, half_b = split_mold(mold, parting_solid)
    timings["split"] = time.time() - t0
    validate.assert_solid_enough(half_a, "mold half A")
    validate.assert_solid_enough(half_b, "mold half B")

    stats = {
        "block_volume_cm3": round(float(block.volume) / 1000.0, 2),
        "cavity_volume_cm3": round(float(cavity.volume) / 1000.0, 2),
        "mold_volume_cm3": round(float(mold.volume) / 1000.0, 2),
        "half_a_volume_cm3": round(float(half_a.volume) / 1000.0, 2),
        "half_b_volume_cm3": round(float(half_b.volume) / 1000.0, 2),
        "split_volume_error_cm3": round(
            abs(float(half_a.volume + half_b.volume - mold.volume)) / 1000.0, 5
        ),
        "parting": surface.report(),
    }
    log.info("mold stats %s", stats)
    report(0.72, "halves cut")

    return MoldResult(
        direction=frame.direction.copy(),
        frame=frame,
        block=block,
        mold=mold,
        half_a=half_a,
        half_b=half_b,
        surface=surface,
        parting_solid=parting_solid,
        local_bounds=local_bounds,
        cavity=cavity,
        timings=timings,
        stats=stats,
    )
