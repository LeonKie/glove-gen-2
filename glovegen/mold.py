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

from . import meshio, parting, validate
from .config import MoldConfig
from .frame import Frame
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


def offset_cavity(mesh: trimesh.Trimesh, delta: float) -> trimesh.Trimesh:
    """Grow the cavity by ``delta`` mm, for cast shrinkage or fit clearance.

    Uses an exact Minkowski sum, which is expensive on a full-resolution scan;
    the caller is expected to decimate first if it matters.
    """
    if abs(delta) < 1e-9:
        return mesh
    if delta < 0:
        raise NotImplementedError(
            "negative cavity_offset (shrinking the cavity) is not supported"
        )
    from manifold3d import Manifold, Mesh

    def to_manifold(m: trimesh.Trimesh) -> "Manifold":
        return Manifold(
            mesh=Mesh(
                vert_properties=np.asarray(m.vertices, dtype=np.float32),
                tri_verts=np.asarray(m.faces, dtype=np.uint32),
            )
        )

    ball = trimesh.creation.icosphere(subdivisions=1, radius=float(delta))
    grown = Manifold.minkowski_sum(to_manifold(mesh), to_manifold(ball)).to_mesh()
    out = trimesh.Trimesh(
        vertices=np.asarray(grown.vert_properties[:, :3], dtype=np.float64),
        faces=np.asarray(grown.tri_verts),
        process=False,
    )
    out.merge_vertices()
    return out


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
    frame = Frame.from_direction(direction)

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
