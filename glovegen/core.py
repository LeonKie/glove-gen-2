"""The core for a hollow cast, and the two things that stop it floating.

Everything before this module casts a **solid** positive: the mold is
``block - part`` and the cavity is filled. A glove is a wall, not a solid, so it
needs a second body inside the cavity -- the core -- with the cast forming in
the gap between them. The wall is then only as consistent as the core's
position, and a core that is merely *in* the cavity is not located: it has six
degrees of freedom and one of them is actively driven. A hollow printed core
displacing ~700 cm3 of silicone at 1.10 g/cm3 sees about 7.7 N of buoyancy
against maybe 1.5 N of self-weight, so it floats, and the wall goes thin on top
before it goes thick underneath.

Two mechanisms hold it, and they are complementary rather than alternative.

**The carrier plate** (:class:`~glovegen.config.CarrierConfig`) is a slab
trimmed off the cuff end of the block perpendicular to the pour axis and
printed as one body with the core hanging beneath it. It locates against the
*assembled* block: its dowels straddle the parting seam, symmetric about it, so
the plate references both halves equally and self-centres rather than
inheriting one half's key clearance. Screws hold it down, because the load is
uplift. It owns gross position and takes the buoyancy, but the core still hangs
off it as a cantilever, and tip deflection goes as length cubed.

**Seam tabs** (:class:`~glovegen.config.CoreTabConfig`) take the moment the
plate cannot. Each is a post reaching from the core out past the cast
silhouette into mold that is solid on both sides of the parting face, where
closing the halves pinches it. They are the one support whose witness mark is
free: a tab crosses the glove wall exactly on the seam line, which is already a
flash ridge that gets trimmed.

The invariant that makes the whole assembly work
------------------------------------------------
**Every core-side feature is centred on the parting surface**, exactly like an
alignment key or the pour spout. That is not decoration; it is what makes the
assembly sequence exist at all. A tab lying in the parting plane, a dowel
running along the pour axis but centred on the seam, the neck likewise: each
leaves a half-round groove in either half, widest exactly at its mouth, so the
whole core assembly lifts straight out of half B along ``+d`` and half A closes
back down onto it. Put a dowel wholly inside one half instead and it has to be
inserted along the pour axis, which the tabs -- trapped sideways in their
grooves -- make impossible. The two options are only compatible because both
obey the same rule.

So the sequence is the ordinary moldmaker's one: core assembly into half B,
half A down on top, screws through the plate.

What this does not solve
------------------------
Tabs stick out sideways through the glove wall, so withdrawing the core along
the cuff axis drags them through the slots they made. On a flexible cast that
stretches; on a stiff one it tears. :func:`fixate` measures the volume the
glove has to stretch over and reports it rather than assuming it is fine.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import ConvexHull

from . import demold, meshio, mold as mold_mod, validate
from .config import MoldConfig
from .features import (
    FeatureSkipped,
    frustum,
    place_local,
    spout_entry,
)
from .frame import Frame, unit
from .mold import local_to_world
from .parting import PartingSurface

log = logging.getLogger(__name__)

# How far the trimming slabs overhang the block, in mm. Same reasoning as
# parting._SOLID_OVERHANG_MM: a cutting solid whose walls land exactly on the
# block's own walls hands the boolean coplanar faces and a fan of needle
# triangles.
_OVERHANG_MM = 20.0

# Boolean merge depth, mm. Two solids that meet exactly face-to-face are a
# coplanar boolean; overlapping them slightly is not.
_MERGE_MM = 1.5


# --------------------------------------------------------------------------
# the core body
# --------------------------------------------------------------------------


def erode(mesh: trimesh.Trimesh, delta: float, *, subdivisions: int = 1) -> trimesh.Trimesh:
    """Shrink ``mesh`` inward by ``delta`` mm: the exact inverse of an offset.

    This is the one genuinely expensive operation a core run adds. It is a
    Minkowski *difference* against a ball, which is exact -- up to the ball's
    tessellation, which undershoots at facet centres -- but scales badly with
    face count, so callers decimate first.
    """
    if delta <= 0:
        raise ValueError(f"erode needs a positive wall, got {delta}")
    from manifold3d import Manifold

    ball = trimesh.creation.icosphere(subdivisions=int(subdivisions), radius=float(delta))
    out = mold_mod.from_manifold(
        Manifold.minkowski_difference(
            mold_mod.to_manifold(mesh), mold_mod.to_manifold(ball)
        )
    )
    if len(out.faces) == 0:
        raise ValueError(
            f"eroding by {delta} mm left nothing: the part is thinner than "
            "twice the wall everywhere"
        )
    return out


def _assert_core_solid(body: trimesh.Trimesh, name: str) -> trimesh.Trimesh:
    """Gate a Minkowski result, which touches itself far more than a boolean does.

    An erosion pinches wherever the part narrows toward twice the wall -- at a
    finger web, at a knuckle -- and every pinch line is a run of legitimate
    position-coincident vertices. Re-welding those on the way back into trimesh
    turns them into non-manifold edges, so the usual 0.2%-of-faces budget fails
    perfectly good geometry: manifold3d's own ``decompose()`` still reports one
    solid, and the downstream booleans consume it without complaint.

    What is *not* relaxed is the check that matters. A boundary edge means a
    hole, and a core with a hole in it is not a core.
    """
    return validate.assert_solid_enough(
        body, name, max_nonmanifold_edges=max(256, int(len(body.faces) * 0.05))
    )


def cuff_offset(cavity: trimesh.Trimesh, pour_axis, cfg: MoldConfig) -> float:
    """Where the glove's rim sits, as a distance along the pour axis.

    Above this the core fills the cavity completely, so no cast forms and the
    cuff is open across the whole wrist section. Without it the erosion caps the
    cuff with a wall-thick membrane and the glove has no opening.
    """
    top = float((np.asarray(cavity.vertices) @ unit(pour_axis)).max())
    return top - float(cfg.core.cuff_depth)


def build_core_body(
    part: trimesh.Trimesh, pour_axis, cfg: MoldConfig
) -> trimesh.Trimesh:
    """The core: the part eroded by the wall, opened out at the cuff.

    The union with the full-section slab above the cuff plane is what turns a
    closed bladder into a glove -- it is also a shut-off, the core touching the
    cavity wall over that band, which is how a real cuff edge is formed.
    """
    p = unit(pour_axis)
    source = part
    if cfg.core.faces and len(part.faces) > cfg.core.faces:
        source = meshio.decimate(part, cfg.core.faces)

    body = erode(source, cfg.core.wall, subdivisions=cfg.core.ball_subdivisions)

    if cfg.core.cuff_depth > 0:
        plug_frame = Frame.from_direction(p)
        cut = cuff_offset(source, p, cfg)
        local = plug_frame.to_local(source.vertices)
        slab = _slab(plug_frame, local, cut, float(local[:, 2].max()) + _OVERHANG_MM)
        plug = trimesh.boolean.intersection(
            [source, slab], engine="manifold", check_volume=False
        )
        if len(plug.faces):
            body = trimesh.boolean.union(
                [body, plug], engine="manifold", check_volume=False
            )

    _assert_core_solid(body, "core body")
    return body


# --------------------------------------------------------------------------
# slabs and trimming
# --------------------------------------------------------------------------


def _slab(frame: Frame, local_pts: np.ndarray, w_lo: float, w_hi: float) -> trimesh.Trimesh:
    """A box spanning ``[w_lo, w_hi]`` along the frame's +Z, wide enough for all of it."""
    lo = np.asarray(local_pts, dtype=np.float64).min(axis=0) - _OVERHANG_MM
    hi = np.asarray(local_pts, dtype=np.float64).max(axis=0) + _OVERHANG_MM
    lo[2], hi[2] = float(w_lo), float(w_hi)
    if hi[2] <= lo[2]:
        raise ValueError(f"empty slab: [{w_lo}, {w_hi}]")
    transform = local_to_world(frame).copy()
    transform[:3, 3] = frame.to_world((lo + hi) / 2.0)
    return trimesh.creation.box(extents=(hi - lo), transform=transform)


def plate_planes(cavity: trimesh.Trimesh, pour_axis, cfg: MoldConfig) -> tuple[float, float]:
    """``(seat, top)`` offsets along the pour axis bounding the carrier plate.

    ``seat`` is the plate's seating face, held clear of the cavity so the plate
    lands on solid mold rather than on the lip of the cuff opening.
    """
    p = unit(pour_axis)
    top_of_cavity = float((np.asarray(cavity.vertices) @ p).max())
    seat = top_of_cavity + float(cfg.carrier.plate_gap)
    return seat, seat + float(cfg.carrier.plate_thickness)


def trim_plate(
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    pour_frame: Frame,
    seat: float,
    top: float,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh]:
    """Cut the plate off the cuff end of the mold.

    The plate is the *cap of the already-featured mold*, which is why this runs
    after the keys, spout and vents are cut rather than before: every hole that
    passes out through the cuff face is then already in the plate, and the pour
    spout becomes a port through it for free.

    Everything above ``top`` is discarded, so the plate has parallel faces
    instead of being a corner wedge sliced off an oblique box.
    """
    corners = np.vstack(
        [pour_frame.to_local(half_a.vertices), pour_frame.to_local(half_b.vertices)]
    )
    above_seat = _slab(pour_frame, corners, seat, float(corners[:, 2].max()) + _OVERHANG_MM)
    plate_box = _slab(pour_frame, corners, seat, top)

    caps = []
    for half, name in ((half_a, "A"), (half_b, "B")):
        cap = trimesh.boolean.intersection(
            [half, plate_box], engine="manifold", check_volume=False
        )
        if len(cap.faces):
            caps.append(cap)
        else:
            log.info("half %s contributes nothing to the carrier plate", name)
    if not caps:
        raise FeatureSkipped(
            "the block does not reach past the cavity along the pour axis, so "
            "there is nothing to make a carrier plate from"
        )
    plate = (
        caps[0]
        if len(caps) == 1
        else trimesh.boolean.union(caps, engine="manifold", check_volume=False)
    )

    half_a = trimesh.boolean.difference(
        [half_a, above_seat], engine="manifold", check_volume=False
    )
    half_b = trimesh.boolean.difference(
        [half_b, above_seat], engine="manifold", check_volume=False
    )
    return half_a, half_b, plate


# --------------------------------------------------------------------------
# reading the parting surface
# --------------------------------------------------------------------------


def _h_at(surface: PartingSurface, local_pts: np.ndarray) -> np.ndarray:
    """Parting height at the nearest grid node to each local (x, y)."""
    pts = np.asarray(local_pts, dtype=np.float64).reshape(-1, 3)
    i = np.clip(np.searchsorted(surface.xs, pts[:, 0]), 0, len(surface.xs) - 1)
    j = np.clip(np.searchsorted(surface.ys, pts[:, 1]), 0, len(surface.ys) - 1)
    return surface.h[i, j]


def seam_drift(
    surface: PartingSurface,
    start_world: np.ndarray,
    axis_world: np.ndarray,
    length: float,
    *,
    samples: int = 13,
) -> float:
    """How far a straight bore wanders off the (curved) parting surface, in mm.

    A channel centred on the parting surface leaves each half a groove that is
    widest exactly at its mouth, so the core lifts out. A channel that drifts
    off it leaves a groove wider inside than at its mouth, which is a lock. This
    is the number that decides which one you have.
    """
    ts = np.linspace(0.0, float(length), int(samples))
    pts = np.asarray(start_world).reshape(1, 3) + ts[:, None] * unit(axis_world)[None, :]
    local = surface.frame.to_local(pts)
    return float(np.abs(local[:, 2] - _h_at(surface, local)).max())


def core_at_parting(core: trimesh.Trimesh, surface: PartingSurface) -> np.ndarray:
    """Grid mask: does the parting surface pass through the core at this node?

    Where the part is thicker than twice the wall, the parting surface -- which
    aims for the middle of the thickest slab of material -- runs *inside* the
    core. Those nodes are where a tab can start, because the tab's inner end is
    then buried in core material and the union is clean.
    """
    xs, ys = surface.xs, surface.ys
    cast = demold.cast_grid(core, surface.frame, xs, ys)
    ncol = cast.ncol
    if len(cast.t) == 0:
        return np.zeros((len(xs), len(ys)), dtype=bool)

    col = np.repeat(np.arange(ncol), cast.counts)
    z = cast.local_z(cast.t)
    h_flat = surface.h.reshape(-1)
    # Crossings are sorted within a column, so a point is inside the solid iff
    # an odd number of them lie below it.
    below = z < h_flat[col]
    n_below = np.bincount(col[below], minlength=ncol)
    return (n_below % 2 == 1).reshape(len(xs), len(ys))


def nodes_below(surface: PartingSurface, pour_axis, offset: float) -> np.ndarray:
    """Grid mask: the parting node's world point is below ``offset`` along the pour axis."""
    p = unit(pour_axis)
    nx, ny = surface.shape
    gx, gy = np.meshgrid(surface.xs, surface.ys, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), surface.h.ravel()])
    return ((surface.frame.to_world(pts) @ p) <= offset).reshape(nx, ny)


def _node_world(surface: PartingSurface, i: int, j: int) -> np.ndarray:
    return surface.frame.to_world(
        np.array([surface.xs[i], surface.ys[j], surface.h[i, j]])
    )


def free_depth(
    cavity: trimesh.Trimesh,
    start: np.ndarray,
    axis: np.ndarray,
    limit: float,
    radius: float,
    *,
    rays: int = 8,
) -> float:
    """How deep a bore of this radius can go before it breaks into the cavity.

    Measured rather than tested pass/fail, because at the cuff the cavity's
    ceiling is only a millimetre below the plate's seating face and a fixed
    depth would reject every position on the seam. A dowel that engages six
    millimetres is a dowel; a dowel that was rejected is nothing.

    Rays around the bore's circumference, not one down its axis: a bore that
    grazes the cavity wall is still a bore that opens into the cast.
    """
    axis = unit(axis)
    frame = Frame.from_direction(axis)
    theta = np.linspace(0.0, 2.0 * np.pi, rays, endpoint=False)
    ring = np.column_stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros(rays)]
    )
    origins = np.vstack([start + frame.to_world(ring), start])
    dirs = np.broadcast_to(axis, origins.shape)
    hits, index_ray = cavity.ray.intersects_location(
        origins, dirs, multiple_hits=False
    )[:2]
    if len(hits) == 0:
        return float(limit)
    reach = (hits - origins[index_ray]) @ axis
    ahead = reach[reach > 0.0]
    if len(ahead) == 0:
        return float(limit)
    return float(min(limit, ahead.min()))


# --------------------------------------------------------------------------
# option C: seam tabs
# --------------------------------------------------------------------------


@dataclass
class TabSite:
    inner: np.ndarray  # world, on the parting surface inside the core
    outer: np.ndarray  # world, on the parting surface in solid mold
    length: float
    clearance: float  # mold around the outer end, beyond the tab's own radius

    def as_dict(self) -> dict:
        return {
            "inner_world": [round(float(v), 2) for v in self.inner],
            "outer_world": [round(float(v), 2) for v in self.outer],
            "length_mm": round(self.length, 2),
            "anchor_clearance_mm": round(self.clearance, 2),
        }


def find_tab_sites(
    surface: PartingSurface,
    in_core: np.ndarray,
    free_dist: np.ndarray,
    allow: np.ndarray,
    cfg: MoldConfig,
) -> list[TabSite]:
    """Pick tab runs from the core out to well-supported mold.

    The placement is the alignment-key logic run in reverse. A key wants a
    column that *misses* the part; a tab wants both -- an outer end in a column
    that misses it, with mold on either side of the parting face, and an inner
    end in a column where the parting surface is inside the core. A distance
    transform away from the core gives every free node both its nearest core
    node and the length of the run between them in one pass.
    """
    tcfg = cfg.core_tabs
    nx, ny = surface.shape
    dx = float(surface.xs[1] - surface.xs[0]) if nx > 1 else 1.0
    dy = float(surface.ys[1] - surface.ys[0]) if ny > 1 else 1.0

    if not in_core.any():
        log.warning("the parting surface never passes through the core; no tabs")
        return []

    dist, idx = ndimage.distance_transform_edt(
        ~in_core, sampling=(dx, dy), return_indices=True
    )

    need = tcfg.radius + tcfg.anchor_margin
    ok = (
        allow
        & ~surface.constrained
        & (free_dist >= need)
        & (dist >= tcfg.min_length)
        & (dist <= tcfg.max_length)
    )
    # Keep clear of the block's own outside walls.
    border_x = int(np.ceil(need / dx))
    border_y = int(np.ceil(need / dy))
    edge = np.zeros_like(ok)
    edge[border_x : nx - border_x, border_y : ny - border_y] = True
    ok &= edge

    cand = np.argwhere(ok)
    if len(cand) == 0:
        log.warning("no room on the parting face for core tabs")
        return []

    # Spread by where each tab *grips the core*, not by where it is anchored:
    # two tabs anchored far apart can still grab the same spot on the core, and
    # bracing a cantilever needs them spread along its length.
    inner_ij = np.column_stack(
        [idx[0][ok], idx[1][ok]]
    )
    inner_pts = np.column_stack(
        [surface.xs[inner_ij[:, 0]], surface.ys[inner_ij[:, 1]]]
    )
    room = free_dist[ok]

    chosen = [int(np.argmax(room))]
    while len(chosen) < tcfg.count:
        gap = np.min(
            np.linalg.norm(
                inner_pts[:, None, :] - inner_pts[chosen][None, :, :], axis=2
            ),
            axis=1,
        )
        gap[gap < tcfg.min_spacing] = -np.inf
        # Among the well-separated candidates, take the most generously
        # supported one rather than merely the furthest.
        score = np.where(np.isfinite(gap), room + 0.25 * gap, -np.inf)
        nxt = int(np.argmax(score))
        if not np.isfinite(score[nxt]):
            break
        chosen.append(nxt)

    sites = []
    for c in chosen:
        i, j = cand[c]
        ii, jj = inner_ij[c]
        sites.append(
            TabSite(
                inner=_node_world(surface, int(ii), int(jj)),
                outer=_node_world(surface, int(i), int(j)),
                length=float(dist[i, j]),
                clearance=float(free_dist[i, j]),
            )
        )
    return sites


def _tab_solids(
    site: TabSite, surface: PartingSurface, cfg: MoldConfig
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """``(tab, pocket)``: the post to add to the core and the groove to cut."""
    tcfg = cfg.core_tabs
    frame = surface.frame
    inner = frame.to_local(site.inner)
    outer = frame.to_local(site.outer)
    axis_local = outer - inner
    length = float(np.linalg.norm(axis_local))
    if length < tcfg.min_length:
        raise FeatureSkipped(f"tab would be only {length:.1f} mm long")

    drift = seam_drift(surface, site.inner, frame.to_world(axis_local), length)
    if drift > tcfg.max_seam_drift:
        raise FeatureSkipped(
            f"the parting surface bends {drift:.2f} mm over the tab's {length:.0f} mm "
            f"run (limit {tcfg.max_seam_drift} mm); its groove would be an undercut"
        )

    axis_local = axis_local / length
    # Start inside the core so the union has something to bite on.
    back = _MERGE_MM + tcfg.radius
    start = inner - axis_local * back
    r, c = tcfg.radius, tcfg.clearance

    tab = place_local(
        frustum(r, r, length + back, sections=32), frame, start, axis_local
    )
    pocket = place_local(
        frustum(r + c, r + c, length + back + c, sections=32), frame, start, axis_local
    )
    return tab, pocket


# --------------------------------------------------------------------------
# option B: the carrier plate's registration
# --------------------------------------------------------------------------


@dataclass
class BoreSite:
    """A hole down the pour axis: a seam dowel or a screw."""

    start: np.ndarray  # world, on the plate's seating plane
    depth: float
    drift: float  # off the parting surface, mm; 0 means dead on the seam
    side: str  # "seam", "a" or "b"

    def as_dict(self) -> dict:
        return {
            "start_world": [round(float(v), 2) for v in self.start],
            "depth_mm": round(self.depth, 2),
            "seam_drift_mm": round(self.drift, 3),
            "side": self.side,
        }


def _seat_candidates(
    surface: PartingSurface, pour_axis, seat: float, tol: float
) -> np.ndarray:
    """Parting-surface nodes whose world point lands on the plate's seating plane.

    These trace the seam line across the seating face -- the only places a dowel
    can go if it is to reference both halves.
    """
    p = unit(pour_axis)
    nx, ny = surface.shape
    gx, gy = np.meshgrid(surface.xs, surface.ys, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), surface.h.ravel()])
    w = surface.frame.to_world(pts) @ p
    keep = np.abs(w - seat) <= tol
    return np.argwhere(keep.reshape(nx, ny))


def find_dowel_sites(
    surface: PartingSurface,
    cavity: trimesh.Trimesh,
    plate: trimesh.Trimesh,
    pour_axis,
    seat: float,
    cfg: MoldConfig,
) -> list[BoreSite]:
    """Two pins on the seam, as far apart as the seating face allows.

    Straddling the seam is the whole point. A plate dowelled into one half
    inherits that half's key clearance on top of its own fit, so the core's
    position error is two interfaces deep instead of one; a pin shared between
    the halves references both equally and self-centres.
    """
    ccfg = cfg.carrier
    p = unit(pour_axis)
    nx, ny = surface.shape
    cell = float(surface.xs[1] - surface.xs[0]) if nx > 1 else 1.0

    idx = _seat_candidates(surface, p, seat, max(cell, 1.0))
    if len(idx) == 0:
        log.warning("the parting seam does not reach the plate's seating face")
        return []

    inside = _inside_footprint(plate, pour_axis, ccfg.dowel_radius + 1.5)

    scored: list[BoreSite] = []
    for i, j in idx:
        start = _node_world(surface, int(i), int(j))
        if not inside(start):
            continue
        drift = seam_drift(surface, start, -p, ccfg.dowel_depth)
        if drift > ccfg.max_seam_drift:
            continue
        room = free_depth(
            cavity, start, -p, ccfg.dowel_depth + _MERGE_MM, ccfg.dowel_radius + 1.0
        )
        depth = min(ccfg.dowel_depth, room - _MERGE_MM)
        if depth < ccfg.dowel_min_depth:
            continue
        scored.append(BoreSite(start=start, depth=depth, drift=drift, side="seam"))

    if not scored:
        log.warning(
            "no seam point on the seating face clears the cavity by %.1f mm, so "
            "the plate has nothing to register against",
            ccfg.dowel_min_depth,
        )
        return []

    # Farthest-point along the seam: two dowels close together are one dowel
    # and a wobble.
    picked = [max(scored, key=lambda s: s.depth)]
    for _ in range(1, ccfg.dowel_count):
        gaps = [
            min(float(np.linalg.norm(s.start - q.start)) for q in picked) for s in scored
        ]
        best = int(np.argmax(gaps))
        if gaps[best] < ccfg.dowel_min_spacing:
            break
        picked.append(scored[best])
    return picked


def _inside_footprint(plate: trimesh.Trimesh, pour_axis, inset: float):
    """Predicate: is this world point inside the plate's outline, inset by ``inset``?"""
    frame = Frame.from_direction(pour_axis)
    uv = frame.to_local(plate.vertices)[:, :2]
    hull = ConvexHull(uv)
    # Inward half-plane test. ConvexHull's equations are outward-facing
    # (n . x + d <= 0 inside), so pushing d by the inset shrinks the polygon.
    a = hull.equations[:, :2]
    b = hull.equations[:, 2]

    def predicate(world_point) -> bool:
        q = frame.to_local(np.asarray(world_point).reshape(1, 3))[0, :2]
        return bool(np.all(a @ q + b + inset <= 0.0))

    return predicate


def find_screw_sites(
    surface: PartingSurface,
    cavity: trimesh.Trimesh,
    plate: trimesh.Trimesh,
    pour_axis,
    seat: float,
    cfg: MoldConfig,
    *,
    avoid: list[np.ndarray],
) -> list[BoreSite]:
    """Clamping screws around the plate's edge, alternating between the halves.

    Unlike everything else here a screw is inserted last, after the mold is
    closed, so it is under no release constraint at all -- it only has to miss
    the cavity and land wholly inside one half rather than straddling the seam,
    where it would jack the halves apart instead of clamping them down.
    """
    ccfg = cfg.carrier
    if ccfg.screw_count <= 0:
        return []
    p = unit(pour_axis)
    frame = Frame.from_direction(p)
    clear = ccfg.screw_radius + ccfg.screw_clearance

    uv = frame.to_local(plate.vertices)[:, :2]
    probes = _ring_inset(uv, clear + 3.0, max(8 * ccfg.screw_count, 32))
    if len(probes) == 0:
        log.warning("the carrier plate is too narrow to take screws")
        return []

    candidates: dict[str, list[BoreSite]] = {"a": [], "b": []}
    for u, v in probes:
        start = frame.to_world(np.array([u, v, seat]))
        if any(np.linalg.norm(start - q) < clear + ccfg.dowel_radius + 4.0 for q in avoid):
            continue
        # Wholly inside one half: sample the bore and demand it stay on one
        # side of the parting surface by more than its own radius.
        ts = np.linspace(0.0, ccfg.screw_depth, 9)
        local = surface.frame.to_local(start[None, :] - ts[:, None] * p[None, :])
        gap = local[:, 2] - _h_at(surface, local)
        if np.abs(gap).min() < clear + 1.0 or not (
            np.all(gap > 0.0) or np.all(gap < 0.0)
        ):
            continue
        room = free_depth(cavity, start, -p, ccfg.screw_depth + _MERGE_MM, clear + 1.0)
        depth = min(ccfg.screw_depth, room - _MERGE_MM)
        if depth < ccfg.screw_min_depth:
            continue
        side = "a" if gap[0] > 0 else "b"
        candidates[side].append(
            BoreSite(
                start=start, depth=depth, drift=float(np.abs(gap).min()), side=side
            )
        )

    # Alternate sides. Taking valid positions in the order they come off the
    # ring walks one side of the seam before reaching the other, so the whole
    # screw budget lands in one half and the other is held down by nothing.
    spacing = 2.5 * clear + 6.0
    picked: list[BoreSite] = []
    while len(picked) < ccfg.screw_count:
        counts = {s: sum(1 for q in picked if q.side == s) for s in ("a", "b")}
        order = sorted(("a", "b"), key=lambda s: counts[s])
        took = False
        for side in order:
            pool = [
                s
                for s in candidates[side]
                if all(np.linalg.norm(s.start - q.start) >= spacing for q in picked)
            ]
            if not pool:
                continue
            picked.append(
                max(
                    pool,
                    key=lambda s: min(
                        (float(np.linalg.norm(s.start - q.start)) for q in picked),
                        default=s.depth,
                    ),
                )
            )
            took = True
            break
        if not took:
            break

    if len(picked) < ccfg.screw_count:
        log.warning(
            "only %d of %d screw positions clear the cavity and stay wholly "
            "inside one half",
            len(picked),
            ccfg.screw_count,
        )
    return picked


def _ring_inset(uv: np.ndarray, inset: float, count: int) -> np.ndarray:
    """``count`` points spread around the outline of ``uv``, ``inset`` mm inside it.

    Walking the outline at even arc length rather than taking its corners is
    what keeps screws spread along a long plate instead of bunched at its ends.
    The inset is per-edge -- scaling the whole polygon toward its centroid pulls
    a 150 x 78 mm plate in twice as far on its long axis as on its short one,
    and puts the screws in the wrong place.
    """
    hull = ConvexHull(uv)
    ring = uv[hull.vertices]
    closed = np.vstack([ring, ring[:1]])

    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    walk = np.concatenate([[0.0], np.cumsum(seg)])
    if walk[-1] <= 1e-9:
        return np.zeros((0, 2))
    targets = np.linspace(0.0, walk[-1], int(count), endpoint=False)
    on_ring = np.column_stack(
        [np.interp(targets, walk, closed[:, 0]), np.interp(targets, walk, closed[:, 1])]
    )

    # Outward half-planes: n . x + d <= 0 holds inside. Pushing every plane in
    # by `inset` and keeping only the points that still satisfy all of them is
    # an exact test, so a point near a corner is dropped rather than nudged to
    # somewhere it does not belong.
    a, b = hull.equations[:, :2], hull.equations[:, 2]
    inward = -a[np.argmin(np.abs(on_ring @ a.T + b), axis=1)]
    moved = on_ring + inward * inset
    # The tolerance is not cosmetic: a point taken from the middle of an edge
    # and pushed in by `inset` lands at exactly `inset` from that edge, so an
    # exact test is decided by the sign of the float error and throws away most
    # of the ring. Corner points still fail it by millimetres.
    keep = np.all(moved @ a.T + b <= -inset + 1e-6, axis=1)
    return moved[keep]


# --------------------------------------------------------------------------
# putting it together
# --------------------------------------------------------------------------


@dataclass
class CoreResult:
    """A hollow-cast core, fixated, plus the mold it fits."""

    core: trimesh.Trimesh  # printable: plate + neck + core body + tabs + dowels
    half_a: trimesh.Trimesh
    half_b: trimesh.Trimesh
    report: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


def _measure_wall(
    core_body: trimesh.Trimesh,
    cavity: trimesh.Trimesh,
    pour_axis,
    cut: float,
    target_mm: float,
    *,
    samples: int = 4000,
) -> dict:
    """Measured wall thickness, not assumed.

    Distance from a point on the core's surface to the cavity's surface *is* the
    wall there, so this checks the erosion delivered what was asked and prices
    the eroding ball's tessellation: a polyhedral ball undershoots at its facet
    centres and never overshoots, costing about 6.5% of the wall at
    ``ball_subdivisions=1``, 1.8% at 2 and 0.4% at 3.

    ``cut`` excludes the cuff, and has to sit a wall *below* the rim rather than
    at it. Above the rim the core deliberately touches the cavity to open the
    glove, and the shut-off disc meets that wall at its edge -- sample a point
    on the rim itself and it reads as near-zero wall, which is not a thin spot
    in the glove but the hole the hand goes through.
    """
    pts = core_body.sample(int(samples))
    keep = (pts @ unit(pour_axis)) < cut
    if keep.sum() < 32:
        return {}
    pts = pts[keep]
    dist = trimesh.proximity.closest_point(cavity, pts)[1]
    target = float(target_mm)
    return {
        "target_mm": round(target, 3),
        "min_mm": round(float(dist.min()), 3),
        "p05_mm": round(float(np.percentile(dist, 5)), 3),
        "median_mm": round(float(np.median(dist)), 3),
        "max_mm": round(float(dist.max()), 3),
        # A single worst sample says nothing about how much of the glove is
        # thin. This does, and it is the number to act on.
        "under_90pct_fraction": round(float((dist < 0.9 * target).mean()), 5),
        "samples": int(len(pts)),
    }


def fixate(
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    ctx,
    part: trimesh.Trimesh,
    pour_axis,
    cfg: MoldConfig,
    *,
    verify: bool = True,
    progress=None,
) -> CoreResult:
    """Build the core and lock it to the mold: options B and C together.

    ``half_a`` / ``half_b`` are the halves *with their features already cut*.
    That ordering is deliberate -- the plate is the cap of the finished mold, so
    the spout and any vents that exit through the cuff face come through into it
    as ports without a line of code.

    ``ctx`` is the :class:`~glovegen.features.FeatureContext` the features were
    cut with: the parting surface, the block's bounds and the cavity.
    """
    report_progress = progress or (lambda *a, **k: None)
    timings: dict[str, float] = {}
    p = unit(pour_axis)
    pour_frame = Frame.from_direction(p)
    up_local = pour_frame.to_local(p)  # +Z by construction; spelled out for clarity
    surface = ctx.surface
    cavity = ctx.cavity
    out: dict = {
        "wall_target_mm": round(float(cfg.core.wall), 3),
        "pour_axis": [round(float(v), 4) for v in p],
        "skipped": [],
    }

    def skip(what: str, why: str) -> None:
        log.info("core: skipping %s: %s", what, why)
        out["skipped"].append({"what": what, "reason": why})

    report_progress(0.05, f"eroding the part by {cfg.core.wall} mm")
    t0 = time.time()
    core_body = build_core_body(part, p, cfg)
    timings["core_erode"] = time.time() - t0
    log.info(
        "core body: %d faces, %.1f cm3 (part %.1f cm3)",
        len(core_body.faces),
        core_body.volume / 1000.0,
        part.volume / 1000.0,
    )

    pieces = [core_body]
    screw_holes: list[trimesh.Trimesh] = []
    seat, top = plate_planes(cavity, p, cfg)

    # -- option B: the plate, its neck, and the registration ---------------
    plate = None
    if cfg.carrier.enabled:
        report_progress(0.35, "trimming the carrier plate")
        t0 = time.time()
        try:
            half_a, half_b, plate = trim_plate(half_a, half_b, pour_frame, seat, top)
        except FeatureSkipped as exc:
            skip("carrier plate", str(exc))
        else:
            pieces.append(plate)
            # Measured, not requested: the block may not reach the full
            # thickness past the cavity, in which case the plate is whatever
            # material was actually there.
            reach = np.asarray(plate.vertices) @ p
            out["plate"] = {
                "seat_offset_mm": round(seat, 2),
                "thickness_mm": round(float(reach.max() - reach.min()), 2),
                "asked_thickness_mm": round(top - seat, 2),
                "volume_cm3": round(float(plate.volume) / 1000.0, 2),
                "pieces": _real_pieces(plate),
            }
        timings["core_plate"] = time.time() - t0

    if plate is not None:
        # The neck: a stem from inside the cuff plug up to the plate's outer
        # face. It runs entirely through mold material above the cavity, so
        # unlike a tab it leaves no witness on the glove at all.
        entry = spout_entry(cavity, ctx, p)
        engage = max(float(cfg.core.cuff_depth), _MERGE_MM)
        start = entry - p * engage
        length = top - float(start @ p)
        drift = seam_drift(surface, start, p, length)
        r, c = cfg.carrier.neck_radius, cfg.carrier.neck_clearance
        if drift > cfg.carrier.max_seam_drift:
            skip(
                "neck",
                f"the parting surface bends {drift:.2f} mm over the neck's "
                f"{length:.0f} mm run (limit {cfg.carrier.max_seam_drift} mm)",
            )
        else:
            neck = place_local(
                frustum(r, r, length, sections=48),
                pour_frame,
                pour_frame.to_local(start),
                up_local,
            )
            channel = place_local(
                frustum(r + c, r + c, length, sections=48),
                pour_frame,
                pour_frame.to_local(start),
                up_local,
            )
            pieces.append(neck)
            half_a = _cut(half_a, channel, "neck -> A")
            half_b = _cut(half_b, channel, "neck -> B")
            out["neck"] = {
                "radius_mm": round(r, 2),
                "length_mm": round(length, 2),
                "seam_drift_mm": round(drift, 3),
            }

        report_progress(0.5, "placing dowels and screws")
        dowels = find_dowel_sites(surface, cavity, plate, p, seat, cfg)
        if not dowels:
            skip("dowels", "no seam point on the seating face can take one")
        for n, site in enumerate(dowels, start=1):
            dr, dc = cfg.carrier.dowel_radius, cfg.carrier.dowel_clearance
            origin = pour_frame.to_local(site.start - p * site.depth)
            pieces.append(
                place_local(
                    frustum(dr, dr, site.depth + _MERGE_MM, sections=32),
                    pour_frame,
                    origin,
                    up_local,
                )
            )
            bore = place_local(
                frustum(dr + dc, dr + dc, site.depth + _MERGE_MM * 2.0, sections=32),
                pour_frame,
                pour_frame.to_local(site.start - p * (site.depth + _MERGE_MM)),
                up_local,
            )
            half_a = _cut(half_a, bore, f"dowel {n} -> A")
            half_b = _cut(half_b, bore, f"dowel {n} -> B")
        out["dowels"] = [s.as_dict() for s in dowels]

        avoid = [s.start for s in dowels] + (
            [np.asarray(entry) + p * (seat - float(np.asarray(entry) @ p))]
            if "neck" in out
            else []
        )
        screws = find_screw_sites(
            surface, cavity, plate, p, seat, cfg, avoid=avoid
        )
        for n, site in enumerate(screws, start=1):
            sr = cfg.carrier.screw_radius
            through = cfg.carrier.screw_radius + cfg.carrier.screw_clearance
            pilot = place_local(
                frustum(sr, sr, site.depth + _MERGE_MM, sections=24),
                pour_frame,
                pour_frame.to_local(site.start - p * (site.depth + _MERGE_MM)),
                up_local,
            )
            half_a = _cut(half_a, pilot, f"screw {n} -> A")
            half_b = _cut(half_b, pilot, f"screw {n} -> B")
            # The plate only becomes part of the assembly at the fuse below,
            # so its through-hole is cut from the assembly afterwards.
            screw_holes.append(
                place_local(
                    frustum(through, through, top - seat + 2.0 * _MERGE_MM, sections=24),
                    pour_frame,
                    pour_frame.to_local(site.start - p * _MERGE_MM),
                    up_local,
                )
            )
        out["screws"] = [s.as_dict() for s in screws]

    # -- option C: seam tabs ------------------------------------------------
    tabs: list[dict] = []
    tab_solids: list[trimesh.Trimesh] = []
    if cfg.core_tabs.enabled:
        report_progress(0.65, "placing seam tabs")
        t0 = time.time()
        in_core = core_at_parting(core_body, surface)
        # A tab that reached above the plate's seating face would be cut in
        # half by the trim. With no plate there is no ceiling to respect.
        allow = (
            nodes_below(surface, p, seat - cfg.core_tabs.radius)
            if plate is not None
            else np.ones(surface.shape, dtype=bool)
        )
        for n, site in enumerate(
            find_tab_sites(surface, in_core, ctx.free_distance(), allow, cfg), start=1
        ):
            entry_report = site.as_dict()
            try:
                tab, pocket = _tab_solids(site, surface, cfg)
            except FeatureSkipped as exc:
                entry_report.update(status="skipped", reason=str(exc))
            else:
                tab_solids.append(tab)
                pieces.append(tab)
                half_a = _cut(half_a, pocket, f"tab {n} -> A")
                half_b = _cut(half_b, pocket, f"tab {n} -> B")
                entry_report.update(status="applied", reason="")
            tabs.append(entry_report)
        timings["core_tabs"] = time.time() - t0
    out["tabs"] = tabs

    # -- fuse the assembly --------------------------------------------------
    report_progress(0.8, "fusing the core assembly")
    t0 = time.time()
    assembly = (
        pieces[0]
        if len(pieces) == 1
        else trimesh.boolean.union(pieces, engine="manifold", check_volume=False)
    )
    for hole in screw_holes:
        assembly = _cut(assembly, hole, "screw clearance -> core")
    timings["core_assembly"] = time.time() - t0
    validate.assert_solid_enough(assembly, "core assembly")
    validate.assert_solid_enough(half_a, "half A (with core fixation)")
    validate.assert_solid_enough(half_b, "half B (with core fixation)")

    out["assembly"] = {
        "volume_cm3": round(float(assembly.volume) / 1000.0, 2),
        "faces": int(len(assembly.faces)),
        "pieces": _real_pieces(assembly),
    }

    if verify:
        report_progress(0.9, "measuring the wall and the core's release")
        t0 = time.time()
        wall = _measure_wall(
            core_body,
            cavity,
            p,
            cuff_offset(cavity, p, cfg) - cfg.core.wall,
            cfg.core.wall,
        )
        if wall:
            out["wall"] = wall
        out["release"] = release_report(assembly, half_a, half_b, ctx.direction)
        if tab_solids:
            out["tab_through_wall_mm3"] = _tab_protrusion(tab_solids, cavity, core_body)
        timings["core_verify"] = time.time() - t0

    report_progress(1.0, "core done")
    log.info("core report %s", {k: v for k, v in out.items() if k != "tabs"})
    return CoreResult(
        core=assembly, half_a=half_a, half_b=half_b, report=out, timings=timings
    )


def _real_pieces(mesh: trimesh.Trimesh, *, min_volume: float = 1.0) -> int:
    """Connected components worth counting.

    manifold3d legitimately emits a scatter of position-coincident, zero-volume
    shells alongside a boolean result -- the same artefact that makes trimesh
    call a perfectly closed solid non-watertight. Counting them would report a
    one-piece core as three.
    """
    return sum(
        1
        for part in mesh.split(only_watertight=False)
        if abs(float(part.volume)) >= min_volume
    )


def _cut(solid: trimesh.Trimesh, tool: trimesh.Trimesh, label: str) -> trimesh.Trimesh:
    out = trimesh.boolean.difference(
        [solid, tool], engine="manifold", check_volume=False
    )
    log.debug("%s -> %d faces", label, len(out.faces))
    return out


def _tab_protrusion(
    tabs: list[trimesh.Trimesh], cavity: trimesh.Trimesh, core_body: trimesh.Trimesh
) -> float:
    """How much tab the glove has to stretch over on its way off the core.

    The part of each tab that sits in the cast -- outside the core, inside the
    cavity -- passes through the glove wall. Pulling the core out along the cuff
    axis drags it through the slot it made, so this is the number that decides
    whether a given cast material tolerates option C or tears on it.
    """
    fused = (
        tabs[0]
        if len(tabs) == 1
        else trimesh.boolean.union(tabs, engine="manifold", check_volume=False)
    )
    inside = trimesh.boolean.intersection(
        [fused, cavity], engine="manifold", check_volume=False
    )
    if len(inside.faces) == 0:
        return 0.0
    in_cast = trimesh.boolean.difference(
        [inside, core_body], engine="manifold", check_volume=False
    )
    return round(abs(float(in_cast.volume)) if len(in_cast.faces) else 0.0, 2)


def release_report(
    core: trimesh.Trimesh,
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    direction,
    *,
    travel: float | None = None,
    tol_fraction: float = 1.0e-6,
) -> dict:
    """Does the core actually come out of the halves along the pull direction?

    Every core-side feature is supposed to be centred on the parting surface, so
    the answer is yes by construction -- which is exactly why it is worth
    measuring. A number here means a tab, dowel or neck drifted off the seam far
    enough for its groove to become a lock, and the mold cannot be opened
    without breaking something.
    """
    d = unit(direction)
    if travel is None:
        travel = max(5.0, 0.1 * float(core.scale))

    # Half A lives above the parting surface and lifts along +d, so relative to
    # A the core leaves along -d; half B is the mirror image. Getting these two
    # the wrong way round measures the core sinking further into each half,
    # which is not a motion anything performs.
    moved = core.copy()
    moved.apply_translation(-d * travel)
    overlap_a = validate.overlap_volume(moved, half_a)
    moved = core.copy()
    moved.apply_translation(d * travel)
    overlap_b = validate.overlap_volume(moved, half_b)

    scale = abs(float(core.volume)) or 1.0
    worst = max(overlap_a, overlap_b)
    return {
        "travel_mm": round(float(travel), 3),
        "core_vs_half_a_mm3": round(overlap_a, 6),
        "core_vs_half_b_mm3": round(overlap_b, 6),
        "fraction": round(worst / scale, 10),
        "releases": bool(worst / scale <= tol_fraction),
    }
