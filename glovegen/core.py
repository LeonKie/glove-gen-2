"""The core for a hollow cast, and the things that stop it floating.

Everything else in the pipeline casts a **solid** positive: the mold is
``block - part`` and the cavity is filled. A glove is a wall, not a solid, so it
needs a second body inside the cavity -- the core -- with the cast forming in
the gap between them. The wall is then only as consistent as the core's
position, and a core that is merely *in* the cavity is not located: it has six
degrees of freedom and one of them is actively driven. A hollow printed core
displacing ~700 cm3 of silicone at 1.10 g/cm3 sees about 7.7 N of buoyancy
against maybe 1.5 N of self-weight, so it floats, and the wall goes thin on top
before it goes thick underneath.

The core alone is what you get by asking for a wall. Everything below is opt-in
on top of it.

The carrier plate, and the cut that makes it
--------------------------------------------
A plate cannot be bolted onto a closed mold; there is nowhere for it to attach
to the core. So adding one **cuts the whole mold** with a plane through the
core and throws away everything on one side of it -- half A, half B and the
core together. That leaves three flat coplanar faces, and the plate is a slab
of the block's own cross-section laid across all of them: it caps the halves,
spans the annulus so the cast is sealed in, and swallows the core's stub so the
two become one printed body. The glove's rim is exactly the cut.

Screws hold the plate down, because the load is uplift.

Sealing the annulus means the pour has to come through the plate, which is what
the port is: a funnel down to the ring of cast at the cut face. With a plate in
the plan the ordinary spout is redundant, and is reported as such rather than
cut into material the plane is about to discard.

Holding the core still: tabs and dowels
---------------------------------------
The plate closes the mold but does not locate the core in it, and the core hangs
off it as a cantilever whose tip deflection goes as length cubed. Both of the
things that fix that run the same line -- from inside the core, out past the
cast silhouette, into mold that is solid on both sides of the parting face --
and differ only in what is done along it.

A **tab** adds material: a post moulded onto the core, pinched flat when the
halves close. No extra part, but it is still there when the core is pulled out
of a cured glove, so the glove has to stretch off it.

A **dowel** takes material away: the same line bored through the core, half A
and half B alike, so a plain rod dropped in from outside the block locks all
three together. The pin is pulled before the mold is opened and the core then
leaves cleanly. Both leave the same hole in the glove; only the tab has to be
dragged out through it.

The bore is blind in the core -- it stops short of breaking out the far side,
which on anything slender would not weaken the core but saw it in half -- and
open to daylight at the other end, because a bore that stops inside the block
traps its own pin.

The invariant that makes the assembly exist
-------------------------------------------
**Every core-side feature is centred on the parting surface**, exactly like an
alignment key or the pour spout. That is not tidiness; it is what makes an
assembly sequence possible at all. A tab lying in the parting plane, a dowel
bore along the same plane: each leaves a half-round groove in either half,
widest exactly at its mouth, so the whole core assembly lifts straight out of
half B along ``+d`` and half A closes back down onto it. A screw is the
exception that proves it -- it runs down the pour axis into one half only, and
it is the one thing here that goes in *after* the mold is shut.

So the sequence is the ordinary moldmaker's one: core into half B, pins in,
half A down on top, screws through the plate. Coming apart is the reverse, and
the pins have to come out first.

What this does not solve
------------------------
A tab is still attached to the core when the core is pulled out of a cured
glove, so it drags through the slot it made. On a flexible cast that stretches;
on a stiff one it tears. The report measures the volume the glove has to
stretch over rather than assuming it is fine -- and a dowel is the way out of
it, at the price of a loose pin to keep track of.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import ConvexHull

from . import demold, meshio, mold as mold_mod, validate
from .config import MoldConfig
from .features import FeatureSkipped, frustum, place_local
from .frame import Frame, unit
from .mold import local_to_world
from .parting import PartingSurface

log = logging.getLogger(__name__)

# How far the cutting slabs overhang the block, in mm. Same reasoning as
# parting._SOLID_OVERHANG_MM: a cutting solid whose walls land exactly on the
# block's own walls hands the boolean coplanar faces and a fan of needle
# triangles.
_OVERHANG_MM = 20.0

# Boolean merge depth, mm. Two solids meeting exactly face to face are a
# coplanar boolean; overlapping them slightly is not.
_MERGE_MM = 1.5


# --------------------------------------------------------------------------
# the core body
# --------------------------------------------------------------------------


def erode(mesh: trimesh.Trimesh, delta: float, *, subdivisions: int = 1) -> trimesh.Trimesh:
    """Shrink ``mesh`` inward by ``delta`` mm: the exact inverse of an offset.

    This is the one genuinely expensive operation a core adds. It is a Minkowski
    *difference* against a ball, exact up to the ball's tessellation, which
    undershoots at facet centres and never overshoots.
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


def assert_core_solid(body: trimesh.Trimesh, name: str) -> trimesh.Trimesh:
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


def build_core_body(part: trimesh.Trimesh, cfg: MoldConfig) -> trimesh.Trimesh:
    """The core: the part, eroded by the wall.

    Nothing opens the cuff here. Without a carrier plate the core is a closed
    bladder and the cast has no rim; it is the plate's plane cut that opens it,
    through the core and both halves at once.
    """
    source = part
    if cfg.core.faces and len(part.faces) > cfg.core.faces:
        source = meshio.decimate(part, cfg.core.faces)
    body = erode(source, cfg.core.wall, subdivisions=cfg.core.ball_subdivisions)
    return assert_core_solid(body, "core body")


# --------------------------------------------------------------------------
# slabs, planes and the cut
# --------------------------------------------------------------------------


def _slab(frame: Frame, local_pts: np.ndarray, w_lo: float, w_hi: float) -> trimesh.Trimesh:
    """A box spanning ``[w_lo, w_hi]`` along the frame's +Z, wide enough for all of it."""
    lo = np.asarray(local_pts, dtype=np.float64).reshape(-1, 3).min(axis=0) - _OVERHANG_MM
    hi = np.asarray(local_pts, dtype=np.float64).reshape(-1, 3).max(axis=0) + _OVERHANG_MM
    lo[2], hi[2] = float(w_lo), float(w_hi)
    if hi[2] <= lo[2]:
        raise ValueError(f"empty slab: [{w_lo}, {w_hi}]")
    transform = local_to_world(frame).copy()
    transform[:3, 3] = frame.to_world((lo + hi) / 2.0)
    return trimesh.creation.box(extents=(hi - lo), transform=transform)


def default_plate_point(cavity: trimesh.Trimesh, pour_axis, cfg: MoldConfig) -> np.ndarray:
    """Where the cut plane sits unless it is moved: just inside the cuff end.

    Right at the cavity's extreme the cut face has no area, so it is set in by
    ``cut_inset``. That much of the scan is thrown away, which is the point --
    on a hand-and-forearm scan the plane goes through the forearm and the arm
    above it goes in the bin.
    """
    p = unit(pour_axis)
    v = np.asarray(cavity.vertices, dtype=np.float64)
    reach = v @ p
    top = float(reach.max())
    offset = top - float(cfg.carrier.cut_inset)
    # Centre it on the cavity's own cross-section, so the point reads as "on the
    # part" in the viewport rather than floating out at the block wall.
    near = v[reach >= offset - 1.0] if (reach >= offset - 1.0).any() else v
    centre = near.mean(axis=0)
    return centre + p * (offset - float(centre @ p))


def plane_offset(point, pour_axis) -> float:
    """The cut plane's distance along the pour axis, from a point on it."""
    return float(np.asarray(point, dtype=np.float64).reshape(3) @ unit(pour_axis))


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


def parting_nodes_world(surface: PartingSurface) -> np.ndarray:
    """Every grid node's point on the parting surface, in world coordinates."""
    gx, gy = np.meshgrid(surface.xs, surface.ys, indexing="ij")
    return surface.frame.to_world(
        np.column_stack([gx.ravel(), gy.ravel(), surface.h.ravel()])
    )


def nodes_below(surface: PartingSurface, pour_axis, offset: float) -> np.ndarray:
    """Grid mask: this node's world point is below ``offset`` along the pour axis."""
    reach = parting_nodes_world(surface) @ unit(pour_axis)
    return (reach <= offset).reshape(surface.shape)


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

    Measured rather than tested pass/fail, because the plane cuts *through* the
    part: at the cut face the cavity's wall is right there, and a fixed depth
    would be rejected at every position on the seam. A dowel that engages five
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


def _inside_footprint(plate: trimesh.Trimesh, pour_axis, inset: float):
    """Predicate: is this world point inside the plate's outline, inset by ``inset``?"""
    frame = Frame.from_direction(pour_axis)
    uv = frame.to_local(plate.vertices)[:, :2]
    hull = ConvexHull(uv)
    a, b = hull.equations[:, :2], hull.equations[:, 2]

    def predicate(world_point) -> bool:
        q = frame.to_local(np.asarray(world_point).reshape(1, 3))[0, :2]
        return bool(np.all(a @ q + b + inset <= 0.0))

    return predicate


def ring_inset(uv: np.ndarray, inset: float, count: int) -> np.ndarray:
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

    # Outward half-planes: n . x + d <= 0 holds inside. The tolerance is not
    # cosmetic: a point taken from the middle of an edge and pushed in by
    # `inset` lands at exactly `inset` from that edge, so an exact test is
    # decided by the sign of the float error and throws away most of the ring.
    # Corner points still fail it by millimetres.
    a, b = hull.equations[:, :2], hull.equations[:, 2]
    inward = -a[np.argmin(np.abs(on_ring @ a.T + b), axis=1)]
    moved = on_ring + inward * inset
    keep = np.all(moved @ a.T + b <= -inset + 1e-6, axis=1)
    return moved[keep]


# --------------------------------------------------------------------------
# what the core items build between them
# --------------------------------------------------------------------------


@dataclass
class CoreState:
    """The core assembly, part-built, as the plan is applied.

    The plate item is applied first and is what fills in ``plane`` and
    ``plate``; every later core item checks that and skips with a reason if the
    plate was switched off or would not fit. That ordering is the whole design:
    a dowel with no plate to stand on is not a dowel.
    """

    body: trimesh.Trimesh  # the eroded core, trimmed once the plate cuts it
    axis: np.ndarray  # the pour axis, and so the cut plane's normal
    pieces: list = field(default_factory=list)  # unioned into the assembly
    holes: list = field(default_factory=list)  # subtracted from it afterwards
    plane: float | None = None  # cut offset along the axis
    plate: trimesh.Trimesh | None = None
    plate_point: np.ndarray | None = None  # a world point on the plane
    discard: trimesh.Trimesh | None = None  # everything past it
    tabs: list = field(default_factory=list)  # kept to price the tear-off
    # The core as it stood before anything was bored into it. Measuring the
    # wall against the bored body reads the inside of a dowel hole as glove,
    # which it is not.
    wall_body: trimesh.Trimesh | None = None
    # Loose rods, one per dowel. Not part of the assembly and not one of the
    # three mold bodies: hardware, which you may well buy rather than print.
    pins: list = field(default_factory=list)
    _mask: np.ndarray | None = field(default=None, repr=False)
    _dist: tuple | None = field(default=None, repr=False)

    def core_mask(self, surface: PartingSurface) -> np.ndarray:
        """Where the parting surface runs inside the core. Cached: every tab asks."""
        if self._mask is None:
            self._mask = core_at_parting(self.body, surface)
        return self._mask

    def core_distance(self, surface: PartingSurface, sampling) -> tuple:
        """Distance away from the core on the parting grid, and the nearest node."""
        if self._dist is None:
            self._dist = ndimage.distance_transform_edt(
                ~self.core_mask(surface), sampling=sampling, return_indices=True
            )
        return self._dist

    def trimmed(self) -> None:
        """The plane moved the core, so anything derived from it is stale."""
        self._mask = None
        self._dist = None

    @property
    def has_plate(self) -> bool:
        return self.plate is not None

    def discarded_cm3(self, ctx) -> float:
        """How much block the cut throws away. Worth saying out loud."""
        if self.discard is None or ctx.block is None:
            return 0.0
        gone = trimesh.boolean.intersection(
            [ctx.block, self.discard], engine="manifold", check_volume=False
        )
        return (abs(float(gone.volume)) if len(gone.faces) else 0.0) / 1000.0

    @property
    def discarded_note(self) -> str:
        return (
            "through the core; everything past it is discarded"
            if self.plane is None
            else f"at {self.plane:.0f} mm along the pour axis; everything past it goes"
        )

    def pin_stock(self) -> trimesh.Trimesh | None:
        """Every dowel pin, in place, as one mesh -- or nothing if there are none."""
        if not self.pins:
            return None
        if len(self.pins) == 1:
            return self.pins[0]
        return trimesh.boolean.union(
            self.pins, engine="manifold", check_volume=False
        )

    def assembly(self) -> trimesh.Trimesh:
        """Fuse everything into the one body that gets printed."""
        parts = [self.body] + self.pieces
        out = (
            parts[0]
            if len(parts) == 1
            else trimesh.boolean.union(parts, engine="manifold", check_volume=False)
        )
        for hole in self.holes:
            out = cut(out, hole, "core hole")
        return assert_core_solid(out, "core assembly")


def cut(solid: trimesh.Trimesh, tool: trimesh.Trimesh, label: str) -> trimesh.Trimesh:
    out = trimesh.boolean.difference(
        [solid, tool], engine="manifold", check_volume=False
    )
    log.debug("%s -> %d faces", label, len(out.faces))
    return out


# --------------------------------------------------------------------------
# the plate, and the cut that makes it
# --------------------------------------------------------------------------


def stage_plate(ctx, point, thickness: float, state: CoreState) -> None:
    """Work out the cut and the plate, and trim the core, without touching the halves.

    Planning needs the geometry the cut will leave -- a dowel proposed where the
    block is about to be discarded is a dowel nobody can use -- so this is
    separated from applying it and runs during the planning pass too.

    The core is trimmed a merge depth *above* the plane rather than at it. The
    stub that leaves sticking up ends up inside the plate, which turns the union
    of core and plate from a coplanar boolean into an overlapping one -- and the
    space it occupies is space the halves are about to vacate, so nothing can
    foul on it.
    """
    if ctx.block is None:
        raise FeatureSkipped(
            "the plate is cut from the block, which this mold did not keep"
        )
    p = state.axis
    offset = plane_offset(point, p)
    frame = Frame.from_direction(p)
    span = frame.to_local(ctx.block.vertices)
    lo, hi = float(span[:, 2].min()), float(span[:, 2].max())
    if not (lo + 1.0 < offset < hi - 0.5):
        raise FeatureSkipped(
            f"the cut plane at {offset:.1f} mm is outside the block "
            f"({lo:.1f} to {hi:.1f} mm along the pour axis)"
        )

    plate = trimesh.boolean.intersection(
        [ctx.block, _slab(frame, span, offset, offset + float(thickness))],
        engine="manifold",
        check_volume=False,
    )
    if len(plate.faces) == 0 or plate.volume <= 0:
        raise FeatureSkipped("the block has no material at the cut plane")

    body = cut(
        state.body, _slab(frame, span, offset + _MERGE_MM, hi + _OVERHANG_MM), "plane -> core"
    )
    if len(body.faces) == 0:
        raise FeatureSkipped("the cut plane leaves no core behind it")

    state.body = body
    state.wall_body = body
    state.plane = offset
    state.plate = plate
    state.plate_point = np.asarray(point, dtype=np.float64).reshape(3)
    state.discard = _slab(frame, span, offset, hi + _OVERHANG_MM)
    state.trimmed()


def apply_plate(bodies, ctx, item, state: CoreState, cfg: MoldConfig) -> dict:
    """Cut the mold in two along the plane and cap it with the plate.

    Everything past the plane goes: half A, half B and the core alike. What is
    left is three coplanar faces, and the plate is a slab of the block's own
    section laid across all of them, so it caps the halves, seals the annulus
    and swallows the core's stub in one body.
    """
    if state.has_plate:
        raise FeatureSkipped(
            f"the mold is already cut at {state.plane:.0f} mm; a second plane "
            "would cut away the plate that is already there"
        )
    thickness = float(item.params["thickness"])
    stage_plate(ctx, item.position, thickness, state)

    bodies.half_a = cut(bodies.half_a, state.discard, "plane -> A")
    bodies.half_b = cut(bodies.half_b, state.discard, "plane -> B")
    state.pieces.append(state.plate)

    measured = np.asarray(state.plate.vertices) @ state.axis
    return {
        "id": item.id,
        "plane_offset_mm": round(float(state.plane), 2),
        "thickness_mm": round(float(measured.max() - measured.min()), 2),
        "asked_thickness_mm": round(thickness, 2),
        "volume_cm3": round(float(state.plate.volume) / 1000.0, 2),
        "discarded_cm3": round(float(state.discarded_cm3(ctx)), 2),
    }


# --------------------------------------------------------------------------
# dowels and screws: registering the plate to the assembled block
# --------------------------------------------------------------------------


def _seam_run(surface, position, ctx, state: CoreState, section):
    """The line a seam feature runs along: from inside the core out to an anchor.

    Only the anchor is in the plan. Which way the run goes is a question about
    where the core is, so it is answered here rather than stored -- which is
    what makes "move this" mean "grip the mold here" rather than "point that
    way". Shared by tabs and dowels, which are the same line: one adds material
    along it, the other takes it away.

    Returns ``(inner_local, outer_local, axis_local, span, anchor_clearance)``.
    """
    outer_local, (i, j) = ctx.on_parting(position)
    if not state.core_mask(surface).any():
        raise FeatureSkipped("the parting surface never passes through the core")

    dx = float(surface.xs[1] - surface.xs[0]) if surface.shape[0] > 1 else 1.0
    dy = float(surface.ys[1] - surface.ys[0]) if surface.shape[1] > 1 else 1.0
    dist, idx = state.core_distance(surface, (dx, dy))
    length = float(dist[i, j])
    if length < section.min_length:
        raise FeatureSkipped(
            "this anchor is already inside the core; the run has to reach out "
            "past the cast to mold that is solid on both sides of the seam"
        )
    if length > section.max_length:
        raise FeatureSkipped(
            f"the nearest core is {length:.0f} mm away "
            f"(limit {section.max_length:.0f} mm)"
        )
    room = float(ctx.free_distance()[i, j])
    if room < section.radius:
        raise FeatureSkipped(
            f"only {room:.1f} mm of mold around this anchor, needs {section.radius:.1f} mm"
        )

    ii, jj = int(idx[0][i, j]), int(idx[1][i, j])
    inner_local = np.array([surface.xs[ii], surface.ys[jj], surface.h[ii, jj]])
    axis_local = outer_local - inner_local
    span = float(np.linalg.norm(axis_local))
    if span < 1e-6:
        raise FeatureSkipped("the run has nowhere to go")
    return inner_local, outer_local, axis_local / span, span, room


def find_seam_sites(ctx, state: CoreState, section, *, avoid=()) -> list[tuple]:
    """Seam runs, as ``(anchor, grip)`` pairs in world coordinates.

    The placement is the alignment-key logic run in reverse. A key wants a
    column that *misses* the part; a seam feature wants both -- an outer end in
    one of those, with mold on either side of the parting face, and an inner end
    where the parting surface runs inside the core. A distance transform away
    from the core gives every free node both its nearest core node and the run
    between them in one pass.
    """
    surface = ctx.surface
    nx, ny = surface.shape
    dx = float(surface.xs[1] - surface.xs[0]) if nx > 1 else 1.0
    dy = float(surface.ys[1] - surface.ys[0]) if ny > 1 else 1.0

    if not state.core_mask(surface).any():
        log.warning("the parting surface never passes through the core")
        return []
    dist, idx = state.core_distance(surface, (dx, dy))

    need = section.radius + section.anchor_margin
    free_dist = ctx.free_distance()
    ok = (
        ~surface.constrained
        & (free_dist >= need)
        & (dist >= section.min_length)
        & (dist <= section.max_length)
    )
    if state.plane is not None:
        # Past the cut there is nothing left to grip.
        ok &= nodes_below(surface, state.axis, state.plane - section.radius)
    border_x, border_y = int(np.ceil(need / dx)), int(np.ceil(need / dy))
    edge = np.zeros_like(ok)
    edge[border_x : nx - border_x, border_y : ny - border_y] = True
    ok &= edge

    cand = np.argwhere(ok)
    if len(cand) == 0:
        log.warning("no room on the parting face for a seam feature")
        return []

    # Spread by where each one *grips the core*, not by where it is anchored:
    # two anchored far apart can still grab the same spot on the core, and
    # bracing a cantilever needs them spread along its length.
    inner_ij = np.column_stack([idx[0][ok], idx[1][ok]])
    inner_pts = np.column_stack(
        [surface.xs[inner_ij[:, 0]], surface.ys[inner_ij[:, 1]]]
    )
    room = free_dist[ok]

    # Keep clear of runs already placed, compared *grip to grip*. That is where
    # they crowd: two runs can be anchored far apart and still take hold of the
    # same few millimetres of core, and a dowel bored through the root of a tab
    # does not weaken it, it cuts it off.
    taken = [surface.frame.to_local(np.asarray(q).reshape(3))[:2] for q in avoid]
    if taken:
        gap = np.min(
            np.linalg.norm(inner_pts[:, None, :] - np.array(taken)[None, :, :], axis=2),
            axis=1,
        )
        room = np.where(gap >= section.min_spacing, room, -np.inf)
        if not np.isfinite(room).any():
            return []

    chosen = [int(np.argmax(room))]
    while len(chosen) < section.count:
        gap = np.min(
            np.linalg.norm(inner_pts[:, None, :] - inner_pts[chosen][None, :, :], axis=2),
            axis=1,
        )
        score = np.where(gap >= section.min_spacing, room + 0.25 * gap, -np.inf)
        nxt = int(np.argmax(score))
        if not np.isfinite(score[nxt]):
            break
        chosen.append(nxt)

    return [
        (
            _node_world(surface, int(cand[c][0]), int(cand[c][1])),
            _node_world(surface, int(inner_ij[c][0]), int(inner_ij[c][1])),
        )
        for c in chosen
    ]


def apply_dowel(bodies, ctx, item, state: CoreState, cfg: MoldConfig) -> dict:
    """Bore one hole through half A, half B *and* the core, and make its pin.

    This is a tab turned inside out. A tab is a post moulded onto the core and
    pinched by the closing halves; a dowel is the same line through the same
    three bodies, cut away instead of added, so that a plain rod dropped down it
    from outside the block registers the core against both halves at once.

    Which is what the extra part buys. Nothing is left attached to the core, so
    the pin is pulled first and the core then leaves the cured glove cleanly,
    rather than dragging a tab through the slot it made. The glove ends up with
    the hole either way.

    The bore is centred on the parting surface for the usual reason: each half
    gets a groove widest exactly at its mouth, so the halves still open and the
    core still lifts out -- once the pin is out of the way, which it has to be.
    """
    dcfg = cfg.core_dowels
    surface = ctx.surface
    r = float(item.params["radius"])
    engagement = float(item.params["engagement"])
    clearance = float(item.params["clearance"])

    inner_local, outer_local, axis_local, _span, room = _seam_run(
        surface, item.position, ctx, state, dcfg
    )
    axis_world = surface.frame.to_world(axis_local)
    inner_world = surface.frame.to_world(inner_local)

    # A blind hole, not a through one. The run enters the core at its outer
    # surface and keeps going; left unchecked it comes out the far side, and on
    # anything slender -- a finger, the web between two -- that does not weaken
    # the core, it saws it in half.
    depth_available = _exit_distance(state.body, inner_world, -axis_world)
    if depth_available is None:
        raise FeatureSkipped("the run does not enter the core at all")
    engagement = min(engagement, depth_available - max(r, 1.0))
    if engagement < max(r, 1.0):
        raise FeatureSkipped(
            f"only {max(depth_available, 0.0):.1f} mm of core along this run; a "
            "bore that deep would break out the other side"
        )
    start_local = inner_local - axis_local * engagement
    start_world = surface.frame.to_world(start_local)

    # Run it out to daylight. A bore that stops inside the block traps its own
    # pin, and a pin that cannot be pulled is a tab with extra steps.
    reach = _exit_distance(ctx.block, start_world, axis_world)
    if reach is None:
        raise FeatureSkipped(
            "this run never reaches the outside of the block, so the pin could "
            "not be pulled back out before the mold is opened"
        )
    drift = seam_drift(surface, start_world, axis_world, reach)
    if drift > dcfg.max_seam_drift:
        raise FeatureSkipped(
            f"the parting surface bends {drift:.2f} mm over the bore's {reach:.0f} mm "
            f"run (limit {dcfg.max_seam_drift} mm); its groove would be an undercut"
        )

    bore = place_local(
        frustum(r + clearance, r + clearance, reach + _MERGE_MM, sections=32),
        surface.frame,
        start_local - axis_local * clearance,
        axis_local,
    )
    pin = place_local(
        frustum(r, r, reach, sections=32), surface.frame, start_local, axis_local
    )

    if state.wall_body is None:
        state.wall_body = state.body
    bodies.half_a = cut(bodies.half_a, bore, "dowel -> A")
    bodies.half_b = cut(bodies.half_b, bore, "dowel -> B")
    state.body = cut(state.body, bore, "dowel -> core")
    # ...and out of the finished assembly too, so a pin whose line happens to
    # pass through a tab still has a channel rather than a interference fit.
    state.holes.append(bore)
    state.trimmed()
    state.pins.append(pin)

    return {
        "id": item.id,
        "anchor_world": [round(float(v), 2) for v in surface.frame.to_world(outer_local)],
        "core_world": [round(float(v), 2) for v in surface.frame.to_world(inner_local)],
        "exit_world": [round(float(v), 2) for v in (start_world + axis_world * reach)],
        "length_mm": round(reach, 2),
        "engagement_mm": round(engagement, 2),
        "pin_diameter_mm": round(2.0 * r, 2),
        "anchor_clearance_mm": round(room, 2),
        "seam_drift_mm": round(drift, 3),
    }


def _exit_distance(block, start, axis) -> float | None:
    """How far along ``axis`` the block's outer surface is, from ``start``."""
    if block is None:
        return None
    hits = block.ray.intersects_location(
        np.asarray(start, dtype=np.float64).reshape(1, 3),
        np.asarray(axis, dtype=np.float64).reshape(1, 3),
        multiple_hits=True,
    )[0]
    if len(hits) == 0:
        return None
    reach = float(((hits - np.asarray(start)) @ unit(axis)).max())
    return reach if reach > 1.0 else None


def _on_plane(position, pour_axis, plane: float) -> np.ndarray:
    """Slide a world point onto the cut plane, so an edited item stays on it."""
    p = unit(pour_axis)
    q = np.asarray(position, dtype=np.float64).reshape(3)
    return q + p * (plane - float(q @ p))


def find_screw_sites(ctx, state: CoreState, cfg: MoldConfig, avoid=()) -> list[np.ndarray]:
    """Clamping screws around the plate's edge, alternating between the halves.

    Unlike everything else here a screw goes in last, after the mold is closed,
    so it is under no release constraint at all -- it only has to miss the cavity
    and land wholly inside one half rather than straddling the seam, where it
    would jack the halves apart instead of clamping them down.
    """
    ccfg = cfg.carrier
    if ccfg.screw_count <= 0:
        return []
    p = state.axis
    frame = Frame.from_direction(p)
    clear = ccfg.screw_radius + ccfg.screw_clearance

    probes = ring_inset(
        frame.to_local(state.plate.vertices)[:, :2],
        clear + 3.0,
        max(8 * ccfg.screw_count, 32),
    )
    if len(probes) == 0:
        log.warning("the carrier plate is too narrow to take screws")
        return []

    candidates: dict[str, list[np.ndarray]] = {"a": [], "b": []}
    for u, v in probes:
        start = frame.to_world(np.array([u, v, state.plane]))
        if any(np.linalg.norm(start - q) < clear + ccfg.port_outer_radius + 4.0 for q in avoid):
            continue
        side = _screw_side(ctx.surface, start, p, ccfg.screw_depth, clear)
        if side is None:
            continue
        room = free_depth(ctx.cavity, start, -p, ccfg.screw_depth + _MERGE_MM, clear + 1.0)
        if min(ccfg.screw_depth, room - _MERGE_MM) < ccfg.screw_min_depth:
            continue
        candidates[side].append(start)

    # Alternate sides. Taking valid positions in the order they come off the
    # ring walks one side of the seam before reaching the other, so the whole
    # screw budget lands in one half and the other is held down by nothing.
    spacing = 2.5 * clear + 6.0
    picked: list[np.ndarray] = []
    sides: list[str] = []
    while len(picked) < ccfg.screw_count:
        order = sorted(("a", "b"), key=lambda s: sides.count(s))
        for side in order:
            pool = [
                q
                for q in candidates[side]
                if all(np.linalg.norm(q - c) >= spacing for c in picked)
            ]
            if not pool:
                continue
            picked.append(
                max(
                    pool,
                    key=lambda q: min(
                        (float(np.linalg.norm(q - c)) for c in picked), default=0.0
                    ),
                )
            )
            sides.append(side)
            break
        else:
            break

    if len(picked) < ccfg.screw_count:
        log.warning(
            "only %d of %d screw positions clear the cavity and stay wholly "
            "inside one half",
            len(picked),
            ccfg.screw_count,
        )
    return picked


def _screw_side(surface, start, p, depth: float, clear: float) -> str | None:
    """Which half this bore lies wholly inside, or None if it straddles the seam."""
    ts = np.linspace(0.0, float(depth), 9)
    local = surface.frame.to_local(start[None, :] - ts[:, None] * unit(p)[None, :])
    gap = local[:, 2] - _h_at(surface, local)
    if np.abs(gap).min() < clear + 1.0:
        return None
    if not (np.all(gap > 0.0) or np.all(gap < 0.0)):
        return None
    return "a" if gap[0] > 0 else "b"


def apply_screw(bodies, ctx, item, state: CoreState, cfg: MoldConfig) -> dict:
    """A pilot bore into one half and a clearance hole through the plate."""
    p = state.axis
    frame = Frame.from_direction(p)
    up = frame.to_local(p)
    r = float(item.params["radius"])
    clear = r + float(item.params["clearance"])

    start = _on_plane(item.position, p, state.plane)
    side = _screw_side(ctx.surface, start, p, float(item.params["depth"]), clear)
    if side is None:
        raise FeatureSkipped(
            "this screw straddles the parting seam; it would jack the halves "
            "apart rather than clamp them down"
        )
    room = free_depth(ctx.cavity, start, -p, float(item.params["depth"]) + _MERGE_MM, clear + 1.0)
    depth = min(float(item.params["depth"]), room - _MERGE_MM)
    if depth < 3.0:
        raise FeatureSkipped(
            f"only {max(depth, 0.0):.1f} mm of mold under this point before the cavity"
        )

    pilot = place_local(
        frustum(r, r, depth + _MERGE_MM, sections=24),
        frame,
        frame.to_local(start - p * (depth + _MERGE_MM)),
        up,
    )
    target = "half_a" if side == "a" else "half_b"
    setattr(bodies, target, cut(getattr(bodies, target), pilot, f"screw -> {side}"))

    plate_top = float((np.asarray(state.plate.vertices) @ p).max())
    state.holes.append(
        place_local(
            frustum(clear, clear, plate_top - state.plane + 2.0 * _MERGE_MM, sections=24),
            frame,
            frame.to_local(start - p * _MERGE_MM),
            up,
        )
    )
    return {
        "id": item.id,
        "start_world": [round(float(v), 2) for v in start],
        "depth_mm": round(depth, 2),
        "side": side,
    }


# --------------------------------------------------------------------------
# the pour port through the plate
# --------------------------------------------------------------------------


def find_port_site(ctx, state: CoreState) -> np.ndarray | None:
    """A spot on the cut face where the plate covers cast rather than core.

    Sealing the annulus is what forces this: with the plate on, the only way in
    is through it, and the only place worth going through is the ring of cast
    between the core and the cavity wall. It is only a wall thick, so the funnel
    necks down to meet it -- the opening is a slot the width of the glove.
    """
    surface, p = ctx.surface, state.axis
    nx = surface.shape[0]
    cell = float(surface.xs[1] - surface.xs[0]) if nx > 1 else 1.0

    nodes = parting_nodes_world(surface)
    on_plane = np.abs((nodes @ p) - state.plane) <= max(cell, 1.0)
    pts = nodes[on_plane]
    if len(pts) == 0:
        return None
    # Slightly below the plane, so "inside the cavity" is a question about the
    # cast rather than about the cut face itself.
    probe = pts - p * max(cell, 1.0)
    try:
        in_cast = ctx.cavity.contains(probe) & ~state.body.contains(probe)
    except Exception:  # pragma: no cover - contains needs a closed mesh
        return None
    if not in_cast.any():
        return None
    # Of the annulus points, the one furthest from the core's surface: that is
    # the widest the ring gets, and so the least bad place to neck down into.
    room = trimesh.proximity.closest_point(state.body, pts[in_cast])[1]
    return pts[in_cast][int(np.argmax(room))]


def apply_port(bodies, ctx, item, state: CoreState, cfg: MoldConfig) -> dict:
    """Cut the funnel through the plate."""
    p = state.axis
    frame = Frame.from_direction(p)
    inner = float(item.params["inner_radius"])
    outer = float(item.params["outer_radius"])

    start = _on_plane(item.position, p, state.plane)
    plate_top = float((np.asarray(state.plate.vertices) @ p).max())
    length = plate_top - state.plane + 2.0 * _MERGE_MM
    if length <= 1.0:
        raise FeatureSkipped("the plate is too thin to take a port")

    funnel = place_local(
        frustum(inner, outer, length, sections=48),
        frame,
        frame.to_local(start - p * _MERGE_MM),
        frame.to_local(p),
    )
    state.holes.append(funnel)
    return {
        "id": item.id,
        "entry_world": [round(float(v), 2) for v in start],
        "length_mm": round(length, 2),
        "inner_radius_mm": round(inner, 2),
        "outer_radius_mm": round(outer, 2),
    }


# --------------------------------------------------------------------------
# seam tabs
# --------------------------------------------------------------------------


def find_tab_sites(ctx, state: CoreState, cfg: MoldConfig, *, avoid=()) -> list[tuple]:
    return find_seam_sites(ctx, state, cfg.core_tabs, avoid=avoid)


def find_dowel_sites(ctx, state: CoreState, cfg: MoldConfig, *, avoid=()) -> list[tuple]:
    return find_seam_sites(ctx, state, cfg.core_dowels, avoid=avoid)


def apply_tab(bodies, ctx, item, state: CoreState, cfg: MoldConfig) -> dict:
    """A post from the core out to an anchor, pinched flat on the parting face.

    The male half of what a dowel does: same line through the same three bodies,
    but material added along it rather than taken away, so no extra part is
    needed -- at the cost of leaving something attached to the core for the
    glove to stretch over on its way off.
    """
    tcfg = cfg.core_tabs
    surface = ctx.surface
    r = float(item.params["radius"])
    clearance = float(item.params["clearance"])

    inner_local, outer_local, axis_local, span, room = _seam_run(
        surface, item.position, ctx, state, tcfg
    )
    drift = seam_drift(
        surface,
        surface.frame.to_world(inner_local),
        surface.frame.to_world(axis_local),
        span,
    )
    if drift > tcfg.max_seam_drift:
        raise FeatureSkipped(
            f"the parting surface bends {drift:.2f} mm over the tab's {span:.0f} mm "
            f"run (limit {tcfg.max_seam_drift} mm); its groove would be an undercut"
        )

    back = _MERGE_MM + r
    start = inner_local - axis_local * back
    tab = place_local(frustum(r, r, span + back, sections=32), surface.frame, start, axis_local)
    pocket = place_local(
        frustum(r + clearance, r + clearance, span + back + clearance, sections=32),
        surface.frame,
        start,
        axis_local,
    )
    state.pieces.append(tab)
    state.tabs.append(tab)
    bodies.half_a = cut(bodies.half_a, pocket, "tab -> A")
    bodies.half_b = cut(bodies.half_b, pocket, "tab -> B")
    return {
        "id": item.id,
        "anchor_world": [round(float(v), 2) for v in surface.frame.to_world(outer_local)],
        # Where it actually grips the core. Only the anchor is in the plan --
        # this end is worked out against the core here -- so handing it back is
        # the only way a viewer can draw the tab rather than guess at it.
        "core_world": [round(float(v), 2) for v in surface.frame.to_world(inner_local)],
        "length_mm": round(span, 2),
        "anchor_clearance_mm": round(room, 2),
        "seam_drift_mm": round(drift, 3),
    }


# --------------------------------------------------------------------------
# what the run measured
# --------------------------------------------------------------------------


def measure_wall(
    core_body: trimesh.Trimesh,
    cavity: trimesh.Trimesh,
    target_mm: float,
    *,
    pour_axis=None,
    cut: float | None = None,
    samples: int = 4000,
) -> dict:
    """Measured wall thickness, not assumed.

    Distance from a point on the core's surface to the cavity's surface *is* the
    wall there, so this checks the erosion delivered what was asked and prices
    the eroding ball's tessellation: a polyhedral ball undershoots at its facet
    centres and never overshoots, costing about 6.5% of the wall at
    ``ball_subdivisions=1``, 1.8% at 2 and 0.4% at 3.

    ``cut`` excludes the cuff, and has to sit a wall *below* the plane rather
    than at it: the core's cut face runs out to the cavity at its rim, so a
    sample taken there reads as near-zero wall. That is not a thin spot in the
    glove, it is the hole the hand goes through.
    """
    pts = core_body.sample(int(samples))
    if cut is not None and pour_axis is not None:
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


def tab_protrusion(
    tabs: list[trimesh.Trimesh], cavity: trimesh.Trimesh, core_body: trimesh.Trimesh
) -> float:
    """How much tab the glove has to stretch over on its way off the core.

    The part of each tab that sits in the cast -- outside the core, inside the
    cavity -- passes through the glove wall. Pulling the core out along the cuff
    axis drags it through the slot it made, so this is the number that decides
    whether a given cast material tolerates seam tabs or tears on them.
    """
    if not tabs:
        return 0.0
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
    measuring. A number here means a tab or a dowel drifted off the seam far
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


def real_pieces(mesh: trimesh.Trimesh, *, min_volume: float = 1.0) -> int:
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
