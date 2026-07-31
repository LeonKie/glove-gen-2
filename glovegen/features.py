"""Mold usability features: alignment keys, a pour spout and vents.

Without these the output is a shape split in half, not a mold: the halves have
nothing to locate each other by, no way to introduce casting material once
they are closed, and no escape path for trapped air.

Two geometric facts drive the design.

**Halves always separate.** Half A is a subset of ``{z > h(x, y)}`` and half B of
``{z < h(x, y)}`` in the pull frame. Translating A by ``+t z`` keeps it inside
``{z > h + t}``, so it can never intersect B. Any feature that respects the
parting surface therefore cannot break the half-to-half split -- which is why
channels can be cut from both halves freely.

**A groove wider inside than at its mouth traps the cast.** A channel of radius
``r`` centred at height ``y0`` and cut by the parting surface at ``h`` leaves half
A a circular segment whose widest point is at ``y0``. If ``h < y0`` that widest
point is *above* the mouth, and the cast inside it is locked in. So channels are
either centred on the parting surface (giving each half a clean half-round
groove, widest exactly at the mouth) or routed along the pull direction, where
they release by construction. Never in between.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy import ndimage

from .config import KeyConfig, MoldConfig, SpoutConfig, VentConfig
from .frame import Frame, principal_axis, unit
from .mold import MoldResult, local_to_world
from .parting import PartingSurface

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def revolved(profile, *, sections: int = 48) -> trimesh.Trimesh:
    """Solid of revolution about the Z axis from a ``[(radius, z), ...]`` profile.

    Radii must be positive and ``z`` strictly increasing; the ends are capped
    flat. trimesh has ``cylinder`` and ``cone`` but nothing that covers a
    drafted key, a funnel and a withdrawal sweep in one primitive.
    """
    prof = np.asarray(profile, dtype=np.float64).reshape(-1, 2)
    if len(prof) < 2:
        raise ValueError("profile needs at least two (radius, z) points")
    radii = np.maximum(prof[:, 0], 1e-4)
    zs = prof[:, 1]
    if np.any(np.diff(zs) <= 0):
        raise ValueError(f"profile z must strictly increase, got {zs}")

    theta = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    ring = np.column_stack([np.cos(theta), np.sin(theta)])

    rings = [np.column_stack([ring * r, np.full(sections, z)]) for r, z in zip(radii, zs)]
    n_rings = len(rings)
    verts = np.vstack(rings + [[[0.0, 0.0, zs[0]]], [[0.0, 0.0, zs[-1]]]])
    i_bottom = n_rings * sections
    i_top = i_bottom + 1

    j = np.arange(sections)
    k = (j + 1) % sections
    faces = []
    for r in range(n_rings - 1):
        lo = r * sections
        hi = (r + 1) * sections
        faces.append(np.column_stack([lo + j, lo + k, hi + k]))
        faces.append(np.column_stack([lo + j, hi + k, hi + j]))
    faces.append(np.column_stack([np.full(sections, i_bottom), k, j]))
    last = (n_rings - 1) * sections
    faces.append(np.column_stack([np.full(sections, i_top), last + j, last + k]))

    mesh = trimesh.Trimesh(vertices=verts, faces=np.vstack(faces), process=False)
    mesh.merge_vertices()
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def frustum(
    r_bottom: float, r_top: float, height: float, *, sections: int = 48
) -> trimesh.Trimesh:
    """A truncated cone along +Z, base at z=0, wide end at z=height."""
    return revolved([(r_bottom, 0.0), (r_top, float(height))], sections=sections)


def _place_local(mesh: trimesh.Trimesh, frame: Frame, origin_local, axis_local=None):
    """Move a +Z-aligned primitive to ``origin_local`` and convert to world."""
    out = mesh.copy()
    if axis_local is not None:
        rot = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], unit(axis_local))
        out.apply_transform(rot)
    out.apply_translation(np.asarray(origin_local, dtype=np.float64))
    out.apply_transform(local_to_world(frame))
    return out


def _boolean(op, meshes, label):
    out = op(meshes, engine="manifold", check_volume=False)
    log.debug("%s -> %d faces", label, len(out.faces))
    return out


def _union(a, b, label="union"):
    return _boolean(trimesh.boolean.union, [a, b], label)


def _difference(a, b, label="difference"):
    return _boolean(trimesh.boolean.difference, [a, b], label)


# --------------------------------------------------------------------------
# alignment keys
# --------------------------------------------------------------------------


@dataclass
class KeySite:
    x: float
    y: float
    z: float  # parting-surface height at (x, y), in the pull frame
    clearance_to_cavity: float

    def as_dict(self) -> dict:
        return {
            "local_xy": [round(self.x, 2), round(self.y, 2)],
            "parting_z": round(self.z, 2),
            "cavity_clearance_mm": round(self.clearance_to_cavity, 2),
        }


def find_key_sites(
    surface: PartingSurface,
    local_bounds: np.ndarray,
    cfg: KeyConfig,
) -> list[KeySite]:
    """Pick well-spread spots on the parting face that clear the cavity.

    Keys must sit where the parting face is solid mold on both sides, i.e. in
    columns that miss the part entirely, with room for the key body above and
    the socket below.
    """
    nx, ny = surface.shape
    dx = float(surface.xs[1] - surface.xs[0]) if nx > 1 else 1.0
    dy = float(surface.ys[1] - surface.ys[0]) if ny > 1 else 1.0

    free = ~surface.constrained
    # Distance from each free cell to the nearest column that hits the part.
    dist = ndimage.distance_transform_edt(free, sampling=(dx, dy))

    need = cfg.radius + cfg.cavity_margin
    ok = free & (dist >= need)

    # Stay clear of the block's outside walls too.
    border_x = int(np.ceil((cfg.radius + 2.0) / dx))
    border_y = int(np.ceil((cfg.radius + 2.0) / dy))
    edge = np.zeros_like(ok)
    edge[border_x : nx - border_x, border_y : ny - border_y] = True
    ok &= edge

    # Vertical room: key body hangs below the surface, socket above it.
    room_below = surface.h - local_bounds[0, 2]
    room_above = local_bounds[1, 2] - surface.h
    ok &= room_below >= (cfg.height + 2.0)
    ok &= room_above >= 2.0

    idx = np.argwhere(ok)
    if len(idx) == 0:
        log.warning("no room on the parting face for alignment keys")
        return []

    pts = np.column_stack([surface.xs[idx[:, 0]], surface.ys[idx[:, 1]]])
    scores = dist[ok]

    # Farthest-point sampling: start at the most generous spot, then keep
    # picking the candidate furthest from everything already chosen, so the
    # keys end up spread around the part instead of clustered.
    chosen: list[int] = [int(np.argmax(scores))]
    min_gap = np.full(len(pts), np.inf)
    while len(chosen) < cfg.count:
        d = np.linalg.norm(pts - pts[chosen[-1]], axis=1)
        min_gap = np.minimum(min_gap, d)
        nxt = int(np.argmax(min_gap))
        if min_gap[nxt] < 2.5 * cfg.radius:
            break
        chosen.append(nxt)

    sites = []
    for c in chosen:
        i, j = idx[c]
        sites.append(
            KeySite(
                x=float(surface.xs[i]),
                y=float(surface.ys[j]),
                z=float(surface.h[i, j]),
                clearance_to_cavity=float(dist[i, j]),
            )
        )
    return sites


def add_keys(
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    surface: PartingSurface,
    local_bounds: np.ndarray,
    cfg: KeyConfig,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Add male keys to half A and matching sockets to half B.

    The key hangs *down* from A's parting face and widens upward at the draft
    angle, so it withdraws from its socket along +d as the halves part.
    """
    sites = find_key_sites(surface, local_bounds, cfg)
    if not sites:
        return half_a, half_b, {"count": 0, "sites": []}

    draft = np.tan(np.radians(cfg.draft_deg))
    overlap = 3.0  # merge depth into A's body, avoids a coplanar boolean

    r_mouth = cfg.radius
    r_bottom = max(r_mouth - cfg.height * draft, 0.35 * r_mouth)
    c = cfg.clearance
    z_sweep = float(local_bounds[1, 2]) + 1.0

    for site in sites:
        # Key: drafted below the parting face so it self-locates, then a plain
        # cylinder above it that just merges into A's body.
        key = revolved(
            [
                (r_bottom, site.z - cfg.height),
                (r_mouth, site.z),
                (r_mouth, site.z + overlap),
            ]
        )
        key = _place_local(key, surface.frame, (site.x, site.y, 0.0))
        half_a = _union(half_a, key, "key -> A")

        # Socket: the key's *swept* volume as it withdraws along +d, not just
        # its resting shape. The parting surface is curved, so half B can hold
        # material above the key's mouth within the key's own footprint; without
        # sweeping the socket to the top of the block, the key would jam on it.
        socket = revolved(
            [
                (r_bottom + c, site.z - cfg.height - c),
                (r_mouth + c, site.z),
                (r_mouth + c, z_sweep),
            ]
        )
        socket = _place_local(socket, surface.frame, (site.x, site.y, 0.0))
        half_b = _difference(half_b, socket, "socket -> B")

    info = {
        "count": len(sites),
        "radius_mm": cfg.radius,
        "height_mm": cfg.height,
        "draft_deg": cfg.draft_deg,
        "clearance_mm": cfg.clearance,
        "sites": [s.as_dict() for s in sites],
    }
    log.info("added %d alignment keys", len(sites))
    return half_a, half_b, info


# --------------------------------------------------------------------------
# pour axis
# --------------------------------------------------------------------------


def choose_pour_axis(mesh: trimesh.Trimesh, cfg: MoldConfig) -> np.ndarray:
    """Which way is up when the assembled mold gets filled.

    Defaults to the part's longest principal axis, oriented so the *fatter* end
    is up. For a hand-and-forearm scan that puts the cut wrist end at the top:
    the mold then fills downward into the fingers with air rising back out
    through the spout, which is the orientation that traps the least air.
    """
    if isinstance(cfg.pour_axis, str) and cfg.pour_axis != "auto":
        raise ValueError(f"pour_axis must be 'auto' or a vector, got {cfg.pour_axis!r}")
    if not isinstance(cfg.pour_axis, str):
        return unit(cfg.pour_axis)

    axis = principal_axis(mesh)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if len(v) > 200_000:
        v = v[:: len(v) // 200_000 + 1]
    proj = v @ axis
    lo_cut, hi_cut = np.percentile(proj, [5.0, 95.0])

    def girth(sel) -> float:
        pts = v[sel]
        if len(pts) < 8:
            return 0.0
        radial = pts - np.outer(pts @ axis, axis)
        return float(np.linalg.norm(radial - radial.mean(axis=0), axis=1).mean())

    if girth(proj <= lo_cut) > girth(proj >= hi_cut):
        axis = -axis
    return unit(axis)


# --------------------------------------------------------------------------
# pour spout
# --------------------------------------------------------------------------


def add_spout(
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    cavity: trimesh.Trimesh,
    surface: PartingSurface,
    local_bounds: np.ndarray,
    pour_axis: np.ndarray,
    cfg: SpoutConfig,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Cut a funnel from the outside of the block into the top of the cavity.

    Placed at the cavity's extreme along the pour axis, which is the highest
    point of the cavity once the mold is stood up to pour, and centred on the
    parting surface so each half gets a clean half-round groove.
    """
    pour_axis = unit(pour_axis)
    v = np.asarray(cavity.vertices, dtype=np.float64)
    proj = v @ pour_axis
    top = proj >= np.percentile(proj, 99.5)
    entry_world = v[top].mean(axis=0)

    # Centre the channel on the parting surface: sample h at the entry's
    # in-plane position. Over the short run through the block wall the surface
    # barely moves, so this keeps the groove half-round to within a fraction of
    # a millimetre.
    entry_local = surface.frame.to_local(entry_world)
    i = int(np.clip(np.searchsorted(surface.xs, entry_local[0]), 0, len(surface.xs) - 1))
    j = int(np.clip(np.searchsorted(surface.ys, entry_local[1]), 0, len(surface.ys) - 1))
    entry_local[2] = float(surface.h[i, j])
    entry_world = surface.frame.to_world(entry_local)

    # Run from just inside the cavity out past the block's furthest face.
    start = entry_world - pour_axis * 3.0
    reach = float((_block_corners_world(local_bounds, surface.frame) @ pour_axis).max())
    length = reach - float(start @ pour_axis) + 3.0

    funnel = frustum(cfg.inner_radius, cfg.outer_radius, length, sections=64)
    rot = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], pour_axis)
    funnel.apply_transform(rot)
    funnel.apply_translation(start)

    half_a = _difference(half_a, funnel, "spout -> A")
    half_b = _difference(half_b, funnel, "spout -> B")

    info = {
        "entry_world": [round(float(x), 2) for x in entry_world],
        "axis": [round(float(x), 4) for x in pour_axis],
        "length_mm": round(length, 2),
        "inner_radius_mm": cfg.inner_radius,
        "outer_radius_mm": cfg.outer_radius,
    }
    log.info("added pour spout at %s", np.round(entry_world, 1))
    return half_a, half_b, info


def _block_corners(local_bounds: np.ndarray) -> np.ndarray:
    lo, hi = local_bounds
    return np.array(
        [
            [x, y, z]
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]
    )


def _block_corners_world(local_bounds: np.ndarray, frame: Frame) -> np.ndarray:
    return frame.to_world(_block_corners(local_bounds))


# --------------------------------------------------------------------------
# vents
# --------------------------------------------------------------------------


def find_air_traps(
    cavity: trimesh.Trimesh,
    pour_axis: np.ndarray,
    cfg: VentConfig,
    *,
    spout_world: np.ndarray | None = None,
    grid: float = 2.0,
) -> list[np.ndarray]:
    """Cavity high points that air would collect under while the mold fills.

    Casts a grid of rays along the pour axis, takes the topmost cavity crossing
    per column to get the cavity's ceiling, and keeps the local maxima of that
    field. Those are the spots liquid closes off before the air can leave.
    """
    from . import demold

    frame = Frame.from_direction(pour_axis)
    hull_local = frame.to_local(cavity.convex_hull.vertices)
    lo, hi = hull_local.min(axis=0), hull_local.max(axis=0)
    xs = np.arange(lo[0], hi[0] + grid, grid)
    ys = np.arange(lo[1], hi[1] + grid, grid)
    cast = demold.cast_grid(cavity, frame, xs, ys)

    nx, ny = len(xs), len(ys)
    ceiling = np.full(nx * ny, -np.inf)
    hit = cast.counts > 0
    if hit.any():
        last = cast.starts[hit] + cast.counts[hit] - 1
        ceiling[hit] = cast.local_z(cast.t[last])
    ceiling = ceiling.reshape(nx, ny)

    filled = np.where(np.isfinite(ceiling), ceiling, -np.inf)
    peak = ndimage.maximum_filter(filled, size=5, mode="nearest")
    is_peak = np.isfinite(ceiling) & (filled >= peak - 1e-9)
    if not is_peak.any():
        return []

    # One vent per *pocket*. A flat ceiling is a plateau of equal-height cells
    # that all pass the local-maximum test, and venting it once at its high
    # point is enough -- without this, a single broad flat face would swallow the
    # whole vent budget and starve genuinely separate pockets.
    labels, n_labels = ndimage.label(is_peak)
    if n_labels == 0:
        return []
    positions = ndimage.maximum_position(
        filled, labels=labels, index=np.arange(1, n_labels + 1)
    )
    idx = np.asarray(positions, dtype=np.int64).reshape(-1, 2)
    heights = ceiling[idx[:, 0], idx[:, 1]]
    order = np.argsort(-heights)
    idx, heights = idx[order], heights[order]

    picked: list[np.ndarray] = []
    for (i, j), z in zip(idx, heights):
        p_local = np.array([xs[i], ys[j], z])
        p_world = frame.to_world(p_local)
        if spout_world is not None and np.linalg.norm(p_world - spout_world) < max(
            cfg.min_spacing, 15.0
        ):
            continue
        if any(np.linalg.norm(p_world - q) < cfg.min_spacing for q in picked):
            continue
        picked.append(p_world)
        if len(picked) >= cfg.count:
            break
    return picked


def add_vents(
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    cavity: trimesh.Trimesh,
    frame: Frame,
    local_bounds: np.ndarray,
    points: list[np.ndarray],
    cfg: VentConfig,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Drill a thin channel from each air trap out to the block's surface.

    Routed along the pull direction, picking the side whose ray does not
    re-enter the cavity, so the channel is a straight run through mold material
    that releases along the direction the half already moves in.
    """
    d = frame.direction
    cut: list[dict] = []

    for p in points:
        chosen_sign = None
        for sign in (1.0, -1.0):
            origin = (p + sign * d * 0.25).reshape(1, 3)
            if not cavity.ray.intersects_any(origin, (sign * d).reshape(1, 3))[0]:
                chosen_sign = sign
                break
        if chosen_sign is None:
            log.info("skipping vent at %s: both routes re-enter the cavity", np.round(p, 1))
            continue

        pz = float(frame.to_local(p)[2])
        if chosen_sign > 0:
            length = float(local_bounds[1, 2]) - pz + 3.0
        else:
            length = pz - float(local_bounds[0, 2]) + 3.0

        channel = frustum(cfg.radius, cfg.radius, length + 2.0, sections=20)
        rot = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], chosen_sign * d)
        channel.apply_transform(rot)
        channel.apply_translation(p - chosen_sign * d * 2.0)

        half_a = _difference(half_a, channel, "vent -> A")
        half_b = _difference(half_b, channel, "vent -> B")
        cut.append(
            {
                "point_world": [round(float(x), 2) for x in p],
                "along": "+d" if chosen_sign > 0 else "-d",
                "length_mm": round(length, 2),
            }
        )

    log.info("added %d vents", len(cut))
    return half_a, half_b, {"count": len(cut), "radius_mm": cfg.radius, "vents": cut}


# --------------------------------------------------------------------------


@dataclass
class FeatureReport:
    keys: dict = field(default_factory=dict)
    spout: dict = field(default_factory=dict)
    vents: dict = field(default_factory=dict)
    pour_axis: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "pour_axis": self.pour_axis,
            "keys": self.keys,
            "spout": self.spout,
            "vents": self.vents,
        }


def apply_features(
    result: MoldResult,
    cavity: trimesh.Trimesh,
    cfg: MoldConfig,
    *,
    progress=None,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, FeatureReport]:
    """Add keys, spout and vents to an already-split mold."""
    report = progress or (lambda *a, **k: None)
    half_a, half_b = result.half_a, result.half_b
    out = FeatureReport()

    pour_axis = choose_pour_axis(cavity, cfg)
    out.pour_axis = [round(float(x), 4) for x in pour_axis]

    if cfg.keys.enabled:
        report(0.76, "adding alignment keys")
        half_a, half_b, out.keys = add_keys(
            half_a, half_b, result.surface, result.local_bounds, cfg.keys
        )

    spout_world = None
    if cfg.spout.enabled:
        report(0.84, "cutting pour spout")
        half_a, half_b, out.spout = add_spout(
            half_a,
            half_b,
            cavity,
            result.surface,
            result.local_bounds,
            pour_axis,
            cfg.spout,
        )
        if out.spout.get("entry_world"):
            spout_world = np.array(out.spout["entry_world"], dtype=np.float64)

    if cfg.vents.enabled:
        report(0.9, "cutting vents")
        traps = find_air_traps(cavity, pour_axis, cfg.vents, spout_world=spout_world)
        half_a, half_b, out.vents = add_vents(
            half_a,
            half_b,
            cavity,
            result.frame,
            result.local_bounds,
            traps,
            cfg.vents,
        )

    return half_a, half_b, out
