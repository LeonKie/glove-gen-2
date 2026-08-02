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

Choosing is separate from cutting
---------------------------------
*Where* the knobs and holes go is a :class:`FeaturePlan`: a plain, serialisable
list of items, each with a world position and its own sizes. Building one is
cheap (ray casts and a distance transform); applying one is the expensive part
(a boolean per item on million-face halves).

Keeping the two apart is what lets the same code be automatic in one place and
interactive in another. The CLI runs :func:`plan_features` and
:func:`apply_plan` back to back, so it stays hands-off. The web app builds the
mold, shows the proposal, and re-applies an edited plan to the *base* halves --
which costs the features only, not the block, the parting surface or the split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

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


def place_local(mesh: trimesh.Trimesh, frame: Frame, origin_local, axis_local=None):
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
# the plan
# --------------------------------------------------------------------------

# The order items are cut in, and so the order they are listed in. ``plate``
# leads because it is the plane cut: it decides how much mold there is at all,
# and every core item after it stands on the face it leaves.
KINDS = ("plate", "key", "spout", "vent", "core_tab", "dowel", "screw", "port")

# Kinds that only mean anything when a core was built. They are proposed only
# for a core run and skipped with a reason otherwise, rather than quietly
# dropped, so an edited plan carried over from a plain mold explains itself.
CORE_KINDS = ("plate", "core_tab", "dowel", "screw", "port")

# ...and of those, the ones that only mean anything once a plate has cut the
# mold. A dowel is not one of them: it locks the core to the halves and needs
# no plate at all, which is the whole difference between it and a screw.
PLATE_KINDS = ("screw", "port")

# Sizes a caller may set per item, with the range each is clamped to. The
# bounds are sanity rails, not opinions: they keep a stray unit (metres for
# millimetres) or a typo'd zero from producing a boolean that runs for minutes
# and returns garbage. The web app mirrors them on its number inputs; this is
# the side that is enforced.
PARAM_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "key": {
        "radius": (0.5, 50.0),
        "height": (0.5, 50.0),
        "draft_deg": (0.0, 45.0),
        "clearance": (0.0, 3.0),
    },
    "spout": {"inner_radius": (0.5, 60.0), "outer_radius": (0.5, 80.0)},
    "vent": {"radius": (0.2, 20.0)},
    "plate": {"thickness": (2.0, 60.0)},
    "core_tab": {"radius": (0.5, 20.0), "clearance": (0.0, 3.0)},
    "dowel": {
        "radius": (1.0, 20.0),
        "engagement": (1.0, 60.0),
        "clearance": (0.0, 2.0),
    },
    "screw": {"radius": (0.5, 10.0), "depth": (2.0, 60.0), "clearance": (0.0, 3.0)},
    "port": {"inner_radius": (0.5, 30.0), "outer_radius": (1.0, 60.0)},
}


class FeatureSkipped(Exception):
    """This item cannot be cut where it was asked for, and why."""


def _defaults_for(kind: str, cfg: MoldConfig | None = None) -> dict[str, float]:
    cfg = cfg or MoldConfig()
    if kind == "key":
        k: KeyConfig = cfg.keys
        return {
            "radius": k.radius,
            "height": k.height,
            "draft_deg": k.draft_deg,
            "clearance": k.clearance,
        }
    if kind == "spout":
        s: SpoutConfig = cfg.spout
        return {"inner_radius": s.inner_radius, "outer_radius": s.outer_radius}
    if kind == "vent":
        v: VentConfig = cfg.vents
        return {"radius": v.radius}
    if kind == "plate":
        return {"thickness": cfg.carrier.plate_thickness}
    if kind == "core_tab":
        return {"radius": cfg.core_tabs.radius, "clearance": cfg.core_tabs.clearance}
    if kind == "dowel":
        d = cfg.core_dowels
        return {
            "radius": d.radius,
            "engagement": d.engagement,
            "clearance": d.clearance,
        }
    if kind == "screw":
        c = cfg.carrier
        return {
            "radius": c.screw_radius,
            "depth": c.screw_depth,
            "clearance": c.screw_clearance,
        }
    if kind == "port":
        c = cfg.carrier
        return {"inner_radius": c.port_inner_radius, "outer_radius": c.port_outer_radius}
    raise ValueError(f"unknown feature kind {kind!r}, expected one of {KINDS}")


def _clean_params(kind: str, params: dict | None, cfg: MoldConfig | None = None) -> dict:
    """Fill in defaults for anything missing and clamp everything else."""
    defaults = _defaults_for(kind, cfg)
    given = dict(params or {})
    out: dict[str, float] = {}
    for name, (lo, hi) in PARAM_BOUNDS[kind].items():
        raw = given.get(name, defaults[name])
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{kind}.{name} must be a number, got {raw!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"{kind}.{name} must be finite, got {raw!r}")
        out[name] = float(np.clip(value, lo, hi))
    return out


def _clean_vector(value, name: str) -> list[float]:
    v = np.asarray(value, dtype=np.float64).reshape(-1)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must be three finite numbers, got {value!r}")
    return [float(x) for x in v]


@dataclass
class FeatureItem:
    """One knob or hole: what it is, where it goes, how big it is."""

    kind: str
    position: list[float]  # world coordinates
    params: dict = field(default_factory=dict)
    enabled: bool = True
    id: str = ""
    source: str = "auto"  # "auto" (proposed) or "user" (placed by hand)
    note: str = ""  # why the automatic pass chose this spot

    def normalised(self, cfg: MoldConfig | None = None) -> "FeatureItem":
        if self.kind not in KINDS:
            raise ValueError(f"unknown feature kind {self.kind!r}, expected one of {KINDS}")
        return replace(
            self,
            position=_clean_vector(self.position, f"{self.kind} position"),
            params=_clean_params(self.kind, self.params, cfg),
            enabled=bool(self.enabled),
            source="user" if self.source == "user" else "auto",
            note=str(self.note or ""),
        )

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "enabled": bool(self.enabled),
            "source": self.source,
            "position": [round(float(v), 3) for v in self.position],
            "params": {k: round(float(v), 4) for k, v in self.params.items()},
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureItem":
        data = dict(data or {})
        return cls(
            kind=str(data.get("kind", "")),
            position=data.get("position", [0.0, 0.0, 0.0]),
            params=data.get("params") or {},
            enabled=bool(data.get("enabled", True)),
            id=str(data.get("id", "")),
            source=str(data.get("source", "auto")),
            note=str(data.get("note", "")),
        )


@dataclass
class FeaturePlan:
    """Every knob and hole to cut, in the order they get cut."""

    items: list[FeatureItem] = field(default_factory=list)
    pour_axis: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])

    def of_kind(self, kind: str) -> list[FeatureItem]:
        return [i for i in self.items if i.kind == kind]

    def enabled_items(self) -> list[FeatureItem]:
        """Enabled items, grouped by kind so a run is reproducible.

        Keys go in before holes are cut: a socket swept out of half B must not
        be re-filled by a later union, and cutting the spout after the keys
        means a spout that clips a key still leaves a usable half.
        """
        order = {kind: n for n, kind in enumerate(KINDS)}
        return sorted(
            (i for i in self.items if i.enabled), key=lambda i: order[i.kind]
        )

    def normalised(self, cfg: MoldConfig | None = None) -> "FeaturePlan":
        """Validate every item, clamp sizes, and give everything a unique id.

        Ids come back from a browser, so they are treated as suggestions:
        blanks and collisions are resolved here rather than trusted, because
        the report keys off them.
        """
        seen: set[str] = set()
        out: list[FeatureItem] = []
        for n, item in enumerate(self.items, start=1):
            item = item.normalised(cfg)
            ident = item.id.strip() or f"{item.kind}-{n}"
            while ident in seen:
                ident = f"{ident}'"
            seen.add(ident)
            out.append(replace(item, id=ident))
        return FeaturePlan(items=out, pour_axis=_clean_vector(self.pour_axis, "pour_axis"))

    def as_dict(self) -> dict:
        return {
            "pour_axis": [round(float(v), 4) for v in self.pour_axis],
            "items": [i.as_dict() for i in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict | "FeaturePlan") -> "FeaturePlan":
        if isinstance(data, FeaturePlan):
            return data
        data = dict(data or {})
        raw = data.get("items")
        if raw is None:
            raw = []
        if not isinstance(raw, (list, tuple)):
            raise ValueError("feature plan 'items' must be a list")
        return cls(
            items=[FeatureItem.from_dict(i) for i in raw],
            pour_axis=data.get("pour_axis") or [0.0, 0.0, 1.0],
        )


@dataclass
class FeatureContext:
    """Everything a feature needs to know about the mold it is cut into.

    Deliberately smaller than :class:`~glovegen.mold.MoldResult`: the block and
    the uncut mold are not needed to place or cut a feature, so a re-apply can
    reconstruct this from the parting surface, the block's bounds and the
    cavity alone -- none of which have to be recomputed.
    """

    surface: PartingSurface
    local_bounds: np.ndarray  # (2, 3) block extent in the pull frame
    cavity: trimesh.Trimesh
    # Only a core run fills these in. The block is what the carrier plate is
    # sliced out of -- the halves cannot stand in for it, because the plate has
    # to span the cavity opening as well, which is exactly the part of the block
    # they are missing. The core is the eroded body before any of the plan
    # touches it, so a re-apply can re-cut from it without eroding again.
    block: trimesh.Trimesh | None = None
    core: trimesh.Trimesh | None = None
    _free_dist: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def from_mold(
        cls,
        result: MoldResult,
        cavity: trimesh.Trimesh | None = None,
        *,
        core: trimesh.Trimesh | None = None,
    ) -> "FeatureContext":
        return cls(
            surface=result.surface,
            local_bounds=np.asarray(result.local_bounds, dtype=np.float64),
            cavity=cavity if cavity is not None else result.cavity,
            block=result.block,
            core=core,
        )

    @property
    def frame(self) -> Frame:
        return self.surface.frame

    @property
    def direction(self) -> np.ndarray:
        return self.surface.frame.direction

    def free_distance(self) -> np.ndarray:
        """Distance from each grid node to the nearest column hitting the part.

        Zero on columns that pass through the cavity, so it doubles as the
        "how much room is there for a knob here" field. Cached: the transform
        is cheap but every key placement asks for it.
        """
        if self._free_dist is None:
            surface = self.surface
            nx, ny = surface.shape
            dx = float(surface.xs[1] - surface.xs[0]) if nx > 1 else 1.0
            dy = float(surface.ys[1] - surface.ys[0]) if ny > 1 else 1.0
            self._free_dist = ndimage.distance_transform_edt(
                ~surface.constrained, sampling=(dx, dy)
            )
        return self._free_dist

    def node_index(self, local_xy) -> tuple[int, int]:
        """Nearest grid node to a local (x, y)."""
        surface = self.surface
        i = int(np.clip(np.searchsorted(surface.xs, local_xy[0]), 0, len(surface.xs) - 1))
        j = int(np.clip(np.searchsorted(surface.ys, local_xy[1]), 0, len(surface.ys) - 1))
        return i, j

    def on_parting(self, world_point) -> tuple[np.ndarray, tuple[int, int]]:
        """Drop a world point onto the parting surface, in local coordinates."""
        local = self.frame.to_local(np.asarray(world_point, dtype=np.float64).reshape(3))
        i, j = self.node_index(local)
        local = local.copy()
        local[2] = float(self.surface.h[i, j])
        return local, (i, j)


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
    ctx: FeatureContext, cfg: KeyConfig, *, allow: np.ndarray | None = None
) -> list[KeySite]:
    """Pick well-spread spots on the parting face that clear the cavity.

    Keys must sit where the parting face is solid mold on both sides, i.e. in
    columns that miss the part entirely, with room for the key body above and
    the socket below.

    ``allow`` masks the grid down to nodes a key may use. A core run passes the
    region below the carrier plate's seating face: a key placed above it would
    be sliced in two when the plate is trimmed off.
    """
    surface = ctx.surface
    nx, ny = surface.shape
    dx = float(surface.xs[1] - surface.xs[0]) if nx > 1 else 1.0
    dy = float(surface.ys[1] - surface.ys[0]) if ny > 1 else 1.0

    free = ~surface.constrained
    dist = ctx.free_distance()

    need = cfg.radius + cfg.cavity_margin
    ok = free & (dist >= need)
    if allow is not None:
        ok &= allow

    # Stay clear of the block's outside walls too.
    border_x = int(np.ceil((cfg.radius + 2.0) / dx))
    border_y = int(np.ceil((cfg.radius + 2.0) / dy))
    edge = np.zeros_like(ok)
    edge[border_x : nx - border_x, border_y : ny - border_y] = True
    ok &= edge

    # Vertical room: key body hangs below the surface, socket above it.
    room_below = surface.h - ctx.local_bounds[0, 2]
    room_above = ctx.local_bounds[1, 2] - surface.h
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


def _apply_key(
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    ctx: FeatureContext,
    item: FeatureItem,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Add one male key to half A and its matching socket to half B.

    The key hangs *down* from A's parting face and widens upward at the draft
    angle, so it withdraws from its socket along +d as the halves part.
    """
    p = item.params
    radius, height = p["radius"], p["height"]
    local, (i, j) = ctx.on_parting(item.position)
    z = float(local[2])

    # A key is only mold-on-both-sides where the column misses the part, and it
    # needs its whole footprint clear -- a knob that breaks into the cavity
    # would print into the cast and open the socket into it.
    clearance = float(ctx.free_distance()[i, j])
    if clearance < radius:
        raise FeatureSkipped(
            f"only {clearance:.1f} mm to the cavity, needs {radius:.1f} mm for the knob"
        )
    if z - height - 2.0 < float(ctx.local_bounds[0, 2]):
        raise FeatureSkipped("not enough mold below the parting face for the knob")
    if float(ctx.local_bounds[1, 2]) - z < 2.0:
        raise FeatureSkipped("not enough mold above the parting face for the socket")

    draft = np.tan(np.radians(p["draft_deg"]))
    r_bottom = max(radius - height * draft, 0.35 * radius)
    c = p["clearance"]
    overlap = 3.0  # merge depth into A's body, avoids a coplanar boolean
    z_sweep = float(ctx.local_bounds[1, 2]) + 1.0

    key = revolved(
        [
            (r_bottom, z - height),
            (radius, z),
            (radius, z + overlap),
        ]
    )
    key = place_local(key, ctx.frame, (local[0], local[1], 0.0))
    half_a = _union(half_a, key, "key -> A")

    # Socket: the key's *swept* volume as it withdraws along +d, not just its
    # resting shape. The parting surface is curved, so half B can hold material
    # above the key's mouth within the key's own footprint; without sweeping the
    # socket to the top of the block, the key would jam on it.
    socket = revolved(
        [
            (r_bottom + c, z - height - c),
            (radius + c, z),
            (radius + c, z_sweep),
        ]
    )
    socket = place_local(socket, ctx.frame, (local[0], local[1], 0.0))
    half_b = _difference(half_b, socket, "socket -> B")

    detail = KeySite(
        x=float(local[0]), y=float(local[1]), z=z, clearance_to_cavity=clearance
    ).as_dict()
    detail["id"] = item.id
    detail["radius_mm"] = round(radius, 2)
    detail["height_mm"] = round(height, 2)
    return half_a, half_b, detail


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


def spout_entry(
    cavity: trimesh.Trimesh, ctx: FeatureContext, pour_axis: np.ndarray
) -> np.ndarray:
    """Where the funnel should break into the cavity, in world coordinates.

    The cavity's extreme along the pour axis -- its highest point once the mold
    is stood up to pour -- pulled onto the parting surface so each half gets a
    clean half-round groove.
    """
    v = np.asarray(cavity.vertices, dtype=np.float64)
    proj = v @ unit(pour_axis)
    top = proj >= np.percentile(proj, 99.5)
    entry_world = v[top].mean(axis=0)
    # Over the short run through the block wall the parting surface barely
    # moves, so centring on it keeps the groove half-round to within a fraction
    # of a millimetre.
    local, _ = ctx.on_parting(entry_world)
    return ctx.frame.to_world(local)


def _apply_spout(
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    ctx: FeatureContext,
    item: FeatureItem,
    pour_axis: np.ndarray,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Cut one funnel from the outside of the block into the cavity."""
    pour_axis = unit(pour_axis)
    local, _ = ctx.on_parting(item.position)
    entry_world = ctx.frame.to_world(local)

    # Run from just inside the cavity out past the block's furthest face.
    start = entry_world - pour_axis * 3.0
    reach = float((_block_corners_world(ctx.local_bounds, ctx.frame) @ pour_axis).max())
    length = reach - float(start @ pour_axis) + 3.0
    if length <= 1.0:
        raise FeatureSkipped("spout entry is outside the block along the pour axis")

    funnel = frustum(
        item.params["inner_radius"], item.params["outer_radius"], length, sections=64
    )
    rot = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], pour_axis)
    funnel.apply_transform(rot)
    funnel.apply_translation(start)

    half_a = _difference(half_a, funnel, "spout -> A")
    half_b = _difference(half_b, funnel, "spout -> B")

    detail = {
        "id": item.id,
        "entry_world": [round(float(x), 2) for x in entry_world],
        "axis": [round(float(x), 4) for x in pour_axis],
        "length_mm": round(length, 2),
        "inner_radius_mm": round(item.params["inner_radius"], 2),
        "outer_radius_mm": round(item.params["outer_radius"], 2),
    }
    log.info("added pour spout at %s", np.round(entry_world, 1))
    return half_a, half_b, detail


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


def _apply_vent(
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    ctx: FeatureContext,
    item: FeatureItem,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Drill one thin channel from an air trap out to the block's surface.

    Routed along the pull direction, picking the side whose ray does not
    re-enter the cavity, so the channel is a straight run through mold material
    that releases along the direction the half already moves in.
    """
    d = ctx.direction
    p = np.asarray(item.position, dtype=np.float64).reshape(3)
    radius = item.params["radius"]

    chosen_sign = None
    for sign in (1.0, -1.0):
        origin = (p + sign * d * 0.25).reshape(1, 3)
        if not ctx.cavity.ray.intersects_any(origin, (sign * d).reshape(1, 3))[0]:
            chosen_sign = sign
            break
    if chosen_sign is None:
        raise FeatureSkipped("both routes out re-enter the cavity")

    pz = float(ctx.frame.to_local(p)[2])
    if chosen_sign > 0:
        length = float(ctx.local_bounds[1, 2]) - pz + 3.0
    else:
        length = pz - float(ctx.local_bounds[0, 2]) + 3.0
    if length <= 1.0:
        raise FeatureSkipped("vent starts outside the block")

    channel = frustum(radius, radius, length + 2.0, sections=20)
    rot = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], chosen_sign * d)
    channel.apply_transform(rot)
    channel.apply_translation(p - chosen_sign * d * 2.0)

    half_a = _difference(half_a, channel, "vent -> A")
    half_b = _difference(half_b, channel, "vent -> B")
    return half_a, half_b, {
        "id": item.id,
        "point_world": [round(float(x), 2) for x in p],
        "along": "+d" if chosen_sign > 0 else "-d",
        "length_mm": round(length, 2),
        "radius_mm": round(radius, 2),
    }


# --------------------------------------------------------------------------
# planning and applying
# --------------------------------------------------------------------------


@dataclass
class Bodies:
    """What a plan is cut into, and what comes out the other side.

    Two halves on a plain run; three once a core is in play. The core is a
    result rather than an input because the plan *builds* it -- the plate, the
    dowels and the tabs are all unioned onto the eroded body.
    """

    half_a: trimesh.Trimesh
    half_b: trimesh.Trimesh
    core: trimesh.Trimesh | None = None
    # The dowel pins, fused into one mesh. Deliberately outside the iteration
    # below: they are not a body of the mold but loose hardware, and a length
    # of the right rod does the job just as well as a printed one.
    pins: trimesh.Trimesh | None = None

    def __iter__(self):
        """So ``half_a, half_b, core = bodies`` reads the obvious way."""
        return iter((self.half_a, self.half_b, self.core))


@dataclass
class FeatureReport:
    keys: dict = field(default_factory=dict)
    spout: dict = field(default_factory=dict)
    vents: dict = field(default_factory=dict)
    # Only a core run fills this in: the wall it measured, the plate it cut and
    # what the seam tabs cost the cast.
    core: dict = field(default_factory=dict)
    pour_axis: list[float] = field(default_factory=list)
    # One entry per planned item, applied or not, so a caller can tell the
    # difference between "you switched it off" and "it would not fit".
    items: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "pour_axis": self.pour_axis,
            "keys": self.keys,
            "spout": self.spout,
            "vents": self.vents,
            "core": self.core,
            "items": self.items,
        }

    def skipped(self) -> list[dict]:
        return [i for i in self.items if i["status"] == "skipped"]


def plan_features(
    ctx: FeatureContext, cfg: MoldConfig | None = None
) -> FeaturePlan:
    """Propose where the knobs and holes go. Cheap: no booleans, no geometry.

    This is the automatic pass. Everything it decides is expressed as plain
    data, so it can be shown, edited and re-applied rather than only obeyed.
    """
    cfg = cfg or MoldConfig()
    pour_axis = choose_pour_axis(ctx.cavity, cfg)
    items: list[FeatureItem] = []

    def add(kind: str, position, note: str, n: int | None = None) -> None:
        items.append(
            FeatureItem(
                id=f"{kind}-{n}" if n else f"{kind}-1",
                kind=kind,
                position=[float(v) for v in position],
                params=_defaults_for(kind, cfg),
                note=note,
            )
        )

    # A carrier plate cuts the mold in two along a plane, so everything else has
    # to be proposed on the side that survives. Staging it here -- one boolean
    # on the block and one on the core, no touching of the halves -- is what
    # lets the rest of the pass see the geometry the cut will actually leave.
    state = _stage_core(ctx, pour_axis, cfg)
    ceiling = state.plane if state is not None else None
    if state is not None and state.has_plate:
        add(
            "plate",
            state.plate_point,
            f"cuts the mold {state.discarded_note}",
        )

    allow = None
    if ceiling is not None:
        from . import core as core_mod

        allow = core_mod.nodes_below(ctx.surface, pour_axis, ceiling - cfg.keys.radius)

    if cfg.keys.enabled:
        for n, site in enumerate(find_key_sites(ctx, cfg.keys, allow=allow), start=1):
            items.append(
                FeatureItem(
                    id=f"key-{n}",
                    kind="key",
                    position=[
                        float(v)
                        for v in ctx.frame.to_world(np.array([site.x, site.y, site.z]))
                    ],
                    params=_defaults_for("key", cfg),
                    note=f"{site.clearance_to_cavity:.1f} mm clear of the cavity",
                )
            )

    # With a plate on, the cavity's high point along the pour axis *is* the cut
    # face, and the plate seals the annulus shut over it. A spout aimed there
    # would be cut into material the plane discards; the port through the plate
    # is the way in instead.
    spout_world = None
    if cfg.spout.enabled and not (state is not None and state.has_plate):
        spout_world = spout_entry(ctx.cavity, ctx, pour_axis)
        add("spout", spout_world, "cavity high point along the pour axis")

    if cfg.vents.enabled:
        traps = find_air_traps(
            ctx.cavity, pour_axis, cfg.vents, spout_world=spout_world
        )
        if ceiling is not None:
            traps = [t for t in traps if float(t @ pour_axis) <= ceiling]
        for n, point in enumerate(traps, start=1):
            add("vent", point, "trapped-air pocket", n)

    if state is not None:
        from . import core as core_mod

        # Tabs and dowels want the same spots -- both run from the core out to
        # solid mold on the seam -- so whichever is planned first is kept clear
        # of by the other.
        seam: list = []
        if cfg.core_tabs.enabled:
            for n, (anchor, grip) in enumerate(
                core_mod.find_tab_sites(ctx, state, cfg), start=1
            ):
                add("core_tab", anchor, "pinched on the parting seam", n)
                seam.append(grip)
        if cfg.core_dowels.enabled:
            for n, (anchor, grip) in enumerate(
                core_mod.find_dowel_sites(ctx, state, cfg, avoid=seam), start=1
            ):
                add("dowel", anchor, "a pin locks the core to both halves", n)
                seam.append(grip)
        if state.has_plate:
            port = core_mod.find_port_site(ctx, state)
            if port is not None:
                add("port", port, "over the ring of cast at the cut face")
            for n, point in enumerate(
                core_mod.find_screw_sites(
                    ctx, state, cfg, [port] if port is not None else []
                ),
                start=1,
            ):
                add("screw", point, "clamps the plate down", n)

    plan = FeaturePlan(items=items, pour_axis=[float(v) for v in pour_axis])
    log.info(
        "planned %d feature(s): %s",
        len(items),
        {k: len(plan.of_kind(k)) for k in KINDS},
    )
    return plan.normalised(cfg)


def _stage_core(ctx: FeatureContext, pour_axis, cfg: MoldConfig):
    """Work out what the cut will leave, without cutting anything yet.

    Placement has to happen against the geometry the plate's plane will produce,
    not the geometry as it stands: a dowel proposed where the block is about to
    be thrown away is a dowel nobody can use. Staging costs one boolean on the
    block and one on the core, which is what buys every later site the right
    answer.
    """
    if ctx.core is None:
        return None
    from . import core as core_mod

    state = core_mod.CoreState(body=ctx.core, axis=unit(pour_axis))
    if not cfg.carrier.enabled:
        return state
    try:
        core_mod.stage_plate(
            ctx,
            core_mod.default_plate_point(ctx.cavity, pour_axis, cfg),
            cfg.carrier.plate_thickness,
            state,
        )
    except FeatureSkipped as exc:
        log.warning("no carrier plate proposed: %s", exc)
    return state


def _aggregate(report: FeatureReport, plan: FeaturePlan, details: dict[str, list[dict]]) -> None:
    """Fill the per-kind summaries the report has always carried."""
    if plan.of_kind("key"):
        sites = details["key"]
        report.keys = {
            "count": len(sites),
            "radius_mm": max((s["radius_mm"] for s in sites), default=0.0),
            "height_mm": max((s["height_mm"] for s in sites), default=0.0),
            "sites": sites,
        }
    if plan.of_kind("spout"):
        spouts = details["spout"]
        report.spout = dict(spouts[0]) if spouts else {"count": 0}
        if len(spouts) > 1:
            report.spout["extra"] = spouts[1:]
    if plan.of_kind("vent"):
        vents = details["vent"]
        report.vents = {
            "count": len(vents),
            "radius_mm": max((v["radius_mm"] for v in vents), default=0.0),
            "vents": vents,
        }
    for kind in CORE_KINDS:
        if kind != "plate" and plan.of_kind(kind):
            report.core[f"{kind}s"] = details[kind]


def apply_plan(
    half_a: trimesh.Trimesh,
    half_b: trimesh.Trimesh,
    ctx: FeatureContext,
    plan: FeaturePlan | dict,
    *,
    cfg: MoldConfig | None = None,
    progress=None,
) -> tuple[Bodies, FeatureReport]:
    """Cut a plan into a pair of base halves.

    Items that cannot be placed are reported and skipped, never fatal: one knob
    landing over the cavity is not a reason to throw away a mold that took a
    minute to build.

    ``progress`` is called with a fraction of *this* step, 0 to 1; the caller
    decides where that sits in its own run.
    """
    from . import core as core_mod  # core builds on this module's primitives

    report_progress = progress or (lambda *a, **k: None)
    cfg = cfg or MoldConfig()
    plan = FeaturePlan.from_dict(plan).normalised(cfg)
    pour_axis = unit(plan.pour_axis)

    bodies = Bodies(half_a=half_a, half_b=half_b)
    state = (
        core_mod.CoreState(body=ctx.core, axis=pour_axis) if ctx.core is not None else None
    )

    out = FeatureReport(pour_axis=[round(float(v), 4) for v in pour_axis])
    details: dict[str, list[dict]] = {k: [] for k in KINDS}
    todo = plan.enabled_items()
    by_id = {item.id: item for item in todo}

    for n, item in enumerate(todo):
        report_progress(n / max(1, len(todo)), f"cutting {item.kind} {n + 1}/{len(todo)}")
        entry = item.as_dict()
        try:
            if item.kind in CORE_KINDS and state is None:
                raise FeatureSkipped(
                    "this mold has no core, so there is nothing for a "
                    f"{item.kind.replace('_', ' ')} to attach to"
                )
            if item.kind in PLATE_KINDS and not state.has_plate:
                raise FeatureSkipped(
                    f"a {item.kind} needs the carrier plate, which is not in "
                    "this plan"
                )
            if item.kind == "key":
                bodies.half_a, bodies.half_b, detail = _apply_key(
                    bodies.half_a, bodies.half_b, ctx, item
                )
            elif item.kind == "spout":
                bodies.half_a, bodies.half_b, detail = _apply_spout(
                    bodies.half_a, bodies.half_b, ctx, item, pour_axis
                )
            elif item.kind == "vent":
                bodies.half_a, bodies.half_b, detail = _apply_vent(
                    bodies.half_a, bodies.half_b, ctx, item
                )
            elif item.kind == "plate":
                detail = core_mod.apply_plate(bodies, ctx, item, state, cfg)
            elif item.kind == "core_tab":
                detail = core_mod.apply_tab(bodies, ctx, item, state, cfg)
            elif item.kind == "dowel":
                detail = core_mod.apply_dowel(bodies, ctx, item, state, cfg)
            elif item.kind == "screw":
                detail = core_mod.apply_screw(bodies, ctx, item, state, cfg)
            else:
                detail = core_mod.apply_port(bodies, ctx, item, state, cfg)
        except FeatureSkipped as exc:
            log.info("skipping %s %s: %s", item.kind, item.id, exc)
            entry.update(status="skipped", reason=str(exc))
        else:
            details[item.kind].append(detail)
            entry.update(status="applied", reason="", detail=detail)
        out.items.append(entry)

    for item in plan.items:
        if item.id not in by_id:
            out.items.append({**item.as_dict(), "status": "off", "reason": ""})

    if state is not None:
        report_progress(0.95, "fusing the core assembly")
        bodies.core = state.assembly()
        bodies.pins = state.pin_stock()
        out.core = _core_summary(state, ctx, cfg, details, bodies.core)

    _aggregate(out, plan, details)
    log.info(
        "applied %d/%d feature(s)",
        sum(1 for i in out.items if i["status"] == "applied"),
        len(plan.items),
    )
    return bodies, out


def _core_summary(state, ctx, cfg: MoldConfig, details: dict, assembly) -> dict:
    """The measured half of a core run: the wall, and what the tabs cost."""
    from . import core as core_mod

    cut = None if state.plane is None else state.plane - cfg.core.wall
    return {
        "wall": core_mod.measure_wall(
            state.wall_body if state.wall_body is not None else state.body,
            ctx.cavity,
            cfg.core.wall,
            pour_axis=state.axis,
            cut=cut,
        ),
        "plate": details["plate"][0] if details["plate"] else None,
        # A core in two pieces is not a core. It happens when a bore crosses
        # the root of a tab, so it is worth saying out loud rather than leaving
        # to be discovered in a slicer.
        "pieces": core_mod.real_pieces(assembly),
        "tab_through_wall_mm3": core_mod.tab_protrusion(
            state.tabs, ctx.cavity, state.body
        ),
    }


def apply_features(
    result: MoldResult,
    cavity: trimesh.Trimesh,
    cfg: MoldConfig,
    *,
    progress=None,
) -> tuple[Bodies, FeatureReport]:
    """Plan and cut features in one go: the fully automatic path."""
    ctx = FeatureContext.from_mold(result, cavity)
    plan = plan_features(ctx, cfg)
    return apply_plan(
        result.half_a, result.half_b, ctx, plan, cfg=cfg, progress=progress
    )
