"""Tunables for the mold pipeline. All lengths are in millimetres."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DemoldConfig:
    """Pull-direction search."""

    # Ray-cast proxy: the direction sweep casts millions of rays, so it runs
    # against a decimated stand-in rather than the full-resolution scan.
    proxy_faces: int = 150_000

    coarse_dirs: int = 220
    coarse_grid: int = 64
    fine_dirs: int = 500
    fine_grid: int = 128
    # Fine sweep is restricted to directions within this cosine similarity of
    # the coarse winner.
    fine_cos_window: float = 0.92

    # Directions whose undercut score is within this much of the best are
    # considered tied, and are broken by the secondary heuristics.
    tie_tol: float = 2.0e-3

    # Weight of the "mold-size" term when breaking ties (fraction of block
    # volume relative to the best candidate's).
    size_weight: float = 0.15

    # Restrict the search to a fixed axis instead of sweeping (unit vector).
    forced_direction: tuple[float, float, float] | None = None


@dataclass
class PartingConfig:
    """Curved parting surface."""

    # Grid cells along the longest footprint axis. The parting surface is a
    # height field on this grid, so this controls how finely it can follow the
    # silhouette. Resolution affects seam quality, not correctness: the surface
    # is pinned inside the cavity either way.
    grid: int = 400

    # Relative weight of the smoothness term against the "sit in the middle of
    # the cast" term. Higher = flatter, less silhouette-hugging parting face.
    smooth_lambda: float = 1.0

    # Active-set passes used to push the solution back inside the feasible band
    # where smoothing pulled it out.
    active_set_rounds: int = 5

    # Keep the parting surface this far inside the cavity band, so the mating
    # face never becomes tangent to the cavity wall and produce slivers.
    band_inset: float = 0.15

    # Fraction of the way into the chosen material interval to aim for
    # (0.5 = the middle of the cast).
    band_target: float = 0.5


@dataclass
class KeyConfig:
    """Alignment keys (registration dowels on the parting face)."""

    enabled: bool = True
    count: int = 4
    radius: float = 5.0
    height: float = 4.0
    # Draft on the frustum: makes the key self-locating and printable without
    # support, and forgiving of small misalignment.
    draft_deg: float = 20.0
    # Gap between the male key and its socket, per side.
    clearance: float = 0.25
    # Keep keys at least this far from the cavity wall.
    cavity_margin: float = 3.0


@dataclass
class SpoutConfig:
    """Pour spout: a funnel from the outside of the block into the cavity."""

    enabled: bool = True
    # Radius where the funnel meets the outside of the block.
    outer_radius: float = 9.0
    # Radius where it breaks into the cavity.
    inner_radius: float = 4.0


@dataclass
class VentConfig:
    """Vents: thin channels letting trapped air escape as the mold fills."""

    enabled: bool = True
    radius: float = 0.9
    # At most this many vents, placed at the most prominent cavity high points.
    count: int = 6
    # Two vent sites must be at least this far apart.
    min_spacing: float = 12.0


@dataclass
class CoreConfig:
    """The core: the body a *hollow* cast is formed against.

    Off by default. With no core the mold produces a solid positive, which is
    what every stage before this one assumes; switching it on adds a third
    printed body and the geometry that locates it.
    """

    enabled: bool = False

    # The glove wall: the gap left between the cavity and the core, and so the
    # thickness of the cast. Everything about fixation is in service of holding
    # this constant, so it is also the tolerance budget.
    wall: float = 2.5

    # Face budget the erosion runs at. The Minkowski difference that shrinks the
    # part into a core is by far the most expensive step of a core run and
    # scales badly with face count, so the core is built from a decimated copy.
    faces: int | None = 40_000

    # Tessellation of the eroding ball. A polyhedral ball undershoots the wall
    # at its facet centres: 1 costs about 4% of the wall, 2 about 1.5%, at
    # roughly 1.5x the time.
    ball_subdivisions: int = 1

    # Depth of *full-section* core at the cuff, measured down the pour axis.
    # Above this the core fills the cavity completely, so no cast forms there
    # and the glove is open across the whole wrist section. Without it the
    # erosion caps the cuff with a wall-thick membrane.
    cuff_depth: float = 5.0


@dataclass
class CarrierConfig:
    """Option B: a carrier plate at the cuff, registered to the assembled block.

    The plate is a slab trimmed off the cuff end of the block perpendicular to
    the pour axis, and it is printed as one body with the core hanging from it.
    It locates against the *assembled* block -- dowels straddling the parting
    seam, so it references both halves equally instead of inheriting one half's
    key clearance -- and is screwed down, because the load case is a core
    floating up, not a core sinking.
    """

    enabled: bool = True

    # Gap between the top of the cavity and the plate's seating face, so the
    # plate seats on solid mold rather than on the edge of the cuff opening.
    plate_gap: float = 1.0
    plate_thickness: float = 10.0

    # The stem from the core up to the plate. It runs entirely through mold
    # material above the cavity, so it leaves no witness on the cast.
    neck_radius: float = 8.0
    neck_clearance: float = 0.2

    dowel_count: int = 2
    dowel_radius: float = 4.0
    dowel_depth: float = 12.0
    dowel_clearance: float = 0.15
    # At the cuff the cavity's ceiling is a millimetre under the seating face,
    # so a bore's usable depth is whatever the cavity leaves rather than
    # whatever was asked for. Half a diameter still registers in plastic; below
    # that a pin is decoration. A block margin close to the dowel diameter is
    # what usually forces the compromise -- there is then no position both
    # clear of the cavity and inside the plate's edge.
    dowel_min_depth: float = 5.0
    # Two dowels this much closer together than the seam is long are not two
    # dowels; they are one dowel and a wobble.
    dowel_min_spacing: float = 25.0

    screw_count: int = 4
    screw_radius: float = 2.0  # pilot bore into the block
    screw_depth: float = 14.0
    screw_min_depth: float = 6.0
    screw_clearance: float = 0.4  # added for the through-hole in the plate

    # How far a bore's axis may drift off the parting surface over its length
    # before its half-round groove stops being widest at its mouth -- past this
    # the groove is an undercut and the core locks into the halves.
    max_seam_drift: float = 0.4


@dataclass
class CoreTabConfig:
    """Option C: tabs from the core, pinched on the parting surface.

    A tab reaches from the core out past the cast silhouette into mold that is
    solid on both sides of the parting face, where closing the halves pinches
    it. Cut on the parting surface it obeys the same groove rule as an
    alignment key, and its witness mark lands on the seam that gets trimmed
    anyway.
    """

    enabled: bool = True
    count: int = 4
    radius: float = 3.0
    clearance: float = 0.2

    # Mold material required around the tab's outer end, beyond its own radius.
    anchor_margin: float = 2.0

    min_length: float = 2.0
    max_length: float = 45.0
    # Spread along the core, so tabs brace the cantilever instead of clustering
    # at its root where the neck already holds it.
    min_spacing: float = 20.0

    max_seam_drift: float = 0.4


@dataclass
class MoldConfig:
    """Top-level mold configuration."""

    # Minimum wall of mold material between the cavity and the outside.
    block_margin: float = 10.0

    # "box"  -> rectangular block (as specified: the mold is a block)
    # "hull" -> convex hull of the part, dilated by block_margin. Same
    #           guarantees, far less material and print time.
    block_shape: str = "box"

    # Uniform offset applied to the cavity, for cast shrinkage / fit. Positive
    # grows the cavity. 0 disables the (expensive) offset pass entirely.
    cavity_offset: float = 0.0

    # Resolution the cavity is cut at. None = full input resolution.
    cavity_faces: int | None = None

    # The pour axis: which way is "up" when the assembled mold is filled.
    # "auto" uses the part's longest principal axis, oriented so the widest end
    # is up. Otherwise a unit vector.
    pour_axis: str | tuple[float, float, float] = "auto"

    demold: DemoldConfig = field(default_factory=DemoldConfig)
    parting: PartingConfig = field(default_factory=PartingConfig)
    keys: KeyConfig = field(default_factory=KeyConfig)
    spout: SpoutConfig = field(default_factory=SpoutConfig)
    vents: VentConfig = field(default_factory=VentConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    core_tabs: CoreTabConfig = field(default_factory=CoreTabConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoldConfig":
        """Build a config from a (possibly partial) nested dict."""
        data = dict(data or {})
        sub = {
            "demold": DemoldConfig,
            "parting": PartingConfig,
            "keys": KeyConfig,
            "spout": SpoutConfig,
            "vents": VentConfig,
            "core": CoreConfig,
            "carrier": CarrierConfig,
            "core_tabs": CoreTabConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sub.items():
            raw = data.pop(name, None) or {}
            valid = {f.name for f in dataclasses.fields(klass)}
            kwargs[name] = klass(**{k: v for k, v in raw.items() if k in valid})
        valid_top = {f.name for f in dataclasses.fields(cls)} - set(sub)
        for k, v in data.items():
            if k in valid_top:
                kwargs[k] = v
        cfg = cls(**kwargs)
        if isinstance(cfg.pour_axis, list):
            cfg.pour_axis = tuple(cfg.pour_axis)
        if isinstance(cfg.demold.forced_direction, list):
            cfg.demold.forced_direction = tuple(cfg.demold.forced_direction)
        return cfg
