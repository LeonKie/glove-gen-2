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
