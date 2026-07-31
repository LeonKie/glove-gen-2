"""Shared fixtures: small synthetic solids with answers we know analytically."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from glovegen.config import MoldConfig


def _union(*meshes) -> trimesh.Trimesh:
    out = trimesh.boolean.union(list(meshes), engine="manifold", check_volume=False)
    out.merge_vertices()
    return out


@pytest.fixture(scope="session")
def sphere() -> trimesh.Trimesh:
    """Demoldable from every direction: no column ever has more than 2 crossings."""
    return trimesh.creation.icosphere(subdivisions=3, radius=20.0)


@pytest.fixture(scope="session")
def dumbbell() -> trimesh.Trimesh:
    """Two balls on a thin rod along Z.

    Pulling along Z is the worst case: columns that miss the rod cross ball,
    gap, ball, so the mold has a trapped fin between them. Pulling across Z is
    perfect. A direction search must find that.
    """
    a = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    a.apply_translation([0, 0, 20])
    b = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    b.apply_translation([0, 0, -20])
    rod = trimesh.creation.cylinder(radius=2.0, height=40.0, sections=32)
    return _union(a, b, rod)


@pytest.fixture(scope="session")
def two_prong() -> trimesh.Trimesh:
    """A bar with two upright prongs of different heights.

    With the pour axis along +Z the taller prong takes the spout, so the shorter
    prong's tip is a genuine trapped-air pocket that vent detection must find.
    """
    base = trimesh.creation.box(extents=[70, 14, 12])
    tall = trimesh.creation.cylinder(radius=5, height=40, sections=24)
    tall.apply_translation([-22, 0, 20])
    short = trimesh.creation.cylinder(radius=5, height=30, sections=24)
    short.apply_translation([22, 0, 15])
    return _union(base, tall, short)


@pytest.fixture(scope="session")
def taper() -> trimesh.Trimesh:
    """An asymmetric capsule: fat end at +Z, so the pour axis must point +Z."""
    a = trimesh.creation.icosphere(subdivisions=3, radius=14.0)
    a.apply_translation([0, 0, 22])
    b = trimesh.creation.icosphere(subdivisions=3, radius=7.0)
    b.apply_translation([0, 0, -22])
    rod = trimesh.creation.cylinder(radius=5.0, height=48.0, sections=32)
    return _union(a, b, rod)


@pytest.fixture
def fast_config() -> MoldConfig:
    """Coarse settings so the whole suite stays quick."""
    cfg = MoldConfig(block_margin=8.0)
    cfg.parting.grid = 90
    cfg.demold.coarse_dirs = 60
    cfg.demold.coarse_grid = 32
    cfg.demold.fine_dirs = 80
    cfg.demold.fine_grid = 48
    cfg.demold.proxy_faces = 20_000
    return cfg


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)
