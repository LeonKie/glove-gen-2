"""Block construction, the boolean split, and the properties that must hold."""

from __future__ import annotations

import numpy as np
import pytest

from glovegen import mold, validate
from glovegen.frame import Frame


class TestBlock:
    def test_box_block_clears_the_part_by_the_margin(self, sphere):
        frame = Frame.from_direction([0, 0, 1])
        block, bounds = mold.build_block(sphere, frame, 8.0, "box")
        local = frame.to_local(sphere.vertices)
        assert np.all(bounds[0] <= local.min(axis=0) - 8.0 + 1e-9)
        assert np.all(bounds[1] >= local.max(axis=0) + 8.0 - 1e-9)
        assert block.is_watertight

    def test_box_block_volume_is_the_bounding_box_plus_margin(self, sphere):
        frame = Frame.from_direction([0, 0, 1])
        block, bounds = mold.build_block(sphere, frame, 8.0, "box")
        assert block.volume == pytest.approx(float(np.prod(bounds[1] - bounds[0])), rel=1e-9)

    def test_hull_block_uses_much_less_material(self, dumbbell):
        frame = Frame.from_direction([1, 0, 0])
        box, _ = mold.build_block(dumbbell, frame, 8.0, "box")
        hull, _ = mold.build_block(dumbbell, frame, 8.0, "hull")
        assert hull.volume < box.volume
        assert hull.is_watertight

    def test_hull_block_still_contains_the_part(self, dumbbell):
        frame = Frame.from_direction([1, 0, 0])
        hull, _ = mold.build_block(dumbbell, frame, 6.0, "hull")
        assert hull.contains(dumbbell.vertices).all()

    def test_unknown_shape_is_rejected(self, sphere):
        frame = Frame.from_direction([0, 0, 1])
        with pytest.raises(ValueError, match="unknown block_shape"):
            mold.build_block(sphere, frame, 8.0, "sphere")


class TestSplit:
    @pytest.fixture(scope="class")
    def built(self, dumbbell):
        from glovegen.config import MoldConfig

        cfg = MoldConfig(block_margin=8.0)
        cfg.parting.grid = 110
        return mold.build_mold(dumbbell, [1, 0, 0], cfg)

    def test_halves_account_for_all_the_mold(self, built):
        """Nothing lost, nothing double-counted by the split."""
        assert built.half_a.volume + built.half_b.volume == pytest.approx(
            built.mold.volume, rel=1e-9
        )

    def test_mold_is_the_block_minus_the_cavity(self, built, dumbbell):
        assert built.mold.volume == pytest.approx(
            built.block.volume - dumbbell.volume, rel=1e-6
        )

    def test_halves_are_closed_solids(self, built):
        for half in (built.half_a, built.half_b):
            rep = validate.solid_report(half)
            assert rep["boundary_edges"] == 0
            assert rep["volume"] > 0

    def test_halves_are_single_components(self, built):
        for half in (built.half_a, built.half_b):
            assert len(half.split(only_watertight=False)) == 1

    def test_half_a_sits_above_the_parting_surface(self, built):
        """Structural check on the invariant that makes the mold openable."""
        frame = built.frame
        surface = built.surface
        local = frame.to_local(built.half_a.vertices)
        i = np.clip(
            np.searchsorted(surface.xs, local[:, 0]), 0, len(surface.xs) - 1
        )
        j = np.clip(
            np.searchsorted(surface.ys, local[:, 1]), 0, len(surface.ys) - 1
        )
        # Allow a cell's worth of slack: vertices land between grid nodes.
        cell = float(surface.xs[1] - surface.xs[0])
        slack = 3.0 * cell + 1e-6
        assert np.all(local[:, 2] >= surface.h[i, j] - slack)

    def test_mold_opens(self, built, dumbbell):
        rep = validate.separation_report(
            built.half_a, built.half_b, dumbbell, built.direction
        )
        assert rep["opens"]
        assert rep["rigid_cast_demolds"]

    def test_a_bad_direction_reports_cast_interference(self, dumbbell):
        from glovegen.config import MoldConfig

        cfg = MoldConfig(block_margin=8.0)
        cfg.parting.grid = 110
        bad = mold.build_mold(dumbbell, [0, 0, 1], cfg)
        rep = validate.separation_report(
            bad.half_a, bad.half_b, dumbbell, bad.direction
        )
        # The halves still come apart -- the parting surface guarantees that --
        # but the cast is trapped, which is the whole point of the metric.
        assert rep["opens"]
        assert not rep["rigid_cast_demolds"]
        assert rep["cast_interference_mm3"] > 100.0

    def test_stats_are_reported(self, built):
        s = built.stats
        assert s["split_volume_error_cm3"] == pytest.approx(0.0, abs=1e-4)
        assert s["block_volume_cm3"] > s["mold_volume_cm3"]
        assert "parting" in s


class TestCavityOffset:
    def test_zero_offset_is_a_no_op(self, sphere):
        assert mold.offset_cavity(sphere, 0.0) is sphere

    def test_negative_offset_is_refused(self, sphere):
        with pytest.raises(NotImplementedError):
            mold.offset_cavity(sphere, -1.0)

    def test_positive_offset_grows_the_cavity(self, sphere):
        grown = mold.offset_cavity(sphere, 2.0)
        assert grown.volume > sphere.volume
        # A 20mm sphere grown by 2mm should approach a 22mm sphere.
        assert grown.volume == pytest.approx(4 / 3 * np.pi * 22**3, rel=0.05)
