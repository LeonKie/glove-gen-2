"""Undercut analysis: the column-crossing model and the direction search."""

from __future__ import annotations

import numpy as np
import pytest

from glovegen import demold
from glovegen.frame import Frame, fibonacci_hemisphere

from .conftest import unit


class TestColumnCast:
    def test_ray_grid_volume_matches_mesh_volume(self, sphere):
        """The crossing model is only right if its volume integral is right.

        Summing the material intervals times the cell area is an independent
        estimate of the solid's volume, so agreeing with ``mesh.volume`` checks
        the entire interval interpretation at once.
        """
        score = demold.score_direction(sphere, [0, 0, 1], 200)
        assert score.part_volume == pytest.approx(sphere.volume, rel=2e-3)

    def test_volume_integral_holds_for_multi_slab_columns(self, dumbbell):
        score = demold.score_direction(dumbbell, [0, 0, 1], 200)
        assert score.part_volume == pytest.approx(dumbbell.volume, rel=5e-3)

    def test_sphere_has_no_undercut_from_any_direction(self, sphere):
        for d in ([0, 0, 1], [1, 0, 0], [0, 1, 0], unit([1, 1, 1]), unit([-2, 1, 3])):
            score = demold.score_direction(sphere, d, 96)
            assert score.trapped_volume == 0.0
            assert score.multi_fraction == 0.0

    def test_dumbbell_axis_is_the_worst_direction(self, dumbbell):
        along = demold.score_direction(dumbbell, [0, 0, 1], 128)
        across = demold.score_direction(dumbbell, [1, 0, 0], 128)
        assert along.undercut_fraction > 0.5
        assert across.trapped_volume == 0.0

    def test_direction_and_its_negation_are_equivalent(self, dumbbell):
        a = demold.score_direction(dumbbell, [0.3, 0.2, 1], 96)
        b = demold.score_direction(dumbbell, [-0.3, -0.2, -1], 96)
        assert a.trapped_volume == pytest.approx(b.trapped_volume, rel=1e-6, abs=1e-9)

    def test_cast_grid_indexes_columns_in_c_order(self, sphere):
        frame = Frame.from_direction([0, 0, 1])
        xs = np.linspace(-25, 25, 7)
        ys = np.linspace(-25, 25, 5)
        cast = demold.cast_grid(sphere, frame, xs, ys)
        assert cast.nx == 7 and cast.ny == 5
        assert cast.counts.shape == (35,)
        # Corner columns miss a 20mm sphere; the centre column must hit it twice.
        centre = 3 * 5 + 2
        assert cast.counts[centre] == 2
        assert cast.counts[0] == 0

    def test_thickest_interval_picks_the_fat_slab(self, dumbbell):
        """Pulled along the axis, the fattest slab in an off-centre column is a ball."""
        frame = Frame.from_direction([0, 0, 1])
        cast = demold.cast_grid(dumbbell, frame, np.array([6.0]), np.array([0.0]))
        cols, lo, hi = cast.thickest_material_interval()
        assert len(cols) == 1
        # That column passes through both balls but not the thin rod, so the
        # chosen slab must be one ball's chord, not the full span.
        assert (hi - lo)[0] < 20.0
        assert (hi - lo)[0] > 5.0

    def test_empty_when_grid_misses_the_mesh(self, sphere):
        frame = Frame.from_direction([0, 0, 1])
        cast = demold.cast_grid(sphere, frame, np.array([500.0]), np.array([500.0]))
        assert cast.counts.sum() == 0
        score = demold.score_columns(cast)
        assert score.part_volume == 0.0
        assert score.undercut_fraction == 0.0


class TestDirectionSearch:
    def test_finds_a_perpendicular_pull_for_the_dumbbell(self, dumbbell, fast_config):
        d, best, ranked = demold.suggest_direction(dumbbell, fast_config.demold)
        assert abs(float(d @ np.array([0.0, 0.0, 1.0]))) < 0.2
        assert best.undercut_fraction < 1e-3
        assert len(ranked) > 1

    def test_forced_direction_short_circuits_the_search(self, dumbbell, fast_config):
        fast_config.demold.forced_direction = (0.0, 0.0, 1.0)
        d, best, ranked = demold.suggest_direction(dumbbell, fast_config.demold)
        assert d == pytest.approx([0, 0, 1])
        assert len(ranked) == 1
        assert best.undercut_fraction > 0.5

    def test_reports_progress(self, dumbbell, fast_config):
        seen = []
        demold.suggest_direction(
            dumbbell, fast_config.demold, progress=lambda f, m: seen.append(f)
        )
        assert seen and 0.0 <= min(seen) and max(seen) <= 1.0
        assert seen == sorted(seen)

    def test_ties_break_towards_a_simpler_parting_surface(self):
        """With undercut tied, prefer fewer multi-slab columns, then less material."""
        mk = lambda uc, multi, block: demold.DirectionScore(  # noqa: E731
            direction=np.array([0, 0, 1.0]),
            trapped_volume=0.0,
            part_volume=1.0,
            undercut_fraction=uc,
            multi_fraction=multi,
            block_volume=block,
        )
        scores = [mk(0.0, 0.9, 100), mk(0.0, 0.1, 100), mk(0.0, 0.1, 500)]
        order = demold._rank(scores, tie_tol=2e-3, size_weight=0.15)
        assert order[0] == 1  # tied undercut, simplest parting, smallest block


class TestHemisphere:
    def test_covers_a_hemisphere_with_unit_vectors(self):
        pts = fibonacci_hemisphere(200)
        assert len(pts) == 200
        assert np.allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-9)
        assert (pts[:, 2] >= 0).all()

    def test_no_duplicate_axes(self):
        pts = fibonacci_hemisphere(64)
        gram = np.abs(pts @ pts.T)
        np.fill_diagonal(gram, 0.0)
        assert gram.max() < 0.9999


class TestFaceHeatmap:
    def test_sphere_is_entirely_releasable(self, sphere):
        info = demold.face_undercut(sphere, [0, 0, 1])
        assert info["undercut_face_count"] == 0
        assert info["undercut_area_fraction"] == 0.0
        assert info["severity"].shape == (len(sphere.faces),)

    def test_dumbbell_axis_traps_faces(self, dumbbell):
        along = demold.face_undercut(dumbbell, [0, 0, 1])
        across = demold.face_undercut(dumbbell, [1, 0, 0])
        assert along["undercut_area_fraction"] > 0.2
        assert across["undercut_area_fraction"] == 0.0

    def test_severity_is_zero_exactly_where_not_trapped(self, dumbbell):
        info = demold.face_undercut(dumbbell, [0, 0, 1])
        sev, stuck = info["severity"], info["undercut"]
        assert np.all(sev[~stuck] == 0.0)
        assert np.all(sev[stuck] > 0.0)
        assert sev.max() <= 1.0
