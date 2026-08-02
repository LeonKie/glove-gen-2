"""The hollow-cast core, and the two things that stop it floating."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from glovegen import core, features, mold, pipeline
from glovegen.config import MoldConfig
from glovegen.frame import Frame, unit


@pytest.fixture
def core_config() -> MoldConfig:
    """Coarse enough to run in the suite, still a real core."""
    # The default block margin, not the tighter one the rest of the suite uses:
    # a margin close to the dowel diameter leaves no position both clear of the
    # cavity and inside the plate's edge, which is a real constraint but not the
    # one these tests are about.
    cfg = MoldConfig(block_margin=10.0)
    cfg.parting.grid = 70
    cfg.core.enabled = True
    cfg.core.wall = 2.0
    cfg.core.faces = 6_000
    cfg.core.cuff_depth = 4.0
    cfg.core_tabs.count = 3
    cfg.core_tabs.min_spacing = 10.0
    return cfg


@pytest.fixture(scope="module")
def built(request):
    """One core run shared by the assertions that only read it."""
    taper = request.getfixturevalue("taper")
    cfg = MoldConfig(block_margin=10.0)
    cfg.parting.grid = 70
    cfg.core.enabled = True
    cfg.core.wall = 2.0
    cfg.core.faces = 6_000
    cfg.core.cuff_depth = 4.0
    cfg.core_tabs.count = 3
    cfg.core_tabs.min_spacing = 10.0
    return pipeline.run(taper, cfg, direction=[1, 0, 0], verify=True)


class TestErode:
    def test_shrinks_by_the_wall_everywhere(self, sphere):
        small = core.erode(sphere, 3.0, subdivisions=2)
        radius = np.linalg.norm(small.vertices - sphere.centroid, axis=1)
        # A polyhedral ball undershoots at its facet centres, never overshoots.
        assert radius.max() == pytest.approx(17.0, abs=0.05)
        assert radius.min() > 16.7

    def test_a_wall_thicker_than_the_part_is_an_error(self, sphere):
        with pytest.raises(ValueError, match="thinner than twice the wall"):
            core.erode(sphere, 25.0)

    def test_rejects_a_zero_wall(self, sphere):
        with pytest.raises(ValueError, match="positive wall"):
            core.erode(sphere, 0.0)


class TestCuff:
    def test_the_glove_is_open_across_the_wrist(self, taper, core_config):
        """No cast forms above the cuff plane, so the glove has a rim not a cap.

        Eroding alone leaves a wall-thick membrane over the wrist; the union
        with the full-section plug is what opens it, and this is the assertion
        that would catch losing that union.
        """
        p = unit(features.choose_pour_axis(taper, core_config))
        body = core.build_core_body(taper, p, core_config)
        cut = core.cuff_offset(taper, p, core_config)

        cast = trimesh.boolean.difference(
            [taper, body], engine="manifold", check_volume=False
        )
        # Volume above the rim, not the topmost vertex: the plug shares the
        # part's outer surface, so that coincident boolean leaves zero-thickness
        # sliver geometry reaching to the top of the cap either way.
        frame = Frame.from_direction(p)
        local = frame.to_local(taper.vertices)
        above = trimesh.boolean.intersection(
            [cast, core._slab(frame, local, cut + 0.5, float(local[:, 2].max()) + 20.0)],
            engine="manifold",
            check_volume=False,
        )
        left = abs(float(above.volume)) if len(above.faces) else 0.0
        assert left < 0.005 * cast.volume, f"{left:.2f} mm3 of cast above the rim"

    def test_without_a_cuff_the_core_is_a_closed_bladder(self, taper, core_config):
        core_config.core.cuff_depth = 0.0
        p = unit(features.choose_pour_axis(taper, core_config))
        body = core.build_core_body(taper, p, core_config)
        cut = core.cuff_offset(taper, p, core_config)
        # The core's own top is a wall below the part's, so cast covers it.
        assert (np.asarray(body.vertices) @ p).max() < cut - 1.0


class TestSeamDrift:
    def test_a_bore_along_a_flat_seam_does_not_drift(self, taper, core_config):
        built = mold.build_mold(taper, [1, 0, 0], core_config)
        surface = built.surface
        start = surface.frame.to_world(
            np.array([surface.xs[len(surface.xs) // 2], surface.ys[len(surface.ys) // 2], 0.0])
        )
        # Along the pull direction the parting surface is a level set, so the
        # only thing that moves is the sample point's own z.
        along_seam = surface.frame.to_world([1.0, 0.0, 0.0])
        drift = core.seam_drift(surface, start, along_seam, 5.0)
        assert drift < 2.0

    def test_a_bore_across_the_seam_drifts(self, taper, core_config):
        built = mold.build_mold(taper, [1, 0, 0], core_config)
        surface = built.surface
        start = surface.frame.to_world(np.array([0.0, 0.0, float(surface.h.mean())]))
        across = surface.frame.direction  # straight out of the parting plane
        assert core.seam_drift(surface, start, across, 10.0) > 5.0


class TestRingInset:
    def test_points_land_inside_at_the_asked_for_inset(self):
        square = np.array([[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]])
        pts = core._ring_inset(square, 5.0, 32)
        assert len(pts) > 0
        assert np.abs(pts).max() <= 15.0 + 1e-6

    def test_mid_edge_points_survive(self):
        """Regression: an exact half-plane test throws away most of the ring.

        A point taken from the middle of an edge and pushed in by the inset sits
        at exactly the inset, so whether it passes is decided by float error --
        which left four screws bunched on one side of a square plate.
        """
        square = np.array([[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]])
        pts = core._ring_inset(square, 5.0, 32)
        # Corners are legitimately dropped; the four edge middles are not.
        assert len(pts) >= 20

    def test_an_inset_wider_than_the_polygon_yields_nothing(self):
        square = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
        assert len(core._ring_inset(square, 10.0, 16)) == 0


class TestCoreAtParting:
    def test_the_parting_surface_runs_through_the_thick_part_of_the_core(
        self, taper, core_config
    ):
        built = mold.build_mold(taper, [1, 0, 0], core_config)
        p = unit(features.choose_pour_axis(taper, core_config))
        body = core.build_core_body(taper, p, core_config)
        mask = core.core_at_parting(body, built.surface)
        # It has to hit *something*, and never more than the columns that pass
        # through the part it was eroded from.
        assert mask.any()
        assert mask.sum() <= built.surface.constrained.sum()


class TestFixated:
    def test_the_core_is_one_printable_solid(self, built):
        report = built.report()["core"]
        assert built.core is not None
        assert report["assembly"]["pieces"] == 1
        assert built.core.volume > 0

    def test_nothing_was_skipped(self, built):
        assert built.report()["core"]["skipped"] == []

    def test_the_wall_matches_what_was_asked_for(self, built):
        wall = built.report()["core"]["wall"]
        # The eroding ball undershoots at its facet centres and never overshoots.
        assert wall["max_mm"] <= wall["target_mm"] + 1e-3
        assert wall["min_mm"] >= wall["target_mm"] * 0.9
        assert wall["under_90pct_fraction"] == 0.0

    def test_the_cuff_rim_is_not_measured_as_a_thin_wall(self, built):
        """Regression: the shut-off disc meets the cavity at its edge.

        Sampling right up to the rim reported a 0.8 mm wall against a 2.5 mm
        target on a hand-shaped part -- not a thin spot in the glove, but the
        hole the hand goes through. The exclusion has to sit a wall below the
        rim, not at it.
        """
        wall = built.report()["core"]["wall"]
        assert wall["min_mm"] > 0.5 * wall["target_mm"]

    def test_the_core_releases_from_both_halves(self, built):
        """The whole point of centring every core feature on the parting surface.

        A number here means a tab, dowel or neck drifted off the seam far enough
        that its groove became a lock and the mold cannot be opened.
        """
        assert built.report()["core"]["release"]["releases"]

    def test_the_halves_still_separate(self, built):
        assert built.separation["opens"]

    def test_the_plate_is_registered_and_clamped(self, built):
        report = built.report()["core"]
        assert len(report["dowels"]) >= 1
        assert all(d["seam_drift_mm"] <= 0.4 for d in report["dowels"])
        # Screws in only one half hold that half down and leave the other loose.
        sides = {s["side"] for s in report["screws"]}
        assert sides == {"a", "b"}

    def test_the_neck_leaves_no_witness_on_the_cast(self, built):
        """It runs from the cuff plug up through mold material, never through cast."""
        report = built.report()["core"]
        assert report["neck"]["seam_drift_mm"] <= 0.4
        assert report["neck"]["length_mm"] > 0

    def test_tabs_were_pinched(self, built):
        tabs = built.report()["core"]["tabs"]
        assert sum(1 for t in tabs if t["status"] == "applied") >= 2

    def test_the_cost_of_the_tabs_is_reported_not_assumed(self, built):
        """Tabs cross the glove wall, so the cast has to stretch off them."""
        assert built.report()["core"]["tab_through_wall_mm3"] > 0

    def test_the_core_is_written_alongside_the_halves(self, built, tmp_path):
        written = built.write(tmp_path, extras=False)
        assert (tmp_path / "core.stl").exists()
        assert written["core"].endswith("core.stl")


class TestGating:
    def test_off_by_default(self, taper, fast_config):
        built = pipeline.run(taper, fast_config, direction=[1, 0, 0], verify=False)
        assert built.core is None
        assert "core" not in built.report()

    def test_tabs_only(self, taper, core_config):
        core_config.carrier.enabled = False
        built = pipeline.run(taper, core_config, direction=[1, 0, 0], verify=False)
        report = built.report()["core"]
        assert "plate" not in report
        assert any(t["status"] == "applied" for t in report["tabs"])

    def test_carrier_only(self, taper, core_config):
        core_config.core_tabs.enabled = False
        built = pipeline.run(taper, core_config, direction=[1, 0, 0], verify=False)
        report = built.report()["core"]
        assert report["tabs"] == []
        assert report["plate"]["thickness_mm"] > 0

    def test_a_tab_that_would_lock_the_mold_is_skipped_with_a_reason(
        self, taper, core_config
    ):
        """The groove rule is enforced, not hoped for."""
        core_config.core_tabs.max_seam_drift = 0.0
        core_config.carrier.enabled = False
        built = pipeline.run(taper, core_config, direction=[1, 0, 0], verify=False)
        tabs = built.report()["core"]["tabs"]
        skipped = [t for t in tabs if t["status"] == "skipped"]
        assert skipped, "a zero drift budget should reject every tab"
        assert "undercut" in skipped[0]["reason"]


class TestKeysMakeRoomForThePlate:
    def test_no_key_is_planned_above_the_seating_face(self, taper, core_config):
        """A key up there would be sliced in two when the plate is trimmed off."""
        built = mold.build_mold(taper, [1, 0, 0], core_config)
        ctx = features.FeatureContext.from_mold(built)
        plan = features.plan_features(ctx, core_config)
        p = unit(plan.pour_axis)
        seat, _ = core.plate_planes(built.cavity, p, core_config)
        for item in plan.of_kind("key"):
            assert np.asarray(item.position) @ p <= seat


class TestConfig:
    def test_a_partial_dict_reaches_the_core_sections(self):
        cfg = MoldConfig.from_dict(
            {
                "core": {"enabled": True, "wall": 1.8},
                "carrier": {"dowel_count": 3},
                "core_tabs": {"count": 6},
            }
        )
        assert cfg.core.enabled and cfg.core.wall == 1.8
        assert cfg.carrier.dowel_count == 3
        assert cfg.core_tabs.count == 6
        # Untouched sections keep their defaults.
        assert cfg.keys.radius == 5.0

    def test_round_trips(self):
        cfg = MoldConfig()
        cfg.core.enabled = True
        cfg.core.wall = 3.25
        assert MoldConfig.from_dict(cfg.to_dict()).core.wall == 3.25


class TestPourAxisSeededFrame:
    def test_a_core_run_squares_the_block_to_the_pour_axis(self, taper, core_config):
        """Otherwise the plate is a corner wedge sliced off an oblique box."""
        built = mold.build_mold(taper, [1, 0, 0], core_config)
        p = unit(features.choose_pour_axis(taper, core_config))
        # Local +X is the pour axis, so a box block has two faces square to it.
        assert abs(abs(built.frame.rot[0] @ p) - 1.0) < 1e-6

    def test_a_plain_run_leaves_the_roll_alone(self, taper, fast_config):
        plain = mold.build_mold(taper, [1, 0, 0], fast_config)
        assert plain.frame.rot[2] == pytest.approx(np.array([1.0, 0.0, 0.0]))


class TestFrameSeed:
    def test_the_seed_sets_the_roll(self):
        frame = Frame.from_direction([0, 0, 1], seed=[1, 1, 0])
        assert frame.rot[0] == pytest.approx(unit([1, 1, 0]))
        assert frame.rot[2] == pytest.approx(np.array([0.0, 0.0, 1.0]))

    def test_a_seed_along_the_direction_says_nothing_and_is_ignored(self):
        seeded = Frame.from_direction([0, 0, 1], seed=[0, 0, 5])
        plain = Frame.from_direction([0, 0, 1])
        assert seeded.rot == pytest.approx(plain.rot)

    def test_the_seed_is_orthonormalised_not_trusted(self):
        frame = Frame.from_direction([0, 0, 1], seed=[3, 0, 9])
        assert frame.rot @ frame.rot.T == pytest.approx(np.eye(3))
