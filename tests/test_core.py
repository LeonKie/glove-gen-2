"""The hollow-cast core: the erosion, the plane cut, and what holds it still."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from glovegen import core, features, mold, pipeline
from glovegen.config import MoldConfig
from glovegen.frame import Frame, unit


def _core_config(**over) -> MoldConfig:
    # The default block margin, not the tighter one the rest of the suite uses:
    # a margin close to the dowel diameter leaves no position both clear of the
    # cavity and inside the plate's edge, which is a real constraint but not the
    # one these tests are about.
    cfg = MoldConfig(block_margin=10.0)
    cfg.parting.grid = 70
    cfg.core.enabled = True
    cfg.core.wall = 2.0
    cfg.core.faces = 6_000
    cfg.core_tabs.count = 3
    cfg.core_tabs.min_spacing = 10.0
    cfg.core_dowels.count = 2
    cfg.core_dowels.min_spacing = 10.0
    for key, value in over.items():
        section, _, name = key.partition("__")
        setattr(getattr(cfg, section), name, value)
    return cfg


@pytest.fixture
def core_config() -> MoldConfig:
    return _core_config()


@pytest.fixture
def full_config() -> MoldConfig:
    """Core, plate, dowels and tabs: everything switched on."""
    return _core_config(
        carrier__enabled=True, core_tabs__enabled=True, core_dowels__enabled=True
    )


@pytest.fixture(scope="module")
def built(request):
    """One full core run shared by the assertions that only read it."""
    taper = request.getfixturevalue("taper")
    cfg = _core_config(
        carrier__enabled=True, core_tabs__enabled=True, core_dowels__enabled=True
    )
    return pipeline.run(taper, cfg, direction=[1, 0, 0], verify=True)


def _ctx(part, cfg):
    result = mold.build_mold(part, [1, 0, 0], cfg)
    body = core.build_core_body(part, cfg)
    return result, features.FeatureContext.from_mold(result, core=body)


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


class TestDefaults:
    def test_a_wall_alone_builds_the_core_and_nothing_else(self, taper, core_config):
        """Asking for a hollow cast should not also rearrange the mold."""
        built = pipeline.run(taper, core_config, direction=[1, 0, 0], verify=False)
        kinds = {i.kind for i in built.feature_plan.items}
        assert built.core is not None
        assert kinds & {"plate", "dowel", "screw", "port", "core_tab"} == set()
        # ...and the ordinary spout is still there, because nothing sealed it off.
        assert "spout" in kinds

    def test_the_plate_and_the_tabs_are_off_in_the_config(self):
        cfg = MoldConfig()
        assert not cfg.carrier.enabled
        assert not cfg.core_tabs.enabled


class TestThePlaneCut:
    def test_it_takes_the_same_bite_out_of_all_three_bodies(self, taper, full_config):
        """The cut is not a trim of the block; it removes the core too."""
        built = pipeline.run(taper, full_config, direction=[1, 0, 0], verify=False)
        p = unit(built.feature_plan.pour_axis)
        plane = built.core_report["plate"]["plane_offset_mm"]

        for name, body in (("half A", built.half_a), ("half B", built.half_b)):
            top = float((np.asarray(body.vertices) @ p).max())
            assert top == pytest.approx(plane, abs=0.05), name
        # The core is the exception, and deliberately: its stub runs a merge
        # depth past the plane so the union with the plate is not coplanar.
        assert float((np.asarray(built.core.vertices) @ p).max()) > plane

    def test_the_plate_caps_everything_the_cut_left(self, built):
        plate = built.report()["core"]["plate"]
        assert plate["thickness_mm"] > 0
        assert plate["volume_cm3"] > 0
        assert plate["discarded_cm3"] > 0

    def test_the_core_and_the_plate_are_one_printable_body(self, built):
        assert core.real_pieces(built.core) == 1

    def test_moving_the_plane_moves_the_cut(self, taper, full_config):
        built = pipeline.run(taper, full_config, direction=[1, 0, 0], verify=False)
        plan = built.feature_plan.as_dict()
        p = unit(plan["pour_axis"])
        before = built.core_report["plate"]["plane_offset_mm"]

        for item in plan["items"]:
            if item["kind"] == "plate":
                item["position"] = list(np.asarray(item["position"]) - p * 6.0)
        moved = pipeline.run(
            taper, full_config, direction=[1, 0, 0], plan=plan, verify=False
        )
        after = moved.core_report["plate"]["plane_offset_mm"]
        assert after == pytest.approx(before - 6.0, abs=0.05)
        assert moved.half_a.volume < built.half_a.volume

    def test_a_plane_outside_the_block_is_skipped_with_a_reason(self, taper, full_config):
        built = pipeline.run(taper, full_config, direction=[1, 0, 0], verify=False)
        plan = built.feature_plan.as_dict()
        p = unit(plan["pour_axis"])
        for item in plan["items"]:
            if item["kind"] == "plate":
                item["position"] = list(np.asarray(item["position"]) + p * 500.0)
        out = pipeline.run(taper, full_config, direction=[1, 0, 0], plan=plan, verify=False)
        entry = next(i for i in out.feature_report.items if i["kind"] == "plate")
        assert entry["status"] == "skipped"
        assert "outside the block" in entry["reason"]

    def test_a_second_plate_is_refused(self, taper, full_config):
        """"+ add" makes two easy, and the second would cut away the first."""
        built = pipeline.run(taper, full_config, direction=[1, 0, 0], verify=False)
        plan = built.feature_plan.as_dict()
        p = unit(plan["pour_axis"])
        original = next(i for i in plan["items"] if i["kind"] == "plate")
        plan["items"].append(
            {
                **original,
                "id": "plate-2",
                "position": list(np.asarray(original["position"]) - p * 8.0),
            }
        )
        out = pipeline.run(taper, full_config, direction=[1, 0, 0], plan=plan, verify=False)
        entries = [i for i in out.feature_report.items if i["kind"] == "plate"]
        assert [e["status"] for e in entries] == ["applied", "skipped"]
        assert "already cut" in entries[1]["reason"]

    def test_aiming_the_pour_aims_the_cut(self, taper, full_config):
        """They are one thing: the port through the plate is the way in.

        A plate square to something other than the way the mold fills would be
        a plate you pour into at an angle, so the plan carries one axis and the
        cut follows it.
        """
        built = pipeline.run(taper, full_config, direction=[1, 0, 0], verify=False)
        plan = built.feature_plan.as_dict()
        before = built.core_report["plate"]

        tilt = np.radians(20.0)
        base = unit(plan["pour_axis"])
        across = unit(np.cross(base, [1.0, 0.0, 0.0]))
        plan["pour_axis"] = [
            float(v) for v in unit(base * np.cos(tilt) + across * np.sin(tilt))
        ]
        out = pipeline.run(taper, full_config, direction=[1, 0, 0], plan=plan, verify=False)

        # The plate is square to the aimed axis, not the one it was built with.
        p = unit(out.feature_plan.pour_axis)
        assert abs(p @ base - np.cos(tilt)) < 1e-6
        reach = np.asarray(out.core.vertices) @ p
        assert float(reach.max()) == pytest.approx(
            out.core_report["plate"]["plane_offset_mm"]
            + out.core_report["plate"]["thickness_mm"],
            abs=0.3,
        )
        assert out.core_report["plate"]["volume_cm3"] > 0
        assert out.core_report["plate"]["plane_offset_mm"] != before["plane_offset_mm"]

    def test_the_things_that_stand_on_the_plate_go_with_it(self, taper, full_config):
        """Screws and the port need the plate; a dowel does not, and stays."""
        built = pipeline.run(taper, full_config, direction=[1, 0, 0], verify=False)
        plan = built.feature_plan.as_dict()
        for item in plan["items"]:
            if item["kind"] == "plate":
                item["enabled"] = False
        out = pipeline.run(taper, full_config, direction=[1, 0, 0], plan=plan, verify=False)
        standing = [i for i in out.feature_report.items if i["kind"] in ("screw", "port")]
        assert standing, "the plan should still carry them"
        for entry in standing:
            assert entry["status"] == "skipped"
            assert "carrier plate" in entry["reason"]
        # A dowel does not stand on the plate, so it is never skipped for the
        # want of one. It may still be skipped on its own merits: without the
        # cut the core is longer, and a run can end up gripping a thinner part
        # of it than it did when the plan was made.
        dowels = [i for i in out.feature_report.items if i["kind"] == "dowel"]
        assert dowels
        assert any(e["status"] == "applied" for e in dowels)
        assert not any("carrier plate" in e.get("reason", "") for e in dowels)


class TestThePourPath:
    def test_a_plate_replaces_the_spout_with_a_port(self, taper, full_config):
        """The plate seals the annulus, so the only way in is through it.

        A spout aimed at the cavity's high point would be cut into material the
        plane is about to discard.
        """
        built = pipeline.run(taper, full_config, direction=[1, 0, 0], verify=False)
        kinds = [i.kind for i in built.feature_plan.items]
        assert "spout" not in kinds
        assert "port" in kinds

    def test_the_port_goes_through_the_plate(self, built):
        entry = next(
            i for i in built.report()["features"]["items"] if i["kind"] == "port"
        )
        assert entry["status"] == "applied", entry.get("reason")
        assert entry["detail"]["length_mm"] > 0


class TestDowels:
    """A tab turned inside out: bored through all three bodies, with a loose pin."""

    def test_the_bore_goes_through_both_halves_and_the_core(self, taper, full_config):
        plain = _core_config()  # a core with nothing holding it
        alone = pipeline.run(taper, plain, direction=[1, 0, 0], verify=False)

        cfg = _core_config(core_dowels__enabled=True)
        bored = pipeline.run(taper, cfg, direction=[1, 0, 0], verify=False)
        assert bored.report()["core"]["dowels"]

        # Every body loses material; none of them gains any.
        assert bored.half_a.volume < alone.half_a.volume
        assert bored.half_b.volume < alone.half_b.volume
        assert bored.core.volume < alone.core.volume

    def test_the_pin_is_a_part_you_can_print(self, taper, full_config):
        built = pipeline.run(taper, full_config, direction=[1, 0, 0], verify=False)
        assert built.pins is not None
        assert built.pins.volume > 0

    def test_the_pin_can_be_pulled(self, built):
        """A bore that stops inside the block traps its own pin."""
        for d in built.report()["core"]["dowels"]:
            assert d["length_mm"] > d["engagement_mm"]
            exit_at = np.asarray(d["exit_world"])
            # Daylight: the far end is on the block's surface, not buried in it.
            assert float(built.mold_result.block.nearest.on_surface([exit_at])[1][0]) < 0.5

    def test_it_sits_on_the_seam(self, built):
        dowels = built.report()["core"]["dowels"]
        assert dowels
        for d in dowels:
            assert d["seam_drift_mm"] <= 0.4

    def test_dowels_need_no_plate(self, taper, core_config):
        """The whole difference from a screw: a dowel locks the core, not the plate."""
        core_config.core_dowels.enabled = True
        built = pipeline.run(taper, core_config, direction=[1, 0, 0], verify=False)
        applied = [
            i
            for i in built.feature_report.items
            if i["kind"] == "dowel" and i["status"] == "applied"
        ]
        assert applied
        assert not any(i.kind == "plate" for i in built.feature_plan.items)

    def test_a_screw_without_a_plate_is_skipped(self, taper, core_config):
        core_config.core_dowels.enabled = True
        built = pipeline.run(taper, core_config, direction=[1, 0, 0], verify=False)
        plan = built.feature_plan.as_dict()
        plan["items"].append(
            {"id": "screw-x", "kind": "screw", "position": [0, 0, 0], "params": {}}
        )
        out = pipeline.run(taper, core_config, direction=[1, 0, 0], plan=plan, verify=False)
        entry = next(i for i in out.feature_report.items if i["kind"] == "screw")
        assert entry["status"] == "skipped"
        assert "carrier plate" in entry["reason"]


class TestRegistration:

    def test_screws_hold_down_both_halves(self, built):
        """All of them in one half clamps that half and leaves the other loose."""
        screws = built.report()["core"].get("screws") or []
        assert {s["side"] for s in screws} == {"a", "b"}

    def test_the_core_releases_from_both_halves(self, built):
        """The whole point of centring every core feature on the parting surface.

        A number here means a tab or a dowel drifted off the seam far enough
        that its groove became a lock and the mold cannot be opened.
        """
        assert built.report()["core"]["release"]["releases"]

    def test_the_halves_still_separate(self, built):
        assert built.separation["opens"]


class TestTabs:
    def test_they_are_pinched_and_priced(self, built):
        report = built.report()
        applied = [
            i
            for i in report["features"]["items"]
            if i["kind"] == "core_tab" and i["status"] == "applied"
        ]
        assert len(applied) >= 2
        # Tabs cross the glove wall, so the cast stretches off them on the way
        # out. That cost is measured, not waved away.
        assert report["core"]["tab_through_wall_mm3"] > 0

    def test_a_tab_that_would_lock_the_mold_is_skipped_with_a_reason(
        self, taper, full_config
    ):
        full_config.core_tabs.max_seam_drift = 0.0
        built = pipeline.run(taper, full_config, direction=[1, 0, 0], verify=False)
        skipped = [
            i
            for i in built.feature_report.items
            if i["kind"] == "core_tab" and i["status"] == "skipped"
        ]
        assert skipped, "a zero drift budget should reject every tab"
        assert "undercut" in skipped[0]["reason"]

    def test_tabs_work_without_a_plate(self, taper, core_config):
        core_config.core_tabs.enabled = True
        built = pipeline.run(taper, core_config, direction=[1, 0, 0], verify=False)
        applied = [
            i
            for i in built.feature_report.items
            if i["kind"] == "core_tab" and i["status"] == "applied"
        ]
        assert applied


class TestTheWall:
    def test_it_matches_what_was_asked_for(self, built):
        wall = built.report()["core"]["wall"]
        # The eroding ball undershoots at its facet centres and never overshoots.
        assert wall["max_mm"] <= wall["target_mm"] + 1e-3
        assert wall["min_mm"] >= wall["target_mm"] * 0.9
        assert wall["under_90pct_fraction"] == 0.0

    def test_the_rim_is_not_measured_as_a_thin_wall(self, built):
        """Regression: the core's cut face runs out to the cavity at its edge.

        Sampling right up to it reported a 0.8 mm wall against a 2.5 mm target
        on a hand-shaped part -- not a thin spot in the glove, but the hole the
        hand goes through. The exclusion has to sit a wall below the plane.
        """
        wall = built.report()["core"]["wall"]
        assert wall["min_mm"] > 0.5 * wall["target_mm"]


class TestCoreItemsWithoutACore:
    def test_they_are_skipped_with_a_reason_rather_than_dropped(self, taper, fast_config):
        """An edited plan can outlive the core run it came from."""
        plain = mold.build_mold(taper, [1, 0, 0], fast_config)
        ctx = features.FeatureContext.from_mold(plain)
        plan = {
            "pour_axis": [0, 0, 1],
            "items": [
                {"id": "plate-1", "kind": "plate", "position": [0, 0, 0], "params": {}}
            ],
        }
        (a, b, core_mesh), report = features.apply_plan(
            plain.half_a, plain.half_b, ctx, plan, cfg=fast_config
        )
        assert core_mesh is None
        assert report.items[0]["status"] == "skipped"
        assert "no core" in report.items[0]["reason"]


class TestPlacementHelpers:
    def test_seam_drift_is_small_along_the_seam(self, taper, core_config):
        built = mold.build_mold(taper, [1, 0, 0], core_config)
        surface = built.surface
        start = surface.frame.to_world(
            np.array([surface.xs[len(surface.xs) // 2], surface.ys[len(surface.ys) // 2], 0.0])
        )
        along = surface.frame.to_world([1.0, 0.0, 0.0])
        assert core.seam_drift(surface, start, along, 5.0) < 2.0

    def test_seam_drift_is_large_across_it(self, taper, core_config):
        built = mold.build_mold(taper, [1, 0, 0], core_config)
        surface = built.surface
        start = surface.frame.to_world(np.array([0.0, 0.0, float(surface.h.mean())]))
        assert core.seam_drift(surface, start, surface.frame.direction, 10.0) > 5.0

    def test_the_parting_surface_runs_through_the_thick_part_of_the_core(
        self, taper, core_config
    ):
        built, ctx = _ctx(taper, core_config)
        mask = core.core_at_parting(ctx.core, built.surface)
        assert mask.any()
        assert mask.sum() <= built.surface.constrained.sum()


class TestRingInset:
    def test_points_land_inside_at_the_asked_for_inset(self):
        square = np.array([[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]])
        pts = core.ring_inset(square, 5.0, 32)
        assert len(pts) > 0
        assert np.abs(pts).max() <= 15.0 + 1e-6

    def test_mid_edge_points_survive(self):
        """Regression: an exact half-plane test throws away most of the ring.

        A point taken from the middle of an edge and pushed in by the inset sits
        at exactly the inset, so whether it passes is decided by float error --
        which left four screws bunched on one side of a square plate.
        """
        square = np.array([[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]])
        assert len(core.ring_inset(square, 5.0, 32)) >= 20

    def test_an_inset_wider_than_the_polygon_yields_nothing(self):
        square = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
        assert len(core.ring_inset(square, 10.0, 16)) == 0


class TestOutputs:
    def test_the_core_is_written_alongside_the_halves(self, built, tmp_path):
        written = built.write(tmp_path, extras=False)
        assert (tmp_path / "core.stl").exists()
        assert written["core"].endswith("core.stl")

    def test_a_plain_run_has_no_core(self, taper, fast_config):
        plain = pipeline.run(taper, fast_config, direction=[1, 0, 0], verify=False)
        assert plain.core is None
        assert "core" not in plain.report()


class TestConfig:
    def test_a_partial_dict_reaches_the_core_sections(self):
        cfg = MoldConfig.from_dict(
            {
                "core": {"enabled": True, "wall": 1.8},
                "carrier": {"enabled": True, "plate_thickness": 14.0},
                "core_dowels": {"enabled": True, "count": 3},
                "core_tabs": {"enabled": True, "count": 6},
            }
        )
        assert cfg.core.enabled and cfg.core.wall == 1.8
        assert cfg.carrier.plate_thickness == 14.0
        assert cfg.core_dowels.count == 3
        assert cfg.core_tabs.count == 6
        assert cfg.keys.radius == 5.0  # untouched sections keep their defaults

    def test_round_trips(self):
        cfg = MoldConfig()
        cfg.core.enabled = True
        cfg.core.wall = 3.25
        assert MoldConfig.from_dict(cfg.to_dict()).core.wall == 3.25


class TestPourAxisSeededFrame:
    def test_a_core_run_squares_the_block_to_the_pour_axis(self, taper, full_config):
        """Otherwise the plate is a corner wedge sliced off an oblique box."""
        built = mold.build_mold(taper, [1, 0, 0], full_config)
        p = unit(features.choose_pour_axis(taper, full_config))
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


def test_taper_cast_is_open_at_the_rim(taper, full_config):
    """The cut is what opens the glove: below it there is cast, above it nothing."""
    built = pipeline.run(taper, full_config, direction=[1, 0, 0], verify=False)
    p = unit(built.feature_plan.pour_axis)
    plane = built.core_report["plate"]["plane_offset_mm"]

    frame = Frame.from_direction(p)
    local = frame.to_local(taper.vertices)
    kept = trimesh.boolean.intersection(
        [taper, core._slab(frame, local, float(local[:, 2].min()) - 20.0, plane)],
        engine="manifold",
        check_volume=False,
    )
    cast = trimesh.boolean.difference(
        [kept, built.core], engine="manifold", check_volume=False
    )
    # A closed bladder would have cast over the top of the core; an open glove
    # has its topmost cast at the rim.
    assert float((np.asarray(cast.vertices) @ p).max()) == pytest.approx(plane, abs=0.2)
