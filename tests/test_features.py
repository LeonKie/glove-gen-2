"""Keys, pour spout and vents."""

from __future__ import annotations

import numpy as np
import pytest

from glovegen import features, mold, validate
from glovegen.config import MoldConfig, VentConfig


def frustum_volume(r0: float, r1: float, h: float) -> float:
    return np.pi * h / 3.0 * (r0 * r0 + r0 * r1 + r1 * r1)


class TestPrimitives:
    def test_frustum_volume_is_correct(self):
        f = features.frustum(4.0, 7.0, 10.0, sections=512)
        assert f.volume == pytest.approx(frustum_volume(4, 7, 10), rel=1e-3)
        assert f.is_watertight

    def test_cylinder_case(self):
        c = features.frustum(5.0, 5.0, 12.0, sections=512)
        assert c.volume == pytest.approx(np.pi * 25 * 12, rel=1e-3)

    def test_revolved_multi_segment_volume(self):
        """A cone stacked under a cylinder: volumes must add."""
        solid = features.revolved(
            [(2.0, 0.0), (5.0, 4.0), (5.0, 9.0)], sections=512
        )
        expected = frustum_volume(2, 5, 4) + np.pi * 25 * 5
        assert solid.volume == pytest.approx(expected, rel=1e-3)
        assert solid.is_watertight

    def test_revolved_rejects_non_monotonic_z(self):
        with pytest.raises(ValueError, match="strictly increase"):
            features.revolved([(2.0, 0.0), (3.0, 0.0)])

    def test_revolved_needs_two_points(self):
        with pytest.raises(ValueError, match="two"):
            features.revolved([(2.0, 0.0)])


class TestPourAxis:
    def test_points_at_the_fat_end(self, taper):
        axis = features.choose_pour_axis(taper, MoldConfig())
        assert axis @ np.array([0, 0, 1.0]) > 0.9

    def test_explicit_axis_is_honoured(self, taper):
        cfg = MoldConfig(pour_axis=(1.0, 0.0, 0.0))
        assert features.choose_pour_axis(taper, cfg) == pytest.approx([1, 0, 0])

    def test_bad_axis_string_is_rejected(self, taper):
        with pytest.raises(ValueError, match="pour_axis"):
            features.choose_pour_axis(taper, MoldConfig(pour_axis="up"))


class TestAirTraps:
    def test_finds_the_shorter_prong(self, two_prong):
        """The prong that does not get the spout is a trapped-air pocket."""
        traps = features.find_air_traps(
            two_prong,
            np.array([0.0, 0.0, 1.0]),
            VentConfig(count=6, min_spacing=8.0),
            spout_world=np.array([-22.0, 0.0, 40.0]),
        )
        assert traps
        top = traps[0]
        assert top[2] == pytest.approx(30.0, abs=1.5)  # the short prong's tip
        assert top[0] > 10.0  # on the +X side

    def test_one_vent_per_pocket_not_per_cell(self, two_prong):
        """A flat ceiling is one pocket, however many grid cells it spans."""
        traps = features.find_air_traps(
            two_prong, np.array([0.0, 0.0, 1.0]), VentConfig(count=50, min_spacing=1.0)
        )
        heights = np.array([t[2] for t in traps])
        # The base's flat top is a single plateau, so it must not contribute
        # dozens of candidates at the same height.
        assert (np.abs(heights - 6.0) < 0.5).sum() <= 6

    def test_respects_min_spacing(self, two_prong):
        traps = features.find_air_traps(
            two_prong, np.array([0.0, 0.0, 1.0]), VentConfig(count=20, min_spacing=25.0)
        )
        for i, a in enumerate(traps):
            for b in traps[i + 1 :]:
                assert np.linalg.norm(a - b) >= 25.0 - 1e-6

    def test_respects_count(self, two_prong):
        traps = features.find_air_traps(
            two_prong, np.array([0.0, 0.0, 1.0]), VentConfig(count=2, min_spacing=5.0)
        )
        assert len(traps) <= 2


class TestApplyFeatures:
    @pytest.fixture(scope="class")
    def built(self, taper):
        cfg = MoldConfig(block_margin=8.0, pour_axis=(0.0, 0.0, 1.0))
        cfg.parting.grid = 110
        res = mold.build_mold(taper, [1, 0, 0], cfg)
        a, b, rep = features.apply_features(res, taper, cfg)
        return res, a, b, rep

    def test_keys_are_placed_and_spread_out(self, built):
        _, _, _, rep = built
        assert rep.keys["count"] >= 2
        pts = np.array([s["local_xy"] for s in rep.keys["sites"]])
        gaps = [
            np.linalg.norm(pts[i] - pts[j])
            for i in range(len(pts))
            for j in range(i + 1, len(pts))
        ]
        assert min(gaps) > 2.0 * rep.keys["radius_mm"]

    def test_keys_clear_the_cavity(self, built):
        _, _, _, rep = built
        for site in rep.keys["sites"]:
            assert site["cavity_clearance_mm"] >= rep.keys["radius_mm"]

    def test_key_adds_material_to_a_and_removes_it_from_b(self, built):
        res, a, b, rep = built
        assert a.volume > res.half_a.volume  # gained keys
        assert b.volume < res.half_b.volume  # lost sockets and spout

    def test_spout_is_cut(self, built):
        _, _, _, rep = built
        assert rep.spout["length_mm"] > 0
        assert rep.spout["outer_radius_mm"] > rep.spout["inner_radius_mm"]

    def test_halves_still_closed_and_still_open(self, built, taper):
        res, a, b, _ = built
        for half in (a, b):
            assert validate.solid_report(half)["boundary_edges"] == 0
        rep = validate.separation_report(a, b, taper, res.direction)
        assert rep["opens"], rep

    def test_features_can_be_disabled(self, taper):
        cfg = MoldConfig(block_margin=8.0)
        cfg.parting.grid = 90
        cfg.keys.enabled = cfg.spout.enabled = cfg.vents.enabled = False
        res = mold.build_mold(taper, [1, 0, 0], cfg)
        a, b, rep = features.apply_features(res, taper, cfg)
        assert a is res.half_a and b is res.half_b
        assert rep.keys == {} and rep.spout == {} and rep.vents == {}
