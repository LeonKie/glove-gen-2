"""The core: the inward offset, and the part it is assembled into."""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from glovegen import core, features, mold, pipeline, validate
from glovegen.config import MoldConfig
from glovegen.core import CoreError
from glovegen.frame import Frame

Z = Frame.from_direction([0.0, 0.0, 1.0])


def sphere_radius(mesh: trimesh.Trimesh) -> float:
    return float((mesh.volume / (4.0 / 3.0 * math.pi)) ** (1.0 / 3.0))


def measured_wall(cavity: trimesh.Trimesh, inset: trimesh.Trimesh, n: int = 4000):
    """Median distance from the inset surface back to the cavity."""
    pts = trimesh.sample.sample_surface(inset, n)[0]
    return float(np.median(trimesh.proximity.closest_point(cavity, pts)[1]))


def rotated(mesh: trimesh.Trimesh, deg: float = 17.0) -> trimesh.Trimesh:
    """Take a shape off the grid axes.

    An axis-aligned box is a degenerate case for a voxel field: the grid is
    derived from the mesh's own bounding box, so every boundary voxel centre
    lands exactly on a face and the inset comes out half a voxel shallow. No
    scan looks like that, and neither should a test that is not about it.
    """
    out = mesh.copy()
    out.apply_transform(
        trimesh.transformations.rotation_matrix(math.radians(deg), [1, 0.4, 0.2])
    )
    return out


# --------------------------------------------------------------------------
# the inward offset
# --------------------------------------------------------------------------


class TestDeflate:
    def test_sphere_inset_is_a_smaller_sphere(self):
        s = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
        got, stats = core.deflate(s, Z, 3.0, pitch=0.5, edge_length=0.5)
        assert sphere_radius(got) == pytest.approx(sphere_radius(s) - 3.0, abs=0.15)
        assert stats["grid_pitch_mm"] == 0.5

    @pytest.mark.parametrize(
        "shape",
        [
            trimesh.creation.icosphere(subdivisions=4, radius=20.0),
            rotated(trimesh.creation.box(extents=(40, 40, 40))),
            trimesh.creation.capsule(radius=12, height=40, count=[40, 40]),
            trimesh.creation.torus(
                major_radius=25, minor_radius=9, major_sections=64, minor_sections=32
            ),
        ],
        ids=["sphere", "rotated-box", "capsule", "torus"],
    )
    def test_wall_comes_out_the_thickness_asked_for(self, shape):
        """The whole point of _LEVEL_BIAS: pin it against curvature and flatness.

        A flat rotated face and a 12 mm-radius capsule have to agree, or the
        calibration is fitting one shape rather than the extractor.
        """
        got, _ = core.deflate(shape, Z, 3.0, pitch=0.5, edge_length=0.5)
        assert measured_wall(shape, got) == pytest.approx(3.0, abs=0.1)

    def test_wall_error_shrinks_with_the_pitch(self):
        s = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
        coarse, _ = core.deflate(s, Z, 3.0, pitch=2.0, edge_length=0.5)
        fine, _ = core.deflate(s, Z, 3.0, pitch=0.5, edge_length=0.5)
        assert abs(measured_wall(s, fine) - 3.0) < abs(measured_wall(s, coarse) - 3.0)

    def test_result_is_a_solid(self):
        s = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
        got, _ = core.deflate(s, Z, 3.0)
        validate.assert_solid_enough(got, "inset")
        assert core.components(got) == 1

    def test_too_thick_a_wall_leaves_nothing(self):
        s = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        with pytest.raises(CoreError, match="no core"):
            core.deflate(s, Z, 12.0)

    def test_thickness_must_be_positive(self):
        s = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
        with pytest.raises(CoreError, match="positive"):
            core.deflate(s, Z, 0.0)

    def test_severing_a_thin_feature_is_reported_not_hidden(self, dumbbell):
        """A wall thicker than half the rod erases it and leaves two balls.

        This is the real failure mode of an inward offset -- it does not get
        thin, it disappears -- so the count has to come out right.
        """
        got, _ = core.deflate(dumbbell, Z, 3.0, pitch=0.5)
        assert core.components(got) == 2

    def test_pitch_is_coarsened_rather_than_blowing_the_budget(self):
        s = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
        _, stats = core.deflate(s, Z, 2.0, pitch=0.05, max_voxels=200_000)
        assert stats["grid_pitch_mm"] > 0.05
        assert stats["grid_voxels"] <= 200_000


class TestOccupancy:
    def test_grid_volume_matches_the_mesh(self):
        """The occupancy comes from ray crossings, so it should integrate to the
        mesh's own volume -- the same check demold makes of its column model."""
        s = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
        occ, _, pitch = core.occupancy_grid(
            s, Z, 0.5, pad=2.0, max_voxels=50_000_000
        )
        assert occ.sum() * pitch**3 == pytest.approx(s.volume, rel=0.01)

    def test_signed_distance_is_positive_inside(self):
        s = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
        occ, _, pitch = core.occupancy_grid(s, Z, 1.0, pad=3.0, max_voxels=10_000_000)
        sdf = core.signed_distance(occ, pitch)
        assert sdf[occ].min() > 0
        assert sdf[~occ].max() < 0
        # Deepest point of a 20 mm sphere is 20 mm in.
        assert sdf.max() == pytest.approx(20.0, abs=1.0)


# --------------------------------------------------------------------------
# the assembled core
# --------------------------------------------------------------------------


def build(part, cfg=None, **core_kwargs):
    """Mold + core for a part, at test resolution."""
    cfg = cfg or MoldConfig(block_margin=8.0)
    cfg.parting.grid = 90
    cfg.core.enabled = True
    cfg.core.thickness = core_kwargs.pop("thickness", 2.0)
    for k, v in core_kwargs.items():
        setattr(cfg.core, k, v)
    mr = mold.build_mold(part, [0, 0, 1], cfg)
    pour = features.choose_pour_axis(mr.cavity, cfg)
    cr = core.build_core(mr.cavity, mr.frame, pour, cfg)
    return cfg, mr, cr


@pytest.fixture(scope="module")
def rod() -> trimesh.Trimesh:
    """A rod across the pull direction, so the pour axis and +/-d stay
    distinct -- the ordinary case for a hand-and-forearm scan."""
    m = trimesh.creation.cylinder(radius=14, height=60, sections=48)
    m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(90), [1, 0, 0]))
    return m


class TestBuildCore:
    def test_core_is_one_solid_piece(self, rod):
        _, _, cr = build(rod)
        validate.assert_solid_enough(cr.core, "core")
        assert cr.stats["components"] == 1

    def test_cuff_sits_a_wall_inside_the_end_so_cap_and_body_meet(self, rod):
        """If the cuff plane is taken at the very extreme the cap floats free of
        the inset body and the core comes out in two pieces."""
        _, mr, cr = build(rod, thickness=2.0)
        extreme = float((np.asarray(mr.cavity.vertices) @ cr.pour_axis).max())
        assert cr.cuff_offset <= extreme - 2.0
        assert cr.stats["components"] == 1

    def test_the_halves_are_untouched_by_the_core(self, rod):
        """Nothing is cut into the mold for the core, so the split is exactly
        what it was without one."""
        cfg = MoldConfig(block_margin=8.0)
        cfg.parting.grid = 90
        plain = mold.build_mold(rod, [0, 0, 1], cfg)
        _, withcore, _ = build(rod)
        assert withcore.half_a.volume == pytest.approx(plain.half_a.volume)
        assert withcore.half_b.volume == pytest.approx(plain.half_b.volume)

    def test_glove_is_open_at_the_cuff(self, rod):
        """No cast beyond the cuff plane -- that is what the full-size cap is for."""
        _, mr, cr = build(rod)
        glove = core.glove_preview(mr.cavity, cr)
        beyond = np.asarray(glove.vertices) @ cr.pour_axis
        assert beyond.max() <= cr.cuff_offset + 1e-6
        assert core.components(glove) == 1

    def test_glove_is_the_gap_between_cavity_and_core(self, rod):
        _, mr, cr = build(rod)
        glove = core.glove_preview(mr.cavity, cr)
        assert glove.volume < mr.cavity.volume - cr.body.volume + 1.0
        assert glove.volume > 0

    def test_split_conserves_volume(self, rod):
        _, mr, cr = build(rod, split=True)
        a, b = mold.split_mold(cr.core, mr.parting_solid)
        assert a.volume + b.volume == pytest.approx(cr.core.volume, rel=1e-4)


class TestComponents:
    def test_counts_solids_not_unwelded_seams(self):
        """trimesh.split reads a boolean result's unwelded coincident vertices as
        component boundaries and reports one sphere as dozens of pieces."""
        a = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        b = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        b.apply_translation([5, 0, 0])
        joined = mold._boolean(trimesh.boolean.union, [a, b], "two overlapping")
        assert core.components(joined) == 1

    def test_counts_genuinely_separate_solids(self):
        a = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
        b = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
        b.apply_translation([40, 0, 0])
        assert core.components(trimesh.util.concatenate([a, b])) == 2


# --------------------------------------------------------------------------
# config plumbing
# --------------------------------------------------------------------------


class TestCoreConfig:
    def test_partial_nesting_survives(self):
        cfg = MoldConfig.from_dict({"core": {"thickness": 1.5}})
        assert cfg.core.thickness == 1.5
        # Everything not mentioned keeps its default rather than vanishing.
        assert cfg.core.max_voxels == 40_000_000
        assert cfg.block_margin == 10.0

    def test_unknown_keys_are_dropped_at_every_level(self):
        cfg = MoldConfig.from_dict({"nope": 1, "core": {"nope": 2, "thickness": 3}})
        assert cfg.core.thickness == 3

    def test_off_by_default(self):
        assert MoldConfig().core.enabled is False

    def test_round_trips_through_a_dict(self):
        cfg = MoldConfig()
        cfg.core.enabled = True
        cfg.core.cuff_offset = -4.0
        assert MoldConfig.from_dict(cfg.to_dict()) == cfg


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


class TestPipelineWithCore:
    @pytest.fixture(scope="class")
    def run(self, rod):
        cfg = MoldConfig(block_margin=8.0)
        cfg.parting.grid = 90
        cfg.core.enabled = True
        cfg.core.thickness = 2.0
        return pipeline.run(rod, cfg, direction=[0, 0, 1], verify=True)

    def test_produces_a_core_and_a_glove(self, run):
        assert run.core is not None and run.glove is not None
        validate.assert_solid_enough(run.core, "core")
        assert run.core_parts() == {"core": run.core}

    def test_halves_still_open(self, run):
        assert run.separation["opens"]

    def test_reports_the_wall_it_achieved(self, run):
        wall = run.core_stats["wall"]
        assert wall["measured"]
        assert wall["median_mm"] == pytest.approx(2.0, abs=0.35)

    def test_reports_a_single_piece(self, run):
        assert run.core_stats["components"] == 1

    def test_report_is_json_able(self, run):
        import json

        assert "core" in json.loads(json.dumps(run.report()))

    def test_writes_the_core(self, run, tmp_path):
        written = run.write(tmp_path, extras=True)
        assert (tmp_path / "core.stl").exists()
        assert (tmp_path / "glove_preview.stl").exists()
        assert "core" in written

    def test_split_writes_two_core_halves(self, rod, tmp_path):
        cfg = MoldConfig(block_margin=8.0)
        cfg.parting.grid = 90
        cfg.core.enabled = True
        cfg.core.split = True
        result = pipeline.run(rod, cfg, direction=[0, 0, 1], verify=True)
        result.write(tmp_path, extras=False)
        assert (tmp_path / "core_a.stl").exists()
        assert (tmp_path / "core_b.stl").exists()
        assert not (tmp_path / "core.stl").exists()
        assert result.core_stats["halves_open"]

    def test_disabled_by_default_leaves_the_old_behaviour(self, rod, tmp_path):
        result = pipeline.run(
            rod, MoldConfig(block_margin=8.0), direction=[0, 0, 1], verify=False
        )
        assert result.core is None
        assert result.core_parts() == {}
        assert "core" not in result.report()
        result.write(tmp_path, extras=False)
        assert not (tmp_path / "core.stl").exists()
