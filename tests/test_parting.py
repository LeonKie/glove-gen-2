"""The parting surface: a constrained height field, and the solid built from it."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from glovegen import parting
from glovegen.config import PartingConfig
from glovegen.frame import Frame


def analytic_prism_volume(surface, z_top: float) -> float:
    """Volume between the height field and ``z_top``, computed independently.

    Uses the same triangulation but sums prism volumes directly, so it is a real
    cross-check on the extrusion, its winding and its caps.
    """
    nx, ny = surface.shape
    dx = float(surface.xs[1] - surface.xs[0])
    dy = float(surface.ys[1] - surface.ys[0])
    idx = np.arange(nx * ny).reshape(nx, ny)
    h = surface.h.ravel()
    total = 0.0
    for a, b, c in (
        (idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:]),
        (idx[:-1, :-1], idx[1:, 1:], idx[:-1, 1:]),
    ):
        tri = (h[a.ravel()] + h[b.ravel()] + h[c.ravel()]) / 3.0
        total += 0.5 * dx * dy * float(np.sum(z_top - tri))
    return total


def build(mesh, direction, margin=8.0, grid=120):
    frame = Frame.from_direction(direction)
    local = frame.to_local(mesh.convex_hull.vertices)
    lo = local.min(axis=0) - margin
    hi = local.max(axis=0) + margin
    surface = parting.build(
        mesh, direction, lo[:2], hi[:2], PartingConfig(grid=grid), frame=frame
    )
    return surface, lo, hi


class TestHeightField:
    def test_sphere_parts_on_its_equator(self, sphere):
        """A centred sphere's mid-surface is z=0, so that is where it must split."""
        surface, _, _ = build(sphere, [0, 0, 1])
        assert np.abs(surface.h).max() < 1e-6

    def test_stays_inside_the_cavity_band(self, dumbbell):
        """The invariant everything else rests on: h never leaves the part."""
        surface, _, _ = build(dumbbell, [1, 0, 0])
        c = surface.constrained
        assert c.any()
        assert np.all(surface.h[c] >= surface.band_lo[c] - 1e-9)
        assert np.all(surface.h[c] <= surface.band_hi[c] + 1e-9)

    def test_band_collapses_at_the_silhouette(self, sphere):
        """Where the part thins to nothing the surface is pinned to it exactly."""
        surface, _, _ = build(sphere, [0, 0, 1], grid=160)
        c = surface.constrained
        thickness = (surface.band_hi - surface.band_lo)[c]
        assert thickness.min() < 0.5  # silhouette columns
        assert thickness.max() > 30.0  # straight through the middle

    def test_unconstrained_columns_interpolate_smoothly(self, dumbbell):
        surface, _, _ = build(dumbbell, [1, 0, 0])
        assert (~surface.constrained).any()
        assert np.all(np.isfinite(surface.h))

    def test_grid_spans_the_requested_footprint(self, sphere):
        surface, lo, hi = build(sphere, [0, 0, 1])
        assert surface.xs[0] == pytest.approx(lo[0])
        assert surface.xs[-1] == pytest.approx(hi[0])
        assert surface.ys[0] == pytest.approx(lo[1])
        assert surface.ys[-1] == pytest.approx(hi[1])

    def test_rejects_a_direction_that_misses_the_part(self, sphere):
        frame = Frame.from_direction([0, 0, 1])
        with pytest.raises(ValueError, match="no ray column"):
            parting.build(
                sphere, [0, 0, 1], [500, 500], [520, 520], PartingConfig(grid=20),
                frame=frame,
            )

    def test_report_is_json_friendly(self, sphere):
        surface, _, _ = build(sphere, [0, 0, 1])
        rep = surface.report()
        assert rep["constrained_columns"] > 0
        assert rep["total_columns"] >= rep["constrained_columns"]
        assert rep["cell_mm"] > 0


class TestPartingSolid:
    def test_volume_matches_the_analytic_prism(self, sphere):
        surface, _, hi = build(sphere, [0, 0, 1])
        z_top = float(hi[2]) + 1.0
        solid = surface.solid(z_top, overhang=0.0)
        assert solid.volume == pytest.approx(
            analytic_prism_volume(surface, z_top), rel=1e-9
        )

    def test_solid_is_watertight(self, dumbbell):
        surface, _, hi = build(dumbbell, [1, 0, 0])
        solid = surface.solid(float(hi[2]) + 1.0)
        assert solid.is_watertight
        assert solid.volume > 0

    def test_solid_covers_the_whole_footprint(self, sphere):
        surface, lo, hi = build(sphere, [0, 0, 1])
        solid = surface.solid(float(hi[2]) + 1.0, overhang=0.0)
        frame = surface.frame
        local = frame.to_local(solid.vertices)
        assert local[:, 0].min() == pytest.approx(surface.xs[0], abs=1e-6)
        assert local[:, 0].max() == pytest.approx(surface.xs[-1], abs=1e-6)

    def test_solid_overhangs_the_block_walls(self, sphere):
        """The side walls must miss the block's, or the boolean emits needles."""
        surface, lo, hi = build(sphere, [0, 0, 1])
        solid = surface.solid(float(hi[2]) + 1.0, overhang=5.0)
        local = surface.frame.to_local(solid.vertices)
        assert local[:, 0].min() == pytest.approx(surface.xs[0] - 5.0, abs=1e-6)
        assert local[:, 0].max() == pytest.approx(surface.xs[-1] + 5.0, abs=1e-6)
        assert local[:, 1].min() == pytest.approx(surface.ys[0] - 5.0, abs=1e-6)
        assert local[:, 1].max() == pytest.approx(surface.ys[-1] + 5.0, abs=1e-6)
        assert solid.is_watertight

    def test_overhang_does_not_move_the_cut(self, dumbbell):
        """The extension is flat at the edge height, so h is untouched: the two
        solids agree wherever the block is, which is all that gets cut."""
        surface, _, hi = build(dumbbell, [1, 0, 0])
        z_top = float(hi[2]) + 1.0
        plain = surface.solid(z_top, overhang=0.0)
        padded = surface.solid(z_top, overhang=5.0)
        # The extra volume is exactly the flat skirt round the outside, so
        # clipping the padded solid back to the footprint must recover the plain
        # one. The clip box spans the footprint in x/y and comfortably brackets
        # the solid in z.
        assert padded.volume > plain.volume
        z_lo = float(surface.h.min()) - 1.0
        lo = np.array([surface.xs[0], surface.ys[0], z_lo])
        hi = np.array([surface.xs[-1], surface.ys[-1], z_top + 1.0])
        box = trimesh.creation.box(extents=(hi - lo))
        box.apply_translation((lo + hi) / 2.0)
        box.apply_transform(np.linalg.inv(surface.frame.transform()))
        clip = trimesh.boolean.intersection([padded, box], engine="manifold")
        assert clip.volume == pytest.approx(plain.volume, rel=1e-6)

    def test_rejects_a_z_top_below_the_surface(self, sphere):
        surface, _, _ = build(sphere, [0, 0, 1])
        with pytest.raises(ValueError, match="not above"):
            surface.solid(float(surface.h.min()) - 5.0)

    def test_surface_mesh_is_an_open_sheet(self, sphere):
        surface, _, _ = build(sphere, [0, 0, 1])
        sheet = surface.surface_mesh()
        nx, ny = surface.shape
        assert len(sheet.faces) == 2 * (nx - 1) * (ny - 1)
        assert not sheet.is_watertight  # it is a sheet, not a solid


class TestPersistence:
    """The surface outlives the run that solved it, so features can be re-cut."""

    def test_round_trips_through_disk(self, dumbbell, tmp_path):
        surface, _, _ = build(dumbbell, [1, 0, 0])
        again = parting.PartingSurface.load(surface.save(tmp_path / "parting.npz"))
        assert np.array_equal(again.h, surface.h)
        assert np.array_equal(again.constrained, surface.constrained)
        assert np.array_equal(again.band_lo, surface.band_lo)
        assert np.array_equal(again.band_hi, surface.band_hi)
        assert np.array_equal(again.xs, surface.xs)
        assert again.report() == surface.report()

    def test_reloaded_frame_still_points_the_same_way(self, dumbbell, tmp_path):
        surface, _, _ = build(dumbbell, [1, 0, 0])
        again = parting.PartingSurface.load(surface.save(tmp_path / "p.npz"))
        assert np.allclose(again.frame.rot, surface.frame.rot)
        assert np.allclose(again.frame.direction, [1, 0, 0])
        # and it still produces the same splitting solid
        z_top = float(surface.h.max()) + 5.0
        assert again.solid(z_top).volume == pytest.approx(surface.solid(z_top).volume)


class TestSmoothing:
    def test_higher_lambda_flattens_the_surface(self, dumbbell):
        frame = Frame.from_direction([1, 0, 0])
        local = frame.to_local(dumbbell.convex_hull.vertices)
        lo, hi = local.min(axis=0) - 8, local.max(axis=0) + 8

        def roughness(lam):
            cfg = PartingConfig(grid=100, smooth_lambda=lam)
            s = parting.build(dumbbell, [1, 0, 0], lo[:2], hi[:2], cfg, frame=frame)
            return float(np.abs(np.diff(s.h, axis=0)).mean())

        assert roughness(50.0) <= roughness(0.05) + 1e-12
