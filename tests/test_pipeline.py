"""End-to-end pipeline, config plumbing, viewer format, CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

from glovegen import cli, meshio, pipeline, validate, viewer_format
from glovegen.config import MoldConfig


class TestConfig:
    def test_partial_dict_keeps_defaults(self):
        cfg = MoldConfig.from_dict({"block_margin": 12, "parting": {"grid": 150}})
        assert cfg.block_margin == 12
        assert cfg.parting.grid == 150
        assert cfg.parting.smooth_lambda == MoldConfig().parting.smooth_lambda
        assert cfg.keys.enabled is True

    def test_unknown_keys_are_ignored(self):
        cfg = MoldConfig.from_dict({"nonsense": 1, "keys": {"nonsense": 2, "count": 3}})
        assert cfg.keys.count == 3

    def test_round_trips_through_dict(self):
        cfg = MoldConfig(block_margin=11.5, block_shape="hull")
        cfg.vents.count = 9
        again = MoldConfig.from_dict(cfg.to_dict())
        assert again.block_margin == 11.5
        assert again.block_shape == "hull"
        assert again.vents.count == 9

    def test_vector_fields_accept_lists(self):
        cfg = MoldConfig.from_dict(
            {"pour_axis": [0, 0, 1], "demold": {"forced_direction": [1, 0, 0]}}
        )
        assert cfg.pour_axis == (0, 0, 1)
        assert cfg.demold.forced_direction == (1, 0, 0)


class TestViewerFormat:
    def test_round_trip(self, sphere):
        tri = viewer_format.decode(viewer_format.encode(sphere))
        assert tri.shape == (len(sphere.faces), 3, 3)
        assert np.allclose(tri, sphere.triangles, atol=1e-3)

    def test_rejects_foreign_payload(self):
        with pytest.raises(ValueError, match="GGM2"):
            viewer_format.decode(b"NOPE" + b"\x00" * 8)

    def test_severity_is_one_byte_per_face(self):
        raw = viewer_format.encode_severity(np.array([0.0, 0.5, 1.0, 2.0, -1.0]))
        assert list(raw) == [0, 128, 255, 255, 0]


class TestPipeline:
    @pytest.fixture(scope="class")
    def result(self, taper):
        cfg = MoldConfig(block_margin=8.0)
        cfg.parting.grid = 100
        return pipeline.run(taper, cfg, direction=[1, 0, 0])

    def test_produces_two_closed_halves(self, result):
        for half in (result.half_a, result.half_b):
            assert validate.solid_report(half)["boundary_edges"] == 0

    def test_mold_opens(self, result):
        assert result.separation["opens"]

    def test_report_is_json_serialisable(self, result):
        blob = json.dumps(result.report())
        assert "pull_direction" in blob
        assert "separation" in blob

    def test_report_covers_every_stage(self, result):
        rep = result.report()
        for key in (
            "part", "mold", "parting_surface", "features", "feature_plan",
            "halves", "timings_s",
        ):
            assert key in rep, key

    def test_report_hands_back_an_editable_plan(self, result):
        """What the run chose, in the form you would hand back to change it."""
        plan = result.report()["feature_plan"]
        assert plan["items"]
        assert {i["kind"] for i in plan["items"]} <= {"key", "spout", "vent"}
        assert all(i["params"] for i in plan["items"])

    def test_writes_the_printable_parts(self, result, tmp_path):
        written = result.write(tmp_path)
        for path in written.values():
            assert (tmp_path / path.split("/")[-1]).stat().st_size > 0
        reloaded = trimesh.load(written["half_a"], process=False)
        reloaded.merge_vertices()
        # STL is float32, so allow a loose tolerance on the volume.
        assert reloaded.volume == pytest.approx(result.half_a.volume, rel=1e-4)

    def test_extras_can_be_skipped(self, result, tmp_path):
        written = result.write(tmp_path, extras=False)
        assert set(written) == {"half_a", "half_b"}

    def test_progress_is_monotonic(self, taper):
        seen = []
        cfg = MoldConfig(block_margin=8.0)
        cfg.parting.grid = 80
        pipeline.run(
            taper, cfg, direction=[1, 0, 0], verify=False,
            progress=lambda f, m: seen.append(f),
        )
        assert seen == sorted(seen)
        assert seen[-1] == pytest.approx(1.0)

    def test_searches_a_direction_when_none_given(self, dumbbell, fast_config):
        result = pipeline.run(dumbbell, fast_config, verify=False)
        assert abs(float(result.direction @ np.array([0, 0, 1.0]))) < 0.25

    def test_fallback_uses_the_thinnest_axis(self, taper):
        cfg = MoldConfig(block_margin=8.0)
        cfg.parting.grid = 80
        result = pipeline.run(taper, cfg, search_direction=False, verify=False)
        # The taper is long in Z, so the thinnest extent is X or Y.
        assert abs(float(result.direction[2])) < 1e-9

    def test_rejects_an_open_input(self, tmp_path):
        broken = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
        broken.update_faces(np.arange(len(broken.faces)) > 4)  # punch a hole
        path = tmp_path / "open.ply"
        broken.export(str(path))
        with pytest.raises(validate.SolidError, match="boundary"):
            pipeline.run(path, MoldConfig())


class TestFeaturePlanning:
    """Automatic by default; an explicit plan wins when there is one."""

    @pytest.fixture(scope="class")
    def cfg(self):
        cfg = MoldConfig(block_margin=8.0, pour_axis=(0.0, 0.0, 1.0))
        cfg.parting.grid = 90
        return cfg

    def test_run_plans_features_by_default(self, taper, cfg):
        result = pipeline.run(taper, cfg, direction=[1, 0, 0], verify=False)
        assert result.feature_plan.items
        assert result.feature_report.keys["count"] > 0

    def test_an_explicit_plan_replaces_the_automatic_one(self, taper, cfg):
        auto = pipeline.run(taper, cfg, direction=[1, 0, 0], verify=False)
        plan = auto.report()["feature_plan"]
        plan["items"] = [i for i in plan["items"] if i["kind"] == "spout"]
        pinned = pipeline.run(
            taper, cfg, direction=[1, 0, 0], verify=False, plan=plan
        )
        assert pinned.feature_report.keys == {}
        assert pinned.feature_report.spout["length_mm"] > 0
        assert pinned.half_a.volume < auto.half_a.volume  # no keys added

    def test_features_can_be_re_cut_without_rebuilding_the_mold(self, taper, cfg):
        """The web app's loop: keep the split, change the knobs."""
        from glovegen import features

        built = pipeline.run(taper, cfg, direction=[1, 0, 0], verify=False)
        ctx = features.FeatureContext.from_mold(built.mold_result)
        plan = built.report()["feature_plan"]
        for item in plan["items"]:
            if item["kind"] == "key":
                item["params"]["radius"] = 3.0

        out = pipeline.apply_feature_plan(
            built.mold_result.half_a, built.mold_result.half_b, ctx, plan, cast=taper
        )
        assert out.separation["opens"]
        assert out.feature_report.keys["radius_mm"] == 3.0
        # smaller knobs than the 5 mm default, so less material added to A
        assert out.half_a.volume < built.half_a.volume
        rep = out.report()
        assert set(rep) >= {"mold", "features", "feature_plan", "separation", "halves"}
        json.dumps(rep)

    def test_re_cutting_starts_from_the_base_halves_every_time(self, taper, cfg):
        """Applying twice must not stack: the second run is not built on the first."""
        from glovegen import features

        built = pipeline.run(taper, cfg, direction=[1, 0, 0], verify=False)
        ctx = features.FeatureContext.from_mold(built.mold_result)
        plan = built.report()["feature_plan"]
        args = (built.mold_result.half_a, built.mold_result.half_b, ctx, plan)
        first = pipeline.apply_feature_plan(*args, verify=False)
        second = pipeline.apply_feature_plan(*args, verify=False)
        assert first.half_a.volume == pytest.approx(second.half_a.volume, rel=1e-9)


class TestHeatmapMesh:
    def test_colours_faces(self, dumbbell):
        mesh, stats = pipeline.heatmap_mesh(dumbbell, [0, 0, 1], proxy_faces=5000)
        assert len(mesh.visual.face_colors) == len(mesh.faces)
        assert stats["undercut_area_fraction"] > 0

    def test_clean_direction_has_no_hot_faces(self, dumbbell):
        _, stats = pipeline.heatmap_mesh(dumbbell, [1, 0, 0], proxy_faces=5000)
        assert stats["undercut_face_count"] == 0


class TestMeshIO:
    def test_keeps_the_largest_shell(self, tmp_path):
        big = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
        speck = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
        speck.apply_translation([100, 0, 0])
        combined = trimesh.util.concatenate([big, speck])
        path = tmp_path / "two.stl"
        combined.export(str(path))
        loaded = meshio.load_mesh(path)
        assert len(loaded.faces) == len(big.faces)

    def test_decimate_reduces_and_stays_closed(self, sphere):
        small = meshio.decimate(sphere, 200)
        assert len(small.faces) <= 260
        assert validate.solid_report(small)["boundary_edges"] == 0

    def test_decimate_is_a_no_op_when_already_small(self, sphere):
        assert meshio.decimate(sphere, 10**9) is sphere

    def test_proxy_is_cached(self, sphere):
        a = meshio.proxy(sphere, 200)
        b = meshio.proxy(sphere, 200)
        assert a is b

    def test_stats_flag_closedness_separately_from_watertight(self, sphere):
        s = meshio.mesh_stats(sphere)
        assert s["closed"] is True
        assert s["boundary_edges"] == 0
        assert s["volume_cm3"] > 0


class TestCLI:
    def test_analyze_emits_one_json_document(self, taper, tmp_path, capsys):
        """stdout must be a single document so it can be piped into jq."""
        path = tmp_path / "part.stl"
        taper.export(str(path))
        code = cli.main(
            ["-q", "analyze", str(path), "--direction", "1", "0", "0", "--top", "0"]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["pull_direction"] == [1.0, 0.0, 0.0]
        assert payload["part"]["closed"] is True
        assert payload["best"]["undercut_percent"] == 0.0

    def test_analyze_writes_a_heatmap(self, dumbbell, tmp_path, capsys):
        path = tmp_path / "part.stl"
        dumbbell.export(str(path))
        heat = tmp_path / "heat.ply"
        code = cli.main(
            ["-q", "analyze", str(path), "--direction", "0", "0", "1",
             "--heatmap", str(heat)]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert heat.stat().st_size > 0
        assert payload["heatmap"]["undercut_face_count"] > 0

    def test_mold_writes_parts_and_report(self, taper, tmp_path, capsys):
        path = tmp_path / "part.stl"
        taper.export(str(path))
        out = tmp_path / "out"
        code = cli.main(
            [
                "-q", "mold", str(path), "-o", str(out),
                "--direction", "1", "0", "0",
                "--margin", "8", "--grid", "90", "--no-vents",
            ]
        )
        assert code == 0
        assert (out / "mold_half_a.stl").exists()
        assert (out / "mold_half_b.stl").exists()
        report = json.loads((out / "report.json").read_text())
        assert report["separation"]["opens"]

    def test_feature_sizes_are_settable(self, taper, tmp_path, capsys):
        path = tmp_path / "part.stl"
        taper.export(str(path))
        out = tmp_path / "out_sized"
        code = cli.main([
            "-q", "mold", str(path), "-o", str(out), "--direction", "1", "0", "0",
            "--margin", "8", "--grid", "90", "--no-vents", "--no-verify",
            "--key-radius", "3", "--key-height", "6", "--spout-outer", "12",
        ])
        assert code == 0
        report = json.loads((out / "report.json").read_text())
        assert report["features"]["keys"]["radius_mm"] == 3.0
        assert report["features"]["keys"]["height_mm"] == 6.0
        assert report["features"]["spout"]["outer_radius_mm"] == 12.0

    def test_a_saved_plan_can_be_applied_back(self, taper, tmp_path, capsys):
        """The command line stays automatic, but an edited plan overrides it."""
        path = tmp_path / "part.stl"
        taper.export(str(path))
        common = [
            "-q", "mold", str(path), "--direction", "1", "0", "0",
            "--margin", "8", "--grid", "90", "--no-vents", "--no-verify",
        ]
        auto = tmp_path / "auto"
        assert cli.main([*common, "-o", str(auto)]) == 0
        report = json.loads((auto / "report.json").read_text())
        assert report["features"]["keys"]["count"] > 0

        plan = report["feature_plan"]
        for item in plan["items"]:
            if item["kind"] == "key":
                item["enabled"] = False
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan))

        edited = tmp_path / "edited"
        assert cli.main([*common, "-o", str(edited), "--plan", str(plan_path)]) == 0
        again = json.loads((edited / "report.json").read_text())
        assert again["features"]["keys"]["count"] == 0
        assert again["features"]["spout"]["length_mm"] > 0

    def test_a_whole_report_is_accepted_as_a_plan(self, taper, tmp_path, capsys):
        path = tmp_path / "part.stl"
        taper.export(str(path))
        common = [
            "-q", "mold", str(path), "--direction", "1", "0", "0",
            "--margin", "8", "--grid", "90", "--no-verify", "--no-extras",
        ]
        first = tmp_path / "first"
        assert cli.main([*common, "-o", str(first)]) == 0
        second = tmp_path / "second"
        assert cli.main([*common, "-o", str(second),
                         "--plan", str(first / "report.json")]) == 0
        a = json.loads((first / "report.json").read_text())["feature_plan"]
        b = json.loads((second / "report.json").read_text())["feature_plan"]
        assert a["items"] == b["items"]

    def test_hull_block_option(self, taper, tmp_path):
        path = tmp_path / "part.stl"
        taper.export(str(path))
        out = tmp_path / "out_hull"
        assert cli.main([
            "-q", "mold", str(path), "-o", str(out), "--direction", "1", "0", "0",
            "--block", "hull", "--grid", "90", "--no-keys", "--no-vents",
            "--no-spout", "--no-extras", "--no-verify",
        ]) == 0
        assert (out / "mold_half_a.stl").exists()
