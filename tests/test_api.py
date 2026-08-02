"""HTTP surface. Geometry runs in a worker process, so these lean on polling."""

from __future__ import annotations

import json
import time

import pytest
import trimesh

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_session=None):
    root = tmp_path_factory.mktemp("apistore")
    import importlib
    import os

    os.environ["GLOVEGEN_STORE"] = str(root)
    os.environ["GLOVEGEN_TTL_HOURS"] = "24"
    from server import app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture(scope="module")
def part_bytes(tmp_path_factory):
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=15.0)
    path = tmp_path_factory.mktemp("mesh") / "ball.stl"
    mesh.export(str(path))
    return path.read_bytes()


def _wait_mesh(client, mesh_id, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get(f"/api/meshes/{mesh_id}").json()["mesh"]["state"]
        if state in ("ready", "failed"):
            return state
        time.sleep(0.3)
    return "timeout"


def _wait_job(client, job_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()["job"]
        if job["state"] in ("done", "failed", "interrupted"):
            return job
        time.sleep(0.3)
    return {"state": "timeout"}


class TestStatus:
    def test_status_exposes_defaults(self, client):
        body = client.get("/api/status").json()
        assert body["defaults"]["block_margin"] > 0
        assert "parting" in body["defaults"]
        assert body["storage"]["ttl_hours"] == 24

    def test_index_is_served(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "glovegen" in res.text


class TestValidation:
    def test_unknown_mesh_is_404(self, client):
        assert client.get("/api/meshes/deadbeef").status_code == 404

    def test_unsupported_file_type(self, client):
        res = client.post("/api/meshes", files={"file": ("a.txt", b"hello")})
        assert res.status_code == 415

    def test_empty_upload(self, client):
        res = client.post("/api/meshes", files={"file": ("a.stl", b"")})
        assert res.status_code == 400

    def test_job_for_unknown_mesh(self, client):
        res = client.post("/api/jobs", json={"mesh_id": "deadbeef", "kind": "mold"})
        assert res.status_code == 404

    def test_bad_job_kind(self, client):
        res = client.post("/api/jobs", json={"mesh_id": "deadbeef", "kind": "polish"})
        assert res.status_code == 422

    def test_features_job_needs_a_source(self, client):
        res = client.post("/api/jobs", json={"kind": "features", "config": {}})
        assert res.status_code == 422

    def test_features_job_for_unknown_source(self, client):
        res = client.post(
            "/api/jobs",
            json={"kind": "features", "config": {"source_job": "deadbeef", "plan": {}}},
        )
        assert res.status_code == 404

    def test_unknown_job_is_404(self, client):
        assert client.get("/api/jobs/deadbeef").status_code == 404
        assert client.get("/api/jobs/deadbeef/files/x.stl").status_code == 404

    def test_unknown_preview_name(self, client):
        assert client.get("/api/jobs/deadbeef/preview/evil.bin").status_code == 404


@pytest.mark.slow
class TestRoundTrip:
    def test_upload_analyze_and_mold(self, client, part_bytes):
        res = client.post("/api/meshes", files={"file": ("ball.stl", part_bytes)})
        assert res.status_code == 200
        mesh_id = res.json()["mesh"]["id"]
        assert _wait_mesh(client, mesh_id) == "ready"

        meta = client.get(f"/api/meshes/{mesh_id}").json()["mesh"]
        assert meta["stats"]["closed"]

        # viewer geometry decodes and its face count matches the heatmap length
        from glovegen import viewer_format

        payload = client.get(f"/api/meshes/{mesh_id}/viewer.bin").content
        tri = viewer_format.decode(payload)

        heat = client.post(
            f"/api/meshes/{mesh_id}/heatmap", json={"direction": [0, 0, 1]}
        )
        assert heat.status_code == 200
        assert len(heat.content) == len(tri)
        assert "X-Glovegen-Stats" in heat.headers
        # a sphere releases everywhere
        assert set(heat.content) == {0}

        job_id = client.post(
            "/api/jobs",
            json={
                "mesh_id": mesh_id,
                "kind": "mold",
                "config": {
                    "block_margin": 6,
                    "parting": {"grid": 80},
                    "direction": [0, 0, 1],
                    "vents": {"enabled": False},
                },
            },
        ).json()["job"]["id"]
        job = _wait_job(client, job_id)
        assert job["state"] == "done", job.get("message")

        report = job["result"]["report"]
        assert report["separation"]["opens"]
        assert report["mold"]["split_volume_error_cm3"] == pytest.approx(0, abs=1e-3)
        assert set(job["parts"]) >= {"mold_half_a.stl", "mold_half_b.stl", "report.json"}

        stl = client.get(f"/api/jobs/{job_id}/files/mold_half_a.stl")
        assert stl.status_code == 200 and len(stl.content) > 1000
        for which in ("half_a", "half_b", "parting"):
            assert client.get(f"/api/jobs/{job_id}/preview/{which}.bin").status_code == 200

    def test_features_are_re_cut_from_the_built_mold(self, client, part_bytes):
        """The web app's second step: edit the proposal, re-apply, no rebuild."""
        mesh_id = client.post(
            "/api/meshes", files={"file": ("ball4.stl", part_bytes)}
        ).json()["mesh"]["id"]
        assert _wait_mesh(client, mesh_id) == "ready"

        mold_id = client.post(
            "/api/jobs",
            json={
                "mesh_id": mesh_id,
                "kind": "mold",
                "config": {
                    "block_margin": 10,
                    "parting": {"grid": 70},
                    "direction": [0, 0, 1],
                    "vents": {"enabled": False},
                },
            },
        ).json()["job"]["id"]
        mold_job = _wait_job(client, mold_id)
        assert mold_job["state"] == "done", mold_job.get("message")

        plan = mold_job["result"]["plan"]
        assert mold_job["result"]["base_job"] == mold_id
        spout = next(i for i in plan["items"] if i["kind"] == "spout")
        spout["params"]["outer_radius"] = 12.0

        job = client.post(
            "/api/jobs",
            json={"kind": "features", "config": {"source_job": mold_id, "plan": plan}},
        ).json()["job"]
        assert job["mesh_id"] == mesh_id
        done = _wait_job(client, job["id"])
        assert done["state"] == "done", done.get("message")

        report = done["result"]["report"]
        assert report["features"]["spout"]["outer_radius_mm"] == 12.0
        assert report["separation"]["opens"]
        assert set(done["parts"]) >= {"mold_half_a.stl", "mold_half_b.stl", "report.json"}
        for which in ("half_a", "half_b", "parting"):
            assert client.get(f"/api/jobs/{job['id']}/preview/{which}.bin").status_code == 200

        # an edit of an edit still re-cuts the original mold, never the edited one
        chained = client.post(
            "/api/jobs",
            json={
                "kind": "features",
                "config": {"source_job": job["id"], "plan": done["result"]["plan"]},
            },
        ).json()["job"]
        assert chained["config"]["base_job"] == mold_id

    def test_features_job_rejects_a_broken_plan(self, client, part_bytes):
        mesh_id = client.post(
            "/api/meshes", files={"file": ("ball5.stl", part_bytes)}
        ).json()["mesh"]["id"]
        assert _wait_mesh(client, mesh_id) == "ready"
        mold_id = client.post(
            "/api/jobs",
            json={
                "mesh_id": mesh_id,
                "kind": "mold",
                "config": {
                    "block_margin": 10,
                    "parting": {"grid": 70},
                    "direction": [0, 0, 1],
                    "vents": {"enabled": False},
                },
            },
        ).json()["job"]["id"]
        assert _wait_job(client, mold_id)["state"] == "done"

        res = client.post(
            "/api/jobs",
            json={
                "kind": "features",
                "config": {
                    "source_job": mold_id,
                    "plan": {"items": [{"kind": "hinge", "position": [0, 0, 0]}]},
                },
            },
        )
        assert res.status_code == 422

    def test_features_job_needs_a_finished_source(self, client, part_bytes):
        """An analyze job has no cached mold to cut into."""
        mesh_id = client.post(
            "/api/meshes", files={"file": ("ball6.stl", part_bytes)}
        ).json()["mesh"]["id"]
        assert _wait_mesh(client, mesh_id) == "ready"
        analyze_id = client.post(
            "/api/jobs",
            json={
                "mesh_id": mesh_id,
                "kind": "analyze",
                "config": {"demold": {"coarse_dirs": 12, "fine_dirs": 12,
                                      "coarse_grid": 24, "fine_grid": 24}},
            },
        ).json()["job"]["id"]
        assert _wait_job(client, analyze_id)["state"] == "done"
        res = client.post(
            "/api/jobs",
            json={"kind": "features", "config": {"source_job": analyze_id, "plan": {}}},
        )
        assert res.status_code == 409

    def test_a_core_run_is_served_end_to_end(self, client, part_bytes):
        """The third body has to reach the browser: preview, download, report."""
        mesh_id = client.post(
            "/api/meshes", files={"file": ("ball5.stl", part_bytes)}
        ).json()["mesh"]["id"]
        assert _wait_mesh(client, mesh_id) == "ready"

        job_id = client.post(
            "/api/jobs",
            json={
                "mesh_id": mesh_id,
                "kind": "mold",
                "config": {
                    "block_margin": 8,
                    "parting": {"grid": 60},
                    "direction": [0, 0, 1],
                    "vents": {"enabled": False},
                    "core": {"enabled": True, "wall": 2.0},
                    "core_tabs": {"count": 2},
                },
            },
        ).json()["job"]["id"]
        job = _wait_job(client, job_id)
        assert job["state"] == "done", job.get("message")

        core = job["result"]["report"]["core"]
        assert core["wall"]["median_mm"] > 0
        assert core["release"]["releases"]
        assert "core" in job["result"]["previews"]

        assert "core.stl" in job["parts"]
        stl = client.get(f"/api/jobs/{job_id}/files/core.stl")
        assert stl.status_code == 200 and len(stl.content) > 1000
        assert client.get(f"/api/jobs/{job_id}/preview/core.bin").status_code == 200

    def test_a_core_mold_can_be_re_cut_without_eroding_again(self, client, part_bytes):
        """The core is a plan like any other, and the erosion is cached.

        Nothing about where the plate's plane sits or how many dowels there are
        depends on the Minkowski difference that built the core, so an edit pays
        for the plan and not for that.
        """
        mesh_id = client.post(
            "/api/meshes", files={"file": ("ball6.stl", part_bytes)}
        ).json()["mesh"]["id"]
        assert _wait_mesh(client, mesh_id) == "ready"

        job_id = client.post(
            "/api/jobs",
            json={
                "mesh_id": mesh_id,
                "kind": "mold",
                "config": {
                    "block_margin": 8,
                    "parting": {"grid": 60},
                    "direction": [0, 0, 1],
                    "vents": {"enabled": False},
                    "core": {"enabled": True, "wall": 2.0},
                    "carrier": {"enabled": True},
                },
            },
        ).json()["job"]["id"]
        built = _wait_job(client, job_id)
        assert built["state"] == "done", built.get("message")
        plan = built["result"]["plan"]
        assert any(i["kind"] == "plate" for i in plan["items"])

        # Move the plate's plane and thin it: the edit has to come back with a
        # core, and with the plate where it was asked for.
        edited = json.loads(json.dumps(plan))
        for item in edited["items"]:
            if item["kind"] == "plate":
                item["params"]["thickness"] = 6.0
                moved = list(item["position"])
                moved[2] -= 2.0
                item["position"] = moved

        edit_id = client.post(
            "/api/jobs",
            json={
                "kind": "features",
                "config": {"source_job": job_id, "plan": edited, "verify": False},
            },
        ).json()["job"]["id"]
        done = _wait_job(client, edit_id)
        assert done["state"] == "done", done.get("message")

        core = done["result"]["report"]["core"]
        assert core["plate"]["asked_thickness_mm"] == pytest.approx(6.0)
        assert "core" in done["result"]["previews"]
        assert "core.stl" in done["parts"]
        # The erosion is the expensive step and it is not repeated.
        assert "core_erode" not in done["result"]["report"]["timings_s"]

    def test_a_plain_mold_offers_no_core(self, client, part_bytes):
        mesh_id = client.post(
            "/api/meshes", files={"file": ("ball7.stl", part_bytes)}
        ).json()["mesh"]["id"]
        assert _wait_mesh(client, mesh_id) == "ready"
        job_id = client.post(
            "/api/jobs",
            json={
                "mesh_id": mesh_id,
                "kind": "mold",
                "config": {
                    "block_margin": 8,
                    "parting": {"grid": 60},
                    "direction": [0, 0, 1],
                    "vents": {"enabled": False},
                },
            },
        ).json()["job"]["id"]
        job = _wait_job(client, job_id)
        assert job["state"] == "done", job.get("message")
        assert "core" not in job["result"]["report"]
        assert "core.stl" not in job["parts"]
        assert client.get(f"/api/jobs/{job_id}/preview/core.bin").status_code == 404

    def test_heatmap_rejects_bad_directions(self, client, part_bytes):
        mesh_id = client.post(
            "/api/meshes", files={"file": ("ball2.stl", part_bytes)}
        ).json()["mesh"]["id"]
        assert _wait_mesh(client, mesh_id) == "ready"
        for body in ({}, {"direction": [0, 0, 0]}, {"direction": [1, 2]}):
            assert client.post(f"/api/meshes/{mesh_id}/heatmap", json=body).status_code == 422

    def test_mesh_can_be_deleted(self, client, part_bytes):
        mesh_id = client.post(
            "/api/meshes", files={"file": ("ball3.stl", part_bytes)}
        ).json()["mesh"]["id"]
        assert _wait_mesh(client, mesh_id) == "ready"
        assert client.delete(f"/api/meshes/{mesh_id}").json()["deleted"]
        assert client.get(f"/api/meshes/{mesh_id}").status_code == 404
