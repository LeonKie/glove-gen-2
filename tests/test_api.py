"""HTTP surface. Geometry runs in a worker process, so these lean on polling."""

from __future__ import annotations

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
