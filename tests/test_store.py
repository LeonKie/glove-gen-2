"""The persistence layer: the thing v1 did not have."""

from __future__ import annotations

import time

import pytest

from server.store import Store, read_json, write_json_atomic


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "store", ttl_seconds=3600.0)


class TestMeshes:
    def test_create_and_read_back(self, store):
        meta = store.create_mesh("hand.stl", b"solid\n")
        assert meta["state"] == "preparing"
        again = store.get_mesh(meta["id"])
        assert again == meta
        assert (store.mesh_dir(meta["id"]) / "original.stl").read_bytes() == b"solid\n"

    def test_update_merges_fields(self, store):
        meta = store.create_mesh("a.stl", b"x")
        store.update_mesh(meta["id"], state="ready", stats={"faces": 10})
        got = store.get_mesh(meta["id"])
        assert got["state"] == "ready"
        assert got["stats"] == {"faces": 10}
        assert got["name"] == "a.stl"  # untouched

    def test_list_is_newest_first(self, store):
        a = store.create_mesh("a.stl", b"x")
        time.sleep(0.01)
        b = store.create_mesh("b.stl", b"y")
        ids = [m["id"] for m in store.list_meshes()]
        assert ids.index(b["id"]) < ids.index(a["id"])

    def test_delete_removes_dependent_jobs(self, store):
        mesh = store.create_mesh("a.stl", b"x")
        job = store.create_job("mold", mesh["id"])
        assert store.delete_mesh(mesh["id"])
        assert store.get_mesh(mesh["id"]) is None
        assert store.get_job(job["id"]) is None

    def test_missing_mesh_reads_as_none(self, store):
        assert store.get_mesh("0123abcd") is None
        assert store.update_mesh("0123abcd", state="ready") is None


class TestJobs:
    def test_lifecycle(self, store):
        mesh = store.create_mesh("a.stl", b"x")
        job = store.create_job("mold", mesh["id"], {"block_margin": 9})
        assert job["state"] == "queued"
        store.update_job(job["id"], state="running", progress=0.5, message="cutting")
        got = store.get_job(job["id"])
        assert (got["state"], got["progress"], got["message"]) == ("running", 0.5, "cutting")
        assert got["config"] == {"block_margin": 9}

    def test_filter_by_mesh(self, store):
        m1 = store.create_mesh("a.stl", b"x")
        m2 = store.create_mesh("b.stl", b"y")
        store.create_job("mold", m1["id"])
        store.create_job("mold", m2["id"])
        assert len(store.list_jobs(m1["id"])) == 1
        assert len(store.list_jobs()) == 2


class TestHousekeeping:
    def test_reap_marks_dead_workers_interrupted(self, store):
        """A restart must not leave a job spinning forever."""
        mesh = store.create_mesh("a.stl", b"x")
        job = store.create_job("mold", mesh["id"])
        # pid 1 exists but is not ours; use an implausible pid that is free.
        store.update_job(job["id"], state="running", pid=2**22 - 1)
        assert store.reap_orphans() >= 1
        got = store.get_job(job["id"])
        assert got["state"] == "interrupted"
        assert "gone" in got["message"]

    def test_reap_leaves_live_workers_alone(self, store):
        import os

        mesh = store.create_mesh("a.stl", b"x")
        job = store.create_job("mold", mesh["id"])
        store.update_job(job["id"], state="running", pid=os.getpid())
        store.update_mesh(mesh["id"], state="ready")
        store.reap_orphans()
        assert store.get_job(job["id"])["state"] == "running"

    def test_reap_fails_stuck_uploads(self, store):
        mesh = store.create_mesh("a.stl", b"x")  # state == preparing
        store.reap_orphans()
        assert store.get_mesh(mesh["id"])["state"] == "failed"

    def test_ttl_is_actually_enforced(self, store):
        mesh = store.create_mesh("a.stl", b"x")
        job = store.create_job("mold", mesh["id"])
        store.update_mesh(mesh["id"], created=time.time() - 10_000)
        store.update_job(job["id"], created=time.time() - 10_000)
        assert store.purge_expired() >= 2
        assert store.get_mesh(mesh["id"]) is None
        assert store.get_job(job["id"]) is None

    def test_ttl_keeps_fresh_records(self, store):
        mesh = store.create_mesh("a.stl", b"x")
        assert store.purge_expired() == 0
        assert store.get_mesh(mesh["id"]) is not None

    def test_ttl_zero_disables_purging(self, store):
        mesh = store.create_mesh("a.stl", b"x")
        store.update_mesh(mesh["id"], created=0.0)
        assert store.purge_expired(0) == 0
        assert store.get_mesh(mesh["id"]) is not None

    def test_usage_summary(self, store):
        store.create_mesh("a.stl", b"x" * 100)
        usage = store.usage()
        assert usage["meshes"] == 1
        assert usage["bytes"] >= 100


class TestSafety:
    @pytest.mark.parametrize("bad", ["../etc", "a/b", "", "a b", "..", "a.b"])
    def test_ids_that_could_escape_are_refused(self, store, bad):
        with pytest.raises(ValueError):
            store.mesh_dir(bad)

    @pytest.mark.parametrize("good", ["abc123", "a-b_c", "DEADBEEF"])
    def test_normal_ids_are_fine(self, store, good):
        assert store.mesh_dir(good).name == good


class TestAtomicJson:
    def test_write_then_read(self, tmp_path):
        p = tmp_path / "x" / "meta.json"
        write_json_atomic(p, {"a": 1})
        assert read_json(p) == {"a": 1}

    def test_no_temp_files_left_behind(self, tmp_path):
        p = tmp_path / "meta.json"
        write_json_atomic(p, {"a": 1})
        assert list(tmp_path.iterdir()) == [p]

    def test_missing_and_corrupt_read_as_none(self, tmp_path):
        assert read_json(tmp_path / "nope.json") is None
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert read_json(bad) is None
