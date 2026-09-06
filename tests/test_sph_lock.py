"""Tests for the GPU-lock self-deadlock fix.

Bug A: ReferenceRunner thread holds an EX flock on <output_dir>/gpu.lock and
then re-opens the same file inside run_reference_storm; two file descriptions
in the same process conflict, so every storm was falsely "SKIPPED". The
held_lock_fd contract lets the caller hand its already-held fd through.
Bug B: _clean_slate never removed the stale gpu.lock.
"""
import fcntl
import os
import numpy as np
import pytest

from hull_opt.sph_gates import run_reference_storm


@pytest.fixture
def config():
    from hull_opt.config import load_config
    return load_config("config.yaml")


@pytest.fixture
def settings():
    from hull_opt.sph_gates import ReferenceSettings
    return ReferenceSettings()


@pytest.fixture
def design_vector():
    return np.array([2.4, 0.50, 0.25, 0.60, 0.75, 45.0, 1.0, 0.20, 0.0012,
                     0.40, 0.20, 12.0, 15.0, 0.15, 0.01, 0.50, 0.42],
                    dtype=np.float64)


class TestHeldLockFd:
    def test_held_fd_skips_reflock(self, config, settings, design_vector,
                                   tmp_path, monkeypatch):
        def _no_solver(*args, **kwargs):
            raise FileNotFoundError("No DualSPHysics solver found (test)")
        monkeypatch.setattr("hull_opt.sph_gates.find_solver", _no_solver)
        monkeypatch.setattr(os.path, "exists", lambda p: True)

        lock_file = tmp_path / "held.lock"
        fd = open(lock_file, "w")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = run_reference_storm(
                design_vector, config, str(tmp_path / "storm_held"),
                "test_held", settings=settings, stl_path="/tmp/hull.stl",
                gpu_lock=str(tmp_path / "no_such_dir" / "gpu.lock"),
                held_lock_fd=fd.fileno())
        finally:
            fd.close()
        assert result["status"] != "SKIPPED"
        assert "GPU locked" not in str(result.get("details", ""))
        assert result["status"] == "FAILED"
        assert "No DualSPHysics solver found" in result["details"]

    def test_held_fd_ignores_invalid_lock_path(self, config, settings,
                                               design_vector, tmp_path,
                                               monkeypatch):
        def _no_solver(*args, **kwargs):
            raise FileNotFoundError("No DualSPHysics solver found (test)")
        monkeypatch.setattr("hull_opt.sph_gates.find_solver", _no_solver)
        monkeypatch.setattr(os.path, "exists", lambda p: True)

        lock_file = tmp_path / "held2.lock"
        fd = open(lock_file, "w")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = run_reference_storm(
                design_vector, config, str(tmp_path / "storm_held2"),
                "test_held2", settings=settings, stl_path="/tmp/hull.stl",
                gpu_lock="/nonexistent-dir/gpu.lock",
                held_lock_fd=fd.fileno())
        finally:
            fd.close()
        assert result["status"] != "SKIPPED"
        assert "GPU locked" not in str(result.get("details", ""))

    def test_without_held_fd_second_open_skips(self, config, settings,
                                               design_vector, tmp_path,
                                               monkeypatch):
        monkeypatch.setattr("hull_opt.sph_gates.find_solver",
                            lambda *a, **k: ("/fake/solver", "cpu",
                                             os.environ.copy()))
        monkeypatch.setattr(os.path, "exists", lambda p: True)
        lock_path = tmp_path / "gpu.lock"
        fd = open(lock_path, "w")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = run_reference_storm(
                design_vector, config, str(tmp_path / "storm_reflock"),
                "test_reflock", settings=settings, stl_path="/tmp/hull.stl",
                gpu_lock=str(lock_path))
        finally:
            fd.close()
        assert result["status"] == "SKIPPED"
        assert "GPU locked" in result["details"]

    def test_without_lock_args_runs(self, config, settings, design_vector,
                                    tmp_path, monkeypatch):
        def _no_solver(*args, **kwargs):
            raise FileNotFoundError("No DualSPHysics solver found (test)")
        monkeypatch.setattr("hull_opt.sph_gates.find_solver", _no_solver)
        monkeypatch.setattr(os.path, "exists", lambda p: True)
        result = run_reference_storm(
            design_vector, config, str(tmp_path / "storm_plain"),
            "test_plain", settings=settings, stl_path="/tmp/hull.stl")
        assert result["status"] == "FAILED"
        assert "GPU locked" not in str(result.get("details", ""))


class TestCleanSlate:
    def test_removes_gpu_lock_and_db(self, tmp_path):
        from run_optimization import _clean_slate
        out = tmp_path / "output"
        out.mkdir()
        (out / "gpu.lock").write_text("stale lock")
        (out / ".run_mode").write_text("full")
        (out / "designs_abc").mkdir()
        db_path = tmp_path / "optimization.db"
        db_path.write_bytes(b"")
        (tmp_path / "optimization.db-wal").write_bytes(b"")
        (tmp_path / "optimization.db-shm").write_bytes(b"")

        _clean_slate(out, db_path)

        assert not (out / "gpu.lock").exists()
        assert not (out / ".run_mode").exists()
        assert not (out / "designs_abc").exists()
        assert not db_path.exists()
        assert not (tmp_path / "optimization.db-wal").exists()
        assert not (tmp_path / "optimization.db-shm").exists()

    def test_clean_slate_idempotent(self, tmp_path):
        from run_optimization import _clean_slate
        out = tmp_path / "output2"
        out.mkdir()
        db_path = tmp_path / "optimization2.db"
        _clean_slate(out, db_path)
        assert not (out / "gpu.lock").exists()
        assert not db_path.exists()

    def test_clean_slate_backs_up_campaign_db(self, tmp_path):
        # Bug #170: a DB holding designs must be moved to a timestamped
        # backup, never unlinked — wipes must be recoverable.
        import sqlite3
        from run_optimization import _clean_slate
        out = tmp_path / "output3"
        out.mkdir()
        db_path = tmp_path / "optimization3.db"
        c = sqlite3.connect(str(db_path))
        c.execute("create table designs (id integer primary key, fom real)")
        c.execute("insert into designs values (1, 3.5)")
        c.commit()
        c.close()
        _clean_slate(out, db_path)
        assert not db_path.exists()
        backups = sorted(out.glob(".backup_*"))
        assert len(backups) == 1
        rescued = sqlite3.connect(f"file:{backups[0] / 'optimization3.db'}?mode=ro", uri=True)
        assert rescued.execute("select count(*) from designs").fetchone()[0] == 1
        rescued.close()
