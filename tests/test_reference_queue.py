"""Tests for ReferenceRunner background thread and BO loop enqueue logic."""
import json
import queue
import sqlite3
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from hull_opt.surrogate import ReferenceRunner, HullOptimizer
from hull_opt.config import Config
from hull_opt.database import OptimizationDatabase
from hull_opt.low_fidelity import EvaluationResult


def _make_db(tmp_path: Path) -> tuple[str, sqlite3.Connection]:
    db_path = str(tmp_path / f"test_ref_{uuid.uuid4().hex[:8]}.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reference_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            design_id INTEGER NOT NULL,
            tool TEXT,
            status TEXT,
            max_accel_g REAL, max_pressure_pa REAL,
            final_orientation TEXT, capsized INTEGER,
            sim_time_s REAL, wall_time_s REAL, details TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_reference_runs_design
            ON reference_runs(design_id);
        CREATE TABLE IF NOT EXISTS designs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iter INTEGER NOT NULL,
            design_vector TEXT NOT NULL,
            feasible INTEGER NOT NULL DEFAULT 0,
            fom REAL, rapid_gates TEXT,
            cad_stl_path TEXT
        );
        CREATE TABLE IF NOT EXISTS corrections (
            key TEXT PRIMARY KEY, value REAL,
            updated_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    return db_path, conn


def _seed_design(conn, design_id: int, stl_path: str):
    """Insert a design row with a valid STL path so the runner doesn't skip it."""
    conn.execute(
        "INSERT OR REPLACE INTO designs(id, iter, design_vector, feasible, fom, cad_stl_path) "
        "VALUES(?, 0, ?, 1, 1.0, ?)",
        (design_id, json.dumps([0.5] * 17), stl_path)
    )
    conn.commit()


def _make_base_config(tmp_path):
    cfg = MagicMock(spec=Config)
    cfg.optimization = MagicMock()
    cfg.optimization.n_initial = 0
    cfg.optimization.n_iter = 0
    cfg.optimization.lhs_min = 0
    cfg.optimization.lhs_increment = 10
    cfg.optimization.lhs_max = 0
    cfg.optimization.lhs_seed = 42
    cfg.optimization.convergence_threshold = 0.01
    cfg.optimization.min_iterations = 5
    cfg.optimization.num_restarts = 5
    cfg.optimization.raw_samples = 20
    cfg.optimization.gp_jitter = 1e-6
    cfg.calibration = MagicMock()
    cfg.calibration.frequency = 999
    cfg.weights = MagicMock()
    cfg.weights.w1 = 1.0
    cfg.bounds = MagicMock()
    cfg.bounds.dim = 2
    cfg.bounds.as_array = lambda: [(-1.0, 1.0), (-1.0, 1.0)]
    cfg.fixed = MagicMock()
    cfg.fixed.LWL = 2.4
    cfg.paths = MagicMock()
    cfg.paths.output_dir = str(tmp_path)
    type(cfg).reference = PropertyMock(return_value=None)
    return cfg


def _mock_eval_result(fom=1.0, feasible=True, **kw):
    r = EvaluationResult()
    r.fom = fom
    r.feasible = feasible
    r.rt_total = 5.0
    r.rt_wave = 3.0
    r.rt_friction = 2.0
    r.stability_index = 1.0
    r.roll_period = 4.0
    r.peak_accel = 5.0
    r.gm = 0.1
    r.cg_z = -0.05
    r.eq_heel_deg = 2.0
    r.righting_energy = 100.0
    r.helm_fwd_deg = 1.0
    r.helm_aft_deg = -0.5
    r.helm_combined_deg = 0.5
    r.physical_params = {}
    r.constraint_values = {"righting_energy": 100.0}
    r.constraint_violations = []
    r.error_code = None
    r.cad_stl_path = None
    r.cad_sac_path = None
    for k, v in kw.items():
        setattr(r, k, v)
    return r


class TestReferenceRunner:
    def _make_runner(self, tmp_path, db_path, q=None):
        if q is None:
            q = queue.Queue()
        runner = ReferenceRunner(
            config=MagicMock(),
            db_path=db_path,
            output_dir=tmp_path,
            gpu_lock_path=str(tmp_path / "gpu.lock"),
            reference_queue=q,
        )
        return runner

    def test_runner_skips_if_already_run(self, tmp_path, monkeypatch):
        db_path, conn = _make_db(tmp_path)
        conn.execute(
            "INSERT INTO reference_runs(design_id,tool,status) VALUES(?,?,?)",
            (42, "dualsphysics", "OK")
        )
        conn.commit()
        conn.close()
        q = queue.Queue()
        runner = self._make_runner(tmp_path, db_path, q)
        storm_mock = MagicMock()
        monkeypatch.setattr("hull_opt.surrogate.run_reference_storm", storm_mock)
        q.put({"design_id": 42, "design_vector": [0.1]*17, "iteration": 5})
        q.put(None)
        runner.start()
        runner.join(timeout=5)
        storm_mock.assert_not_called()
        conn2 = sqlite3.connect(db_path)
        conn2.row_factory = sqlite3.Row
        row = conn2.execute(
            "SELECT status FROM reference_runs WHERE design_id=?", (42,)
        ).fetchone()
        assert row["status"] == "OK"
        conn2.close()

    def test_runner_processes_queue(self, tmp_path, monkeypatch):
        db_path, conn = _make_db(tmp_path)
        stl_file = tmp_path / "ref_ds_7" / "hull.stl"
        stl_file.parent.mkdir(parents=True, exist_ok=True)
        stl_file.write_text("stub")
        _seed_design(conn, 7, str(stl_file))
        conn.close()
        q = queue.Queue()
        runner = self._make_runner(tmp_path, db_path, q)
        storm_mock = MagicMock(return_value={
            "tool": "dualsphysics",
            "status": "OK",
            "max_accel_g": 12.5,
            "max_pressure_pa": 85000.0,
            "final_orientation": [0.0, 0.0, 5.0],
            "capsized": False,
            "sim_time_s": 10.0,
            "wall_time_s": 120.0,
            "details": "ran gpu solver",
        })
        monkeypatch.setattr("hull_opt.surrogate.run_reference_storm", storm_mock)

        def _mock_flock(fd, op):
            pass
        monkeypatch.setattr("fcntl.flock", _mock_flock)

        dv = [0.5] * 17
        q.put({"design_id": 7, "design_vector": dv, "iteration": 3})
        q.put(None)
        runner.start()
        runner.join(timeout=5)
        storm_mock.assert_called_once()
        conn2 = sqlite3.connect(db_path)
        conn2.row_factory = sqlite3.Row
        row = conn2.execute(
            "SELECT * FROM reference_runs WHERE design_id=?", (7,)
        ).fetchone()
        assert row is not None
        assert row["status"] == "OK"
        assert row["max_accel_g"] == pytest.approx(12.5)
        assert row["max_pressure_pa"] == pytest.approx(85000.0)
        assert row["capsized"] == 0
        conn2.close()

    def test_runner_timeout_handling(self, tmp_path, monkeypatch):
        db_path, conn = _make_db(tmp_path)
        stl_file = tmp_path / "ref_ds_99" / "hull.stl"
        stl_file.parent.mkdir(parents=True, exist_ok=True)
        stl_file.write_text("stub")
        _seed_design(conn, 99, str(stl_file))
        conn.close()
        q = queue.Queue()
        runner = self._make_runner(tmp_path, db_path, q)
        storm_mock = MagicMock(side_effect=RuntimeError("Solver crashed"))
        monkeypatch.setattr("hull_opt.surrogate.run_reference_storm", storm_mock)

        def _mock_flock(fd, op):
            pass
        monkeypatch.setattr("fcntl.flock", _mock_flock)

        q.put({"design_id": 99, "design_vector": [0.1]*17, "iteration": 10})
        q.put(None)
        runner.start()
        runner.join(timeout=5)
        conn2 = sqlite3.connect(db_path)
        conn2.row_factory = sqlite3.Row
        row = conn2.execute(
            "SELECT * FROM reference_runs WHERE design_id=?", (99,)
        ).fetchone()
        assert row is not None
        assert row["status"] == "FAILED"
        assert "Solver crashed" in (row["details"] or "")
        conn2.close()

    def test_runner_gpu_lock_skips_without_row(self, tmp_path, monkeypatch):
        """Lock-busy must NOT write a SKIPPED row: the design stays eligible
        and is retried on the next enqueue (a transient collision must not
        permanently burn the design's reference storm)."""
        db_path, conn = _make_db(tmp_path)
        conn.close()
        q = queue.Queue()
        runner = self._make_runner(tmp_path, db_path, q)
        storm_mock = MagicMock()
        monkeypatch.setattr("hull_opt.surrogate.run_reference_storm", storm_mock)

        def _mock_flock_block(fd, op):
            raise BlockingIOError(11, "Resource temporarily unavailable")
        monkeypatch.setattr("fcntl.flock", _mock_flock_block)

        q.put({"design_id": 55, "design_vector": [0.1]*17, "iteration": 7})
        q.put(None)
        runner.start()
        runner.join(timeout=5)
        storm_mock.assert_not_called()
        conn2 = sqlite3.connect(db_path)
        conn2.row_factory = sqlite3.Row
        row = conn2.execute(
            "SELECT * FROM reference_runs WHERE design_id=?", (55,)
        ).fetchone()
        assert row is None, "no SKIPPED row may be written on lock-busy"
        conn2.close()

    def test_runner_stop_via_sentinel(self, tmp_path):
        db_path, conn = _make_db(tmp_path)
        conn.close()
        q = queue.Queue()
        runner = self._make_runner(tmp_path, db_path, q)
        q.put(None)
        runner.start()
        runner.join(timeout=5)
        assert not runner.is_alive()

    def test_runner_sentinel_and_stop_event(self, tmp_path):
        db_path, conn = _make_db(tmp_path)
        conn.close()
        q = queue.Queue()
        runner = self._make_runner(tmp_path, db_path, q)
        runner.start()
        runner.stop()
        q.put(None)
        runner.join(timeout=5)
        assert not runner.is_alive()


class TestEnqueueInBOLoop:
    def test_enqueue_every_n_iterations(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "opt.db")
        db = OptimizationDatabase(db_path)
        cfg = _make_base_config(tmp_path)
        cfg.optimization.n_iter = 30
        cfg.optimization.convergence_threshold = 1e-6
        cfg.optimization.min_iterations = 3

        optimizer = HullOptimizer(cfg, db)
        optimizer._pool_type = "serial"
        optimizer._process_pool = None
        optimizer._ray_remote = None
        q = queue.Queue()
        optimizer._reference_queue = q
        optimizer._reference_thread = MagicMock()
        optimizer._reference_thread.is_alive.return_value = True

        monkeypatch.setattr(optimizer, "_eval_one", lambda dv, **kw: _mock_eval_result())
        monkeypatch.setattr(optimizer, "_propose_candidate",
                            lambda: np.array([0.0, 0.0]))

        optimizer._bo_loop(0)

        items = []
        while not q.empty():
            try:
                items.append(q.get_nowait())
                q.task_done()
            except queue.Empty:
                break

        eligible = [i for i in items if i["iteration"] % 10 == 0]
        assert len(eligible) >= 1
        db.close()

    def test_no_enqueue_when_thread_dead(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "opt2.db")
        db = OptimizationDatabase(db_path)
        cfg = _make_base_config(tmp_path)
        cfg.optimization.n_iter = 5
        cfg.optimization.convergence_threshold = 1e-6
        cfg.optimization.min_iterations = 3

        optimizer = HullOptimizer(cfg, db)
        optimizer._pool_type = "serial"
        optimizer._process_pool = None
        optimizer._ray_remote = None
        q = queue.Queue()
        optimizer._reference_queue = q
        optimizer._reference_thread = MagicMock()
        optimizer._reference_thread.is_alive.return_value = False

        monkeypatch.setattr(optimizer, "_eval_one", lambda dv, **kw: _mock_eval_result())
        monkeypatch.setattr(optimizer, "_propose_candidate",
                            lambda: np.array([0.0, 0.0]))

        optimizer._bo_loop(0)

        assert q.empty()
        db.close()
