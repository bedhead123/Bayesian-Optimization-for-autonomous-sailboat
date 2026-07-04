"""
SQLite database for storing optimization results.
Tables: designs (iteration data), calibration (mid-fi deltas),
validation (gate results).
Key exports: OptimizationDatabase
"""
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime


class OptimizationDatabase:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).resolve())
        db_parent = Path(self.db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)
        # Clean stale WAL/SHM files from aborted runs
        for sfx in ["-wal", "-shm"]:
            (db_parent / f"{Path(self.db_path).name}{sfx}").unlink(missing_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS designs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iter INTEGER NOT NULL,
                design_vector TEXT NOT NULL,
                feasible INTEGER NOT NULL DEFAULT 0,
                fom REAL,
                rt_total REAL,
                rt_wave REAL,
                rt_friction REAL,
                stability_index REAL,
                roll_period REAL,
                peak_accel REAL,
                constraint_values TEXT,
                constraint_violations TEXT,
                error_code TEXT,
                cad_stl_path TEXT,
                cad_sac_path TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                design_id INTEGER,
                iter INTEGER NOT NULL,
                rt_michlet REAL NOT NULL,
                rt_cfd REAL NOT NULL,
                delta REAL NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS validation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                design_id INTEGER NOT NULL,
                gate_name TEXT NOT NULL,
                passed INTEGER NOT NULL,
                measured_value REAL,
                threshold REAL,
                details TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_designs_iter ON designs(iter);
            CREATE INDEX IF NOT EXISTS idx_designs_feasible ON designs(feasible);
            CREATE INDEX IF NOT EXISTS idx_calibration_iter ON calibration(iter);
        """)
        self._conn.commit()

    def insert_design(self, iter_num: int, design_vector: np.ndarray,
                      feasible: bool, fom: Optional[float] = None,
                      rt_total: Optional[float] = None,
                      rt_wave: Optional[float] = None,
                      rt_friction: Optional[float] = None,
                      stability_index: Optional[float] = None,
                      roll_period: Optional[float] = None,
                      peak_accel: Optional[float] = None,
                      constraint_values: Optional[dict] = None,
                      constraint_violations: Optional[list] = None,
                      error_code: Optional[str] = None,
                      cad_stl_path: Optional[str] = None,
                      cad_sac_path: Optional[str] = None) -> int:
        self._conn.execute("""
            INSERT OR REPLACE INTO designs
                (iter, design_vector, feasible, fom, rt_total, rt_wave,
                 rt_friction, stability_index, roll_period, peak_accel,
                 constraint_values, constraint_violations, error_code,
                 cad_stl_path, cad_sac_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            iter_num,
            json.dumps(design_vector.tolist() if isinstance(design_vector, np.ndarray) else design_vector),
            int(feasible), fom, rt_total, rt_wave,
            rt_friction, stability_index, roll_period, peak_accel,
            json.dumps(constraint_values) if constraint_values else None,
            json.dumps(constraint_violations) if constraint_violations else None,
            error_code, cad_stl_path, cad_sac_path
        ))
        self._conn.commit()
        return self._conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_design(self, design_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM designs WHERE id = ?", (design_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_designs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM designs ORDER BY iter"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_feasible_designs(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM designs WHERE feasible = 1 ORDER BY fom DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_best_feasible(self) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM designs WHERE feasible = 1 ORDER BY fom DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def get_top_n(self, n: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM designs WHERE feasible = 1 ORDER BY fom DESC LIMIT ?",
            (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def store_calibration(self, design_id: int, iter_num: int,
                          rt_michlet: float, rt_cfd: float, delta: float):
        self._conn.execute("""
            INSERT INTO calibration (design_id, iter, rt_michlet, rt_cfd, delta)
            VALUES (?, ?, ?, ?, ?)
        """, (design_id, iter_num, rt_michlet, rt_cfd, delta))
        self._conn.commit()

    def get_calibrations(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM calibration ORDER BY iter"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_calibration(self) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM calibration ORDER BY iter DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def store_validation(self, design_id: int, gate_name: str,
                         passed: bool, measured_value: float,
                         threshold: float, details: Optional[str] = None):
        self._conn.execute("""
            INSERT INTO validation (design_id, gate_name, passed, measured_value, threshold, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (design_id, gate_name, int(passed), measured_value, threshold, details))
        self._conn.commit()

    def get_validation_results(self, design_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM validation WHERE design_id = ?", (design_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_iteration_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM designs").fetchone()
        return row[0]

    def close(self):
        self._conn.close()
