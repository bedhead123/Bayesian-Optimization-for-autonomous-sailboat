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
        # Clean stale WAL/SHM files from aborted runs — but ONLY when no
        # other process holds the DB (Bug #165: a second run wiped output/
        # while the first was mid-write, then unlinked live WAL files).
        import fcntl as _fc
        _probe = None
        try:
            _probe = open(self.db_path + ".gclock", "w")
            _fc.flock(_probe, _fc.LOCK_EX | _fc.LOCK_NB)
            for sfx in ["-wal", "-shm"]:
                (db_parent / f"{Path(self.db_path).name}{sfx}").unlink(missing_ok=True)
            _fc.flock(_probe, _fc.LOCK_UN)
        except BlockingIOError:
            pass
        finally:
            if _probe is not None:
                try:
                    _probe.close()
                except Exception:
                    pass
        self._conn = sqlite3.connect(self.db_path, timeout=30.0,
                                     check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        # Scrub stale 'running' calibration attempts from crashed runs: the
        # selector excludes 'running' rows, so a crash mid-calibration would
        # otherwise block that design from every future attempt.
        self._conn.execute(
            "UPDATE calibration_attempts SET status='failed' WHERE status='running'"
        )
        self._conn.commit()

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
                righting_energy REAL,
                gm REAL,
                cg_z REAL,
                eq_heel_deg REAL,
                helm_fwd_deg REAL,
                helm_aft_deg REAL,
                helm_combined_deg REAL,
                physical_params TEXT,
                constraint_values TEXT,
                constraint_violations TEXT,
                error_code TEXT,
                cad_stl_path TEXT,
                cad_sac_path TEXT,
                avs_deg REAL,
                capsize_margin REAL,
                storm_peak_accel_g REAL,
                roll_sigma_deg REAL,
                parametric_roll INTEGER,
                slam_pressure_pa REAL,
                inverted_pressure_pa REAL,
                storm_wind_heel_deg REAL,
                rapid_gates TEXT,
                gate_margins TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                design_id INTEGER,
                iter INTEGER NOT NULL,
                rt_michlet REAL NOT NULL,
                rt_cfd REAL NOT NULL,
                delta REAL NOT NULL,
                factor REAL NOT NULL DEFAULT 1.0,
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

            CREATE TABLE IF NOT EXISTS calibration_attempts (
                design_id INTEGER NOT NULL,
                iter INTEGER NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('running','ok','failed','timeout','invalid')),
                factor REAL,
                ts TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (design_id, iter)
            );

            CREATE TABLE IF NOT EXISTS campaign (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS bem_runs (
                bem_hash TEXT NOT NULL,
                design_id INTEGER,
                omega REAL,
                heading_deg REAL,
                heave_rao REAL,
                pitch_rao REAL,
                roll_rao REAL,
                wall_time_s REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS reference_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                design_id INTEGER NOT NULL,
                tool TEXT,
                status TEXT,
                max_accel_g REAL,
                max_pressure_pa REAL,
                final_orientation TEXT,
                capsized INTEGER,
                sim_time_s REAL,
                wall_time_s REAL,
                details TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS corrections (
                key TEXT PRIMARY KEY,
                value REAL,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_designs_iter ON designs(iter);
            CREATE INDEX IF NOT EXISTS idx_designs_feasible ON designs(feasible);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_calibration_design_iter ON calibration(design_id, iter);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_validation_design_gate ON validation(design_id, gate_name);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_bem_runs_hash ON bem_runs(bem_hash, omega, heading_deg);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_reference_runs_design ON reference_runs(design_id);
        """)
        # Migration for pre-factor databases
        try:
            self._conn.execute("ALTER TABLE calibration ADD COLUMN factor REAL NOT NULL DEFAULT 1.0")
        except Exception:
            pass
        # Migration: per-design drag factor under which the CURRENT stored
        # fom/rt_total were computed (updated by rescore_foms whenever the
        # calibration factor changes).
        try:
            self._conn.execute("ALTER TABLE designs ADD COLUMN drag_factor REAL")
        except Exception:
            pass
        # Migration: gate status classification (PASS/FAIL/CRASH/TIMEOUT) —
        # a solver crash or timeout is not a gate FAIL.
        try:
            self._conn.execute("ALTER TABLE validation ADD COLUMN status TEXT")
        except Exception:
            pass
        # Migration: per-design results columns (additive; fresh DBs already have them)
        for col in ("righting_energy", "gm", "cg_z", "eq_heel_deg",
                    "helm_fwd_deg", "helm_aft_deg", "helm_combined_deg",
                    "lead_pct_lwl", "T_over_L", "T_total_m",
                    "physical_params"):
            try:
                self._conn.execute(f"ALTER TABLE designs ADD COLUMN {col} REAL"
                                   if col != "physical_params"
                                   else "ALTER TABLE designs ADD COLUMN physical_params TEXT")
            except Exception:
                pass
        for col in ("avs_deg", "capsize_margin", "storm_peak_accel_g",
                    "roll_sigma_deg", "slam_pressure_pa",
                    "inverted_pressure_pa", "storm_wind_heel_deg"):
            try:
                self._conn.execute(f"ALTER TABLE designs ADD COLUMN {col} REAL")
            except Exception:
                pass
        # Migration: mission-FoM rescore columns (Bug #171 follow-up: the
        # CSV "lagged" because rescores lived only in chat). Campaign fom
        # is NEVER overwritten — mission_fom sits alongside with provenance.
        for col in ("mission_fom", "mission_drive", "gust_margin",
                    "heavy_leeway_deg", "draft_logistics_cost"):
            try:
                self._conn.execute(f"ALTER TABLE designs ADD COLUMN {col} REAL")
            except Exception:
                pass
        try:
            self._conn.execute("ALTER TABLE designs ADD COLUMN parametric_roll INTEGER")
        except Exception:
            pass
        for col in ("rapid_gates", "gate_margins"):
            try:
                self._conn.execute(f"ALTER TABLE designs ADD COLUMN {col} TEXT")
            except Exception:
                pass
        # Balance 360° polar columns (system-level whole-boat balance)
        for col in ("balance_worst_heel_ops", "balance_worst_heel_storm",
                    "balance_worst_leeway_ops", "balance_mean_drive", "balance_vmg_up"):
            try:
                self._conn.execute(f"ALTER TABLE designs ADD COLUMN {col} REAL")
            except Exception:
                pass
        # Migration: allow 'invalid' status in calibration_attempts (B2:
        # sub-friction Rt or non-steady trace). SQLite cannot ALTER a CHECK
        # constraint, so recreate the table.
        try:
            _probe = self._conn.execute(
                "SELECT status FROM calibration_attempts LIMIT 0"
            )
            # Recreate only when the table exists and the CHECK is the old one
            _sql = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' "
                "AND name='calibration_attempts'"
            ).fetchone()
            if _sql and "'invalid'" not in (_sql[0] or ""):
                self._conn.execute(
                    "ALTER TABLE calibration_attempts RENAME TO calibration_attempts_old"
                )
                self._conn.execute("""
                    CREATE TABLE calibration_attempts (
                        design_id INTEGER PRIMARY KEY,
                        iter INTEGER NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('running','ok','failed','timeout','invalid')),
                        factor REAL,
                        ts TEXT DEFAULT (datetime('now'))
                    )
                """)
                self._conn.execute("""
                    INSERT OR REPLACE INTO calibration_attempts
                        (design_id, iter, status, factor, ts)
                    SELECT design_id, iter, status, factor, ts
                    FROM calibration_attempts_old
                """)
                self._conn.execute("DROP TABLE calibration_attempts_old")
        except Exception:
            pass
        self._conn.commit()

    def insert_design(self, iter_num: int, design_vector: np.ndarray,
                      feasible: bool, fom: Optional[float] = None,
                      rt_total: Optional[float] = None,
                      rt_wave: Optional[float] = None,
                      rt_friction: Optional[float] = None,
                      stability_index: Optional[float] = None,
                      roll_period: Optional[float] = None,
                      peak_accel: Optional[float] = None,
                      righting_energy: Optional[float] = None,
                      gm: Optional[float] = None,
                      cg_z: Optional[float] = None,
                      eq_heel_deg: Optional[float] = None,
                      helm_fwd_deg: Optional[float] = None,
                      helm_aft_deg: Optional[float] = None,
                      helm_combined_deg: Optional[float] = None,
                      lead_pct_lwl: Optional[float] = None,
                      T_over_L: Optional[float] = None,
                      T_total_m: Optional[float] = None,
                      physical_params: Optional[dict] = None,
                      constraint_values: Optional[dict] = None,
                      constraint_violations: Optional[list] = None,
                      error_code: Optional[str] = None,
                      cad_stl_path: Optional[str] = None,
                      cad_sac_path: Optional[str] = None,
                      drag_factor: Optional[float] = None) -> int:
        self._conn.execute("""
            INSERT OR REPLACE INTO designs
                (iter, design_vector, feasible, fom, rt_total, rt_wave,
                 rt_friction, stability_index, roll_period, peak_accel,
                 righting_energy, gm, cg_z, eq_heel_deg,
                 helm_fwd_deg, helm_aft_deg, helm_combined_deg,
                 lead_pct_lwl, T_over_L, T_total_m,
                 physical_params,
                 constraint_values, constraint_violations, error_code,
                 cad_stl_path, cad_sac_path, drag_factor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            iter_num,
            json.dumps(design_vector.tolist() if isinstance(design_vector, np.ndarray) else design_vector),
            int(feasible), fom, rt_total, rt_wave,
            rt_friction, stability_index, roll_period, peak_accel,
            righting_energy, gm, cg_z, eq_heel_deg,
            helm_fwd_deg, helm_aft_deg, helm_combined_deg,
            lead_pct_lwl, T_over_L, T_total_m,
            json.dumps(physical_params) if physical_params else None,
            json.dumps(constraint_values) if constraint_values else None,
            json.dumps(constraint_violations) if constraint_violations else None,
            error_code, cad_stl_path, cad_sac_path, drag_factor
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

    def get_best_feasible_uncalibrated(self, retry_failed: bool = False,
                                       fallback_to_least_infeasible: bool = False) -> Optional[dict]:
        """Best feasible design without a (blocking) calibration attempt.

        With ``retry_failed=False`` any prior attempt (ok/failed/timeout/running)
        excludes the design: a 4-h run that already failed must not be re-picked.
        With ``retry_failed=True`` only 'failed' attempts may be retried; 'ok',
        'timeout' and 'running' (possibly still in flight) still exclude.

        With ``fallback_to_least_infeasible=True``, when no feasible design
        exists, fall back to the least-infeasible evaluated design (fewest
        constraint violations, then highest fom, then lowest id). Failed
        evaluations (non-NULL error_code) are never eligible.
        """
        if retry_failed:
            excl = "AND ca.status IN ('ok', 'timeout', 'running', 'invalid')"
        else:
            excl = ""
        row = self._conn.execute(f"""
            SELECT d.* FROM designs d
            WHERE d.feasible = 1
            AND NOT EXISTS (
                SELECT 1 FROM calibration_attempts ca
                WHERE ca.design_id = d.id {excl}
            )
            ORDER BY d.fom DESC LIMIT 1
        """).fetchone()
        if row is None and fallback_to_least_infeasible:
            rows = self._conn.execute(f"""
                SELECT d.* FROM designs d
                WHERE d.feasible = 0
                AND d.error_code IS NULL
                AND NOT EXISTS (
                    SELECT 1 FROM calibration_attempts ca
                    WHERE ca.design_id = d.id {excl}
                )
            """).fetchall()
            if rows:
                def _violations(r):
                    raw = r["constraint_violations"]
                    if not raw:
                        return 0
                    try:
                        return len(json.loads(raw))
                    except Exception:
                        return 0
                best = min(rows, key=lambda r: (_violations(r), -float(r["fom"] or 0.0), r["id"]))
                return dict(best)
        return dict(row) if row else None

    def mark_calibration_attempt(self, design_id: int, iter_num: int,
                                 status: str, factor: Optional[float] = None):
        self._conn.execute("""
            INSERT OR REPLACE INTO calibration_attempts (design_id, iter, status, factor)
            VALUES (?, ?, ?, ?)
        """, (design_id, iter_num, status, factor))
        self._conn.commit()

    def get_calibration_attempt(self, design_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM calibration_attempts WHERE design_id = ?", (design_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_top_n(self, n: int) -> list[dict]:
        # Fail-closed selection: exclude designs that errored during
        # evaluation (error_code set, e.g. E_FOM) and any non-finite FoM
        # (SQLite stores NaN as NULL; ±Inf survives the round-trip).
        rows = self._conn.execute(
            "SELECT * FROM designs WHERE feasible = 1 AND error_code IS NULL "
            "ORDER BY fom DESC"
        ).fetchall()
        top = []
        for r in rows:
            fom = r["fom"]
            if fom is None:
                continue
            try:
                if not np.isfinite(float(fom)):
                    continue
            except (TypeError, ValueError):
                continue
            top.append(dict(r))
            if len(top) >= n:
                break
        return top

    def store_calibration(self, design_id: int, iter_num: int,
                          rt_michlet: float, rt_cfd: float, delta: float,
                          factor: float = 1.0):
        self._conn.execute("""
            INSERT OR REPLACE INTO calibration (design_id, iter, rt_michlet, rt_cfd, delta, factor)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (design_id, iter_num, rt_michlet, rt_cfd, delta, factor))
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

    def rescore_foms(self, factor: float, w1: float = 1.0, min_rt: float = 0.5) -> int:
        """Recompute rt_total and FoM for all stored designs under a new
        drag factor.

        The drag term in the FoM is ``w1 / max(min_rt, Rt)`` with
        ``Rt = max(Rf*0.5, Rf + Rw*factor)``; every other FoM term
        (stability, self-righting, accelerations, penalties) is
        factor-independent. rt_wave/rt_friction are stored unscaled, so the
        rescale is exact and needs no knowledge of the factor each design
        was originally evaluated under.

        Returns the number of designs rescored.
        """
        rows = self._conn.execute(
            "SELECT id, fom, rt_total, rt_wave, rt_friction FROM designs "
            "WHERE feasible = 1 AND error_code IS NULL "
            "AND fom IS NOT NULL AND rt_total IS NOT NULL "
            "AND rt_wave IS NOT NULL AND rt_friction IS NOT NULL"
        ).fetchall()
        n = 0
        for r in rows:
            try:
                rf = float(r["rt_friction"])
                rw = float(r["rt_wave"])
                rt_old = float(r["rt_total"])
                if not (np.isfinite(rf) and np.isfinite(rw) and np.isfinite(rt_old) and rt_old > 0):
                    continue
                rt_new = max(rf * 0.5, rf + rw * max(0.0, float(factor)))
                if not np.isfinite(rt_new) or rt_new <= 0:
                    continue
                fom_old = float(r["fom"])
                if not np.isfinite(fom_old):
                    continue
                fom_new = fom_old + w1 * (1.0 / max(min_rt, rt_new)
                                         - 1.0 / max(min_rt, rt_old))
                if not np.isfinite(fom_new):
                    continue
                self._conn.execute(
                    "UPDATE designs SET rt_total = ?, fom = ?, drag_factor = ? WHERE id = ?",
                    (rt_new, fom_new, factor, r["id"]),
                )
                n += 1
            except (TypeError, ValueError):
                continue
        self._conn.commit()
        return n

    def store_validation(self, design_id: int, gate_name: str,
                         passed: bool, measured_value: float,
                         threshold: float, details: Optional[str] = None,
                         status: Optional[str] = None):
        if status is None:
            status = "PASS" if passed else "FAIL"
        self._conn.execute("""
            INSERT OR REPLACE INTO validation (design_id, gate_name, passed, measured_value, threshold, details, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (design_id, gate_name, int(passed), measured_value, threshold, details, status))
        self._conn.commit()

    def get_validation_results(self, design_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM validation WHERE design_id = ?", (design_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_iteration_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM designs").fetchone()
        return row[0]

    def check_config_signature(self, signature: str) -> tuple[bool, Optional[str]]:
        """Return (matches, previous_signature).

        First run on a DB stores the signature and returns (True, None).
        Subsequent runs compare; a mismatch means the objective-affecting
        config changed since the campaign was created.
        """
        row = self._conn.execute(
            "SELECT value FROM campaign WHERE key = 'config_signature'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT OR REPLACE INTO campaign (key, value) VALUES ('config_signature', ?)",
                (signature,),
            )
            self._conn.commit()
            return True, None
        prev = row["value"]
        return prev == signature, prev

    def update_design_rapid_gates(self, design_id: int, result) -> None:
        """Store rapid-gate scalars + JSON payloads for a design.

        Accepts either a plain dict (legacy callers/tests) or a
        RapidGateResult (all surrogate.py call sites). For the dataclass the
        JSON columns are derived: gate_margins <- result.margins, rapid_gates
        <- the scalar metrics (NaN sanitized to None so the JSON stays clean).
        """
        def _clean(v):
            if v is None:
                return None
            try:
                return None if not np.isfinite(v) else v
            except TypeError:
                return v

        fixed_names = ("avs_deg", "capsize_margin", "storm_peak_accel_g",
                       "roll_sigma_deg", "parametric_roll", "slam_pressure_pa",
                       "inverted_pressure_pa", "storm_wind_heel_deg")
        scalar_keys = ("avs_deg", "capsize_margin", "storm_peak_accel_g",
                       "roll_sigma_deg", "slam_pressure_pa", "slam_accel_g",
                       "inverted_pressure_pa", "storm_wind_heel_deg",
                       "roll_period_s", "max_gz_m", "gz_area_30",
                       "gz_area_40_90")

        gates = None
        margins = None
        fixed = {}
        if isinstance(result, dict):
            gates = result.get("rapid_gates")
            margins = result.get("gate_margins")
            if margins is None and "margins" in result:
                margins = result["margins"]
            for name in fixed_names:
                fixed[name] = result.get(name)
        elif hasattr(result, "margins"):
            # RapidGateResult (duck-typed; no import of rapid_gates needed)
            gates = {}
            for name in scalar_keys:
                gates[name] = _clean(getattr(result, name, None))
            gates["parametric_roll_risk"] = bool(getattr(result, "parametric_roll_risk", False))
            gates["self_right"] = bool(getattr(result, "self_right", False))
            margins = result.margins
            fixed = {
                "avs_deg": result.avs_deg,
                "capsize_margin": result.capsize_margin,
                "storm_peak_accel_g": result.storm_peak_accel_g,
                "roll_sigma_deg": result.roll_sigma_deg,
                "parametric_roll": int(bool(result.parametric_roll_risk)),
                "slam_pressure_pa": result.slam_pressure_pa,
                "inverted_pressure_pa": result.inverted_pressure_pa,
                "storm_wind_heel_deg": result.storm_wind_heel_deg,
            }
        else:
            for name in fixed_names:
                fixed[name] = getattr(result, name, None)

        self._conn.execute("""
            UPDATE designs SET
                avs_deg = ?, capsize_margin = ?, storm_peak_accel_g = ?,
                roll_sigma_deg = ?, parametric_roll = ?,
                slam_pressure_pa = ?, inverted_pressure_pa = ?,
                storm_wind_heel_deg = ?, rapid_gates = ?, gate_margins = ?
            WHERE id = ?
        """, (
            fixed["avs_deg"], fixed["capsize_margin"],
            fixed["storm_peak_accel_g"], fixed["roll_sigma_deg"],
            fixed["parametric_roll"], fixed["slam_pressure_pa"],
            fixed["inverted_pressure_pa"], fixed["storm_wind_heel_deg"],
            json.dumps(gates) if gates else None,
            json.dumps(margins) if margins else None,
            design_id,
        ))
        self._conn.commit()

    def update_design_balance(self, design_id: int, balance: dict) -> None:
        """Store 360° balance polar scalars for a design."""
        if not isinstance(balance, dict):
            return
        def _c(v):
            try:
                return None if v is None or not np.isfinite(v) else float(v)
            except Exception:
                return None
        self._conn.execute("""
            UPDATE designs SET
                balance_worst_heel_ops = ?, balance_worst_heel_storm = ?,
                balance_worst_leeway_ops = ?, balance_mean_drive = ?, balance_vmg_up = ?
            WHERE id = ?
        """, (
            _c(balance.get("worst_heel_ops_deg")),
            _c(balance.get("worst_heel_storm_deg")),
            _c(balance.get("worst_leeway_ops_deg")),
            _c(balance.get("mean_drive_ops_N")),
            _c(balance.get("vmg_up_N")),
            design_id,
        ))
        self._conn.commit()

    def update_design_mission(self, design_id: int, mission: dict) -> None:
        """Store mission-FoM rescore for a design. Never touches campaign fom."""
        if not isinstance(mission, dict):
            return
        import numpy as _np

        def _c(v):
            try:
                return None if v is None or not _np.isfinite(v) else float(v)
            except Exception:
                return None
        self._conn.execute("""
            UPDATE designs SET
                mission_fom = ?, mission_drive = ?, gust_margin = ?,
                heavy_leeway_deg = ?, draft_logistics_cost = ?
            WHERE id = ?
        """, (
            _c(mission.get("mission_fom")),
            _c(mission.get("mission_drive")),
            _c(mission.get("gust_margin")),
            _c(mission.get("heavy_leeway_deg")),
            _c(mission.get("draft_logistics_cost")),
            design_id,
        ))
        self._conn.commit()

    def store_bem_run(self, bem_hash: str, design_id: int, omega: float,
                      heading_deg: float, heave: float, pitch: float,
                      roll: float, wall_time_s: float) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO bem_runs
                (bem_hash, design_id, omega, heading_deg, heave_rao,
                 pitch_rao, roll_rao, wall_time_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (bem_hash, design_id, omega, heading_deg, heave, pitch, roll,
              wall_time_s))
        self._conn.commit()

    def store_reference_result(self, design_id: int, tool: str, status: str,
                               **metrics) -> None:
        self._conn.execute("""
            INSERT OR REPLACE INTO reference_runs
                (design_id, tool, status, max_accel_g, max_pressure_pa,
                 final_orientation, capsized, sim_time_s, wall_time_s, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            design_id, tool, status,
            metrics.get("max_accel_g"), metrics.get("max_pressure_pa"),
            metrics.get("final_orientation"), metrics.get("capsized"),
            metrics.get("sim_time_s"), metrics.get("wall_time_s"),
            metrics.get("details"),
        ))
        self._conn.commit()

    def get_reference_results(self, design_id: Optional[int] = None) -> list:
        if design_id is None:
            rows = self._conn.execute(
                "SELECT * FROM reference_runs ORDER BY design_id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM reference_runs WHERE design_id = ?",
                (design_id,),
            ).fetchall()
        return list(rows)

    def set_correction(self, key: str, value: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO corrections (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_corrections(self) -> dict:
        rows = self._conn.execute(
            "SELECT key, value FROM corrections"
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def close(self):
        self._conn.close()
