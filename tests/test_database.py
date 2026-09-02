"""Unit tests for the calibration-attempt bookkeeping (no-repeat calibrations)."""
import json
import sqlite3
import numpy as np
import pytest

from hull_opt.database import OptimizationDatabase


@pytest.fixture
def db(tmp_path):
    return OptimizationDatabase(str(tmp_path / "test.db"))


def _insert_design(db, iter_num, fom, feasible=True):
    db.insert_design(
        iter_num=iter_num,
        design_vector=np.zeros(17),
        feasible=feasible,
        fom=fom,
    )


def test_uncalibrated_picks_best_untouched(db):
    _insert_design(db, 0, fom=1.0)
    _insert_design(db, 1, fom=2.0)
    best = db.get_best_feasible_uncalibrated()
    assert best is not None
    assert best["iter"] == 1  # highest FoM


def test_attempted_design_excluded(db):
    _insert_design(db, 0, fom=3.0)
    _insert_design(db, 1, fom=2.0)
    db.mark_calibration_attempt(1, 100, "failed")  # design id 1 = iter 1? no:
    # design ids are autoincrement; find the top design's id
    d0 = db.get_design(1)  # first inserted
    db.mark_calibration_attempt(d0["id"], 100, "failed")
    best = db.get_best_feasible_uncalibrated()
    assert best is not None
    assert best["id"] != d0["id"]


def test_all_attempted_returns_none(db):
    _insert_design(db, 0, fom=3.0)
    d0 = db.get_design(1)
    db.mark_calibration_attempt(d0["id"], 100, "ok")
    assert db.get_best_feasible_uncalibrated() is None


def test_retry_failed_allows_only_failed(db):
    _insert_design(db, 0, fom=3.0)
    _insert_design(db, 1, fom=2.0)
    d0 = db.get_design(1)
    d1 = db.get_design(2)
    db.mark_calibration_attempt(d0["id"], 100, "failed")
    db.mark_calibration_attempt(d1["id"], 100, "timeout")
    # default: neither retryable
    assert db.get_best_feasible_uncalibrated() is None
    # retry_failed: only the 'failed' one is eligible again
    best = db.get_best_feasible_uncalibrated(retry_failed=True)
    assert best is not None
    assert best["id"] == d0["id"]


def test_invalid_status_accepted_and_not_retryable(db):
    """B2: an invalid calibration (sub-friction Rt / non-steady trace) is
    stored with status 'invalid' and must not be re-picked even with
    retry_failed=True (the physics-level rejection will repeat)."""
    _insert_design(db, 0, fom=3.0)
    d0 = db.get_design(1)
    db.mark_calibration_attempt(d0["id"], 100, "invalid", factor=1.0)
    assert db.get_best_feasible_uncalibrated() is None
    assert db.get_best_feasible_uncalibrated(retry_failed=True) is None
    att = db.get_calibration_attempt(d0["id"])
    assert att["status"] == "invalid"
    assert att["factor"] == 1.0


def test_running_attempt_blocks(db):
    _insert_design(db, 0, fom=3.0)
    d0 = db.get_design(1)
    db.mark_calibration_attempt(d0["id"], 100, "running")
    assert db.get_best_feasible_uncalibrated() is None
    # 'running' is not retryable even with retry_failed
    assert db.get_best_feasible_uncalibrated(retry_failed=True) is None


def test_infeasible_designs_never_picked(db):
    _insert_design(db, 0, fom=3.0, feasible=False)
    assert db.get_best_feasible_uncalibrated() is None


def test_factor_roundtrip_on_attempt(db):
    _insert_design(db, 0, fom=3.0)
    d0 = db.get_design(1)
    db.mark_calibration_attempt(d0["id"], 100, "ok", factor=0.7009)
    row = db.get_calibration_attempt(d0["id"])
    assert row is not None
    assert row["factor"] == pytest.approx(0.7009)


def test_rapid_gate_columns_migration(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "old.db"))
    conn.execute("""
        CREATE TABLE designs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iter INTEGER NOT NULL,
            design_vector TEXT NOT NULL,
            feasible INTEGER NOT NULL DEFAULT 0,
            fom REAL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE UNIQUE INDEX idx_designs_iter ON designs(iter)")
    conn.close()
    db = OptimizationDatabase(str(tmp_path / "old.db"))
    assert hasattr(db, "update_design_rapid_gates")
    cols = [r[1] for r in db._conn.execute("PRAGMA table_info(designs)").fetchall()]
    for c in ("avs_deg", "capsize_margin", "storm_peak_accel_g",
              "roll_sigma_deg", "parametric_roll", "slam_pressure_pa",
              "inverted_pressure_pa", "storm_wind_heel_deg",
              "rapid_gates", "gate_margins"):
        assert c in cols, f"missing column {c}"
    db.close()


def test_store_bem_run_roundtrip(db):
    db.store_bem_run("abc123", 1, 0.5, 180.0, 0.1, 0.02, 0.01, 3.5)
    rows = db._conn.execute(
        "SELECT * FROM bem_runs WHERE bem_hash = ?", ("abc123",)
    ).fetchall()
    assert len(rows) == 1
    r = dict(rows[0])
    assert r["omega"] == 0.5
    assert r["heading_deg"] == 180.0
    assert r["heave_rao"] == 0.1
    assert r["pitch_rao"] == 0.02
    assert r["roll_rao"] == 0.01
    assert r["wall_time_s"] == 3.5


def test_reference_result_store_roundtrip(db):
    db.store_reference_result(1, "DualSPHysics", "ok", max_accel_g=12.5,
                              max_pressure_pa=5000.0, final_orientation="upright",
                              capsized=0, sim_time_s=4.0, wall_time_s=120.0)
    rows = db.get_reference_results(design_id=1)
    assert len(rows) == 1
    r = dict(rows[0])
    assert r["tool"] == "DualSPHysics"
    assert r["max_accel_g"] == 12.5
    assert r["capsized"] == 0

    results = db.get_reference_results()
    assert len(results) >= 1


def test_corrections_roundtrip(db):
    db.set_correction("drag_factor", 1.25)
    db.set_correction("wave_amplification", 0.92)
    corr = db.get_corrections()
    assert corr["drag_factor"] == 1.25
    assert corr["wave_amplification"] == 0.92

    db.set_correction("drag_factor", 1.30)
    corr = db.get_corrections()
    assert corr["drag_factor"] == 1.30
    assert len(corr) == 2


def test_update_design_rapid_gates(db):
    did = db.insert_design(iter_num=0, design_vector=np.zeros(17),
                           feasible=True, fom=1.0)
    result = {
        "avs_deg": 125.0,
        "capsize_margin": 0.45,
        "storm_peak_accel_g": 2.3,
        "roll_sigma_deg": 8.1,
        "parametric_roll": 0,
        "slam_pressure_pa": 4500.0,
        "inverted_pressure_pa": 12000.0,
        "storm_wind_heel_deg": 22.5,
        "rapid_gates": {"avs": 125.0},
        "gate_margins": {"avs": 0.45, "stability": 0.12},
    }
    db.update_design_rapid_gates(did, result)
    d = db.get_design(did)
    assert d["avs_deg"] == 125.0
    assert d["capsize_margin"] == 0.45
    assert d["storm_peak_accel_g"] == 2.3
    assert d["roll_sigma_deg"] == 8.1
    assert d["parametric_roll"] == 0
    assert d["slam_pressure_pa"] == 4500.0
    assert d["inverted_pressure_pa"] == 12000.0
    assert d["storm_wind_heel_deg"] == 22.5
    gm = json.loads(d["gate_margins"])
    assert gm["avs"] == 0.45


MARGIN_KEYS_15 = [
    "avs", "capsize", "storm_accel", "roll_sigma",
    "slam_pressure", "inverted_pressure", "wind_heel", "parametric_roll",
    "righting_energy", "slam_accel", "max_gz_m",
    "gz_area_30", "gz_area_40_90", "self_right", "roll_period",
]


def test_update_design_rapid_gates_with_result_object(db):
    from hull_opt.rapid_gates import RapidGateResult
    did = db.insert_design(iter_num=0, design_vector=np.zeros(17),
                           feasible=True, fom=1.0)
    res = RapidGateResult(
        avs_deg=125.0, capsize_margin=0.45,
        storm_peak_accel_g=2.3, roll_sigma_deg=8.1,
        roll_period_s=float("nan"),
        slam_pressure_pa=4500.0, slam_accel_g=1.2,
        inverted_pressure_pa=12000.0, storm_wind_heel_deg=22.5,
        max_gz_m=0.08, gz_area_30=0.012, gz_area_40_90=0.015,
        self_right=True, parametric_roll_risk=False,
    )
    res.margins = {k: 0.5 for k in MARGIN_KEYS_15}
    db.update_design_rapid_gates(did, res)
    d = db.get_design(did)
    assert d["rapid_gates"] is not None
    assert d["gate_margins"] is not None
    assert d["avs_deg"] == 125.0
    assert d["parametric_roll"] == 0

    rg = json.loads(d["rapid_gates"])
    assert rg["storm_peak_accel_g"] == 2.3
    assert rg["roll_sigma_deg"] == 8.1
    assert rg["roll_period_s"] is None  # NaN sanitized to JSON null
    assert rg["slam_accel_g"] == 1.2
    assert rg["self_right"] is True

    gm = json.loads(d["gate_margins"])
    assert set(gm.keys()) == set(MARGIN_KEYS_15)
    assert all(gm[k] == 0.5 for k in MARGIN_KEYS_15)


def test_update_design_rapid_gates_dict_margins_only(db):
    did = db.insert_design(iter_num=0, design_vector=np.zeros(17),
                           feasible=True, fom=1.0)
    result = {"margins": {"avs": 0.45, "capsize": 0.55}}
    db.update_design_rapid_gates(did, result)
    d = db.get_design(did)
    assert d["gate_margins"] is not None
    gm = json.loads(d["gate_margins"])
    assert gm["avs"] == 0.45
    assert gm["capsize"] == 0.55
