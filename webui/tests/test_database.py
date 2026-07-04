import numpy as np
import tempfile
from pathlib import Path
from hull_opt.database import OptimizationDatabase


def _make_db():
    tmp = tempfile.mkdtemp()
    db_path = str(Path(tmp) / "test.db")
    return OptimizationDatabase(db_path), tmp


def test_insert_and_get_design():
    db, tmp = _make_db()
    dv = np.array([2.4, 0.5, 0.2, 0.6, 0.75, 10.0, 1.0, 0.2, 0.003, 0.4,
                   0.2, 0.15, 0.8, 12.0, 0.1, 0.005, 0.55])
    design_id = db.insert_design(
        iter_num=0, design_vector=dv, feasible=True, fom=1.5,
        rt_total=10.0, rt_wave=3.0, rt_friction=7.0,
        stability_index=1.0, roll_period=5.0, peak_accel=10.0,
        constraint_values={"B/LWL": 0.21}, constraint_violations=[],
    )
    assert design_id > 0
    d = db.get_design(design_id)
    assert d is not None
    assert d["iter"] == 0
    assert d["feasible"] == 1
    assert abs(d["fom"] - 1.5) < 1e-6
    assert d["rt_total"] == 10.0


def test_get_iteration_count_empty():
    db, tmp = _make_db()
    assert db.get_iteration_count() == 0


def test_get_iteration_count():
    db, tmp = _make_db()
    dv = np.array([2.4, 0.5, 0.2, 0.6, 0.75, 10.0, 1.0, 0.2, 0.003, 0.4,
                   0.2, 0.15, 0.8, 12.0, 0.1, 0.005, 0.55])
    db.insert_design(iter_num=5, design_vector=dv, feasible=True, fom=1.0)
    assert db.get_iteration_count() == 1


def test_get_feasible_designs():
    db, tmp = _make_db()
    dv = np.array([2.4, 0.5, 0.2, 0.6, 0.75, 10.0, 1.0, 0.2, 0.003, 0.4,
                   0.2, 0.15, 0.8, 12.0, 0.1, 0.005, 0.55])
    db.insert_design(iter_num=0, design_vector=dv, feasible=False, fom=-1.0)
    db.insert_design(iter_num=1, design_vector=dv, feasible=True, fom=2.0)
    db.insert_design(iter_num=2, design_vector=dv, feasible=True, fom=1.0)
    feasible = db.get_feasible_designs()
    assert len(feasible) == 2
    assert feasible[0]["fom"] == 2.0  # sorted DESC


def test_get_best_feasible():
    db, tmp = _make_db()
    dv = np.array([2.4, 0.5, 0.2, 0.6, 0.75, 10.0, 1.0, 0.2, 0.003, 0.4,
                   0.2, 0.15, 0.8, 12.0, 0.1, 0.005, 0.55])
    db.insert_design(iter_num=0, design_vector=dv, feasible=True, fom=1.0)
    db.insert_design(iter_num=1, design_vector=dv, feasible=True, fom=3.0)
    best = db.get_best_feasible()
    assert best is not None
    assert best["fom"] == 3.0


def test_calibration_roundtrip():
    db, tmp = _make_db()
    dv = np.array([2.4, 0.5, 0.2, 0.6, 0.75, 10.0, 1.0, 0.2, 0.003, 0.4,
                   0.2, 0.15, 0.8, 12.0, 0.1, 0.005, 0.55])
    did = db.insert_design(iter_num=0, design_vector=dv, feasible=True, fom=1.0)
    db.store_calibration(did, 0, 10.0, 12.0, 2.0)
    calibs = db.get_calibrations()
    assert len(calibs) == 1
    assert abs(calibs[0]["delta"] - 2.0) < 1e-6
    latest = db.get_latest_calibration()
    assert latest is not None


def test_validation_roundtrip():
    db, tmp = _make_db()
    dv = np.array([2.4, 0.5, 0.2, 0.6, 0.75, 10.0, 1.0, 0.2, 0.003, 0.4,
                   0.2, 0.15, 0.8, 12.0, 0.1, 0.005, 0.55])
    did = db.insert_design(iter_num=0, design_vector=dv, feasible=True, fom=1.0)
    db.store_validation(did, "gate1", True, 10.0, 20.0, "OK")
    results = db.get_validation_results(did)
    assert len(results) == 1
    assert results[0]["gate_name"] == "gate1"


def test_get_top_n():
    db, tmp = _make_db()
    dv = np.array([2.4, 0.5, 0.2, 0.6, 0.75, 10.0, 1.0, 0.2, 0.003, 0.4,
                   0.2, 0.15, 0.8, 12.0, 0.1, 0.005, 0.55])
    for i in range(5):
        db.insert_design(iter_num=i, design_vector=dv, feasible=True, fom=float(5 - i))
    top3 = db.get_top_n(3)
    assert len(top3) == 3
    assert top3[0]["fom"] == 5.0
    assert top3[2]["fom"] == 3.0
