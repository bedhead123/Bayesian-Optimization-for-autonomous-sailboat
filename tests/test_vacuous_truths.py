"""Tests for the vacuous-truth fixes: calibration least-infeasible fallback,
LHS plateau gating on feasible designs, and systemic-error BO abort logic."""
import numpy as np
import pytest

from hull_opt.database import OptimizationDatabase
from hull_opt.surrogate import _abort_reason


@pytest.fixture
def db(tmp_path):
    return OptimizationDatabase(str(tmp_path / "test.db"))


def _insert(db, iter_num, fom, feasible=True, violations=None, error_code=None):
    db.insert_design(
        iter_num=iter_num,
        design_vector=np.zeros(17),
        feasible=feasible,
        fom=fom,
        constraint_violations=violations,
        error_code=error_code,
    )


def test_fallback_returns_none_without_flag(db):
    _insert(db, 0, fom=-5.0, feasible=False, violations=["roll_period_margin=-1.0 < 0"])
    assert db.get_best_feasible_uncalibrated() is None


def test_fallback_picks_least_infeasible(db):
    _insert(db, 0, fom=-5.0, feasible=False,
            violations=["a_margin=-1.0 < 0", "b_margin=-1.0 < 0"])
    _insert(db, 1, fom=-2.0, feasible=False,
            violations=["a_margin=-1.0 < 0"])
    best = db.get_best_feasible_uncalibrated(fallback_to_least_infeasible=True)
    assert best is not None
    assert best["iter"] == 1
    assert best["feasible"] == 0


def test_fallback_skips_error_rows(db):
    _insert(db, 0, fom=-5.0, feasible=False,
            violations=["a_margin=-1.0 < 0"])
    _insert(db, 1, fom=-2.0, feasible=False, error_code="E_RAO:boom")
    best = db.get_best_feasible_uncalibrated(fallback_to_least_infeasible=True)
    assert best is not None
    assert best["iter"] == 0


def test_fallback_prefers_feasible_when_any_exist(db):
    _insert(db, 0, fom=-5.0, feasible=False, violations=["a_margin=-1.0 < 0"])
    _insert(db, 1, fom=3.0, feasible=True)
    best = db.get_best_feasible_uncalibrated(fallback_to_least_infeasible=True)
    assert best["iter"] == 1
    assert best["feasible"] == 1


def test_fallback_respects_calibration_attempts(db):
    _insert(db, 0, fom=-5.0, feasible=False, violations=["a_margin=-1.0 < 0"])
    d0 = db.get_design(1)
    db.mark_calibration_attempt(d0["id"], 100, "ok")
    assert db.get_best_feasible_uncalibrated(fallback_to_least_infeasible=True) is None


def test_abort_reason_requires_10_bo_iters():
    assert _abort_reason(bo_n=9, bo_err_frac=1.0, lhs_err_frac=0.0, bo_env_frac=1.0) is None


def test_abort_reason_requires_env_dominance():
    assert _abort_reason(bo_n=10, bo_err_frac=0.9, lhs_err_frac=0.0, bo_env_frac=0.3) is None


def test_abort_reason_requires_rate_gap():
    # LHS phase also heavily errored: no clear environment regression -> no abort
    assert _abort_reason(bo_n=10, bo_err_frac=0.9, lhs_err_frac=0.8, bo_env_frac=0.9) is None


def test_abort_reason_fires_on_systemic_env_failure():
    reason = _abort_reason(bo_n=10, bo_err_frac=1.0, lhs_err_frac=0.0, bo_env_frac=1.0)
    assert reason is not None
    assert "environment" in reason.lower()
