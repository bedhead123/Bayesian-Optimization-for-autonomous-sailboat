"""Unit tests for drag-factor smoothing (WP2.4) and B2 calibration handling."""
import pytest

from hull_opt.surrogate import smooth_drag_factor


def test_first_factor_passes_through():
    # No first-calibration short-circuit (no instant 1.0→X jumps): the first
    # factor is smoothed from prev=1.0 like any other step.
    val, clipped = smooth_drag_factor(1.0, 0.7009, 0.20)
    assert val < 1.0 and val > 0.7009  # moved toward target, bounded step
    assert not clipped


def test_small_step_not_clipped():
    # old=1.0956, new=1.0 -> cand=1.0478, +4.4% < 20% tolerance
    val, clipped = smooth_drag_factor(1.0956, 1.0, 0.20)
    assert val == pytest.approx(1.0478)
    assert not clipped


def test_large_swing_clipped_to_tolerance():
    # The observed -36% swing: 1.0956 -> 0.7009. Cand = 0.8982 is -18% below
    # old, within 20%: so it passes. Use an extreme case to force clipping.
    val, clipped = smooth_drag_factor(1.0956, 0.1, 0.20)
    assert clipped
    assert val == pytest.approx(1.0956 * 0.8, rel=1e-6)


def test_upward_swing_clipped():
    val, clipped = smooth_drag_factor(0.7009, 3.0, 0.20)
    assert clipped
    assert val == pytest.approx(0.7009 * 1.2, rel=1e-6)


def test_exact_tolerance_boundary_not_clipped():
    old, new = 1.0, 1.0
    val, clipped = smooth_drag_factor(old, new, 0.20)
    assert not clipped
    assert val == pytest.approx(1.0)


# ── B2: invalid calibration resets the factor, skips storage ────────────────

class _FakeDB:
    """Records calls instead of touching SQLite."""

    def __init__(self):
        self.calls = []
        self.stored = []

    def mark_calibration_attempt(self, design_id, it, status, factor=None):
        self.calls.append(("attempt", design_id, it, status, factor))

    def store_calibration(self, *args):
        self.stored.append(args)

    def rescore_foms(self, *args):
        return 0

    def get_all_designs(self):
        return []


def _optimizer(config, db):
    from hull_opt.surrogate import HullOptimizer
    opt = object.__new__(HullOptimizer)
    opt.config = config
    opt.db = db
    opt.drag_factor = 1.4
    opt._sync_history_from_db = lambda: None
    return opt


def test_invalid_calibration_resets_factor_and_skips_store():
    from hull_opt.config import load_config
    cfg = load_config("config.yaml")
    db = _FakeDB()
    opt = _optimizer(cfg, db)
    cal = {"valid": False, "reason": "rt_sph=5.0 N < Rf=21.0 N"}
    opt._handle_calibration_result(cal, design_id=7, it=3)
    assert opt.drag_factor == 1.0
    assert db.calls == [("attempt", 7, 3, "invalid", 1.0)]
    assert db.stored == []  # no store_calibration from bad data


def test_valid_calibration_stores_and_smooths():
    from hull_opt.config import load_config
    cfg = load_config("config.yaml")
    db = _FakeDB()
    opt = _optimizer(cfg, db)
    opt._handle_calibration_result(
        {"valid": True, "factor": 1.2, "rt_lowfi": 20.0,
         "rt_cfd": 25.0, "delta": 5.0},
        design_id=7, it=3,
    )
    assert opt.drag_factor == pytest.approx(1.3)  # 1.4 blended halfway to 1.2
    assert db.calls == [("attempt", 7, 3, "ok", pytest.approx(1.3))]
    assert len(db.stored) == 1
    assert db.stored[0][:2] == (7, 3)
    assert db.stored[0][5] == pytest.approx(1.3)


def test_none_calibration_marks_failed():
    from hull_opt.config import load_config
    cfg = load_config("config.yaml")
    db = _FakeDB()
    opt = _optimizer(cfg, db)
    opt._handle_calibration_result(None, design_id=7, it=3)
    assert opt.drag_factor == 1.4  # unchanged
    assert db.calls == [("attempt", 7, 3, "failed", None)]
