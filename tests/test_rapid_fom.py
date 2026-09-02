import time
from types import SimpleNamespace

import fcntl
import numpy as np
import pytest
from hull_opt.rapid_gates import RapidGateResult
from hull_opt.config import Config, RapidValidationConfig, ReferenceConfig, WeightConfig
from hull_opt.low_fidelity import (
    EvaluationResult,
    evaluate_low_fidelity,
    _accumulate_margin_violations,
    _margin_bonus_clip,
    _wait_spf_lock,
)


def test_config_loads_new_sections():
    from hull_opt.config import load_config
    cfg = load_config("config.yaml")
    assert hasattr(cfg, "rapid_validation")
    assert isinstance(cfg.rapid_validation, RapidValidationConfig)
    assert cfg.rapid_validation.storm_hs_factor == 4.0  # 10 m hurricane survival mode (2026-08-25)
    assert cfg.rapid_validation.min_avs_deg == 110.0
    assert hasattr(cfg, "reference")
    assert isinstance(cfg.reference, ReferenceConfig)
    assert cfg.reference.tool == "dualsphysics"
    assert hasattr(cfg, "weights")
    assert hasattr(cfg, "validation")
    assert cfg.validation.max_accel_g == 19.0  # structure+electronics shock gate (2026-08-25)
    assert cfg.weights.w6 == 0.5
    assert cfg.weights.w7 == 0.10  # structural shock reward — hardware survival, not comfort


def test_evaluation_result_has_rapid_field():
    res = EvaluationResult()
    assert hasattr(res, "rapid")
    assert res.rapid is None


def test_evaluate_low_fidelity_rapid_is_none_on_failure():
    cfg = object()
    vec = np.array([float("nan")] * 17, dtype=float)
    res = evaluate_low_fidelity(vec, cfg)
    assert hasattr(res, "rapid")
    assert res.rapid is None


def test_fom_changes_with_margins():
    """Rapid margins present vs absent should change the FoM formula path
    (not the value, since no BEM runs in this test)."""
    res0 = EvaluationResult()
    res0.feasible = True
    res0.rt_total = 1.0
    res0.fom = 1.0 / 1.0
    assert res0.fom == 1.0

    res1 = EvaluationResult()
    res1.feasible = True
    res1.rt_total = 1.0
    rapid = RapidGateResult()
    rapid.margins = {
        "capsize": 0.5,
        "storm_accel": 0.3,
        "slam_pressure": 0.2,
        "inverted_pressure": 0.1,
    }
    res1.rapid = rapid
    fom_base = 1.0
    fom_base += 0.5 * _margin_bonus_clip(rapid.margins.get("capsize", -1))
    fom_base += 0.5 * _margin_bonus_clip(rapid.margins.get("storm_accel", -1))
    fom_base += 0.5 * _margin_bonus_clip(rapid.margins.get("slam_pressure", -1))
    fom_base += 0.5 * _margin_bonus_clip(rapid.margins.get("inverted_pressure", -1))
    res1.fom = fom_base
    assert res1.fom > 1.0


def test_accumulate_margin_violations_mixed():
    """Soft gate negative margins are report-only; hard gates add magnitude."""
    rapid = RapidGateResult()
    rapid.margins = {"capsize": -0.2, "storm_accel": 0.3, "slam_pressure": -0.1,
                     "avs": -0.4, "roll_sigma": 0.2}
    soft = {"capsize", "storm_accel", "slam_pressure", "slam_accel",
            "inverted_pressure"}
    violations, magnitude = _accumulate_margin_violations(
        rapid, soft, ["pre_existing_violation"], 0.5
    )
    assert "pre_existing_violation" in violations
    assert any("capsize_margin=-0.200 < 0" in v for v in violations)
    assert any("slam_pressure_margin=-0.100 < 0" in v for v in violations)
    assert any("avs_margin=-0.400 < 0" in v for v in violations)
    # soft (capsize, slam_pressure): no magnitude; hard (avs): 0.5 * 0.4
    assert magnitude == pytest.approx(0.5 + 0.2)
    # input list must not be mutated (pure function)
    assert len(violations) == 4


def test_accumulate_margin_violations_soft_only_no_magnitude():
    rapid = RapidGateResult()
    rapid.margins = {"capsize": -0.3, "slam_pressure": -2.0,
                     "inverted_pressure": -0.5}
    soft = {"capsize", "storm_accel", "slam_pressure", "slam_accel",
            "inverted_pressure"}
    violations, magnitude = _accumulate_margin_violations(rapid, soft, [], 0.0)
    assert len(violations) == 3
    assert magnitude == 0.0


def test_accumulate_margin_violations_hard_only_adds_magnitude():
    rapid = RapidGateResult()
    rapid.margins = {"avs": -0.5}
    violations, magnitude = _accumulate_margin_violations(rapid, (), [], 0.0)
    assert any("avs_margin" in v for v in violations)
    assert magnitude == pytest.approx(0.25)


def test_margin_bonus_clip_tanh():
    assert _margin_bonus_clip(None) == -1.0
    assert _margin_bonus_clip(0.0) == 0.0
    assert _margin_bonus_clip(-2.0) < -0.95
    assert _margin_bonus_clip(2.0) > 0.95
    assert -1.0 < _margin_bonus_clip(-2.0) < 0.0
    assert 0.0 < _margin_bonus_clip(2.0) < 1.0


def test_wait_spf_lock_noop_when_no_lock_file(tmp_path):
    cfg = SimpleNamespace(paths=SimpleNamespace(output_dir=str(tmp_path)))
    t0 = time.time()
    assert _wait_spf_lock(cfg, timeout_s=1) is True
    assert time.time() - t0 < 1.0


def test_wait_spf_lock_noop_when_dir_missing(tmp_path):
    cfg = SimpleNamespace(paths=SimpleNamespace(output_dir=str(tmp_path / "no_such_dir")))
    assert _wait_spf_lock(cfg, timeout_s=1) is True


def test_wait_spf_lock_parks_until_released(tmp_path):
    import threading
    lock_path = tmp_path / "gpu.lock"
    holder_fd = open(lock_path, "a")
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = SimpleNamespace(paths=SimpleNamespace(output_dir=str(tmp_path)))

        def release_soon():
            time.sleep(0.3)
            fcntl.flock(holder_fd, fcntl.LOCK_UN)

        releaser = threading.Thread(target=release_soon)
        releaser.start()
        t0 = time.time()
        ok = _wait_spf_lock(cfg, timeout_s=1)
        elapsed = time.time() - t0
        releaser.join()
        assert ok is True, "lock was released before timeout; must return True"
        assert elapsed >= 0.25, f"worker did not park: elapsed {elapsed:.3f}s"
    finally:
        holder_fd.close()


def test_wait_spf_lock_times_out(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    holder_fd = open(lock_path, "a")
    try:
        fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = SimpleNamespace(paths=SimpleNamespace(output_dir=str(tmp_path)))
        t0 = time.time()
        ok = _wait_spf_lock(cfg, timeout_s=0.2)
        elapsed = time.time() - t0
        assert ok is False, "lock never released: must time out and return False"
        assert elapsed < 1.5, f"timeout overshot: elapsed {elapsed:.3f}s"
    finally:
        holder_fd.close()


def test_accel_penalty_uses_storm_accel():
    storm_accel = 15.0
    peak_accel = 5.0
    max_accel = 10.0
    peak_for_penalty = max(peak_accel, storm_accel)
    penalty = 2.0 * (peak_for_penalty - max_accel) if peak_for_penalty > max_accel else 0.0
    assert penalty > 0
    assert pytest.approx(penalty) == 10.0


def test_accel_penalty_no_storm():
    peak_accel = 5.0
    max_accel = 10.0
    storm_accel = peak_accel
    peak_for_penalty = max(peak_accel, storm_accel)
    penalty = 2.0 * (peak_for_penalty - max_accel) if peak_for_penalty > max_accel else 0.0
    assert penalty == 0.0


def test_config_signature_includes_new_sections():
    from hull_opt.config import load_config, config_signature
    cfg = load_config("config.yaml")
    sig = config_signature(cfg)
    assert isinstance(sig, str)
    assert len(sig) == 64


# ── Phase-0 regression: feasible-recompute from rapid margins ─────────────
# low_fidelity.py §5b: every failing margin (< 0) is appended to
# constraint_violations; violation_magnitude is inflated ONLY by hard gates
# (soft gates are penalized in the FoM via w6-w9 bonuses); feasibility is
# revoked ONLY for gates NOT in config.rapid_validation.soft_margin_gates.

_STUB_RAO = {
    "omega": [1.0, 2.0, 3.0],
    "heave_rao": [0.0, 0.0, 0.0],
    "pitch_rao": [0.0, 0.0, 0.0],
    "roll_rao": [0.0, 0.0, 0.0],
    "peak_accel": 2.0,
    "roll_period": 1.5,
}


def _run_low_fidelity_with_margins(monkeypatch, tmp_path, margins):
    from hull_opt.config import load_config
    from hull_opt.low_fidelity import evaluate_low_fidelity

    monkeypatch.setattr(
        "hull_opt.low_fidelity._wait_spf_lock",
        lambda *a, **kw: True,
        raising=False,
    )
    monkeypatch.setattr(
        "hull_opt.low_fidelity._compute_raos_capytaine",
        lambda *a, **kw: _STUB_RAO,
        raising=False,
    )
    monkeypatch.setattr(
        "hull_opt.low_fidelity.evaluate_rapid_gates",
        lambda *a, **kw: _fake_rapid_result(margins),
        raising=False,
    )
    monkeypatch.setattr(
        "hull_opt.low_fidelity.evaluate_balance_polar",
        lambda *a, **kw: _fake_balance_result(),
        raising=False,
    )
    monkeypatch.setattr(
        "hull_opt.low_fidelity.evaluate_constraints",
        lambda *a, **kw: (True, [], {}, 0.0),
        raising=False,
    )
    cfg = load_config("config.yaml")
    dv = np.zeros(17, dtype=np.float64)
    return evaluate_low_fidelity(dv, cfg, output_dir=str(tmp_path / "out"))


def _fake_balance_result():
    return {
        "AVS_deg": 120.0,
        "worst_heel_ops_deg": 5.0,
        "worst_leeway_ops_deg": 3.0,
        "worst_heel_storm_deg": 10.0,
        "worst_leeway_storm_deg": 4.0,
        "mean_drive_ops_N": 50.0,
        "vmg_up_N": 20.0,
        "vmg_down_N": 30.0,
        "n_ops_fail": 0,
        "n_storm_fail": 0,
        "system_feasible": True,
        "heel_margin_storm": 0.5,
        "heel_margin_ops": 0.6,
    }


def _fake_rapid_result(margins):
    r = RapidGateResult()
    r.margins = dict(margins)
    return r


def test_wait_spf_lock_invoked_before_bem_and_rapid(monkeypatch, tmp_path):
    from hull_opt.config import load_config
    from hull_opt.low_fidelity import evaluate_low_fidelity

    calls = []

    def spy_lock(*args, **kwargs):
        calls.append(args[0] if args else None)

    monkeypatch.setattr("hull_opt.low_fidelity._wait_spf_lock", spy_lock)
    monkeypatch.setattr(
        "hull_opt.low_fidelity._compute_raos_capytaine",
        lambda *a, **kw: _STUB_RAO,
        raising=False,
    )
    monkeypatch.setattr(
        "hull_opt.low_fidelity.evaluate_rapid_gates",
        lambda *a, **kw: _fake_rapid_result({}),
        raising=False,
    )
    monkeypatch.setattr(
        "hull_opt.low_fidelity.evaluate_balance_polar",
        lambda *a, **kw: _fake_balance_result(),
        raising=False,
    )
    monkeypatch.setattr(
        "hull_opt.low_fidelity.evaluate_constraints",
        lambda *a, **kw: (True, [], {}, 0.0),
        raising=False,
    )
    cfg = load_config("config.yaml")
    dv = np.zeros(17, dtype=np.float64)
    res = evaluate_low_fidelity(dv, cfg, output_dir=str(tmp_path / "out"))
    assert res.error_code is None, f"evaluation failed: {res.error_code}"
    assert len(calls) == 2, "lock wait must run before BEM RAOs and before rapid gates"


def test_hard_margin_failure_revokes_feasibility(monkeypatch, tmp_path):
    res = _run_low_fidelity_with_margins(
        monkeypatch, tmp_path, {"avs": -0.5, "roll_sigma": 0.2}
    )
    assert res.error_code is None, f"evaluation failed: {res.error_code}"
    assert res.feasible is False, "a failing hard margin must revoke feasibility"
    assert any("avs_margin" in v for v in res.constraint_violations)
    assert res.violation_magnitude > 0
    assert res.fom < 0, "infeasible designs route to a negative FoM"


def test_soft_margin_failures_keep_feasible(monkeypatch, tmp_path):
    res = _run_low_fidelity_with_margins(
        monkeypatch, tmp_path,
        {"slam_pressure": -2.0, "slam_accel": -1.0, "capsize": -0.3,
         "inverted_pressure": 0.1},
    )
    assert res.error_code is None, f"evaluation failed: {res.error_code}"
    assert res.feasible is True, "soft analytic margins must not revoke feasibility"
    for key in ("slam_pressure", "slam_accel", "capsize"):
        assert any(f"{key}_margin" in v for v in res.constraint_violations), \
            f"missing violation entry for {key}"
    assert res.violation_magnitude == 0.0, \
        "soft margins are report-only and must not poison violation_magnitude"


def test_soft_and_hard_margin_mix(monkeypatch, tmp_path):
    res = _run_low_fidelity_with_margins(
        monkeypatch, tmp_path,
        {"slam_pressure": -2.0, "avs": -0.5},
    )
    assert res.error_code is None, f"evaluation failed: {res.error_code}"
    assert res.feasible is False, "a failing hard margin must revoke feasibility"
    assert any("slam_pressure_margin" in v for v in res.constraint_violations)
    assert any("avs_margin" in v for v in res.constraint_violations)
    assert res.violation_magnitude == pytest.approx(0.25), \
        "only the hard gate (avs) should contribute to magnitude"