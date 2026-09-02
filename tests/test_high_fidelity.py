"""Unit tests for high_fidelity._extract_peak_accel_from_motion data guards."""
import numpy as np
import pytest
from pathlib import Path

from hull_opt.high_fidelity import _extract_peak_accel_from_motion


def _state_file(base: Path, t: str, vel=(0.0, 0.0, 0.0), accel=(0.0, 0.0, 0.0),
                orient=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    d = base / t / "uniform"
    d.mkdir(parents=True, exist_ok=True)
    v = " ".join(str(x) for x in vel)
    a = " ".join(str(x) for x in accel)
    o = " ".join(str(x) for x in orient)
    (d / "sixDoFRigidBodyMotionState").write_text(
        f"""centreOfRotation ( 0 0 0 );

orientation     ( {o} );

velocity        ( {v} );

acceleration    ( {a} );
"""
    )


def _valid_states(base: Path, n=10, dt=0.01):
    for i in range(1, n + 1):
        a = (0.0, 0.0, 2.0 * i)  # grows to 20 m/s^2 ~ 2 g
        _state_file(base, f"{i * dt:.3f}", vel=(0.5, 0.0, 0.0), accel=a)


def test_valid_run_returns_peak_g(tmp_path):
    _valid_states(tmp_path)
    val = _extract_peak_accel_from_motion(tmp_path, min_sim_time=0.0)
    assert val is not None
    assert val == pytest.approx(20.0 / 9.81, rel=1e-6)


def test_single_t0_state_rejected(tmp_path):
    # The 0.01 g t=0 garbage: one partial state only
    _state_file(tmp_path, "0.003", vel=(0.0, 0.0, -7.67), accel=(0.0, 0.0, 0.075))
    assert _extract_peak_accel_from_motion(tmp_path, min_sim_time=0.03) is None


def test_crashed_short_run_rejected_by_min_sim_time(tmp_path):
    # Many states but sim time stalls well short of the gate minimum
    for i in range(1, 6):
        _state_file(tmp_path, f"{i * 0.001:.3f}", accel=(0.0, 0.0, 1.0))
    assert _extract_peak_accel_from_motion(tmp_path, min_sim_time=0.03) is None
    # ...but passes a lenient minimum
    assert _extract_peak_accel_from_motion(tmp_path, min_sim_time=0.0) is not None


def test_nan_accel_rejected(tmp_path):
    _valid_states(tmp_path)
    _state_file(tmp_path, "0.11", accel=(float("nan"), 0.0, 0.0))
    assert _extract_peak_accel_from_motion(tmp_path, min_sim_time=0.0) is None


def test_inf_velocity_blowup_rejected(tmp_path):
    _valid_states(tmp_path)
    _state_file(tmp_path, "0.12", vel=(1e52, 0.0, 0.0), accel=(1.0, 0.0, 0.0))
    assert _extract_peak_accel_from_motion(tmp_path, min_sim_time=0.0) is None


def test_huge_accel_rejected(tmp_path):
    _valid_states(tmp_path)
    _state_file(tmp_path, "0.13", accel=(20000.0, 0.0, 0.0))  # > 1000 g
    assert _extract_peak_accel_from_motion(tmp_path, min_sim_time=0.0) is None


def test_processor_dir_history_read(tmp_path):
    # Parallel runs keep full history in processor dirs, main case has 1 state
    _state_file(tmp_path / "processor0", "0.001", accel=(1.0, 0.0, 0.0))
    _state_file(tmp_path / "processor0", "0.002", accel=(2.0, 0.0, 0.0))
    _state_file(tmp_path / "processor1", "0.002", accel=(2.0, 0.0, 0.0))  # dup time
    _state_file(tmp_path / "processor0", "0.003", accel=(3.0, 0.0, 0.0))
    _state_file(tmp_path, "0.003", accel=(3.0, 0.0, 0.0))  # reconstructed latest
    val = _extract_peak_accel_from_motion(tmp_path, min_sim_time=0.0)
    assert val is not None
    assert val == pytest.approx(3.0 / 9.81, rel=1e-6)
