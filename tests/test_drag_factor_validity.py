"""Drag-factor validity tests (B1/B3): the calibration factor must never be
stored from a sub-friction Rt (physically impossible), and the ratio must be
de-amplified so SPH noise cannot swing it across its full range."""
import pytest

from hull_opt.mid_fidelity import calibration_factor


# ── B1: validity gate ────────────────────────────────────────────────────────

def test_sub_friction_rt_is_invalid():
    factor, reason = calibration_factor(rt_sph=5.0, Rf=21.0, Rw=10.0)
    assert factor is None
    assert reason is not None and "sub-friction" in reason


def test_exact_friction_boundary_is_invalid():
    factor, reason = calibration_factor(rt_sph=21.0, Rf=21.0, Rw=10.0)
    assert factor is None


def test_non_finite_inputs_invalid():
    factor, reason = calibration_factor(rt_sph=float("nan"), Rf=21.0, Rw=10.0)
    assert factor is None
    assert reason is not None and "non-finite" in reason


def test_zero_wave_resistance_invalid():
    factor, reason = calibration_factor(rt_sph=30.0, Rf=21.0, Rw=0.0)
    assert factor is None
    assert "no wave resistance" in reason


def test_valid_rt_above_friction_passes():
    factor, reason = calibration_factor(rt_sph=30.0, Rf=21.0, Rw=10.0)
    assert reason is None
    assert factor == pytest.approx((30.0 - 21.0) / 10.0)


# ── B3: de-amplified denominator ────────────────────────────────────────────

    def test_denominator_floored_at_03_rf(self):
        # Rw tiny (0.1 N) but non-zero: denom = 0.3*Rf = 6.3, not 0.1.
        # f_low = (30-21)/6.3 = 1.43 — in band [0.5, 2.0]. A ±5 N SPH noise
        # swing pushes the raw factor to 2.22 > 2.0 → out-of-band → None
        # (invalid), which is exactly the bounded behavior we want.
        f_low, _ = calibration_factor(rt_sph=30.0, Rf=21.0, Rw=0.1)
        f_high, _ = calibration_factor(rt_sph=30.0 + 5.0, Rf=21.0, Rw=0.1)
        assert f_low == pytest.approx((30.0 - 21.0) / 6.3)
        assert f_high is None


def test_noise_swing_stays_bounded_within_one_step():
    # Rf=20, Rw=10, rt_sph=30±5: old formula (no floor) gave (30±5-20)/10
    # = 1.0±0.5. With the 0.3*Rf floor (denom=10 still larger than 6) the
    # swing is identical here, but when Rw < 0.3*Rf the floor binds.
    f_low, _ = calibration_factor(rt_sph=25.0, Rf=20.0, Rw=10.0)
    f_high, _ = calibration_factor(rt_sph=35.0, Rf=20.0, Rw=10.0)
    assert f_low == pytest.approx(0.5)
    assert f_high == pytest.approx(1.5)
    assert (f_high - f_low) <= 1.0


def test_factor_out_of_band_returns_none():
    # factor_bounds [0.5, 2.0]: out-of-band raw factors return (None, reason)
    # → calibration INVALID, factor stays 1.0 (Bug #157: the old clip-to-5.0
    # corrupted every FoM at iter 110 of the 2026-08-24 campaign)
    f, _ = calibration_factor(rt_sph=1e6, Rf=20.0, Rw=10.0)
    assert f is None
    f2, _ = calibration_factor(rt_sph=20.01, Rf=20.0, Rw=10.0)
    assert f2 is None