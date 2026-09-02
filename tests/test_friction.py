import numpy as np
from hull_opt.friction import compute_frictional_resistance, compute_total_resistance


def test_friction_positive():
    Rf = compute_frictional_resistance(speed_ms=2.0, wetted_area=1.0, LWL=2.4)
    assert Rf > 0


def test_friction_zero_at_zero_speed():
    Rf = compute_frictional_resistance(speed_ms=0.0, wetted_area=1.0, LWL=2.4)
    assert Rf == 0.0


def test_friction_increases_with_speed():
    Rf1 = compute_frictional_resistance(speed_ms=1.0, wetted_area=1.0, LWL=2.4)
    Rf2 = compute_frictional_resistance(speed_ms=2.0, wetted_area=1.0, LWL=2.4)
    assert Rf2 > Rf1


def test_friction_increases_with_area():
    Rf1 = compute_frictional_resistance(speed_ms=2.0, wetted_area=0.5, LWL=2.4)
    Rf2 = compute_frictional_resistance(speed_ms=2.0, wetted_area=1.0, LWL=2.4)
    assert Rf2 > Rf1


def test_total_equals_friction_with_no_wave():
    speed = 2.0
    area = 1.0
    LWL = 2.4
    Rt, Rf, Rw = compute_total_resistance(speed, area, LWL, wave_resistance=0.0)
    assert Rw == 0.0
    assert Rt == Rf


def test_total_adds_wave_resistance():
    speed = 2.0
    area = 1.0
    LWL = 2.4
    Rt, Rf, Rw = compute_total_resistance(speed, area, LWL, wave_resistance=5.0)
    assert Rw == 5.0
    assert Rt == Rf + 5.0


def test_friction_very_low_reynolds():
    Rf = compute_frictional_resistance(speed_ms=1e-6, wetted_area=1.0, LWL=2.4)
    assert Rf == 0.0
