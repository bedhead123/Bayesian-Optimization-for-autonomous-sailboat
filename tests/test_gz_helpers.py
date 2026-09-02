import numpy as np
import pytest
from hull_opt.hydrostatics import compute_avs, compute_gz_area, compute_righting_energy

RHO = 1025.0
G = 9.81
DISP = 0.25


def _synthetic():
    angles = np.linspace(0.0, 50.0, 6)
    gz = np.array([0.0, 0.05, 0.10, 0.05, 0.0, -0.05])
    return np.column_stack([angles, gz, np.full(6, DISP)])


def test_gz_area_matches_trapezoid():
    curve = _synthetic()
    area = compute_gz_area(curve, 0.0, 30.0)
    angles = curve[:, 0]
    gz = curve[:, 1]
    expected = np.trapezoid(np.maximum(gz[:4], 0.0), np.deg2rad(angles[:4]))
    assert area == pytest.approx(expected, abs=1e-12)
    assert area == pytest.approx(0.175 * np.deg2rad(10.0), abs=1e-12)


def test_gz_area_clips_negative():
    curve = _synthetic()
    full = compute_gz_area(curve, 0.0, 50.0)
    angles = curve[:, 0]
    gz = curve[:, 1]
    expected = np.trapezoid(np.maximum(gz, 0.0), np.deg2rad(angles))
    assert full == pytest.approx(expected, abs=1e-12)
    assert full == pytest.approx(compute_gz_area(curve, 0.0, 40.0), abs=1e-12)


def test_gz_area_lo_gt_hi():
    curve = _synthetic()
    assert compute_gz_area(curve, 40.0, 10.0) == 0.0


def test_gz_area_bad_input():
    assert compute_gz_area(np.zeros((0, 3)), 0.0, 90.0) == 0.0
    assert compute_gz_area(np.array([[0.0, 1.0, 1.0]]), 0.0, 90.0) == 0.0
    nan_curve = np.column_stack([[0.0, 30.0, 90.0], [np.nan, 0.1, 0.0],
                                 [0.25, 0.25, 0.25]])
    assert compute_gz_area(nan_curve, 0.0, 90.0) == 0.0


def test_gz_area_single_point_mask():
    assert compute_gz_area(_synthetic(), 50.0, 50.0) == 0.0


def _avs_curve():
    angles = np.array([0.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0])
    gz = np.array([0.0, 0.05, 0.08, 0.10, 0.09, 0.02, -0.02])
    return np.column_stack([angles, gz, np.zeros(7)])


def test_avs_interpolated_crossing():
    curve = _avs_curve()
    assert compute_avs(curve) == pytest.approx(45.0, abs=1e-9)


def test_avs_no_crossing_returns_max_angle():
    angles = np.linspace(0.0, 180.0, 37)
    gz = np.full(37, 0.05)
    curve = np.column_stack([angles, gz, np.zeros(37)])
    assert compute_avs(curve) == pytest.approx(180.0, abs=1e-12)


def test_avs_all_negative():
    angles = np.linspace(0.0, 180.0, 19)
    gz = np.full(19, -0.03)
    curve = np.column_stack([angles, gz, np.zeros(19)])
    assert compute_avs(curve) == 0.0


def test_avs_flat_zero():
    angles = np.linspace(0.0, 180.0, 37)
    curve = np.column_stack([angles, np.zeros(37), np.zeros(37)])
    assert compute_avs(curve) == 0.0


def test_avs_max_at_curve_end():
    angles = np.array([0.0, 30.0, 60.0])
    gz = np.array([0.0, 0.05, 0.08])
    curve = np.column_stack([angles, gz, np.zeros(3)])
    assert compute_avs(curve) == pytest.approx(60.0, abs=1e-12)


def test_righting_energy_equals_area_scaled():
    curve = _synthetic()
    energy = compute_righting_energy(curve, max_heel_deg=30.0,
                                     rho=RHO, g=G, displacement=DISP)
    area = compute_gz_area(curve, 0.0, 30.0)
    assert energy == pytest.approx(RHO * G * DISP * area, rel=1e-12)


def test_righting_energy_constant_gz():
    angles = np.linspace(0.0, 90.0, 10)
    gz = np.full(10, 0.1)
    curve = np.column_stack([angles, gz, np.full(10, DISP)])
    energy = compute_righting_energy(curve, max_heel_deg=90.0,
                                     rho=RHO, g=G, displacement=DISP)
    expected = RHO * G * DISP * 0.1 * (np.pi / 2.0)
    assert energy == pytest.approx(expected, rel=1e-12)
