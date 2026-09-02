"""Tests for fidelity shrink (rapid-gate bypass in high/mid fidelity)."""
import numpy as np
import pytest
from pathlib import Path

from hull_opt.rapid_gates import RapidGateResult


def test_rapid_gates_used_in_high_fi(monkeypatch):
    """Gate 2/4/5 values come from pre-computed rapid result, not OF."""
    from hull_opt.high_fidelity import _validate_single

    rapid = RapidGateResult()
    rapid.storm_peak_accel_g = 5.2
    rapid.slam_pressure_pa = 45000.0
    rapid.inverted_pressure_pa = 38000.0

    monkeypatch.setattr("hull_opt.geometry_validator.validate_design_vector",
                        lambda *a, **kw: (True, "ok"))
    monkeypatch.setattr("hull_opt.high_fidelity.generate_hull",
                        lambda *a, **kw: ("/tmp/dummy.stl", "/tmp/dummy.sac",
                                          {"nabla": 0.25, "waterplane_area": 1.0},
                                          "/tmp/hull.stl"))
    monkeypatch.setattr("hull_opt.high_fidelity._gate_fine_cfd",
                        lambda *a, **kw: (True, 50.0, 200.0, "mocked gate1"))
    monkeypatch.setattr("hull_opt.high_fidelity._gate_self_righting",
                        lambda *a, **kw: (True, 10.0, 30.0, "mocked gate3"))
    monkeypatch.setattr("hull_opt.high_fidelity._gate_inverted_pressure",
                        lambda *a, **kw: (True, 30000.0, 100000.0,
                                          "mocked gate5"))
    from hull_opt.config import load_config
    config = load_config("config.yaml")

    design_vector = np.zeros(17)
    result = _validate_single(design_vector, 1, config, rapid=rapid)

    g2 = result.gates["wave_motions_accel"]
    assert g2["status"] == "OK"
    assert g2["value"] == pytest.approx(5.2)
    assert g2["passed"]

    g4 = result.gates["drop_impact_accel"]
    assert g4["status"] == "OK"
    assert g4["value"] == pytest.approx(45.0)
    assert "rapid-gate" in g4["details"]

    g5 = result.gates["inverted_pressure"]
    assert g5["passed"]
    assert "rapid-gate" in g5["details"]


def test_high_fi_gate1_stays_cfd(monkeypatch):
    """Gate 1 still runs CFD (not short-circuited by rapid gates)."""
    from hull_opt.high_fidelity import _validate_single

    rapid = RapidGateResult()
    rapid.storm_peak_accel_g = 0.1
    rapid.slam_pressure_pa = 1000.0
    rapid.inverted_pressure_pa = 500.0

    gate1_called = False

    def mock_gate1(*a, **kw):
        nonlocal gate1_called
        gate1_called = True
        return True, 50.0, 200.0, "mocked gate1"

    monkeypatch.setattr("hull_opt.geometry_validator.validate_design_vector",
                        lambda *a, **kw: (True, "ok"))
    monkeypatch.setattr("hull_opt.high_fidelity.generate_hull",
                        lambda *a, **kw: ("/tmp/dummy.stl", "/tmp/dummy.sac",
                                          {"nabla": 0.25, "waterplane_area": 1.0},
                                          "/tmp/hull.stl"))
    monkeypatch.setattr("hull_opt.high_fidelity._gate_fine_cfd", mock_gate1)
    monkeypatch.setattr("hull_opt.high_fidelity._gate_self_righting",
                        lambda *a, **kw: (True, 10.0, 30.0, "mocked gate3"))
    monkeypatch.setattr("hull_opt.high_fidelity._gate_inverted_pressure",
                        lambda *a, **kw: (True, 30000.0, 100000.0,
                                          "mocked gate5"))
    from hull_opt.config import load_config
    config = load_config("config.yaml")

    design_vector = np.zeros(17)
    _validate_single(design_vector, 1, config, rapid=rapid)

    assert gate1_called, "gate1 should still run CFD even with rapid gates"


def test_mid_fi_defaults(monkeypatch):
    """Mid-fidelity defaults use SPH solver by default."""
    class MinimalCalibration:
        solver = "sph"
        sph_dp = 0.03
        sph_sim_time = 10.0
        timeout = 7200

    class MinimalPaths:
        output_dir = "/tmp"
        dualsphysics_dir = "/home/anon/apps/DualSPHysics_v5.4"

    class MinimalFixed:
        target_speed_knots = 8.0
        target_displacement = 0.25
        rho_water = 1025.0
        nu_water = 1.2e-6
        gravity = 9.81
        electronics_bay = (0.0, 0.0, 0.0)

    class MinimalBounds:
        LWL = (2.3, 2.5)
        BWL = (0.40, 0.60)
        T_canoe = (0.15, 0.35)
        Cp = (0.55, 0.65)
        Cm = (0.60, 0.90)
        LCB = (30.0, 60.0)
        D_keel = (0.85, 1.20)
        keel_chord = (0.15, 0.25)
        bulb_vol = (0.001, 0.004)
        bulb_pos = (0.30, 0.50)
        E = (0.15, 0.30)
        flare = (8.0, 15.0)
        deadrise = (5.0, 25.0)
        bilge_r = (0.05, 0.30)
        keel_rake = (0.001, 0.02)
        ballast_frac = (0.30, 0.70)
        wingsail_pos = (0.30, 0.55)

    class MinimalConfig:
        calibration = MinimalCalibration()
        paths = MinimalPaths()
        fixed = MinimalFixed()
        bounds = MinimalBounds()
        optimization = None
        validation = None
        wave_spectrum = None

    cfg = MinimalConfig()

    from hull_opt.mid_fidelity import run_mid_fidelity_calibration

    dv = np.zeros(17)
    monkeypatch.setattr("hull_opt.geometry_validator.validate_design_vector",
                        lambda *a, **kw: (True, "ok"))
    monkeypatch.setattr("hull_opt.mid_fidelity.generate_hull",
                        lambda *a, **kw: ("/tmp/dummy.stl", "/tmp/dummy.sac",
                                          {"nabla": 0.25}, "/tmp/hull.stl"))
    monkeypatch.setattr("hull_opt.preflight.preflight_case",
                        lambda *a, **kw: (True, []))
    monkeypatch.setattr("hull_opt.sph_resistance.run_towing_resistance",
                        lambda *a, **kw: {"rt_n": None, "status": "FAILED",
                                           "details": "mocked failure"})

    result = run_mid_fidelity_calibration(dv, 1, 0, cfg)
    assert result is None
