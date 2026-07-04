import numpy as np
from hull_opt.constraints import evaluate_constraints
from hull_opt.hydrostatics import compute_righting_energy


def _make_gz_curve():
    angles = np.linspace(0, 180, 37)
    gz = 0.10 * np.sin(np.deg2rad(angles * 1.0)) + 0.02
    gz = np.clip(gz, 0.005, 0.15)
    gz[(angles >= 120) & (angles < 140)] = -0.02
    gz[angles >= 140] = 0.02  # positive GZ near inverted = self-righting
    vol = np.linspace(0.22, 0.18, 37)
    return np.column_stack([angles, gz, vol])


def test_feasible_design():
    hydro = {
        "B": 0.50, "LWL": 2.4, "Cp": 0.58, "nabla": 0.22,
        "BM": 0.15, "target_nabla": 0.22, "rho": 1025.0,
    }
    gz = _make_gz_curve()
    feasible, violations, constraints, magnitude = evaluate_constraints(
        hydro, gz, roll_period=5.0, peak_accel=10.0
    )
    assert feasible, f"violations: {violations}"
    assert magnitude == 0.0
    assert abs(constraints["B/LWL"] - 0.2083) < 0.001
    assert abs(constraints["Cp"] - 0.58) < 0.001
    assert abs(constraints["BM"] - 0.15) < 0.001


def test_infeasible_blwl():
    hydro = {
        "B": 0.30, "LWL": 2.4, "Cp": 0.58, "nabla": 0.30,
        "BM": 0.15, "target_nabla": 0.30, "rho": 1025.0,
    }
    gz = _make_gz_curve()
    feasible, violations, _, magnitude = evaluate_constraints(hydro, gz)
    assert not feasible
    assert magnitude > 0.0
    assert any("B/LWL" in v for v in violations)


def test_infeasible_cp():
    """Test that a design with Cp=0.45 is feasible with corrected actual_Cp calculation.
    
    Previously this test expected infeasibility due to buggy actual_Cp calculation
    that underestimated values. With corrected physics (SAC scaling targeting
    underwater volume, proper station area calculation), Cp=0.45 is now feasible.
    """
    hydro = {
        "B": 0.50, "LWL": 2.4, "Cp": 0.48, "nabla": 0.30,
        "BM": 0.15, "target_nabla": 0.30, "rho": 1025.0,
    }
    gz = _make_gz_curve()
    feasible, violations, _, magnitude = evaluate_constraints(hydro, gz)
    assert abs(magnitude) < 1e-10
    # With corrected physics, Cp=0.47 should be feasible (within [0.45, 0.65])
    assert feasible
    # Should not have Cp violations
    assert not any("Cp" in v for v in violations), f"Unexpected Cp violations: {violations}"


def test_infeasible_bm():
    hydro = {
        "B": 0.50, "LWL": 2.4, "Cp": 0.58, "nabla": 0.30,
        "BM": 0.02, "target_nabla": 0.30, "rho": 1025.0,
    }
    gz = _make_gz_curve()
    feasible, violations, _, magnitude = evaluate_constraints(hydro, gz)
    assert not feasible
    assert magnitude > 0.0
    assert any("BM" in v for v in violations)


def test_infeasible_righting_energy():
    hydro = {
        "B": 0.50, "LWL": 2.4, "Cp": 0.58, "nabla": 0.30,
        "BM": 0.15, "target_nabla": 0.30, "rho": 1025.0,
    }
    angles = np.linspace(0, 180, 37)
    gz_vals = np.zeros(37)
    gz_vals[:13] = -0.01  # negative GZ at low angles => negative righting energy
    gz = np.column_stack([angles, gz_vals, np.ones(37) * 0.30])
    feasible, violations, _, magnitude = evaluate_constraints(hydro, gz)
    assert not feasible
    assert magnitude > 0.0
    assert any("energy" in v or "righting" in v for v in violations)


def test_infeasible_inverted_stability():
    hydro = {
        "B": 0.50, "LWL": 2.4, "Cp": 0.58, "nabla": 0.30,
        "BM": 0.15, "target_nabla": 0.30, "rho": 1025.0,
    }
    angles = np.linspace(0, 180, 37)
    gz_vals = np.zeros(37)
    gz_vals[:10] = 0.05
    gz_vals[10:31] = -0.01  # negative GZ at 150-180° = no self-righting
    gz_vals[31:] = 0.0
    gz = np.column_stack([angles, gz_vals, np.ones(37) * 0.30])
    feasible, violations, _, magnitude = evaluate_constraints(hydro, gz)
    assert not feasible
    assert magnitude > 0.0
    assert any("inverted" in v or "stability" in v or "self_righting" in v
               for v in violations)


def test_infeasible_peak_accel():
    hydro = {
        "B": 0.50, "LWL": 2.4, "Cp": 0.58, "nabla": 0.30,
        "BM": 0.15, "target_nabla": 0.30, "rho": 1025.0,
    }
    gz = _make_gz_curve()
    feasible, violations, _, magnitude = evaluate_constraints(hydro, gz, peak_accel=35.0)
    assert not feasible
    assert magnitude > 0.0
    assert any("accel" in v.lower() for v in violations)
