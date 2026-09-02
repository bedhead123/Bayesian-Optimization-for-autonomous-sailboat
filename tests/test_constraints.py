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
        "B": 0.55, "LWL": 2.4, "Cp": 0.58, "nabla": 0.22,
        "BM": 0.15, "target_nabla": 0.22, "rho": 1025.0,
    }
    gz = _make_gz_curve()
    feasible, violations, constraints, magnitude = evaluate_constraints(
        hydro, gz, roll_period=5.0, peak_accel=10.0
    )
    assert feasible, f"violations: {violations}"
    assert magnitude == 0.0
    assert abs(constraints["B/LWL"] - 0.2292) < 0.001
    assert abs(constraints["Cp"] - 0.58) < 0.001
    assert abs(constraints["BM"] - 0.15) < 0.001


def test_infeasible_blwl():
    # B/LWL = 0.083 -> dev 2.17 > 2.0 = degenerate band (fatal). The penalty
    # band (dev 1-2) is a soft FoM signal since the 2026-08-25 severity split.
    hydro = {
        "B": 0.20, "LWL": 2.4, "Cp": 0.58, "nabla": 0.30,
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
        "B": 0.55, "LWL": 2.4, "Cp": 0.48, "nabla": 0.30,
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
    # BM = 0.003 < 0.005 = degenerate (fatal). The 0.005-0.03 band is a soft
    # FoM penalty since the 2026-08-25 severity split.
    hydro = {
        "B": 0.50, "LWL": 2.4, "Cp": 0.58, "nabla": 0.30,
        "BM": 0.003, "target_nabla": 0.30, "rho": 1025.0,
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


def test_volume_gated_against_true_target_not_target_eff():
    """SAC-gaming regression: a hull displacing only 27% of the config target
    must be infeasible even when hydro['target_nabla_eff'] matches the shrunken
    volume (the adaptive SAC target the geometry scaled toward)."""
    from hull_opt.config import load_config

    config = load_config("config.yaml")
    true_target = config.fixed.target_displacement
    shrunk = 0.27 * true_target  # design 16 pattern: 27% of 0.145 m³
    hydro = {
        "B": 0.40, "LWL": 2.4, "Cp": 0.55, "nabla": shrunk,
        "underwater_volume": shrunk,
        "BM": 0.15, "rho": 1025.0,
        "target_nabla": true_target,
        "target_nabla_eff": shrunk,  # matches the shrunken volume exactly
        "sac_scale_factor": 1.30,  # at the SAC cap
        "sac_scale_std": 0.10,
    }
    gz = _make_gz_curve()
    feasible, violations, constraints, magnitude = evaluate_constraints(
        hydro, gz, config=config,
    )
    assert not feasible, f"27%-of-target hull must be infeasible: {violations}"
    assert constraints["vol_ratio_error"] > 0.5
    assert any("Volume error" in v or "volume_error" in v for v in violations), violations
    assert magnitude > 0.0


def test_full_displacement_passes_volume_gate():
    """A hull at the true target passes the volume gate (control for the
    target_eff->true_target change)."""
    from hull_opt.config import load_config

    config = load_config("config.yaml")
    true_target = config.fixed.target_displacement
    hydro = {
        "B": 0.55, "LWL": 2.4, "Cp": 0.60, "nabla": true_target,
        "underwater_volume": true_target,
        "BM": 0.15, "rho": 1025.0,
        "target_nabla": true_target,
        "target_nabla_eff": true_target,
        "sac_scale_factor": 1.05,
        "sac_scale_std": 0.05,
    }
    gz = _make_gz_curve()
    feasible, violations, constraints, magnitude = evaluate_constraints(
        hydro, gz, config=config,
    )
    assert constraints["vol_ratio_error"] < 0.05
    assert not any("volume_error" in v for v in violations), violations
    assert feasible, f"full-displacement hull should stay feasible: {violations}"
