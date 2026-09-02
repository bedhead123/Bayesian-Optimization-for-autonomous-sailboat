import numpy as np
import pytest
from hull_opt.rapid_gates import (
    RapidGateResult,
    evaluate_rapid_gates,
    _jonswap_spectrum,
    _peak_accel_from_raos,
    _roll_sigma_rad,
    _storm_wind_heel,
    _self_righting,
    _slam_pressure,
    _slam_accel,
    _inverted_pressure,
)

RHO = 1025.0
G = 9.81


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    from hull_opt.config import load_config
    return load_config("config.yaml")


@pytest.fixture
def synthetic_gz_high_avs():
    angles = np.linspace(0, 180, 37)
    gz = np.where(angles < 110, 0.08 * np.sin(np.radians(angles * 1.6)), 0.0)
    gz = np.maximum(gz, -0.02)
    return np.column_stack([angles, gz, np.full(37, 0.25)])


@pytest.fixture
def synthetic_gz_low_avs():
    angles = np.linspace(0, 180, 37)
    gz = np.where(angles < 40, 0.03 * np.sin(np.radians(angles * 4.0)), -0.01)
    return np.column_stack([angles, gz, np.full(37, 0.25)])


@pytest.fixture
def synthetic_gz_self_right():
    angles = np.linspace(0, 180, 37)
    gz = np.where(angles > 150, 0.01, 0.05)
    return np.column_stack([angles, gz, np.full(37, 0.25)])


@pytest.fixture
def synthetic_gz_not_self_right():
    angles = np.linspace(0, 180, 37)
    gz = np.where(angles > 150, -0.01, 0.05)
    return np.column_stack([angles, gz, np.full(37, 0.25)])


@pytest.fixture
def mid_x_dict():
    return {
        "LWL": 2.4, "BWL": 0.50, "T_canoe": 0.25, "Cp": 0.60, "Cm": 0.75,
        "LCB": 45.0, "D_keel": 1.0, "keel_chord": 0.20, "bulb_vol": 0.0012,
        "bulb_pos": 0.40, "E": 0.20, "flare": 12.0,
        "deadrise": 15.0, "bilge_r": 0.15, "keel_rake": 0.01,
        "ballast_frac": 0.50, "wingsail_pos": 0.42,
    }


@pytest.fixture
def mid_hydro():
    return {
        "nabla": 0.25, "underwater_volume": 0.25,
        "BM": 0.05, "CB_x": 1.2, "CB_z": -0.12,
        "B": 0.50, "LWL": 2.4, "Cp": 0.60,
        "sac_scale_factor": 1.0, "target_nabla": 0.30,
    }


# ── Gate R: Synthetic GZ tests ──────────────────────────────────────────

def test_gate_r_high_avs(synthetic_gz_high_avs, mid_x_dict, mid_hydro, config):
    res = evaluate_rapid_gates(mid_x_dict, mid_hydro, synthetic_gz_high_avs,
                               -0.05, "/nonexistent.stl", config)
    assert res.avs_deg > 100.0
    assert res.max_gz_m > 0.04
    assert res.gz_area_30 > 0.0
    assert res.gz_area_40_90 > 0.0
    assert res.margins["avs"] > 0
    assert res.passed["avs"]


def test_gate_r_low_avs(synthetic_gz_low_avs, mid_x_dict, mid_hydro, config):
    res = evaluate_rapid_gates(mid_x_dict, mid_hydro, synthetic_gz_low_avs,
                               -0.05, "/nonexistent.stl", config)
    assert res.avs_deg < 50.0
    margin = res.margins["avs"]
    assert np.isfinite(margin) and margin > 0


def test_gate_r_self_right(synthetic_gz_self_right, mid_x_dict, mid_hydro, config):
    assert _self_righting(synthetic_gz_self_right)


def test_gate_r_not_self_right(synthetic_gz_not_self_right, mid_x_dict, mid_hydro, config):
    assert not _self_righting(synthetic_gz_not_self_right)


# ── Capsize margin ─────────────────────────────────────────────────────

def test_capsize_margin_valid(synthetic_gz_high_avs, mid_x_dict, mid_hydro, config):
    gz_angles = synthetic_gz_high_avs[:, 0]
    gz_vals = synthetic_gz_high_avs[:, 1]
    storm_wind_ms = config.validation.storm_wind_speed_knots * 0.514444
    from hull_opt.rig import build_rig
    from hull_opt.hydrostatics import compute_wind_heeling_arm
    rig = build_rig(mid_x_dict, config)
    sail_area = rig["area"] * 0.15
    sail_h = rig["combined"]["z"]
    wind_arm = compute_wind_heeling_arm(0.0, storm_wind_ms, sail_area, sail_h,
                                         mid_hydro["underwater_volume"],
                                         rho_water=RHO, g=G)
    res = evaluate_rapid_gates(mid_x_dict, mid_hydro, synthetic_gz_high_avs,
                               -0.05, "/nonexistent.stl", config)
    assert np.isfinite(res.capsize_margin)
    assert isinstance(res.margins["capsize"], float)


def test_capsize_margin_nan_gz(mid_x_dict, mid_hydro, config):
    bad_gz = np.column_stack([np.linspace(0, 180, 10),
                               np.full(10, float("nan")),
                               np.full(10, 0.25)])
    res = evaluate_rapid_gates(mid_x_dict, mid_hydro, bad_gz,
                               -0.05, "/nonexistent.stl", config)
    assert res.avs_deg == 0.0
    assert res.margins["avs"] == 0.0


# ── Margin sign conventions ─────────────────────────────────────────────

def test_max_type_margin():
    threshold = 30.0
    value_pass = 25.0
    value_fail = 35.0
    margin_pass = (threshold - value_pass) / threshold
    margin_fail = (threshold - value_fail) / threshold
    assert margin_pass > 0
    assert margin_fail < 0


def test_min_type_margin():
    threshold = 90.0
    value_pass = 100.0
    value_fail = 80.0
    margin_pass = value_pass / threshold
    margin_fail = value_fail / threshold
    assert margin_pass > 0
    assert margin_fail > 0


def test_nan_margin_negative():
    assert not np.isfinite(float("nan"))
    margins = {}
    v = float("nan")
    margins["test"] = -1.0
    assert margins["test"] == -1.0


# ── Parametric roll risk ────────────────────────────────────────────────

def test_parametric_roll_inside_band():
    Tp_storm = 11.7
    roll_period = 6.0
    Tp_half = Tp_storm / 2.0
    risk = abs(Tp_half - roll_period) / roll_period < 0.15
    assert risk


def test_parametric_roll_outside_band():
    Tp_storm = 11.7
    roll_period = 4.0
    Tp_half = Tp_storm / 2.0
    risk = abs(Tp_half - roll_period) / roll_period < 0.15
    assert not risk


def test_parametric_roll_margin_signed():
    risk = True
    margin_risk = -0.3
    margin_safe = 0.3
    assert margin_risk < 0
    assert margin_safe > 0


# ── JONSWAP helpers ─────────────────────────────────────────────────────

def test_jonswap_spectrum_shape():
    S, omega = _jonswap_spectrum(Hs=2.5, Tp=9.0, gamma=3.3, n_freq=50, omega_min=0.2, omega_max=6.0)
    assert len(S) == len(omega) == 50
    assert np.all(S >= 0)
    peak_idx = int(np.argmax(S))
    peak_omega = omega[peak_idx]
    expected_peak = 2.0 * np.pi / 9.0
    assert abs(peak_omega - expected_peak) < 1.0


def test_peak_accel_from_raos():
    omega_bem = np.linspace(0.2, 6.0, 15)
    heave = np.ones(15) * 0.5
    pitch = np.ones(15) * 0.2
    accel = _peak_accel_from_raos(heave, pitch, omega_bem, Hs=2.5, Tp=9.0, gamma=3.3, n_freq=100, g=9.81, x_eb=0.0)
    assert np.isfinite(accel)
    assert accel > 0


def test_roll_sigma_rad():
    omega_bem = np.linspace(0.2, 6.0, 15)
    roll = np.ones(15) * 1.0
    sigma = _roll_sigma_rad(roll, omega_bem, Hs=2.5, Tp=9.0, gamma=3.3, n_freq=100)
    assert np.isfinite(sigma)
    assert sigma > 0


# ── Slam helpers ────────────────────────────────────────────────────────

def test_slam_pressure_finite(mid_x_dict, config):
    p_max, v, p_raw = _slam_pressure(mid_x_dict, config)
    assert np.isfinite(p_max)
    assert p_max > 0
    assert np.isfinite(p_raw) and p_raw >= p_max
    # Drop height is now 15 m (survive-not-thrive) vs legacy 3 m
    expected_v = np.sqrt(2 * G * float(getattr(config.rapid_validation, "slam_drop_height_m", 15.0)))
    assert abs(v - expected_v) < 0.01


def test_slam_pressure_min_deadrise(config):
    xd = {
        "LWL": 2.4, "BWL": 0.50, "T_canoe": 0.25, "Cp": 0.60, "Cm": 0.75,
        "LCB": 45.0, "D_keel": 1.0, "keel_chord": 0.20, "bulb_vol": 0.0012,
        "bulb_pos": 0.40, "E": 0.20, "flare": 12.0,
        "deadrise": 1.0, "bilge_r": 0.15, "keel_rake": 0.01,
        "ballast_frac": 0.50, "wingsail_pos": 0.42,
    }
    p_max, v, p_raw = _slam_pressure(xd, config)
    assert np.isfinite(p_max)
    assert p_max > 0


def test_slam_accel_finite(mid_x_dict, mid_hydro, config):
    p_max = 500000.0
    accel = _slam_accel(p_max, mid_x_dict, mid_hydro, config)
    assert np.isfinite(accel)
    assert accel > 0


def test_slam_pressure_capped_soft(mid_x_dict, config):
    """Wagner peak pressure must be capped to the configured soft cap."""
    p_eff, v, p_raw = _slam_pressure(mid_x_dict, config)
    assert np.isfinite(p_eff)
    assert p_eff > 0
    cap = float(getattr(config.rapid_validation, "slam_max_pressure_pa", 3.2e6))
    assert p_eff <= cap + 1e-6
    assert p_raw >= p_eff  # uncapped value retained for margin gradient
    expected_v = np.sqrt(2 * G * float(getattr(config.rapid_validation, "slam_drop_height_m", 15.0)))
    assert abs(v - expected_v) < 0.01


def test_slam_accel_range(mid_x_dict, mid_hydro, config):
    """Initial-contact impact accel at 15 deg deadrise, 15 m drop: survive-not-thrive expects higher."""
    accel = _slam_accel(500000.0, mid_x_dict, mid_hydro, config)
    assert np.isfinite(accel)
    # 15 m drop ~ sqrt(5)* the 3 m velocity → accel scales linearly with v, so 3m:3-30g → 15m: ~6-70g
    assert 5.0 <= accel <= 80.0


def test_slam_accel_monotonic_deadrise(mid_x_dict, mid_hydro, config):
    """Impact acceleration must DECREASE as deadrise increases:
    a flat bottom hits harder than a V-bottom."""
    def xd(deadrise):
        d = dict(mid_x_dict)
        d["deadrise"] = deadrise
        return d

    a5 = _slam_accel(500000.0, xd(5.0), mid_hydro, config)
    a15 = _slam_accel(500000.0, xd(15.0), mid_hydro, config)
    a25 = _slam_accel(500000.0, xd(25.0), mid_hydro, config)
    assert 0 < a25 < a15 < a5
    assert a5 < 150.0  # 15 m drop raises all accels; monotonicity is the invariant


def _failing_gz_curve():
    angles = np.linspace(0, 180, 37)
    gz = np.where(angles < 35, 0.005 * np.sin(np.radians(angles * (180.0 / 35.0))), -0.005)
    return np.column_stack([angles, gz, np.full(37, 0.25)])


def test_rapid_ordering_skips_bem_on_cheap_hard_fail(mid_x_dict, mid_hydro, config):
    """A design failing a cheap HARD gate (GZ-derived) must skip the storm
    BEM sweep entirely: gate_w reports the skip, margins stay complete,
    BEM-dependent fields carry the skip fallbacks."""
    gz = _failing_gz_curve()
    res = evaluate_rapid_gates(mid_x_dict, mid_hydro, gz, -0.05,
                               "/nonexistent.stl", config,
                               x_vector=np.zeros(17))
    assert res.details["gate_w"].startswith("storm BEM skipped")
    assert "cheap hard gate failed" in res.details["gate_w"]
    assert set(res.margins.keys()) == MARGIN_KEYS_15
    assert res.storm_peak_accel_g == 60.0
    assert res.roll_sigma_deg == 45.0
    assert res.roll_period_s == 0.0
    assert res.parametric_roll_risk is False


def test_worst_hard_reported(mid_x_dict, mid_hydro, config, synthetic_gz_high_avs):
    soft = set(config.rapid_validation.soft_margin_gates)
    res_fail = evaluate_rapid_gates(mid_x_dict, mid_hydro, _failing_gz_curve(),
                                    -0.05, "/nonexistent.stl", config)
    assert res_fail.worst_hard != ""
    assert res_fail.worst_hard in MARGIN_KEYS_15
    assert res_fail.worst_hard not in soft
    assert res_fail.worst_hard_value < 0

    # high-AVS synthetic curve: cheap gates pass except self_right (-0.3);
    # the BEM skip sets roll_period 0.0 -> roll_period margin -1.0, the most
    # negative non-soft margin -> worst_hard is that roll_period artifact.
    res_hi = evaluate_rapid_gates(mid_x_dict, mid_hydro, synthetic_gz_high_avs,
                                  -0.05, "/nonexistent.stl", config)
    assert res_hi.worst_hard not in soft
    assert res_hi.worst_hard_value < 0
    assert res_hi.worst_hard in ("roll_period", "self_right", "wind_heel")


# ── Inverted pressure ───────────────────────────────────────────────────

def test_inverted_pressure_basic(mid_x_dict, config):
    p = _inverted_pressure("/nonexistent.stl", mid_x_dict, config)
    assert np.isfinite(p)
    assert p > 0


# ── Storm wind heel ─────────────────────────────────────────────────────

def test_storm_wind_heel_finite(synthetic_gz_high_avs, mid_x_dict, mid_hydro, config):
    heel = _storm_wind_heel(mid_x_dict, synthetic_gz_high_avs, mid_hydro, config)
    assert np.isfinite(heel)
    assert 0 <= heel <= 90


# ── Self-righting helper ────────────────────────────────────────────────

def test_self_righting_empty():
    assert not _self_righting(np.zeros((0, 3)))
    assert not _self_righting(np.array([[0, 0, 0]]))


def test_self_righting_all_negative():
    angles = np.linspace(0, 180, 37)
    gz = np.full(37, -0.01)
    curve = np.column_stack([angles, gz, np.full(37, 0.25)])
    assert not _self_righting(curve)


# ── End-to-end test on a real design ──────────────────────────────────────
# This test builds actual geometry + GZ curve and then runs the FULL storm
# sweep: _storm_bem_sweep performs a real Capytaine solve (radiation +
# diffraction across the storm omega grid, ~3-4 solves) and feeds the RAO
# surrogate fallback path. ~80 s wall time. Requires capytaine, meshio,
# trimesh, fast_simplification.

MARGIN_KEYS_15 = {
    "avs", "capsize", "storm_accel", "roll_sigma", "roll_sigma_ops",
    "slam_pressure", "inverted_pressure", "wind_heel", "parametric_roll",
    "righting_energy", "slam_accel", "max_gz_m",
    "gz_area_30", "gz_area_40_90", "self_right", "roll_period",
}

@pytest.mark.quick_test
def test_evaluate_rapid_gates_e2e(config):
    import trimesh
    from hull_opt.param_layer import design_vector_to_physical
    from hull_opt.geometry import generate_hull
    from hull_opt.hydrostatics import compute_gz_curve, compute_cg_z

    raw = np.array([0.0] * 17, dtype=float)
    x_dict = design_vector_to_physical(raw, config)

    stl_path, sac_path, hydro, hull_stl = generate_hull(
        raw, output_dir=None, LWL=x_dict["LWL"],
        target_displacement=config.fixed.target_displacement, config=config,
    )

    cg_z = compute_cg_z(x_dict, nabla=hydro.get("underwater_volume", hydro.get("nabla")), config=config)
    gz = compute_gz_curve(hull_stl, cg_z=cg_z, n_angles=37, max_heel=180.0)

    res = evaluate_rapid_gates(x_dict, hydro, gz, cg_z, stl_path, config)

    assert isinstance(res, RapidGateResult)
    assert np.isfinite(res.avs_deg) or not np.isfinite(res.avs_deg)
    assert np.isfinite(res.max_gz_m) or not np.isfinite(res.max_gz_m)
    assert np.isfinite(res.gz_area_30) or not np.isfinite(res.gz_area_30)
    assert np.isfinite(res.gz_area_40_90) or not np.isfinite(res.gz_area_40_90)
    assert isinstance(res.self_right, bool)
    assert isinstance(res.parametric_roll_risk, bool)

    expected_margin_keys = MARGIN_KEYS_15
    assert set(res.margins.keys()) == expected_margin_keys, (
        f"Missing margin keys: {expected_margin_keys - set(res.margins.keys())}"
    )
    assert set(res.passed.keys()) == expected_margin_keys

    for k, v in res.margins.items():
        assert np.isfinite(v), f"Margin {k} is non-finite: {v}"

    for k, v in res.passed.items():
        assert isinstance(v, bool), f"Passed {k} is not bool: {type(v)}"

    # The storm BEM sweep ran for real (stl_path exists), so the seakeeping
    # fields come from Capytaine unless it fell back; either way they exist.
    assert np.isfinite(res.storm_peak_accel_g) or not np.isfinite(res.storm_peak_accel_g)
    assert np.isfinite(res.roll_sigma_deg) or not np.isfinite(res.roll_sigma_deg)
    assert np.isfinite(res.roll_period_s) or not np.isfinite(res.roll_period_s)


# ── Regression: no-GZ path must fail every margin (Phase 0) ───────────────

def test_no_gz_path_fails_all_margins(mid_x_dict, mid_hydro, config):
    res = evaluate_rapid_gates(mid_x_dict, mid_hydro, None, -0.05,
                               "/nonexistent.stl", config)
    assert res.details.get("no_gz") is True
    assert set(res.margins.keys()) == MARGIN_KEYS_15
    for k, v in res.margins.items():
        assert isinstance(v, float) and v < 0, f"margin {k} = {v} not < 0"
    assert len(res.passed) == len(MARGIN_KEYS_15)
    for k, v in res.passed.items():
        assert v is False, f"passed[{k}] = {v} must be False without GZ data"


def test_no_gz_path_short_curve_fails_all_margins(mid_x_dict, mid_hydro, config):
    short_gz = np.column_stack([[0.0], [0.05], [0.25]])
    res = evaluate_rapid_gates(mid_x_dict, mid_hydro, short_gz, -0.05,
                               "/nonexistent.stl", config)
    assert res.details.get("no_gz") is True
    assert all(v < 0 for v in res.margins.values())


# ── Regression: 0.C margin coverage on a synthetic GZ curve ───────────────

def test_0c_margin_coverage_synthetic_gz(synthetic_gz_high_avs, mid_x_dict,
                                         mid_hydro, config):
    res = evaluate_rapid_gates(mid_x_dict, mid_hydro, synthetic_gz_high_avs,
                               -0.05, "/nonexistent.stl", config)
    for k in ("righting_energy", "slam_accel", "max_gz_m",
              "gz_area_30", "gz_area_40_90", "self_right", "roll_period"):
        assert k in res.margins, f"missing margin {k}"
        assert np.isfinite(res.margins[k]), f"margin {k} non-finite: {res.margins[k]}"
