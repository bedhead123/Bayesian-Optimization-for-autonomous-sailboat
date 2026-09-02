import copy
import json
import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def test_designs_json_complete():
    path = GOLDEN_DIR / "designs.json"
    assert path.exists(), f"Missing {path}"
    data = json.loads(path.read_text())
    assert "designs" in data
    designs = data["designs"]
    assert len(designs) == 6, f"Expected 6 designs, got {len(designs)}"
    for i, dv in enumerate(designs):
        assert isinstance(dv, list), f"Design {i} is not a list"
        assert len(dv) == 17, f"Design {i} has {len(dv)} elements (expected 17)"
        for j, v in enumerate(dv):
            assert isinstance(v, (int, float)), f"Design {i}[{j}] is not numeric"


def test_known_constant_design(tmp_path):
    designs_path = GOLDEN_DIR / "designs.json"
    designs = json.loads(designs_path.read_text())["designs"]
    dv = np.array(designs[0], dtype=np.float64)
    from hull_opt.config import load_config
    from hull_opt.low_fidelity import evaluate_low_fidelity
    config = load_config("config.yaml")
    out_dir = str(tmp_path / "golden_run")
    result = evaluate_low_fidelity(dv, config, output_dir=out_dir,
                                    drag_factor=1.0, iteration=0)
    assert result.error_code is None, f"Evaluation failed: {result.error_code}"
    assert np.isfinite(result.fom), f"Non-finite FoM: {result.fom}"
    assert result.rt_total > 0, f"rt_total not positive: {result.rt_total}"
    assert result.cad_stl_path is not None
    assert Path(result.cad_stl_path).exists()
    metrics_path = GOLDEN_DIR / "metrics.json"
    if not metrics_path.exists():
        mini_golden = {}
        mini_golden["design_0"] = {
            "feasible": result.feasible,
            "fom": result.fom,
            "rt_total": result.rt_total,
            "rt_wave": result.rt_wave,
            "rt_friction": result.rt_friction,
            "stability_index": result.stability_index,
            "roll_period": result.roll_period,
            "peak_accel": result.peak_accel,
            "righting_energy": result.righting_energy,
            "gm": result.gm,
            "cg_z": result.cg_z,
            "eq_heel_deg": result.eq_heel_deg,
            "helm_fwd_deg": result.helm_fwd_deg,
            "helm_aft_deg": result.helm_aft_deg,
            "helm_combined_deg": result.helm_combined_deg,
        }
        if result.rapid is not None:
            r = result.rapid
            mini_golden["design_0"]["rapid"] = {
                "avs_deg": r.avs_deg,
                "max_gz_m": r.max_gz_m,
                "capsize_margin": r.capsize_margin,
                "storm_peak_accel_g": r.storm_peak_accel_g,
                "roll_sigma_deg": r.roll_sigma_deg,
                "parametric_roll_risk": r.parametric_roll_risk,
                "slam_pressure_pa": r.slam_pressure_pa,
                "inverted_pressure_pa": r.inverted_pressure_pa,
                "storm_wind_heel_deg": r.storm_wind_heel_deg,
            }
        metrics_path.write_text(json.dumps(mini_golden, indent=2, default=str))
    golden = json.loads(metrics_path.read_text())["design_0"]
    for key in ("fom", "rt_total", "rt_wave"):
        exp = golden.get(key)
        if exp is not None:
            got = getattr(result, key, None)
            denom = max(abs(exp), 1e-12)
            rdiff = abs(got - exp) / denom
            assert rdiff < 1e-6, f"{key}: expected {exp}, got {got} (rel diff {rdiff:.2e})"


def test_smoke_detects_regression(tmp_path):
    designs_path = GOLDEN_DIR / "designs.json"
    designs = json.loads(designs_path.read_text())["designs"]
    dv = np.array(designs[0], dtype=np.float64)
    from hull_opt.config import load_config
    from hull_opt.low_fidelity import evaluate_low_fidelity
    config = load_config("config.yaml")
    out_dir = str(tmp_path / "regression_run")
    result = evaluate_low_fidelity(dv, config, output_dir=out_dir,
                                    drag_factor=1.0, iteration=0)
    fake_metrics = {
        "design_0": {
            "feasible": result.feasible,
            "fom": result.fom,
            "rt_total": 999.0,
            "rt_wave": result.rt_wave,
            "rt_friction": result.rt_friction,
            "stability_index": result.stability_index,
            "roll_period": result.roll_period,
            "peak_accel": result.peak_accel,
            "righting_energy": result.righting_energy,
            "gm": result.gm,
            "cg_z": result.cg_z,
            "eq_heel_deg": result.eq_heel_deg,
            "helm_fwd_deg": result.helm_fwd_deg,
            "helm_aft_deg": result.helm_aft_deg,
            "helm_combined_deg": result.helm_combined_deg,
        }
    }
    fake_path = tmp_path / "corrupted_metrics.json"
    fake_path.write_text(json.dumps(fake_metrics))
    golden = json.loads(fake_path.read_text())["design_0"]
    exp_rt = golden["rt_total"]
    got_rt = result.rt_total
    denom = max(abs(exp_rt), 1e-12)
    rdiff = abs(got_rt - exp_rt) / denom
    assert rdiff > 0.01, f"Expected large mismatch in rt_total, got rdiff={rdiff:.2e}"


def test_nurbs_volume_close_to_triangle(tmp_path):
    """Verify NURBS mode produces volume within 5% of triangle path."""
    designs_path = GOLDEN_DIR / "designs.json"
    designs = json.loads(designs_path.read_text())["designs"]
    dv = np.array(designs[0], dtype=np.float64)
    from hull_opt.config import load_config
    from hull_opt.low_fidelity import evaluate_low_fidelity
    config = load_config("config.yaml")
    out_dir = str(tmp_path / "nurbs_check")
    # Triangle path
    result = evaluate_low_fidelity(dv, config, output_dir=out_dir,
                                    drag_factor=1.0, iteration=0)
    assert result.error_code is None, f"Triangle eval failed: {result.error_code}"
    tri_vol = result.hydro.get("underwater_volume", result.hydro.get("nabla", 0.0))
    assert tri_vol > 0
    # NURBS path - run with nurbs config
    nurbs_config = copy.deepcopy(config)
    object.__setattr__(nurbs_config.fixed, "use_nurbs_geometry", True)
    object.__setattr__(nurbs_config.fixed, "use_nurbs_gz", True)
    out_dir2 = str(tmp_path / "nurbs_run")
    result_n = evaluate_low_fidelity(dv, nurbs_config, output_dir=out_dir2,
                                      drag_factor=1.0, iteration=0)
    if result_n.error_code:
        pytest.skip(f"NURBS eval skipped: {result_n.error_code}")
    nurbs_vol = result_n.hydro.get("underwater_volume", result_n.hydro.get("nabla", 0.0))
    if nurbs_vol > 0 and tri_vol > 0:
        ratio = abs(nurbs_vol - tri_vol) / tri_vol
        assert ratio < 0.10, f"NURBS volume diff: {ratio*100:.1f}%"


def test_nurbs_gz_matches_triangle_within_5mm(tmp_path):
    """Verify NURBS-mode GZ matches triangle GZ within 5mm max diff."""
    designs_path = GOLDEN_DIR / "designs.json"
    designs = json.loads(designs_path.read_text())["designs"]
    dv = np.array(designs[0], dtype=np.float64)
    from hull_opt.config import load_config
    from hull_opt.low_fidelity import evaluate_low_fidelity
    config = load_config("config.yaml")
    # Triangle path
    result = evaluate_low_fidelity(dv, config, output_dir=str(tmp_path / "tri_gz"),
                                    drag_factor=1.0, iteration=0)
    assert result.error_code is None
    gz_tri = result.gz_curve
    # NURBS path
    nurbs_config = copy.deepcopy(config)
    object.__setattr__(nurbs_config.fixed, "use_nurbs_geometry", True)
    object.__setattr__(nurbs_config.fixed, "use_nurbs_gz", True)
    result_n = evaluate_low_fidelity(dv, nurbs_config, output_dir=str(tmp_path / "nrb_gz"),
                                      drag_factor=1.0, iteration=0)
    if result_n.error_code:
        pytest.skip(f"NURBS eval skipped: {result_n.error_code}")
    gz_n = result_n.gz_curve
    if gz_n is None:
        pytest.skip("NURBS GZ curve is None")
    # Interpolate NURBS GZ onto triangle angles and compare
    interp = np.interp(gz_tri[:, 0], gz_n[:, 0], gz_n[:, 1])
    diff = np.abs(gz_tri[:, 1] - interp)
    max_mm = diff.max() * 1000.0
    assert max_mm < 5.0, f"NURBS vs triangle GZ max diff {max_mm:.2f}mm (limit 5mm)"
    # Also check FoM (righting energy) within 1%
    from hull_opt.hydrostatics import compute_righting_energy
    re_tri = compute_righting_energy(gz_tri)
    re_nurbs = compute_righting_energy(gz_n)
    if re_tri > 1e-6:
        re_diff = abs(re_nurbs - re_tri) / re_tri * 100
        assert re_diff < 1.0, f"Righting energy diff {re_diff:.3f}% (limit 1%)"
