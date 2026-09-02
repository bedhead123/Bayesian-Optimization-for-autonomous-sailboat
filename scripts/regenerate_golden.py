import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

GOLDEN_DIR = Path("tests/golden")

def run_once(out_dir):
    designs = json.loads((GOLDEN_DIR / "designs.json").read_text())["designs"]
    dv = np.array(designs[0], dtype=np.float64)
    from hull_opt.config import load_config
    from hull_opt.low_fidelity import evaluate_low_fidelity
    config = load_config("config.yaml")
    result = evaluate_low_fidelity(dv, config, output_dir=out_dir,
                                   drag_factor=1.0, iteration=0)
    assert result.error_code is None, f"Evaluation failed: {result.error_code}"
    assert np.isfinite(result.fom), f"Non-finite FoM: {result.fom}"
    mini = {
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
        mini["rapid"] = {
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
    return mini

if __name__ == "__main__":
    import tempfile
    r1 = run_once(tempfile.mkdtemp(prefix="golden_run1_"))
    print("RUN1:", json.dumps(r1, indent=2))
    r2 = run_once(tempfile.mkdtemp(prefix="golden_run2_"))
    print("RUN2:", json.dumps(r2, indent=2))
    same = {k: (r1[k] == r2[k] or (isinstance(r1[k], float) and abs(r1[k] - r2[k]) < 1e-12))
            for k in r1 if k != "rapid"}
    if "rapid" in r1 and "rapid" in r2:
        same["rapid"] = {k: (r1["rapid"][k] == r2["rapid"][k]) for k in r1["rapid"]}
    print("STABLE:", same)
    if all(bool(v) for v in same.values()):
        (GOLDEN_DIR / "metrics.json").write_text(
            json.dumps({"design_0": r1}, indent=2, default=str) + "\n")
        print("WROTE tests/golden/metrics.json")
    else:
        print("NOT STABLE - not writing")
