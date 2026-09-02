import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hull_opt.config import load_config
from hull_opt.low_fidelity import evaluate_low_fidelity

config = load_config("config.yaml")
designs_path = Path("tests/golden/designs.json")
designs = json.loads(designs_path.read_text())["designs"]

results = {}
for i, dv in enumerate(designs):
    vec = np.array(dv, dtype=np.float64)
    output_dir = f"/tmp/golden_out/design_{i}"
    result = evaluate_low_fidelity(vec, config, output_dir=output_dir,
                                    drag_factor=1.0, iteration=i)
    entry = {
        "design_vector": dv,
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
        entry["rapid"] = {
            "avs_deg": r.avs_deg,
            "max_gz_m": r.max_gz_m,
            "capsize_margin": r.capsize_margin,
            "storm_peak_accel_g": r.storm_peak_accel_g,
            "roll_sigma_deg": r.roll_sigma_deg,
            "parametric_roll_risk": r.parametric_roll_risk,
            "slam_pressure_pa": r.slam_pressure_pa,
            "inverted_pressure_pa": r.inverted_pressure_pa,
            "storm_wind_heel_deg": r.storm_wind_heel_deg,
            "margins": {k: float(v) if v is not None else None
                       for k, v in (r.margins or {}).items()},
        }
    results[f"design_{i}"] = entry

out_path = Path("tests/golden/metrics.json")
out_path.write_text(json.dumps(results, indent=2, default=str))
print(f"Golden metrics written: {len(results)} designs -> {out_path}")
