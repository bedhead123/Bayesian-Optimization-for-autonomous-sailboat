#!/home/anon/apps/boat/venv/bin/python
"""P0.5 SPH smoke test: rebuild design 38 at dp=0.05, sim 2 s."""
import sys, os, json, logging, time, shutil
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from hull_opt.config import load_config
from hull_opt.param_layer import design_vector_to_physical
from hull_opt.geometry import generate_hull
from hull_opt.sph_resistance import (
    build_towing_case, run_sph_case, compute_forces, extract_towing_force
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sph_smoke")

config = load_config("config.yaml")
object.__setattr__(config.calibration, "sph_dp", 0.05)
object.__setattr__(config.calibration, "sph_sim_time", 2.0)
object.__setattr__(config.calibration, "timeout", 1200)

dv = np.array([6.7355, 1.2905, 2.1148, -8.9733, -9.0206, 5.6420,
               0.0742, 8.3527, -3.1485, -1.0567, 7.1563, -0.1374,
               -6.5480, 4.8676, 7.2190, 7.8127, 8.0476, -9.1720], dtype=np.float64)

smoke_root = Path("/tmp/sph_smoke")
if smoke_root.exists():
    shutil.rmtree(smoke_root)
smoke_root.mkdir(parents=True)

x_dict = design_vector_to_physical(dv, config)
logger.info("Design: LWL=%.3f BWL=%.3f T_c=%.3f D_k=%.3f deadrise=%.1f",
            x_dict["LWL"], x_dict["BWL"], x_dict["T_canoe"], x_dict.get("D_keel",0), x_dict.get("deadrise",0))

# Generate hull (pass the raw design vector, not x_dict)
geo_result = generate_hull(
    dv, output_dir=str(smoke_root / "hull"),
    LWL=x_dict["LWL"],
    target_displacement=config.fixed.target_displacement,
    config=config,
)
stl_path, sac_path, hydro, hull_stl = geo_result
if not hull_stl or not Path(hull_stl).exists():
    logger.error("STL generation failed"); sys.exit(1)

case_dir = smoke_root / "tow_case"
case_dir.mkdir(parents=True)

build_towing_case(case_dir, hull_stl, x_dict, config,
                  speed_ms=config.fixed.target_speed_knots * 0.514444)

gpu_lock = str(smoke_root / "gpu.lock")

result = run_sph_case(case_dir, config, gpu_lock=gpu_lock)
status = result.get("status")
np_p = result.get("np_particles", 0)
t_wall = result.get("wall_time_s", 0)
logger.info("run_sph_case status=%s np=%d wall=%.1fs", status, np_p, t_wall)

if status != "OK":
    logger.error("SPH failed: %s", result.get("details",""))
    sys.exit(1)

# ComputeForces
csv = compute_forces(case_dir)
if csv is None:
    logger.error("ComputeForces produced no output")
    sys.exit(1)
logger.info("Forces CSV: %s", csv)

rt = extract_towing_force(csv, tail_frac=0.2)
if rt is None:
    logger.error("Rt extraction failed")
    sys.exit(1)
logger.info("Rt = %.4f N", rt)

rho = config.fixed.rho_water
speed_ms = config.fixed.target_speed_knots * 0.514444
drag = rt / (0.5 * rho * speed_ms ** 2 * (x_dict["BWL"] * x_dict["LWL"]))
logger.info("Drag factor = %.4f", drag)

if 0.05 <= drag <= 5.0:
    logger.info("P0.5 SMOKE TEST PASS: drag_factor=%.4f", drag)
else:
    logger.error("P0.5 SMOKE TEST FAIL: drag_factor=%.4f", drag)
    sys.exit(1)
