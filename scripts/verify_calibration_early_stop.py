"""Live single-design calibration with the convergence early-stop.

Re-meshes + runs design 60 (from the production DB) through the WP2 mid-fi
path: blockMesh -> snappyHexMesh (new quality settings) -> setFields ->
interFoam with the _ForceConvergencePoller -> tail-window extraction.

Usage: venv/bin/python scripts/verify_calibration_early_stop.py [iter]
Expected: solver stops at ~10-11 s sim time instead of 20 s; factor extracted
from the window_s=3 window matches the archived design-60 value (~1.0956).
"""
import sys
from pathlib import Path

ROOT = Path("/home/anon/apps/boat")
sys.path.insert(0, str(ROOT))

import json
import sqlite3

from hull_opt.config import load_config
from hull_opt.mid_fidelity import run_mid_fidelity_calibration

ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 9999


def main():
    config = load_config(str(ROOT / "config.yaml"))
    conn = sqlite3.connect(str(ROOT / "output" / "optimization.db"))
    row = conn.execute(
        "SELECT design_vector FROM designs WHERE id = 60"
    ).fetchone()
    conn.close()
    if row is None:
        print("design 60 not found in DB")
        return 1
    design_vector = json.loads(row[0])
    print(f"calibrating design 60 at iter {ITER} -> "
          f"output/calibration/design_60_iter_{ITER}")
    cal = run_mid_fidelity_calibration(
        design_vector, design_id=60, iteration=ITER, config=config
    )
    print(f"\nRESULT: {cal}")
    if cal is None:
        return 1
    print(f"factor={cal['factor']:.4f} rt_cfd={cal['rt_cfd']:.4f} "
          f"(archived design-60 factor=1.0956, rt_cfd=92.87 N)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
