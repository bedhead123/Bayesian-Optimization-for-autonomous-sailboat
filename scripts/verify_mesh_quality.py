"""WP2.6 mesh-quality verification: re-mesh the archived design_60 STL with the
new snappyHexMesh settings (nCellsBetweenLevels 4, maxNonOrtho 55, skewness 4,
relaxed 65, keel box level 5, n_layers 2) and run checkMesh.

Usage: venv/bin/python scripts/verify_mesh_quality.py [case_dir]
Expected: checkMesh maxNonOrtho < 55, max skewness < 4, mesh OK.
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hull_opt.config import load_config
from hull_opt.of_legacy import write_openfoam_case, run_of_command
from hull_opt.utils import knots_to_ms

ROOT = Path("/home/anon/apps/boat")
OUT = ROOT / "output" / "mesh_quality_check"


def main():
    config = load_config(str(ROOT / "config.yaml"))
    of_env = config.paths.openfoam_env

    conn = sqlite3.connect(str(ROOT / "output" / "optimization.db"))
    row = conn.execute(
        "SELECT design_vector, iter FROM designs WHERE id = 60"
    ).fetchone()
    conn.close()
    if row is None:
        print("design 60 not found in DB")
        return 1
    design_vector = json.loads(row[0])

    from hull_opt.param_layer import design_vector_to_physical
    x_dict = design_vector_to_physical(design_vector, config)
    hull_lwl = float(x_dict["LWL"])
    B = x_dict["BWL"]
    T_hull = x_dict["T_canoe"]
    D_keel = x_dict.get("D_keel", 0.0)
    T_total = T_hull + D_keel
    speed_ms = knots_to_ms(config.fixed.target_speed_knots)

    src_stl = ROOT / "output" / "calibration" / "design_60_iter_60" / "constant" / "triSurface" / "hull.stl"
    if not src_stl.exists():
        print(f"archived STL missing: {src_stl}")
        return 1

    case_dir = OUT / "design_60_remesh"
    if case_dir.exists():
        import shutil
        shutil.rmtree(case_dir)
    write_openfoam_case(
        case_dir=case_dir,
        stl_path=str(src_stl),
        speed_ms=speed_ms,
        LWL=hull_lwl, B=B, T=T_hull,
        rho=config.fixed.rho_water,
        nu=config.fixed.nu_water,
        gravity=config.fixed.gravity,
        mesh_levels=(2, 3),
        n_layers=2,
        solver="interFoam",
        end_time=20.0,
        delta_t=0.001,
        write_interval=0.1,
        max_cells=config.calibration.coarse_cells,
        max_co=config.calibration.max_co,
        T_total=T_total,
        keel_x=(0.4 * hull_lwl if x_dict.get("bulb_vol", 0.0) < 1e-6
                else x_dict.get("bulb_pos", 0.4) * hull_lwl),
        keel_chord=x_dict.get("keel_chord", 0.2),
        keel_depth=D_keel,
    )
    print(f"case written to {case_dir}")

    for cmd, timeout in ((["blockMesh"], 600), (["snappyHexMesh", "-overwrite"], 7200)):
        print(f"--- {cmd[0]} ---")
        proc = run_of_command(cmd + ["-case", str(case_dir)], case_dir, of_env,
                              timeout=timeout,
                              log_file=case_dir / f"log.{cmd[0]}")
        if proc.returncode != 0:
            print(f"{cmd[0]} FAILED rc={proc.returncode}")
            return 1
        print(f"{cmd[0]} OK")

    print("--- checkMesh ---")
    proc = run_of_command(["checkMesh", "-case", str(case_dir)], case_dir,
                          of_env, timeout=600)
    out = proc.stdout
    import re
    for pat in [r"Mesh non-orthogonality Max: [\d.]+ average: [\d.]+",
                r"Max skewness = [\d.]+",
                r"Min volume = [^\s]+\. Max volume = [^\s]+\.\s+Total volume = [^\s]+\.",
                r"Mesh OK\.",
                r"\*\*.*Mesh.*NOT OK"]:
        m = re.search(pat, out)
        if m:
            print("  " + m.group(0))
    if "Mesh OK." not in out:
        print("  checkMesh did not report Mesh OK")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
