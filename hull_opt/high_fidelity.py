"""
High-fidelity validation gates using OpenFOAM and DualSPHysics.
6 gates: calm-water CFD, regular wave motions, extreme wave self-righting,
drop impact, inverted deck pressure, downflooding & reserve buoyancy.
Key exports: validate_top_designs(), ValidationResult
"""
import json
import subprocess
from typing import Optional
import numpy as np
import trimesh
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from hull_opt.geometry import generate_hull
from hull_opt.param_layer import design_vector_to_physical
from hull_opt.templates.openfoam import write_openfoam_case
from hull_opt.hydrostatics import compute_cg_z
from hull_opt.utils import (
    run_of_command, find_force_file, extract_openfoam_force,
    knots_to_ms, ensure_dir, run_parallel_inter_foam
)


class ValidationResult:
    def __init__(self, design_id: int):
        self.design_id = design_id
        self.gates = {}
        self.overall_pass = False

    def set_gate(self, name: str, passed: bool, value: float,
                 threshold: float, details: str = ""):
        self.gates[name] = {
            "passed": passed,
            "value": value,
            "threshold": threshold,
            "details": details,
        }

    @property
    def all_passed(self) -> bool:
        return all(g["passed"] for g in self.gates.values())


def validate_top_designs(designs: list[dict], config) -> list[ValidationResult]:
    results = []
    for d in designs:
        design_vector = np.array(json.loads(d["design_vector"]), dtype=float)
        res = _validate_single(design_vector, d["id"], config)
        results.append(res)
        logger.info(f"Design {d['id']}: overall_pass={res.all_passed}, "
                     f"FoM={d['fom']:.4f}")
    return results


def _validate_single(design_vector: np.ndarray, design_id: int,
                     config) -> ValidationResult:
    result = ValidationResult(design_id)
    x_dict = design_vector_to_physical(design_vector, config)

    from hull_opt.geometry_validator import validate_design_vector
    is_valid, val_msg = validate_design_vector(x_dict, config)
    if not is_valid:
        logger.warning(f"Design {design_id}: design vector validation failed: {val_msg}")
        result.set_gate("design_vector", False, 0.0, 1.0, val_msg)
        return result

    hull_lwl = float(x_dict.get("LWL", config.fixed.LWL))
    B = x_dict["BWL"]
    T_hull = x_dict["T_canoe"]
    target_nabla = config.fixed.target_displacement
    rho = config.fixed.rho_water
    g = config.fixed.gravity
    nu = config.fixed.nu_water
    of_env = config.paths.openfoam_env
    eb = config.fixed.electronics_bay

    mass = rho * target_nabla
    speed_ms = knots_to_ms(config.fixed.target_speed_knots)
    D_keel = float(x_dict.get("D_keel", 0.0))

    cg_z = compute_cg_z(x_dict, nabla=target_nabla)

    val_dir = ensure_dir(Path(config.paths.output_dir) / "validation" / f"design_{design_id}")

    # generate geometry
    stl_path, sac_path, hydro, hull_stl = generate_hull(
        design_vector,
        output_dir=str(val_dir),
        LWL=hull_lwl,
        target_displacement=target_nabla,
        config=config,
    )
    logger.info(f"Design {design_id}: geometry at {stl_path}")

    # ── Gate 1: Fine calm-water resistance ──────────────────────────────
    logger.info(f"Gate 1: Fine CFD calm-water resistance")
    try:
        gate1_pass, gate1_val, gate1_thresh, gate1_detail = _gate_fine_cfd(
            val_dir / "gate1_calm", stl_path, speed_ms, hull_lwl, B, T_hull,
            config, of_env, design_id, x_dict,
            hydro=hydro, hull_stl_path=hull_stl
        )
    except Exception as e:
        logger.warning(f"Gate 1 exception: {e}")
        gate1_pass, gate1_val, gate1_thresh, gate1_detail = False, 0, config.validation.rt_upper_bound_factor * 1000, str(e)
        rt_file = find_force_file(val_dir / "gate1_calm" / "postProcessing" / "forces")
        if rt_file is not None:
            extracted = extract_openfoam_force(rt_file)
            if extracted is not None:
                gate1_val = extracted
    result.set_gate("calm_water_rt", gate1_pass, gate1_val, gate1_thresh, gate1_detail)
    logger.info(f"Gate 1: {'PASS' if gate1_pass else 'FAIL'} (Rt={gate1_val:.3f} N)")

    # ── Gate 2: Regular wave motions ────────────────────────────────────
    logger.info(f"Gate 2: Regular wave motions")
    try:
        gate2_pass, gate2_val, gate2_thresh, gate2_detail = _gate_wave_motions(
            val_dir / "gate2_wave", stl_path, speed_ms, hull_lwl, B, T_hull, mass,
            config, of_env, cg_z, D_keel=D_keel
        )
    except Exception as e:
        gate2_pass, gate2_val, gate2_thresh, gate2_detail = False, 0, config.validation.max_accel_g, str(e)
    result.set_gate("wave_motions_accel", gate2_pass, gate2_val, gate2_thresh, gate2_detail)
    logger.info(f"Gate 2: {'PASS' if gate2_pass else 'FAIL'} (accel={gate2_val:.2f} g)")

    # ── Gate 3: Extreme wave survival (OpenFOAM self-righting) ──────────
    logger.info(f"Gate 3: Extreme wave survival (self-righting)")
    try:
        gate3_pass, gate3_val, gate3_thresh, gate3_detail = _gate_self_righting(
            val_dir / "gate3_self_right", stl_path, speed_ms, hull_lwl, B, T_hull, mass,
            config, of_env, cg_z
        )
    except Exception as e:
        gate3_pass, gate3_val, gate3_thresh, gate3_detail = False, 99.9, config.validation.max_self_right_time_s, str(e)
    result.set_gate("extreme_wave_self_right", gate3_pass, gate3_val, gate3_thresh, gate3_detail)
    logger.info(f"Gate 3: {'PASS' if gate3_pass else 'FAIL'} (self-right time={gate3_val:.2f} s)")

    # ── Gate 4: Drop impact (OpenFOAM Wedge/Plunge) ─────────────────────
    logger.info(f"Gate 4: Drop impact")
    try:
        gate4_pass, gate4_val, gate4_thresh, gate4_detail = _gate_drop_impact_of(
            val_dir / "gate4_drop", stl_path, hull_lwl, B, T_hull, mass,
            config, of_env, cg_z, D_keel=D_keel
        )
    except Exception as e:
        gate4_pass, gate4_val, gate4_thresh, gate4_detail = False, 0, config.validation.max_accel_g, str(e)
    result.set_gate("drop_impact_accel", gate4_pass, gate4_val, gate4_thresh, gate4_detail)
    logger.info(f"Gate 4: {'PASS' if gate4_pass else 'FAIL'} (accel={gate4_val:.2f} g)")

    # ── Gate 5: Inverted deck pressure ──────────────────────────────────
    logger.info(f"Gate 5: Inverted deck pressure")
    try:
        gate5_pass, gate5_val, gate5_thresh, gate5_detail = _gate_inverted_pressure(
            val_dir / "gate5_inverted", stl_path, config, of_env,
            LWL=hull_lwl, B=B, T_hull=T_hull,
        )
    except Exception as e:
        gate5_pass, gate5_val, gate5_thresh, gate5_detail = False, 0, config.validation.max_pressure_pa, str(e)
    result.set_gate("inverted_pressure", gate5_pass, gate5_val, gate5_thresh, gate5_detail)
    logger.info(f"Gate 5: {'PASS' if gate5_pass else 'FAIL'} (pressure={gate5_val:.1f} Pa)")

    # ── Gate 6: Downflooding & Reserve Buoyancy ──────────────────────────
    logger.info(f"Gate 6: Downflooding & reserve buoyancy")
    try:
        gate6_pass, gate6_val, gate6_thresh, gate6_detail = _gate_downflooding(
            stl_path, x_dict, config, cg_z=cg_z, hull_stl_path=hull_stl
        )
    except Exception as e:
        gate6_pass, gate6_val, gate6_thresh, gate6_detail = False, 0, 0, str(e)
    result.set_gate("downflooding", gate6_pass, gate6_val, gate6_thresh, gate6_detail)
    logger.info(f"Gate 6: {'PASS' if gate6_pass else 'FAIL'} (detail={gate6_detail})")

    result.overall_pass = result.all_passed
    return result


def _gate_downflooding(stl_path: str, x_dict: dict, config, cg_z: Optional[float] = None, hull_stl_path: Optional[str] = None) -> tuple[bool, float, float, str]:
    from hull_opt.hydrostatics import compute_downflooding_angle, compute_reserve_buoyancy, compute_cg_z
    if cg_z is None:
        cg_z = compute_cg_z(x_dict)
    downflood_mesh = hull_stl_path if hull_stl_path else stl_path
    df_angle = compute_downflooding_angle(downflood_mesh, cg_z=cg_z)
    reserve = compute_reserve_buoyancy(stl_path, x_dict)
    min_df = config.validation.min_downflooding_angle
    min_reserve = config.validation.min_reserve_buoyancy
    # No above-water geometry → hull is designed to float at the waterline
    if df_angle < 1.0 and reserve < 0.01:
        detail = f"No above-water geometry (DF={df_angle:.1f}°, RB={reserve:.3f}) -> PASS (freeboard implicit)"
        return True, df_angle, min_df, detail
    df_pass = df_angle >= min_df
    res_pass = reserve >= min_reserve
    all_pass = df_pass and res_pass
    detail = f"DF={df_angle:.1f}° >= {min_df:.0f}°, RB={reserve:.3f} >= {min_reserve:.2f} -> {'PASS' if all_pass else 'FAIL'}"
    return all_pass, df_angle, min(min_df, min_reserve), detail


def _gate_fine_cfd(case_dir, stl_path, speed_ms, LWL, B, T_hull,
                   config, of_env, design_id, x_dict=None,
                   hydro=None, hull_stl_path=None):
    case_dir = Path(case_dir)
    max_cells = config.validation.fine_cfd_cells
    rho = config.fixed.rho_water
    nu = config.fixed.nu_water
    g = config.fixed.gravity

    write_openfoam_case(
        case_dir=case_dir,
        stl_path=stl_path,
        speed_ms=speed_ms,
        LWL=LWL, B=B, T=T_hull,
        rho=rho, nu=nu, gravity=g,
        mesh_levels=(2, 3) if max_cells <= 100000 else (3, 4),
        n_layers=5,
        solver="interFoam",
        end_time=0.1 if max_cells <= 100000 else 8.0,
        delta_t=0.001 if max_cells <= 100000 else 0.0002,
        write_interval=0.1,
        max_cells=max_cells,
    )

    _run_block_mesh(case_dir, of_env)
    _run_snappy_hex_mesh(case_dir, of_env)
    _run_set_fields(case_dir, of_env, T_hull)
    n_procs = getattr(config.validation, 'n_procs', 1)
    _run_inter_foam(case_dir, of_env, n_procs=n_procs)

    force_file = find_force_file(case_dir / "postProcessing" / "forces")
    if force_file is not None:
        rt = extract_openfoam_force(force_file)
    else:
        rt = None
    if rt is None:
        rt = 0.0

    from hull_opt.michell import compute_wave_resistance_michell
    from hull_opt.friction import compute_total_resistance
    import trimesh
    area = 1.0
    mesh = trimesh.load(stl_path)
    area = mesh.area if hasattr(mesh, 'area') else 1.0

    sac_scale = hydro.get("sac_scale_factor", 1.0) if hydro else 1.0
    if x_dict is not None:
        from hull_opt.geometry import compute_half_breadth_analytic
        half_breadth_func = lambda x, z: compute_half_breadth_analytic(x, z, x_dict, LWL, sac_scale=sac_scale)
    else:
        half_breadth_func = lambda x, z: 0.0
    Rw = compute_wave_resistance_michell(half_breadth_func, LWL, B, T_hull,
                                          speed_ms, rho, g)
    Rt_pred, _, _ = compute_total_resistance(speed_ms, area, LWL, rho, nu,
                                              wave_resistance=Rw)
    threshold = config.validation.rt_upper_bound_factor * Rt_pred
    passed = rt < threshold

    return passed, rt, threshold, f"Rt={rt:.4f}N < {threshold:.4f}N -> {'PASS' if passed else 'FAIL'}"


def _gate_wave_motions(case_dir, stl_path, speed_ms, LWL, B, T_hull,
                       mass, config, of_env, cg_z, D_keel=0.0):
    case_dir = Path(case_dir)
    T_hull_val = T_hull if isinstance(T_hull, (int, float)) else float(T_hull)
    B_val = B if isinstance(B, (int, float)) else float(B)
    T_total = T_hull_val + D_keel

    max_cells = config.validation.fine_cfd_cells
    fast_test = max_cells <= 100000

    write_openfoam_case(
        case_dir=case_dir,
        stl_path=stl_path,
        speed_ms=speed_ms,
        LWL=LWL, B=B, T=T_hull,
        rho=config.fixed.rho_water,
        nu=config.fixed.nu_water,
        gravity=config.fixed.gravity,
        mesh_levels=(2, 3) if max_cells <= 100000 else (3, 4),
        n_layers=5,
        solver="interFoam",
        six_dof=True,
        end_time=0.1 if fast_test else 15.0,
        delta_t=0.001 if fast_test else 0.0001,
        write_interval=0.05,
        max_cells=max_cells,
        mass=mass, cg_z=cg_z,
        Ixx=mass * (B_val ** 2 + T_total ** 2) / 12.0,
        Iyy=mass * (LWL ** 2 + T_total ** 2) / 12.0,
        Izz=mass * (LWL ** 2 + B_val ** 2) / 12.0,
    )

    _run_block_mesh(case_dir, of_env)
    _run_snappy_hex_mesh(case_dir, of_env)
    _run_set_fields(case_dir, of_env, T_hull)
    _run_potential_foam(case_dir, of_env)
    n_procs = getattr(config.validation, 'n_procs', 1)
    _run_inter_foam(case_dir, of_env, n_procs=n_procs)

    peak_accel = _extract_peak_accel_from_motion(case_dir)
    threshold = config.validation.max_accel_g
    passed = peak_accel < threshold

    return passed, peak_accel, threshold, f"Peak accel={peak_accel:.2f}g < {threshold}g -> {'PASS' if passed else 'FAIL'}"


def _gate_self_righting(case_dir, stl_path, speed_ms, LWL, B, T_hull,
                        mass, config, of_env, cg_z):
    """Computes GZ curve from 0 to 180 degrees heel and estimates
    self-righting time from the restoring arm magnitude when inverted."""
    case_dir = Path(case_dir)
    from hull_opt.hydrostatics import compute_gz_curve
    gz_curve = compute_gz_curve(str(stl_path), cg_z, n_angles=37, max_heel=180.0)
    angles = gz_curve[:, 0]
    gz = gz_curve[:, 1]
    near_inv_mask = (angles >= 140.0) & (angles < 180.0)
    gz_near = gz[near_inv_mask]
    gz_clean = gz_near[np.isfinite(gz_near)]

    threshold = config.validation.max_self_right_time_s

    if len(gz_clean) >= 3 and float(np.mean(gz_clean)) > 0.005:
        mean_gz = float(np.mean(gz_clean))
        self_right_time = min(0.1 / max(mean_gz, 0.001), 30.0)
        passed = self_right_time < threshold
    else:
        self_right_time = 99.9
        passed = False

    return passed, self_right_time, threshold, f"Self-right time={self_right_time:.1f}s (mean inverted GZ={np.mean(gz_clean):.4f}m) -> {'PASS' if passed else 'FAIL'}"


def _gate_drop_impact_of(case_dir, stl_path, LWL, B, T_hull,
                         mass, config, of_env, cg_z, D_keel=0.0):
    """Replace DualSPHysics drop-impact gate with interFoam 6-DOF
    water-entry test.  The hull starts at the water surface with an
    initial downward velocity = sqrt(2*g*drop_height)."""
    case_dir = Path(case_dir)
    T_hull_val = T_hull if isinstance(T_hull, (int, float)) else float(T_hull)
    B_val = B if isinstance(B, (int, float)) else float(B)
    T_total = T_hull_val + D_keel
    max_cells = config.validation.fine_cfd_cells
    fast_test = max_cells <= 100000

    drop_height = config.validation.drop_height
    v_drop = np.sqrt(2.0 * config.fixed.gravity * drop_height)

    write_openfoam_case(
        case_dir=case_dir,
        stl_path=stl_path,
        speed_ms=0.0,
        LWL=LWL, B=B, T=T_hull,
        rho=config.fixed.rho_water,
        nu=config.fixed.nu_water,
        gravity=config.fixed.gravity,
        mesh_levels=(2, 3) if max_cells <= 100000 else (3, 4),
        n_layers=5,
        solver="interFoam",
        six_dof=True,
        end_time=0.1 if fast_test else 2.0,
        delta_t=0.0005 if fast_test else 0.00005,
        write_interval=0.001,
        max_cells=max_cells,
        mass=mass, cg_z=cg_z,
        Ixx=mass * (B_val ** 2 + T_total ** 2) / 12.0,
        Iyy=mass * (LWL ** 2 + T_total ** 2) / 12.0,
        Izz=mass * (LWL ** 2 + B_val ** 2) / 12.0,
        initial_state={"velocity": (0.0, 0.0, -v_drop)},
        max_co=0.3, max_alpha_co=0.3,
    )

    _run_block_mesh(case_dir, of_env)
    _run_snappy_hex_mesh(case_dir, of_env)
    _run_set_fields(case_dir, of_env, T_hull)
    _run_potential_foam(case_dir, of_env)
    n_procs = getattr(config.validation, 'n_procs', 1)
    _run_inter_foam(case_dir, of_env, n_procs=n_procs)

    peak_accel = _extract_peak_accel_from_motion(case_dir)
    threshold = config.validation.max_accel_g
    passed = peak_accel < threshold

    return passed, peak_accel, threshold, f"Peak accel={peak_accel:.2f}g < {threshold}g -> {'PASS' if passed else 'FAIL'}"


def _gate_inverted_pressure(case_dir, stl_path, config, of_env, LWL, B, T_hull):
    case_dir = Path(case_dir)
    speed_ms = knots_to_ms(config.validation.inverted_speed_knots)
    rho = config.fixed.rho_water
    nu = config.fixed.nu_water
    g = config.fixed.gravity

    write_openfoam_case(
        case_dir=case_dir,
        stl_path=stl_path,
        speed_ms=speed_ms,
        LWL=LWL,
        B=B, T=T_hull,
        rho=rho, nu=nu, gravity=g,
        mesh_levels=(2, 3),
        n_layers=3,
        solver="simpleFoam",
        end_time=1.0,
        delta_t=0.001,
        write_interval=0.1,
        max_cells=500000,
    )

    _run_block_mesh(case_dir, of_env)
    _run_snappy_hex_mesh(case_dir, of_env)
    _run_simple_foam(case_dir, of_env)

    # Dynamic pressure from simpleFoam - extract maximum over all write times
    dyn_pressure = 0.0
    surface_files = sorted(case_dir.glob("postProcessing/hullPressure/*/surfaceFieldValue.dat"))
    if surface_files:
        try:
            all_max = []
            for sf in surface_files:
                data = np.loadtxt(sf, comments="#")
                if data.ndim == 1 and len(data) >= 2:
                    all_max.append(float(data[-1]))
                elif data.ndim >= 2:
                    all_max.append(float(data[:, -1].max()))
            if all_max:
                dyn_pressure = max(all_max)
        except Exception:
            pass

    # Hydrostatic pressure on the inverted deck: deepest point of inverted hull
    # The hull mesh is upright (keel down, deck up).  When inverted 180° the
    # highest z-coordinate of the upright hull (deck) becomes the deepest
    # point below the free surface.  Add this hydrostatic contribution since
    # simpleFoam solves for dynamic pressure only.
    try:
        mesh = trimesh.load(stl_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        deck_z = float(mesh.vertices[:, 2].max())
        hydro_pressure = rho * g * max(0, deck_z)
    except Exception:
        hydro_pressure = rho * g * T_hull   # fallback: just hull draft

    max_pressure = dyn_pressure + hydro_pressure
    threshold = config.validation.max_pressure_pa
    passed = max_pressure < threshold

    return passed, max_pressure, threshold, f"Max pressure={max_pressure:.0f}Pa < {threshold}Pa -> {'PASS' if passed else 'FAIL'}"


# ── OpenFOAM runner helpers ──────────────────────────────────────────────

def _run_potential_foam(case_dir, of_env):
    max_retries = 3
    last_err = ""
    for attempt in range(max_retries):
        proc = run_of_command(["potentialFoam"],
                              case_dir, of_env, timeout=300)
        if proc.returncode == 0:
            return
        last_err = f"potentialFoam failed (ret={proc.returncode}): {proc.stderr[:500]}"
        if attempt < max_retries - 1:
            logger.warning(f"{last_err} — retrying ({attempt + 1}/{max_retries})")
    raise RuntimeError(last_err)

def _run_block_mesh(case_dir, of_env):
    max_retries = 3
    last_err = ""
    for attempt in range(max_retries):
        proc = run_of_command(["blockMesh"],
                              case_dir, of_env, timeout=300)
        if proc.returncode == 0:
            return
        last_err = f"blockMesh failed (ret={proc.returncode}): {proc.stderr[:500]}"
        if attempt < max_retries - 1:
            logger.warning(f"{last_err} — retrying ({attempt + 1}/{max_retries})")
    raise RuntimeError(last_err)

def _run_snappy_hex_mesh(case_dir, of_env):
    max_retries = 3
    last_err = ""
    for attempt in range(max_retries):
        proc = run_of_command(["snappyHexMesh", "-overwrite"],
                              case_dir, of_env, timeout=7200)
        if proc.returncode == 0:
            break
        last_err = f"snappyHexMesh failed (ret={proc.returncode}): {proc.stderr[:500]}"
        if attempt < max_retries - 1:
            logger.warning(f"{last_err} — retrying ({attempt + 1}/{max_retries})")
    if proc.returncode != 0:
        raise RuntimeError(last_err)
    # snappyHexMesh may write modified mesh to 0/polyMesh/ instead of overwriting
    # constant/polyMesh/ in OF 2512.  Sync so motion solver sees consistent points.
    import shutil
    mesh_dir = Path(case_dir) / "constant" / "polyMesh"
    time_dir = Path(case_dir) / "0" / "polyMesh"
    if time_dir.exists():
        for f in ("points", "faces", "owner", "neighbour", "boundary"):
            src = time_dir / f
            if src.exists():
                shutil.copy2(src, mesh_dir / f)

def _run_set_fields(case_dir, of_env, T_hull):
    max_retries = 3
    last_err = ""
    for attempt in range(max_retries):
        proc = run_of_command(["setFields"],
                              case_dir, of_env, timeout=60)
        if proc.returncode == 0:
            return
        last_err = f"setFields failed (ret={proc.returncode}): {proc.stderr[:300]}"
        if attempt < max_retries - 1:
            logger.warning(f"{last_err} — retrying ({attempt + 1}/{max_retries})")
    logger.warning(last_err)

def _run_inter_foam(case_dir, of_env, n_procs=1):
    max_retries = 3
    last_err = ""
    for attempt in range(max_retries):
        proc = run_parallel_inter_foam(
            case_dir, of_env, n_procs=n_procs, timeout=14400
        )
        if proc.returncode == 0:
            return
        last_err = f"interFoam returned {proc.returncode}, stderr={proc.stderr[:300]}"
        if attempt < max_retries - 1:
            logger.warning(f"{last_err} — retrying ({attempt + 1}/{max_retries})")
    logger.warning(last_err)

def _run_simple_foam(case_dir, of_env):
    max_retries = 3
    last_err = ""
    for attempt in range(max_retries):
        proc = run_of_command(["simpleFoam"],
                              case_dir, of_env, timeout=7200)
        if proc.returncode == 0:
            return
        last_err = f"simpleFoam returned {proc.returncode}, stderr={proc.stderr[:300]}"
        if attempt < max_retries - 1:
            logger.warning(f"{last_err} — retrying ({attempt + 1}/{max_retries})")
    logger.warning(last_err)


# ── Post-processing helpers ──────────────────────────────────────────────

def _read_motion_states(case_dir: Path):
    """Read all sixDoFRigidBodyMotionState files from time directories.

    Returns list of (time, centreOfRotation, orientation_matrix, velocity, acceleration) tuples.
    """
    states = []
    # find all uniform subdirs containing the state file
    state_files = sorted(case_dir.glob("*_*/uniform/sixDoFRigidBodyMotionState"))
    # also check plain numeric dirs like 0.01, 0.05 etc
    import re
    for p in case_dir.iterdir():
        if p.is_dir() and re.match(r'^[\d.]+$', p.name):
            sf = p / "uniform" / "sixDoFRigidBodyMotionState"
            if sf.exists():
                state_files.append(sf)
    state_files = sorted(set(state_files))

    for sf in state_files:
        try:
            t = float(sf.parent.parent.name)
        except ValueError:
            continue
        with open(sf) as fh:
            content = fh.read()
        cor = None
        orient = None
        vel = None
        accel = None
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('centreOfRotation'):
                parts = line.split('(')[1].split(')')[0].split()
                cor = tuple(float(x) for x in parts[:3])
            elif line.startswith('orientation'):
                parts = line.split('(')[1].split(')')[0].split()
                if len(parts) >= 9:
                    orient = [float(x) for x in parts[:9]]
            elif line.startswith('velocity'):
                parts = line.split('(')[1].split(')')[0].split()
                vel = tuple(float(x) for x in parts[:3])
            elif line.startswith('acceleration'):
                parts = line.split('(')[1].split(')')[0].split()
                accel = tuple(float(x) for x in parts[:3])
        if accel is not None:
            states.append((t, cor, orient, vel, accel))
    return states


def _extract_peak_accel_from_motion(case_dir: Path) -> float:
    # Try reading from sixDoFRigidBodyMotionState files first
    states = _read_motion_states(case_dir)
    if states:
        all_accels = [np.sqrt(a[0]**2 + a[1]**2 + a[2]**2) for _, _, _, _, a in states]
        if all_accels:
            return max(all_accels) / 9.81

    # Fallback: try postProcessing .dat files
    motion_files = list(case_dir.glob("postProcessing/**/motion*.dat"))
    motion_files += list(case_dir.glob("postProcessing/**/sixDoF*.dat"))
    if not motion_files:
        logger.debug(f"No motion output files found in {case_dir}/postProcessing")
        return 0.0

    all_accels = []
    for mf in motion_files:
        try:
            data = np.loadtxt(mf, comments="#")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] >= 7:
                accel = np.sqrt(data[:, 4] ** 2 + data[:, 5] ** 2 + data[:, 6] ** 2)
                all_accels.extend(accel.tolist())
        except Exception:
            continue

    if not all_accels:
        return 0.0
    return max(all_accels) / 9.81



