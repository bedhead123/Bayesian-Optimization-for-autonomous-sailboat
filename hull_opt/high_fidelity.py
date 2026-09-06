"""
High-fidelity validation gates using DualSPHysics (SPH) and analytic rapid gates.
5 gates: calm-water SPH Rt, rapid wave motions, analytic self-righting,
rapid drop impact, inverted deck pressure (SPH).
Key exports: validate_top_designs(), ValidationResult
"""
import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

from hull_opt.geometry import generate_hull
from hull_opt.param_layer import design_vector_to_physical
from hull_opt.utils import knots_to_ms, ensure_dir, log_rapid_summary, log_design_diagnostics


class GateCrashError(RuntimeError):
    """Solver ran but crashed after all retries."""


class GateTimeoutError(RuntimeError):
    """Solver exceeded the gate wall-time budget."""


class GateSetupError(RuntimeError):
    """The gate case was set up wrong. A pipeline bug, not a design failure."""


class ValidationResult:
    def __init__(self, design_id: int):
        self.design_id = design_id
        self.gates = {}
        self.overall_pass = False

    def set_gate(self, name: str, passed: bool, value: float,
                 threshold: float, details: str = "", status: str = None):
        if status is None:
            status = "PASS" if passed else "FAIL"
        self.gates[name] = {
            "passed": passed,
            "value": value,
            "threshold": threshold,
            "details": details,
            "status": status,
        }

    @property
    def all_passed(self) -> bool:
        return all(g["passed"] for g in self.gates.values())


def validate_top_designs(designs: list[dict], config, rapid_results: list | None = None) -> list[ValidationResult]:
    results = []
    for i, d in enumerate(designs):
        design_vector = np.array(json.loads(d["design_vector"]), dtype=float)
        rapid = rapid_results[i] if rapid_results is not None and i < len(rapid_results) else None
        res = _validate_single(design_vector, d["id"], config, rapid=rapid)
        results.append(res)
        logger.info(f"Design {d['id']}: overall_pass={res.all_passed}, "
                     f"FoM={d['fom']:.4f}")
    return results


def _classify_gate_exception(exc: Exception, default_thresh: float):
    if isinstance(exc, GateTimeoutError):
        return 0.0, default_thresh, str(exc), "TIMEOUT"
    if isinstance(exc, (GateCrashError, GateSetupError)):
        return 0.0, default_thresh, str(exc), "CRASH"
    return 0.0, default_thresh, str(exc), "FAIL"


def _validate_single(design_vector: np.ndarray, design_id: int,
                     config, rapid=None) -> ValidationResult:
    result = ValidationResult(design_id)
    x_dict = design_vector_to_physical(design_vector, config)

    from hull_opt.geometry_validator import validate_design_vector
    is_valid, val_msg = validate_design_vector(x_dict, config)
    if not is_valid:
        logger.warning(f"Design {design_id}: design vector validation failed: {val_msg}")
        result.set_gate("design_vector", False, 0.0, 1.0, val_msg)
        return result

    hull_lwl = float(x_dict.get("LWL", config.fixed.LWL))
    B = x_dict["BWL"]  # placeholder; re-resolved to measured beam after generate_hull below
    T_hull = x_dict["T_canoe"]
    target_nabla = config.fixed.target_displacement
    rho = config.fixed.rho_water
    g = config.fixed.gravity
    eb = config.fixed.electronics_bay

    mass = rho * target_nabla
    speed_ms = knots_to_ms(config.fixed.target_speed_knots)
    D_keel = float(x_dict.get("D_keel", 0.0))

    from hull_opt.hydrostatics import compute_cg_z
    cg_z = compute_cg_z(x_dict, nabla=target_nabla)

    val_dir = ensure_dir(Path(config.paths.output_dir) / "validation" / f"design_{design_id}")

    for gate_dir in val_dir.glob("gate*"):
        if gate_dir.is_dir():
            shutil.rmtree(gate_dir, ignore_errors=True)

    stl_path, sac_path, hydro, hull_stl = generate_hull(
        design_vector,
        output_dir=str(val_dir),
        LWL=hull_lwl,
        target_displacement=target_nabla,
        config=config,
    )
    logger.info(f"Design {design_id}: geometry at {stl_path}")
    # Bug #172-B: price the built beam in all gates below.
    from hull_opt.hydrostatics import measured_beam as _mb
    try:
        B = _mb(x_dict, hydro)
    except Exception:
        pass

    # ── Rapid gate evaluation (Tier-1) for gates 2/4/5 ────────────────
    if rapid is None:
        try:
            from hull_opt.rapid_gates import evaluate_rapid_gates
            from hull_opt.hydrostatics import compute_gz_curve
            gz_curve = compute_gz_curve(str(stl_path), cg_z, n_angles=37, max_heel=180.0)
            rapid = evaluate_rapid_gates(x_dict, hydro, gz_curve, cg_z, stl_path, config)
            log_rapid_summary(design_id, rapid)
            log_design_diagnostics({
                "feasible": True, "fom": 0, "rapid": rapid,
                "rt_total": None, "rt_wave": None, "rt_friction": None,
                "roll_period": None,
                "peak_accel": getattr(rapid, "storm_peak_accel_g", None),
                "helm_combined_deg": None, "gm": None,
                "righting_energy": None, "stability_index": None,
            }, design_id)
        except Exception:
            rapid = None

    # ── Gate 1: Calm-water resistance (SPH towing) ────────────────────
    logger.info(f"Gate 1: SPH calm-water resistance")
    try:
        gate1_pass, gate1_val, gate1_thresh, gate1_detail = _gate_fine_cfd(
            val_dir / "gate1_calm", stl_path, speed_ms, hull_lwl, B, T_hull,
            config, design_id, x_dict,
            hydro=hydro, hull_stl_path=hull_stl,
        )
        gate1_status = None
    except Exception as e:
        logger.warning(f"Gate 1 exception: {e}")
        gate1_pass = False
        gate1_val, gate1_thresh, gate1_detail, gate1_status = _classify_gate_exception(
            e, config.validation.rt_upper_bound_factor * 1000)
    result.set_gate("calm_water_rt", gate1_pass, gate1_val, gate1_thresh, gate1_detail,
                    status=gate1_status)
    logger.info(f"Gate 1: {'PASS' if gate1_pass else gate1_status or 'FAIL'} "
                f"(Rt={gate1_val:.3f} N)")

    # ── Gate 2: Regular wave motions (rapid-gate) ──────────────────────
    logger.info(f"Gate 2: Regular wave motions (rapid-gate)")
    if rapid is not None:
        gate2_val = rapid.storm_peak_accel_g
        gate2_passed = gate2_val < config.validation.max_accel_g
        gate2_status = "OK"
        gate2_detail = f"rapid-gate storm peak accel = {gate2_val:.3f} g (threshold {config.validation.max_accel_g} g)"
        gate2_thresh = config.validation.max_accel_g
    else:
        gate2_passed = False
        gate2_val = 0.0
        gate2_thresh = config.validation.max_accel_g
        gate2_detail = "no rapid gate result available"
        gate2_status = "FAILED"
    result.set_gate("wave_motions_accel", gate2_passed, gate2_val, gate2_thresh, gate2_detail,
                    status=gate2_status)
    logger.info(f"Gate 2: {'PASS' if gate2_passed else gate2_status or 'FAIL'} "
                f"(accel={gate2_val:.2f} g)")

    # ── Gate 3: Extreme wave survival (analytic self-righting) ─────────
    logger.info(f"Gate 3: Extreme wave survival (self-righting)")
    try:
        gate3_pass, gate3_val, gate3_thresh, gate3_detail = _gate_self_righting(
            val_dir / "gate3_self_right", stl_path, speed_ms, hull_lwl, B, T_hull, mass,
            config, cg_z,
        )
    except Exception as e:
        gate3_pass, gate3_val, gate3_thresh, gate3_detail = False, 99.9, config.validation.max_self_right_time_s, str(e)
    result.set_gate("extreme_wave_self_right", gate3_pass, gate3_val, gate3_thresh, gate3_detail)
    logger.info(f"Gate 3: {'PASS' if gate3_pass else 'FAIL'} (self-right time={gate3_val:.2f} s)")

    # ── Gate 4: Drop impact (rapid-gate slam pressure) ─────────────────
    logger.info(f"Gate 4: Drop impact (rapid-gate)")
    if rapid is not None:
        # Capped value decides the gate (<= so the 3.2 MPa cap boundary is
        # inclusive — the exact-cap equality otherwise failed every low-
        # deadrise design on a boundary artifact); the FoM deadrise gradient
        # comes from the UNCAPPED Wagner value in the rapid margins.
        gate4_val = rapid.slam_pressure_pa / 1000.0
        gate4_passed = rapid.slam_pressure_pa <= config.validation.max_pressure_pa
        gate4_status = "OK"
        gate4_detail = (f"rapid-gate slam pressure = {rapid.slam_pressure_pa:.1f} Pa "
                        f"(wagner_raw={rapid.slam_pressure_raw_pa:.1f} Pa; "
                        f"threshold {config.validation.max_pressure_pa} Pa)")
        gate4_thresh = config.validation.max_pressure_pa
    else:
        gate4_passed = False
        gate4_val = 0.0
        gate4_thresh = config.validation.max_pressure_pa
        gate4_detail = "no rapid gate result available"
        gate4_status = "FAILED"
    result.set_gate("drop_impact_accel", gate4_passed, gate4_val, gate4_thresh, gate4_detail,
                    status=gate4_status)
    logger.info(f"Gate 4: {'PASS' if gate4_passed else gate4_status or 'FAIL'} "
                f"(pressure={gate4_val:.1f} kPa)")

    # ── Gate 5: Inverted deck pressure ─────────────────────────────────
    logger.info(f"Gate 5: Inverted deck pressure")
    gate5_rapid_val = rapid.inverted_pressure_pa if rapid is not None else None
    try:
        gate5_pass, gate5_val, gate5_thresh, gate5_detail = _gate_inverted_pressure(
            val_dir / "gate5_inverted", stl_path, config,
            LWL=hull_lwl, B=B, T_hull=T_hull, D_keel=D_keel, x_dict=x_dict,
            timeout=3600,
        )
        gate5_status = None
        if gate5_rapid_val is not None:
            gate5_val = max(gate5_val, gate5_rapid_val)
            if gate5_val >= config.validation.max_pressure_pa:
                gate5_pass = False
            gate5_detail += f" | rapid-gate inverted pressure = {gate5_rapid_val:.0f} Pa"
    except Exception as e:
        logger.warning(f"Gate 5 exception: {e}")
        if gate5_rapid_val is not None:
            gate5_pass = gate5_rapid_val < config.validation.max_pressure_pa
            gate5_val = gate5_rapid_val
            gate5_thresh = config.validation.max_pressure_pa
            gate5_detail = f"rapid-gate fallback: inverted pressure = {gate5_rapid_val:.0f} Pa"
            gate5_status = "OK"
        else:
            gate5_pass = False
            gate5_val, gate5_thresh, gate5_detail, gate5_status = _classify_gate_exception(
                e, config.validation.max_pressure_pa)
    result.set_gate("inverted_pressure", gate5_pass, gate5_val, gate5_thresh, gate5_detail,
                    status=gate5_status)
    logger.info(f"Gate 5: {'PASS' if gate5_pass else gate5_status or 'FAIL'} "
                f"(pressure={gate5_val:.1f} Pa)")

    result.overall_pass = result.all_passed
    return result


# Cost model for SPH: particles ~ (tank_volume)/dp^3, ~3e-8 s/particle/step.
# Measured on this box (RTX 3050, design 140 gate-1 tank): 19 s for
# 26,423 particles x 25,000 steps => ~2.9e-8 s/particle/step. The previous
# 1e-5 overestimated wall time by ~350x and blocked any dp finer than 0.09.
_SEC_PER_PARTICLE_PER_STEP = 3e-8
_SEC_PER_CELL_PER_STEP = _SEC_PER_PARTICLE_PER_STEP  # backward compat alias


def _expected_steps(end_time, delta_t):
    return int(round(end_time / delta_t))


def _keel_case_kwargs(x_dict, LWL, T_hull, D_keel):
    """Generate OpenFOAM case kwargs for keel geometry.

    Returns T_total and keel-box parameters. Omits keel keys when
    D_keel is near-zero (no keel).
    """
    T_total = T_hull + D_keel
    kwargs = {"T_total": T_total}
    if D_keel > 0.01 and x_dict.get("keel_chord", 0.0) > 0.01:
        kwargs["keel_x"] = (
            0.4 * LWL if x_dict.get("bulb_vol", 0.0) < 1e-6
            else x_dict.get("bulb_pos", 0.4) * LWL
        )
        kwargs["keel_chord"] = x_dict.get("keel_chord", 0.2)
        kwargs["keel_depth"] = D_keel
    return kwargs


def _check_force_steady(force_file, gate_name, max_osc_ratio=1.0,
                        max_slope_ratio=0.25):
    """Check that the force trace has reached a steady state.

    Returns (is_steady, message). Raises GateSetupError only when the force
    file is unreadable or too few samples (a pipeline bug, not a design
    failure). A non-steady trace (still trending, or oscillating beyond
    max_osc_ratio std/mean) returns (False, reason).

    Thresholds are Phase-0 calibrated (design 140, gate-1 tank, RTX 3050):
    the converged 6 s trace sits at |slope/mean| ~0.12 and std/mean ~0.5;
    all 2.5 s startup-transient traces sit at |slope/mean| >= 0.36.
    max_slope_ratio=0.25 and max_osc_ratio=1.0 split the gap, so a trace is
    only rejected when its trend or oscillation exceeds the measurement's
    own noise floor.
    """
    from hull_opt.sph_resistance import read_force_trace_csv
    trace = read_force_trace_csv(force_file)
    if trace is None:
        raise GateSetupError(
            f"{gate_name}: force file {force_file} not readable"
        )
    times, fx = trace
    if len(times) < 20:
        raise GateSetupError(
            f"{gate_name}: too few samples ({len(times)}) "
            f"for steady-state check"
        )

    # Check the last 30 % of data for steady state
    n = max(10, len(fx) // 3)
    tail_fx = fx[-n:]
    tail_t = times[-n:]

    t_norm = tail_t - tail_t[0]
    mean_fx = float(np.mean(tail_fx))
    if t_norm[-1] > 0 and mean_fx > 0:
        slope = np.polyfit(t_norm, tail_fx, 1)[0]
        if abs(slope) / mean_fx > max_slope_ratio:
            direction = "rising" if slope > 0 else "decaying"
            return False, (
                f"{gate_name}: force still {direction} "
                f"(slope={slope:.4f} N/s, mean={mean_fx:.4f} N)"
            )
        std_fx = float(np.std(tail_fx))
        if std_fx / mean_fx > max_osc_ratio:
            return False, (
                f"{gate_name}: force oscillating beyond guard "
                f"(std/mean={std_fx / mean_fx:.2f} > {max_osc_ratio}, "
                f"mean={mean_fx:.4f} N)"
            )
    return True, f"{gate_name}: force trace steady (mean={mean_fx:.4f} N)"


def _extract_peak_accel_from_motion(case_dir, min_sim_time=0.0):
    """Extract peak acceleration (in g) from sixDoFRigidBodyMotionState files.

    Reads motion state data from the case directory and optionally from
    processor* directories (which retain full history in parallel runs).
    Returns the peak |acceleration| / 9.81, or None if data-quality guards
    fail (too few samples, short sim time, NaN/Inf, velocities > 100 m/s,
    accelerations > 1000 g).
    """
    case_dir = Path(case_dir)
    times_accels = {}

    def _read_state_dir(base_dir):
        if not base_dir.is_dir():
            return
        for tdir in sorted(base_dir.iterdir()):
            state_file = tdir / "uniform" / "sixDoFRigidBodyMotionState"
            if not state_file.exists():
                continue
            try:
                text = state_file.read_text()
                t = float(tdir.name)
                m_a = re.search(r"acceleration\s*\(([^)]+)\)", text)
                m_v = re.search(r"velocity\s*\(([^)]+)\)", text)
                if m_a and m_v:
                    accel = [float(x) for x in m_a.group(1).split()]
                    vel = [float(x) for x in m_v.group(1).split()]
                    times_accels[t] = (accel, vel)
            except (ValueError, OSError):
                pass

    _read_state_dir(case_dir)
    for pd in sorted(case_dir.glob("processor*")):
        _read_state_dir(pd)  # processor dirs mirror case structure

    if len(times_accels) < 3:
        return None

    times = sorted(times_accels.keys())
    t_span = float(times[-1]) - float(times[0])
    if t_span < min_sim_time:
        return None

    peak_g = None
    for t in times:
        accel, vel = times_accels[t]
        if not all(np.isfinite(a) for a in accel) or not all(np.isfinite(v) for v in vel):
            return None
        if any(abs(v) > 100.0 for v in vel):
            return None
        a_mag = np.linalg.norm(accel)
        if a_mag > 1000.0 * 9.81:
            return None
        g_val = a_mag / 9.81
        if peak_g is None or g_val > peak_g:
            peak_g = g_val

    return peak_g


def _check_gate_feasibility(gate_name, dp, sim_time, n_particles_est,
                             timeout):
    """Fail fast if the SPH case cannot finish within its wall-time budget."""
    steps = int(np.ceil(sim_time / max(0.0001, 1e-4)))
    est_wall = steps * n_particles_est * _SEC_PER_PARTICLE_PER_STEP
    budget = 0.85 * timeout
    if est_wall > budget:
        raise RuntimeError(
            f"{gate_name}: infeasible cost model — est {steps} steps x "
            f"{n_particles_est} particles ~ {est_wall / 3600:.1f} h > "
            f"{budget / 3600:.1f} h budget ({timeout} s). Reduce dp/sim_time."
        )
    logger.info(f"{gate_name}: cost model {steps} steps ~ {est_wall / 60:.0f} min "
                f"(particles~{n_particles_est}, budget {timeout}s)")


def _gate_fine_cfd(case_dir, stl_path, speed_ms, LWL, B, T_hull,
                   config, design_id, x_dict=None,
                   hydro=None, hull_stl_path=None):
    """Gate 1: calm-water towing resistance.

    PASS/FAIL is decided by the OF-validated low-fi Rt (ITTC friction +
    Michell wave; OpenFOAM-era calibrations measured CFD/low-fi ratios
    0.73-1.08, see docs/POST_RUN_ANALYSIS.md). The fixed-hull-in-current SPH
    setup measures momentum-flux/form drag ~14x low-fi (Bug #157: Fz at speed
    exceeds full-submersion buoyancy, so the dynamic pressure field is not
    tow-equivalent) — the SPH tow is RUN ONLY as an audit when
    config.validation.sph_towing_audit is true, and its measured Rt is
    reported but never decides the gate.
    """
    from hull_opt.michell import compute_wave_resistance_michell, capped_wave_resistance
    from hull_opt.friction import compute_total_resistance
    import trimesh

    T_hull_val = T_hull if isinstance(T_hull, (int, float)) else float(T_hull)
    D_keel = float(x_dict.get("D_keel", 0.0)) if x_dict else 0.0

    # ── Validated low-fi prediction (the gate basis) ────────────────────
    mesh = trimesh.load(stl_path)
    try:
        uw = trimesh.intersections.slice_mesh_plane(mesh, plane_origin=[0,0,0], plane_normal=[0,0,1], cap=True)
        area = float(uw.area) if uw is not None and hasattr(uw,'area') else float(mesh.area)
    except Exception:
        area = float(mesh.area) if hasattr(mesh,'area') else 1.0

    sac_scale = hydro.get("sac_scale_factor", 1.0) if hydro else 1.0
    if x_dict is not None:
        from hull_opt.geometry import compute_half_breadth_analytic, keel_half_breadth
        keel_chord = float(x_dict.get("keel_chord", 0.0))
        def half_breadth_func(x, z, _ss=sac_scale, _dk=D_keel, _kc=keel_chord):
            hb = compute_half_breadth_analytic(x, z, x_dict, LWL, sac_scale=_ss)
            if _dk > 0.01 and _kc > 0.01:
                hb = hb + keel_half_breadth(x, z, x_dict, LWL)
            return hb
    else:
        half_breadth_func = lambda x, z: 0.0
    from hull_opt.hydrostatics import eff_draft as _ed
    Rw_raw = compute_wave_resistance_michell(half_breadth_func, LWL, B, _ed(x_dict, hydro) + D_keel,
                                              speed_ms, config.fixed.rho_water,
                                              config.fixed.gravity,
                                              n_z=max(20, int(np.ceil(20 * (T_hull + D_keel) / max(T_hull, 1e-6)))))
    target_nabla = config.fixed.target_displacement
    from hull_opt.michell import delft_cap_frac as _delft_cap_frac
    _Fn_gate = float(speed_ms) / max(1e-9, (config.fixed.gravity * max(1e-9, LWL)) ** 0.5)
    Rw = capped_wave_resistance(Rw_raw, target_nabla, config.fixed.rho_water,
                                config.fixed.gravity,
                                cap_frac=_delft_cap_frac(_Fn_gate))
    Rt_pred, _, _ = compute_total_resistance(speed_ms, area, LWL,
                                              config.fixed.rho_water,
                                              config.fixed.nu_water,
                                              wave_resistance=Rw)
    # Independent Delft-series envelope (2-5% of displacement weight at
    # Fn 0.40-0.45, PYD Fig 5.20) — the old threshold was 2.0·Rt_pred,
    # i.e. mathematically always PASS (audit: Gate 1 was a tautology).
    ref_rt = 0.035 * config.fixed.rho_water * config.fixed.gravity * target_nabla
    threshold = max(config.validation.rt_upper_bound_factor * ref_rt, 1.0)
    passed = (Rt_pred < threshold) and (Rt_pred > 1e-6)

    detail = (f"Rt_lowfi={Rt_pred:.4f}N < Delft-envelope {threshold:.4f}N -> "
              f"{'PASS' if passed else 'FAIL'}"
              f" (gate basis = Delft-anchored low-fi, Bug #157)")

    # ── SPH tow as AUDIT ONLY (never decides the gate) ──────────────────
    if not getattr(config.validation, "sph_towing_audit", False):
        return passed, Rt_pred, threshold, detail

    try:
        from hull_opt.sph_resistance import run_towing_resistance

        case_dir = Path(case_dir)
        dp = getattr(config.validation, "sph_dp", 0.05)
        sim_time = getattr(config.validation, "sph_sim_time", 6.0)
        timeout = getattr(config.validation, "gate_timeout", 14400)

        # Cost-model feasibility check — must OVER-estimate the fillbox
        # rendered by write_towing_case (moving-hull tow, AGENTS.md v3):
        # tank length = travel + front 0.6·LWL + hull + back 0.5·LWL ≈
        # 2.1·LWL + travel, travel = u0·sim_time (capped only by the writer's
        # particle budget — over-estimating is safe here); width
        # 2·max(B, 0.55); depth = keel tip + 1.5·keel_chord floor;
        # +0.25 m covers the writer's 4·dp fillbox inset.
        keel_chord_c = float(x_dict.get("keel_chord", 0.2)) if x_dict else 0.2
        tank_vol = (2.1 * LWL + speed_ms * sim_time + 0.25) \
            * (2.0 * max(1.0 * B, 0.55)) \
            * ((T_hull_val + D_keel) + max(1.5 * keel_chord_c, 0.4))
        n_particles_est = int(tank_vol / (dp ** 3))
        _check_gate_feasibility("gate1_calm", dp, sim_time,
                                n_particles_est, timeout)

        result = run_towing_resistance(
            case_dir, stl_path, x_dict or {}, config, speed_ms,
            dp=dp, sim_time=sim_time, timeout_s=timeout,
        )
        rt = result.get("rt_n")
        if rt is None or result.get("status") != "OK":
            detail += (f" | SPH audit: {result.get('status', 'FAILED')} "
                       f"({result.get('details', '')[:120]})")
        else:
            steady_ok, steady_msg = _check_force_steady(
                result.get("force_csv"), "gate1_calm",
            )
            detail += (f" | SPH audit Rt={rt:.4f}N "
                       f"(steady={steady_ok}; NOT gate-deciding)")
            logger.info(
                f"Gate 1 SPH audit design {design_id}: Rt={rt:.4f} N vs "
                f"low-fi {Rt_pred:.4f} N (ratio {rt / max(Rt_pred, 1e-9):.2f})"
            )
    except Exception as e:
        logger.warning(f"Gate 1 SPH audit skipped: {e}")

    return passed, Rt_pred, threshold, detail


def _gate_self_righting(case_dir, stl_path, speed_ms, LWL, B, T_hull,
                        mass, config, cg_z):
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


def _gate_inverted_pressure(case_dir, stl_path, config, LWL, B, T_hull,
                            D_keel=0.0, x_dict=None, timeout=7200):
    """Gate 5: SPH inverted deck pressure (hard CRASH on any solver failure)."""
    from hull_opt.sph_resistance import run_inverted_pressure

    case_dir = Path(case_dir)
    rho = config.fixed.rho_water

    result = run_inverted_pressure(
        case_dir, stl_path, x_dict or {}, config,
        timeout_s=timeout,
    )

    # Gate 5 is a hard CRASH gate: a fabricated analytic value must never be
    # scored as a measured deck pressure (bug: solver AbortBoundOut was
    # silently replaced with the fallback, and the gate PASSed on it).
    if result.get("status") == "TIMEOUT":
        raise GateTimeoutError(f"SPH inverted timed out: {result.get('details', '')}")
    max_pressure = result.get("max_pressure_pa")
    if result.get("status") != "OK" or max_pressure is None:
        fb = result.get("fallback_pressure_pa")
        msg = f"SPH inverted failed: {result.get('details', '')}"
        if fb is not None:
            msg += f" (analytic fallback {fb:.1f} Pa NOT used as a measurement)"
        raise GateCrashError(msg)

    threshold = config.validation.max_pressure_pa
    passed = max_pressure < threshold

    return passed, max_pressure, threshold, f"Max pressure={max_pressure:.0f}Pa < {threshold}Pa -> {'PASS' if passed else 'FAIL'}"
