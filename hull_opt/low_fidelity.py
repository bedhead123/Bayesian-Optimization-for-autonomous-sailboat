"""
Low-fidelity evaluation pipeline: geometry → GZ curve → Michell wave
resistance → friction → Capytaine RAOs → constraints → FoM.
Primary evaluation function used by the BO surrogate.
Key exports: evaluate_low_fidelity(), EvaluationResult
Bugs fixed: GM sign error in roll period estimate (#4)
"""
import fcntl
import hashlib
import json
import time
import numpy as np
import trimesh
from pathlib import Path
from typing import Optional
import logging


def _trapz(y, x, **kwargs):
    return np.trapezoid(y, x, **kwargs)
import warnings

_DEBUG_LOG = None  # Will be set from config if available
PEAK_ACCEL_FALLBACK_G = 60.0


def _agent_log(location: str, message: str, data: dict, hypothesis_id: str, run_id: str = "pre-fix"):
    if _DEBUG_LOG is None:
        return
    # #region agent log
    try:
        with open(_DEBUG_LOG, "a") as f:
            f.write(json.dumps({
                "sessionId": "b990d8", "location": location, "message": message,
                "data": data, "hypothesisId": hypothesis_id, "runId": run_id,
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        logger.debug("Failed to write agent log", exc_info=True)
    # #endregion

# Only suppress known-harmless Capytaine warnings (e.g. intermittent
# iterative solver non-convergence for panels near resonance frequencies).
# Other warnings (dimension mismatches, geometry issues) are left visible.
warnings.filterwarnings("ignore", message=".*Capytaine failed.*")
logging.getLogger("capytaine").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

from hull_opt.geometry import generate_hull, compute_half_breadth_analytic, keel_half_breadth
try:
    from hull_opt.geometry import mesh_displacement
except ImportError:
    mesh_displacement = None
from hull_opt.param_layer import design_vector_to_physical
from hull_opt.geometry_validator import validate_design_vector
from hull_opt.hydrostatics import compute_gz_curve, compute_righting_energy, compute_cg_z, \
    compute_lumped_inertia
from hull_opt.constraints import evaluate_constraints
from hull_opt.michell import compute_wave_resistance_michell, capped_wave_resistance
from hull_opt.friction import compute_total_resistance
from hull_opt.rapid_gates import evaluate_rapid_gates, RapidGateResult
from hull_opt.balance import evaluate_balance_polar


class EvaluationResult:
    def __init__(self):
        self.feasible = False
        self.fom = -float("inf")
        self.rt_total = 0.0
        self.rt_wave = 0.0
        self.rt_friction = 0.0
        self.stability_index = 0.0
        self.roll_period = 0.0
        self.peak_accel = 0.0
        self.helm_fwd_deg = 0.0
        self.helm_aft_deg = 0.0
        self.helm_combined_deg = 0.0
        self.gm = 0.0
        self.cg_z = 0.0
        self.eq_heel_deg = 0.0
        self.righting_energy = 0.0
        self.physical_params = None
        self.constraint_values = {}
        self.constraint_violations = []
        self.error_code = None
        self.cad_stl_path = None
        self.cad_sac_path = None
        self.hydro = {}
        self.gz_curve = None
        self.rao_data = None
        self.rapid = None
        self.balance = None


def _wait_spf_lock(config, timeout_s: int = 3600):
    """Park this worker while the SPH solver holds output/gpu.lock.

    Polls flock(LOCK_EX|LOCK_NB) on <config.paths.output_dir>/gpu.lock
    every 5 s until it can acquire (then releases immediately) or the
    timeout elapses (then returns False). timeout_s <= 0 means a single
    attempt with no waiting. When timeout_s < 5 s the poll interval
    shrinks to timeout_s so short waits finish promptly. The lock file is
    opened in append mode (never truncated). No-op if the file can't be
    opened (returns True)."""
    try:
        lock_path = Path(config.paths.output_dir) / "gpu.lock"
        fd = open(lock_path, "a")
    except OSError:
        return True
    parked = False
    waited = 0.0
    poll_dt = min(5.0, max(0.05, float(timeout_s))) if float(timeout_s) > 0 else 0.0
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                if parked:
                    logger.debug("BEM worker resumed after %.1fs wait on gpu.lock", waited)
                return True
            except BlockingIOError:
                if not parked:
                    parked = True
                    logger.info("BEM worker parked: SPH solver holds gpu.lock")
                logger.debug("BEM worker waiting on gpu.lock (%.1fs elapsed)", waited)
                if float(timeout_s) <= 0 or waited >= float(timeout_s):
                    logger.debug("BEM worker gave up on gpu.lock after %.1fs", waited)
                    return False
                time.sleep(poll_dt)
                waited += poll_dt
            except OSError:
                return True
    finally:
        try:
            fd.close()
        except OSError:
            logger.debug("Failed to close gpu.lock fd", exc_info=True)


def _accumulate_margin_violations(rapid, soft, constraint_violations,
                                  violation_magnitude):
    """Append failing rapid-gate margins to the violations list and inflate
    the infeasibility magnitude for HARD gates only.

    Soft margins are appended to constraint_violations for reporting but do
    NOT add to violation_magnitude (they are already penalized via the w6-w9
    FoM bonuses); revoking feasibility for failing hard gates stays in the
    caller. Pure function: returns (violations, magnitude) without mutating
    its arguments. `rapid` must expose `.margins` (a dict of gate -> margin)."""
    soft_set = set(soft)
    violations = list(constraint_violations)
    magnitude = float(violation_magnitude)
    for gate, margin in rapid.margins.items():
        if isinstance(margin, (int, float)) and margin < 0:
            violations.append(f"{gate}_margin={margin:.3f} < 0")
            if gate not in soft_set:
                magnitude += max(0.0, abs(float(margin))) * 0.5
    return violations, magnitude


def _margin_bonus_clip(v):
    """Smooth tanh saturation for rapid-gate margin bonuses: keeps the bonus
    bounded in (-1, 1) while preserving a GP gradient beyond ±1 (a hard clip
    kills the gradient there). None (missing margin) maps to -1.0."""
    if v is None:
        return -1.0
    return float(np.tanh(v))


def draft_logistics_cost(x_dict: dict, config) -> float:
    """Priced inconvenience of draft (Bug #169), not a wall.

    Ramp-launchable draft (fixed.draft_free_m) is free; over that, handling
    cost ramps linearly at fixed.draft_logistics_per_m FoM per meter.
    Benefits CAN outweigh it — BO finds the equilibrium. Pure function.
    """
    try:
        t_tot = float(x_dict.get("T_canoe", 0.0)) + float(x_dict.get("D_keel", 0.0))
        free = float(getattr(config.fixed, "draft_free_m", 1.0))
        rate = float(getattr(config.fixed, "draft_logistics_per_m", 0.3))
        return max(0.0, rate * (t_tot - free))
    except Exception:
        return 0.0


def leeway_penalty_deg(heavy_leeway_deg: float, config) -> float:
    """VMG-priced leeway (Bug #169): degrees past the 4° designer norm
    (PYD: 3-5° ideal helm balance) cost drive on every point of sail.
    Linear slope w_leeway = stated judgment. Pure function."""
    try:
        hl = float(heavy_leeway_deg)
        if not np.isfinite(hl) or hl <= 4.0:
            return 0.0
        return float(getattr(config.weights, "w_leeway", 0.5)) * (hl - 4.0) / 4.0
    except Exception:
        return 0.0


def evaluate_low_fidelity(design_vector: np.ndarray, config,
                          output_dir: Optional[str] = None,
                          drag_factor: float = 1.0,
                          iteration: Optional[int] = None,
                          bem_skip: bool = False) -> EvaluationResult:
    result = EvaluationResult()

    try:
        x_dict = design_vector_to_physical(design_vector, config)
    except Exception as e:
        result.error_code = f"E_DESERIALIZE:{e}"
        return result

    # 0. Validate design vector bounds before geometry generation
    is_valid, val_msg = validate_design_vector(x_dict, config)
    if not is_valid:
        result.error_code = f"E_DV_INVALID:{val_msg}"
        return result

    # 1. Generate geometry (LWL from design vector, with fixed LWL as default)
    try:
        hull_lwl = float(x_dict["LWL"])
        if output_dir is not None:
            dv_key = hashlib.md5(np.asarray(design_vector, dtype=float).tobytes()).hexdigest()[:12]
            output_dir = str(Path(output_dir) / f"design_{dv_key}")
        stl_path, sac_path, hydro, hull_stl = generate_hull(
            design_vector,
            output_dir=output_dir,
            LWL=hull_lwl,
            target_displacement=config.fixed.target_displacement,
            config=config,
        )
        _agent_log("low_fidelity.py:generate", "geometry paths", {
            "stl_path": stl_path, "hull_stl": hull_stl,
        }, "A")
        result.cad_stl_path = stl_path
        result.cad_sac_path = sac_path
        result.hydro = hydro
        result.physical_params = {
            **x_dict,
            "sac_scale_factor": hydro.get("sac_scale_factor"),
            "sac_scale_std": hydro.get("sac_scale_std"),
            "actual_Cp": hydro.get("actual_Cp"),
            "BM": hydro.get("BM"),
            "CB_x": hydro.get("CB_x"),
            "waterplane_area": hydro.get("waterplane_area"),
        }

        # NURBS geometry validation is handled in generate_hull via validate_nurbs_patches
    except Exception as e:
        result.error_code = f"E_GEOM:{e}"
        return result

    # 2. GZ curve with ballast-derived CG (hull-only mesh, keel CG approximated)
    try:
        cg_z = hydro.get("cg_z", compute_cg_z(x_dict, nabla=hydro.get("underwater_volume", hydro.get("nabla")), config=config))
        if not np.isfinite(cg_z):
            raise ValueError(f"Non-finite CG_z: {cg_z}")
        gz_patches = None
        if config.fixed.use_nurbs_gz and "nurbs_patches_file" in hydro:
            from hull_opt.geometry import _load_nurbs_patches
            try:
                gz_patches = _load_nurbs_patches(hydro["nurbs_patches_file"])
            except Exception as e:
                logger.warning(f"NURBS patch loading failed, falling back to mesh GZ: {e}")
        gz = compute_gz_curve(
            hull_stl,
            cg_z=cg_z,
            n_angles=37,
            max_heel=180.0,
            patches=gz_patches,
        )
        result.gz_curve = gz
        if not np.all(np.isfinite(gz[:, 1])):
            result.error_code = "E_GZ_NAN"
            return result
    except Exception as e:
        result.error_code = f"E_GZ:{e}"
        return result

    # 3. Compute wave resistance via Michell integral
    try:
        mesh = trimesh.load(hull_stl)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        speed_ms = config.fixed.target_speed_knots * 0.514444
        # Use underwater-only wetted area (slice at z=0)
        try:
            underwater_mesh = trimesh.intersections.slice_mesh_plane(
                mesh, [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], cap=False
            )
            if underwater_mesh is not None and hasattr(underwater_mesh, 'area') and underwater_mesh.area > 0:
                wetted_area = underwater_mesh.area
            else:
                wetted_area = mesh.area
        except Exception as e:
            logger.warning(f"Underwater mesh slicing failed, using total mesh area: {e}")
            wetted_area = mesh.area

        sac_scale = hydro.get("sac_scale_factor", 1.0)
        D_keel = x_dict.get("D_keel", 0.0)
        keel_chord = x_dict.get("keel_chord", 0.0)
        def half_breadth_func(xq, zq, _ss=sac_scale, _dk=D_keel, _kc=keel_chord):
            hb = compute_half_breadth_analytic(xq, zq, x_dict, hull_lwl, sac_scale=_ss)
            if _dk > 0.01 and _kc > 0.01:
                hb = hb + keel_half_breadth(xq, zq, x_dict, hull_lwl)
            return hb

        Rw_raw = compute_wave_resistance_michell(
            half_breadth_func,
            LWL=hull_lwl,
            B=x_dict["BWL"],
            T=x_dict["T_canoe"] + D_keel,
            speed_ms=speed_ms,
            rho=config.fixed.rho_water,
            g=config.fixed.gravity,
            n_z=max(20, int(np.ceil(20 * (x_dict["T_canoe"] + D_keel) / max(x_dict["T_canoe"], 1e-6)))),
        )
        # PYD Ch.5 residuary envelope (2-5% of displacement weight at Fn 0.4-0.45):
        # Michell thin-ship overpredicts 2-5x at B/L~0.25; cap to the Delft band
        # so the FoM tracks reality and the SPH calibration band is reachable.
        nabla_ref = hydro.get("underwater_volume",
                              hydro.get("nabla", config.fixed.target_displacement))
        from hull_opt.michell import delft_cap_frac as _delft_cap_frac
        _Fn_des = float(speed_ms) / max(1e-9, (config.fixed.gravity * max(1e-9, hull_lwl)) ** 0.5)
        Rw = capped_wave_resistance(Rw_raw, nabla_ref, config.fixed.rho_water,
                                    config.fixed.gravity,
                                    cap_frac=_delft_cap_frac(_Fn_des))
        # Appendage wetted area: keel (both sides + tip, root fillet) and full
        # ellipsoid bulb surface — the old hull-only slice omitted ~half the
        # viscous surface. Bulb: full ellipsoid 4π(a²b²+a²c²+b²c²)/3 approx via
        # Knud Thomsen p=1.6 on semi-axes (a=x, b=y, c=z).
        _br = (max(1e-9, 3.0 * x_dict["bulb_vol"] / (4.0 * np.pi))) ** (1.0 / 3.0)
        _ba, _bb, _bc = _br * 1.333, _br, _br * 0.8
        _bulb_s = 4.0 * np.pi * (((_ba * _bb) ** 1.6 + (_ba * _bc) ** 1.6 + (_bb * _bc) ** 1.6) / 3.0) ** (1.0 / 1.6)
        A_app = (2.0 * D_keel * 0.75 * keel_chord * 1.15 * 1.06
                 + _bulb_s)
        Rt, Rf, Rw_out = compute_total_resistance(
            speed_ms, wetted_area, hull_lwl,
            rho=config.fixed.rho_water, nu=config.fixed.nu_water,
            wave_resistance=Rw,
            appendage_area=A_app,
            roughness_mult=float(getattr(config.fixed, "fouling_cf_mult", 1.0)),
        )
        # Calibrated wave-resistance factor from mid-fidelity CFD. The ITTC
        # friction line is reliable; only the wave part is scaled. Floor at
        # 50% of friction to avoid non-physical corrected drags.
        Rt = max(Rf * 0.5, Rf + Rw * max(0.0, drag_factor))
        result.rt_total = Rt
        result.rt_wave = Rw
        result.rt_wave_raw = Rw_raw
        result.rt_friction = Rf
        if result.physical_params is not None:
            result.physical_params["wetted_area"] = wetted_area
            result.physical_params["appendage_area"] = A_app
        logger.debug(f"Resistance: Rw={Rw:.4f} (raw {Rw_raw:.4f}), Rf={Rf:.4f}, Rt={Rt:.4f}, "
                     f"wetted_area={wetted_area:.6f}, appendage_area={A_app:.4f}, "
                     f"drag_factor={drag_factor:.4f}, speed_ms={speed_ms:.4f}")
        for val_name, val in [("Rt", Rt), ("Rf", Rf), ("Rw", Rw)]:
            if not np.isfinite(val) or val < 0:
                raise ValueError(f"Non-finite or negative {val_name}: {val}")
    except Exception as e:
        import traceback
        result.error_code = f"E_MICHELL:{e}\n{traceback.format_exc()}"
        return result

    # 3b. Upright displaced volume from the SAME mesh used for the GZ curve
    # (actual mesh volume, not SAC-derived). Reused by the displacement-mismatch
    # penalty in the FoM block below; falls back to SAC volume if unavailable.
    try:
        mesh_disp = float(mesh_displacement(mesh, 0.0))
    except Exception as e:
        logger.warning(f"Mesh displacement unavailable, falling back to SAC volume: {e}")
        mesh_disp = 0.0
    result.hydro["mesh_displacement_m3"] = mesh_disp

    # 4. Capytaine RAOs (heave, pitch, roll) — required, no fallback
    try:
        if bem_skip:
            gm = hydro.get("GM", hydro.get("gm", 0.1))
            bwl = x_dict.get("BWL", 0.5)
            result.roll_period = 2.0 * np.pi * 0.35 * bwl / max(np.sqrt(9.81 * max(gm, 0.01)), 1e-6)
            result.peak_accel = float(getattr(getattr(config, "validation", None), "max_accel_g", 30.0)) * 2.0  # 60g penalty, was 0.0 which incorrectly passed constraints
            result.rao_data = {"omega": [0.0], "heave_rao": [0.0], "pitch_rao": [0.0],
                               "roll_rao": [0.0], "peak_accel": result.peak_accel, "roll_period": result.roll_period}
        else:
            try:
                _wait_spf_lock(config, timeout_s=120)
            except Exception:
                logger.debug("gpu.lock wait failed; continuing without parking")
            rao_data = _compute_raos_capytaine(
                stl_path, config, x_dict, hydro, speed_ms
            )
            result.roll_period = rao_data.get("roll_period", 0.0)
            result.peak_accel = rao_data.get("peak_accel", 0.0)
            if not np.isfinite(result.peak_accel) or result.peak_accel <= 0:
                logger.warning("Capytaine peak_accel invalid (%s), falling back to 60.0g penalty", result.peak_accel)
                result.peak_accel = 60.0
            if not np.isfinite(result.roll_period):
                result.roll_period = 0.0
            result.rao_data = {k: v for k, v in rao_data.items()
                               if k in ("omega", "heave_rao", "pitch_rao", "roll_rao",
                                        "peak_accel", "roll_period", "wall_time_s")}
    except BemDecimationEnvError as e:
        result.error_code = f"E_RAO_ENV:{e}"
        return result
    except Exception as e:
        result.error_code = f"E_RAO:{e}"
        return result

    # 4b. Rapid gates (BEM-dependent storm seakeeping + GZ-based stability)
    try:
        _wait_spf_lock(config, timeout_s=120)
    except Exception:
        logger.debug("gpu.lock wait failed; continuing without parking")
    try:
        rapid_result = evaluate_rapid_gates(x_dict, result.hydro, result.gz_curve, result.cg_z, stl_path, config, x_vector=design_vector, bem_skip=bem_skip)
    except Exception:
        rapid_result = None
    result.rapid = rapid_result

    # 4c. System-level 360° aero-hydro balance (cheap analytic, whole-system)
    try:
        balance = evaluate_balance_polar(x_dict, config, gz, hydro,
                                         boat_speed_ms=speed_ms)
    except Exception as e:
        logger.debug(f"Balance polar failed: {e}")
        balance = None
    result.balance = balance

    # 5. Constraints (pass x_dict, config, stl_path for new checks)
    try:
        feasible, violations, constraints, violation_magnitude = evaluate_constraints(
            hydro, gz, roll_period=result.roll_period,
            peak_accel=result.peak_accel,
            x_dict=x_dict, config=config, stl_path=stl_path,
            hull_stl_path=hull_stl,
            iteration=iteration,
            balance=balance,
        )
        _agent_log("low_fidelity.py:constraints", "constraint result", {
            "feasible": feasible,
            "downflooding_angle": constraints.get("downflooding_angle"),
            "fom_will_compute": feasible,
            "rt_total": result.rt_total,
            "n_violations": len(violations),
            "violations_head": violations[:3],
        }, "C")
        result.feasible = feasible
        result.constraint_violations = violations
        result.constraint_values = constraints
        result.violation_magnitude = violation_magnitude
        result.helm_fwd_deg = constraints.get("helm_fwd_deg", 0.0)
        result.helm_aft_deg = constraints.get("helm_aft_deg", 0.0)
        result.helm_combined_deg = constraints.get("helm_combined_deg", 0.0)
        result.gm = constraints.get("gm", 0.0)
        result.cg_z = constraints.get("cg_z", hydro.get("cg_z", 0.0))
        result.eq_heel_deg = constraints.get("eq_heel_feathered_deg", 0.0)
        result.righting_energy = constraints.get("righting_energy", 0.0)
    except Exception as e:
        result.error_code = f"E_CONSTRAINTS:{e}"
        return result

    # 5b. Append rapid-gate margin violations to constraint violations.
    # Soft margins (capsize/storm_accel/slam_pressure/slam_accel/
    # inverted_pressure) are already penalized in the FoM via the w6-w9
    # bonuses and use conservative analytic estimates, so they do NOT revoke
    # feasibility. Any other (stability/seakeeping) margin failing revokes
    # feasibility; the FoM block below routes infeasible designs to a graded
    # negative FoM.
    if result.rapid is not None and result.rapid.margins:
        soft = set(getattr(getattr(config, "rapid_validation", None),
                           "soft_margin_gates", ()))
        result.constraint_violations, result.violation_magnitude = \
            _accumulate_margin_violations(
                result.rapid, soft,
                result.constraint_violations, result.violation_magnitude,
            )
        if any(isinstance(m, (int, float)) and m < 0 and gate not in soft
               for gate, m in result.rapid.margins.items()):
            result.feasible = False

    # 5c. System-level balance gate: storm heel/leeway must stay inside AVS,
    # and ops polar must be largely feasible. Hard failure only on storm
    # infeasibility; ops single-point failure is a soft FoM shaping signal
    # (keeps gradient). This prevents every design going infeasible due to
    # marginal VMG while still enforcing survival at 80kt feathered.
    if result.balance is not None:
        bal = result.balance
        # Record balance scalars into constraints for reporting
        constraints["balance_worst_heel_ops"] = bal.get("worst_heel_ops_deg")
        constraints["balance_worst_leeway_ops"] = bal.get("worst_leeway_ops_deg")
        constraints["balance_worst_heel_storm"] = bal.get("worst_heel_storm_deg")
        constraints["balance_worst_leeway_storm"] = bal.get("worst_leeway_storm_deg")
        constraints["balance_mean_drive"] = bal.get("mean_drive_ops_N")
        constraints["balance_vmg_up"] = bal.get("vmg_up_N")
        constraints["balance_vmg_down"] = bal.get("vmg_down_N")
        constraints["balance_n_ops_fail"] = bal.get("n_ops_fail")
        constraints["balance_n_storm_fail"] = bal.get("n_storm_fail")
        constraints["balance_heel_margin_storm"] = bal.get("heel_margin_storm")
        # Hard gate: storm must be feasible
        if bal.get("n_storm_fail", 0) > 0 or not bal.get("system_feasible", True):
            # Only revoke if storm truly fails (survival critical)
            if bal.get("n_storm_fail", 0) > 0:
                result.constraint_violations.append(
                    f"balance_storm_fail n_fail={bal.get('n_storm_fail')} worst_heel_storm={bal.get('worst_heel_storm_deg', 0):.1f}°"
                )
                result.violation_magnitude += 1.0 + abs(float(bal.get("heel_margin_storm", -1)))
                result.feasible = False
            else:
                # system_feasible False due to heel margin but no point fail — soft
                result.constraint_violations.append(
                    f"balance_heel_margin_storm={bal.get('heel_margin_storm', -1):.3f} < 0"
                )
                result.violation_magnitude += 0.3 * abs(float(bal.get("heel_margin_storm", -1)))
        # Soft: ops failures -> penalty not fatal
        if bal.get("n_ops_fail", 0) > 1:
            result.constraint_violations.append(f"balance_ops_fail n={bal.get('n_ops_fail')}")
            result.violation_magnitude += 0.2 * float(bal.get("n_ops_fail", 0))

    # 6. FoM
    if not result.feasible:
        # Rule 3: graded negative FoM so GP learns constraint boundary
        violation_mag = result.violation_magnitude if hasattr(result, 'violation_magnitude') else 1.0
        result.fom = -max(violation_mag, 0.01)
    else:
        try:
            righting_energy = constraints.get("righting_energy", 0.0)
            # Bug #168: single stability term, Michaelis-Menten saturation.
            # The old stack (w2*min(RE/160,1.25) + w3*self_right + w6 capsize
            # + drive/storm/VMG/leeway tanhs) rewarded the SAME GZ signal six
            # times with six saturating nonlinearities — depth upside capped
            # while depth costs grew unbounded. Now: ONE continuous term.
            # Half-saturation at 2x the survival threshold (80 J) is a stated
            # mission judgment (diminishing returns past self-righting are
            # real capsize mechanics; the 2x multiple is a choice, sens-tested
            # in test_plan_impl). Self-righting stays a hard gate, not FoM.
            re_half = 2.0 * max(1.0, getattr(config.validation, "min_righting_energy", 40.0))
            re_val = max(0.0, righting_energy) if np.isfinite(righting_energy) else 0.0
            stability_index = re_val / (re_val + re_half)
            if not np.isfinite(stability_index):
                stability_index = 0.0
            result.stability_index = stability_index

            # No crew comfort roll band penalty (autonomous vessel)

            storm_accel = result.rapid.storm_peak_accel_g if result.rapid else result.peak_accel
            accel_penalty = 0.0
            peak_accel_for_penalty = max(result.peak_accel, storm_accel)
            if peak_accel_for_penalty > config.validation.max_accel_g:
                accel_penalty = 2.0 * (peak_accel_for_penalty - config.validation.max_accel_g)

            # Displacement mismatch penalty: prevent AI from undersizing hull.
            # Penalize the actual mesh displacement (upright, same mesh as the
            # GZ curve) so SAC scaling can't game the penalty; fall back to the
            # SAC-derived volume when the mesh volume is unavailable.
            target_nabla = hydro.get("target_nabla", config.fixed.target_displacement)
            if mesh_disp > 0:
                nabla = float(mesh_disp)
            else:
                nabla = float(hydro.get("underwater_volume", hydro.get("nabla", target_nabla)))
            vol_delta = abs(nabla - target_nabla) / max(1e-10, target_nabla)
            disp_penalty = 0.5 * vol_delta

            # Helm balance shaping term (w5): soft penalty on feasible designs
            # with poor CE-CLR balance. Hard gate is the constraint in §8.2.
            helm_penalty = 0.0
            if config.weights.w5 > 0:
                max_helm = max(config.validation.single_sail_max_helm_angle_deg, 1.0)
                helm_worst = max(result.helm_combined_deg, 0.0)
                helm_penalty = (helm_worst / max_helm) * config.weights.w5

            # Light-wind bonus for bimodal El Niño conditions
            # Designs with good sail area-to-drag at low speed get bonus.
            # Deep fins pay here too: the fin is wetted at ALL speeds
            # (Bug #166 — the old T_canoe-only call gave depth a free ride).
            light_wind_bonus = 0.0
            if hasattr(config.weights, 'light_wind_bonus') and config.weights.light_wind_bonus > 0:
                # Lower wave resistance at low speed = better light-wind performance
                speed_low_ms = 0.5 * config.fixed.target_speed_knots * 0.514444
                try:
                    Rw_low_raw = compute_wave_resistance_michell(
                        half_breadth_func,
                        LWL=hull_lwl,
                        B=x_dict["BWL"],
                        T=x_dict["T_canoe"] + x_dict.get("D_keel", 0.0),
                        speed_ms=speed_low_ms,
                        rho=config.fixed.rho_water,
                        g=config.fixed.gravity,
                    )
                    # Same Fn-based Delft envelope as the design speed (one
                    # physics, one curve — Bug #171 killed the 0.02/0.035 pair).
                    _Fn_low = float(speed_low_ms) / max(1e-9, (config.fixed.gravity * max(1e-9, hull_lwl)) ** 0.5)
                    Rw_low = capped_wave_resistance(
                        Rw_low_raw, nabla_ref, config.fixed.rho_water,
                        config.fixed.gravity,
                        cap_frac=_delft_cap_frac(_Fn_low))
                    # Normalize: lower Rw at low speed → higher bonus
                    rw_norm = min(Rw_low / max(1e-10, config.fixed.target_displacement), 10.0)
                    light_wind_bonus = config.weights.light_wind_bonus * max(0, 1.0 - rw_norm / 5.0)
                except Exception:
                    pass

            # Bug #168: no feasible-path draft debit. The old quadratic/cubic
            # penalty was tuned backwards from "dethrone design 157" — an
            # outcome target, not physics. Bug #169: depth logistics is a
            # priced inconvenience (draft_logistics_cost below), never a
            # per-Newton tax and never a wall — only T_total > LWL fails
            # (physical cap, constraints.py). Depth still pays honestly via
            # wetted/induced drag, light-wind Rw at full draft, and root
            # structure mass+VCG.

            # w3 (self-right) is gate-only as of Bug #168: it duplicated the
            # GZ reward already carried by stability_index. Computed for the
            # log; not added to FoM.
            self_right_score = constraints.get("self_righting", 0.0)
            if not np.isfinite(self_right_score):
                self_right_score = 0.0

            w = config.weights
            # Ocean-state total resistance: upright Rt + induced drag (keel
            # lift at the equilibrium leeway the balance solver already
            # finds — PYD Ch.5: "induced resistance is normally the most
            # important" heel-related term) + added resistance in waves
            # (Schaaf-style ~9.4% of Rt at modest seas, PYD Fig 5.26).
            # Gives BO the heel/leeway/seaway drag gradient the old
            # upright-only model hid.
            rt_ocean = float(result.rt_total)
            if result.balance is not None:
                try:
                    lw = max(float(result.balance.get("worst_leeway_ops_deg", 4.0)), 0.5)
                    Dk = float(x_dict.get("D_keel", 0.5))
                    kc = float(x_dict.get("keel_chord", 0.2))
                    A_plan = Dk * 0.75 * kc
                    AR_e = 2.0 * Dk / max(1e-9, 0.75 * kc)
                    cl_keel = (2.0 * np.pi * AR_e / max(1e-9, AR_e + 2.0)) * np.radians(min(lw, 12.0))
                    q = 0.5 * config.fixed.rho_water * speed_ms ** 2
                    ri = q * A_plan * cl_keel ** 2 / max(1e-9, np.pi * AR_e)
                    # Added resistance is second-order in wave height
                    # (Bug #171: was linear, overpredicting moderate seas ~2x).
                    _hsr = min(1.0, getattr(config.rapid_validation, "ops_Hs", 1.2) / 1.2)
                    raw = 0.12 * (result.rt_friction + result.rt_wave) * _hsr ** 2
                    rt_ocean = float(result.rt_total) + ri + raw
                except Exception:
                    rt_ocean = float(result.rt_total)
            result.rt_ocean = rt_ocean
            Rt_safe = max(0.5, rt_ocean)
            # Drag normalized to a reference resistance so w1/Rt spans
            # ~0.3-0.5 FoM instead of 0.004 (old: drowned by saturated
            # stability/self-right constants — audit A3).
            drag_ref = float(getattr(config.weights, "fom_drag_reference_n", 45.0))
            result.fom = (
                w.w1 * (drag_ref / Rt_safe)
                + w.w2 * stability_index
                + light_wind_bonus
                - w.w4 * accel_penalty
                - disp_penalty
                - helm_penalty
            )
            # Rapid-gate margin bonuses (w6–w9)
            if result.rapid is not None and result.rapid.margins:
                w6 = w.w6 if hasattr(w, 'w6') else 0.5
                w7 = w.w7 if hasattr(w, 'w7') else 0.5
                w8 = w.w8 if hasattr(w, 'w8') else 0.5
                w9 = w.w9 if hasattr(w, 'w9') else 0.5
                result.fom += w6 * _margin_bonus_clip(result.rapid.margins.get("capsize", -1))
                result.fom += w7 * _margin_bonus_clip(result.rapid.margins.get("storm_accel", -1))
                result.fom += w8 * _margin_bonus_clip(result.rapid.margins.get("slam_pressure", -1))
                result.fom += w9 * _margin_bonus_clip(result.rapid.margins.get("inverted_pressure", -1))

            # System-level mission bonus: ocean crossing lives on reach/run,
            # scored in three wind bands through the SAME balance solver
            # (Bug #169). Stored result.balance is the 10 kt medium band;
            # light (5 kt) + heavy (22 kt) are scoring-only extras (<5 ms each).
            # Reach drive per band vs Rt: deep keels earn here (drive at heel
            # with small leeway); stubby keels pay in leeway/heel. Gust
            # readiness = heavy-band heel headroom with a REAL gradient
            # (replaces the old saturated storm tanh). Upwind VMG kept for
            # wind shifts. Leeway priced linearly past the 4° designer norm
            # (PYD: 3-5° ideal) instead of the old 0.2-capped token.
            if result.balance is not None:
                bal = result.balance
                band_wts = (
                    float(getattr(w, "band_light_wt", 0.25)),
                    float(getattr(w, "band_medium_wt", 0.45)),
                    float(getattr(w, "band_heavy_wt", 0.30)),
                )
                ws = sum(band_wts)
                band_wts = tuple(b / ws for b in band_wts) if ws > 0 else (0.25, 0.45, 0.30)
                w_md = float(getattr(w, "w_mission_drive", 1.2))
                w_gu = float(getattr(w, "w_gust", 0.6))
                mission_drive = 0.0
                gust_margin = float(bal.get("heel_margin_ops", 0.0))
                heavy_leeway = float(bal.get("worst_leeway_ops_deg", 0.0))
                try:
                    med_reach = float(bal.get("reach_drive_ops_N", bal.get("mean_drive_ops_N", 0.0)))
                    if np.isfinite(med_reach) and med_reach > 0:
                        mission_drive += band_wts[1] * med_reach / Rt_safe
                    # Per-band boat speeds (Bug #169): drift at 2 kt, work at
                    # target, breeze-on at the 5 kt user max — never score a
                    # band at a speed the boat can't sail.
                    kt2ms = 0.514444
                    v_light = float(getattr(config.fixed, "band_light_kt", 2.0)) * kt2ms
                    v_heavy = float(getattr(config.fixed, "band_heavy_kt", 5.0)) * kt2ms
                    light = evaluate_balance_polar(
                        x_dict, config, gz, hydro,
                        boat_speed_ms=v_light, tws_ops_kt=5.0)
                    l_reach = float(light.get("reach_drive_ops_N", 0.0))
                    if np.isfinite(l_reach) and l_reach > 0:
                        mission_drive += band_wts[0] * l_reach / Rt_safe
                    heavy = evaluate_balance_polar(
                        x_dict, config, gz, hydro,
                        boat_speed_ms=v_heavy, tws_ops_kt=22.0)
                    h_reach = float(heavy.get("reach_drive_ops_N", 0.0))
                    if np.isfinite(h_reach) and h_reach > 0:
                        mission_drive += band_wts[2] * h_reach / Rt_safe
                    gm = heavy.get("heel_margin_ops", gust_margin)
                    if np.isfinite(gm):
                        gust_margin = float(gm)
                    hl = heavy.get("worst_leeway_ops_deg", heavy_leeway)
                    if np.isfinite(hl):
                        heavy_leeway = float(hl)
                except Exception:
                    pass
                result.fom += w_md * mission_drive
                result.fom += w_gu * gust_margin
                # Upwind VMG bonus (wind shifts happen mid-ocean)
                vmg_up = bal.get("vmg_up_N", 0.0)
                if np.isfinite(vmg_up) and vmg_up > 0:
                    result.fom += 0.3 * float(np.tanh(vmg_up / 50.0))
                # Leeway penalty, VMG-priced helper (linear past 4° norm)
                result.fom -= leeway_penalty_deg(heavy_leeway, config)
                result.mission_drive = mission_drive
                result.gust_margin = gust_margin
                result.heavy_leeway_deg = heavy_leeway
                # Persist mission scalars into constraints (→ DB constraint_values
                # JSON → results.md/CSV), same pattern as the balance scalars.
                constraints["mission_drive"] = mission_drive
                constraints["gust_margin"] = gust_margin
                constraints["heavy_leeway_deg"] = heavy_leeway

            # Draft logistics (Bug #169): inconvenience with a price, not a
            # wall (helper above). Absolute physical cap (T_total > LWL)
            # stays a hard gate in constraints.py.
            try:
                result.draft_logistics_cost = draft_logistics_cost(x_dict, config)
                result.fom -= result.draft_logistics_cost
                constraints["draft_logistics_cost"] = result.draft_logistics_cost
            except Exception:
                result.draft_logistics_cost = 0.0

            if not np.isfinite(result.fom):
                logger.warning(f"Non-finite FoM: {result.fom}, resetting to large penalty")
                result.error_code = "E_FOM_NAN"
                result.fom = -1e10
                result.feasible = False

            _agent_log("low_fidelity.py:fom", "FoM computed", {
                "fom": result.fom, "rt_total": result.rt_total,
                "rt_wave": result.rt_wave, "stability_index": stability_index,
                "helm_fwd_deg": result.helm_fwd_deg,
                "helm_aft_deg": result.helm_aft_deg,
                "helm_penalty": helm_penalty,
            }, "D")
            try:
                result.physical_params["struct_mass_kg"] = hydro.get("struct_mass_kg")
                result.physical_params["ballast_tip_kg"] = hydro.get("ballast_tip_kg")
                result.physical_params["ballast_fin_kg"] = hydro.get("ballast_fin_kg")
            except Exception:
                pass
        except Exception as e:
            result.error_code = f"E_FOM:{e}"
            # Fail-closed: a feasible design whose FoM crashed must not be
            # selected for validation.  -1e10 keeps the GP target finite
            # (never -inf) while get_top_n excludes it via error_code.
            result.fom = -1e10

    return result


# Absolute ceiling on BEM panel count. Capytaine builds dense N x N complex
# matrices (16 bytes/entry) for S and K per solve, so an undecimated NURBS
# tessellation (~28k faces, ~22k immersed) would need ~15GB per matrix pair.
# Never let a solve run above this even if the config requests more.
BEM_MAX_PANELS = 4000


class BemDecimationEnvError(RuntimeError):
    """BEM mesh decimation failed because a backend module is missing from the
    environment (e.g. fast_simplification not installed), not because the
    geometry is bad. Callers can distinguish this from a genuine ValueError
    (degenerate/undecimatable mesh)."""


def _truncate_msg(msg, limit=200):
    text = str(msg).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _decimate_bem_mesh(msh, config, n_target=None):
    """Quadric-decimate the BEM mesh: Capytaine factorizes a dense N x N
    matrix (O(N^3) LU), so halving panel count cuts solve time ~8x with
    negligible RAO accuracy loss at these hull sizes.

    Returns a mesh with at most min(bem_n_panels, BEM_MAX_PANELS) faces, or
    raises ValueError if no decimation backend succeeds. It must NEVER return
    the full mesh when decimation fails: running Capytaine on a ~20k-face
    mesh consumes ~10GB+ of RAM (see the 2026-08 Ray OOM incident) and kills
    the whole worker pool. If a backend module is missing (ImportError) and
    no backend produced a usable result, raises BemDecimationEnvError instead
    so the environment problem is distinguishable from geometry failure.
    n_target overrides the config target (used by generate_hull's
    generation-time decimation, where config may be None).
    """
    import meshio
    import numpy as _np
    try:
        n_before = len(msh.cells_dict["triangle"])
    except (KeyError, AttributeError):
        raise ValueError(f"BEM mesh has no triangle cells: {type(msh).__name__}")

    if n_target is None:
        configured = int(getattr(config.wave_spectrum, "bem_n_panels", 2500)) if config is not None else 2500
        n_target = configured
    n_target = int(n_target)
    if n_target <= 0:
        n_target = BEM_MAX_PANELS
    n_target = min(n_target, BEM_MAX_PANELS)
    if n_before <= n_target:
        return msh

    verts = _np.asarray(msh.points, dtype=_np.float64)
    faces = _np.asarray(msh.cells_dict["triangle"], dtype=_np.int32)
    if _np.isnan(verts).any() or not np.isfinite(verts).all():
        raise ValueError(
            f"BEM mesh contains NaN/Inf vertices ({int(_np.isnan(verts).sum())}): "
            "refusing to run Capytaine on corrupt geometry"
        )

    backends = [
        ("fast_simplification", _decimate_fast_simplification),
        ("trimesh_quadric", _decimate_trimesh_quadric),
    ]
    import_failures = {}
    for name, fn in backends:
        try:
            dv, df = fn(verts, faces, n_target)
            if df is None or len(df) < 500:
                logger.debug(f"BEM decimation {name} produced degenerate result")
                continue
            if len(df) > n_target:
                logger.debug(f"BEM decimation {name} exceeded target: {len(df)}")
                continue
            if not _np.isfinite(dv).all():
                logger.debug(f"BEM decimation {name} produced NaN vertices")
                continue
            logger.info(f"BEM mesh decimated: {n_before} -> {len(df)} faces via {name}")
            return meshio.Mesh(points=dv, cells=[("triangle", df)])
        except ImportError as e:
            import_failures[name] = getattr(e, "name", None) or str(e)
            logger.warning(
                f"BEM mesh decimation ({name}) failed: {type(e).__name__}: {_truncate_msg(e)}"
            )
        except Exception as e:
            logger.warning(
                f"BEM mesh decimation ({name}) failed: {type(e).__name__}: {_truncate_msg(e)}"
            )

    if import_failures:
        missing = sorted({str(m) for m in import_failures.values() if m})
        raise BemDecimationEnvError(
            f"BEM mesh decimation failed: missing backend module(s) "
            f"({', '.join(missing)}). This is an environment problem, not a "
            f"geometry problem. Install with: "
            f"/home/anon/apps/boat/venv/bin/pip install fast_simplification"
        )
    raise ValueError(
        f"BEM mesh could not be decimated to <= {n_target} faces (input {n_before}); "
        "refusing to run Capytaine on an oversized mesh"
    )


def _decimate_fast_simplification(verts, faces, n_target):
    """Quadric collapse via fast_simplification (meshoptimizer)."""
    import fast_simplification as _fs
    # aggressiveness=3 (positional 5th arg): higher values can produce
    # non-manifold edges that crash Capytaine's heal_mesh().
    return _fs.simplify(verts, faces, None, n_target, 3)


def _decimate_trimesh_quadric(verts, faces, n_target):
    """Quadric decimation via trimesh (falls back when fast_simplification
    chokes on non-manifold input)."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    decimated = mesh.simplify_quadric_decimation(face_count=n_target)
    return decimated.vertices, decimated.faces


def _compute_raos_capytaine(stl_path: str, config, x_dict: dict,
                            hydro: dict, speed_ms: float) -> dict:
    """Compute RAOs via Capytaine (zero-speed BEM).
    Note: speed_ms is accepted but unused — Capytaine solves zero-speed
    diffraction/radiation only (no forward-speed Green function).
    Forward-speed effects are neglected at this fidelity level.
    Returns the RAO dict including wall_time_s (total BEM wall time)."""
    _t_bem0 = time.time()
    import capytaine as cpy
    from capytaine.post_pro.rao import rao as compute_rao
    from capytaine.io.xarray import assemble_dataset
    import meshio
    import xarray as xr
    from hull_opt.utils import reassert_pipeline_logging
    reassert_pipeline_logging(config)

    # Prefer the generation-time decimated copy (R2.2) when present: it sits
    # next to stl_path in the same output dir and skips per-eval decimation.
    bem_stl = str(Path(stl_path).parent / "hull_geometry_bem.stl")
    if Path(bem_stl).is_file():
        stl_path = bem_stl
    msh = meshio.read(stl_path)
    msh = _decimate_bem_mesh(msh, config)
    n_panels = len(msh.cells_dict["triangle"])
    if n_panels > BEM_MAX_PANELS:
        raise ValueError(
            f"BEM mesh has {n_panels} panels > cap {BEM_MAX_PANELS}; "
            "aborting solve to avoid OOM"
        )
    body = cpy.FloatingBody.from_meshio(msh, name="hull")
    # capytaine 2.x: keep_immersed_part() mutates in place; 3.x: immersed_part() returns new body
    if hasattr(body, "keep_immersed_part"):
        body.keep_immersed_part()
    else:
        body = body.immersed_part()

    nabla = hydro.get("nabla", 0.25)
    rho = config.fixed.rho_water
    g = config.fixed.gravity

    BWL = x_dict.get("BWL", 0.5)
    T_canoe = x_dict.get("T_canoe", 0.2)

    # CRITICAL ORDER: set center_of_mass BEFORE adding the rigid-body dofs —
    # the rotation dofs are created about center_of_mass; adding them first
    # (mesh centroid ~ origin) produced spurious roll-yaw / heave-pitch
    # hydrostatic coupling (-1131 N·m/rad) that polluted the roll RAO via
    # the singular yaw row (audit: storm roll sigma ~280°).
    from hull_opt.hydrostatics import compute_cg_z as _cg, compute_cg_x as _cgx
    cg_z = hydro.get("cg_z", _cg(x_dict, nabla=hydro.get("underwater_volume", hydro.get("nabla")), config=config))
    cg_x = hydro.get("cg_x", _cgx(x_dict, config, cb_x=float(hydro.get("CB_x", 0.0))))
    body.center_of_mass = np.array([cg_x, 0.0, cg_z])
    body.add_all_rigid_body_dofs()

    # Inertia from the REAL lumped mass model (hull shell, keel, bulb+ballast
    # at keel depth, payload at deck, mast) — replaces Capytaine's
    # uniform-density panel inertia, which overestimates roll inertia by
    # ~15-25% and ignores the 100+ kg ballast at keel depth (audit A9).
    _std_dofs = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]
    _dof_names = list(body.dofs)
    _sel = [_std_dofs.index(n) for n in _dof_names]
    body.inertia_matrix = xr.DataArray(
        compute_lumped_inertia(x_dict, hydro, config)[np.ix_(_sel, _sel)],
        dims=["influenced_dof", "radiating_dof"],
        coords={"influenced_dof": _dof_names, "radiating_dof": _dof_names},
        name="inertia_matrix")
    body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness(rho=rho, g=g)

    solver = cpy.BEMSolver()

    bem = config.wave_spectrum
    omega_range = np.linspace(
        bem.bem_omega_min, bem.bem_omega_max, bem.bem_n_freq
    )

    # Head-seas diffraction (heave/pitch/accel metrics) + radiation.
    problems = []
    for omega in omega_range:
        problems.append(cpy.DiffractionProblem(
            body=body, omega=omega, wave_direction=np.pi, rho=rho, g=g
        ))
        for dof_name in body.dofs:
            problems.append(cpy.RadiationProblem(
                body=body, omega=omega, radiating_dof=dof_name, rho=rho, g=g
            ))

    results = solver.solve_all(problems, n_jobs=1)
    dataset = assemble_dataset(results, hydrostatics=True)

    dataset["inertia_matrix"] = body.inertia_matrix
    dataset["hydrostatic_stiffness"] = body.hydrostatic_stiffness

    rao_result = compute_rao(dataset)

    omega_vals = dataset.omega.values

    roll_raos = np.abs(rao_result.sel(radiating_dof="Roll").values.squeeze())
    heave_raos = np.abs(rao_result.sel(radiating_dof="Heave").values.squeeze())
    pitch_raos = np.abs(rao_result.sel(radiating_dof="Pitch").values.squeeze())

    # Beam-seas roll: head-seas roll RAO is ~zero by symmetry (symmetric
    # hull), so the old roll_period was a grid histogram artifact. Solve
    # the beam heading for roll; reuse the radiation results.
    try:
        beam_diff = [cpy.DiffractionProblem(
            body=body, omega=omega, wave_direction=np.pi / 2.0, rho=rho, g=g
        ) for omega in omega_range]
        beam_res = solver.solve_all(beam_diff, n_jobs=1)
        beam_dataset = assemble_dataset(results + beam_res, hydrostatics=True)
        beam_dataset["inertia_matrix"] = body.inertia_matrix
        beam_dataset["hydrostatic_stiffness"] = body.hydrostatic_stiffness
        beam_rao = compute_rao(beam_dataset)
        roll_raos_beam = np.abs(np.asarray(
            beam_rao.sel(radiating_dof="Roll")
            .sel(wave_direction=np.pi / 2.0, method="nearest").values).squeeze())
        if np.asarray(roll_raos_beam).ndim == 1 and len(roll_raos_beam) == len(omega_vals):
            roll_raos = roll_raos_beam
    except Exception as e:
        logger.warning(f"Beam-seas roll RAO failed, keeping head-seas: {e}")

    # Guarantee a 1-D roll spectrum before damping/peak logic.
    roll_raos = np.asarray(roll_raos)
    if roll_raos.ndim > 1:
        roll_raos = roll_raos.reshape(-1)[:len(omega_vals)]

    # Keel-vortex roll damping (PYD p.56: "by far the most important" roll
    # damping source; SIMP-paper anchor zeta=0.42). Capytaine's RAO includes
    # only radiation damping (small); the vortex term is added by scaling
    # the RAO by the damped/undamped amplification ratio. The natural
    # frequency is taken from the RAO's OWN peak (the analytic 0.35·B
    # estimate sits far from the BEM resonance for a ballast-heavy hull —
    # anchoring on it left the attenuation factor ~1 at the wrong omega).
    try:
        _peak_i = int(np.argmax(roll_raos)) if len(roll_raos) > 1 else 0
        wn = float(omega_vals[_peak_i]) if 0 < _peak_i < len(omega_vals) - 1 \
            and omega_vals[_peak_i] > 0 else None
        if wn is None or not np.isfinite(wn):
            gm_h = hydro.get("GM", hydro.get("gm", 0.1))
            t_roll_an = 2.0 * np.pi * 0.35 * x_dict.get("BWL", 0.55) / \
                max(np.sqrt(config.fixed.gravity * max(gm_h, 0.01)), 1e-6)
            wn = 2.0 * np.pi / max(t_roll_an, 1e-6)
        r = omega_vals / max(wn, 1e-6)
        zeta = 0.3 + 0.25 * (x_dict.get("D_keel", 0.5) / max(1e-9, x_dict["LWL"]))
        zeta_rad = 0.01
        h_rad = np.sqrt((1.0 - r ** 2) ** 2 + (2.0 * zeta_rad * r) ** 2)
        h_tot = np.sqrt((1.0 - r ** 2) ** 2 + (2.0 * zeta * r) ** 2)
        roll_raos = roll_raos * (h_rad / np.maximum(h_tot, 1e-9))
    except Exception as e:
        logger.warning(f"Roll-damping correction failed: {e}")

    roll_period = 0.0
    if len(roll_raos) > 2:
        peak_idx = np.argmax(roll_raos)
        if 0 < peak_idx < len(omega_vals) - 1 and omega_vals[peak_idx] > 0:
            # Parabolic interpolation for better peak resolution
            y0, y1, y2 = roll_raos[peak_idx-1], roll_raos[peak_idx], roll_raos[peak_idx+1]
            x0, x1, x2 = omega_vals[peak_idx-1], omega_vals[peak_idx], omega_vals[peak_idx+1]
            denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
            if abs(denom) > 1e-15:
                a = ((x2 - x1) * y0 + (x0 - x2) * y1 + (x1 - x0) * y2) / denom
                b = ((x1**2 - x2**2) * y0 + (x2**2 - x0**2) * y1 + (x0**2 - x1**2) * y2) / denom
                if a < 0:
                    omega_peak = -b / (2 * a)
                    if omega_vals[peak_idx-1] <= omega_peak <= omega_vals[peak_idx+1]:
                        roll_period = 2 * np.pi / omega_peak
                    else:
                        roll_period = 2 * np.pi / omega_vals[peak_idx]
                else:
                    roll_period = 2 * np.pi / omega_vals[peak_idx]
            else:
                roll_period = 2 * np.pi / omega_vals[peak_idx]

    peak_accel = _compute_peak_accel(
        heave_raos, pitch_raos, omega_vals,
        config, g, body
    )

    return {
        "roll_period": float(roll_period),
        "peak_accel": peak_accel,
        "omega": omega_vals.tolist(),
        "roll_rao": roll_raos.tolist(),
        "heave_rao": heave_raos.tolist(),
        "pitch_rao": pitch_raos.tolist(),
        "wall_time_s": float(time.time() - _t_bem0),
    }


def _compute_peak_accel(heave_rao, pitch_rao, omega, config,
                        g, body) -> float:
    try:
        Hs = config.wave_spectrum.Hs
        gamma = config.wave_spectrum.gamma
        n_freq = config.wave_spectrum.n_freq
        Tp = getattr(config.wave_spectrum, 'Tp', 9.0)
        if Tp <= 0 or Hs <= 0:
            logger.warning(f"Invalid JONSWAP parameters: Hs={Hs}, Tp={Tp}")
            return PEAK_ACCEL_FALLBACK_G
        electronics_z = config.fixed.electronics_bay[2]

        omega_min = max(0.1, min(omega) if len(omega) > 0 else 0.1)
        omega_max = max(omega) if len(omega) > 0 else 6.0
        omega_sp = np.linspace(omega_min, omega_max, n_freq)

        f = omega_sp / (2 * np.pi)
        fp = 1.0 / Tp

        S_Hz = np.zeros_like(omega_sp)
        for i, fi in enumerate(f):
            if fi <= 0:
                continue
            sigma = 0.07 if fi <= fp else 0.09
            # Bug #171 (regression of Bug #71): see rapid_gates.py — 0.0081.
            alpha = 0.0081
            beta = -1.25 * (fp / fi) ** 4
            gamma_term = gamma ** np.exp(-0.5 * ((fi - fp) / (sigma * fp)) ** 2)
            S_Hz[i] = alpha * Hs ** 2 * (fp / fi) ** 4 * np.exp(beta) * gamma_term / fi

        # Convert JONSWAP from Hz to rad/s: S(ω) = S(f) / (2π)
        # because S(ω) dω = S(f) df with dω = 2π df.
        # Bug #171: self-normalize to m0 = (Hs/4)^2 (see rapid_gates.py).
        S = S_Hz / (2.0 * np.pi)
        try:
            _m0 = float(np.trapezoid(S, omega_sp))
            _tgt = (float(Hs) / 4.0) ** 2
            if _m0 > 0 and np.isfinite(_m0):
                S = S * (_tgt / _m0)
        except Exception:
            pass

        heave_interp = np.interp(omega_sp, omega, heave_rao, left=0, right=0)
        pitch_interp = np.interp(omega_sp, omega, pitch_rao, left=0, right=0)

        x_eb = config.fixed.electronics_bay[0]
        # Guard: a payload at x=0 has zero pitch lever (the old default
        # [0,0,-0.05] made the pitch term vanish identically). Fall back to
        # the bow quarter where the IMU/electronics actually mount.
        if abs(x_eb) < 1e-9:
            x_eb = 0.25 * 2.4
        accel_response = heave_interp + pitch_interp * abs(x_eb)
        accel_squared = (accel_response * omega_sp ** 2) ** 2

        m0 = _trapz(accel_squared * S, omega_sp)
        significant_accel = 4.0 * np.sqrt(max(0, m0))
        peak_accel = 1.86 * significant_accel / g
        # 1.86 converts significant double-amplitude (4σ) to expected
        # peak-to-peak maximum over ~1000 cycles: σ·√(2·ln(1000))/σ·4 * 4σ
        # = √(2·ln(1000))/4 * significant_accel ≈ 0.93 * significant_accel / g
        # for single-amplitude. 1.86 gives peak-to-peak, which is the
        # convention used for structural load qualification in this design.

        return float(peak_accel)
    except Exception as e:
        logger.warning(f"Peak acceleration computation failed: {e}")
        return PEAK_ACCEL_FALLBACK_G
