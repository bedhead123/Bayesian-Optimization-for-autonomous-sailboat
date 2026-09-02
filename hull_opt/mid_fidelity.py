"""
Mid-fidelity calibration using DualSPHysics (SPH) resistance simulation.

Generates geometry, builds SPH towing case, runs GenCase + solver + ComputeForces,
extracts towing force, and computes calibration delta between low-fi and SPH.
Key exports: run_mid_fidelity_calibration()
"""
import logging
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

from hull_opt.geometry import generate_hull
from hull_opt.utils import knots_to_ms, ensure_dir


def calibration_factor(rt_sph: float, Rf: float, Rw: float,
                       factor_bounds: tuple = (0.5, 2.0)) -> tuple[float | None, str | None]:
    """B1+B3: drag-factor validity gate and de-amplified ratio.

    Returns (factor, reason). factor=None with a reason means the
    calibration is INVALID:
    - rt_sph < Rf (sub-friction, physically impossible for a towed hull)
    - Rw <= 0 (no wave resistance to calibrate)
    - raw factor outside ``factor_bounds``: the SPH tow disagrees with the
      CFD-validated low-fi model by more than the band allows, so the
      measurement is out-of-family (Bug #157: dp-coarse SPH tows read
      20x; correcting by a clipped 5.0 corrupted every FoM). The tow is
      treated as an audit failure — the factor stays at 1.0.

    Inside the band the factor is clip(raw, *factor_bounds) — the
    denominator is floored at 0.3*Rf so ±5 N of SPH noise on a ~10 N wave
    term cannot swing the factor across its whole range.
    """
    lo, hi = float(factor_bounds[0]), float(factor_bounds[1])
    if not np.isfinite(rt_sph) or not np.isfinite(Rf) or not np.isfinite(Rw):
        return None, f"non-finite inputs (rt_sph={rt_sph}, Rf={Rf}, Rw={Rw})"
    # rt_sph <= Rf: sub-friction OR exactly-friction (factor would be 0).
    # Both are physically implausible for a towed hull and would pollute
    # the stored factor with a floor-clipped value.
    if rt_sph <= Rf:
        return None, f"rt_sph={rt_sph:.4f} N <= Rf={Rf:.4f} N (sub-friction, invalid)"
    if Rw > 1e-6:
        denom = max(Rw, 0.3 * Rf)
        raw = (rt_sph - Rf) / denom
        if raw < lo or raw > hi:
            return None, (
                f"raw drag factor {raw:.2f} outside validated band "
                f"[{lo}, {hi}] — SPH tow out-of-family, refusing to correct"
            )
        return float(np.clip(raw, lo, hi)), None
    return None, f"Rw={Rw:.4f} N <= 0 (no wave resistance to calibrate)"


def run_mid_fidelity_calibration(design_vector: np.ndarray,
                                  design_id: int, iteration: int,
                                  config) -> dict | None:
    logger.info(f"Mid-fidelity calibration: design {design_id}, iteration {iteration}")

    solver = getattr(config.calibration, "solver", "sph")
    if solver != "sph":
        raise ValueError("OpenFOAM solver no longer supported, use 'sph'")

    try:
        output_root = ensure_dir(Path(config.paths.output_dir) / "calibration")
        case_dir = output_root / f"design_{design_id}_iter_{iteration}"
        if case_dir.exists():
            shutil.rmtree(case_dir)

        from hull_opt.param_layer import design_vector_to_physical
        x_dict = design_vector_to_physical(design_vector, config)
        hull_lwl = float(x_dict.get("LWL", config.fixed.LWL))
        B = x_dict["BWL"]
        T_hull = x_dict["T_canoe"]
        D_keel = x_dict.get("D_keel", 0.0)
        speed_knots = config.fixed.target_speed_knots
        speed_ms = knots_to_ms(speed_knots)
        rho = config.fixed.rho_water
        g = config.fixed.gravity

        # Validate design vector before geometry generation
        from hull_opt.geometry_validator import validate_design_vector
        is_valid, msg = validate_design_vector(x_dict, config)
        if not is_valid:
            raise ValueError(f"Design vector validation failed: {msg}")

        # Generate geometry
        stl_path, sac_path, hydro, hull_stl = generate_hull(
            design_vector,
            output_dir=str(case_dir),
            LWL=hull_lwl,
            target_displacement=config.fixed.target_displacement,
            config=config,
        )
        logger.info(f"Geometry generated: {stl_path}")

        # ── SPH calibration path ──────────────────────────────────────────
        from hull_opt.sph_resistance import build_towing_case, run_towing_resistance

        dp = getattr(config.calibration, "sph_dp", 0.05)
        sim_time = getattr(config.calibration, "sph_sim_time", 6.0)
        timeout = getattr(config.calibration, "timeout", 3600)
        conv = getattr(config.calibration, "convergence", {}) or {}

        # Render the towing case BEFORE preflight so the preflight validates
        # the actual case that will be run (previously it ran against an
        # empty case_dir and always aborted on the missing XML).
        # run_towing_resistance re-renders idempotently before running.
        build_towing_case(case_dir, stl_path, x_dict, config, speed_ms,
                          dp=dp, sim_time=sim_time)

        # SPH preflight: check solver binary, STL, etc.
        from hull_opt.preflight import preflight_case
        pf_ok, pf_msgs = preflight_case(case_dir, config)
        for line in pf_msgs:
            logger.info(f"  PREFLIGHT {line}")
        if not pf_ok:
            logger.error("Preflight failed; aborting calibration")
            return None

        result = run_towing_resistance(
            case_dir, stl_path, x_dict, config, speed_ms,
            dp=dp, sim_time=sim_time, timeout_s=timeout,
            gpu_lock=str(Path(config.paths.output_dir) / "gpu.lock"),
            lock_wait=True,
            window_s=float(conv.get("window_s", 3.0)),
            min_sim_time=float(conv.get("min_sim_time", sim_time * 0.8)),
            # Fz sanity: a hull seeing vertical fluid force beyond 2x the
            # DESIGN displacement buoyancy is measuring tank pressurization
            # (Bug #157 signature: 6-12 kN runaways), not drag. Uses the
            # design displacement (config), not hydro["nabla"] — the towed
            # body is the full hull+keel+bulb and nabla is hull-only, which
            # made the bound tighter than honest dynamic Fz (observed:
            # 2038 N measured vs 1866 N nabla-bound -> false rejection).
            max_fz_n=2.0 * rho * g * float(config.fixed.target_displacement),
        )

        rt_sph = result["rt_n"]
        if rt_sph is None or result["status"] != "OK":
            logger.error(f"SPH towing failed: {result['details']}")
            logger.error(
                f"<calib FAILED design={design_id} iter={iteration} rc={result['status']} "
                f"t={result.get('wall_time_s', 0.0):.1f}s np={result.get('np_particles', 0)}>"
            )
            return None

        # Steady-state check: same guard as the high-fi gate — a calibration
        # from a non-steady force trace is invalid and must not move the factor.
        from hull_opt.high_fidelity import _check_force_steady
        steady_ok, steady_msg = _check_force_steady(
            result.get("force_csv"), "calib",
        )
        if not steady_ok:
            logger.warning(f"Calibration non-steady trace: {steady_msg}")
            logger.warning(
                f"<calib INVALID design={design_id} iter={iteration} "
                f"reason=non-steady trace>"
            )
            return {"valid": False, "reason": steady_msg}

        logger.info(f"Rt_SPH = {rt_sph:.4f} N")

        # Compute low-fidelity prediction to get wave-resistance correction factor
        try:
            from hull_opt.geometry import compute_half_breadth_analytic, keel_half_breadth
            from hull_opt.michell import compute_wave_resistance_michell
            from hull_opt.friction import compute_total_resistance
            import trimesh

            sac_scale = hydro.get("sac_scale_factor", 1.0)

            def hb_func(xq, zq, _ss=sac_scale):
                hb = compute_half_breadth_analytic(xq, zq, x_dict, hull_lwl, sac_scale=_ss)
                if D_keel > 0.01 and x_dict.get("keel_chord", 0.0) > 0.01:
                    hb = hb + keel_half_breadth(xq, zq, x_dict, hull_lwl)
                return hb

            Rw = compute_wave_resistance_michell(
                hb_func, hull_lwl, B, T_hull + D_keel, speed_ms, rho, g,
                n_z=max(20, int(np.ceil(20 * (T_hull + D_keel) / max(T_hull, 1e-6)))),
            )

            hull_mesh_obj = trimesh.load(hull_stl)
            if isinstance(hull_mesh_obj, trimesh.Scene):
                hull_mesh_obj = hull_mesh_obj.dump(concatenate=True)
            try:
                underwater = trimesh.intersections.slice_mesh_plane(
                    hull_mesh_obj, [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], cap=False,
                )
                if underwater is not None and hasattr(underwater, 'area') and underwater.area > 0:
                    area = underwater.area
                else:
                    area = hull_mesh_obj.area
            except Exception:
                area = hull_mesh_obj.area if hasattr(hull_mesh_obj, 'area') else 1.0

            Rt_lowfi, Rf, Rw_out = compute_total_resistance(
                speed_ms, area, hull_lwl, rho, config.fixed.nu_water, wave_resistance=Rw,
            )
            logger.info(
                f"Low-fi components: Rw={Rw_out:.4f} N, Rf={Rf:.4f} N, "
                f"Rt={Rt_lowfi:.4f} N, wetted_area={area:.6f} m²"
            )

            # B1+B3: validity gate + de-amplified factor (pure function).
            # factor_bounds: outside the band the tow is out-of-family vs
            # the CFD-validated low-fi model (0.73-1.08 historically) and
            # the calibration is refused instead of clipping (Bug #157).
            factor_bounds = tuple(getattr(config.calibration, "factor_bounds", (0.5, 2.0)))
            factor, invalid_reason = calibration_factor(rt_sph, Rf, Rw_out,
                                                        factor_bounds=factor_bounds)
            if invalid_reason is not None:
                logger.warning(f"Calibration invalid: {invalid_reason}")
                logger.warning(
                    f"<calib INVALID design={design_id} iter={iteration} "
                    f"reason={invalid_reason}>"
                )
                return {"valid": False, "reason": invalid_reason}
            delta = float(rt_sph - Rt_lowfi)
            logger.info(
                f"Calibration: factor={factor:.4f} (SPH={rt_sph:.4f}, "
                f"low-fi={Rt_lowfi:.4f}, delta={delta:.4f} N)"
            )
            return {
                "valid": True,
                "factor": factor,
                "rt_cfd": rt_sph,
                "rt_lowfi": Rt_lowfi,
                "delta": delta,
                "rf": Rf,
                "rw": Rw_out,
            }
        except Exception as e:
            logger.warning(f"Low-fi prediction failed, returning None: {e}")
            return None

    except Exception as e:
        logger.error(f"Mid-fidelity calibration failed: {e}")
        return None
