"""
Low-fidelity evaluation pipeline: geometry → GZ curve → Michell wave
resistance → friction → Capytaine RAOs → constraints → FoM.
Primary evaluation function used by the BO surrogate.
Key exports: evaluate_low_fidelity(), EvaluationResult
Bugs fixed: GM sign error in roll period estimate (#4)
"""
import hashlib
import json
import time
import numpy as np
import trimesh
from pathlib import Path
from typing import Optional
import logging
import warnings

_DEBUG_LOG = "/home/anon/apps/boat/.cursor/debug-b990d8.log"


def _agent_log(location: str, message: str, data: dict, hypothesis_id: str, run_id: str = "pre-fix"):
    # #region agent log
    try:
        with open(_DEBUG_LOG, "a") as f:
            f.write(json.dumps({
                "sessionId": "b990d8", "location": location, "message": message,
                "data": data, "hypothesisId": hypothesis_id, "runId": run_id,
                "timestamp": int(time.time() * 1000),
            }) + "\n")
    except Exception:
        pass
    # #endregion

warnings.filterwarnings("ignore", message=".*different number of dimensions.*")
warnings.filterwarnings("ignore", message=".*Capytaine failed.*")
logging.getLogger("capytaine").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

from hull_opt.geometry import generate_hull, compute_half_breadth_analytic
from hull_opt.param_layer import design_vector_to_physical
from hull_opt.geometry_validator import validate_hull_geometry, validate_design_vector
from hull_opt.hydrostatics import compute_gz_curve, compute_righting_energy, compute_cg_z
from hull_opt.constraints import evaluate_constraints
from hull_opt.michell import compute_wave_resistance_michell
from hull_opt.friction import compute_total_resistance


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
        self.constraint_values = {}
        self.constraint_violations = []
        self.error_code = None
        self.cad_stl_path = None
        self.cad_sac_path = None
        self.hydro = {}
        self.gz_curve = None


def evaluate_low_fidelity(design_vector: np.ndarray, config,
                          output_dir: Optional[str] = None,
                          drag_correction: float = 0.0,
                          iteration: Optional[int] = None) -> EvaluationResult:
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

        # Validate geometry for degenerate shapes (hull-only; keel/bulb are appendages)
        is_valid, val_msg = validate_hull_geometry(hull_stl)
        if not is_valid:
            result.error_code = f"E_VALIDATE:{val_msg}"
            return result
    except Exception as e:
        result.error_code = f"E_GEOM:{e}"
        return result

    # 2. GZ curve with ballast-derived CG (hull-only mesh, keel CG approximated)
    try:
        cg_z = hydro.get("cg_z", compute_cg_z(x_dict, nabla=hydro.get("underwater_volume", hydro.get("nabla"))))
        if not np.isfinite(cg_z):
            raise ValueError(f"Non-finite CG_z: {cg_z}")
        gz = compute_gz_curve(
            hull_stl,
            cg_z=cg_z,
            n_angles=37,
            max_heel=180.0,
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
        except Exception:
            wetted_area = mesh.area

        sac_scale = hydro.get("sac_scale_factor", 1.0)
        def half_breadth_func(xq, zq, _ss=sac_scale):
            return compute_half_breadth_analytic(xq, zq, x_dict, hull_lwl, sac_scale=_ss)

        Rw = compute_wave_resistance_michell(
            half_breadth_func,
            LWL=hull_lwl,
            B=x_dict["BWL"],
            T=x_dict["T_canoe"],
            speed_ms=speed_ms,
            rho=config.fixed.rho_water,
            g=config.fixed.gravity,
        )
        Rt, Rf, Rw_out = compute_total_resistance(
            speed_ms, wetted_area, hull_lwl,
            rho=config.fixed.rho_water, nu=config.fixed.nu_water,
            wave_resistance=Rw,
        )
        Rt += drag_correction
        result.rt_total = Rt
        result.rt_wave = Rw
        result.rt_friction = Rf
        logger.debug(f"Resistance: Rw={Rw:.4f}, Rf={Rf:.4f}, Rt={Rt:.4f}, "
                     f"wetted_area={wetted_area:.6f}, drag_corr={drag_correction:.4f}, "
                     f"speed_ms={speed_ms:.4f}")
        for val_name, val in [("Rt", Rt), ("Rf", Rf), ("Rw", Rw)]:
            if not np.isfinite(val) or val < 0:
                raise ValueError(f"Non-finite or negative {val_name}: {val}")
    except Exception as e:
        import traceback
        result.error_code = f"E_MICHELL:{e}\n{traceback.format_exc()}"
        return result

    # 4. Capytaine RAOs (heave, pitch, roll)
    try:
        rao_data = _compute_raos_capytaine(
            stl_path, config, x_dict, hydro, speed_ms
        )
        result.roll_period = rao_data.get("roll_period", 0.0)
        result.peak_accel = rao_data.get("peak_accel", 0.0)
        if not np.isfinite(result.peak_accel):
            result.peak_accel = 60.0
        if not np.isfinite(result.roll_period):
            result.roll_period = 0.0
    except ImportError:
        logger.warning("Capytaine not available; using RAO estimates")
        result.roll_period = _estimate_roll_period(hydro, x_dict)
        result.peak_accel = 60.0
    except Exception as e:
        logger.error(f"Capytaine RAO computation failed: {e}", exc_info=True)
        result.roll_period = _estimate_roll_period(hydro, x_dict)
        result.peak_accel = 60.0

    # 5. Constraints (pass x_dict, config, stl_path for new checks)
    try:
        feasible, violations, constraints, violation_magnitude = evaluate_constraints(
            hydro, gz, roll_period=result.roll_period,
            peak_accel=result.peak_accel,
            x_dict=x_dict, config=config, stl_path=stl_path,
            hull_stl_path=hull_stl,
            iteration=iteration,
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
    except Exception as e:
        result.error_code = f"E_CONSTRAINTS:{e}"
        return result

    # 6. FoM
    if not result.feasible:
        # Rule 3: graded negative FoM so GP learns constraint boundary
        violation_mag = result.violation_magnitude if hasattr(result, 'violation_magnitude') else 1.0
        result.fom = -max(violation_mag, 0.01)
    else:
        try:
            righting_energy = constraints.get("righting_energy", 0.0)
            stability_index = righting_energy / max(
                1e-10, config.weights.stability_normalization
            )
            if not np.isfinite(stability_index):
                stability_index = 0.0
            else:
                stability_index = max(0.0, min(stability_index, 2.0))
            result.stability_index = stability_index

            # No crew comfort roll band penalty (autonomous vessel)

            accel_penalty = 0.0
            if result.peak_accel > config.validation.max_accel_g:
                accel_penalty = 2.0 * (result.peak_accel - config.validation.max_accel_g)

            # Displacement mismatch penalty: prevent AI from undersizing hull
            target_nabla = hydro.get("target_nabla", config.fixed.target_displacement)
            nabla = hydro.get("underwater_volume", hydro.get("nabla", target_nabla))
            vol_delta = abs(nabla - target_nabla) / max(1e-10, target_nabla)
            disp_penalty = 0.5 * vol_delta

            # Light-wind bonus for bimodal El Niño conditions
            # Designs with good sail area-to-drag at low speed get bonus
            light_wind_bonus = 0.0
            if hasattr(config.weights, 'light_wind_bonus') and config.weights.light_wind_bonus > 0:
                # Lower wave resistance at low speed = better light-wind performance
                speed_low_ms = 0.5 * config.fixed.target_speed_knots * 0.514444
                try:
                    Rw_low = compute_wave_resistance_michell(
                        half_breadth_func,
                        LWL=hull_lwl,
                        B=x_dict["BWL"],
                        T=x_dict["T_canoe"],
                        speed_ms=speed_low_ms,
                        rho=config.fixed.rho_water,
                        g=config.fixed.gravity,
                    )
                    # Normalize: lower Rw at low speed → higher bonus
                    rw_norm = min(Rw_low / max(1e-10, config.fixed.target_displacement), 10.0)
                    light_wind_bonus = config.weights.light_wind_bonus * max(0, 1.0 - rw_norm / 5.0)
                except Exception:
                    pass

            self_right_score = constraints.get("self_righting", 0.0)
            if not np.isfinite(self_right_score):
                self_right_score = 0.0

            w = config.weights
            Rt_safe = max(0.5, result.rt_total)
            result.fom = (
                w.w1 / Rt_safe
                + w.w2 * stability_index
                + w.w3 * self_right_score
                + light_wind_bonus
                - w.w4 * accel_penalty
                - disp_penalty
            )
            if not np.isfinite(result.fom):
                logger.warning(f"Non-finite FoM: {result.fom}, resetting to large penalty")
                result.fom = 1e10

            _agent_log("low_fidelity.py:fom", "FoM computed", {
                "fom": result.fom, "rt_total": result.rt_total,
                "rt_wave": result.rt_wave, "stability_index": stability_index,
            }, "D")
        except Exception as e:
            result.error_code = f"E_FOM:{e}"

    return result


def _compute_raos_capytaine(stl_path: str, config, x_dict: dict,
                            hydro: dict, speed_ms: float) -> dict:
    # speed_ms is accepted but unused — Capytaine solves zero-speed
    # diffraction/radiation only (no forward-speed Green function).
    import capytaine as cpy
    from capytaine.post_pro.rao import rao as compute_rao
    from capytaine.io.xarray import assemble_dataset
    import meshio
    import xarray as xr

    msh = meshio.read(stl_path)
    body = cpy.FloatingBody.from_meshio(msh, name="hull")
    body.keep_immersed_part()

    nabla = hydro.get("nabla", 0.25)
    rho = config.fixed.rho_water
    g = config.fixed.gravity

    BWL = x_dict.get("BWL", 0.5)
    T_canoe = x_dict.get("T_canoe", 0.2)

    body.add_all_rigid_body_dofs()

    from hull_opt.hydrostatics import compute_cg_z as _cg
    cg_z = hydro.get("cg_z", _cg(x_dict, nabla=hydro.get("underwater_volume", hydro.get("nabla"))))
    cg = np.array([0.0, 0.0, cg_z])
    body.center_of_mass = cg

    body.inertia_matrix = body.compute_rigid_body_inertia(rho=rho)
    body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness(rho=rho, g=g)

    solver = cpy.BEMSolver()

    omega_range = np.linspace(0.2, 6.0, 15)

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

    roll_period = 0.0
    if len(roll_raos) > 2:
        peak_idx = np.argmax(roll_raos)
        if peak_idx < len(omega_vals) and omega_vals[peak_idx] > 0:
            roll_period = 2 * np.pi / omega_vals[peak_idx]

    peak_accel = _compute_peak_accel(
        heave_raos, pitch_raos, omega_vals,
        config, stl_path, g, body
    )

    return {
        "roll_period": float(roll_period),
        "peak_accel": peak_accel,
        "omega": omega_vals.tolist(),
        "roll_rao": roll_raos.tolist(),
        "heave_rao": heave_raos.tolist(),
        "pitch_rao": pitch_raos.tolist(),
    }


def _compute_peak_accel(heave_rao, pitch_rao, omega, config,
                        stl_path, g, body) -> float:
    try:
        import capytaine as cpy
    except ImportError:
        return 10.0

    try:
        Hs = config.wave_spectrum.Hs
        gamma = config.wave_spectrum.gamma
        n_freq = config.wave_spectrum.n_freq
        Tp = getattr(config.wave_spectrum, 'Tp', 9.0)
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
            alpha = 5.0 / 16.0
            beta = -1.25 * (fp / fi) ** 4
            gamma_term = gamma ** np.exp(-0.5 * ((fi - fp) / (sigma * fp)) ** 2)
            S_Hz[i] = alpha * Hs ** 2 * (fp / fi) ** 4 * np.exp(beta) * gamma_term / fi

        # Convert JONSWAP from Hz to rad/s: S(ω) = S(f) / (2π)
        # because S(ω) dω = S(f) df with dω = 2π df
        S = S_Hz / (2.0 * np.pi)

        heave_interp = np.interp(omega_sp, omega, heave_rao, left=0, right=0)
        pitch_interp = np.interp(omega_sp, omega, pitch_rao, left=0, right=0)

        x_eb = config.fixed.electronics_bay[0]
        accel_response = heave_interp + pitch_interp * abs(x_eb)
        accel_squared = (accel_response * omega_sp ** 2) ** 2

        m0 = np.trapezoid(accel_squared * S, omega_sp)
        significant_accel = 4.0 * np.sqrt(max(0, m0))
        peak_accel = 1.86 * significant_accel / g
        # 1.86 converts significant double-amplitude (4σ) to expected
        # peak-to-peak maximum over ~1000 cycles: σ·√(2·ln(1000))/σ·4 * 4σ
        # = √(2·ln(1000))/4 * significant_accel ≈ 0.93 * significant_accel / g
        # for single-amplitude. 1.86 gives peak-to-peak, which is the
        # convention used for structural load qualification in this design.

        return float(peak_accel)
    except Exception:
        return 60.0


def _estimate_roll_period(hydro: dict, x_dict: Optional[dict] = None) -> float:
    BM = hydro.get("BM", 0.5)
    nabla = hydro.get("underwater_volume", hydro.get("nabla", 0.25))
    B = hydro.get("B", 1.0)
    rho = 1025.0
    g = 9.81
    mass = rho * nabla
    if x_dict is not None:
        B = x_dict.get("BWL", B)
    T_canoe = hydro.get("T_canoe", 0.3) or 0.3
    D_keel = hydro.get("D_keel", 0.0)
    T_total = T_canoe + D_keel
    Ixx = mass * (B ** 2 + T_total ** 2) / 12.0

    if x_dict is not None:
        from hull_opt.hydrostatics import compute_cg_z as _cg
        cg_z = _cg(x_dict)
        CB_z = hydro.get("CB_z", -0.08)
        if CB_z > 0.0 or CB_z < -T_total:
            CB_z = -T_total * 0.4
        # Hull-only CB_z is too shallow for deep-keel designs; shift
        # downward by a fraction of keel depth weighted by keel volume
        if D_keel > T_canoe * 0.5:
            keel_vol_ratio = (D_keel * hydro.get("sac_scale_factor", 1.0)) / max(1e-10, T_canoe + D_keel)
            keel_contribution = min(keel_vol_ratio * D_keel * 0.25, D_keel * 0.5)
            CB_z = CB_z - keel_contribution
        GM = BM + CB_z - cg_z
    else:
        GM = BM - 0.05

    if GM <= 0:
        return 0.0
    T_phi = 2 * np.pi * np.sqrt(Ixx / (rho * g * nabla * GM))
    return T_phi
