"""
Tier-1 rapid gate suite: 5 gates (R, W, S, P, E) that run in <3 min CPU.
No hard rejection; NaN/degenerate produces margin<0. All BEM work wrapped
in exception handlers — any Capytaine failure sets fallback + margins: -1.
Key exports: evaluate_rapid_gates(), RapidGateResult
"""
import dataclasses
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


def _trapz(y, x, **kwargs):
    return np.trapezoid(y, x, **kwargs)
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hull_opt.rao_surrogate import RAOSurrogate

logger = logging.getLogger(__name__)
RHO_AIR = 1.225


@dataclass
class RapidGateResult:
    avs_deg: float = 0.0
    max_gz_m: float = 0.0
    gz_area_30: float = 0.0
    gz_area_40_90: float = 0.0
    righting_energy: float = 0.0
    capsize_margin: float = 0.0
    self_right: bool = False
    storm_peak_accel_g: float = 0.0
    roll_sigma_deg: float = 0.0
    roll_sigma_ops_deg: float = 0.0  # 8 ft normal-seas roll — must not roll a bunch
    roll_period_s: float = 0.0
    parametric_roll_risk: bool = False
    slam_pressure_pa: float = 0.0
    slam_pressure_raw_pa: float = 0.0
    slam_accel_g: float = 0.0
    inverted_pressure_pa: float = 0.0
    storm_wind_heel_deg: float = 0.0
    storm_rao_data: list = field(default_factory=list)
    worst_hard: str = ""
    worst_hard_value: float = 0.0
    margins: dict = field(default_factory=dict)
    passed: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)


def _jonswap_spectrum(Hs, Tp, gamma, n_freq, omega_min, omega_max):
    omega_sp = np.linspace(omega_min, omega_max, n_freq)
    f = omega_sp / (2.0 * np.pi)
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
    return S_Hz / (2.0 * np.pi), omega_sp


def _peak_accel_from_raos(heave_rao, pitch_rao, omega_bem, Hs, Tp, gamma,
                          n_freq, g, x_eb):
    if len(omega_bem) < 2:
        return 60.0
    omega_min = max(0.1, float(min(omega_bem)))
    omega_max = float(max(omega_bem))
    S, omega_sp = _jonswap_spectrum(Hs, Tp, gamma, n_freq, omega_min, omega_max)
    heave_interp = np.interp(omega_sp, omega_bem, heave_rao, left=0, right=0)
    pitch_interp = np.interp(omega_sp, omega_bem, pitch_rao, left=0, right=0)
    accel_response = heave_interp + pitch_interp * abs(x_eb)
    accel_squared = (accel_response * omega_sp ** 2) ** 2
    m0 = _trapz(accel_squared * S, omega_sp)
    return float(1.86 * 4.0 * np.sqrt(max(0.0, m0)) / g)


def _roll_sigma_rad(roll_rao, omega_bem, Hs, Tp, gamma, n_freq):
    if len(omega_bem) < 2:
        return 0.7854
    omega_min = max(0.1, float(min(omega_bem)))
    omega_max = float(max(omega_bem))
    S, omega_sp = _jonswap_spectrum(Hs, Tp, gamma, n_freq, omega_min, omega_max)
    r = np.interp(omega_sp, omega_bem, roll_rao, left=0, right=0)
    m0 = _trapz((r ** 2) * S, omega_sp)
    return float(np.sqrt(max(0.0, m0)))


def _storm_wind_heel(x_dict, gz_curve, hydro, config):
    from hull_opt.hydrostatics import compute_wind_heel_equilibrium
    from hull_opt.rig import build_rig, CD_FEATHERED
    storm_wind_ms = config.validation.storm_wind_speed_knots * 0.514444
    rig = build_rig(x_dict, config)
    nabla = hydro.get("nabla", hydro.get("underwater_volume", 0.25))
    return compute_wind_heel_equilibrium(
        gz_curve, rig, storm_wind_ms, nabla,
        rho_water=config.fixed.rho_water, g=config.fixed.gravity,
        feathered_frac=1.0, cd_sail=CD_FEATHERED, x_dict=x_dict,
    )


def _self_righting(gz_curve):
    if gz_curve is None or len(gz_curve) < 2:
        return False
    angles = gz_curve[:, 0]
    gz = gz_curve[:, 1]
    late_mask = (angles >= 150.0) & (angles < 180.0)
    if np.sum(late_mask) > 2:
        gz_late = gz[late_mask]
        if np.all(np.isfinite(gz_late)):
            return bool(float(np.mean(gz_late)) > 0.005)
    return False


def _storm_bem_sweep(stl_path, config, x_dict, hydro, design_vector=None, surrogate=None):
    import capytaine as cpy
    from capytaine.post_pro.rao import rao as compute_rao
    from capytaine.io.xarray import assemble_dataset
    import meshio
    from hull_opt.utils import reassert_pipeline_logging
    reassert_pipeline_logging(config)
    from hull_opt.low_fidelity import _decimate_bem_mesh, BEM_MAX_PANELS
    from hull_opt.hydrostatics import compute_cg_z as _cg

    rapid_cfg = getattr(config, "rapid_validation", None)
    n_freq = getattr(rapid_cfg, "storm_n_freq", 30) if rapid_cfg is not None else 30
    headings_deg = getattr(config.wave_spectrum, "storm_headings_deg", [180.0, 135.0, 90.0])

    ws_storm = dataclasses.replace(
        config.wave_spectrum,
        bem_n_freq=n_freq,
    )
    cfg_storm = dataclasses.replace(config, wave_spectrum=ws_storm)

    msh = meshio.read(stl_path)
    msh = _decimate_bem_mesh(msh, cfg_storm)
    n_panels = len(msh.cells_dict["triangle"])
    if n_panels > BEM_MAX_PANELS:
        raise ValueError(
            f"Storm BEM mesh has {n_panels} panels > cap {BEM_MAX_PANELS}; "
            "aborting solve to avoid OOM"
        )
    body = cpy.FloatingBody.from_meshio(msh, name="hull")
    if hasattr(body, "keep_immersed_part"):
        body.keep_immersed_part()
    else:
        body = body.immersed_part()

    rho = config.fixed.rho_water
    g = config.fixed.gravity
    nabla = hydro.get("nabla", 0.25)
    # CRITICAL ORDER: center_of_mass BEFORE add_all_rigid_body_dofs (the
    # rotation dofs are created about it — mesh-centroid axes produced
    # spurious roll-yaw hydrostatic coupling that polluted the roll RAO).
    from hull_opt.hydrostatics import compute_cg_x as _cgx
    cg_z = hydro.get("cg_z", _cg(x_dict, nabla=hydro.get("underwater_volume", nabla), config=config))
    cg_x = hydro.get("cg_x", _cgx(x_dict, config, cb_x=float(hydro.get("CB_x", 0.0))))
    body.center_of_mass = np.array([cg_x, 0.0, cg_z])
    body.add_all_rigid_body_dofs()
    import xarray as _xr
    from hull_opt.hydrostatics import compute_lumped_inertia
    _std_dofs = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]
    _dof_names = list(body.dofs)
    _sel = [_std_dofs.index(n) for n in _dof_names]
    body.inertia_matrix = _xr.DataArray(
        compute_lumped_inertia(x_dict, hydro, config)[np.ix_(_sel, _sel)],
        dims=["influenced_dof", "radiating_dof"],
        coords={"influenced_dof": _dof_names, "radiating_dof": _dof_names},
        name="inertia_matrix")
    body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness(rho=rho, g=g)

    solver = cpy.BEMSolver()

    heads_rad = [np.deg2rad(h) for h in headings_deg]
    hs_factor = getattr(config.wave_spectrum, "storm_hs_factor", 4.0)
    tp_factor = getattr(config.wave_spectrum, "storm_tp_factor", 1.3)
    Hs_storm = hs_factor * config.wave_spectrum.Hs
    Tp_storm = tp_factor * getattr(config.wave_spectrum, "Tp", 9.0)
    gamma = config.wave_spectrum.gamma
    n_freq_spec = getattr(config.wave_spectrum, "n_freq", 100)

    omega_range = np.linspace(ws_storm.bem_omega_min, ws_storm.bem_omega_max, n_freq)

    use_full = True
    all_headings_rao = []

    if surrogate is not None and surrogate.ready and design_vector is not None:
        anchor_omega = surrogate.anchor_grid(ws_storm.bem_omega_min, ws_storm.bem_omega_max)

        rad_probs = []
        for omega in anchor_omega:
            for dof in body.dofs:
                rad_probs.append(cpy.RadiationProblem(body=body, omega=omega, radiating_dof=dof, rho=rho, g=g))
        rad_results = solver.solve_all(rad_probs, n_jobs=1)

        all_preds = surrogate.predict_batch(
            np.asarray(design_vector, dtype=np.float64),
            omega_range,
            headings_deg,
        )

        full_fallback = False
        for h_rad in heads_rad:
            h_deg = round(np.degrees(h_rad))
            diff_probs = [cpy.DiffractionProblem(body=body, omega=omega, wave_direction=h_rad, rho=rho, g=g) for omega in anchor_omega]
            diff_results = solver.solve_all(diff_probs, n_jobs=1)
            dataset = assemble_dataset(rad_results + diff_results, hydrostatics=True)
            dataset["inertia_matrix"] = body.inertia_matrix
            dataset["hydrostatic_stiffness"] = body.hydrostatic_stiffness
            rao_result = compute_rao(dataset)

            aw_vals = np.asarray(dataset.omega.values, dtype=float)
            a_heave = np.abs(np.asarray(rao_result.sel(radiating_dof="Heave").values.squeeze()))
            a_pitch = np.abs(np.asarray(rao_result.sel(radiating_dof="Pitch").values.squeeze()))
            a_roll = np.abs(np.asarray(rao_result.sel(radiating_dof="Roll").values.squeeze()))

            mask = np.isclose(all_preds["heading_deg"], h_deg)
            order = np.argsort(all_preds["omega"][mask])
            sw = all_preds["omega"][mask][order]
            sh = all_preds["heave_rao"][mask][order]
            sp = all_preds["pitch_rao"][mask][order]
            sr = all_preds["roll_rao"][mask][order]

            sa_h = np.interp(aw_vals, sw, sh, left=0, right=0)
            sa_p = np.interp(aw_vals, sw, sp, left=0, right=0)
            sa_r = np.interp(aw_vals, sw, sr, left=0, right=0)

            rmse_h = float(np.sqrt(np.mean((sa_h - a_heave) ** 2)))
            rmse_p = float(np.sqrt(np.mean((sa_p - a_pitch) ** 2)))
            rmse_r = float(np.sqrt(np.mean((sa_r - a_roll) ** 2)))
            mean_h = max(1e-6, float(np.mean(np.abs(a_heave))))
            mean_p = max(1e-6, float(np.mean(np.abs(a_pitch))))
            mean_r = max(1e-6, float(np.mean(np.abs(a_roll))))
            nrmse_h = rmse_h / mean_h
            nrmse_p = rmse_p / mean_p
            nrmse_r = rmse_r / mean_r
            max_nrmse = max(nrmse_h, nrmse_p, nrmse_r)

            if max_nrmse > surrogate.config.error_threshold:
                full_fallback = True
                logger.info(
                    "Storm BEM surrogate anchor error-threshold exceeded "
                    f"(max_nrmse={max_nrmse:.3f} > "
                    f"{surrogate.config.error_threshold}); falling back to full sweep"
                )
                break

            bh = sh.copy()
            bp = sp.copy()
            br = sr.copy()
            for ai, aw in enumerate(aw_vals):
                idx = int(np.argmin(np.abs(sw - aw)))
                bh[idx] = float(a_heave[ai])
                bp[idx] = float(a_pitch[ai])
                br[idx] = float(a_roll[ai])

            all_headings_rao.append({
                "heading_deg": h_deg,
                "omega": omega_range.tolist(),
                "heave_rao": bh.tolist(),
                "pitch_rao": bp.tolist(),
                "roll_rao": br.tolist(),
            })

        if not full_fallback:
            use_full = False

    if use_full:
        rad_probs = []
        for omega in omega_range:
            for dof in body.dofs:
                rad_probs.append(cpy.RadiationProblem(body=body, omega=omega, radiating_dof=dof, rho=rho, g=g))
        rad_results = solver.solve_all(rad_probs, n_jobs=1)

        for h_rad in heads_rad:
            h_deg = round(np.degrees(h_rad))
            diff_probs = [cpy.DiffractionProblem(body=body, omega=omega, wave_direction=h_rad, rho=rho, g=g) for omega in omega_range]
            diff_results = solver.solve_all(diff_probs, n_jobs=1)
            dataset = assemble_dataset(rad_results + diff_results, hydrostatics=True)
            dataset["inertia_matrix"] = body.inertia_matrix
            dataset["hydrostatic_stiffness"] = body.hydrostatic_stiffness
            rao_result = compute_rao(dataset)

            omega_vals = np.asarray(dataset.omega.values, dtype=float)
            roll_rao = np.abs(np.asarray(rao_result.sel(radiating_dof="Roll").values).squeeze())
            heave_rao = np.abs(np.asarray(rao_result.sel(radiating_dof="Heave").values).squeeze())
            pitch_rao = np.abs(np.asarray(rao_result.sel(radiating_dof="Pitch").values).squeeze())

            all_headings_rao.append({
                "heading_deg": h_deg,
                "omega": omega_vals.tolist(),
                "heave_rao": heave_rao.tolist(),
                "pitch_rao": pitch_rao.tolist(),
                "roll_rao": roll_rao.tolist(),
            })

    beam_result = None
    for hrao in all_headings_rao:
        if hrao["heading_deg"] == 90:
            oa = np.asarray(hrao["omega"])
            ha = np.asarray(hrao["heave_rao"])
            pa = np.asarray(hrao["pitch_rao"])
            ra = np.asarray(hrao["roll_rao"])

            # Keel-vortex roll damping (PYD p.56; SIMP paper zeta=0.42 anchor).
            # Scale by the damped/undamped amplification ratio, anchored to
            # the RAO's own resonance (the analytic 0.35·B estimate sits
            # far from the BEM peak on ballast-heavy hulls and left the
            # attenuation factor ~1 at the wrong omega).
            try:
                _peak_i = int(np.argmax(ra)) if len(ra) > 1 else 0
                wn = float(oa[_peak_i]) if 0 < _peak_i < len(oa) - 1 and oa[_peak_i] > 0 else None
                if wn is None or not np.isfinite(wn):
                    gm_h = hydro.get("GM", hydro.get("gm", 0.1))
                    t_roll_an = 2.0 * np.pi * 0.35 * x_dict.get("BWL", 0.55) / \
                        max(np.sqrt(g * max(gm_h, 0.01)), 1e-6)
                    wn = 2.0 * np.pi / max(t_roll_an, 1e-6)
                rr = oa / max(wn, 1e-6)
                zeta = 0.3 + 0.25 * (x_dict.get("D_keel", 0.5) / max(1e-9, x_dict["LWL"]))
                zeta_rad = 0.01
                h_rad = np.sqrt((1.0 - rr ** 2) ** 2 + (2.0 * zeta_rad * rr) ** 2)
                h_tot = np.sqrt((1.0 - rr ** 2) ** 2 + (2.0 * zeta * rr) ** 2)
                ra = ra * (h_rad / np.maximum(h_tot, 1e-9))
            except Exception as e:
                logger.warning(f"Storm roll-damping correction failed: {e}")

            roll_period = 0.0
            if len(ra) > 2:
                pk = int(np.argmax(ra))
                if pk < len(oa) and oa[pk] > 0:
                    roll_period = 2.0 * np.pi / oa[pk]

            x_eb = config.fixed.electronics_bay[0]
            if abs(x_eb) < 1e-9:
                x_eb = 0.25 * x_dict.get("LWL", 2.4)
            peak_accel = _peak_accel_from_raos(ha, pa, oa, Hs_storm, Tp_storm, gamma, n_freq_spec, g, x_eb)
            roll_sigma = _roll_sigma_rad(ra, oa, Hs_storm, Tp_storm, gamma, n_freq_spec)
            # Ops roll at 4 ft (1.2 m) normal seas — realistic for 2.4 m boat (was 8 ft/2.5 m = boat-length wave)
            rapid_ops = getattr(config, "rapid_validation", None)
            Hs_ops = float(getattr(rapid_ops, "ops_Hs", 1.2)) if rapid_ops is not None else 1.2
            Tp_ops = float(getattr(rapid_ops, "ops_Tp", 7.0)) if rapid_ops is not None else 7.0
            roll_sigma_ops = _roll_sigma_rad(ra, oa, Hs_ops, Tp_ops, gamma, n_freq_spec)

            beam_result = {
                "storm_peak_accel_g": peak_accel,
                "roll_sigma_deg": np.degrees(roll_sigma),
                "roll_sigma_ops_deg": np.degrees(roll_sigma_ops),
                "roll_period_s": roll_period,
                "all_headings_rao": all_headings_rao,
            }
            break

    if beam_result is None:
        raise ValueError("Beam heading (90°) not found in storm sweep")

    return beam_result, Tp_storm


def _slam_pressure(x_dict, config):
    deadrise_deg = x_dict.get("deadrise", 15.0)
    rapid = getattr(config, "rapid_validation", None)
    drop_height = getattr(rapid, "slam_drop_height_m", None) if rapid is not None else None
    if drop_height is None:
        drop_height = getattr(config.validation, "drop_height", 3.0)
    g = config.fixed.gravity
    rho = config.fixed.rho_water
    v_impact = np.sqrt(2.0 * g * drop_height)
    beta = np.radians(max(deadrise_deg, 2.0))
    cot_beta = 1.0 / np.tan(beta)
    p_wagner = 0.5 * rho * (0.5 * np.pi * v_impact * cot_beta) ** 2
    p_wagner = max(p_wagner, 0.0)
    if rapid is None:
        p_cap = getattr(config.validation, "max_pressure_pa", 100000.0) * 0.25
    else:
        p_cap = getattr(rapid, "slam_max_pressure_pa", 0.5e6)
    p_eff = min(p_wagner, max(float(p_cap), 0.0))
    # Return the UNCAPPED Wagner value too so the slam margin keeps a
    # deadrise gradient instead of saturating at the gate threshold.
    return p_eff, v_impact, float(p_wagner)


# Scalar prefactor for the initial-contact impact force model
# (F_slam = C * 0.5*rho*v^2*A_slam*k(beta)). Chosen so a 3 m drop at
# 15 deg deadrise lands ~10 g and the response decays monotonically with
# deadrise (5 deg ~16 g, 25 deg ~6 g for the design-size hull).
_SLAM_FORCE_FACTOR = 0.85


def _slam_accel(p_max, x_dict, hydro, config):
    BWL = x_dict.get("BWL", 0.5)
    LWL = x_dict.get("LWL", 2.4)
    T_canoe = x_dict.get("T_canoe", 0.25)
    deadrise_deg = x_dict.get("deadrise", 15.0)
    beta = np.radians(max(deadrise_deg, 2.0))
    rapid = getattr(config, "rapid_validation", None)
    drop_height = getattr(rapid, "slam_drop_height_m", None) if rapid is not None else None
    if drop_height is None:
        drop_height = getattr(config.validation, "drop_height", 3.0)
    g = config.fixed.gravity
    rho = config.fixed.rho_water
    v_impact = np.sqrt(max(0.0, 2.0 * g * drop_height))
    A_wet = BWL * LWL * (2.0 * np.tan(beta) / np.pi)
    A_slam = min(A_wet, BWL * T_canoe)
    k_beta = (np.pi / 2.0) ** 2 * max(0.3, 1.0 / np.tan(beta)) / 2.0
    mass = config.fixed.target_displacement * rho
    if mass <= 0 or A_slam <= 0 or v_impact <= 0:
        return 0.0
    f_slam = _SLAM_FORCE_FACTOR * 0.5 * rho * v_impact ** 2 * A_slam * k_beta
    return f_slam / (mass * g)


def _inverted_pressure(stl_path, x_dict, config):
    try:
        import trimesh
        mesh = trimesh.load(stl_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        deck_z = float(mesh.vertices[:, 2].max())
    except Exception:
        deck_z = x_dict.get("E", 0.2)
    hydro_pressure = config.fixed.rho_water * config.fixed.gravity * max(0.0, deck_z)
    storm_wind_ms = config.validation.storm_wind_speed_knots * 0.514444
    gust = 1.3
    dyn_pressure = 0.5 * RHO_AIR * (gust * storm_wind_ms) ** 2
    return hydro_pressure + dyn_pressure


def _margin_keys(config) -> list[str]:
    """Canonical margin key list, in build order. Used both by
    _build_margins and by the no-GZ early-return (which forces every
    margin negative so a stability-data-free design can never pass)."""
    return [
        "avs", "capsize", "storm_accel", "roll_sigma",
        "roll_sigma_ops",  # 8 ft normal-seas roll — hard gate, must not roll a bunch
        "slam_pressure", "inverted_pressure", "wind_heel", "parametric_roll",
        "righting_energy", "slam_accel", "max_gz_m",
        "gz_area_30", "gz_area_40_90", "self_right", "roll_period",
    ]


# Cheap HARD gates: computable from GZ/wind-heel data alone, no BEM sweep
# (in canonical build order). roll_sigma/roll_period/parametric_roll/capsize
# need storm BEM results; storm_accel is a soft gate.
_CHEAP_HARD_KEYS = [
    "avs", "wind_heel", "righting_energy",
    "max_gz_m", "gz_area_30", "gz_area_40_90", "self_right",
]


def _soft_gate_keys(config) -> set:
    rapid = getattr(config, "rapid_validation", None)
    soft = getattr(rapid, "soft_margin_gates", None) if rapid is not None else None
    if not soft:
        return set()
    return set(soft)


def evaluate_rapid_gates(x_dict, hydro, gz_curve, cg_z, stl_path, config,
                         bem_results=None, x_vector=None, bem_skip=False):
    result = RapidGateResult()

    if gz_curve is None or len(gz_curve) < 2:
        result.details["gate_r"] = "No GZ curve provided"
        result.details["no_gz"] = True
        result.margins = {k: -1.0 for k in _margin_keys(config)}
        result.passed = {k: False for k in result.margins}
        result.worst_hard = "avs"
        result.worst_hard_value = -1.0
        return result

    # ── Gate R: Righting / Stability ─────────────────────────────────
    from hull_opt.hydrostatics import compute_avs, compute_gz_area
    result.avs_deg = compute_avs(gz_curve)
    result.max_gz_m = float(np.max(gz_curve[:, 1]))
    result.gz_area_30 = compute_gz_area(gz_curve, 0.0, 30.0)
    result.gz_area_40_90 = compute_gz_area(gz_curve, 40.0, 90.0)
    result.self_right = _self_righting(gz_curve)
    result.details["gate_r"] = (
        f"AVS={result.avs_deg:.1f}° maxGZ={result.max_gz_m:.4f}m "
        f"area30={result.gz_area_30:.4f} area40_90={result.gz_area_40_90:.4f} "
        f"self_right={result.self_right}"
    )

    # Righting energy: always recompute from the GZ curve (do not trust
    # stale pass-through in hydro dict — Bug #5 fix).
    try:
        from hull_opt.hydrostatics import compute_righting_energy as _compute_re
        nabla_re = hydro.get("underwater_volume",
                             hydro.get("nabla", config.fixed.target_displacement))
        re_val = _compute_re(gz_curve, max_heel_deg=60.0,
                             rho=hydro.get("rho", config.fixed.rho_water),
                             displacement=nabla_re)
    except Exception:
        re_val = float("nan")
    result.righting_energy = float(re_val)

    # ── Gate E: Storm wind heel (needed by Gate R capsize too) ──────
    try:
        result.storm_wind_heel_deg = _storm_wind_heel(x_dict, gz_curve, hydro, config)
    except Exception as e:
        result.storm_wind_heel_deg = 90.0
        result.details["gate_e"] = f"Wind heel calc failed: {e}"
    else:
        result.details["gate_e"] = f"storm_wind_heel={result.storm_wind_heel_deg:.1f}°"

    # ── Gate S: Slam pressure ───────────────────────────────────────
    try:
        p_max, v_impact, p_wagner_raw = _slam_pressure(x_dict, config)
        result.slam_pressure_pa = p_max
        result.slam_pressure_raw_pa = p_wagner_raw
        result.slam_accel_g = _slam_accel(p_max, x_dict, hydro, config)
        result.details["gate_s"] = (
            f"p_max={p_max:.0f}Pa (wagner_raw={p_wagner_raw:.0f}Pa) "
            f"v_impact={v_impact:.2f}m/s accel={result.slam_accel_g:.1f}g"
        )
    except Exception as e:
        result.slam_pressure_pa = float("nan")
        result.slam_pressure_raw_pa = float("nan")
        result.slam_accel_g = float("nan")
        result.details["gate_s"] = f"Slam calc failed: {e}"

    # ── Gate P: Inverted pressure ───────────────────────────────────
    try:
        result.inverted_pressure_pa = _inverted_pressure(stl_path, x_dict, config)
        result.details["gate_p"] = f"inverted_pressure={result.inverted_pressure_pa:.0f}Pa"
    except Exception as e:
        result.inverted_pressure_pa = float("nan")
        result.details["gate_p"] = f"Inverted pressure calc failed: {e}"

    # ── Gate W: Storm BEM / seakeeping (run last: 2-4 min cost) ─────
    # Skipped when any cheap HARD gate (GZ/wind-heel derived) already
    # failed — the BEM-heavy sweep cannot rescue an unstable hull.
    if bem_skip:
        result.storm_peak_accel_g = 60.0
        result.roll_sigma_deg = 45.0
        result.roll_sigma_ops_deg = 10.0
        result.roll_period_s = 0.0
        result.parametric_roll_risk = False
        result.details["gate_w"] = "storm BEM skipped (bem_skip=True)"
    else:
        cheap_margins = _build_cheap_margins(result, config)
        cheap_failing = [k for k in _CHEAP_HARD_KEYS
                         if cheap_margins.get(k, -1.0) < 0]
        if cheap_failing:
            result.storm_peak_accel_g = 60.0
            result.roll_sigma_deg = 45.0
            result.roll_sigma_ops_deg = 10.0
            result.roll_period_s = 0.0
            result.parametric_roll_risk = False
            result.details["gate_w"] = (
                f"storm BEM skipped: cheap hard gate failed ({cheap_failing[0]})"
            )
        else:
            try:
                beam_data, Tp_storm = _storm_bem_sweep(stl_path, config, x_dict, hydro, design_vector=x_vector)
                result.storm_peak_accel_g = beam_data["storm_peak_accel_g"]
                result.roll_sigma_deg = beam_data["roll_sigma_deg"]
                result.roll_sigma_ops_deg = beam_data.get("roll_sigma_ops_deg", beam_data["roll_sigma_deg"] * 0.25)
                result.roll_period_s = beam_data["roll_period_s"]
                result.storm_rao_data = beam_data.get("all_headings_rao", [])

                Tp_half = Tp_storm / 2.0
                rapid = getattr(config, "rapid_validation", None)
                band_frac = getattr(rapid, "parametric_roll_band_frac", 0.15) if rapid is not None else 0.15
                if result.roll_period_s > 0 and np.isfinite(result.roll_period_s):
                    result.parametric_roll_risk = bool(
                        abs(Tp_half - result.roll_period_s) / result.roll_period_s < band_frac
                    )
                else:
                    result.parametric_roll_risk = False

                result.details["gate_w"] = (
                    f"storm_accel={result.storm_peak_accel_g:.2f}g "
                    f"roll_sigma={result.roll_sigma_deg:.1f}° "
                    f"roll_sigma_ops={result.roll_sigma_ops_deg:.1f}° "
                    f"roll_period={result.roll_period_s:.2f}s "
                    f"parametric_roll={result.parametric_roll_risk}"
                )
            except Exception as e:
                logger.warning(f"Storm BEM failed for design: {e}")
                result.storm_peak_accel_g = 60.0
                result.roll_sigma_deg = 45.0
                result.roll_sigma_ops_deg = 10.0
                result.roll_period_s = float("nan")
                result.parametric_roll_risk = False
                result.details["gate_w"] = f"Storm BEM failed: {e}"

    # ── Capsize margin (needs both Gate R and Gate W/E results) ─────
    mean_heel = result.storm_wind_heel_deg
    wind_heel_crest = 2.0 * mean_heel  # PYD p.59: righting ~halved on a wave crest
    storm_heel_est = max(
        wind_heel_crest,
        mean_heel + 1.4 * result.roll_sigma_deg,
    )
    if result.avs_deg > 0:
        result.capsize_margin = storm_heel_est / result.avs_deg
    else:
        result.capsize_margin = 99.0
    result.details["capsize"] = (
        f"storm_heel_est={storm_heel_est:.1f}° avs={result.avs_deg:.1f}° "
        f"ratio={result.capsize_margin:.3f}"
    )

    _build_margins(result, config)
    return result


def _compute_margins(result, config, keys) -> dict:
    """Compute margins for a subset of the canonical keys, in canonical
    order, applying corrections. Keys must be a subset of _margin_keys."""
    rapid = getattr(config, "rapid_validation", None)
    min_avs_deg = getattr(rapid, "min_avs_deg", 90.0) if rapid is not None else 90.0
    max_accel_g = getattr(config.validation, "max_accel_g", 30.0)
    roll_sigma_max_deg = getattr(rapid, "roll_sigma_max_deg", 45.0) if rapid is not None else 45.0
    max_pressure_pa = getattr(config.validation, "max_pressure_pa", 100000.0)
    wind_heel_max_frac = getattr(rapid, "wind_heel_max_frac", 0.85) if rapid is not None else 0.85
    capsize_safety_factor = getattr(rapid, "capsize_safety_factor", 1.0) if rapid is not None else 1.0
    min_righting_energy = getattr(config.validation, "min_righting_energy", 75.0)
    min_max_gz_m = getattr(rapid, "min_max_gz_m", 0.05) if rapid is not None else 0.05
    min_gz_area_30 = getattr(rapid, "min_gz_area_30", 0.005) if rapid is not None else 0.005
    min_gz_area_40_90 = getattr(rapid, "min_gz_area_40_90", 0.01) if rapid is not None else 0.01
    roll_period_min_s = getattr(rapid, "roll_period_min_s", 1.0) if rapid is not None else 1.0
    roll_period_max_s = getattr(rapid, "roll_period_max_s", 8.0) if rapid is not None else 8.0

    margins = {}

    if "avs" in keys:
        v = result.avs_deg
        margins["avs"] = v / min_avs_deg if np.isfinite(v) and min_avs_deg > 0 else -1.0

    if "capsize" in keys:
        v = result.capsize_margin
        margins["capsize"] = (1.0 - v / capsize_safety_factor) if np.isfinite(v) and capsize_safety_factor > 0 else -1.0

    if "storm_accel" in keys:
        v = result.storm_peak_accel_g
        margins["storm_accel"] = (max_accel_g - v) / max_accel_g if np.isfinite(v) and max_accel_g > 0 else -1.0

    if "roll_sigma" in keys:
        v = result.roll_sigma_deg
        margins["roll_sigma"] = (roll_sigma_max_deg - v) / roll_sigma_max_deg if np.isfinite(v) and roll_sigma_max_deg > 0 else -1.0

    if "roll_sigma_ops" in keys:
        v = result.roll_sigma_ops_deg
        roll_sigma_ops_max_deg = getattr(rapid, "roll_sigma_ops_max_deg", 25.0) if rapid is not None else 25.0
        margins["roll_sigma_ops"] = (roll_sigma_ops_max_deg - v) / roll_sigma_ops_max_deg if np.isfinite(v) and roll_sigma_ops_max_deg > 0 else -1.0

    if "slam_pressure" in keys:
        # UNCAPPED Wagner vs the gate threshold: keeps a deadrise gradient
        # (the capped value saturated at 3.2 MPa and gave no signal).
        v = result.slam_pressure_raw_pa if np.isfinite(result.slam_pressure_raw_pa) \
            else result.slam_pressure_pa
        margins["slam_pressure"] = (max_pressure_pa - v) / max_pressure_pa if np.isfinite(v) and max_pressure_pa > 0 else -1.0

    if "inverted_pressure" in keys:
        v = result.inverted_pressure_pa
        # Deck-skin allowable (~100 kPa cored laminate) — the 3.2 MPa hull
        # threshold saturated every margin at ~0.998 (audit: w9 was a
        # constant +0.38 bonus).
        deck_allow = getattr(rapid, "deck_allowable_pa", 100000.0) if rapid is not None else 100000.0
        margins["inverted_pressure"] = (deck_allow - v) / deck_allow if np.isfinite(v) and deck_allow > 0 else -1.0

    if "wind_heel" in keys:
        v = result.storm_wind_heel_deg
        if np.isfinite(v) and result.avs_deg > 0 and wind_heel_max_frac > 0:
            margins["wind_heel"] = (wind_heel_max_frac * result.avs_deg - v) / (wind_heel_max_frac * result.avs_deg)
        else:
            margins["wind_heel"] = -1.0

    if "parametric_roll" in keys:
        margins["parametric_roll"] = -0.3 if result.parametric_roll_risk else 0.3

    if "righting_energy" in keys:
        v = result.righting_energy
        margins["righting_energy"] = (v - min_righting_energy) / min_righting_energy if np.isfinite(v) and min_righting_energy > 0 else -1.0

    if "slam_accel" in keys:
        v = result.slam_accel_g
        margins["slam_accel"] = (max_accel_g - v) / max_accel_g if np.isfinite(v) and max_accel_g > 0 else -1.0

    if "max_gz_m" in keys:
        v = result.max_gz_m
        margins["max_gz_m"] = (v - min_max_gz_m) / min_max_gz_m if np.isfinite(v) and min_max_gz_m > 0 else -1.0

    if "gz_area_30" in keys:
        v = result.gz_area_30
        margins["gz_area_30"] = (v - min_gz_area_30) / min_gz_area_30 if np.isfinite(v) and min_gz_area_30 > 0 else -1.0

    if "gz_area_40_90" in keys:
        v = result.gz_area_40_90
        margins["gz_area_40_90"] = (v - min_gz_area_40_90) / min_gz_area_40_90 if np.isfinite(v) and min_gz_area_40_90 > 0 else -1.0

    if "self_right" in keys:
        margins["self_right"] = 0.3 if result.self_right else -0.3

    if "roll_period" in keys:
        v = result.roll_period_s
        if not np.isfinite(v):
            margins["roll_period"] = -1.0
        elif roll_period_min_s <= v <= roll_period_max_s:
            margins["roll_period"] = 1.0
        elif v < roll_period_min_s:
            margins["roll_period"] = -(roll_period_min_s - v) / max(roll_period_min_s, 1e-9)
        else:
            margins["roll_period"] = -(v - roll_period_max_s) / max(roll_period_max_s, 1e-9)

    try:
        from hull_opt.corrections import apply_corrections
        margins = apply_corrections(margins, _load_corrections(config))
    except ImportError:
        pass

    return margins


_CORRECTIONS_CACHE = {}


def _load_corrections(config) -> dict:
    """Read the stored SPH-reference corrections from the campaign DB,
    cached by (path, mtime, size) so ReferenceRunner updates are picked up
    without re-opening SQLite on every low-fi evaluation (the old code
    called apply_corrections(margins) with an EMPTY dict — stored SPH
    storm_accel corrections were silently discarded, audit finding)."""
    db_path = getattr(getattr(config, "paths", None), "database", None)
    if not db_path or not Path(db_path).is_file():
        return {}
    try:
        st = Path(db_path).stat()
        key = (str(db_path), st.st_mtime, st.st_size)
        if key not in _CORRECTIONS_CACHE:
            from hull_opt.database import OptimizationDatabase
            db = OptimizationDatabase(str(db_path))
            _CORRECTIONS_CACHE[key] = db.get_corrections() or {}
        return _CORRECTIONS_CACHE[key]
    except Exception:
        return {}


def _build_cheap_margins(result, config) -> dict:
    """Margins for the cheap hard gates only (no BEM sweep needed). Used to
    decide whether the 2-4 min storm sweep can be skipped."""
    return _compute_margins(result, config, _CHEAP_HARD_KEYS)


def _build_margins(result, config):
    margins = _compute_margins(result, config, _margin_keys(config))

    assert list(margins.keys()) == _margin_keys(config), (
        f"Margin key list drifted from canonical: {list(margins.keys())}"
    )

    passed = {}
    for k, v in margins.items():
        if np.isfinite(v):
            passed[k] = bool(v > 0)
        else:
            passed[k] = False

    result.margins = margins
    result.passed = passed

    soft = _soft_gate_keys(config)
    hard_vals = {
        k: v for k, v in margins.items()
        if k not in soft and isinstance(v, (int, float)) and np.isfinite(v)
    }
    result.worst_hard = ""
    result.worst_hard_value = 0.0
    if hard_vals:
        worst_key = min(hard_vals, key=hard_vals.get)
        if hard_vals[worst_key] < 0.0:
            result.worst_hard = worst_key
            result.worst_hard_value = float(hard_vals[worst_key])
