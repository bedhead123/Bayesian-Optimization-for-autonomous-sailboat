"""
Coupled aero-hydro equilibrium solver for 360° weathervaning wingsail.

Evaluates the full system together (not isolated rig or hull):

  Aero side/heeling  <->  Hydro side (keel lift @ leeway)
  Heeling moment     <->  Righting moment (GZ curve)

For each true-wind angle around the compass the solver finds the
heel + leeway where side forces and heeling/righting moments balance.
If no balance exists before AVS / max heel the point is infeasible.

Key exports: evaluate_balance_polar(), hydro_sideforce()

Hydro model is analytic (lifting-line keel + hull side area), intentionally
cheap (~0.5 ms per point) so the BO loop can call it for every design.
"""

import numpy as np
from typing import Optional

from hull_opt.rig import compute_aero_forces_360


def hydro_sideforce(leeway_deg: float, boat_speed_ms: float, x_dict: dict,
                    rho_water: float = 1025.0) -> float:
    """Side force from keel + hull at a given leeway (deg).

    Lifting-line: CL = 2π·AR/(AR+2) · sin(leeway), capped at stall ~12°.
    Keel area = D_keel · mean_chord (trapezoidal taper factor 0.75).
    Hull side area adds ~15% of LWL·T for low-aspect hull lift (small slope).
    """
    leeway = np.radians(float(leeway_deg))
    if abs(leeway) < 1e-6 or boat_speed_ms < 0.1:
        return 0.0

    D_keel = float(x_dict.get("D_keel", 1.0))
    keel_chord = float(x_dict.get("keel_chord", 0.2))
    LWL = float(x_dict.get("LWL", 2.4))
    T_canoe = float(x_dict.get("T_canoe", 0.25))

    mean_chord = max(1e-9, keel_chord * 0.75)
    A_keel = max(1e-9, D_keel * mean_chord)
    # Hull-attached keel: effective AR doubles via the hull-bottom mirror
    # (PYD Ch.6/Fig 6.5) — the old geometric-only AR under-predicted lift.
    AR_keel = 2.0 * D_keel / max(1e-9, mean_chord)
    # Lifting-line slope
    slope_keel = 2.0 * np.pi * AR_keel / max(1e-9, AR_keel + 2.0)
    # Stall model: linear to 12°, then flat
    leeway_eff = np.clip(leeway, -np.radians(12.0), np.radians(12.0))
    # Beyond stall add little extra
    if abs(leeway) > np.radians(12.0):
        extra = 0.15 * np.sin(abs(leeway) - np.radians(12.0)) * np.sign(leeway)
        leeway_eff = leeway_eff + extra
    CL_keel = slope_keel * leeway_eff
    CL_keel = float(np.clip(CL_keel, -1.2, 1.2))

    q = 0.5 * rho_water * boat_speed_ms * boat_speed_ms
    F_keel = q * A_keel * CL_keel

    # Hull side contribution (low AR, small slope)
    A_hull = 0.15 * LWL * T_canoe
    slope_hull = 0.8  # very low aspect hull
    CL_hull = slope_hull * np.sin(leeway)  # small
    F_hull = q * A_hull * CL_hull

    return float(F_keel + F_hull)


def _interp_gz(gz_curve: np.ndarray, heel_deg: float) -> float:
    """Linear interpolation of GZ at heel_deg."""
    if gz_curve is None or len(gz_curve) < 2:
        return 0.0
    return float(np.interp(heel_deg, gz_curve[:, 0], gz_curve[:, 1],
                           left=0.0, right=gz_curve[-1, 1]))


def solve_one_point(x_dict: dict, config, gz_curve: np.ndarray,
                    disp_m3: float, TWA_deg: float,
                    TWS_ms: float, boat_speed_ms: float,
                    AVS_deg: float, feathered: bool = False,
                    rho_water: float = 1025.0, g: float = 9.81) -> dict:
    """Find heel/leeway equilibrium for one wind point.

    Brute sweep over heel [0, AVS) and leeway [0, 15°] minimizing:
      err_side = |aero_side - hydro_side| / max(1, |aero_side|)
      err_heel = |heeling - righting| / max(1, righting_at_30°)

    The heel side dominates feasibility; leeway is chosen to match side force
    at that heel. Returns dict with equil or infeasible flag.
    """
    # GZ-derived righting scale for normalization
    gz30 = abs(_interp_gz(gz_curve, 30.0))
    righting_scale = max(1.0, rho_water * g * disp_m3 * max(0.01, gz30))

    # Heel candidates up to min(AVS-2, 70°) – need margin before capsize
    heel_max = min(float(AVS_deg) - 2.0, 70.0) if np.isfinite(AVS_deg) and AVS_deg > 10 else 60.0
    if heel_max < 5.0:
        heel_max = 5.0
    heels = np.linspace(0.0, heel_max, 31)
    leeways = np.linspace(0.0, 15.0, 61)  # 0.25° resolution for accurate side match

    best = None
    best_err = 1e12

    # Precompute aero at each heel (side/moment scale with cos(heel))
    for heel in heels:
        aero = compute_aero_forces_360(
            x_dict, config, TWS_ms, TWA_deg, boat_speed_ms, heel_deg=heel,
            feathered=feathered
        )
        aero_side = abs(float(aero["side_N"]))
        heeling = float(aero["heeling_moment_Nm"])
        gz = _interp_gz(gz_curve, heel)
        righting = rho_water * g * disp_m3 * max(0.0, gz)

        # If heel beyond AVS, righting negative -> infeasible region
        if gz < 0 and heel > 5:
            err_heel = 2.0  # large penalty
        else:
            err_heel = abs(heeling - righting) / righting_scale

        # Find leeway that best matches aero_side at this heel
        best_leeway = 0.0
        best_side_err = 1e12
        for lw in leeways:
            hyd = abs(hydro_sideforce(lw, boat_speed_ms, x_dict, rho_water))
            # For zero aero (head to wind feathered), any leeway far is bad; keep small
            if aero_side < 1.0:
                side_err = hyd / (righting_scale) * 0.001 + lw * 0.01
            else:
                side_err = abs(hyd - aero_side) / max(1.0, aero_side)
            if side_err < best_side_err:
                best_side_err = side_err
                best_leeway = lw

        err = err_heel + 0.5 * best_side_err

        # Heel feasibility: must have positive righting margin
        # Prefer solutions with small heel (comfort/VMG)
        err += heel * 0.001

        if err < best_err:
            best_err = err
            best = {
                "heel_deg": float(heel),
                "leeway_deg": float(best_leeway),
                "aero_side_N": float(aero_side),
                "hydro_side_N": float(abs(hydro_sideforce(best_leeway, boat_speed_ms, x_dict, rho_water))),
                "heeling_Nm": heeling,
                "righting_Nm": float(righting),
                "drive_N": float(aero["drive_N"]),
                "side_N_raw": float(aero["side_N"]),
                "AWS_ms": float(aero["AWS_ms"]),
                "AWA_deg": float(aero["AWA_deg"]),
                "err_heel": float(err_heel),
                "err_side": float(best_side_err),
                "err_total": float(err),
                "feathered": feathered,
            }

    if best is None:
        return {"feasible": False, "reason": "no sweep candidate"}

    # Feasibility criteria – relaxed to 0.6 to account for analytic model
    # uncertainty and discretization; hydro side is approximate lifting-line.
    heel_ok = best["heel_deg"] < (AVS_deg * 0.85) if np.isfinite(AVS_deg) and AVS_deg > 0 else best["heel_deg"] < 45.0
    leeway_ok = best["leeway_deg"] < 12.0
    side_match = best["err_side"] < 0.25
    heel_match = best["err_heel"] < 0.60

    feasible = heel_ok and leeway_ok and side_match and heel_match
    # In feathered storm case we allow larger heel_match tolerance (righting vs small heeling)
    if feathered and heel_ok and leeway_ok and best["err_side"] < 0.30:
        feasible = True

    best["feasible"] = bool(feasible)
    best["heel_ok"] = bool(heel_ok)
    best["leeway_ok"] = bool(leeway_ok)
    best["side_match"] = bool(side_match)
    best["heel_match"] = bool(heel_match)
    best["TWA_deg"] = float(TWA_deg)
    best["TWS_ms"] = float(TWS_ms)
    return best


def evaluate_balance_polar(x_dict: dict, config, gz_curve: np.ndarray,
                           hydro: dict, boat_speed_ms: Optional[float] = None,
                           tws_ops_kt: float = 10.0,
                           tws_storm_kt: Optional[float] = None) -> dict:
    """System-level 360° balance across compass points.

    Evaluates 8 TWA points at operational wind and at storm feathered wind.
    Returns summary dict with worst-case heel/leeway, mean drive/VMG, per-point
    results, and overall feasible flag. Cheap (<5 ms).

    Bug #169: tws_ops_kt parametrized so the mission FoM can score light /
    medium / heavy bands through the same solver. Defaults reproduce the
    legacy 10 kt ops call exactly (gates + DB read these keys).
    """
    from hull_opt.hydrostatics import compute_avs

    LWL = float(x_dict.get("LWL", 2.4))
    disp = float(hydro.get("underwater_volume",
                           hydro.get("nabla", config.fixed.target_displacement)))
    if not np.isfinite(disp) or disp <= 0:
        disp = float(config.fixed.target_displacement)

    if boat_speed_ms is None:
        boat_speed_ms = float(config.fixed.target_speed_knots * 0.514444)

    AVS = compute_avs(gz_curve) if gz_curve is not None else 90.0
    if not np.isfinite(AVS) or AVS <= 0:
        AVS = 90.0

    # Operational wind (default 10 kt typical sailing) and storm feathered wind.
    # Bug #169: ops TWS is a parameter (mission bands); storm defaults to config.
    TWS_ops = float(tws_ops_kt) * 0.514444
    if tws_storm_kt is None:
        TWS_storm = float(config.validation.storm_wind_speed_knots * 0.514444)
    else:
        TWS_storm = float(tws_storm_kt) * 0.514444

    TWAs = [0, 45, 90, 135, 180, 225, 270, 315]  # full circle; symmetry will duplicate but checks both tacks
    # Unique for scoring: 0,45,90,135,180 (others mirror)
    TWAs_ops = [0, 45, 90, 135, 180]
    TWAs_storm = [0, 45, 90, 135, 180]

    ops_points = []
    for twa in TWAs_ops:
        res = solve_one_point(x_dict, config, gz_curve, disp, twa, TWS_ops,
                              boat_speed_ms, AVS, feathered=False)
        ops_points.append(res)

    storm_points = []
    for twa in TWAs_storm:
        res = solve_one_point(x_dict, config, gz_curve, disp, twa, TWS_storm,
                              boat_speed_ms, AVS, feathered=True)
        storm_points.append(res)

    # Summary metrics
    def _worst(points, key):
        vals = [p.get(key, 0) for p in points if np.isfinite(p.get(key, 0))]
        return float(max(vals)) if vals else 0.0

    def _mean_drive(points):
        vals = [p.get("drive_N", 0) for p in points if p.get("feasible", False) and p.get("drive_N", 0) > 0]
        return float(np.mean(vals)) if vals else 0.0

    def _reach_drive(points):
        # Bug #169: ocean-crossing mission lives on reach/run (TWA 90/135/180).
        # Mean drive over feasible reach/run points; upwind (45) priced via VMG.
        vals = [p.get("drive_N", 0) for p in points
                if p.get("feasible", False) and p.get("TWA_deg", 0) in (90, 135, 180)
                and p.get("drive_N", 0) > 0]
        return float(np.mean(vals)) if vals else 0.0

    # VMG: drive component projected onto course? Approx VMG = boat_speed * cos(TWA) if drive>0,
    # but with fixed boat speed we use drive * cos(TWA) as proxy: upwind VMG uses 45° point
    upwind_drive = next((p["drive_N"] for p in ops_points if p["TWA_deg"] == 45 and p["feasible"]), 0.0)
    downwind_drive = next((p["drive_N"] for p in ops_points if p["TWA_deg"] == 180 and p["feasible"]), 0.0)
    vmg_up = upwind_drive * np.cos(np.radians(45)) if upwind_drive > 0 else -abs(upwind_drive) * 0.1
    vmg_down = downwind_drive  # running is already downwind

    worst_heel_ops = _worst(ops_points, "heel_deg")
    worst_leeway_ops = _worst(ops_points, "leeway_deg")
    worst_heel_storm = _worst(storm_points, "heel_deg")
    worst_leeway_storm = _worst(storm_points, "leeway_deg")

    # Helm: compute yaw moment worst across ops points
    yaw_vals = [abs(p.get("side_N_raw", 0) * (p.get("leeway_deg", 0))) for p in ops_points]
    # Better: use stored aero yaw proxy: side * arm_long
    worst_yaw = 0.0
    for p in ops_points:
        if not p.get("feasible"):
            continue
        aero = compute_aero_forces_360(x_dict, config, TWS_ops, p["TWA_deg"], boat_speed_ms, p["heel_deg"], False)
        worst_yaw = max(worst_yaw, abs(aero.get("yaw_moment_Nm", 0)))

    # Overall feasibility: all ops storm points must have heel < AVS*0.85 and leeway <12°
    all_feasible = all(p.get("feasible", False) for p in ops_points) and all(p.get("feasible", False) for p in storm_points)
    # Soft: if >1 point fails, overall fails; single marginal failure is reported but not fatal if storm ok
    n_ops_fail = sum(1 for p in ops_points if not p.get("feasible", False))
    n_storm_fail = sum(1 for p in storm_points if not p.get("feasible", False))

    # Allow 1 ops failure as soft (e.g. head to wind drive small)
    system_feasible = (n_storm_fail == 0) and (n_ops_fail <= 1) and (worst_heel_storm < AVS * 0.85)

    return {
        "AVS_deg": float(AVS),
        "TWS_ops_ms": float(TWS_ops),
        "TWS_storm_ms": float(TWS_storm),
        "boat_speed_ms": float(boat_speed_ms),
        "ops_points": ops_points,
        "storm_points": storm_points,
        "worst_heel_ops_deg": float(worst_heel_ops),
        "worst_leeway_ops_deg": float(worst_leeway_ops),
        "worst_heel_storm_deg": float(worst_heel_storm),
        "worst_leeway_storm_deg": float(worst_leeway_storm),
        "mean_drive_ops_N": float(_mean_drive(ops_points)),
        "reach_drive_ops_N": float(_reach_drive(ops_points)),
        "vmg_up_N": float(vmg_up),
        "vmg_down_N": float(vmg_down),
        "worst_yaw_Nm": float(worst_yaw),
        "n_ops_fail": int(n_ops_fail),
        "n_storm_fail": int(n_storm_fail),
        "system_feasible": bool(system_feasible),
        "heel_margin_ops": float((AVS * 0.85 - worst_heel_ops) / max(1.0, AVS * 0.85)) if AVS > 0 else -1.0,
        "heel_margin_storm": float((AVS * 0.85 - worst_heel_storm) / max(1.0, AVS * 0.85)) if AVS > 0 else -1.0,
    }
