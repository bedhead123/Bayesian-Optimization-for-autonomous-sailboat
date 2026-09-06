"""
Design constraint evaluation for hull feasibility.
Checks B/LWL, Cp, BM, volume error, righting energy, self-righting,
keel aspect ratio, ballast moment, reserve buoyancy, downflooding angle,
wind heeling equilibrium, and wingsail helm balance (CE vs CLR).
Key exports: evaluate_constraints()
"""
import numpy as np
from typing import Optional
from hull_opt.hydrostatics import (
    compute_righting_energy, compute_cg_z, compute_cg_x,
    compute_downflooding_angle, compute_reserve_buoyancy,
    compute_wind_heeling_arm, compute_wind_heel_equilibrium,
)
from hull_opt.rig import build_rig, helm_angle_deg, helm_arm, CD_FEATHERED, \
    extended_keel_clr_x, signed_lead_frac


def evaluate_constraints(hydro: dict, gz_curve: np.ndarray,
                         roll_period: float = 0.0,
                         peak_accel: float = 0.0,
                         x_dict: Optional[dict] = None,
                         config=None,
                         stl_path: Optional[str] = None,
                         hull_stl_path: Optional[str] = None,
                         iteration: Optional[int] = None,
                         balance: Optional[dict] = None) -> tuple[bool, list, dict, float]:
    violation_magnitude = 0.0
    violations = []
    constraints = {}
    reserve_fatal = False  # set True only by the survival-critical reserve-buoyancy check

    relaxation = 0.0
    if iteration is not None:
        relaxation = max(0.0, 1.0 - iteration / 200.0)

    # Blanket NaN/Inf guard: reject any design with non-finite constraint values
    for k, v in hydro.items():
        if isinstance(v, (int, float, np.floating)) and not np.isfinite(v):
            return False, [f"Non-finite hydro value {k}={v}"], {f"{k}_finite": v}, 5.0

    B = hydro.get("B", 0.0)
    LWL = hydro.get("LWL", 2.4)
    Cp = hydro.get("Cp", 0.0)
    nabla = hydro.get("nabla", 0.0)
    BM = hydro.get("BM", 0.0)
    target_nabla = hydro.get("target_nabla", 0.25)
    underwater_vol = hydro.get("underwater_volume", nabla)

    for k, v in [("B", B), ("LWL", LWL), ("Cp", Cp), ("nabla", nabla), ("BM", BM), ("target_nabla", target_nabla), ("underwater_vol", underwater_vol)]:
        if isinstance(v, (int, float, np.floating)) and not np.isfinite(v):
            return False, [f"Non-finite value {k}={v} before constraint eval"], {}, 5.0

    sac_scale = hydro.get("sac_scale_factor", 1.0)
    constraints["B/LWL"] = B / max(1e-10, LWL)
    constraints["B/LWL_scaled"] = constraints["B/LWL"] * sac_scale
    constraints["Cp"] = Cp
    constraints["actual_Cp"] = hydro.get("actual_Cp", Cp)
    constraints["BM"] = BM
    constraints["nabla"] = nabla
    constraints["underwater_volume"] = underwater_vol
    constraints["target_nabla"] = target_nabla

    rho = hydro.get("rho", 1025.0)
    righting_energy = compute_righting_energy(gz_curve, max_heel_deg=60.0,
                                                displacement=underwater_vol, rho=rho)
    constraints["righting_energy"] = righting_energy

    high_angles = gz_curve[:, 0]
    high_gz = gz_curve[:, 1]
    late_mask = (high_angles >= 150.0) & (high_angles < 180.0)
    self_right = False
    mean_gz_high = 0.0
    if np.sum(late_mask) > 2:
        gz_high_valid = high_gz[late_mask]
        if np.all(np.isfinite(gz_high_valid)):
            mean_gz_high = float(np.mean(gz_high_valid))
            self_right = mean_gz_high > 0.005
    # Geometry heuristic fallback: only apply when GZ angular resolution is poor
    if not self_right and x_dict is not None and np.sum(late_mask) <= 2:
        if np.all(np.isfinite(high_gz)):
            T_canoe = x_dict.get("T_canoe", 0.2)
            D_keel = x_dict.get("D_keel", 0.5)
            ballast_frac = x_dict.get("ballast_frac", 0.3)
            min_ballast = 0.3
            if config is not None and hasattr(config, 'validation'):
                min_ballast = config.validation.min_ballast_ratio
            if (D_keel > T_canoe * 3.0 and ballast_frac > min_ballast + 0.1 and D_keel >= 1.0):
                self_right = True
    constraints["self_righting"] = 1.0 if self_right else 0.0

    constraints["roll_period"] = roll_period
    constraints["peak_accel"] = peak_accel

    # Phase 5: B/LWL floating penalty (with relaxation)
    blwl_actual = constraints["B/LWL"]
    blwl_target = 0.30  # was 0.225 for old BWL 0.40-0.60, now BWL 0.60-0.85/LWL2.4 => mean 0.30, update to match bounds
    blwl_range = 0.10   # 0.20-0.40 feasible band, was 0.075
    blwl_dev = abs(blwl_actual - blwl_target) / blwl_range
    blwl_degenerate = 2.0 + relaxation * 2.0  # start at 4.0, tighten to 2.0
    if blwl_dev > blwl_degenerate:
        return False, [f"B/LWL {blwl_actual:.4f} out of control (dev={blwl_dev:.2f})"], constraints, violation_magnitude + 10.0
    elif blwl_dev > 1.0:
        blwl_penalty = 8.0 * (blwl_dev - 1.0) ** 2
        violation_magnitude += blwl_penalty
        violations.append(f"B/LWL: {blwl_actual:.4f} (target {blwl_target:.3f}, penalty {blwl_penalty:.2f})")
        constraints['B/LWL'] = blwl_actual
    elif blwl_dev > 0.8:
        blwl_penalty = 8.0 * 0.05 * (blwl_dev - 0.8) / 0.2
        violation_magnitude += blwl_penalty
        violations.append(f"B/LWL: {blwl_actual:.4f} (near boundary, penalty {blwl_penalty:.2f})")
        constraints['B/LWL'] = blwl_actual
    else:
        constraints['B/LWL'] = blwl_actual

    # B/LWL_scaled: effective beam after volumetric SAC scaling.
    # Previously 0.03-0.75 never bound (typical scaled beam 0.25-0.34
    # passes), so the optimizer could produce sideways-stretched slabs
    # with L/B ~2:1. Tight band 0.20-0.38 steers BO toward boat-like
    # beams; volumetric scaling (y+z via sqrt) already reduced distortion.
    if constraints["B/LWL_scaled"] < (0.20 - relaxation * 0.05):
        lo = 0.20 - relaxation * 0.05
        viol = lo - constraints["B/LWL_scaled"]
        violations.append(f"B/LWL_scaled={constraints['B/LWL_scaled']:.3f} < {lo:.2f}")
        violation_magnitude += viol * 5.0
    elif constraints["B/LWL_scaled"] > (0.38 + relaxation * 0.10):
        max_blwl_scaled = 0.38 + relaxation * 0.10
        viol = constraints["B/LWL_scaled"] - max_blwl_scaled
        violations.append(f"B/LWL_scaled={constraints['B/LWL_scaled']:.3f} > {max_blwl_scaled:.2f}")
        violation_magnitude += viol * 5.0

    # Deck beam constraint: cap the flared deck slab that dominates the
    # visual L/B. Deck half-breadth at midship after SAC scaling should
    # not push deck/L beyond ~0.45.
    try:
        E_val = float(x_dict.get("E", 0.0))
        flare_val = float(x_dict.get("flare", 0.0))
        if E_val > 1e-6 and flare_val > 1e-6:
            from hull_opt.geometry import _topside_wall_angle as _tw, _sheer_height as _sh
            sb = float(x_dict.get("sheer_bow", 0.0)); ss = float(x_dict.get("sheer_stern", 0.0))
            # Estimate deck beam at midship (x_norm≈0.5) where flare is maximal
            y_wl_mid = B / 2.0  # approx midship WL half-beam before scaling
            zsh_mid = _sh(0.5, E_val, sb, ss)
            fl_mid = _tw(0.5, flare_val, B, E_val, y_wl_mid, zsh_mid)
            deck_half = (y_wl_mid + zsh_mid * np.tan(fl_mid)) * sac_scale ** 0.5  # y+z volumetric -> sqrt
            deck_beam_L = (2.0 * deck_half) / max(1e-10, LWL)
            constraints["deck_beam_L"] = deck_beam_L
            # Bow kick raises reserve without tripping midship slab gate: sample bow too (info-only)
            zsh_bow = _sh(0.05, E_val, sb, ss)
            constraints["sheer_bow_m"] = sb
            constraints["sheer_stern_m"] = ss
            constraints["stem_rake_deg"] = float(x_dict.get("stem_rake_deg", 0.0))
            constraints["forefoot_cut"] = float(x_dict.get("forefoot_cut", 0.35))
            constraints["z_sheer_bow_m"] = zsh_bow
            if deck_beam_L > (0.50 + relaxation * 0.10):
                max_deck = 0.50 + relaxation * 0.10
                viol = deck_beam_L - max_deck
                violations.append(f"deck_beam/L {deck_beam_L:.3f} > {max_deck:.2f}")
                violation_magnitude += viol * 3.0
    except Exception:
        pass

    # Beam-to-draft ratio: prevent cartoonishly flat hulls (excessive beam vs depth)
    # Use actual beam after SAC scaling, not the unscaled design-parameter BWL
    T_canoe = hydro.get("T_canoe", 0.3)
    if not np.isfinite(T_canoe) or T_canoe <= 0:
        T_canoe = 0.3
    B_actual = B * sac_scale
    if T_canoe > 0:
        beam_draft = B_actual / T_canoe
        constraints["beam_draft_ratio"] = beam_draft
        max_beam_draft = 4.5 + relaxation * 2.0  # start at 6.5, tighten to 4.5
        if beam_draft > max_beam_draft:
            viol = beam_draft - max_beam_draft
            violations.append(f"B_actual/T_canoe={beam_draft:.2f} > {max_beam_draft:.2f} (hull too flat)")
            violation_magnitude += viol * 0.2

    # Volume mismatch: compare underwater volume against the TRUE target
    # displacement (config.fixed.target_displacement), not the adaptive
    # per-design target (capped SAC scaling). SAC-capped designs must
    # accumulate the volume penalty/violation instead of passing against
    # a shrunken target_eff.
    if config is not None and hasattr(config, 'fixed'):
        true_target = config.fixed.target_displacement
    else:
        true_target = target_nabla
    if true_target is None or not np.isfinite(true_target) or true_target <= 0:
        true_target = underwater_vol
    vol_ratio = abs(underwater_vol - true_target) / max(1e-10, true_target)
    constraints["vol_ratio_error"] = vol_ratio
    # Phase 5: Volume error floating penalty — relaxed for 200kg target (was 0.25, killed 60% LHS)
    vol_degenerate = 0.70 + relaxation * 0.30  # start at 100%, tighten to 70%
    vol_penalty_thresh = 0.40 + relaxation * 0.20  # start at 60%, tighten to 40% (was 25%)
    if vol_ratio > vol_degenerate:
        return False, [f"Volume error {vol_ratio:.2%} > {vol_degenerate:.0%}"], constraints, violation_magnitude + 10.0
    elif vol_ratio > vol_penalty_thresh:
        vol_penalty = 3.0 * ((vol_ratio - vol_penalty_thresh) / max(vol_penalty_thresh, 1e-6)) ** 2
        violation_magnitude += vol_penalty
        violations.append(f"volume_error: {vol_ratio:.2%} (penalty {vol_penalty:.2f})")
        constraints['volume_error'] = vol_ratio
    elif vol_ratio > 0.20:
        vol_penalty = 3.0 * 0.05 * (vol_ratio - 0.20) / 0.05
        violation_magnitude += vol_penalty
        violations.append(f"volume_error: {vol_ratio:.2%} (near boundary, penalty {vol_penalty:.2f})")
        constraints['volume_error'] = vol_ratio
    else:
        constraints['volume_error'] = vol_ratio

    # Minimum displacement: prevent pancake hulls
    if nabla < 0.02:
        viol = 0.02 - nabla
        violations.append(f"hull_volume={nabla:.4f} m³ < 0.02 m³ (minimum displacement)")
        violation_magnitude += viol * 20.0
    if true_target > 0 and nabla < 0.5 * true_target:
        viol = 0.5 - nabla / true_target
        violations.append(f"hull_volume={nabla:.4f} m³ < 50% of target={true_target:.4f} m³")
        violation_magnitude += viol * 2.0

    # SAC scale factor: prevent extreme SAC scaling producing balloon sections.
    # Geometry caps scale at 1.30; keep the constraint aligned with that cap.
    constraints["sac_scale_factor"] = sac_scale
    max_sac = 1.35
    if sac_scale > max_sac:
        viol = sac_scale - max_sac
        violations.append(f"sac_scale_factor={sac_scale:.2f} > {max_sac:.2f}")
        violation_magnitude += viol * 0.5
    if sac_scale < 0.5:
        viol = 0.5 - sac_scale
        violations.append(f"sac_scale_factor={sac_scale:.2f} < 0.5")
        violation_magnitude += viol * 0.5
    sac_scale_std = hydro.get("sac_scale_std", 0.0)
    constraints["sac_scale_std"] = sac_scale_std
    if sac_scale_std > 0.5:
        viol = sac_scale_std - 0.5
        violations.append(f"sac_scale_std={sac_scale_std:.3f} > 0.5 (per-station SAC scaling varies too much)")
        violation_magnitude += viol

    # Phase 5: Cp floating penalty
    cp_check = constraints.get("actual_Cp", constraints["Cp"])
    if config is not None:
        cp_low = config.bounds.Cp[0] * 0.9
        cp_high = config.bounds.Cp[1] * 1.1
    else:
        cp_low = 0.45
        cp_high = 0.65
    if not np.isfinite(cp_check):
        return False, [f"actual_Cp non-finite: {cp_check}"], constraints, violation_magnitude + 10.0
    cp_target = (cp_low + cp_high) / 2.0
    cp_range = (cp_high - cp_low) / 2.0
    cp_dev = abs(cp_check - cp_target) / max(cp_range, 1e-6)
    if cp_dev > 3.0:  # >3x range = degenerate
        return False, [f"actual_Cp {cp_check:.4f} out of control (dev={cp_dev:.2f})"], constraints, violation_magnitude + 10.0
    elif cp_dev > 1.0:
        cp_penalty = 5.0 * (cp_dev - 1.0) ** 2
        violation_magnitude += cp_penalty
        violations.append(f"actual_Cp: {cp_check:.4f} (penalty {cp_penalty:.2f})")
        constraints['actual_Cp'] = cp_check
    elif cp_dev > 0.8:
        cp_penalty = 5.0 * 0.05 * (cp_dev - 0.8) / 0.2
        violation_magnitude += cp_penalty
        violations.append(f"actual_Cp: {cp_check:.4f} (near boundary, penalty {cp_penalty:.2f})")
        constraints['actual_Cp'] = cp_check
    else:
        constraints['actual_Cp'] = cp_check
    # Phase 5: BM floating penalty (with relaxation)
    if BM < 0.005:  # Near-zero BM = degenerate
        return False, [f"BM {BM:.4f}m < 0.005m"], constraints, violation_magnitude + 10.0
    bm_threshold = max(0.03 - relaxation * 0.025, 0.005)  # start at 0.005, ramp to 0.03
    if bm_threshold > 0.005 and BM < bm_threshold:
        bm_penalty = 10.0 * (bm_threshold - BM) / bm_threshold
        violation_magnitude += bm_penalty
        violations.append(f"BM: {BM:.4f}m (penalty {bm_penalty:.2f})")
        constraints['BM'] = BM
    elif BM < 0.04:
        bm_penalty = 10.0 * 0.05 * (0.04 - BM) / 0.01
        violation_magnitude += bm_penalty
        violations.append(f"BM: {BM:.4f}m (near boundary, penalty {bm_penalty:.2f})")
        constraints['BM'] = BM
    else:
        constraints['BM'] = BM
    if config is not None:
        min_re = config.validation.min_righting_energy
        min_re_relaxed = min_re * (1.0 - relaxation * 0.7)  # start at 30% of nominal, ramp up
        if not np.isfinite(constraints["righting_energy"]) or constraints["righting_energy"] < min_re_relaxed:
            viol = max(0, min_re_relaxed - constraints["righting_energy"]) if np.isfinite(constraints["righting_energy"]) else min_re_relaxed
            violations.append(f"righting_energy={constraints['righting_energy']:.1f} J < {min_re_relaxed:.1f} J")
            violation_magnitude += viol / max(min_re_relaxed, 1e-6)
            reserve_fatal = True  # SURVIVAL: righting energy is what recovers the boat after a knockdown
    elif not np.isfinite(constraints["righting_energy"]) or constraints["righting_energy"] <= 0.0:
        violations.append(f"righting_energy={constraints['righting_energy']:.1f} J <= 0 J")
        violation_magnitude += 1.0
        reserve_fatal = True
    if not np.isfinite(constraints["self_righting"]) or constraints["self_righting"] < 0.5:
        violation_magnitude += 1.0 - (constraints["self_righting"] if np.isfinite(constraints["self_righting"]) else 0.0)
        violations.append("inverted stability FAIL")
        reserve_fatal = True  # SURVIVAL: must self-right after inversion
    max_accel_gate = 30.0
    if config is not None and hasattr(config, 'validation'):
        max_accel_gate = getattr(config.validation, 'max_accel_gate', getattr(config.validation, 'max_accel_g', 30.0))
    if not np.isfinite(constraints["peak_accel"]) or constraints["peak_accel"] > max_accel_gate:
        viol = constraints["peak_accel"] - max_accel_gate if np.isfinite(constraints["peak_accel"]) else max_accel_gate
        violations.append(f"peak_accel={constraints['peak_accel']:.1f}g > {max_accel_gate:.0f}g")
        violation_magnitude += viol / max_accel_gate
        reserve_fatal = True  # SURVIVAL: structural survival of wave-slam deceleration

    if roll_period > 0 and (not np.isfinite(roll_period) or roll_period < 0.5 or roll_period > 8.0):
        if np.isfinite(roll_period) and roll_period < 0.5:
            viol = 0.5 - roll_period
        elif np.isfinite(roll_period):
            viol = roll_period - 8.0
        else:
            viol = 0.5
        violations.append(f"roll_period={roll_period:.2f}s not in [0.5, 8.0] s")
        violation_magnitude += viol

    if x_dict is not None:
        D_keel = x_dict.get("D_keel", 0.5)
        keel_chord = x_dict.get("keel_chord", 0.2)
        ballast_frac = x_dict.get("ballast_frac", 0.3)

        # Keel aspect ratio: prevent stubby or absurdly slender keels
        mean_chord = keel_chord * 0.75
        keel_ar = D_keel / max(1e-10, mean_chord)
        constraints["keel_aspect_ratio"] = keel_ar
        ar_upper = (config.bounds.D_keel[1] / (config.bounds.keel_chord[0] * 0.75)
                    if config is not None else 10.67)
        if keel_ar < 2.0:
            viol = 2.0 - keel_ar
            violations.append(f"keel_AR={keel_ar:.3f} < 2.0")
            violation_magnitude += viol * 0.5
        if keel_ar > ar_upper:
            viol = keel_ar - ar_upper
            violations.append(f"keel_AR={keel_ar:.3f} > {ar_upper:.2f}")
            violation_magnitude += viol * 0.5

        # Bug #168: ballast_moment cap DELETED. It was reverse-engineered
        # from "allow 2.2*0.75=1.65" — an admissibility target, not physics.
        # Ballast is genuinely constrained by AVS/RE/GM gates + bulb fit,
        # which all still apply. Record the product as info only.
        constraints["ballast_moment"] = ballast_frac * D_keel

        # Bilge radius vs beam: prevent extreme bilge radius causing bulging sections
        bilge_r = x_dict.get("bilge_r", 0.0)
        if B > 0:
            br_ratio = bilge_r / B
            constraints["bilge_r_BWL_ratio"] = br_ratio
            max_br = 0.5 + relaxation * 0.3  # start at 0.8, tighten to 0.5
            if br_ratio > max_br:
                viol = br_ratio - max_br
                violations.append(f"bilge_r/BWL={br_ratio:.3f} > {max_br:.2f} (causes non-monotonic cross-section)")
                violation_magnitude += viol * 2.0

        constraints["ballast_ratio"] = ballast_frac
        min_ballast = config.validation.min_ballast_ratio if config is not None else 0.25
        if ballast_frac < min_ballast:
            viol = min_ballast - ballast_frac
            violations.append(f"ballast_ratio={ballast_frac:.3f} < {min_ballast:.2f}")
            violation_magnitude += viol * 3.0
        # Bulb capacity: lead must fit in the bulb (PYD ballast-IN-bulb).
        # Demand from TRUE displacement (measured underwater volume, else
        # target_displacement — never the raw box, which overstates mass
        # 2-3x) + rig/payload from config.fixed. Undersized bulbs accumulate
        # violation so new campaigns size the bulb honestly.
        try:
            _uw = hydro.get("underwater_volume", hydro.get("nabla", 0.0))
            if _uw is not None and np.isfinite(_uw) and _uw > 0:
                _tgt = float(_uw)
            elif config is not None:
                _tgt = float(config.fixed.target_displacement)
            else:
                _tgt = 0.10
            _mast = 10.0 if config is None else float(
                config.fixed.wingsail_mast_mass + config.fixed.wingsail_nose_pod_mass)
            _pay = 15.0 if config is None else float(config.fixed.payload_mass_kg)
            _cap_base = _tgt * 1025.0 + _mast + _pay
            _need = ballast_frac * max(_cap_base, 40.0)
            _have = float(x_dict.get("bulb_vol", 0.0)) * 11340.0
            constraints["bulb_capacity_kg"] = _have
            constraints["ballast_need_kg"] = _need
            if _need > _have + 1e-9:
                violations.append(f"bulb undersized: need {_need:.1f}kg > cap {_have:.1f}kg")
                violation_magnitude += (_need - _have) / max(10.0, _have) * 2.0
        except Exception:
            pass

    if x_dict is not None and config is not None and (stl_path is not None or hull_stl_path is not None):
        reserve = compute_reserve_buoyancy(stl_path, x_dict)
        constraints["reserve_buoyancy"] = reserve
        min_reserve = config.validation.min_reserve_buoyancy
        if reserve < min_reserve:
            viol = min_reserve - reserve
            violations.append(f"reserve_buoyancy={reserve:.3f} < {min_reserve:.2f}")
            violation_magnitude += viol * 3.0
            # SURVIVAL-CRITICAL: reserve buoyancy is the energy that rightes
            # the boat after a knockdown — below minimum is not survivable.
            # Bug #171: latch ONLY — never clear. An earlier `else:
            # reserve_fatal = False` wiped fatals set by righting-energy /
            # self-right / accel above, marking unsurvivable designs feasible.
            reserve_fatal = True

        # Downflooding: vessel is fully enclosed/watertight - skip this check
        # df_angle is still computed for info but never triggers a violation
        try:
            df_angle = compute_downflooding_angle(hull_stl_path or stl_path, cg_z=compute_cg_z(x_dict, nabla=nabla), x_dict=x_dict)
        except Exception:
            df_angle = 180.0
        constraints["downflooding_angle"] = df_angle
        # No violation enforced (watertight vessel)

        storm_wind_ms = config.validation.storm_wind_speed_knots * 0.514444
        rig = build_rig(x_dict, config)
        eq_heel = compute_wind_heel_equilibrium(
            gz_curve, rig, storm_wind_ms, nabla,
            rho_water=rho, g=config.fixed.gravity, feathered_frac=1.0,
            cd_sail=CD_FEATHERED, x_dict=x_dict,
        )
        constraints["eq_heel_feathered_deg"] = eq_heel
        max_eq_heel = 45.0 + relaxation * 30.0  # start at 75°, tighten to 45°
        if eq_heel > max_eq_heel:
            viol = eq_heel - max_eq_heel
            violations.append(f"equilibrium heel={eq_heel:.1f}° > {max_eq_heel:.0f}° with feathered wings")
            violation_magnitude += viol / max(45.0, 1e-6)

        # ── Draft: info + soft T/L shaping + priced inconvenience ──
        # T/L>0.30 accumulates soft violation (JFR band label; slope is a
        # judgment, stated). Bug #169 (user directive): draft is an
        # INCONVENIENCE WITH A PRICE, not a wall — benefits can outweigh
        # handling cost, so over-deep is never infeasible for logistics.
        # The price lives in FoM (draft_logistics_cost, low_fidelity.py).
        # Only hard draft failure left: T_total > LWL (physical RealityCheck).
        t_total = x_dict["T_canoe"] + x_dict["D_keel"]
        t_over_l = t_total / max(1e-9, LWL)
        constraints["T_over_L"] = t_over_l
        constraints["T_total_m"] = t_total
        if t_over_l > 0.30:
            ex = t_over_l - 0.30
            violations.append(f"T/L={t_over_l:.2f} > 0.30 (draft {t_total:.2f} m — launch/debris risk)")
            violation_magnitude += 8.0 * ex
        if t_total > LWL + 1e-9:
            return False, [f"total draft={t_total:.2f} m > LWL={LWL:.2f} m (physical cap)"], \
                constraints, violation_magnitude + 4.0

        # ── JFR envelope monitors (info-only: LDR, L/B, SA/D, Wb/DT) ──
        try:
            nabla_info = hydro.get("underwater_volume", hydro.get("nabla", 0.0))
            if nabla_info and np.isfinite(nabla_info) and nabla_info > 0:
                constraints["LDR"] = LWL / (nabla_info ** (1.0 / 3.0))
                B_ac = B * sac_scale
                constraints["L_over_B"] = LWL / max(1e-9, B_ac)
                try:
                    constraints["SA_over_D"] = rig["area"] / max(1e-9, (nabla_info ** (2.0 / 3.0)))
                except Exception:
                    pass
                try:
                    tot_m = nabla_info * 1025.0 + float(getattr(config.fixed, "payload_mass_kg", 0.0)) \
                        + float(config.fixed.wingsail_mast_mass + config.fixed.wingsail_nose_pod_mass)
                    constraints["Wb_over_DT"] = (hydro.get("bulb_mass_kg", 0.0) + hydro.get("ballast_mass_kg", 0.0)) / max(1e-9, tot_m)
                except Exception:
                    pass
        except Exception:
            pass

        # ── Wingsail helm balance: SIGNED lead hard gate ───────────────
        # PYD Ch.9: hydrodynamic CLR on the extended keel (25%-chord line at
        # 45% draft, sweep-corrected); lead = CE_x - CLR_x must be POSITIVE
        # (weather helm) — negative lead (lee helm) is hard-infeasible: the
        # boat cannot hold a course without a rudder (forum #15; PYD p.201
        # gust-luffing safety). Recommended fin-keel band 2-6% LWL (p.205).
        clr_x = extended_keel_clr_x(x_dict)
        lead_frac = signed_lead_frac(rig["combined"]["x"], clr_x, LWL)
        arm = helm_arm(rig["combined"]["z"], x_dict)
        helm_cmb = helm_angle_deg(rig["combined"]["x"], clr_x, arm)  # report-only vertical proxy
        constraints["ce_x"] = rig["combined"]["x"]
        constraints["ce_z"] = rig["combined"]["z"]
        constraints["clr_x"] = clr_x
        constraints["clr_method"] = "extended_keel_45pct_draft"
        constraints["lead_pct_lwl"] = lead_frac * 100.0
        constraints["helm_fwd_deg"] = helm_cmb
        constraints["helm_aft_deg"] = helm_cmb
        constraints["helm_combined_deg"] = helm_cmb
        if lead_frac < -0.005:
            return False, [f"lee helm: lead={lead_frac*100:.2f}% LWL (CE fwd of CLR)"], \
                constraints, violation_magnitude + 4.0
        if lead_frac > 0.085:
            ex = lead_frac - 0.085
            violations.append(f"weather helm {lead_frac*100:.2f}% LWL > 8.5% (rec. 2-6%)")
            violation_magnitude += 20.0 * ex

    # ── Report metrics (informational — no constraint) ─────────────────
    cg_z_hydro = hydro.get("cg_z", None)
    if cg_z_hydro is not None and np.isfinite(cg_z_hydro):
        constraints["cg_z"] = cg_z_hydro
        CB_z = hydro.get("CB_z", 0.0)
        if np.isfinite(CB_z):
            constraints["gm"] = BM + CB_z - cg_z_hydro
    if x_dict is not None and config is not None:
        try:
            cb_x = hydro.get("CB_x", None)
            cg_x = compute_cg_x(x_dict, config, cb_x=cb_x if cb_x is not None else 0.0)
            constraints["cg_x"] = cg_x
            clr_arm = helm_arm(
                rig["combined"]["z"] if 'rig' in locals() and rig else 1.5, x_dict)
            constraints["trim_deg"] = float(
                np.degrees(np.arctan2(abs(cg_x - cb_x) if cb_x is not None else 0.0,
                                      max(clr_arm, 1e-9))))
        except Exception:
            pass
    # System-level balance metrics (informational; hard gate handled in low_fidelity)
    if balance is not None:
        try:
            constraints["balance_worst_heel_ops"] = balance.get("worst_heel_ops_deg")
            constraints["balance_worst_heel_storm"] = balance.get("worst_heel_storm_deg")
            constraints["balance_worst_leeway_ops"] = balance.get("worst_leeway_ops_deg")
            constraints["balance_mean_drive"] = balance.get("mean_drive_ops_N")
            constraints["balance_vmg_up"] = balance.get("vmg_up_N")
            constraints["balance_n_ops_fail"] = balance.get("n_ops_fail")
            constraints["balance_n_storm_fail"] = balance.get("n_storm_fail")
            constraints["balance_heel_margin_storm"] = balance.get("heel_margin_storm")
        except Exception:
            pass

    # Feasibility = no FATAL violation. Soft penalties (sac_scale_std,
    # actual_Cp/volume near-boundary, ballast/BM ramps, soft storm gates)
    # are reported and FoM-shaped, not feasibility-revoking: marking every
    # penalized design infeasible gave the optimizer zero gradient (0/68
    # feasible in the 2026-08-24/25 campaigns). Hard storm-gate failures
    # still revoke feasibility in low_fidelity via the rapid margins.
    feasible = not reserve_fatal
    return feasible, violations, constraints, float(violation_magnitude)
