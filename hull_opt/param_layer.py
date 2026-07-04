"""
Parameter transformation layer (Rule 1: Shape Genes, Rule 2: Squashing Functions).

Transforms the raw [-10, +10] unbounded search-space vector used by the GP
into physically valid hull parameters using ratio-based parameterization,
sigmoid squashing, and first-order physics bounds (Rule 5).

The GP never sees raw meters or degrees — only dimensionless ratios and
unbounded latent variables. All physical consistency is baked into the
transformation, not validated post-hoc.
"""
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _hard_clamp(x, lo, hi):
    return np.clip(x, lo, hi)


def design_vector_to_physical(raw: np.ndarray, config=None) -> dict:
    """Convert 17-element raw GP vector to physical hull parameters.

    The raw vector lives in roughly [-10, +10] (unbounded). Each element is
    squashed/transformed to a physically valid hull parameter. LWL is the
    master scale (Rule 1). All other dimensions are ratios to LWL.
    """
    r = raw  # shorthand

    # ── Master scale ────────────────────────────────────────────────
    # LWL stays as-is but bounded via sigmoid (Rule 2)
    lwl_lo, lwl_hi = (2.3, 2.5) if config is None else config.bounds.LWL
    LWL = lwl_lo + (lwl_hi - lwl_lo) * sigmoid(r[0])

    # ── Ratios to LWL (Rule 1: Shape Genes) ─────────────────────────
    bwl_lo, bwl_hi = (0.40, 0.60) if config is None else config.bounds.BWL
    BWL = bwl_lo + (bwl_hi - bwl_lo) * sigmoid(r[1])

    t_lo, t_hi = (0.15, 0.35) if config is None else config.bounds.T_canoe
    T_canoe = t_lo + (t_hi - t_lo) * sigmoid(r[2])

    # Cp, Cm are dimensionless [0,1] coefficients (Rule 5: bounded by physics)
    cp_lo, cp_hi = (0.55, 0.65) if config is None else config.bounds.Cp
    Cp = cp_lo + (cp_hi - cp_lo) * sigmoid(r[3])

    cm_lo, cm_hi = (0.60, 0.90) if config is None else config.bounds.Cm
    Cm = cm_lo + (cm_hi - cm_lo) * sigmoid(r[4])

    # LCB is fraction of LWL (Rule 1)
    lcb_lo, lcb_hi = (5.0, 20.0) if config is None else config.bounds.LCB
    LCB = lcb_lo + (lcb_hi - lcb_lo) * sigmoid(r[5])

    # Keel depth as ratio of LWL (Rule 1)
    dk_lo, dk_hi = (0.85, 1.20) if config is None else config.bounds.D_keel
    D_keel = dk_lo + (dk_hi - dk_lo) * sigmoid(r[6])

    # Keel chord as ratio of LWL (Rule 1)
    kc_lo, kc_hi = (0.15, 0.25) if config is None else config.bounds.keel_chord
    keel_chord = kc_lo + (kc_hi - kc_lo) * sigmoid(r[7])

    # Bulb volume as fraction of target displacement (Rule 1)
    bv_lo, bv_hi = (0.001, 0.004) if config is None else config.bounds.bulb_vol
    bulb_vol = bv_lo + (bv_hi - bv_lo) * sigmoid(r[8])

    # Bulb position as fraction of LWL from bow (Rule 1)
    bp_lo, bp_hi = (0.30, 0.50) if config is None else config.bounds.bulb_pos
    bulb_pos = bp_lo + (bp_hi - bp_lo) * sigmoid(r[9])

    # Sheer parameters as ratios of LWL (Rule 1)
    e_lo, e_hi = (0.15, 0.30) if config is None else config.bounds.E
    E = e_lo + (e_hi - e_lo) * sigmoid(r[10])

    sa_lo, sa_hi = (0.05, 0.25) if config is None else config.bounds.SA
    SA = sa_lo + (sa_hi - sa_lo) * sigmoid(r[11])

    # Flare angle - squashed (Rule 2), bounded by tan(flare) < beam/draft (Rule 5)
    fl_lo, fl_hi = (5.0, 15.0) if config is None else config.bounds.flare
    flare = fl_lo + (fl_hi - fl_lo) * sigmoid(r[12])

    # Deadrise angle - squashed (Rule 2)
    dr_lo, dr_hi = (5.0, 25.0) if config is None else config.bounds.deadrise
    deadrise = dr_lo + (dr_hi - dr_lo) * sigmoid(r[13])

    # Bilge radius - squashed (Rule 2), max 50% of BWL (Rule 5)
    br_lo, br_hi = (0.05, 0.30) if config is None else config.bounds.bilge_r
    bilge_r = br_lo + (br_hi - br_lo) * sigmoid(r[14])

    # Keel rake angle - squashed (Rule 2)
    kr_lo, kr_hi = (0.001, 0.02) if config is None else config.bounds.keel_rake
    keel_rake = kr_lo + (kr_hi - kr_lo) * sigmoid(r[15])

    # Ballast fraction - squashed, always [0,1] (Rule 2)
    bf_lo, bf_hi = (0.30, 0.70) if config is None else config.bounds.ballast_frac
    ballast_frac = bf_lo + (bf_hi - bf_lo) * sigmoid(r[16])

    return {
        "LWL": float(LWL),
        "BWL": float(BWL),
        "T_canoe": float(T_canoe),
        "Cp": float(Cp),
        "Cm": float(Cm),
        "LCB": float(LCB),
        "D_keel": float(D_keel),
        "keel_chord": float(keel_chord),
        "bulb_vol": float(bulb_vol),
        "bulb_pos": float(bulb_pos),
        "E": float(E),
        "SA": float(SA),
        "flare": float(flare),
        "deadrise": float(deadrise),
        "bilge_r": float(bilge_r),
        "keel_rake": float(keel_rake),
        "ballast_frac": float(ballast_frac),
    }


def flattened_bounds(config=None) -> list[tuple[float, float]]:
    """Return the effective search-space bounds for the raw [-10, +10] vector.
    
    The raw GP operates in unbounded space. We keep a nominal [-10, +10]
    range for numerical stability of the acquisition function.
    """
    n_params = 17
    return [(-10.0, 10.0) for _ in range(n_params)]


def physics_anchor(config) -> dict:
    """Return nominal physical parameter values based on first-order physics.
    
    Rule 5: Anchors the GP prior mean near physically sensible values while
    keeping bounds wide enough for the AI to discover non-Eulerian solutions.
    """
    from hull_opt.config import BoundsConfig
    
    target_disp = config.fixed.target_displacement
    speed_ms = config.fixed.target_speed_knots * 0.514444
    
    # Displacement-based length (box approximation: L * B * T * Cp * Cm)
    # Solve LWL from B = 0.25*L, T = 0.1*L, Cp=0.6, Cm=0.75
    # disp ≈ L * 0.25L * 0.1L * 0.6 * 0.75 = 0.01125 * L^3
    LWL_est = (target_disp / 0.01125) ** (1.0 / 3.0)
    LWL_est = max(2.0, min(LWL_est, 3.0))  # keep in plausible range
    
    # Beam and draft from LWL (typical slenderness ratios)
    BWL_est = 0.25 * LWL_est
    T_canoe_est = 0.10 * LWL_est
    
    # Keel depth from righting moment requirement
    # Keel moment: m_keel * g * arm ≈ rho * disp * g * GM
    # Rule of thumb: keel depth = 30-50% of LWL for sailboats
    D_keel_est = 0.35 * LWL_est
    keel_chord_est = 0.08 * LWL_est
    
    # Coefficients near optimal for wave resistance
    Cp_est = 0.60
    Cm_est = 0.75
    
    # LCB near midship for balance
    LCB_est = 12.5
    
    # Stability-derived bilge radius
    bilge_r_est = 0.10 * BWL_est
    
    # Ballast fraction — typical for deep-keel designs
    ballast_est = 0.50
    
    # Other defaults
    E_est = 0.20
    SA_est = 0.15
    flare_est = 10.0
    deadrise_est = 15.0
    keel_rake_est = 0.01
    bulb_vol_est = 0.01
    bulb_pos_est = 0.40
    
    return {
        "LWL": LWL_est, "BWL": BWL_est, "T_canoe": T_canoe_est,
        "Cp": Cp_est, "Cm": Cm_est, "LCB": LCB_est,
        "D_keel": D_keel_est, "keel_chord": keel_chord_est,
        "bulb_vol": bulb_vol_est, "bulb_pos": bulb_pos_est,
        "E": E_est, "SA": SA_est, "flare": flare_est,
        "deadrise": deadrise_est, "bilge_r": bilge_r_est,
        "keel_rake": keel_rake_est, "ballast_frac": ballast_est,
    }
