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


def bulb_vol_max_for(keel_chord: float) -> float:
    """Geometric max bulb volume (m³) for a given keel chord.

    Mirrors the build-time cap in geometry.py:991-998, where the bulb radius
    is limited to half the keel chord: V_max = 4/3·pi·(keel_chord/2)^3.
    Replicated locally because geometry.py imports param_layer, so importing
    it here would cycle.
    """
    return 4.0 / 3.0 * np.pi * (keel_chord * 0.5) ** 3


def design_vector_to_physical(raw: np.ndarray, config=None) -> dict:
    """Convert 18-element raw GP vector to physical hull parameters.

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
    lcb_lo, lcb_hi = (45.0, 56.0) if config is None else config.bounds.LCB
    LCB = lcb_lo + (lcb_hi - lcb_lo) * sigmoid(r[5])

    # Keel depth as ratio of LWL (Rule 1)
    dk_lo, dk_hi = (0.45, 0.65) if config is None else config.bounds.D_keel
    D_keel = dk_lo + (dk_hi - dk_lo) * sigmoid(r[6])

    # Keel chord as ratio of LWL (Rule 1)
    kc_lo, kc_hi = (0.18, 0.26) if config is None else config.bounds.keel_chord
    keel_chord = kc_lo + (kc_hi - kc_lo) * sigmoid(r[7])

    # Bulb volume as fraction of target displacement (Rule 1). The geometric
    # max scales with keel_chord (bulb radius capped at chord/2, geometry.py
    # lines 991-998), so the effective upper bound shrinks with keel_chord:
    # the sigmoid then saturates AT the geometric max instead of being
    # silently clipped past it.
    bv_lo, bv_hi = (0.0015, 0.004) if config is None else config.bounds.bulb_vol
    if config is not None:
        bv_hi = max(bv_lo, min(bv_hi, bulb_vol_max_for(keel_chord)))
    bulb_vol = bv_lo + (bv_hi - bv_lo) * sigmoid(r[8])

    # Bulb position as fraction of LWL from bow (Rule 1)
    bp_lo, bp_hi = (0.30, 0.50) if config is None else config.bounds.bulb_pos
    bulb_pos = bp_lo + (bp_hi - bp_lo) * sigmoid(r[9])

    # Sheer parameters as ratios of LWL (Rule 1)
    e_lo, e_hi = (0.15, 0.30) if config is None else config.bounds.E
    E = e_lo + (e_hi - e_lo) * sigmoid(r[10])

    # Flare angle - squashed (Rule 2), bounded by tan(flare) < beam/draft (Rule 5)
    fl_lo, fl_hi = (8.0, 15.0) if config is None else config.bounds.flare
    flare = fl_lo + (fl_hi - fl_lo) * sigmoid(r[11])

    # Deadrise angle - squashed (Rule 2)
    dr_lo, dr_hi = (5.0, 25.0) if config is None else config.bounds.deadrise
    deadrise = dr_lo + (dr_hi - dr_lo) * sigmoid(r[12])

    # Bilge radius - squashed (Rule 2), max 50% of BWL (Rule 5)
    br_lo, br_hi = (0.05, 0.30) if config is None else config.bounds.bilge_r
    bilge_r = br_lo + (br_hi - br_lo) * sigmoid(r[13])

    # Keel rake angle - squashed (Rule 2), DEGREES (geometry uses deg2rad)
    kr_lo, kr_hi = (15.0, 30.0) if config is None else config.bounds.keel_rake
    keel_rake = kr_lo + (kr_hi - kr_lo) * sigmoid(r[14])

    # Ballast fraction - squashed, always [0,1] (Rule 2)
    bf_lo, bf_hi = (0.35, 0.55) if config is None else config.bounds.ballast_frac
    ballast_frac = bf_lo + (bf_hi - bf_lo) * sigmoid(r[15])

    # Wingsail mast position as fraction of LWL from bow (Rule 1)
    wp_lo, wp_hi = (0.30, 0.55) if config is None else config.bounds.wingsail_pos
    wingsail_pos = wp_lo + (wp_hi - wp_lo) * sigmoid(r[16])

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
        "flare": float(flare),
        "deadrise": float(deadrise),
        "bilge_r": float(bilge_r),
        "keel_rake": float(keel_rake),
        "ballast_frac": float(ballast_frac),
        "wingsail_pos": float(wingsail_pos),
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
    # Rule of thumb: keel depth = 25-30% of LWL for uncrewed boats
    # (Review T/L band 0.15-0.3 keeps T_total ~0.6-0.72 m at 2.4 m LWL)
    D_keel_est = 0.26 * LWL_est
    keel_chord_est = 0.09 * LWL_est

    # Coefficients near optimal for wave resistance
    Cp_est = 0.60
    Cm_est = 0.75

    # LCB near midship for balance
    LCB_est = 50.0

    # Stability-derived bilge radius
    bilge_r_est = 0.10 * BWL_est

    # Ballast fraction — PYD fleet 0.25-0.50 for deep-keel designs
    ballast_est = 0.45

    # Other defaults
    E_est = 0.30
    flare_est = 17.0
    deadrise_est = 22.0
    keel_rake_est = 22.0
    bulb_vol_est = 0.002
    bulb_pos_est = 0.40
    
    return {
        "LWL": LWL_est, "BWL": BWL_est, "T_canoe": T_canoe_est,
        "Cp": Cp_est, "Cm": Cm_est, "LCB": LCB_est,
        "D_keel": D_keel_est, "keel_chord": keel_chord_est,
        "bulb_vol": bulb_vol_est, "bulb_pos": bulb_pos_est,
        "E": E_est, "flare": flare_est,
        "deadrise": deadrise_est, "bilge_r": bilge_r_est,
        "keel_rake": keel_rake_est, "ballast_frac": ballast_est,
        "wingsail_pos": 0.42,
    }
