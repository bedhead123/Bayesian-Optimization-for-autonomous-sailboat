"""
Single wingsail rig model: wing + trim tail, combined CE (center of effort),
and weather-helm proxy angle — now with 360° weathervaning aero polar.

Geometry scales with LWL:
    cr   = wingsail_cr_frac * LWL            root chord
    ct   = wingsail_tip_taper * cr           tip chord
    b    = wingsail_span_frac * LWL          span
    S_w  = (cr + ct)/2 * b                   wing area
    S_t  = wingsail_tail_area_frac * S_w     tail area (same span)
    mast = quarter-chord station of the wing (wingsail_mast_chord_frac * cr
           from the leading edge); the mast axis is at wingsail_pos * LWL.
    tail = wingsail_tail_arm_chords * cr aft of the wing trailing edge.
    CE_z = 0.35 * b (taper lowers the aerodynamic center).

360° model (added for system-level balance):
    The wingsail weathervanes to set an optimal AoA (~12°). For any
    apparent wind angle (AWA) the mast spins so the wing is at that AoA,
    maximizing L/D. Apparent wind follows AWS/AWA from TWS/TWA and boat
    speed; aero force is decomposed into drive (x) and side (y) via the
    lift/drag vectors. Feathered mode (storm) sets AoA=0, CL~0, small CD.
    Heel reduces projected area by cos(heel).

All coordinates in mesh frame (bow at x = 0, stern at x = +LWL,
waterline z = 0) — consistent with hydro["CB_x"] from geometry.py.
Key exports: build_rig(), sail_pos_x(), helm_angle_deg(), helm_arm(),
             combined_ce_z(), single_sail_ce(), wing_polar(),
             compute_aero_forces_360()
"""
import numpy as np


def sail_pos_x(pos_frac: float, LWL: float) -> float:
    """Convert mast position (%LWL from bow, [0,1]) to mesh-frame x.

    pos_frac=0.5 (midship) maps to x=LWL/2; bow (0) to 0; stern (1) to LWL.
    """
    return float(pos_frac) * LWL


def extended_keel_clr_x(x_dict: dict) -> float:
    """PYD Ch.9 fin-keel hydrodynamic CLR: point on the 25%-chord line at
    45% of the keel draft (keel extended to the waterline). NOT CB_x, NOT
    the geometric/area centroid. keel_rake is in DEGREES (geometry uses
    deg2rad). The ~30° debris-shedding sweep moves the CLR aft by up to
    0.5·D·tan(rake) — the forum's "sweep loads the bottom" correction."""
    LWL = x_dict["LWL"]
    D = float(x_dict.get("D_keel", 1.0))
    kc = float(x_dict.get("keel_chord", 0.2))
    bp = float(x_dict.get("bulb_pos", 0.4))
    rake = float(x_dict.get("keel_rake", 15.0))
    x_le = bp * LWL - 0.4 * kc          # keel root LE anchor (geometry.py)
    clr = x_le + 0.25 * kc + 0.45 * D * np.tan(np.radians(rake))
    return float(clr)


def signed_lead_frac(ce_x: float, clr_x: float, LWL: float) -> float:
    """lead = (CE_x - CLR_x) / LWL, SIGHED.

    + = weather helm (safe — boat rounds up and sheds load in a gust,
    PYD p.201); - = lee helm (bear-away, "almost impossible to counteract
    with your rudder" — forum #15). Recommended fin-keel band 2-6% LWL
    (PYD p.205)."""
    return float((ce_x - clr_x) / max(1e-9, LWL))


def build_rig(x_dict: dict, config) -> dict:
    """Build the single-wingsail rig geometry from a decoded design vector.

    Returns:
        main:     {area, height, x, cr, ct, span}   wing CE state (main only)
        tail:     {area, height, x}                 trim-tail CE state
        combined: {x, z}                            area-weighted CE (wing+tail)
        single:   {x, z}                            wing-only CE state
        area:     total sail area (wing + tail), m²
        ce_z:     combined CE height above WL, m
    """
    LWL = x_dict["LWL"]
    f = config.fixed
    cr = f.wingsail_cr_frac * LWL
    ct = f.wingsail_tip_taper * cr
    span = f.wingsail_span_frac * LWL
    S_w = 0.5 * (cr + ct) * span
    S_t = f.wingsail_tail_area_frac * S_w

    x_mast = sail_pos_x(x_dict["wingsail_pos"], LWL)
    # Mast axis sits mast_chord_frac * cr from the wing LE; the wing CE is at
    # the quarter-chord station, i.e. at the mast axis.
    wing_ce_x = x_mast
    wing_te_x = x_mast + (1.0 - f.wingsail_mast_chord_frac) * cr
    tail_chord = S_t / max(span, 1e-9)
    tail_ce_x = wing_te_x + f.wingsail_tail_arm_chords * cr + 0.25 * tail_chord

    ce_z = 0.35 * span
    total_area = S_w + S_t
    combined_x = (S_w * wing_ce_x + S_t * tail_ce_x) / max(total_area, 1e-10)

    return {
        "main": {"area": float(S_w), "height": float(ce_z), "x": float(wing_ce_x),
                 "cr": float(cr), "ct": float(ct), "span": float(span)},
        "tail": {"area": float(S_t), "height": float(ce_z), "x": float(tail_ce_x)},
        "combined": {"x": float(combined_x), "z": float(ce_z)},
        "single": {"x": float(wing_ce_x), "z": float(ce_z)},
        "area": float(total_area),
        "ce_z": float(ce_z),
    }


def helm_angle_deg(ce_x: float, clr_x: float, arm: float) -> float:
    """Weather-helm proxy: the angle the sail-force resultant must tilt
    (in the vertical plane) to counter the CE-CLR longitudinal offset.

    ce_x  — center of effort x (mesh frame)
    clr_x — center of lateral resistance x (mesh frame, ~CB_x)
    arm   — vertical lever arm between CE and CLR (m)
    """
    return float(np.degrees(np.arctan2(abs(ce_x - clr_x), max(arm, 1e-9))))


def helm_arm(z_ce: float, x_dict: dict) -> float:
    """Vertical lever arm for the helm proxy: CE height + keel depth + half draft."""
    return max(z_ce, 0.1) + x_dict["D_keel"] + x_dict["T_canoe"] * 0.5


def combined_ce_z(x_dict: dict, config) -> float:
    """Area-weighted CE height above waterline for the combined sail set."""
    return build_rig(x_dict, config)["combined"]["z"]


def single_sail_ce(x_dict: dict, config, which: str = "fwd") -> tuple[float, float]:
    """(x, z) center of effort for the single (wing-only) sail state.

    `which` is accepted for backward compatibility; a wingsail has no
    fwd/aft sail split, so both values return the wing-only state.
    """
    rig = build_rig(x_dict, config)
    return rig["single"]["x"], rig["single"]["z"]


# ── 360° weathervaning aero polar ──────────────────────────────────────
RHO_AIR = 1.225
CL_OPT = 1.0           # JMSE combined wing+tail CLmax ~1.16 @15°; AR-corrected below
AOA_OPT_RAD = np.radians(12.0)
CD0 = 0.022            # parasitic drag at AoA=0 (JMSE Table-3 level, was 0.015)
CD_FEATHERED = 0.020   # drag when feathered (aligned with wind)


def rig_ar_eff(x_dict: dict, config) -> float:
    """Effective aspect ratio of the wingsail, doubled for the deck
    end-plate mirror (PYD p.105 / Hazen model)."""
    f = config.fixed
    cr = f.wingsail_cr_frac * x_dict["LWL"]
    ct = f.wingsail_tip_taper * cr
    span = f.wingsail_span_frac * x_dict["LWL"]
    ar_geo = span / max(1e-9, 0.5 * (cr + ct))
    return 2.0 * ar_geo


def wing_polar(aoa_rad: float, feathered: bool = False,
               ar_eff: float = None) -> tuple[float, float]:
    """Lift/drag coefficients for the wingsail at a given AoA.

    CL is finite-AR corrected via the Prandtl lifting-line slope factor
    AR/(AR+2) (PYD Fig 6.6: at ARe=3, 5° AoA gives <1/3 of the 2D lift);
    induced drag uses K = 1/(π·e·ARe) with e=0.75 instead of the old
    fixed 0.06. Feathered returns CL≈0 and a small CD.
    """
    if feathered:
        return 0.0, CD_FEATHERED
    aoa = float(aoa_rad)
    if ar_eff is None or ar_eff <= 0:
        ar_eff = 5.5   # deck end-plated AR 2.74·2
    k_ind = 1.0 / (np.pi * 0.75 * ar_eff)
    slope_factor = ar_eff / max(1e-9, ar_eff + 2.0)
    # Normalize so peak at AOA_OPT
    denom = np.sin(2.0 * AOA_OPT_RAD)
    if abs(denom) < 1e-9:
        denom = 1.0
    cl = CL_OPT * np.sin(2.0 * aoa) / denom * slope_factor
    # Clip to physical range
    cl = float(np.clip(cl, -1.4, 1.4))
    stall = 0.25 * np.sin(aoa) ** 2
    cd = CD0 + k_ind * cl * cl + stall
    return float(cl), float(cd)


def compute_aero_forces_360(
    x_dict: dict,
    config,
    TWS_ms: float,
    TWA_deg: float,
    boat_speed_ms: float = 0.0,
    heel_deg: float = 0.0,
    feathered: bool = False,
) -> dict:
    """Aero forces for a wingsail that weathervanes 360°.

    The mast spins to put the wing at AOA_OPT to the apparent wind, so
    CL/CD are always optimal (or small if feathered). The resulting
    vector is projected into boat axes: drive (x, +forward) and side (y).

    Returns dict with drive_N, side_N, heeling_moment_Nm, yaw_moment_Nm,
    AWS_ms, AWA_deg, CL, CD, FA_mag.

    Heel reduces effective projected area by cos(heel) and heeling moment
    arm by cos(heel) as the sail tips to leeward.
    """
    rig = build_rig(x_dict, config)
    area = rig["area"]
    ce_z = rig["combined"]["z"]

    # Heel correction
    heel_rad = np.radians(float(heel_deg))
    area_eff = area * max(0.05, np.cos(heel_rad))
    ce_z_eff = ce_z * max(0.05, np.cos(heel_rad))

    # Apparent wind
    twa = np.radians(float(TWA_deg))
    tws = float(TWS_ms)
    bs = float(boat_speed_ms)
    # True wind vector FROM direction: [cos(TWA), sin(TWA)] * TWS
    # Apparent wind = true wind - boat velocity (boat moving +x)
    aw_x = tws * np.cos(twa) - bs
    aw_y = tws * np.sin(twa)
    AWS = float(np.hypot(aw_x, aw_y))
    if AWS < 0.1:
        return {
            "drive_N": 0.0, "side_N": 0.0, "heeling_moment_Nm": 0.0,
            "yaw_moment_Nm": 0.0, "AWS_ms": AWS, "AWA_deg": 0.0,
            "CL": 0.0, "CD": CD0, "FA_mag": 0.0,
            "area_eff": area_eff, "ce_z_eff": ce_z_eff,
        }
    AWA = float(np.arctan2(aw_y, aw_x))  # FROM angle relative to bow

    # Wing always trims to optimal AoA, so CL/CD are optimal (AR-corrected)
    ar_eff = rig_ar_eff(x_dict, config)
    if feathered:
        cl, cd = wing_polar(0.0, feathered=True)
        # Small feathered factor also scales with residual area fraction
        # already applied via CL~0; CD stays small
    else:
        cl, cd = wing_polar(AOA_OPT_RAD, feathered=False, ar_eff=ar_eff)

    q = 0.5 * RHO_AIR * AWS * AWS
    L = q * area_eff * abs(cl)
    D = q * area_eff * cd

    # Wind FROM unit vector
    cos_awa = np.cos(AWA)
    sin_awa = np.sin(AWA)
    # Drag acts opposite FROM vector (downwind), i.e. -w_from
    D_vec = np.array([-D * cos_awa, -D * sin_awa])
    # Lift perpendicular to wind direction (two signs)
    # Rotate -w_from 90° cw/ccw to get lift directions
    L_vec_a = np.array([L * sin_awa, -L * cos_awa])
    L_vec_b = -L_vec_a

    # Pick tack that maximizes drive (Fx)
    F_a = D_vec + L_vec_a
    F_b = D_vec + L_vec_b
    F = F_a if F_a[0] >= F_b[0] else F_b

    drive = float(F[0])
    side = float(F[1])

    # Moments: heeling from side * ce_z; yaw from side*long arm + drive*small arm
    # Use side absolute for heeling; sign matters for yaw direction
    heeling = float(abs(side) * ce_z_eff)

    # Yaw moment about CLR (positive = weather helm)
    try:
        clr_x_hint = float(x_dict.get("LCB", 45.0) / 100.0 * x_dict["LWL"])
    except Exception:
        clr_x_hint = 1.2
    arm_long = rig["combined"]["x"] - clr_x_hint
    yaw = float(side * arm_long)  # side creates yaw; drive lever is small, ignore

    FA_mag = float(np.hypot(drive, side))

    return {
        "drive_N": drive,
        "side_N": side,
        "heeling_moment_Nm": heeling,
        "yaw_moment_Nm": yaw,
        "AWS_ms": AWS,
        "AWA_deg": float(np.degrees(AWA)),
        "CL": float(cl),
        "CD": float(cd),
        "FA_mag": FA_mag,
        "area_eff": float(area_eff),
        "ce_z_eff": float(ce_z_eff),
    }
