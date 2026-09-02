"""
Configuration loading and dataclass definitions.
Loads YAML config files into frozen dataclasses with bounds, fixed parameters,
optimization settings, and paths.
Key exports: load_config(), Config, BoundsConfig, FixedConfig, OptimizationConfig
"""
import hashlib
import yaml
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BoundsConfig:
    LWL: tuple[float, float] = (2.30, 2.50)
    BWL: tuple[float, float] = (0.55, 0.68)
    T_canoe: tuple[float, float] = (0.28, 0.36)
    Cp: tuple[float, float] = (0.55, 0.60)
    Cm: tuple[float, float] = (0.78, 0.90)
    LCB: tuple[float, float] = (45.0, 56.0)
    D_keel: tuple[float, float] = (0.45, 0.65)
    keel_chord: tuple[float, float] = (0.18, 0.26)
    bulb_vol: tuple[float, float] = (0.0015, 0.0025)
    bulb_pos: tuple[float, float] = (0.30, 0.50)
    E: tuple[float, float] = (0.28, 0.40)
    flare: tuple[float, float] = (10.0, 24.0)
    deadrise: tuple[float, float] = (16.0, 28.0)
    bilge_r: tuple[float, float] = (0.18, 0.30)
    keel_rake: tuple[float, float] = (15.0, 30.0)
    ballast_frac: tuple[float, float] = (0.35, 0.55)
    wingsail_pos: tuple[float, float] = (0.30, 0.75)

    def as_array(self) -> list[tuple[float, float]]:
        return [self.LWL, self.BWL, self.T_canoe, self.Cp, self.Cm,
                self.LCB, self.D_keel, self.keel_chord, self.bulb_vol,
                self.bulb_pos, self.E, self.flare,
                self.deadrise, self.bilge_r, self.keel_rake,
                self.ballast_frac, self.wingsail_pos]

    @property
    def dim(self) -> int:
        return len(self.as_array())

    def to_dict(self) -> dict:
        return {k: list(v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class FixedConfig:
    LWL: float = 2.4
    target_speed_knots: float = 3.5
    target_displacement: float = 0.10
    use_nurbs_geometry: bool = True
    use_nurbs_gz: bool = True
    gravity: float = 9.81
    rho_water: float = 1025.0
    nu_water: float = 1.19e-6
    electronics_bay: list[float] = field(default_factory=lambda: [0.6, 0.0, 0.05])
    wingsail_cr_frac: float = 0.34
    wingsail_tip_taper: float = 0.7
    wingsail_span_frac: float = 1.00
    wingsail_tail_area_frac: float = 0.35
    wingsail_tail_arm_chords: float = 1.75
    wingsail_mast_chord_frac: float = 0.25
    wingsail_mast_mass: float = 7.0
    wingsail_nose_pod_mass: float = 3.0
    mast_cg_height_frac: float = 0.50
    payload_mass_kg: float = 15.0
    payload_cg_z: float = 0.30
    hull_mass_floor_kg: float = 20.0
    fouling_cf_mult: float = 1.25


@dataclass(frozen=True)
class WeightConfig:
    w1: float = 1.8
    w2: float = 0.25
    w3: float = 0.6
    w4: float = 2.0
    w5: float = 0.8
    w6: float = 0.5
    w7: float = 0.10
    w8: float = 0.35
    w9: float = 0.5
    stability_normalization: float = 30.0
    light_wind_bonus: float = 1.2
    fom_drag_reference_n: float = 45.0


@dataclass(frozen=True)
class RapidValidationConfig:
    storm_hs_factor: float = 4.0
    storm_tp_factor: float = 1.3
    storm_n_freq: int = 15
    storm_headings_deg: tuple = (180.0, 135.0, 90.0)
    min_avs_deg: float = 110.0
    slam_drop_height_m: float = 15.0
    slam_max_pressure_pa: float = 3200000.0
    ops_Hs: float = 1.2  # normal seas for 2.4m boat — 4 ft, not 8 ft
    ops_Tp: float = 7.0
    parametric_roll_band_frac: float = 0.15
    roll_sigma_max_deg: float = 120.0
    roll_sigma_ops_max_deg: float = 35.0  # was 25.0 hard for 8 ft seas — 35° for 4 ft ops + soft (autonomous)
    capsize_safety_factor: float = 1.0
    wind_heel_max_frac: float = 0.85
    deck_allowable_pa: float = 100000.0
    min_max_gz_m: float = 0.05
    min_gz_area_30: float = 0.005
    min_gz_area_40_90: float = 0.01
    roll_period_min_s: float = 1.0
    roll_period_max_s: float = 8.0
    soft_margin_gates: tuple = (
        "capsize", "storm_accel", "slam_pressure", "slam_accel",
        "inverted_pressure", "roll_sigma", "roll_sigma_ops",
    )


@dataclass(frozen=True)
class ReferenceConfig:
    tool: str = "dualsphysics"
    sim_time_s: float = 10.0
    timeout_min: float = 180.0
    dp: float = 0.06
    dtout: float = 0.1
    reference_every_n: int = 10
    Hs: float = 0.6
    Tp: float = 2.5
    gamma: float = 3.3


@dataclass(frozen=True)
class OptimizationConfig:
    n_initial: int = 80
    n_iter: int = 300
    convergence_threshold: float = 0.01
    min_iterations: int = 100
    num_restarts: int = 30
    raw_samples: int = 256
    gp_jitter: float = 1.0e-4
    lhs_min: int = 30
    lhs_increment: int = 15
    lhs_max: int = 60
    lhs_seed: int = 42


@dataclass(frozen=True)
class CalibrationConfig:
    frequency: int = 0
    coarse_cells: int = 200000
    tolerance: float = 0.20
    timeout: int = 3600
    n_procs: int = 8
    # DualSPHysics towing calibration. solver="sph" selects the SPH towing
    # resistance (sph_resistance.run_towing_resistance) instead of OpenFOAM
    # interFoam. Runtime trade: fluid particles ∝ 1/dp³; at dp=0.05 the
    # towing tank holds ~230k particles and a 6 s calm-water tow takes
    # roughly 20-40 min on the GPU solver. Cases whose GenCase particle
    # count exceeds max_particles are aborted before the solver launches
    # (P0.2 particle preflight).
    solver: str = "sph"
    sph_dp: float = 0.05
    sph_sim_time: float = 6.0
    max_particles: int = 400000
    # Validated band for the SPH-vs-low-fi drag factor. A raw factor outside
    # the band invalidates the calibration (tow out-of-family vs the
    # CFD-validated low-fi model) instead of clipping — Bug #157: dp-coarse
    # SPH tows read ~20x and the old clip-at-5.0 corrupted every FoM.
    factor_bounds: tuple = (0.5, 2.0)
    end_time: float = 20.0
    delta_t: float = 0.001
    max_co: float = 0.7
    retry_failed: bool = True
    convergence: dict = field(default_factory=lambda: {
        "min_sim_time": 2.0,
        "window_s": 1.2,
        "rel_tol": 0.015,
        "n_checks": 2,
        "poll_interval_s": 60,
    })


@dataclass(frozen=True)
class ValidationConfig:
    fine_cfd_cells: int = 500000
    regular_wave_H: float = 2.0
    regular_wave_T: float = 6.0
    # Physical caps for the analytic gate-2 wave (wave_fields.py): the
    # regular_wave_H/T config values describe full-size vessels, not a ~2 m
    # hull in a ~1.7 m deep channel; the caps bind first for this hull class.
    wave_height_cap_frac: float = 0.5        # H <= frac * T_hull
    wave_steepness_cap: float = 0.10         # H <= frac * wavelength
    wave_span_frac: float = 0.9              # wavelength <= frac * domain span
    extreme_wave_H_factor: float = 4.0
    drop_height: float = 15.0
    inverted_speed_knots: float = 25.0
    max_accel_g: float = 19.0
    max_pressure_pa: float = 3200000.0
    max_self_right_time_s: float = 10.0
    rt_upper_bound_factor: float = 2.0
    storm_wind_speed_knots: float = 80.0
    safety_factor_composite: float = 2.0
    min_righting_energy: float = 40.0
    min_ballast_ratio: float = 0.25
    min_reserve_buoyancy: float = 0.40
    single_sail_max_helm_angle_deg: float = 5.0
    n_procs: int = 8
    # Wall-time budgets per gate. Gate 2 (wave motions) needs more: a full
    # regular-wave period + response at a coarser dt still costs 6-8k steps.
    gate_timeout: int = 14400
    gate2_timeout: int = 28800
    # DualSPHysics towing/validation settings. Gate 1 (calm-water Rt) uses
    # sph_dp/sph_sim_time; gate 5 (inverted deck pressure) uses the inverted
    # pair. Runtime trade: particles ∝ 1/dp³, and each case costs roughly
    # 30-60 min on the GPU at dp≈0.02-0.03 (the inverted tank is shallower
    # and cheaper than the towing tank, hence the larger default dp).
    sph_dp: float = 0.05
    sph_sim_time: float = 6.0
    # Gate 1 PASS/FAIL uses the OF-validated low-fi Rt; the SPH tow runs only
    # as an audit when sph_towing_audit is true (Bug #157: the fixed-hull
    # current setup measures momentum-flux drag ~14x low-fi).
    sph_towing_audit: bool = False
    sph_inverted_dp: float = 0.075
    sph_inverted_sim_time: float = 6.0
    sph_calibration: bool = True


@dataclass(frozen=True)
class WaveSpectrumConfig:
    type: str = "JONSWAP"
    Hs: float = 2.5
    Tp: float = 9.0
    gamma: float = 3.3
    n_freq: int = 100
    n_dir: int = 24
    min_wind_speed_kt: float = 6.0
    # Capytaine BEM sweep: dense N^3 LU factorization dominates each low-fi
    # evaluation, so keep the panel count moderate.
    bem_omega_min: float = 0.2
    bem_omega_max: float = 6.0
    bem_n_freq: int = 15
    bem_n_panels: int = 2500


@dataclass(frozen=True)
class PathConfig:
    output_dir: str = "./output"
    openfoam_env: str = "/opt/openfoam2512/etc/bashrc"
    dualsphysics_dir: str = "/home/anon/apps/DualSPHysics_v5.4"
    database: str = "./output/optimization.db"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    console: bool = True
    file: str = "./output/pipeline.log"


@dataclass(frozen=True)
class Config:
    bounds: BoundsConfig = field(default_factory=BoundsConfig)
    fixed: FixedConfig = field(default_factory=FixedConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    wave_spectrum: WaveSpectrumConfig = field(default_factory=WaveSpectrumConfig)
    rapid_validation: RapidValidationConfig = field(default_factory=RapidValidationConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def config_signature(config: Config) -> str:
    """Deterministic hash of the objective-affecting config (bounds, fixed
    parameters, weights, validation gates).

    Stored in the DB at campaign start; any run (optimize, resume,
    validate-only) whose signature differs is refused so that FoMs and gate
    thresholds are never computed under a silently different operating point.
    """
    payload = {
        "bounds": asdict(config.bounds),
        "fixed": asdict(config.fixed),
        "weights": asdict(config.weights),
        "validation": asdict(config.validation),
        "rapid_validation": asdict(config.rapid_validation),
        "reference": asdict(config.reference),
        "optimization": asdict(config.optimization),
        "calibration": asdict(config.calibration),
        "wave_spectrum": asdict(config.wave_spectrum),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: str = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    config_dir = Path(path).resolve().parent

    paths_raw = raw.get("paths", {})
    for path_key in ("output_dir", "database", "dualsphysics_dir", "openfoam_env"):
        if path_key in paths_raw:
            p = paths_raw[path_key]
            if not Path(p).is_absolute():
                paths_raw[path_key] = str((config_dir / p).resolve())

    log_raw = raw.get("logging", {})
    if "file" in log_raw:
        lf = log_raw["file"]
        if not Path(lf).is_absolute():
            log_raw["file"] = str((config_dir / lf).resolve())

    raw_rapid = raw.get("rapid_validation", {})
    if "storm_headings_deg" in raw_rapid:
        raw_rapid["storm_headings_deg"] = tuple(raw_rapid["storm_headings_deg"])
    if "soft_margin_gates" in raw_rapid:
        raw_rapid["soft_margin_gates"] = tuple(raw_rapid["soft_margin_gates"])
    config = Config(
        bounds=BoundsConfig(**{
            k: tuple(v) if isinstance(v, list) else v
            for k, v in raw.get("bounds", {}).items()
        }),
        fixed=FixedConfig(**raw.get("fixed", {})),
        weights=WeightConfig(**raw.get("weights", {})),
        optimization=OptimizationConfig(**raw.get("optimization", {})),
        calibration=CalibrationConfig(**raw.get("calibration", {})),
        validation=ValidationConfig(**raw.get("validation", {})),
        wave_spectrum=WaveSpectrumConfig(**raw.get("wave_spectrum", {})),
        rapid_validation=RapidValidationConfig(**raw_rapid),
        reference=ReferenceConfig(**raw.get("reference", {})),
        paths=PathConfig(**paths_raw),
        logging=LoggingConfig(**log_raw),
    )

    # Validate parameter/bounds consistency
    _validate_param_bounds_consistency(config)
    _validate_hull_fabric(config)
    _validate_wave_fabric(config)
    return config


def _validate_hull_fabric(config: Config) -> None:
    """Fabric check: max-bounds hull must be able to reach target_displacement with SAC cap 1.3.
    If not, bounds and target are incoherent — LHS will be 100% E_GEOM (1-2s evals)."""
    max_vec = np.array([hi for _, hi in config.bounds.as_array()])
    from hull_opt.geometry import design_vector_to_dict
    x = design_vector_to_dict(max_vec)
    # analytic raw capacity estimate (L*B*T*Cp*Cm*0.7 prismatic)
    raw = x["LWL"] * x["BWL"] * x["T_canoe"] * x["Cp"] * x["Cm"] * 0.7
    cap = raw * 1.3
    target = config.fixed.target_displacement
    if cap < target * 0.9:
        raise ValueError(f"FABRIC: max-bounds hull raw {raw:.3f}*1.3={cap:.3f} < target {target:.3f} — bounds cannot reach displacement, will be 100% E_GEOM. Increase BWL/T/Cp or lower target.")


def _validate_wave_fabric(config: Config) -> None:
    """Fabric check: storm Hs vs hull size. The survival requirement is an
    absolute 10 m hurricane sea (user directive 2026-08-25) — rolling to
    extreme angles is EXPECTED (roll_sigma/capsize are soft, informational);
    survival is judged by reserve buoyancy, AVS, self-righting and slam
    gates. So exceeding 2.5*LWL is a notice, not a failure prediction."""
    hs_storm = config.wave_spectrum.Hs * config.rapid_validation.storm_hs_factor
    if hs_storm > config.bounds.LWL[1] * 2.5:
        import warnings
        warnings.warn(f"FABRIC: Hs_storm {hs_storm:.1f}m > 2.5*LWL {config.bounds.LWL[1]*2.5:.1f}m "
                      f"— survival-mode storm: roll_sigma/capsize are informational; "
                      f"feasibility rides on reserve_buoyancy/AVS/self-right/slam gates")


def _validate_param_bounds_consistency(config: Config) -> None:
    """Validate that design_vector_names() order matches bounds order."""
    from hull_opt.geometry import design_vector_to_dict

    names = design_vector_names()
    bounds_array = config.bounds.as_array()

    assert len(names) == len(bounds_array), (
        f"Parameter name count ({len(names)}) != bounds count ({len(bounds_array)})"
    )

    # Round-trip test: create a vector from the midpoint of bounds
    test_vector = np.array([(lo + hi) / 2 for lo, hi in bounds_array])
    x_dict = design_vector_to_dict(test_vector)

    # Verify every name maps to a value
    for name in names:
        assert name in x_dict, f"Parameter '{name}' not found in design_vector_to_dict output"

    # Verify every parameter has a non-NaN value
    for k, v in x_dict.items():
        assert np.isfinite(v), f"Parameter {k} has non-finite round-trip value: {v}"


def design_vector_names() -> list[str]:
    return ["LWL", "BWL", "T_canoe", "Cp", "Cm", "LCB",
            "D_keel", "keel_chord", "bulb_vol", "bulb_pos",
            "E", "flare", "deadrise",
            "bilge_r", "keel_rake", "ballast_frac",
            "wingsail_pos"]
