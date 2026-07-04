"""
Configuration loading and dataclass definitions.
Loads YAML config files into frozen dataclasses with bounds, fixed parameters,
optimization settings, and paths.
Key exports: load_config(), Config, BoundsConfig, FixedConfig, OptimizationConfig
"""
import yaml
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BoundsConfig:
    LWL: tuple[float, float] = (2.30, 2.50)
    BWL: tuple[float, float] = (0.40, 0.60)
    T_canoe: tuple[float, float] = (0.15, 0.35)
    Cp: tuple[float, float] = (0.55, 0.65)
    Cm: tuple[float, float] = (0.60, 0.90)
    LCB: tuple[float, float] = (5.0, 20.0)
    D_keel: tuple[float, float] = (0.85, 1.20)
    keel_chord: tuple[float, float] = (0.15, 0.25)
    bulb_vol: tuple[float, float] = (0.001, 0.004)
    bulb_pos: tuple[float, float] = (0.30, 0.50)
    E: tuple[float, float] = (0.15, 0.30)
    SA: tuple[float, float] = (0.05, 0.25)
    flare: tuple[float, float] = (5.0, 15.0)
    deadrise: tuple[float, float] = (5.0, 25.0)
    bilge_r: tuple[float, float] = (0.05, 0.30)
    keel_rake: tuple[float, float] = (0.001, 0.02)
    ballast_frac: tuple[float, float] = (0.30, 0.70)

    def as_array(self) -> list[tuple[float, float]]:
        return [self.LWL, self.BWL, self.T_canoe, self.Cp, self.Cm,
                self.LCB, self.D_keel, self.keel_chord, self.bulb_vol,
                self.bulb_pos, self.E, self.SA, self.flare,
                self.deadrise, self.bilge_r, self.keel_rake,
                self.ballast_frac]

    @property
    def dim(self) -> int:
        return len(self.as_array())

    def to_dict(self) -> dict:
        return {k: list(v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class FixedConfig:
    LWL: float = 2.4
    target_speed_knots: float = 4.0
    target_displacement: float = 0.30
    gravity: float = 9.81
    rho_water: float = 1025.0
    nu_water: float = 1.19e-6
    electronics_bay: list[float] = field(default_factory=lambda: [0.0, 0.0, -0.05])
    wing_sail_area: float = 2.0
    wing_sail_height: float = 1.5


@dataclass(frozen=True)
class WeightConfig:
    w1: float = 1.0
    w2: float = 0.5
    w3: float = 0.5
    w4: float = 2.0
    stability_normalization: float = 30.0
    light_wind_bonus: float = 0.2


@dataclass(frozen=True)
class OptimizationConfig:
    n_initial: int = 80  # keep for backward compat, but Agents 1+2 use lhs_min/max
    n_iter: int = 300
    convergence_threshold: float = 0.01
    min_iterations: int = 30
    num_restarts: int = 20
    raw_samples: int = 100
    gp_jitter: float = 1.0e-6
    lhs_min: int = 20
    lhs_increment: int = 10
    lhs_max: int = 40
    lhs_seed: int = 42


@dataclass(frozen=True)
class CalibrationConfig:
    frequency: int = 20
    coarse_cells: int = 200000
    tolerance: float = 0.20
    timeout: int = 7200
    n_procs: int = 1


@dataclass(frozen=True)
class ValidationConfig:
    fine_cfd_cells: int = 1500000
    regular_wave_H: float = 2.0
    regular_wave_T: float = 6.0
    extreme_wave_H_factor: float = 4.0
    drop_height: float = 3.0
    inverted_speed_knots: float = 25.0
    max_accel_g: float = 30.0
    max_pressure_pa: float = 100000.0
    max_self_right_time_s: float = 10.0
    rt_upper_bound_factor: float = 2.0
    storm_wind_speed_knots: float = 80.0
    safety_factor_composite: float = 2.0
    min_righting_energy: float = 75.0
    min_ballast_ratio: float = 0.25
    min_reserve_buoyancy: float = 0.30
    min_downflooding_angle: float = 85.0
    n_procs: int = 1


@dataclass(frozen=True)
class WaveSpectrumConfig:
    type: str = "JONSWAP"
    Hs: float = 2.5
    Tp: float = 9.0
    gamma: float = 3.3
    n_freq: int = 100
    n_dir: int = 24
    min_wind_speed_kt: float = 6.0


@dataclass(frozen=True)
class PathConfig:
    output_dir: str = "./output"
    openfoam_env: str = "/opt/openfoam2512/etc/bashrc"
    dualsphysics_dir: str = "/opt/dualsphysics/5.4"
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
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


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
        paths=PathConfig(**paths_raw),
        logging=LoggingConfig(**log_raw),
    )

    # Validate parameter/bounds consistency
    _validate_param_bounds_consistency(config)
    return config


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
            "E", "SA", "flare", "deadrise",
            "bilge_r", "keel_rake", "ballast_frac"]
