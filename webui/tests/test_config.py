import tempfile
import yaml
from hull_opt.config import load_config, Config, design_vector_names


def test_load_config_defaults():
    config = load_config("config.yaml")
    assert isinstance(config, Config)
    assert abs(config.fixed.LWL - 2.4) < 1e-6
    assert abs(config.fixed.target_speed_knots - 4.0) < 1e-6
    assert abs(config.fixed.target_displacement - 0.30) < 1e-6
    assert abs(config.weights.w1 - 1.0) < 1e-6
    assert abs(config.weights.w2 - 0.5) < 1e-6


def test_bounds_as_array():
    config = load_config("config.yaml")
    bounds = config.bounds.as_array()
    assert len(bounds) == 17
    for lo, hi in bounds:
        assert lo < hi


def test_bounds_dim():
    config = load_config("config.yaml")
    assert config.bounds.dim == 17


def test_design_vector_names():
    names = design_vector_names()
    assert len(names) == 17
    assert names[0] == "LWL"
    assert names[1] == "BWL"
    assert names[3] == "Cp"
    assert names[-1] == "ballast_frac"


def test_bounds_to_dict():
    config = load_config("config.yaml")
    d = config.bounds.to_dict()
    for k in ["LWL", "BWL", "T_canoe", "Cp", "Cm", "LCB",
              "D_keel", "keel_chord", "bulb_vol", "bulb_pos",
              "E", "SA", "flare", "deadrise",
              "bilge_r", "keel_rake", "ballast_frac"]:
        assert k in d
        assert len(d[k]) == 2
        assert d[k][0] < d[k][1]


def test_config_is_frozen():
    config = load_config("config.yaml")
    import dataclasses
    assert dataclasses.is_dataclass(config)
    assert config.__dataclass_fields__ is not None


def test_new_config_fields():
    config = load_config("config.yaml")
    assert abs(config.wave_spectrum.Tp - 9.0) < 1e-6
    assert abs(config.wave_spectrum.Hs - 2.5) < 1e-6
    assert abs(config.validation.min_righting_energy - 75.0) < 1e-6
    assert abs(config.validation.storm_wind_speed_knots - 80.0) < 1e-6
    assert abs(config.validation.safety_factor_composite - 2.0) < 1e-6
    assert abs(config.validation.extreme_wave_H_factor - 4.0) < 1e-6
    assert abs(config.fixed.wing_sail_area - 2.0) < 1e-6
