import tempfile
import yaml
from hull_opt.config import load_config, Config, design_vector_names


def test_load_config_defaults():
    config = load_config("config.yaml")
    assert isinstance(config, Config)
    assert abs(config.fixed.LWL - 2.4) < 1e-6  # dataclass fallback (LWL is a design variable)
    assert abs(config.fixed.target_speed_knots - 3.5) < 1e-6
    assert abs(config.fixed.target_displacement - 0.10) < 1e-6
    assert abs(config.weights.w1 - 1.8) < 1e-6
    assert abs(config.weights.w2 - 0.25) < 1e-6
    assert abs(config.weights.w5 - 0.8) < 1e-6


def test_bounds_as_array():
    config = load_config("config.yaml")
    bounds = config.bounds.as_array()
    assert len(bounds) == 21
    for lo, hi in bounds:
        assert lo <= hi
    # flat-deck directive: sheer/rake pinned
    d = config.bounds.to_dict()
    assert d["sheer_bow"] == [0.0, 0.0]
    assert d["forefoot_cut"] == [0.0, 0.6]


def test_bounds_dim():
    config = load_config("config.yaml")
    assert config.bounds.dim == 21


def test_design_vector_names():
    names = design_vector_names()
    assert len(names) == 21
    assert names[0] == "LWL"
    assert names[1] == "BWL"
    assert names[3] == "Cp"
    assert names[-1] == "forefoot_cut"
    assert names[-2] == "stem_rake_deg"


def test_bounds_to_dict():
    config = load_config("config.yaml")
    d = config.bounds.to_dict()
    for k in ["LWL", "BWL", "T_canoe", "Cp", "Cm", "LCB",
              "D_keel", "keel_chord", "bulb_vol", "bulb_pos",
              "E", "flare", "deadrise",
              "bilge_r", "keel_rake", "ballast_frac",
              "wingsail_pos", "sheer_bow", "sheer_stern", "stem_rake_deg",
              "forefoot_cut"]:
        assert k in d
        assert len(d[k]) == 2
        assert d[k][0] <= d[k][1]  # pinned [0,0] flat-deck bounds allowed


def test_config_is_frozen():
    config = load_config("config.yaml")
    import dataclasses
    assert dataclasses.is_dataclass(config)
    assert config.__dataclass_fields__ is not None


def test_new_config_fields():
    config = load_config("config.yaml")
    assert abs(config.wave_spectrum.Tp - 9.0) < 1e-6
    assert abs(config.wave_spectrum.Hs - 2.5) < 1e-6
    assert abs(config.validation.min_righting_energy - 40.0) < 1e-6
    assert abs(config.validation.storm_wind_speed_knots - 80.0) < 1e-6
    assert abs(config.validation.safety_factor_composite - 2.0) < 1e-6
    assert abs(config.validation.extreme_wave_H_factor - 4.0) < 1e-6
    assert abs(config.validation.single_sail_max_helm_angle_deg - 5.0) < 1e-6
    assert abs(config.fixed.wingsail_cr_frac - 0.34) < 1e-6
    assert abs(config.fixed.wingsail_tip_taper - 0.7) < 1e-6
    assert abs(config.fixed.wingsail_span_frac - 1.00) < 1e-6
    assert abs(config.fixed.wingsail_tail_area_frac - 0.35) < 1e-6
    assert abs(config.fixed.wingsail_tail_arm_chords - 1.75) < 1e-6
    assert abs(config.fixed.wingsail_mast_mass - 7.0) < 1e-6
    assert abs(config.fixed.wingsail_nose_pod_mass - 3.0) < 1e-6
    assert abs(config.fixed.mast_cg_height_frac - 0.50) < 1e-6
