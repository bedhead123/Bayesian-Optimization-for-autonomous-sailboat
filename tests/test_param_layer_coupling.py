import numpy as np
import pytest

from hull_opt.config import load_config
from hull_opt.param_layer import (
    bulb_vol_max_for,
    design_vector_to_physical,
)


def test_bulb_vol_max_for_matches_geometry_cap():
    # geometry.py:991-998 caps bulb radius at keel_chord/2:
    # V_max = 4/3 * pi * (keel_chord/2)^3
    assert bulb_vol_max_for(0.15) == pytest.approx(
        4.0 / 3.0 * np.pi * (0.15 * 0.5) ** 3
    )
    assert bulb_vol_max_for(0.25) == pytest.approx(
        4.0 / 3.0 * np.pi * (0.25 * 0.5) ** 3
    )


def test_bulb_vol_max_for_monotonic_in_keel_chord():
    chords = np.linspace(0.15, 0.25, 11)
    vols = [bulb_vol_max_for(c) for c in chords]
    assert all(v_next > v_prev for v_prev, v_next in zip(vols, vols[1:]))


def test_bulb_vol_clamped_to_geometric_max_at_min_chord():
    config = load_config("config.yaml")
    raw = np.zeros(17)
    raw[7] = -10.0  # keel_chord -> lower bound
    raw[8] = 20.0   # bulb_vol -> upper bound (saturated)
    x = design_vector_to_physical(raw, config)
    assert x["keel_chord"] == pytest.approx(config.bounds.keel_chord[0], abs=1e-3)
    max_vol = bulb_vol_max_for(x["keel_chord"])
    assert x["bulb_vol"] <= max_vol
    # sigmoid saturates AT the effective (coupled) upper bound, not past it;
    # the effective bound is the config bound capped by the geometric max
    eff_hi = max(config.bounds.bulb_vol[0],
                 min(config.bounds.bulb_vol[1], max_vol))
    assert x["bulb_vol"] == pytest.approx(eff_hi, abs=1e-9)


def test_bulb_vol_config_upper_bound_reachable_at_max_chord():
    config = load_config("config.yaml")
    raw = np.zeros(17)
    raw[7] = 20.0  # keel_chord -> upper bound
    raw[8] = 20.0  # bulb_vol -> upper bound (saturated)
    x = design_vector_to_physical(raw, config)
    assert x["keel_chord"] == pytest.approx(config.bounds.keel_chord[1], abs=1e-3)
    eff_hi = max(config.bounds.bulb_vol[0],
                 min(config.bounds.bulb_vol[1],
                     bulb_vol_max_for(x["keel_chord"])))
    assert x["bulb_vol"] == pytest.approx(eff_hi, abs=1e-9)
    if bulb_vol_max_for(x["keel_chord"]) >= config.bounds.bulb_vol[1]:
        assert x["bulb_vol"] == pytest.approx(config.bounds.bulb_vol[1], abs=1e-9)


def test_bulb_vol_coupling_monotone_across_chord_range():
    config = load_config("config.yaml")
    raw = np.zeros(17)
    raw[8] = 20.0
    prev = None
    for r7 in np.linspace(-10.0, 10.0, 11):
        raw[7] = r7
        x = design_vector_to_physical(raw, config)
        assert x["bulb_vol"] <= bulb_vol_max_for(x["keel_chord"])
        if prev is not None:
            assert x["keel_chord"] > prev["keel_chord"]
            assert x["bulb_vol"] >= prev["bulb_vol"]
        prev = x


def test_bulb_vol_decouple_when_config_none():
    # config=None fallback path keeps hardcoded bounds (no coupling)
    raw = np.zeros(17)
    raw[7] = -10.0  # keel_chord near lower bound
    raw[8] = 20.0   # bulb_vol saturated
    x = design_vector_to_physical(raw, None)
    assert x["keel_chord"] < 0.181
    assert x["bulb_vol"] == pytest.approx(0.004, abs=1e-9)
