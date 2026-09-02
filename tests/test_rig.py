"""Unit tests for hull_opt.rig (single wingsail / CE / helm model)."""
import numpy as np
import pytest

from hull_opt.rig import (
    build_rig, sail_pos_x, helm_angle_deg, helm_arm,
    combined_ce_z, single_sail_ce,
)
from hull_opt.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config("config.yaml")


def _xdict(LWL=2.4, wingsail_pos=0.42, D_keel=1.0, T_canoe=0.25):
    return {
        "LWL": LWL, "wingsail_pos": wingsail_pos,
        "D_keel": D_keel, "T_canoe": T_canoe,
    }


def test_sail_pos_x_frame_conversion():
    # Mesh frame: bow at x=0, stern at x=+LWL
    assert sail_pos_x(0.5, 2.4) == pytest.approx(1.2)
    assert sail_pos_x(0.0, 2.4) == pytest.approx(0.0)
    assert sail_pos_x(1.0, 2.4) == pytest.approx(2.4)
    assert sail_pos_x(0.25, 2.4) == pytest.approx(0.6)


def test_build_rig_geometry_scales_with_lwl(config):
    # Derive expected geometry from the live config (cr/ct/b are LWL-scaled)
    f = config.fixed
    LWL = 2.4
    cr = f.wingsail_cr_frac * LWL
    ct = f.wingsail_tip_taper * cr
    span = f.wingsail_span_frac * LWL
    rig = build_rig(_xdict(LWL=LWL), config)
    assert rig["main"]["cr"] == pytest.approx(cr)
    assert rig["main"]["ct"] == pytest.approx(ct)
    assert rig["main"]["span"] == pytest.approx(span)
    assert rig["main"]["area"] == pytest.approx(0.5 * (cr + ct) * span)
    assert rig["tail"]["area"] == pytest.approx(f.wingsail_tail_area_frac * rig["main"]["area"])
    assert rig["area"] == pytest.approx(rig["main"]["area"] + rig["tail"]["area"])
    # CE height at 0.35 × span
    assert rig["combined"]["z"] == pytest.approx(0.35 * span)


def test_build_rig_mast_at_quarter_chord(config):
    # Mast axis = wingsail_pos × LWL = the wing's quarter-chord station
    f = config.fixed
    LWL, pos = 2.4, 0.40
    cr = f.wingsail_cr_frac * LWL
    rig = build_rig(_xdict(LWL=LWL, wingsail_pos=pos), config)
    x_mast = sail_pos_x(pos, LWL)
    assert rig["main"]["x"] == pytest.approx(x_mast)
    # Tail sits tail_arm_chords aft of the wing trailing edge (+ quarter-chord)
    te = x_mast + (1.0 - f.wingsail_mast_chord_frac) * cr
    tail_chord = rig["tail"]["area"] / rig["main"]["span"]
    assert rig["tail"]["x"] == pytest.approx(
        te + f.wingsail_tail_arm_chords * cr + 0.25 * tail_chord)


def test_build_rig_combined_ce_between_wing_and_tail(config):
    rig = build_rig(_xdict(LWL=2.4), config)
    # Area-weighted CE must lie between the wing CE and the tail CE
    assert rig["main"]["x"] < rig["combined"]["x"] < rig["tail"]["x"]
    # Weighted combination: x_c = (S_w·x_w + S_t·x_t) / (S_w + S_t)
    denom = rig["main"]["area"] + rig["tail"]["area"]
    expected = (rig["main"]["area"] * rig["main"]["x"]
                + rig["tail"]["area"] * rig["tail"]["x"]) / denom
    assert rig["combined"]["x"] == pytest.approx(expected)


def test_build_rig_single_state_is_wing_only(config):
    rig = build_rig(_xdict(LWL=2.4, wingsail_pos=0.35), config)
    assert rig["single"]["x"] == sail_pos_x(0.35, 2.4)
    assert rig["single"]["z"] == pytest.approx(
        0.35 * config.fixed.wingsail_span_frac * 2.4)


def test_build_rig_scales_with_lwl(config):
    rig_a = build_rig(_xdict(LWL=2.4), config)
    rig_b = build_rig(_xdict(LWL=2.5), config)
    # Areas scale with LWL², CE x with LWL, CE z with LWL
    assert rig_b["area"] == pytest.approx(rig_a["area"] * (2.5 / 2.4) ** 2)
    assert rig_b["combined"]["x"] == pytest.approx(
        rig_a["combined"]["x"] * 2.5 / 2.4, rel=1e-9)
    assert rig_b["combined"]["z"] == pytest.approx(
        rig_a["combined"]["z"] * 2.5 / 2.4, rel=1e-9)


def test_helm_angle_monotonic_in_delta_x():
    arm = 2.0
    base = helm_angle_deg(0.0, 0.0, arm)
    assert base == pytest.approx(0.0)
    assert helm_angle_deg(0.1, 0.0, arm) > base
    assert helm_angle_deg(0.2, 0.0, arm) > helm_angle_deg(0.1, 0.0, arm)
    assert helm_angle_deg(0.1, 0.0, arm) == helm_angle_deg(-0.1, 0.0, arm)


def test_helm_angle_formula():
    # atan2(delta, arm) = atan2(1, 1) = 45°
    assert helm_angle_deg(1.0, 0.0, 1.0) == pytest.approx(45.0)


def test_helm_arm_includes_keel_and_draft():
    arm = helm_arm(1.5, _xdict(D_keel=1.0, T_canoe=0.25))
    assert arm == pytest.approx(1.5 + 1.0 + 0.125)
    # z_ce floored at 0.1
    arm_floor = helm_arm(0.0, _xdict(D_keel=1.0, T_canoe=0.25))
    assert arm_floor == pytest.approx(0.1 + 1.0 + 0.125)


def test_combined_ce_z_helper(config):
    ce_z = 0.35 * config.fixed.wingsail_span_frac * 2.4
    assert combined_ce_z(_xdict(), config) == pytest.approx(ce_z)


def test_single_sail_ce_helper(config):
    # Backward-compat helper: returns the wing-only state for either arg
    x, z = single_sail_ce(_xdict(wingsail_pos=0.40), config, which="fwd")
    assert x == sail_pos_x(0.40, 2.4)
    assert z == pytest.approx(0.35 * config.fixed.wingsail_span_frac * 2.4)
    x2, z2 = single_sail_ce(_xdict(wingsail_pos=0.40), config, which="aft")
    assert (x2, z2) == (x, z)


def test_build_rig_requires_mast_position(config):
    # wingsail_pos comes from the decoded design vector
    xd = _xdict()
    rig = build_rig(xd, config)
    assert np.isfinite(rig["combined"]["x"])
    assert np.isfinite(rig["combined"]["z"])


def test_build_rig_production_wingsail_config():
    # config.yaml wingsail: cr = cr_frac·LWL, ct = taper·cr, b = span_frac·LWL
    cfg = load_config("config.yaml")
    f = cfg.fixed
    LWL = _xdict()["LWL"]
    cr = f.wingsail_cr_frac * LWL
    ct = f.wingsail_tip_taper * cr
    span = f.wingsail_span_frac * LWL
    s_main = 0.5 * (cr + ct) * span
    rig = build_rig(_xdict(), cfg)
    assert rig["main"]["area"] == pytest.approx(s_main, rel=1e-3)
    assert rig["tail"]["area"] == pytest.approx(f.wingsail_tail_area_frac * s_main, rel=1e-3)
    assert rig["area"] == pytest.approx(s_main * (1.0 + f.wingsail_tail_area_frac), rel=1e-3)
    # Wingsail position bounds sit within (0, 1)
    assert cfg.bounds.wingsail_pos[0] > 0.0 and cfg.bounds.wingsail_pos[1] < 1.0
