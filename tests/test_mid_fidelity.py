"""
Mid-fidelity (DualSPHysics) calibration tests.

Regression for the preflight-before-render crash: mid_fidelity.py used to run
preflight_case() against an empty case_dir (the towing XML is rendered later
inside run_towing_resistance), so the "Case XML file found (none)" FAIL always
aborted calibration. The pipeline is now generate_hull -> build_towing_case ->
preflight_case -> run_towing_resistance.

These tests exercise the real geometry generation and the real DualSPHysics
toolchain on this machine, so they are marked slow (excluded by default).
"""
import numpy as np
import pytest
from pathlib import Path

pytestmark = pytest.mark.slow

from hull_opt.config import load_config
from hull_opt.geometry import generate_hull
from hull_opt.utils import knots_to_ms

_CONFIG = load_config("config.yaml")

DESIGN = np.array([
    2.40,   # LWL
    0.50,   # BWL
    0.20,   # T_canoe
    0.60,   # Cp
    0.75,   # Cm
    10.0,   # LCB
    1.00,   # D_keel
    0.20,   # keel_chord
    0.003,  # bulb_vol
    0.45,   # bulb_pos
    0.20,   # E
    0.80,   # flare
    12.0,   # deadrise
    0.10,   # bilge_r
    0.005,  # keel_rake
    0.55,   # ballast_frac
    0.42,   # wingsail_pos
])


def _x_dict():
    return {
        "LWL": float(DESIGN[0]),
        "BWL": float(DESIGN[1]),
        "T_canoe": float(DESIGN[2]),
        "D_keel": float(DESIGN[5]),
        "keel_chord": float(DESIGN[7]),
    }


def test_preflight_passes_after_case_render(tmp_path):
    """generate_hull -> build_towing_case -> preflight_case must pass, proving
    the calibration pipeline no longer aborts on a missing case XML."""
    from hull_opt.preflight import preflight_case
    from hull_opt.sph_resistance import build_towing_case

    stl_path, sac_path, hydro, hull_stl = generate_hull(
        DESIGN, output_dir=str(tmp_path), config=_CONFIG,
    )
    assert Path(stl_path).exists()

    case_dir = tmp_path / "case"
    build_towing_case(
        case_dir, stl_path, _x_dict(), _CONFIG,
        speed_ms=knots_to_ms(_CONFIG.fixed.target_speed_knots),
    )
    assert sorted(case_dir.glob("case_*.xml")), "towing case XML must exist"

    pf_ok, pf_msgs = preflight_case(case_dir, _CONFIG)
    assert pf_ok, f"preflight must pass on the rendered case:\n" + "\n".join(pf_msgs)


def test_preflight_warns_but_passes_on_non_watertight_stl(tmp_path):
    """DualSPHysics does not need a watertight mesh (hull_full.stl has open
    keel/bulb patches by design): a non-watertight STL is a WARN, not a FAIL.
    Uses a synthetic open mesh so the outcome is deterministic."""
    import trimesh
    from hull_opt.preflight import preflight_case
    from hull_opt.sph_resistance import build_towing_case

    box = trimesh.creation.box(extents=(1.0, 0.5, 0.3))
    box.faces = box.faces[:-2]  # drop the last face -> open boundary
    box.remove_unreferenced_vertices()
    assert not box.is_watertight, "test mesh must be non-watertight"
    stl_path = tmp_path / "open_box.stl"
    box.export(stl_path)

    case_dir = tmp_path / "case_warn"
    build_towing_case(
        case_dir, str(stl_path), _x_dict(), _CONFIG,
        speed_ms=knots_to_ms(_CONFIG.fixed.target_speed_knots),
    )

    pf_ok, pf_msgs = preflight_case(case_dir, _CONFIG)
    assert pf_ok, f"non-watertight STL must not fail preflight:\n" + "\n".join(pf_msgs)
    watertight_msgs = [m for m in pf_msgs if "watertight" in m.lower()]
    assert watertight_msgs, "preflight must report the watertightness state"
    assert all(m.startswith("WARN") for m in watertight_msgs), pf_msgs


def test_mid_fidelity_aborts_before_solver_when_preflight_fails(tmp_path, monkeypatch):
    """If the rendered case is genuinely broken (STL missing), calibration
    returns None without touching the solver."""
    from hull_opt.mid_fidelity import run_mid_fidelity_calibration

    def _bad_generate(*a, **kw):
        return "/nonexistent/stl", "/nonexistent/sac", {"nabla": 0.1}, "/nonexistent/hull.stl"

    def _bad_validate(*a, **kw):
        return True, "ok"

    monkeypatch.setattr("hull_opt.geometry_validator.validate_design_vector", _bad_validate)
    monkeypatch.setattr("hull_opt.mid_fidelity.generate_hull", _bad_generate)

    result = run_mid_fidelity_calibration(DESIGN, 1, 0, _CONFIG)
    assert result is None
