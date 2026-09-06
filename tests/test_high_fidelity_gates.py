"""Gate-settings tests: the PRODUCTION (500k-cell) path must be covered.

The old gate settings (mesh (3,4), 5 layers, end_time 8 s, deltaT 2e-4) were
only ever exercised through the fast-test branch (max_cells <= 100000, which
short-circuits to a 0.1 s case). That masking alias let the production path
ship infeasible: design_98 gate-1 built a 1.39M-cell mesh and needed 32-65 h
of solver time against a 4 h timeout. These tests lock the production
settings + the cost-model math so that can never silently happen again.
"""
from pathlib import Path
import pytest
import numpy as np

from hull_opt.config import load_config
from hull_opt.high_fidelity import (
    _expected_steps,
    _check_gate_feasibility,
    _check_force_steady,
    _SEC_PER_CELL_PER_STEP,
    GateCrashError,
    GateSetupError,
)
from hull_opt.of_legacy import read_force_trace


# ── Production gate-1 settings (option-b sizing) ───────────────────────────

def test_gate1_production_settings():
    cfg = load_config("config.yaml")
    assert cfg.validation.sph_dp == 0.05
    assert cfg.validation.sph_sim_time == 6.0


def test_fast_branch_stays_short():
    cfg = load_config("config.yaml")
    assert cfg.validation.sph_dp == 0.05


# ── Cost-model / feasibility math ──────────────────────────────────────────

def test_cost_model_constant_recalibrated():
    # Measured on this box (design 140 gate-1 tank): 19 s for 26,423
    # particles x 25,000 steps => ~3e-8 s/particle/step. The old 1e-5
    # overestimated wall time ~350x and blocked any dp finer than 0.09.
    assert _SEC_PER_CELL_PER_STEP == pytest.approx(3e-8)


def test_cost_model_gate1_production_fits_budget():
    # dp=0.05, 6.0 s: ~151k particles x 60k steps x 3e-8 = ~271 s.
    _check_gate_feasibility("gate1_calm", 0.05, 6.0, 150784, 14400)


def test_cost_model_catches_old_bloated_settings():
    with pytest.raises(RuntimeError, match="infeasible"):
        _check_gate_feasibility("gate1_calm", 0.02, 8.0, 9154687, 14400)


def test_expected_steps_arithmetic():
    assert _expected_steps(3.0, 0.001) == 3000
    assert _expected_steps(8.0, 0.001) == 8000
    assert _expected_steps(1.0, 0.00025) == 4000
    assert _expected_steps(8.0, 0.0002) == 40000


def test_gate1_tank_vol_matches_towing_writer_and_fits_budget(tmp_path):
    """The gate-1 cost model in _gate_fine_cfd must OVER-estimate the
    moving-hull towing tank rendered by write_towing_case (tank length =
    2.1·LWL + travel, width 2·max(B,0.55), depth keel-tip + 1.5·chord),
    and that estimate at the production settings (dp=0.05, sim 6.0 s) must
    fit the gate-1 wall-time budget for the largest allowed design."""
    import xml.etree.ElementTree as ET
    import numpy as np
    from hull_opt.templates.dualsphysics import write_towing_case
    from hull_opt.geometry import generate_hull
    from hull_opt.config import load_config

    cfg = load_config("config.yaml")
    b = cfg.bounds
    LWL, B, T_canoe, D_keel = b.LWL[1], b.BWL[1], b.T_canoe[1], b.D_keel[1]
    keel_chord = b.keel_chord[1]
    sim_time = cfg.validation.sph_sim_time
    speed_ms = 1.65

    # Generate a current-bounds max-draft hull (legacy output/design_* STLs
    # carry the pre-2026-09-01 deep keels and must not drive this check).
    # Insane-mode D_keel max can exceed LWL-T_canoe (RealityCheck rejects
    # T+D>LWL), so clamp to the largest BUILDABLE draft, not the raw max.
    from hull_opt.config import design_vector_names
    _names = design_vector_names()
    max_vec = np.array([hi for _, hi in b.as_array()], dtype=np.float64)
    _di = _names.index("D_keel")
    D_keel = min(D_keel, LWL - T_canoe - 0.01)
    _p = (D_keel - b.D_keel[0]) / max(1e-9, b.D_keel[1] - b.D_keel[0])
    _p = min(max(_p, 1e-6), 1.0 - 1e-6)
    max_vec[_di] = -np.log(1.0 / _p - 1.0)  # inverse-sigmoid to raw space
    stl_path, _, _, _ = generate_hull(
        max_vec, output_dir=str(tmp_path / "hull"),
        LWL=LWL, target_displacement=cfg.fixed.target_displacement,
        config=cfg)

    xml = write_towing_case(tmp_path / "tow", stl_path, LWL, B, T_canoe,
                            D_keel, speed_ms=speed_ms, dp=cfg.validation.sph_dp,
                            sim_time=cfg.validation.sph_sim_time,
                            mass=150.0)
    root = ET.fromstring(xml.read_text())
    fillbox = next(el for el in root.iter() if el.tag == "fillbox")
    size = fillbox.find("size")
    tank_vol = (float(size.get("x")) * float(size.get("y"))
                * float(size.get("z")))

    # Mirrors _gate_fine_cfd (moving-hull tow geometry) and must cover the
    # actual fillbox (conservative; travel = u0·sim_time, writer's budget
    # cap only shrinks the real tank; +0.25 m covers the 4·dp inset).
    est = (2.1 * LWL + speed_ms * sim_time + 0.25) * (2.0 * max(1.0 * B, 0.55)) \
        * ((T_canoe + D_keel) + max(1.5 * keel_chord, 0.4))
    assert est >= tank_vol

    n_particles_est = int(est / (cfg.validation.sph_dp ** 3))
    _check_gate_feasibility("gate1_calm", cfg.validation.sph_dp,
                            cfg.validation.sph_sim_time, n_particles_est,
                            cfg.validation.gate_timeout)


# ── Config carries the per-gate timeouts ───────────────────────────────────

def test_config_has_per_gate_timeouts():
    cfg = load_config("config.yaml")
    assert cfg.validation.gate_timeout == 14400
    assert cfg.validation.gate2_timeout == 28800
    assert cfg.validation.gate2_timeout > cfg.validation.gate_timeout


# ── Phase 0: gate-1 steady-state guard ─────────────────────────────────────

def _write_force_trace(path: Path, times: list, fx: list):
    lines = ["Part;Time [s];Np;ForceFluid.x [N];ForceFluid.y [N];ForceFluid.z [N];ForceFluid [N]"]
    for i, (t, f) in enumerate(zip(times, fx)):
        lines.append(f"{i};{t:.6f};123;{f:.6f};0;0;{f:.6f}")
    path.write_text("\n".join(lines) + "\n")


def test_gate1_rejects_still_rising_force(tmp_path):
    t = np.linspace(0.0, 3.0, 300)
    fx = 10.0 * (t / 3.0) ** 2
    f = tmp_path / "force.dat"
    _write_force_trace(f, t.tolist(), fx.tolist())
    steady, msg = _check_force_steady(f, "gate1_calm")
    assert not steady
    assert "still rising" in msg


def test_gate1_rejects_decaying_transient(tmp_path):
    # Startup-transient signature seen in Phase 0: force spikes then decays
    # through the tail window (slope/mean ~ -0.5 > 0.25 guard).
    t = np.linspace(0.0, 3.0, 300)
    fx = 500.0 * np.exp(-2.0 * t / 3.0) + 10.0
    f = tmp_path / "force.dat"
    _write_force_trace(f, t.tolist(), fx.tolist())
    steady, msg = _check_force_steady(f, "gate1_calm")
    assert not steady
    assert "decaying" in msg


def test_gate1_accepts_quasi_steady_force(tmp_path):
    t = np.linspace(0.0, 3.0, 300)
    fx = 10.0 * np.exp(-4.0 * t / 3.0) + 25.0
    f = tmp_path / "force.dat"
    _write_force_trace(f, t.tolist(), fx.tolist())
    steady, msg = _check_force_steady(f, "gate1_calm")
    assert steady


def test_gate1_rejects_oscillating_force(tmp_path):
    # Exact-period sinusoid: dt=0.02, 5 Hz -> 10 samples/period. The tail
    # window (last 1/3 = 100 samples = 10 periods) integrates to slope ~0,
    # so only the std/mean guard (25/2^0.5 on mean 30 = 0.59 > 0.5) trips.
    t = np.arange(300) * 0.02
    fx = 30.0 + 25.0 * np.sin(2 * np.pi * 5.0 * t)
    f = tmp_path / "force.dat"
    _write_force_trace(f, t.tolist(), fx.tolist())
    steady, msg = _check_force_steady(f, "gate1_calm", max_osc_ratio=0.5)
    assert not steady
    assert "oscillating" in msg


def test_read_force_trace_both_formats(tmp_path):
    old = tmp_path / "old.dat"
    old.write_text(
        "# Time    forces(pressure, viscous)\n"
        "0.001 ((1.0 0 0) (0.5 0 0))\n"
        "0.002 ((1.5 0 0) (0.5 0 0))\n"
    )
    t, fx = read_force_trace(old)
    assert t.tolist() == [0.001, 0.002]
    assert fx.tolist() == [1.0, 1.5]
    from hull_opt.sph_resistance import read_force_trace_csv
    new = tmp_path / "new.csv"
    _write_force_trace(new, [0.001, 0.002], [2.0, 2.5])
    t2, fx2 = read_force_trace_csv(new)
    assert fx2.tolist() == [2.0, 2.5]
