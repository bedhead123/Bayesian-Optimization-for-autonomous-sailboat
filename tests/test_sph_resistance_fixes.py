"""
Regression tests for the SPH pipeline P0 fixes:

1. build_towing_case signature: mid_fidelity.py passes speed_ms as the 5th
   positional arg; the old signature (dp, sim_time, speed_ms, ...) bound
   speed_ms to dp and then raised TypeError on the dp= keyword — every
   calibration attempt failed before the solver.
2. compute_forces mk filter: the v5.4 classic dialect remaps mkbound with a
   +10 offset, so the hull's actual mk is 11, not the template's 1. The old
   -onlymk:1 selected the main fluid and ComputeForces crashed with "no
   boundary particles selected" (gate 1 + mid-fi).
3. Gate 5 hard-CRASH semantics: run_inverted_pressure must never hand the
   analytic fallback back as max_pressure_pa (a solver AbortBoundOut used to
   be silently scored as a PASS), and _gate_inverted_pressure must raise
   GateCrashError instead of scoring the fallback.
4. P0 OOM survivability: particle-count preflight aborts oversized GenCase
   layouts before the solver launch, and mid-fidelity threads its run
   through the shared gpu.lock.
5. lock_wait=True: calibration blocks on a busy gpu.lock instead of being
   silently starved (SKIPPED -> marked failed).
"""
import time
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hull_opt.config import load_config
from hull_opt.high_fidelity import GateCrashError

_CONFIG = load_config("config.yaml")

_X_DICT = {
    "LWL": 2.4,
    "BWL": 0.5,
    "T_canoe": 0.2,
    "D_keel": 1.0,
    "keel_chord": 0.2,
}

# Raw [-10, +10] GP-space vector (matches tests/test_mid_fidelity.py).
_RAW_DESIGN = np.array([
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

_REAL_GENCASE_LINE = (
    "Total particles: 1,083,974 (bound=66672 (fx=66672 mv=0 ft=0) fluid=1017302)"
)


# ── RC1: build_towing_case call pattern ─────────────────────────────────────

def test_build_towing_case_mid_fi_call_pattern(tmp_path, monkeypatch):
    """Mid-fi's exact positional call must bind speed_ms correctly and not
    raise TypeError on the dp/sim_time keywords (bug RC1)."""
    import hull_opt.templates.dualsphysics as tpl

    captured = {}

    def _fake_write(case_dir, **kw):
        captured.update(kw)
        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "case_towing.xml").write_text("<case/>")
        return case_dir / "case_towing.xml"

    monkeypatch.setattr(tpl, "write_towing_case", _fake_write)

    from hull_opt.sph_resistance import build_towing_case

    speed_ms = 1.65
    build_towing_case(
        tmp_path / "case", "/fake/hull.stl", _X_DICT, _CONFIG,
        speed_ms,           # positional, 5th arg (as mid_fidelity.py:73 does)
        dp=0.03,
        sim_time=10.0,
    )
    assert captured["speed_ms"] == pytest.approx(speed_ms)
    assert captured["dp"] == pytest.approx(0.03)
    assert captured["sim_time"] == pytest.approx(10.0)


def test_build_towing_case_keyword_call_still_works(tmp_path, monkeypatch):
    """Keyword-only callers (tests, run_towing_resistance) keep working."""
    import hull_opt.templates.dualsphysics as tpl

    captured = {}

    def _fake_write(case_dir, **kw):
        captured.update(kw)
        (case_dir).mkdir(parents=True, exist_ok=True)
        (case_dir / "case_towing.xml").write_text("<case/>")
        return case_dir / "case_towing.xml"

    monkeypatch.setattr(tpl, "write_towing_case", _fake_write)

    from hull_opt.sph_resistance import build_towing_case

    build_towing_case(
        tmp_path / "case", "/fake/hull.stl", _X_DICT, _CONFIG,
        speed_ms=1.65, dp=0.03,
    )
    assert captured["speed_ms"] == pytest.approx(1.65)


# ── RC2: ComputeForces mk filter ────────────────────────────────────────────

def test_hull_boundary_mk_is_11():
    """v5.4 classic dialect: template mk 1 + bound offset 10 -> actual mk 11."""
    from hull_opt.sph_resistance import hull_boundary_mk

    assert hull_boundary_mk() == 11


def test_compute_forces_uses_hull_mk_by_default(tmp_path, monkeypatch):
    """compute_forces() without onlymk must pass -onlymk:11 (the hull's
    actual mk), never the template mk 1 (main fluid — RC2)."""
    import hull_opt.sph_resistance as sr

    data_dir = tmp_path / "case_out" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "Part_0001.bi4").write_bytes(b"fake")
    (tmp_path / "gencase.xml").write_text("<gencase/>")
    fake_tool = tmp_path / "ComputeForces_linux64"
    fake_tool.write_bytes(b"fake")

    commands = []

    def _fake_find_tool(name, ds_dir):
        return str(fake_tool)

    def _fake_run(cmd, capture_output=True, text=True, timeout=900, check=False):
        commands.append(list(cmd))
        for i, c in enumerate(cmd):
            if c == "-savecsv" and i + 1 < len(cmd):
                Path(cmd[i + 1]).write_text("ok")
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(sr, "_find_tool", _fake_find_tool)
    monkeypatch.setattr(sr.subprocess, "run", _fake_run)

    csv = sr.compute_forces(tmp_path)
    assert csv is not None
    onlymk_args = [c for c in commands[0] if str(c).startswith("-onlymk:")]
    assert onlymk_args == ["-onlymk:11"]


def test_compute_forces_explicit_onlymk_respected(tmp_path, monkeypatch):
    """Callers may still override the mk filter explicitly."""
    import hull_opt.sph_resistance as sr

    data_dir = tmp_path / "case_out" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "Part_0001.bi4").write_bytes(b"fake")
    (tmp_path / "gencase.xml").write_text("<gencase/>")
    fake_tool = tmp_path / "ComputeForces_linux64"
    fake_tool.write_bytes(b"fake")

    commands = []

    def _fake_find_tool(name, ds_dir):
        return str(fake_tool)

    def _fake_run(cmd, capture_output=True, text=True, timeout=900, check=False):
        commands.append(list(cmd))
        for i, c in enumerate(cmd):
            if c == "-savecsv" and i + 1 < len(cmd):
                Path(cmd[i + 1]).write_text("ok")
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(sr, "_find_tool", _fake_find_tool)
    monkeypatch.setattr(sr.subprocess, "run", _fake_run)

    sr.compute_forces(tmp_path, onlymk=10)
    onlymk_args = [c for c in commands[0] if str(c).startswith("-onlymk:")]
    assert onlymk_args == ["-onlymk:10"]


# ── RC3: Gate 5 hard-CRASH semantics ────────────────────────────────────────

def test_run_inverted_pressure_solver_fail_never_fabricates_measurement(
        tmp_path, monkeypatch):
    """Solver status != OK must leave max_pressure_pa None and return the
    analytic estimate under fallback_pressure_pa only (bug RC3)."""
    import hull_opt.sph_resistance as sr

    monkeypatch.setattr(
        "hull_opt.templates.dualsphysics.write_inverted_case",
        lambda *a, **kw: tmp_path / "case_inverted.xml",
    )

    def _fail_run(*a, **kw):
        return {"status": "FAILED", "wall_time_s": 1.0, "details": "solver abort",
                "tool_kind": "gpu", "out_dir": str(tmp_path / "case_out")}

    monkeypatch.setattr(sr, "run_sph_case", _fail_run)

    result = sr.run_inverted_pressure(tmp_path, "/fake/hull.stl", _X_DICT, _CONFIG)
    assert result["status"] == "FAILED"
    assert result["max_pressure_pa"] is None
    assert result["fallback_pressure_pa"] is not None
    assert "analytic fallback" in result["details"]


def test_run_inverted_pressure_measuretool_fail_keeps_measurement_none(
        tmp_path, monkeypatch):
    """Solver OK but no MeasureTool data: no fabricated max_pressure_pa."""
    import hull_opt.sph_resistance as sr

    monkeypatch.setattr(
        "hull_opt.templates.dualsphysics.write_inverted_case",
        lambda *a, **kw: tmp_path / "case_inverted.xml",
    )

    def _ok_run(*a, **kw):
        return {"status": "OK", "wall_time_s": 1.0, "details": "ran gpu solver",
                "tool_kind": "gpu", "out_dir": str(tmp_path / "case_out")}

    monkeypatch.setattr(sr, "run_sph_case", _ok_run)
    monkeypatch.setattr(sr, "_run_measuretool", lambda *a, **kw: None)

    result = sr.run_inverted_pressure(tmp_path, "/fake/hull.stl", _X_DICT, _CONFIG)
    assert result["status"] == "OK"
    assert result["max_pressure_pa"] is None
    assert result["fallback_pressure_pa"] is not None


def test_gate5_crashes_on_solver_failure(monkeypatch):
    """_gate_inverted_pressure must raise GateCrashError when the solver
    failed, even though an analytic fallback exists (bug: false PASS)."""
    from hull_opt import high_fidelity as hf

    def _fake_run(*a, **kw):
        return {"max_pressure_pa": None, "fallback_pressure_pa": 3766.0,
                "status": "FAILED", "wall_time_s": 1.0,
                "details": "solver abort (AbortBoundOut)"}

    monkeypatch.setattr("hull_opt.sph_resistance.run_inverted_pressure", _fake_run)

    with pytest.raises(GateCrashError, match="3766.0"):
        hf._gate_inverted_pressure(
            Path("/tmp"), "/fake/hull.stl", _CONFIG, 2.4, 0.5, 0.2, 1.0,
            x_dict=_X_DICT, timeout=7200,
        )


def test_gate5_crashes_on_measuretool_failure(monkeypatch):
    """No measurement (status OK but max_pressure_pa None) is still a crash."""
    from hull_opt import high_fidelity as hf

    def _fake_run(*a, **kw):
        return {"max_pressure_pa": None, "fallback_pressure_pa": 3766.0,
                "status": "OK", "wall_time_s": 1.0,
                "details": "MeasureTool produced no output"}

    monkeypatch.setattr("hull_opt.sph_resistance.run_inverted_pressure", _fake_run)

    with pytest.raises(GateCrashError):
        hf._gate_inverted_pressure(
            Path("/tmp"), "/fake/hull.stl", _CONFIG, 2.4, 0.5, 0.2, 1.0,
            x_dict=_X_DICT, timeout=7200,
        )


def test_gate5_uses_measured_value(monkeypatch):
    """A real MeasureTool measurement still produces a PASS/FAIL verdict."""
    from hull_opt import high_fidelity as hf

    def _fake_run(*a, **kw):
        return {"max_pressure_pa": 1500.0, "fallback_pressure_pa": None,
                "status": "OK", "wall_time_s": 1.0,
                "details": "max deck pressure 1500.0 Pa"}

    monkeypatch.setattr("hull_opt.sph_resistance.run_inverted_pressure", _fake_run)

    passed, p, threshold, msg = hf._gate_inverted_pressure(
        Path("/tmp"), "/fake/hull.stl", _CONFIG, 2.4, 0.5, 0.2, 1.0,
        x_dict=_X_DICT, timeout=7200,
    )
    assert passed is True
    assert p == pytest.approx(1500.0)
    assert threshold == _CONFIG.validation.max_pressure_pa


# ── P0: OOM-survivability (particle preflight, gpu_lock threading) ─────────

def test_parse_particle_count_real_line():
    """The exact GenCase line observed on this machine parses to 1083974."""
    from hull_opt.sph_resistance import _parse_particle_count

    assert _parse_particle_count(_REAL_GENCASE_LINE) == 1083974


def test_parse_particle_count_handles_commas_and_garbage():
    """Comma-separated counts, plain counts and unparseable text."""
    from hull_opt.sph_resistance import _parse_particle_count

    assert _parse_particle_count("Total particles: 400,000") == 400000
    assert _parse_particle_count("Total particles: 123") == 123
    assert _parse_particle_count("Total particles:") == 0
    assert _parse_particle_count("Total particles: nope") == 0
    assert _parse_particle_count("") == 0
    assert _parse_particle_count(None) == 0


def test_run_sph_case_aborts_on_particle_cap(tmp_path, monkeypatch):
    """A GenCase layout above calibration.max_particles must FAIL before the
    solver is launched (never burn GPU time / RAM on an oversized case)."""
    import hull_opt.sph_resistance as sr

    (tmp_path / "case_towing.xml").write_text("<case/>")
    (tmp_path / "gencase.xml").write_text("<gencase/>")
    (tmp_path / "gencase.out").write_text(_REAL_GENCASE_LINE)
    fake_gencase = tmp_path / "GenCase_linux64"
    fake_gencase.write_bytes(b"fake")
    fake_solver = tmp_path / "DualSPHysics54.linux64"
    fake_solver.write_bytes(b"fake")

    calls = []

    def _fake_find_tool(name, ds_dir):
        return str(fake_gencase)

    def _fake_find_solver(ds_dir):
        return str(fake_solver), "gpu", {}

    def _fake_run(cmd, **kw):
        calls.append(list(cmd))
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(sr, "_find_tool", _fake_find_tool)
    monkeypatch.setattr(sr, "_find_solver_impl", _fake_find_solver)
    monkeypatch.setattr(sr.subprocess, "run", _fake_run)

    result = sr.run_sph_case(tmp_path, _CONFIG)

    assert result["status"] == "FAILED"
    assert "exceeds cap" in result["details"]
    assert result["np_particles"] == 1083974
    assert len(calls) == 1, "solver must never be launched"
    assert any("GenCase" in c for c in calls[0])


def test_run_sph_case_under_cap_launches_solver(tmp_path, monkeypatch):
    """A layout within max_particles proceeds: solver gets GenCase layout and
    solver.log redirection exists (no capture_output)."""
    import hull_opt.sph_resistance as sr

    (tmp_path / "case_towing.xml").write_text("<case/>")
    (tmp_path / "gencase.xml").write_text("<gencase/>")
    (tmp_path / "gencase.out").write_text("Total particles: 230,000")
    fake_gencase = tmp_path / "GenCase_linux64"
    fake_gencase.write_bytes(b"fake")
    fake_solver = tmp_path / "DualSPHysics54.linux64"
    fake_solver.write_bytes(b"fake")

    calls = []

    def _fake_find_tool(name, ds_dir):
        return str(fake_gencase)

    def _fake_find_solver(ds_dir):
        return str(fake_solver), "gpu", {}

    def _fake_run(cmd, **kw):
        calls.append((list(cmd), kw))
        if cmd[0].endswith("GenCase_linux64"):
            return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        if cmd[0] == "nvidia-smi":
            return type("P", (), {"returncode": 0, "stdout": "100000\n"})()
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(sr, "_find_tool", _fake_find_tool)
    monkeypatch.setattr(sr, "_find_solver_impl", _fake_find_solver)
    monkeypatch.setattr(sr.subprocess, "run", _fake_run)

    result = sr.run_sph_case(tmp_path, _CONFIG, timeout_s=60)

    assert result["status"] == "OK"
    assert result["np_particles"] == 230000
    solver_calls = [c for c in calls if "-dirout" in c[0]]
    assert len(solver_calls) == 1
    solver_cmd, solver_kw = solver_calls[0]
    assert "-gpu" in solver_cmd
    assert "capture_output" not in solver_kw, "solver output must go to solver.log"
    assert (tmp_path / "solver.log").exists()


def test_run_sph_case_skipped_check_failure_then_locked(tmp_path, monkeypatch):
    """gpu_lock held only around the solver: GenCase can run while the lock is
    busy, and a busy lock yields SKIPPED with the lock message."""
    import hull_opt.sph_resistance as sr

    (tmp_path / "case_towing.xml").write_text("<case/>")
    (tmp_path / "gencase.xml").write_text("<gencase/>")
    (tmp_path / "gencase.out").write_text("Total particles: 230,000")
    fake_gencase = tmp_path / "GenCase_linux64"
    fake_gencase.write_bytes(b"fake")
    fake_solver = tmp_path / "DualSPHysics54.linux64"
    fake_solver.write_bytes(b"fake")

    lock_path = tmp_path / "gpu.lock"
    held = open(lock_path, "w")
    import fcntl
    fcntl.flock(held, fcntl.LOCK_EX)

    def _fake_find_tool(name, ds_dir):
        return str(fake_gencase)

    def _fake_find_solver(ds_dir):
        return str(fake_solver), "gpu", {}

    def _fake_run(cmd, **kw):
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(sr, "_find_tool", _fake_find_tool)
    monkeypatch.setattr(sr, "_find_solver_impl", _fake_find_solver)
    monkeypatch.setattr(sr.subprocess, "run", _fake_run)

    result = sr.run_sph_case(tmp_path, _CONFIG, gpu_lock=str(lock_path))

    assert result["status"] == "SKIPPED"
    assert "GPU locked" in result["details"]
    assert result["np_particles"] == 0
    fcntl.flock(held, fcntl.LOCK_UN)
    held.close()


def test_run_sph_case_lock_wait_blocks_then_runs(tmp_path, monkeypatch):
    """lock_wait=True must block on a busy gpu.lock until it is released,
    then launch the solver — calibration waits out a running storm instead
    of being starved (P0.5; a SKIPPED calibration used to be marked failed)."""
    import hull_opt.sph_resistance as sr
    import fcntl

    (tmp_path / "case_towing.xml").write_text("<case/>")
    (tmp_path / "gencase.xml").write_text("<gencase/>")
    (tmp_path / "gencase.out").write_text("Total particles: 230,000")
    fake_gencase = tmp_path / "GenCase_linux64"
    fake_gencase.write_bytes(b"fake")
    fake_solver = tmp_path / "DualSPHysics54.linux64"
    fake_solver.write_bytes(b"fake")

    lock_path = tmp_path / "gpu.lock"
    held = open(lock_path, "w")
    fcntl.flock(held, fcntl.LOCK_EX)

    def _release_after_1s():
        time.sleep(1.0)
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()
    threading.Thread(target=_release_after_1s, daemon=True).start()

    def _fake_find_tool(name, ds_dir):
        return str(fake_gencase)

    def _fake_find_solver(ds_dir):
        return str(fake_solver), "gpu", {}

    def _fake_run(cmd, **kw):
        if cmd[0] == "nvidia-smi":
            return type("P", (), {"returncode": 0, "stdout": "6000\n", "stderr": ""})()
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(sr, "_find_tool", _fake_find_tool)
    monkeypatch.setattr(sr, "_find_solver_impl", _fake_find_solver)
    monkeypatch.setattr(sr.subprocess, "run", _fake_run)

    t0 = time.time()
    result = sr.run_sph_case(tmp_path, _CONFIG, timeout_s=60,
                             gpu_lock=str(lock_path), lock_wait=True)
    dt = time.time() - t0

    assert result["status"] == "OK"
    assert dt >= 0.9, f"must wait out the held lock, took {dt:.2f}s"


def test_run_towing_resistance_threads_lock_wait(tmp_path, monkeypatch):
    """run_towing_resistance must forward lock_wait to run_sph_case so
    calibration's blocking wait survives the wrapper."""
    import hull_opt.sph_resistance as sr

    captured = {}

    def _fake_sph_case(*a, **kw):
        captured.update(kw)
        return {"status": "OK", "wall_time_s": 1.0, "tool_kind": "gpu",
                "out_dir": "/x", "np_particles": 10, "details": "ok"}

    def _fake_build(*a, **kw):
        Path(a[0]).mkdir(parents=True, exist_ok=True)
        return Path(a[0]) / "case_towing.xml"

    def _fake_cf(*a, **kw):
        csv = tmp_path / "forces.csv"
        csv.write_text("Time;ForceFluidX\n0.0;3.0\n0.1;3.2\n")
        return str(csv)

    monkeypatch.setattr(sr, "run_sph_case", _fake_sph_case)
    monkeypatch.setattr(sr, "build_towing_case", _fake_build)
    monkeypatch.setattr(sr, "compute_forces", _fake_cf)
    monkeypatch.setattr(sr, "extract_towing_force",
                        lambda csv, **kw: 3.1)

    result = sr.run_towing_resistance(
        tmp_path, "/fake/hull.stl", _X_DICT, _CONFIG, 1.65,
        gpu_lock="/tmp/gpu.lock", lock_wait=True)

    assert result["status"] == "OK"
    assert captured.get("lock_wait") is True


def test_mid_fidelity_passes_gpu_lock(tmp_path, monkeypatch, caplog):
    """run_mid_fidelity_calibration must hand gpu_lock=<out>/gpu.lock to
    run_towing_resistance (P0.4; low_fidelity.py waits on the same path)."""
    from hull_opt.mid_fidelity import run_mid_fidelity_calibration

    cfg = replace(_CONFIG, paths=replace(_CONFIG.paths, output_dir=str(tmp_path)))
    captured = {}

    def _fake_generate(*a, **kw):
        return "/fake/stl", "/fake/sac", {"nabla": 0.1}, "/fake/hull.stl"

    def _fake_validate(*a, **kw):
        return True, "ok"

    def _fake_build(*a, **kw):
        Path(a[0]).mkdir(parents=True, exist_ok=True)
        return Path(a[0]) / "case_towing.xml"

    def _fake_preflight(*a, **kw):
        return True, []

    def _fake_tow(*a, **kw):
        captured.update(kw)
        return {"rt_n": None, "status": "FAILED", "wall_time_s": 3.4,
                "details": "solver exit 9: oom", "np_particles": 123456,
                "tool_kind": "gpu", "force_csv": None}

    monkeypatch.setattr("hull_opt.mid_fidelity.generate_hull", _fake_generate)
    monkeypatch.setattr("hull_opt.geometry_validator.validate_design_vector", _fake_validate)
    monkeypatch.setattr("hull_opt.sph_resistance.build_towing_case", _fake_build)
    monkeypatch.setattr("hull_opt.preflight.preflight_case", _fake_preflight)
    monkeypatch.setattr("hull_opt.sph_resistance.run_towing_resistance", _fake_tow)

    result = run_mid_fidelity_calibration(_RAW_DESIGN, 7, 3, cfg)

    assert result is None
    assert captured["gpu_lock"] == str(tmp_path / "gpu.lock")
    assert "<calib FAILED design=7 iter=3 rc=FAILED" in caplog.text
    assert "np=123456" in caplog.text
