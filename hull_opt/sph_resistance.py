"""
SPH calm-water towing resistance and inverted-deck-pressure gates
(DualSPHysics v5.4, classic <casedef> XML dialect).

Towing setup
------------
The calm-water tow fixes the hull at its design draft in a long tank and
drives a uniform current past it with the native <inout> open-boundary
mechanism (Tafuni et al. 2018; see examples/inletoutlet/07_CurrentHull in
the DualSPHysics distribution). This is the Galilean equivalent of towing
the hull: the hull sees the same relative flow while remaining fixed, which
avoids moving-hull stability issues and keeps the domain length manageable.
Total resistance Rt is the x-force the fluid exerts on the hull boundary
particles (motion-type 2, ComputeForces with -onlymk:<hull mk>, see
hull_boundary_mk), averaged over the tail of the run once the steady wave
pattern has developed. Flow is +x, so drag on the hull is -x and Rt is
reported positive.

Inverted setup
--------------
Gate 5 capsizes the hull (deck down, keel up, STL pre-rotated 180 degrees
about x by write_inverted_case) and lets it settle under its own mass. The
peak pressure on the deck is interpolated with MeasureTool at the
measure_points.txt locations written by write_inverted_case at the
equilibrium deck depth.

Runtime budget
--------------
Particles scale as dp^-3. At calibration cost (dp=0.05) a towing case is
~230k particles and ~20-40 min on the RTX 3050 GPU solver
(DualSPHysics54.linux64); dp=0.03 would be ~1.1M particles — too big
for this 6 GiB VRAM box. The inverted case is cheaper (no inlet/outlet
zones, shallower tank).
"""
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

from hull_opt.sph_gates import (
    ReferenceSettings,
    load_reference_settings,
    _find_tool,
    _parse_measuretool_csv,
)
from hull_opt.sph_gates import find_solver as _find_solver_impl

logger = logging.getLogger(__name__)

_DS_DEFAULT_DIR = "/home/anon/apps/DualSPHysics_v5.4"

# v5.4 classic dialect mk remapping (verified in gencase.out): GenCase adds
# +10 to every mkbound and +1 to every mkfluid when it writes the particle
# data. Templates declare the hull with <setmkbound mk="1">, so its ACTUAL
# mk in the BI4/XML particle data is 11. ComputeForces -onlymk filters on
# the actual mk; passing the template value would select the main fluid
# (actual mk 1) and match zero boundary particles.
_DS5_BOUND_MK_OFFSET = 10
_DS5_FLUID_MK_OFFSET = 1
_HULL_MKBOUND = 1


def hull_boundary_mk() -> int:
    """Actual (post-GenCase remap) mk of the hull boundary particles.

    Returns _DS5_BOUND_MK_OFFSET + _HULL_MKBOUND (= 11): the template's
    <setmkbound mk="1"> hull plus the v5.4 classic +10 bound offset.
    ComputeForces and any other tool that filters particle data by mk must
    use this value, not the template one.
    """
    return _DS5_BOUND_MK_OFFSET + _HULL_MKBOUND


def find_solver(config):
    """Resolve the DualSPHysics solver path from config.dualsphysics_dir.
    
    Wraps sph_gates.find_solver with a config-based interface.
    Returns (solver_path, tool_kind, env_dict) or raises FileNotFoundError.
    """
    ds_dir = getattr(config.paths, "dualsphysics_dir", _DS_DEFAULT_DIR)
    return _find_solver_impl(ds_dir)


def _resolve_val(config, key: str, default: float):
    """Pull a value from config.calibration with a validation fallback."""
    cal = getattr(config, "calibration", None)
    if cal is not None:
        val = getattr(cal, key, None)
        if val is not None:
            return val
    val_cfg = getattr(config, "validation", None)
    if val_cfg is not None:
        val = getattr(val_cfg, key, None)
        if val is not None:
            return val
    return default


_PARTICLES_RE = re.compile(r"Total particles:\s*([\d,]+)")


def _parse_particle_count(text: str) -> int:
    """Total particle count from a GenCase summary line, e.g.
    'Total particles: 1,083,974 (bound=66672 (fx=66672 mv=0 ft=0) fluid=1017302)'
    Returns 0 when unparseable (never raises)."""
    m = _PARTICLES_RE.search(text or "")
    if not m:
        return 0
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return 0


def _tail(path, n: int = 250) -> str:
    """Last n chars of a file, or '' if it cannot be read."""
    try:
        return (Path(path).read_text(errors="replace"))[-n:]
    except OSError:
        return ""


def build_towing_case(case_dir, stl_path, x_dict, config,
                      speed_ms=None, dp=None, sim_time=None, dt_out=None):
    """Write the calm-water towing case XML for a design.

    Signature mirrors run_towing_resistance: speed_ms is the 5th positional
    argument (mid_fidelity.py passes it positionally). Do NOT reorder.

    Config-derived defaults (calibration first, validation fallback):
    dp = calibration.sph_dp, sim_time = calibration.sph_sim_time;
    speed_ms defaults to the design target speed (config.fixed.target_speed_knots).
    Returns the case XML path.
    """
    from hull_opt.templates.dualsphysics import write_towing_case
    from hull_opt.utils import knots_to_ms

    if dp is None:
        dp = _resolve_val(config, "sph_dp", 0.05)
    if sim_time is None:
        sim_time = _resolve_val(config, "sph_sim_time", 6.0)
    if speed_ms is None:
        speed_ms = knots_to_ms(getattr(config.fixed, "target_speed_knots", 3.2))
    if dt_out is None:
        dt_out = max(0.02, min(0.1, sim_time / 50.0))

    rho = getattr(config.fixed, "rho_water", 1025.0)
    target_disp = getattr(config.fixed, "target_displacement", 0.145)
    mass = rho * target_disp

    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    return write_towing_case(
        case_dir,
        hull_stl_path=str(stl_path),
        LWL=float(x_dict["LWL"]),
        B=float(x_dict["BWL"]),
        T_canoe=float(x_dict["T_canoe"]),
        D_keel=float(x_dict["D_keel"]),
        speed_ms=float(speed_ms),
        dp=float(dp),
        sim_time=float(sim_time),
        dt_out=float(dt_out),
        mass=float(mass),
        rho=float(rho),
        gravity=getattr(config.fixed, "gravity", 9.81),
        eb_coords=getattr(config.fixed, "electronics_bay", [0.0, 0.0, -0.05]),
        keel_chord=float(x_dict.get("keel_chord", 0.0)),
    )


def run_sph_case(case_dir, config, xml_path=None, timeout_s=None, gpu_lock=None,
                 lock_wait=False) -> dict:
    """Run GenCase + the DualSPHysics solver on the case XML in case_dir.

    If xml_path is None, auto-discovers the first case_*.xml in case_dir.
    Returns {"status": OK|TIMEOUT|FAILED|SKIPPED, "wall_time_s", "tool_kind",
    "details", "out_dir", "np_particles"}. Never raises. The solver output
    (Part_XXXX.bi4) lands in <case_dir>/case_out/data.

    lock_wait=True makes the gpu_lock acquisition BLOCKING (fcntl.LOCK_EX)
    instead of the non-blocking skip: calibration must wait out a running
    storm rather than be silently starved (status SKIPPED).
    """
    case_dir = Path(case_dir)
    wall_t0 = time.time()
    result_base = {"wall_time_s": 0.0, "tool_kind": None,
                   "out_dir": str(case_dir / "case_out"), "np_particles": 0}

    if xml_path is None:
        xml_files = sorted(case_dir.glob("case_*.xml"))
        if not xml_files:
            return {"status": "FAILED", "details": f"no case_*.xml in {case_dir}", **result_base}
        xml_path = xml_files[0]
    else:
        xml_path = Path(xml_path)

    if timeout_s is None:
        settings = load_reference_settings(config)
        timeout_s = int(settings.timeout_min * 60)

    ds_dir = getattr(config.paths, "dualsphysics_dir", _DS_DEFAULT_DIR)
    try:
        solver_path, tool_kind, env = _find_solver_impl(ds_dir)
    except FileNotFoundError as e:
        return {"status": "FAILED", "details": str(e),
                "wall_time_s": time.time() - wall_t0, "tool_kind": None,
                "out_dir": result_base["out_dir"], "np_particles": 0}

    try:
        # GenCase: initial particle layout (v5.4). GenCase reads case_stem.xml
        # and outputs {gc_stem}.bi4 + {gc_stem}.xml (which has particles info).
        case_stem = Path(xml_path).stem
        gc_stem = "gencase"
        gencase_path = _find_tool("GenCase_linux64", ds_dir)
        try:
            gc_proc = subprocess.run(
                [gencase_path, case_stem, gc_stem, "-save:bi"],
                capture_output=True, text=True, timeout=600, check=False,
                cwd=str(case_dir),
            )
        except subprocess.TimeoutExpired as _gc_timeout:
            _gc_stdout = (_gc_timeout.stdout or "")[-300:]
            _gc_stderr = (_gc_timeout.stderr or "")[-300:]
            logger.warning("GenCase timed out (600s):\n  stdout: %s\n  stderr: %s",
                           _gc_stdout.replace('\n', '\\n'),
                           _gc_stderr.replace('\n', '\\n'))
            return {"status": "FAILED", "details": "GenCase timed out",
                    "wall_time_s": time.time() - wall_t0, "tool_kind": None,
                    "out_dir": str(case_dir / "case_out"), "np_particles": 0}
        if gc_proc.returncode != 0:
            tail = (gc_proc.stderr or gc_proc.stdout or "")[-250:]
            logger.error("GenCase failed (exit %d): %s", gc_proc.returncode, tail)
            return {"status": "FAILED",
                    "details": f"GenCase exit {gc_proc.returncode}: {tail}",
                    "wall_time_s": time.time() - wall_t0, "tool_kind": None,
                    "out_dir": str(case_dir / "case_out"), "np_particles": 0}
        if not (case_dir / "gencase.xml").exists():
            return {"status": "FAILED", "details": "GenCase produced no gencase.xml",
                    "wall_time_s": time.time() - wall_t0, "tool_kind": None,
                    "out_dir": str(case_dir / "case_out"), "np_particles": 0}

        # Particle-count preflight: abort before the solver launch if the
        # GenCase layout would blow the RAM/VRAM budget of this box.
        np_particles = 0
        try:
            np_particles = _parse_particle_count((case_dir / "gencase.out").read_text(errors="replace"))
        except OSError:
            pass
        if not np_particles:
            np_particles = _parse_particle_count(gc_proc.stdout)
        max_part = int(_resolve_val(config, "max_particles", 400000))
        if np_particles > max_part:
            logger.error(f"particle count {np_particles} exceeds cap {max_part} (gencase)")
            return {"status": "FAILED",
                    "details": f"particle count {np_particles} exceeds cap {max_part} (gencase)",
                    "wall_time_s": time.time() - wall_t0, "tool_kind": tool_kind,
                    "out_dir": str(case_dir / "case_out"), "np_particles": np_particles}

        # Solver reads the GenCase-output XML (which contains case.execution.particles)
        out_dir = case_dir / "case_out"
        solver_cmd = [solver_path, gc_stem, "-dirout", str(out_dir), "-svres"]
        if tool_kind == "gpu":
            solver_cmd.append("-gpu")

        # VRAM guard: ~1.5 kB/particle estimate vs. free GPU memory.
        if tool_kind == "gpu" and np_particles > 0:
            need_mib = np_particles * 1536.0 / 1024.0 / 1024.0
            try:
                smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=15, check=False,
                )
                free_mib = int(smi.stdout.strip().splitlines()[0].strip())
                if free_mib < need_mib:
                    logger.error(
                        f"VRAM guard: need ~{int(need_mib)} MiB for {np_particles} particles, "
                        f"only {free_mib} MiB free"
                    )
                    return {"status": "FAILED",
                            "details": f"VRAM guard: need ~{int(need_mib)} MiB for "
                                       f"{np_particles} particles, only {free_mib} MiB free",
                            "wall_time_s": time.time() - wall_t0, "tool_kind": tool_kind,
                            "out_dir": str(out_dir), "np_particles": np_particles}
            except Exception as e:
                logger.debug("nvidia-smi unavailable (%s); continuing without VRAM guard", e)

        # gpu_lock is held ONLY around the solver run (GenCase and
        # ComputeForces run outside it).
        lock_fd = None
        if gpu_lock is not None:
            import fcntl
            try:
                lock_fd = open(gpu_lock, "w")
                if lock_wait:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                else:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, IOError):
                return {"status": "SKIPPED", "details": f"GPU locked by {gpu_lock}", **result_base}

        solver_log_path = case_dir / "solver.log"
        logger.debug("launching solver %s (log: %s)", solver_cmd, solver_log_path)
        try:
            solver_log = open(solver_log_path, "w")
            try:
                proc = subprocess.run(solver_cmd, stdout=solver_log,
                                      stderr=subprocess.STDOUT,
                                      timeout=timeout_s, check=False, env=env,
                                      cwd=str(case_dir))
            finally:
                solver_log.close()
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT", "details": f"TIMEOUT after {timeout_s}s",
                    "wall_time_s": time.time() - wall_t0, "tool_kind": tool_kind,
                    "out_dir": str(out_dir), "np_particles": np_particles}
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception:
                    pass

        wall_time = time.time() - wall_t0
        if proc.returncode != 0:
            tail = _tail(solver_log_path, 250)
            logger.warning("Solver exited %d: %s", proc.returncode, tail)
            return {"status": "FAILED", "details": f"solver exit {proc.returncode}: {tail}",
                    "wall_time_s": wall_time, "tool_kind": tool_kind,
                    "out_dir": str(out_dir), "np_particles": np_particles}
        return {"status": "OK", "details": f"ran {tool_kind} solver", "wall_time_s": wall_time,
                "tool_kind": tool_kind, "out_dir": str(out_dir), "np_particles": np_particles}
    finally:
        pass


def compute_forces(case_dir, onlymk=None) -> Optional[Path]:
    """Run ComputeForces on the solver output and return the forces CSV path.

    ComputeForces reads the saved BI4 particle files in <case_dir>/case_out/data
    (written by the solver with -svres) plus the case XML for the mk mapping,
    and computes the fluid force on the hull boundary particles:
        ComputeForces_linux64 -dirdata <data> -filexml <xml> -onlymk:<mk> \\
                              -viscoauto -savecsv forces.csv
    onlymk is the ACTUAL mkbound value in the particle data (post-GenCase
    remap): the hull is actual mk 11 (template mk 1 + v5.4 classic bound
    offset 10, see hull_boundary_mk); walls are actual mk 10. The default
    None resolves to the hull mk. Returns None if the particle data is
    missing or the tool fails.
    """
    if onlymk is None:
        onlymk = hull_boundary_mk()
    case_dir = Path(case_dir)
    data_dir = case_dir / "case_out" / "data"
    if not data_dir.is_dir():
        data_dir = case_dir / "case_out"
    if not sorted(data_dir.glob("Part_*.bi4")):
        logger.warning("No Part_*.bi4 in %s", data_dir)
        return None

    # Use GenCase output XML (has mk mapping); fallback to case XML
    xml_path = case_dir / "gencase.xml"
    if not xml_path.exists():
        xml_files = sorted(case_dir.glob("case_*.xml"))
        if not xml_files:
            return None
        xml_path = xml_files[0]

    # Fabric: try config dir first, then default, then local bin — not hard-coded single path
    ds_candidates = [_DS_DEFAULT_DIR, "./bin/dualsphysics/5.4", str(Path(__file__).parent.parent / "bin/dualsphysics/5.4")]
    tool = None
    for d in ds_candidates:
        cand = _find_tool("ComputeForces_linux64", d)
        if os.path.exists(cand):
            tool = cand
            break
    if tool is None:
        tool = _find_tool("ComputeForces_linux64", _DS_DEFAULT_DIR)
    if not os.path.exists(tool):
        logger.warning("ComputeForces_linux64 not found at %s", tool)
        return None

    csv_path = case_dir / "forces.csv"
    cmd = [tool, "-dirdata", str(data_dir), "-filexml", str(xml_path),
           f"-onlymk:{onlymk}", "-viscoauto", "-savecsv", str(csv_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
        if proc.returncode != 0:
            logger.warning("ComputeForces exit %d: %s", proc.returncode,
                           (proc.stderr or proc.stdout or "")[-300:])
    except subprocess.TimeoutExpired:
        logger.warning("ComputeForces timed out")
        return None
    return csv_path if csv_path.exists() else None


def read_force_trace_csv(forces_csv) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Parse a v5.4 ComputeForces CSV into a (times, |ForceFluidX|) trace.

    Returns None on any parse problem. Reuses the same column-key
    normalisation as extract_towing_force so both stay consistent.
    """
    try:
        with open(forces_csv) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        if len(lines) < 2:
            return None
        header = lines[0].lstrip("#")
        sep = ";" if ";" in header else ","
        cols_raw = [c.strip().lower() for c in header.split(sep)]

        def _col_key(c):
            return "".join(ch for ch in c.split("[")[0].split("(")[0] if ch.isalnum()).lower()

        cols_key = [_col_key(c) for c in cols_raw]
        fx_idx = None
        time_idx = None
        for i, ck in enumerate(cols_key):
            if ck == "forcefluidx" or ck.startswith("forcefluidx"):
                fx_idx = i
            if ck == "time" or ck.startswith("time"):
                time_idx = i
        if fx_idx is None:
            return None
        rows = []
        for line in lines[1:]:
            parts = line.split(sep)
            try:
                t = float(parts[time_idx]) if time_idx is not None else 0.0
                fx = float(parts[fx_idx])
            except (ValueError, IndexError):
                continue
            rows.append((t, fx))
        if not rows:
            return None
        times = np.array([r[0] for r in rows])
        fxs = np.abs(np.array([r[1] for r in rows]))
        return times, fxs
    except Exception as e:
        logger.warning("read_force_trace_csv failed: %s", e)
        return None


def read_force_tail_stats(csv_path, window_s=None, tail_frac=0.2) -> Optional[dict]:
    """Read the tail-window mean of |Fx| and |Fz| from a ComputeForces CSV.

    Returns {"t_end", "fx_mean", "fz_mean"} or None on parse failure.
    The Fz mean is the vertical-force sanity signal: on a clean tow it must
    stay within a fraction of the displacement buoyancy. Bug #157's
    contaminated runs show |Fz| climbing to 4x full-submersion buoyancy
    while Fx looks deceptively steady.
    """
    try:
        with open(csv_path) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        if len(lines) < 2:
            return None
        header = lines[0].lstrip("#")
        sep = ";" if ";" in header else ","
        cols_key = [
            "".join(ch for ch in c.split("[")[0].split("(")[0] if ch.isalnum()).lower()
            for c in header.split(sep)
        ]
        fx_idx = fz_idx = time_idx = None
        for i, ck in enumerate(cols_key):
            if ck == "forcefluidx" or ck.startswith("forcefluidx"):
                fx_idx = i
            elif ck == "forcefluidz" or ck.startswith("forcefluidz"):
                fz_idx = i
            if ck == "time" or ck.startswith("time"):
                time_idx = i
        if fx_idx is None:
            return None
        rows = []
        for line in lines[1:]:
            parts = line.split(sep)
            try:
                t = float(parts[time_idx]) if time_idx is not None else 0.0
                fx = float(parts[fx_idx])
                fz = float(parts[fz_idx]) if fz_idx is not None else 0.0
            except (ValueError, IndexError):
                continue
            rows.append((t, fx, fz))
        if not rows:
            return None
        times = np.array([r[0] for r in rows])
        if window_s is not None and time_idx is not None and len(times) > 1:
            mask = times >= float(times[-1]) - window_s
        else:
            n_tail = max(1, int(round(len(rows) * tail_frac)))
            mask = np.zeros(len(rows), dtype=bool)
            mask[-n_tail:] = True
        return {
            "t_end": float(times[-1]),
            "fx_mean": float(np.mean(np.abs(np.array([r[1] for r in rows]))[mask])),
            "fz_mean": float(np.mean(np.abs(np.array([r[2] for r in rows]))[mask])),
        }
    except Exception as e:
        logger.warning("read_force_tail_stats failed: %s", e)
        return None


def extract_towing_force(forces_csv, tail_frac=0.2, window_s=None) -> Optional[float]:
    """Extract total hull resistance (N) from a v5.4 ComputeForces CSV.

    v5.4 CSV layout: header 'Time;ForceFluidX;ForceFluidY;ForceFluidZ;
    WeightX;...;ForceTotalX;...' (semicolon-separated by default). For a
    fixed hull Weight=0 so ForceFluidX is the drag. The hull is fixed and
    the current is +x, so drag is -x: Rt is the sign-corrected (positive)
    tail-window mean of ForceFluidX. Returns None on any parse problem.
    """
    try:
        with open(forces_csv) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        if len(lines) < 2:
            return None
        header = lines[0].lstrip("#")
        sep = ";" if ";" in header else ","
        cols_raw = [c.strip().lower() for c in header.split(sep)]

        def _col_key(c):
            """Normalise a ComputeForces column header to a canonical key.
            Strips units suffix after '[' or '(', removes all non-alnum chars,
            then lowercases.  'ForceFluid.x [N]' -> 'forcefluidx'."""
            return "".join(ch for ch in c.split("[")[0].split("(")[0] if ch.isalnum()).lower()

        cols_key = [_col_key(c) for c in cols_raw]
        fx_idx = None
        time_idx = None
        for i, ck in enumerate(cols_key):
            if ck == "forcefluidx" or ck.startswith("forcefluidx"):
                fx_idx = i
            if ck == "time" or ck.startswith("time"):
                time_idx = i
        if fx_idx is None:
            logger.warning("No ForceFluidX column in %s (cols_key: %s)", forces_csv, cols_key)
            return None
        rows = []
        for line in lines[1:]:
            parts = line.split(sep)
            try:
                t = float(parts[time_idx]) if time_idx is not None else 0.0
                fx = float(parts[fx_idx])
            except (ValueError, IndexError):
                continue
            rows.append((t, fx))
        if not rows:
            return None
        times = np.array([r[0] for r in rows])
        fxs = np.array([r[1] for r in rows])
        if window_s is not None and time_idx is not None and len(times) > 1:
            t_max = float(times[-1])
            mask = times >= t_max - window_s
        else:
            n_tail = max(1, int(round(len(rows) * tail_frac)))
            mask = np.zeros(len(rows), dtype=bool)
            mask[-n_tail:] = True
        mean_fx = float(np.mean(fxs[mask]))
        return abs(mean_fx)
    except Exception as e:
        logger.warning("extract_towing_force failed: %s", e)
        return None


def _run_measuretool(case_dir, out_name="measuretool.csv"):
    """Run MeasureTool with the case's measure_points.txt on the solver output."""
    case_dir = Path(case_dir)
    data_dir = case_dir / "case_out" / "data"
    if not data_dir.is_dir():
        data_dir = case_dir / "case_out"
    pts = case_dir / "measure_points.txt"
    xml_path = case_dir / "gencase.xml"
    if not xml_path.exists():
        xml_files = sorted(case_dir.glob("case_*.xml"))
        if not xml_files:
            return None
        xml_path = xml_files[0]
    bi4_files = sorted(data_dir.glob("Part_*.bi4"))
    if not pts.exists() or not bi4_files:
        return None
    ds_candidates = [_DS_DEFAULT_DIR, "./bin/dualsphysics/5.4", str(Path(__file__).parent.parent / "bin/dualsphysics/5.4")]
    tool = None
    for d in ds_candidates:
        cand = _find_tool("MeasureTool_linux64", d)
        if os.path.exists(cand):
            tool = cand
            break
    if tool is None:
        tool = _find_tool("MeasureTool_linux64", _DS_DEFAULT_DIR)
    if not os.path.exists(tool):
        logger.warning("MeasureTool_linux64 not found at %s", tool)
        return None
    # MeasureTool -savecsv <stem> writes one CSV per variable:
    # <stem>_Press.csv / _Rhop.csv / _Vel.csv (v5.4). The literal <stem>
    # file never exists; the pressure CSV is what the gates need.
    stem = case_dir / out_name
    cmd = [tool, "-dirdata", str(data_dir), "-filexml", str(xml_path),
           "-points", str(pts), "-savecsv", str(stem)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
        if proc.returncode != 0:
            logger.warning("MeasureTool exit %d: %s", proc.returncode,
                           (proc.stderr or proc.stdout or "")[-300:])
            return None
    except (subprocess.TimeoutExpired, OSError):
        return None
    # MeasureTool appends the variable name to the stem MINUS its extension:
    # -savecsv measuretool.csv -> measuretool_Press.csv (v5.4).
    base = str(stem)
    if base.endswith(".csv"):
        base = base[:-4]
    press_csv = Path(base + "_Press.csv")
    if press_csv.exists():
        return press_csv
    for f in sorted(case_dir.glob(f"{Path(out_name).stem}_*.csv")):
        if "_Press" in f.name:
            return f
    return None


def _max_pressure_any_column(csv_path):
    """Max pressure column value from a MeasureTool CSV, matching Press/Pressure."""
    try:
        with open(csv_path) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        header = None
        for l in lines:
            if l.startswith("#Time") or l.startswith("Time"):
                header = l.lstrip("#")
                break
        if header is None:
            header = lines[0].lstrip("#")
        sep = ";" if ";" in header else ","
        cols = [c.strip().lower() for c in header.split(sep)]
        idxs = [i for i, c in enumerate(cols) if "press" in c]
        if not idxs:
            return None
        vals = []
        for line in lines[1:]:
            parts = line.split(sep)
            for i in idxs:
                try:
                    vals.append(abs(float(parts[i])))
                except (ValueError, IndexError):
                    continue
        return max(vals) if vals else None
    except Exception:
        return None


def parse_measuretool_pressure(case_dir):
    """Find and parse the MeasureTool CSV in case_dir/sph_out/.

    Returns max_pressure_pa (float) or None if the CSV is missing or
    unparseable. Uses _parse_measuretool_csv from sph_gates with a
    _max_pressure_any_column fallback.
    """
    try:
        case_dir = Path(case_dir)
        mt_csv = case_dir / "sph_out" / "MeasureTool.csv"
        if not mt_csv.exists():
            csv_list = sorted(case_dir.glob("**/*MeasureTool*.csv"))
            if csv_list:
                mt_csv = csv_list[0]
            else:
                return None
        parsed = _parse_measuretool_csv(mt_csv)
        p = parsed.get("max_pressure_pa")
        if p is None or (isinstance(p, float) and (np.isnan(p) or p == 0.0)):
            p = _max_pressure_any_column(mt_csv)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return None
        return float(p)
    except Exception as e:
        logger.warning("parse_measuretool_pressure: %s", e)
        return None


def _exit_code_from_details(details: str) -> str:
    """Exit code from a run_sph_case details string, or '-' when absent."""
    m = re.search(r"exit (\d+)", details or "")
    return m.group(1) if m else "-"


def run_towing_resistance(case_dir, stl_path, x_dict, config, speed_ms,
                          dp=None, sim_time=None, timeout_s=None, gpu_lock=None,
                          lock_wait=False, window_s=None, min_sim_time=None,
                          max_fz_n=None) -> dict:
    """Full towing gate: build case -> run solver -> ComputeForces -> extract Rt.

    Returns {"rt_n": float|None, "status", "wall_time_s", "tool_kind",
    "force_csv", "details", "np_particles"}. rt_n is None on any failure;
    never raises. lock_wait=True waits (blocking) on gpu_lock instead of
    returning SKIPPED when the GPU is busy.

    Guards (Bug #157 follow-up):
    - window_s: average |Fx| only over the last window_s seconds (steady
      state) instead of the default tail-20%.
    - min_sim_time: reject the run if the force trace ends before this
      simulation time (unconverged tow).
    - max_fz_n: reject the run if the tail-window mean |Fz| exceeds this
      bound. A fixed hull whose vertical fluid force approaches or exceeds
      the full-submersion buoyancy (rho*g*nabla) is measuring tank
      pressurization artifacts, not drag.
    """
    case_dir = Path(case_dir)
    wall_t0 = time.time()
    result = {"rt_n": None, "status": "FAILED", "wall_time_s": 0.0,
              "tool_kind": None, "force_csv": None, "details": "",
              "np_particles": 0}
    try:
        xml_path = build_towing_case(case_dir, stl_path, x_dict, config,
                                     dp=dp, sim_time=sim_time, speed_ms=speed_ms)
        run = run_sph_case(case_dir, config, timeout_s=timeout_s,
                           gpu_lock=gpu_lock, lock_wait=lock_wait)
        result["status"] = run["status"]
        result["tool_kind"] = run["tool_kind"]
        result["wall_time_s"] = time.time() - wall_t0
        result["details"] = run["details"]
        result["np_particles"] = run.get("np_particles", 0)
        if run["status"] != "OK":
            rc = "TIMEOUT" if run["status"] == "TIMEOUT" else _exit_code_from_details(run["details"])
            sim = sim_time if sim_time is not None else _resolve_val(config, "sph_sim_time", 6.0)
            logger.warning(
                f"<calib FAILED rc={rc} t={result['wall_time_s']:.1f}/{sim:.1f}s "
                f"np={result['np_particles']}>"
            )
            return result
        csv_path = compute_forces(case_dir)
        if csv_path is None:
            result["details"] = "ComputeForces produced no output"
            return result
        result["force_csv"] = str(csv_path)
        stats = read_force_tail_stats(csv_path, window_s=window_s)
        if stats is None:
            result["details"] = "force CSV parsed but no valid ForceFluidX"
            return result
        if min_sim_time is not None and stats["t_end"] < min_sim_time:
            result["details"] = (
                f"force trace ends at t={stats['t_end']:.2f}s < "
                f"min_sim_time={min_sim_time:.2f}s (unconverged tow)"
            )
            logger.warning(f"<calib FAILED reason='{result['details']}'>")
            return result
        if max_fz_n is not None and stats["fz_mean"] > max_fz_n:
            result["details"] = (
                f"Fz sanity: tail |Fz|={stats['fz_mean']:.1f} N > "
                f"{max_fz_n:.1f} N (tank pressurization artifact, not drag)"
            )
            logger.warning(f"<calib FAILED reason='{result['details']}'>")
            return result
        rt = stats["fx_mean"]
        result["rt_n"] = rt
        if rt is not None:
            result["status"] = "OK"
            fz_note = f" Fz={stats['fz_mean']:.1f} N" if max_fz_n is not None else ""
            result["details"] = f"Rt={rt:.3f} N{fz_note}"
        else:
            result["details"] = "force CSV parsed but no valid ForceFluidX"
        return result
    except Exception as e:
        logger.exception("run_towing_resistance failed")
        result["details"] = f"exception: {e}"
        result["wall_time_s"] = time.time() - wall_t0
        return result


def run_inverted_pressure(case_dir, stl_path, x_dict, config,
                          timeout_s=None, gpu_lock=None) -> dict:
    """Inverted self-righting gate: build case -> run -> MeasureTool max pressure.

    Returns {"max_pressure_pa": float|None, "fallback_pressure_pa": float|None,
    "status", "wall_time_s", "details"}.

    max_pressure_pa is ONLY ever a real MeasureTool measurement. If the
    solver fails (status != OK) or MeasureTool produces no measurement, an
    analytic hydrostatic estimate (rho*g*deck-depth-below-surface at the
    equilibrium draft) is returned under fallback_pressure_pa with
    max_pressure_pa = None, so callers can never mistake a fabricated value
    for a measurement (Gate 5 previously PASSed on the fallback).
    """
    from hull_opt.templates.dualsphysics import write_inverted_case

    case_dir = Path(case_dir)
    wall_t0 = time.time()
    rho = getattr(config.fixed, "rho_water", 1025.0)
    g = getattr(config.fixed, "gravity", 9.81)
    result = {"max_pressure_pa": None, "fallback_pressure_pa": None,
              "status": "FAILED", "wall_time_s": 0.0, "details": ""}

    def _analytic_fallback():
        pts = case_dir / "measure_points.txt"
        try:
            lines = [l for l in pts.read_text().splitlines() if l.strip()][1:]
            depths = [abs(float(l.split()[2])) for l in lines if len(l.split()) >= 3]
            if depths:
                return rho * g * float(np.mean(depths))
        except Exception:
            pass
        return rho * g * float(x_dict.get("T_canoe", 0.25))

    try:
        dp = getattr(config.validation, "sph_inverted_dp", 0.03)
        sim_time = getattr(config.validation, "sph_inverted_sim_time", 6.0)
        dt_out = max(0.02, min(0.1, sim_time / 50.0))
        mass = rho * getattr(config.fixed, "target_displacement", 0.145)
        write_inverted_case(
            case_dir,
            hull_stl_path=str(stl_path),
            LWL=float(x_dict["LWL"]),
            B=float(x_dict["BWL"]),
            T_canoe=float(x_dict["T_canoe"]),
            D_keel=float(x_dict["D_keel"]),
            mass=float(mass),
            dp=float(dp),
            sim_time=float(sim_time),
            dt_out=float(dt_out),
            rho=float(rho),
            gravity=g,
            eb_coords=getattr(config.fixed, "electronics_bay", [0.0, 0.0, -0.05]),
            keel_chord=float(x_dict.get("keel_chord", 0.0)),
        )
        run = run_sph_case(case_dir, config, timeout_s=timeout_s, gpu_lock=gpu_lock)
        result["status"] = run["status"]
        result["wall_time_s"] = time.time() - wall_t0
        result["details"] = run["details"]
        if run["status"] != "OK":
            result["fallback_pressure_pa"] = _analytic_fallback()
            result["details"] += f" | analytic fallback (unverified, status != OK)"
            return result
        mt_csv = _run_measuretool(case_dir)
        if mt_csv is None:
            result["fallback_pressure_pa"] = _analytic_fallback()
            result["details"] = "MeasureTool produced no output | analytic fallback (unverified)"
            return result
        parsed = _parse_measuretool_csv(mt_csv)
        p = parsed.get("max_pressure_pa")
        if p is None or (isinstance(p, float) and (np.isnan(p) or p == 0.0)):
            p = _max_pressure_any_column(mt_csv)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            result["fallback_pressure_pa"] = _analytic_fallback()
            result["details"] = "no pressure column in MeasureTool CSV | analytic fallback (unverified)"
        else:
            result["details"] = f"max deck pressure {p:.1f} Pa"
            result["max_pressure_pa"] = float(p)
        result["status"] = "OK"
        return result
    except Exception as e:
        logger.exception("run_inverted_pressure failed")
        result["details"] = f"exception: {e}"
        result["wall_time_s"] = time.time() - wall_t0
        return result
