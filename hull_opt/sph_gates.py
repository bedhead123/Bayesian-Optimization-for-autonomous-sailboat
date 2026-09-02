import logging
import math
import os
import re
import subprocess
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fcntl

from hull_opt.templates.dualsphysics import (DEF_MARGIN, STORM_XML,
                                             storm_domain, write_storm_case)

logger = logging.getLogger(__name__)

# Reference storm particle cap: above this the case would exceed the 6 GiB
# VRAM budget (~1.5 kB/particle) or never finish. ~330-440k at reference
# dp=0.06; dp=0.02 produced 9-12M and produced 1 frame in 30 min.
_STORM_MAX_PARTICLES = 500_000
# Adaptive-dp budgeting: pick dp so the expected layout lands here (19%
# headroom below the cap). See _estimate_storm_dp.
_STORM_TARGET_PARTICLES = 420_000
# Empirical fill fraction np ~= k * V_defbox / dp^3. Fitted to 6 real GenCase
# storm layouts (k = 0.620-0.661, mean 0.638); the upper edge keeps the
# estimate conservative against the cap.
_STORM_VOL_COEFF = 0.65
# Coarser than this the storm resolution is meaningless; cap stays 500k.
_STORM_MAX_DP = 0.10
# Only coarsen when the nominal-dp estimate approaches the cap (worst real
# fill fraction is k=0.661 vs the 0.65 estimate -> +1.7%): hulls below this
# keep dp_nominal exactly, so previously-passing storms are untouched.
_STORM_COARSEN_THRESHOLD = 480_000


def _storm_defbox_volume(LWL: float, B: float, T: float, Hs: float,
                         dp: float) -> float:
    """Definition-box volume (m^3) of a storm case: the domain from
    storm_domain plus DEF_MARGIN on every side (mirrors the XML template)."""
    dom = storm_domain(LWL, B, T, Hs, dp)
    dx = 2.0 * (dom["xdomain"] + DEF_MARGIN)
    dy = 2.0 * (dom["ydomain"] + DEF_MARGIN)
    dz = (dom["zmax"] + DEF_MARGIN) + (dom["zmin"] + DEF_MARGIN)
    return dx * dy * dz


def _estimate_storm_dp(LWL: float, B: float, T: float, Hs: float,
                       dp_nominal: float,
                       n_target: int = _STORM_TARGET_PARTICLES,
                       cap: int = _STORM_MAX_PARTICLES) -> float:
    """Design-adaptive reference-storm dp.

    Small hulls keep dp_nominal exactly (their layout is already ~380-440k).
    Oversized hulls — whose tank volume would trip the 500k cap at
    dp_nominal (observed 501,868 / 505,841 particles) — are coarsened just
    enough to land near n_target. Non-finite inputs fall back to dp_nominal
    (the post-GenCase cap check still guards those cases).
    """
    try:
        LWL, B, T, Hs, dp_nominal = (float(v)
                                     for v in (LWL, B, T, Hs, dp_nominal))
    except (TypeError, ValueError):
        return dp_nominal
    if not all(np.isfinite(v) for v in (LWL, B, T, Hs, dp_nominal)):
        return dp_nominal
    if dp_nominal <= 0.0:
        return dp_nominal
    T = max(T, 0.05)

    def _estimate(dp: float) -> float:
        return (_STORM_VOL_COEFF * _storm_defbox_volume(LWL, B, T, Hs, dp)
                / dp ** 3.0)

    # Leave hulls that already fit comfortably alone: zero behavior change
    # for previously-passing storms.
    if _estimate(dp_nominal) <= _STORM_COARSEN_THRESHOLD:
        return dp_nominal

    # Volume is dp-independent for T >= ~0.4 (zmin = 3*T dominates the 5*dp
    # floors), but iterate once so degenerate shallow hulls stay exact.
    dp_est = (_STORM_VOL_COEFF * _storm_defbox_volume(LWL, B, T, Hs,
                                                      dp_nominal)
              / n_target) ** (1.0 / 3.0)
    dp_est = min(max(dp_est, dp_nominal), _STORM_MAX_DP)
    dp_est = (_STORM_VOL_COEFF * _storm_defbox_volume(LWL, B, T, Hs, dp_est)
              / n_target) ** (1.0 / 3.0)
    dp_est = min(max(dp_est, dp_nominal), _STORM_MAX_DP)
    # Round UP so rounding never pushes the layout over the estimate.
    dp_eff = min(max(math.ceil(dp_est * 1000.0) / 1000.0, dp_nominal),
                 _STORM_MAX_DP)
    # Self-check: even the upper-edge estimate must clear the cap. If not,
    # fall back to the nominal dp and let the post-GenCase check abort as
    # before (no worse than the pre-fix behavior).
    if _estimate(dp_eff) > cap:
        return dp_nominal
    return dp_eff


@dataclass(frozen=True)
class ReferenceSettings:
    tool: str = "dualsphysics"
    sim_time_s: float = 10.0
    timeout_min: float = 30.0
    dp: float = 0.02
    dtout: float = 0.1


def load_reference_settings(config) -> ReferenceSettings:
    ref = getattr(config, "reference", None)
    if ref is None:
        return ReferenceSettings()
    return ReferenceSettings(
        tool=getattr(ref, "tool", "dualsphysics"),
        sim_time_s=getattr(ref, "sim_time_s", 10.0),
        timeout_min=getattr(ref, "timeout_min", 30.0),
        dp=getattr(ref, "dp", 0.02),
        dtout=getattr(ref, "dtout", 0.1),
    )


def find_solver(dualsphysics_dir: str) -> tuple[str, str, dict]:
    gpu_path = str(Path(dualsphysics_dir) / "bin" / "linux" / "DSGcc7" / "DualSPHysics54.linux64")
    if os.path.exists(gpu_path):
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(Path(gpu_path).parent)
        return (gpu_path, "gpu", env)
    cpu_path = str(Path(dualsphysics_dir) / "bin" / "linux" / "DualSPHysics5.4CPU_linux64")
    if os.path.exists(cpu_path):
        return (cpu_path, "cpu", os.environ.copy())
    raise FileNotFoundError(
        f"No DualSPHysics solver found (tried GPU:{gpu_path}, CPU:{cpu_path})"
    )


def _find_tool(tool_name: str, dualsphysics_dir: str) -> str:
    candidates = [
        str(Path(dualsphysics_dir) / "bin" / "linux" / tool_name),
        str(Path(dualsphysics_dir) / "bin" / "Linux" / tool_name),
        f"/home/anon/apps/DualSPHysics_v5.4/bin/linux/{tool_name}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


def _normalize_csv_col(col: str) -> str:
    return re.sub(r"[^a-z]", "", col.lower())


def _parse_measuretool_csv(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"max_accel_g": float("nan"), "max_pressure_pa": float("nan")}
    text = csv_path.read_text(errors="replace")
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("Time")]
    if not lines:
        return {"max_accel_g": float("nan"), "max_pressure_pa": float("nan")}
    header_line = None
    for l in text.splitlines():
        s = l.strip().lstrip("#")
        # MeasureTool v5.4 header: "Part;Time [s];Press_0 [Pa];Press_1 [Pa];..."
        # (one column per measurement point). Older/other headers start with
        # "Time". Match either.
        if s.startswith("Time") or (s.startswith("Part") and "Time" in s.split(";")[1]):
            header_line = s
            break
    if header_line is None:
        header_line = text.splitlines()[0].strip().lstrip("#")
    cols = [_normalize_csv_col(c) for c in header_line.split(";")]
    max_accel = 0.0
    max_pressure = 0.0
    for line in lines:
        parts = line.split(";")
        if len(parts) < len(cols):
            continue
        try:
            for i, col in enumerate(cols):
                val = float(parts[i].strip())
                if col.startswith("accel") or col.startswith("ace"):
                    max_accel = max(max_accel, abs(val))
                if "press" in col:
                    max_pressure = max(max_pressure, abs(val))
        except (ValueError, IndexError):
            continue
    accel_g = max_accel / 9.81 if max_accel > 0 else 0.0
    return {"max_accel_g": accel_g, "max_pressure_pa": max_pressure}


def _find_measuretool_csv(case_dir: Path) -> list[Path]:
    pattern = "*MeasureTool*.csv"
    return sorted(case_dir.glob(pattern))


def _run_measuretool(case_dir: Path, ds_dir: str) -> Optional[Path]:
    """Run MeasureTool on the solver output in case_dir, return CSV path."""
    data_dir = case_dir / "storm_out" / "data"
    if not data_dir.is_dir():
        data_dir = case_dir / "storm_out"
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
    tool = _find_tool("MeasureTool_linux64", ds_dir)
    if not os.path.exists(tool):
        return None
    # MeasureTool -savecsv <stem> writes one CSV per variable:
    # <stem>_Press.csv / _Rhop.csv / _Vel.csv (v5.4). The literal <stem>
    # file never exists, so check for the pressure CSV.
    stem = case_dir / "storm_out" / "MeasureTool.csv"
    cmd = [tool, "-dirdata", str(data_dir), "-filexml", str(xml_path),
           "-points", str(pts), "-savecsv", str(stem)]
    try:
        mt_proc = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=900, check=False)
        if mt_proc.returncode != 0:
            logger.warning("MeasureTool exit %d: %s", mt_proc.returncode,
                           (mt_proc.stderr or mt_proc.stdout or "")[-300:])
            return None
    except (subprocess.TimeoutExpired, OSError, PermissionError, FileNotFoundError):
        return None
    base = str(stem)
    if base.endswith(".csv"):
        base = base[:-4]
    press_csv = Path(base + "_Press.csv")
    if press_csv.exists():
        return press_csv
    for f in sorted((case_dir / "storm_out").glob("MeasureTool_*.csv")):
        if "_Press" in f.name:
            return f
    return None


def _check_capsize(case_dir: Path) -> bool:
    float_info_dir = case_dir / "floatinginfo"
    if not float_info_dir.is_dir():
        return False
    csv_files = sorted(float_info_dir.glob("*.csv"))
    if not csv_files:
        return False
    last_csv = csv_files[-1]
    try:
        data = np.genfromtxt(last_csv, delimiter=";", skip_header=1, names=True, invalid_raise=False)
        if data is None or len(data) == 0:
            return False
        if "orientation_x" in data.dtype.names:
            last_orient = float(data["orientation_x"][-1])
        elif "PosX" in data.dtype.names:
            logger.warning("check_capsize: found PosX column instead of orientation_x")
            return False
        else:
            return False
        return abs(last_orient) > 90.0
    except Exception:
        return False


def _storm_failed_result(wall_time_s, details) -> dict:
    return {
        "tool": "dualsphysics",
        "status": "FAILED",
        "max_accel_g": float("nan"),
        "max_pressure_pa": float("nan"),
        "final_orientation": [float("nan")] * 3,
        "capsized": None,
        "sim_time_s": 0.0,
        "wall_time_s": wall_time_s,
        "details": details,
    }


def run_reference_storm(design_vector, config, case_dir, design_id,
                        settings=None, stl_path=None, gpu_lock=None,
                        held_lock_fd=None) -> dict:
    if settings is None:
        settings = load_reference_settings(config)
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    design_id = str(design_id)
    stl_path = str(stl_path) if stl_path is not None else ""
    if not stl_path or not os.path.exists(stl_path):
        raise FileNotFoundError(
            f"run_reference_storm: STL path is empty or file not found "
            f"(stl_path={stl_path!r}). The caller must provide a valid STL path."
        )

    # held_lock_fd: the caller already opened+flocked gpu_lock (ReferenceRunner
    # thread). Re-opening the same file here would create a second file
    # description whose flock conflicts with the held one, falsely skipping
    # every storm ("GPU locked by ..." self-deadlock). Skip the lock block.
    lock_fd = None
    if gpu_lock is not None and held_lock_fd is None:
        try:
            lock_fd = open(gpu_lock, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, IOError):
            logger.info(f"GPU locked by {gpu_lock}, skipping {design_id}")
            return {
                "tool": "dualsphysics",
                "status": "SKIPPED",
                "max_accel_g": float("nan"),
                "max_pressure_pa": float("nan"),
                "final_orientation": [float("nan")] * 3,
                "capsized": False,
                "sim_time_s": settings.sim_time_s,
                "wall_time_s": 0.0,
                "details": f"GPU locked by {gpu_lock}",
            }

    try:
        wall_t0 = time.time()

        try:
            solver_path, tool_kind, env = find_solver(config.paths.dualsphysics_dir)
        except FileNotFoundError as e:
            return {
                "tool": "dualsphysics",
                "status": "FAILED",
                "max_accel_g": float("nan"),
                "max_pressure_pa": float("nan"),
                "final_orientation": [float("nan")] * 3,
                "capsized": False,
                "sim_time_s": settings.sim_time_s,
                "wall_time_s": time.time() - wall_t0,
                "details": str(e),
            }

        ds_dir = config.paths.dualsphysics_dir
        gencase_path = _find_tool("GenCase_linux64", ds_dir)
        measuretool_path = _find_tool("MeasureTool_linux64", ds_dir)

        try:
            from hull_opt.param_layer import design_vector_to_physical
            phys = design_vector_to_physical(design_vector, config)
            LWL = phys["LWL"]
            B = phys["BWL"]
            T_canoe = phys["T_canoe"]
            D_keel = phys["D_keel"]
        except (IndexError, TypeError):
            LWL = getattr(config.fixed, "LWL", 2.4) if hasattr(config, 'fixed') else 2.4
            B = 0.5
            T_canoe = 0.25
            D_keel = 1.0

        T_total = T_canoe + D_keel
        target_disp = getattr(config.fixed, "target_displacement", None)
        if hasattr(config.fixed, "rho_water") and target_disp is not None:
            mass = config.fixed.rho_water * target_disp
        else:
            mass = 300.0
        rho = config.fixed.rho_water if hasattr(config.fixed, "rho_water") else 1025.0

        try:
            from hull_opt.hydrostatics import compute_cg_z
            cg_z = compute_cg_z(phys, nabla=None, config=config)
        except Exception:
            cg_z = getattr(config.fixed, "cg_z", -0.2)

        ref_cfg = getattr(config, "reference", None)
        Hs = getattr(ref_cfg, "Hs", 0.25) if ref_cfg else 0.25
        Tp = getattr(ref_cfg, "Tp", 1.3) if ref_cfg else 1.3
        gamma = getattr(ref_cfg, "gamma", 3.3) if ref_cfg else 3.3

        # Design-adaptive dp: oversized hulls trip the 500k particle cap at
        # the nominal dp (observed 501,868 / 505,841 particles), so coarsen
        # the layout just enough to land near the 420k target. Small hulls
        # keep settings.dp unchanged.
        storm_dp = _estimate_storm_dp(LWL, B, T_total, Hs, settings.dp)
        if storm_dp != settings.dp:
            logger.info(
                f"reference storm design {design_id}: adaptive dp "
                f"{settings.dp} -> {storm_dp} (hull LWL={LWL:.2f} m "
                f"B={B:.2f} m T_total={T_total:.2f} m)"
            )

        write_storm_case(
            case_dir,
            hull_stl_path=stl_path,
            LWL=LWL,
            B=B,
            T=T_total,
            mass=mass,
            gravity=config.fixed.gravity,
            rho=rho,
            nu=config.fixed.nu_water,
            cg_z=cg_z,
            Hs=Hs,
            Tp=Tp,
            gamma=gamma,
            sim_time=settings.sim_time_s,
            dt_out=settings.dtout,
            dp=storm_dp,
            eb_coords=getattr(config.fixed, "electronics_bay", [0.0, 0.0, -0.05]),
        )

        xml_path = case_dir / "case_storm.xml"

        # DualSPHysics tools append ".xml" to the case name themselves, so pass
        # bare stems (see JSph.cpp: FileXml = DirCase + CaseName + ".xml").
        case_stem = xml_path.stem
        gc_stem = "gencase"
        gencase_cmd = [gencase_path, case_stem, gc_stem, "-save:bi"]
        # Log STL face count for diagnostics
        try:
            import trimesh
            _mesh = trimesh.load(stl_path, force='mesh')
            logger.info("GenCase input STL for design %s has %d faces",
                        design_id, len(_mesh.faces))
        except Exception:
            pass
        try:
            gc_result = subprocess.run(gencase_cmd, capture_output=True, text=True,
                                       timeout=600, check=False, cwd=str(case_dir))
        except subprocess.TimeoutExpired as _gc_timeout:
            _gc_stdout = (_gc_timeout.stdout or "")[-300:]
            _gc_stderr = (_gc_timeout.stderr or "")[-300:]
            logger.warning("GenCase timed out (600s):\n  stdout: %s\n  stderr: %s",
                           _gc_stdout.replace('\n', '\\n'),
                           _gc_stderr.replace('\n', '\\n'))
            return _storm_failed_result(time.time() - wall_t0,
                                        "GenCase timed out")
        if gc_result.returncode != 0:
            tail = (gc_result.stderr or gc_result.stdout or "")[-500:]
            logger.error("GenCase failed (exit %d): %s", gc_result.returncode, tail)
            return _storm_failed_result(time.time() - wall_t0,
                                        f"GenCase exit {gc_result.returncode}: {tail}")
        if not (case_dir / "gencase.xml").exists():
            return _storm_failed_result(time.time() - wall_t0,
                                        "GenCase produced no gencase.xml")

        # Particle-count preflight: abort before the solver launch if the
        # GenCase layout would blow the RAM/VRAM budget of this box
        # (mirrors run_sph_case in sph_resistance.py).
        np_particles = 0
        try:
            from hull_opt.sph_resistance import _parse_particle_count
            try:
                np_particles = _parse_particle_count(
                    (case_dir / "gencase.out").read_text(errors="replace"))
            except OSError:
                pass
            if not np_particles:
                np_particles = _parse_particle_count(gc_result.stdout)
        except ImportError:
            np_particles = 0
        if np_particles > _STORM_MAX_PARTICLES:
            logger.error(f"particle count {np_particles} exceeds cap "
                         f"{_STORM_MAX_PARTICLES} (gencase)")
            return _storm_failed_result(
                time.time() - wall_t0,
                f"particle count {np_particles} exceeds cap {_STORM_MAX_PARTICLES} (gencase)")

        timeout_s = int(settings.timeout_min * 60)
        solver_cmd = [solver_path, gc_stem, "-dirout", str(case_dir / "storm_out"), "-svres"]
        if tool_kind == "gpu":
            solver_cmd.append("-gpu")

        # VRAM guard: ~1.5 kB/particle estimate vs. free GPU memory
        # (mirrors run_sph_case in sph_resistance.py).
        if tool_kind == "gpu" and np_particles > 0:
            need_mib = np_particles * 1536.0 / 1024.0 / 1024.0
            try:
                smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=15, check=False,
                )
                free_mib = int(smi.stdout.strip().splitlines()[0].strip())
                if free_mib < need_mib:
                    logger.error(
                        f"VRAM guard: need ~{int(need_mib)} MiB for "
                        f"{np_particles} particles, only {free_mib} MiB free"
                    )
                    return _storm_failed_result(
                        time.time() - wall_t0,
                        f"VRAM guard: need ~{int(need_mib)} MiB for "
                        f"{np_particles} particles, only {free_mib} MiB free")
            except Exception as e:
                logger.debug("nvidia-smi unavailable (%s); continuing without "
                             "VRAM guard", e)
        try:
            proc = subprocess.run(solver_cmd, capture_output=True, text=True,
                                   timeout=timeout_s, check=False, env=env, cwd=str(case_dir))
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-500:]
                logger.warning(f"Solver exited with code {proc.returncode}: {tail}")
                wall_time = time.time() - wall_t0
                return _storm_failed_result(wall_time,
                                            f"solver exit {proc.returncode}: {tail}")
        except subprocess.TimeoutExpired:
            wall_time = time.time() - wall_t0
            return {
                "tool": "dualsphysics",
                "status": "TIMEOUT",
                "max_accel_g": float("nan"),
                "max_pressure_pa": float("nan"),
                "final_orientation": [float("nan")] * 3,
                "capsized": False,
                "sim_time_s": settings.sim_time_s,
                "wall_time_s": wall_time,
                "details": f"TIMEOUT after {timeout_s}s",
            }

        wall_time = time.time() - wall_t0
        mt_csv_path = case_dir / "storm_out" / "MeasureTool.csv"
        if not mt_csv_path.exists():
            csv_list = _find_measuretool_csv(case_dir / "storm_out")
            if csv_list:
                mt_csv_path = csv_list[0]
            else:
                run_mt = _run_measuretool(case_dir, ds_dir)
                if run_mt is not None:
                    mt_csv_path = run_mt

        parsed = _parse_measuretool_csv(mt_csv_path)
        capsized = _check_capsize(case_dir / "storm_out")

        float_info_dir = case_dir / "storm_out" / "floatinginfo"
        orientation = [0.0, 0.0, 0.0]
        if float_info_dir.is_dir():
            csv_files = sorted(float_info_dir.glob("*.csv"))
            if csv_files:
                try:
                    data = np.genfromtxt(csv_files[-1], delimiter=";", skip_header=1, names=True, invalid_raise=False)
                    if data is not None and len(data) > 0:
                        for i, axis in enumerate(["orientation_x", "orientation_y", "orientation_z"]):
                            if axis in data.dtype.names:
                                orientation[i] = float(data[axis][-1])
                except Exception:
                    pass

        result = {
            "tool": "dualsphysics",
            "status": "OK",
            "max_accel_g": parsed["max_accel_g"],
            "max_pressure_pa": parsed["max_pressure_pa"],
            "final_orientation": orientation,
            "capsized": capsized,
            "sim_time_s": settings.sim_time_s,
            "wall_time_s": wall_time,
            "details": f"ran {tool_kind} solver",
        }
        return result
    finally:
        if gpu_lock is not None and held_lock_fd is None and lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass
