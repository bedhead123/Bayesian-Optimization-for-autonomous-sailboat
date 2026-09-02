"""
DEPRECATED — OpenFOAM utility functions.

Moved from hull_opt.utils for Phase 1B cleanup. All OpenFOAM runner
functions, template writers, and force parsers are consolidated here.
These functions are no longer used by the pipeline (SPH backend is the
only calibration/validation path). They remain importable for existing
scripts and tests that may refer to them.
"""
import logging
import os
import re
import signal
import shlex
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from jinja2 import Template

logger = logging.getLogger(__name__)

_DECOMPOSE_TEMPLATE = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}

numberOfSubdomains {{ n_procs }};

method          scotch;
""")


def run_of_command(cmd: list[str], case_dir: Path,
                   of_env: str, timeout: int = 3600,
                   log_file: Optional[Path] = None) -> subprocess.CompletedProcess:
    env_check = subprocess.run(
        ["bash", "-c", f"source {of_env} 2>&1 && echo OF_ENV_OK"],
        capture_output=True, text=True, timeout=30,
    )
    if "OF_ENV_OK" not in env_check.stdout:
        logger.warning(f"OpenFOAM env source failed: {env_check.stderr[:200]}")
    case_dir = Path(case_dir).resolve()
    safe_dir = str(case_dir)
    escaped_cmd = ' '.join(shlex.quote(c) for c in cmd)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        full_cmd = (f"source {of_env} 2>/dev/null && set -o pipefail && "
                    f"{escaped_cmd} 2>&1 | tee {shlex.quote(str(log_file))}")
    else:
        full_cmd = f"source {of_env} 2>/dev/null && set -o pipefail && {escaped_cmd} 2>&1"
    return subprocess.run(
        ["bash", "-c", full_cmd],
        cwd=safe_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False
    )


def tail_log(log_file: Path, n_lines: int = 40) -> str:
    try:
        lines = Path(log_file).read_text(errors="replace").strip().splitlines()
    except OSError:
        return f"(no log at {log_file})"
    if not lines:
        return "(empty log)"
    return "\n".join(lines[-n_lines:])


def kill_orphaned_solvers(case_dir: Path) -> int:
    case_dir = Path(case_dir).resolve()
    out = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True
    ).stdout
    killed = 0
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid, args = parts
        args_lower = args.lower()
        if str(case_dir) not in args:
            continue
        if "interfoam" in args_lower or "mpirun" in args_lower:
            try:
                os.kill(int(pid), signal.SIGKILL)
                killed += 1
            except (OSError, ValueError):
                pass
    if killed:
        logger.warning(f"Killed {killed} orphaned solver process(es) for {case_dir.name}")
    return killed


def terminate_interfoam(case_dir: Path) -> int:
    case_dir = Path(case_dir).resolve()
    out = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True
    ).stdout
    signalled = 0
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid, args = parts
        args_lower = args.lower()
        if str(case_dir) not in args:
            continue
        if "interfoam" in args_lower or "mpirun" in args_lower:
            try:
                os.kill(int(pid), signal.SIGTERM)
                signalled += 1
            except (OSError, ValueError):
                pass
    if signalled:
        logger.info(f"Sent SIGTERM to {signalled} process(es) for {case_dir.name}")
    return signalled


def find_force_file(forces_dir: Path) -> Optional[Path]:
    if not forces_dir.exists():
        logger.debug(f"Forces directory not found: {forces_dir}")
        return None
    time_dirs = sorted(
        [d for d in forces_dir.iterdir() if d.is_dir()],
        key=lambda d: float(d.name)
    )
    if not time_dirs:
        logger.debug(f"No time subdirectories in {forces_dir}")
        return None
    latest = time_dirs[-1] / "force.dat"
    if latest.exists():
        return latest
    for d in reversed(time_dirs):
        candidate = d / "force.dat"
        if candidate.exists():
            return candidate
    logger.debug(f"No force.dat found in any time directory under {forces_dir}")
    return None


def read_force_trace(force_file: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if force_file is None or not Path(force_file).exists():
        return None
    lines = Path(force_file).read_text().strip().splitlines()
    data_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    if not data_lines:
        return None
    header = next((l for l in lines if l.strip().startswith("# Time")), "")
    has_total_col = "total_x" in header

    times = []
    fx_vals = []
    for line in data_lines:
        line = line.strip()
        m = re.search(r'\([-eE\d.\s]+\)\s*\([-eE\d.\s]+\)', line)
        if m:
            inner = m.group().split(') (')[0].lstrip('(')
            parts_inner = inner.split()
            if len(parts_inner) >= 1:
                try:
                    times.append(float(line.split()[0]))
                    fx_vals.append(abs(float(parts_inner[0])))
                except (ValueError, IndexError):
                    pass
                continue
        parts = line.split()
        if len(parts) >= 7:
            try:
                if has_total_col and len(parts) >= 10:
                    fx_vals.append(abs(float(parts[1])))
                    times.append(float(parts[0]))
                else:
                    fx_vals.append(abs(float(parts[1]) + float(parts[4])))
                    times.append(float(parts[0]))
            except (ValueError, IndexError):
                pass
    if not fx_vals or len(fx_vals) != len(times):
        return None
    return np.asarray(times, dtype=float), np.asarray(fx_vals, dtype=float)


def extract_openfoam_force(force_file: Path, tail_frac: float = 0.2,
                           window_s: Optional[float] = None) -> Optional[float]:
    trace = read_force_trace(force_file)
    if trace is None:
        return None
    times, fx_vals = trace

    if window_s is not None and window_s > 0 and times.size >= 10:
        t_span = float(times[-1] - times[0])
        if t_span >= 0.5 * window_s:
            mask = times >= times[-1] - window_s
            if int(mask.sum()) >= 10:
                result = float(np.mean(fx_vals[mask]))
                logger.debug(f"extract_openfoam_force: window_s={window_s}s "
                             f"({int(mask.sum())}/{len(fx_vals)} samples, "
                             f"t={times[-1]:.3f}s), mean={result:.4f} N")
                return result

    tail_frac = max(0.0, min(1.0, tail_frac))
    n_tail = max(1, int(round(len(fx_vals) * tail_frac)))
    result = float(np.mean(fx_vals[-n_tail:]))
    logger.debug(f"extract_openfoam_force: {n_tail}/{len(fx_vals)} samples averaged, "
                 f"tail mean={result:.4f} N (last={fx_vals[-1]:.4f} N)")
    return result


def write_decompose_par_dict(case_dir: Path, n_procs: int):
    out_path = Path(case_dir) / "system" / "decomposeParDict"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = _DECOMPOSE_TEMPLATE.render(n_procs=n_procs)
    out_path.write_text(content)
    logger.info(f"decomposeParDict written for {n_procs} processes")


def run_decompose_par(case_dir: Path, of_env: str, timeout: int = 300) -> bool:
    proc = run_of_command(
        ["decomposePar", "-case", str(case_dir), "-force"],
        case_dir, of_env, timeout=timeout,
        log_file=case_dir / "log.decomposePar",
    )
    if proc.returncode != 0:
        logger.error(f"decomposePar failed: {tail_log(case_dir / 'log.decomposePar')}")
        return False
    logger.info("decomposePar OK")
    return True


def run_reconstruct_par(case_dir: Path, of_env: str, timeout: int = 600) -> bool:
    proc_dirs = list(Path(case_dir).glob("processor*"))
    if not proc_dirs:
        logger.info("No processor directories found, skipping reconstructPar")
        return True
    proc = run_of_command(
        ["reconstructPar", "-case", str(case_dir), "-latestTime"],
        case_dir, of_env, timeout=timeout,
        log_file=case_dir / "log.reconstructPar",
    )
    if proc.returncode != 0:
        logger.warning(f"reconstructPar exited non-zero (rc={proc.returncode}): "
                       f"{tail_log(case_dir / 'log.reconstructPar')}")
        return False
    logger.info("reconstructPar OK")
    return True


def run_parallel_inter_foam(case_dir: Path, of_env: str, n_procs: int,
                            timeout: int = 14400) -> subprocess.CompletedProcess:
    if n_procs <= 1:
        return run_of_command(
            ["interFoam", "-case", str(case_dir)],
            case_dir, of_env, timeout=timeout,
            log_file=case_dir / "log.interFoam",
        )

    kill_orphaned_solvers(case_dir)
    write_decompose_par_dict(case_dir, n_procs)

    if not run_decompose_par(case_dir, of_env, timeout=min(timeout, 300)):
        logger.warning("decomposePar failed, falling back to serial interFoam")
        return run_of_command(
            ["interFoam", "-case", str(case_dir)],
            case_dir, of_env, timeout=timeout
        )

    logger.info(f"Running interFoam in parallel on {n_procs} cores...")
    proc = run_of_command(
        ["mpirun", "-np", str(n_procs), "--oversubscribe", "--allow-run-as-root",
         "interFoam", "-case", str(case_dir), "-parallel"],
        case_dir, of_env, timeout=timeout,
        log_file=case_dir / "log.interFoam",
    )
    if proc.returncode != 0:
        logger.error(f"interFoam (parallel) failed (rc={proc.returncode}):\n"
                     f"{tail_log(case_dir / 'log.interFoam', n_lines=60)}")

    if not run_reconstruct_par(case_dir, of_env, timeout=min(timeout, 600)):
        logger.error("reconstructPar failed — force extraction may rely on processor-local data")

    return proc


# ── Backward-compat: deprecated OpenFOAM case templates ────────────────────

def compute_domain(LWL: float, B: float, T: float,
                   T_total: float | None = None) -> dict:
    """DEPRECATED: OpenFOAM blockMesh domain computation."""
    T_total = T_total if T_total is not None and T_total > 0 else T
    xmin = 1.0 * LWL
    xmax = 2.5 * LWL
    ymax = 3.5 * B
    zmin = max(2.0 * T, T_total + 0.5)
    zmax = max(2.0 * T, 0.5)
    return {
        "xmin": xmin, "xmax": xmax, "ymax": ymax,
        "zmin": zmin, "zmax": zmax, "T_total": T_total,
        "span": xmin + xmax,
    }


def write_openfoam_case(case_dir, stl_path, speed_ms, LWL, B, T,
                        rho=1025.0, nu=1.19e-6, gravity=9.81,
                        mesh_levels=(2, 3), n_layers=3, solver="interFoam",
                        six_dof=False, end_time=10.0, delta_t=0.001,
                        write_interval=0.1, max_cells=500000, max_co=0.7,
                        max_alpha_co=0.5, mass=0.0, cg_z=-0.1,
                        Ixx=0.0, Iyy=0.0, Izz=0.0, initial_state=None,
                        T_total=None, keel_x=None, keel_chord=None,
                        keel_depth=None, keel_box_level=None,
                        keel_tip_box_level=None, keel_tip_height=0.35,
                        free_surface_refinement=False, fsr_z_half=0.0,
                        wave=None):
    """DEPRECATED — OpenFOAM case writer removed.
    
    Raises RuntimeError to indicate this functionality is no longer available.
    Use DualSPHysics templates (write_towing_case / write_inverted_case)
    instead.
    """
    raise RuntimeError(
        "write_openfoam_case has been removed. "
        "Use DualSPHysics templates from hull_opt.templates.dualsphysics instead."
    )
