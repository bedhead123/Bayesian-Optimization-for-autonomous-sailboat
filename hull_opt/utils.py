import os
import logging
import subprocess
import shlex
import psutil
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def check_external_tools(config) -> list[str]:
    missing = []
    of_env = config.paths.openfoam_env
    if os.path.exists(of_env):
        test_cmd = f"source {of_env} && which interFoam"
        try:
            result = subprocess.run(["bash", "-c", test_cmd], capture_output=True,
                                    text=True, timeout=30)
            if result.returncode != 0:
                missing.append("OpenFOAM (interFoam not found after sourcing)")
        except FileNotFoundError:
            missing.append("OpenFOAM (bash not found)")
    else:
        missing.append(f"OpenFOAM env file not found: {of_env}")

    ds_dir = config.paths.dualsphysics_dir
    ds_candidates = [
        Path(ds_dir) / "bin" / "linux" / "DualSPHysics5.4_linux64",
        Path(ds_dir) / "bin" / "linux" / "DualSPHysics",
        Path(ds_dir) / "bin" / "Linux" / "DualSPHysics5.4_linux64",
        Path(ds_dir) / "bin" / "Linux" / "DualSPHysics",
    ]
    ds_found = any(c.exists() for c in ds_candidates)
    if not ds_found:
        missing.append(f"DualSPHysics binary not found in {ds_dir}")

    return missing


def run_of_command(cmd: list[str], case_dir: Path,
                   of_env: str, timeout: int = 3600) -> subprocess.CompletedProcess:
    # Source env first and verify it works, then run command separately
    # This avoids silent failures when sourcing the env file fails
    env_check = subprocess.run(
        ["bash", "-c", f"source {of_env} 2>&1 && echo OF_ENV_OK"],
        capture_output=True, text=True, timeout=30,
    )
    if "OF_ENV_OK" not in env_check.stdout:
        logger.warning(f"OpenFOAM env source failed: {env_check.stderr[:200]}")
    case_dir = Path(case_dir).resolve()
    safe_dir = str(case_dir)
    escaped_cmd = ' '.join(shlex.quote(c) for c in cmd)
    full_cmd = f"source {of_env} 2>/dev/null && set -o pipefail && {escaped_cmd} 2>&1"
    return subprocess.run(
        ["bash", "-c", full_cmd],
        cwd=safe_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False
    )


def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    a_shifted = a - a_max
    return a_max.squeeze(axis=axis) + np.log(np.sum(np.exp(a_shifted), axis=axis))


def find_force_file(forces_dir: Path) -> Optional[Path]:
    """Find the latest force.dat in postProcessing/forces/<time>/ directories.

    OpenFOAM writes force output to postProcessing/forces/<sim_time>/force.dat
    where <sim_time> is the simulation time at write (e.g. 0.0005 after the
    first timestep). The time directory name is NOT guaranteed to be '0'.
    """
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
    # fallback: check all time dirs for force.dat
    for d in reversed(time_dirs):
        candidate = d / "force.dat"
        if candidate.exists():
            return candidate
    logger.debug(f"No force.dat found in any time directory under {forces_dir}")
    return None


def extract_openfoam_force(force_file: Path) -> Optional[float]:
    if not force_file.exists():
        logger.debug(f"Force file not found: {force_file}")
        return None
    lines = force_file.read_text().strip().splitlines()
    data_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    if not data_lines:
        logger.debug(f"Force file has no data lines: {force_file}")
        return None
    last_line = data_lines[-1].strip()
    import re

    # Format 1 (OpenFOAM <=v2206): time  ((p_x p_y p_z) (v_x v_y v_z))
    m = re.search(r'\([-eE\d.\s]+\)\s*\([-eE\d.\s]+\)', last_line)
    if m:
        inner = m.group().split(') (')[0].lstrip('(')
        parts_inner = inner.split()
        if len(parts_inner) >= 1:
            fx = float(parts_inner[0])
            result = abs(fx)
            logger.debug(f"extract_openfoam_force (old format): fx={fx}, result={result}")
            return result

    # Format 2 (OpenFOAM v2512+): tabular columns
    # Time  total_x total_y total_z  pressure_x pressure_y pressure_z  viscous_x viscous_y viscous_z
    parts = last_line.split()
    if len(parts) >= 10:
        try:
            total_x = float(parts[1])
            result = abs(total_x)
            logger.debug(f"extract_openfoam_force (v2512 format): total_x={total_x}, result={result}")
            return result
        except (ValueError, IndexError):
            pass

    logger.debug(f"Could not parse force from: {last_line}")
    return None


def knots_to_ms(knots: float) -> float:
    return knots * 0.514444


def ms_to_knots(ms: float) -> float:
    return ms / 0.514444


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def latin_hypercube_sample(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        samples[:, j] = (perm + rng.uniform(0, 1, n)) / n
    return samples


def scale_lhs_to_bounds(lhs_samples: np.ndarray,
                        bounds: list[tuple[float, float]]) -> np.ndarray:
    scaled = np.zeros_like(lhs_samples)
    for j, (lo, hi) in enumerate(bounds):
        scaled[:, j] = lo + lhs_samples[:, j] * (hi - lo)
    return scaled


class MemoryManager:
    def __init__(self):
        self.gpu_available = False
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_total = torch.cuda.get_device_properties(0).total_memory
                logger.info(f"GPU detected: {torch.cuda.get_device_properties(0).name}, "
                           f"{self.gpu_total / 1e9:.1f}GB VRAM")
            else:
                logger.info("CUDA GPU not available, using CPU only")
        except ImportError:
            logger.info("PyTorch not available for GPU detection")
            self.gpu_available = False

    def safe_worker_count(self, requested: int, per_process_gb: float = 2.5) -> int:
        free_ram = psutil.virtual_memory().available / 1e9
        max_by_ram = max(1, int(free_ram / per_process_gb))
        actual = min(max(requested, 1), max_by_ram)
        if actual < requested:
            logger.info(f"Memory throttle: {requested} workers requested, "
                       f"{free_ram:.1f}GB free RAM allows {actual} "
                       f"(need {per_process_gb}GB/worker)")
        return actual

    def device_for_torch(self) -> str:
        if not self.gpu_available:
            return "cpu"
        try:
            import torch
            free_vram = torch.cuda.mem_get_info()[0] / 1e9
            if free_vram > 2.0:
                return "cuda"
            logger.info(f"GPU VRAM low ({free_vram:.1f}GB free < 2GB), falling back to CPU")
            return "cpu"
        except Exception:
            return "cpu"

    def gpu_fit_or_fallback(self, model_fn, *args, **kwargs):
        if not self.gpu_available:
            return model_fn(*args, **kwargs)
        try:
            import torch
            return model_fn(*args, **kwargs)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.warning(f"GPU OOM ({e}), retrying on CPU")
            torch.cuda.empty_cache()
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    arg.data = arg.data.cpu()
            return model_fn(*args, **kwargs)


def write_decompose_par_dict(case_dir: Path, n_procs: int):
    """Write decomposeParDict for parallel OpenFOAM execution."""
    from hull_opt.templates.openfoam import TEMPLATES
    out_path = Path(case_dir) / "system" / "decomposeParDict"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = TEMPLATES["decomposeParDict"].render(n_procs=n_procs)
    out_path.write_text(content)
    logger.info(f"decomposeParDict written for {n_procs} processes")


def run_decompose_par(case_dir: Path, of_env: str, timeout: int = 300) -> bool:
    """Run decomposePar with -force to split mesh and fields across processors."""
    proc = run_of_command(
        ["decomposePar", "-case", str(case_dir), "-force"],
        case_dir, of_env, timeout=timeout
    )
    if proc.returncode != 0:
        logger.error(f"decomposePar failed: {proc.stdout[:500]}")
        return False
    logger.info("decomposePar OK")
    return True


def run_reconstruct_par(case_dir: Path, of_env: str, timeout: int = 600) -> bool:
    """Run reconstructPar to merge processor results back to single case."""
    proc_dirs = list(Path(case_dir).glob("processor*"))
    if not proc_dirs:
        logger.info("No processor directories found, skipping reconstructPar")
        return True
    proc = run_of_command(
        ["reconstructPar", "-case", str(case_dir), "-latestTime"],
        case_dir, of_env, timeout=timeout
    )
    if proc.returncode != 0:
        logger.warning(f"reconstructPar exited non-zero (rc={proc.returncode}): {proc.stdout[:2000]}")
        return False
    logger.info("reconstructPar OK")
    return True


def run_parallel_inter_foam(case_dir: Path, of_env: str, n_procs: int,
                            timeout: int = 14400) -> subprocess.CompletedProcess:
    """Run interFoam in parallel with decomposePar/reconstructPar.

    For n_procs <= 1, falls back to serial interFoam.
    Returns the CompletedProcess of the solver run.
    """
    if n_procs <= 1:
        return run_of_command(
            ["interFoam", "-case", str(case_dir)],
            case_dir, of_env, timeout=timeout
        )

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
        case_dir, of_env, timeout=timeout
    )
    if proc.returncode != 0:
        logger.error(f"interFoam (parallel) failed (rc={proc.returncode}):\n{proc.stdout[:1000]}")

    if not run_reconstruct_par(case_dir, of_env, timeout=min(timeout, 600)):
        logger.error("reconstructPar failed — force extraction may rely on processor-local data")

    return proc

