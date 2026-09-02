import os
import sys
import json
import logging
import threading
import warnings
import psutil
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Backward-compat imports for legacy OpenFOAM functions
_DEPRECATED_FUNCS = {
    "run_of_command", "tail_log", "kill_orphaned_solvers",
    "terminate_interfoam", "find_force_file", "read_force_trace",
    "extract_openfoam_force", "write_decompose_par_dict",
    "run_decompose_par", "run_reconstruct_par", "run_parallel_inter_foam",
}


def __getattr__(name):
    if name in _DEPRECATED_FUNCS:
        from hull_opt.of_legacy import __dict__ as legacy
        if name in legacy:
            warnings.warn(
                f"hull_opt.utils.{name} is deprecated — import from hull_opt.of_legacy instead",
                DeprecationWarning, stacklevel=2,
            )
            return legacy[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def check_external_tools(config) -> list[str]:
    missing = []
    ds_dir = config.paths.dualsphysics_dir
    # also try fallback absolute path for legacy installs
    fallback_dirs = [ds_dir, "/home/anon/apps/DualSPHysics_v5.4", "./bin/dualsphysics/5.4"]
    found = False
    for d in fallback_dirs:
        ds_candidates = [
            Path(d) / "bin" / "linux" / "GenCase_linux64",
            Path(d) / "bin" / "Linux" / "GenCase_linux64",
        ]
        if any(c.exists() for c in ds_candidates):
            found = True
            break
    if not found:
        missing.append(f"DualSPHysics (GenCase_linux64) not found in {ds_dir}")

    return missing


def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    a_shifted = a - a_max
    return a_max.squeeze(axis=axis) + np.log(np.sum(np.exp(a_shifted), axis=axis))


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


def _rapid_field(result, name, default=None):
    """Fetch a field from a RapidGateResult-like object or dict, never raising."""
    if result is None:
        return default
    try:
        if isinstance(result, dict):
            return result.get(name, default)
        return getattr(result, name, default)
    except Exception:
        return default


class _RapidSummarySource:
    """Attribute/dict view merging a rapid-gate result with caller extras."""
    __slots__ = ("_result", "_extra")

    def __init__(self, result, extra):
        self._result = result
        self._extra = extra or {}

    def __getattr__(self, name):
        extra = object.__getattribute__(self, "_extra")
        if name in extra:
            return extra[name]
        result = object.__getattribute__(self, "_result")
        if isinstance(result, dict):
            return result.get(name)
        return getattr(result, name, None)


def format_rapid_summary(result) -> str:
    """One-line rapid-gate summary: avs, capsize/storm/roll/slam/inverted/wind
    margins, parametric risk, min margin and worst gate. Tolerant of missing
    fields: absent numerics render as '--', missing margins as '--'/'n/a'.
    Never raises."""
    def _num(v, spec, unit=""):
        if v is None:
            return "--"
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return "--"
        if not np.isfinite(fv):
            return "--"
        return f"{fv:{spec}}{unit}"

    def _kpa(v):
        if v is None:
            return "--"
        try:
            fv = float(v) / 1000.0
        except (TypeError, ValueError):
            return "--"
        if not np.isfinite(fv):
            return "--"
        return f"{fv:.1f}kPa"

    iter_num = _rapid_field(result, "iter", _rapid_field(result, "iteration"))
    iter_s = "--" if iter_num is None else str(iter_num)
    fom = _rapid_field(result, "fom")
    fom_s = _num(fom, ".4f")
    feasible = _rapid_field(result, "feasible")
    feas_s = "--" if feasible is None else ("T" if feasible else "F")
    param_risk = _rapid_field(result, "parametric_roll_risk")
    param_s = "--" if param_risk is None else ("OK" if not param_risk else "RISK")

    margins = _rapid_field(result, "margins") or {}
    if isinstance(margins, str):
        try:
            margins = json.loads(margins)
        except Exception:
            margins = {}
    if not isinstance(margins, dict):
        margins = {}
    finite_margins = {
        k: v for k, v in margins.items()
        if isinstance(v, (int, float)) and np.isfinite(v)
    }
    if finite_margins:
        worst = min(finite_margins, key=finite_margins.get)
        min_s = f"{finite_margins[worst]:.3f}"
    else:
        worst, min_s = "n/a", "--"
    worst_hard = _rapid_field(result, "worst_hard") or "n/a"

    return (
        f"rapid iter={iter_s} fom={fom_s} feasible={feas_s} "
        f"avs={_num(_rapid_field(result, 'avs_deg'), '.1f')} "
        f"capsize={_num(_rapid_field(result, 'capsize_margin'), '.3f')} "
        f"storm_accel={_num(_rapid_field(result, 'storm_peak_accel_g'), '.1f', 'g')} "
        f"roll_sigma={_num(_rapid_field(result, 'roll_sigma_deg'), '.1f')} "
        f"roll_period={_num(_rapid_field(result, 'roll_period_s'), '.1f', 's')} "
        f"slam_p={_kpa(_rapid_field(result, 'slam_pressure_pa'))} "
        f"inverted_p={_kpa(_rapid_field(result, 'inverted_pressure_pa'))} "
        f"wind_heel={_num(_rapid_field(result, 'storm_wind_heel_deg'), '.1f')} "
        f"param={param_s} min_margin={min_s} worst={worst} "
        f"worst_hard={worst_hard}"
    )


def log_rapid_summary(design_id, result, extra=None) -> None:
    """INFO-log the rapid-gate summary to terminal and output/pipeline.log
    with a [rapid] prefix. Emits via a dedicated handler pair instead of the
    root logger because importing capytaine resets the root logger to WARNING
    and replaces its handlers, which would silently drop the line mid-run.
    Never raises: missing attributes degrade to '--'/n/a, and any failure
    logs a short fallback line instead of crashing the pipeline."""
    try:
        line = format_rapid_summary(_RapidSummarySource(result, extra))
    except Exception:
        line = "rapid summary unavailable"
    try:
        _rapid_logger_get().info(f"[rapid] design_id={design_id} {line}")
    except Exception:
        try:
            logger.info(f"[rapid] design_id={design_id} {line}")
        except Exception:
            pass


_rapid_logger = None
_rapid_logger_lock = threading.Lock()


def _rapid_logger_get() -> logging.Logger:
    """INFO logger with console + pipeline-log handlers, configured once."""
    global _rapid_logger
    if _rapid_logger is None:
        with _rapid_logger_lock:
            if _rapid_logger is None:
                lg = logging.getLogger("hull_opt.rapid")
                lg.setLevel(logging.INFO)
                lg.propagate = False
                fmt = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
                out = logging.StreamHandler(sys.stdout)
                out.setFormatter(fmt)
                lg.addHandler(out)
                log_path = None
                for h in logging.getLogger().handlers:
                    if isinstance(h, logging.FileHandler):
                        log_path = h.baseFilename
                        break
                if log_path is None:
                    log_path = str(Path("output") / "pipeline.log")
                try:
                    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                    fh = logging.FileHandler(log_path)
                    fh.setFormatter(fmt)
                    lg.addHandler(fh)
                except Exception:
                    pass
                _rapid_logger = lg
    return _rapid_logger


def reassert_pipeline_logging(config=None) -> None:
    """Force root logging back to the pipeline's console + file handlers.

    Importing capytaine replaces the root logger's handlers with a
    RichHandler and resets its level to WARNING, silently killing all
    logger.info output (terminal and pipeline.log) for the rest of the
    process. Call this after any capytaine import to restore the pipeline's
    logging configuration. Idempotent; safe to call repeatedly.
    """
    log_config = getattr(config, "logging", None)
    level_name = getattr(log_config, "level", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    console = getattr(log_config, "console", True)
    log_file = getattr(log_config, "file", None)
    if log_file is None:
        log_file = str(Path("output") / "pipeline.log")

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception:
            pass


def log_design_diagnostics(result, design_id, iteration=None):
    """Print all evaluation metrics to terminal + pipeline.log.

    Emits two logger.info lines with [DGN] prefix so they're easily grepable:
      1. Core metrics: fom, feasible, Rt, Rw, Rf, roll_period, peak_accel,
         helm, GM, righting_energy, stability_index
      2. All rapid margins: avs, capsize, storm_accel, roll_sigma, wind_heel,
         param_roll, slam_pressure, inverted_pressure, slam_accel, max_gz_m,
         gz_area_30, gz_area_40_90, self_right, roll_period, min, worst_gate
    Never crashes (full try/except with fallback short line).
    """
    try:
        def _fmt(v, spec="", unit=""):
            if v is None:
                return "--"
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return "--"
            if not np.isfinite(fv):
                return "--"
            return f"{fv:{spec}}{unit}"

        def _bool_s(v):
            if v is None:
                return "--"
            try:
                return "T" if v else "F"
            except Exception:
                return "--"

        def _okfail(v):
            if v is None:
                return "--"
            try:
                return "OK" if v else "FAIL"
            except Exception:
                return "--"

        def _kpa(v):
            if v is None:
                return "--"
            try:
                fv = float(v) / 1000.0
            except (TypeError, ValueError):
                return "--"
            if not np.isfinite(fv):
                return "--"
            return f"{fv:.1f}kPa"

        iter_s = "" if iteration is None else f" iter={iteration}"
        core = (
            f"[DGN design_id={design_id}{iter_s}] "
            f"fom={_fmt(_rapid_field(result, 'fom'), '.4f')} "
            f"feasible={_bool_s(_rapid_field(result, 'feasible'))} "
            f"Rt={_fmt(_rapid_field(result, 'rt_total'), '.4f')} "
            f"Rw={_fmt(_rapid_field(result, 'rt_wave'), '.4f')} "
            f"Rf={_fmt(_rapid_field(result, 'rt_friction'), '.4f')} "
            f"roll={_fmt(_rapid_field(result, 'roll_period'), '.2f')}s "
            f"accel={_fmt(_rapid_field(result, 'peak_accel'), '.2f')}g "
            f"helm={_fmt(_rapid_field(result, 'helm_combined_deg'), '.1f')}\u00b0 "
            f"GM={_fmt(_rapid_field(result, 'gm'), '.4f')} "
            f"right_energy={_fmt(_rapid_field(result, 'righting_energy'), '.1f')}J "
            f"stab_idx={_fmt(_rapid_field(result, 'stability_index'), '.3f')}"
        )

        rapid = _rapid_field(result, "rapid")
        if rapid is not None:
            margins = _rapid_field(rapid, "margins") or {}
            if isinstance(margins, str):
                try:
                    margins = json.loads(margins)
                except Exception:
                    margins = {}
            if not isinstance(margins, dict):
                margins = {}
            finite_margins = {
                k: v for k, v in margins.items()
                if isinstance(v, (int, float)) and np.isfinite(v)
            }
            if finite_margins:
                worst = min(finite_margins, key=finite_margins.get)
                min_s = f"{finite_margins[worst]:.3f}"
            else:
                worst, min_s = "n/a", "--"
            param_risk = _rapid_field(rapid, "parametric_roll_risk")
            self_r = _rapid_field(rapid, "self_right")
            param_s = "--" if param_risk is None else ("OK" if not param_risk else "RISK")
            margins_line = (
                f"[DGN design_id={design_id}] margins: "
                f"avs={_fmt(_rapid_field(rapid, 'avs_deg'), '.1f')}\u00b0 "
                f"capsize={_fmt(_rapid_field(rapid, 'capsize_margin'), '.3f')} "
                f"storm={_fmt(_rapid_field(rapid, 'storm_peak_accel_g'), '.1f')}g "
                f"roll\u03c3={_fmt(_rapid_field(rapid, 'roll_sigma_deg'), '.1f')}\u00b0 "
                f"windheel={_fmt(_rapid_field(rapid, 'storm_wind_heel_deg'), '.1f')}\u00b0 "
                f"param={param_s} "
                f"slamP={_kpa(_rapid_field(rapid, 'slam_pressure_pa'))} "
                f"invP={_kpa(_rapid_field(rapid, 'inverted_pressure_pa'))} "
                f"slamA={_fmt(_rapid_field(rapid, 'slam_accel_g'), '.1f')}g "
                f"maxGZ={_fmt(_rapid_field(rapid, 'max_gz_m'), '.3f')}m "
                f"gz30={_fmt(_rapid_field(rapid, 'gz_area_30'), '.2f')} "
                f"gz40_90={_fmt(_rapid_field(rapid, 'gz_area_40_90'), '.2f')} "
                f"selfR={_okfail(self_r)} "
                f"rollP={_fmt(_rapid_field(rapid, 'roll_period_s'), '.2f')}s "
                f"min={min_s} worst={worst}"
            )
        else:
            margins_line = f"[DGN design_id={design_id}] margins: no rapid data"

        lg = logging.getLogger("hull_opt")
        lg.info(core)
        lg.info(margins_line)

    except Exception:
        try:
            logging.getLogger("hull_opt").info(
                f"[DGN design_id={design_id}] diagnostics unavailable")
        except Exception:
            pass

