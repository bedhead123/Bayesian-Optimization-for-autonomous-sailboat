"""
Bayesian Optimization surrogate using BoTorch (SingleTaskGP + LogEI).
LHS initial sampling → BO loop with mid-fidelity calibration →
convergence checking. Uses Ray or ProcessPoolExecutor for parallel evaluation.
Key exports: HullOptimizer
Bugs fixed: GP likelihood disconnected from SingleTaskGP (#2)
"""
import json
import os
import time
import numpy as np
import torch
import logging
import signal
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Optional
import queue
import threading
import sqlite3

from hull_opt.sph_gates import run_reference_storm, load_reference_settings, find_solver
from hull_opt.corrections import update_correction

logger = logging.getLogger(__name__)

from hull_opt.database import OptimizationDatabase
from hull_opt.low_fidelity import evaluate_low_fidelity, EvaluationResult
from hull_opt.config import Config, design_vector_names
from hull_opt.param_layer import flattened_bounds
from hull_opt.utils import (latin_hypercube_sample, scale_lhs_to_bounds,
                             ensure_dir, MemoryManager, log_rapid_summary,
                             log_design_diagnostics)
from hull_opt.mid_fidelity import run_mid_fidelity_calibration


class ReferenceRunner(threading.Thread):
    """Daemon thread: runs DualSPHysics reference storms on designs enqueued by the BO loop."""
    def __init__(self, config, db_path: str, output_dir: Path, gpu_lock_path: str,
                 reference_queue: queue.Queue):
        super().__init__(daemon=True)
        self.config = config
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.gpu_lock_path = gpu_lock_path
        self.queue = reference_queue
        self.settings = load_reference_settings(config)
        self._stop_event = threading.Event()

    def run(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=10)
            except queue.Empty:
                continue
            if item is None:
                self.queue.task_done()
                break
            design_id = item["design_id"]
            design_vector = item["design_vector"]
            iteration = item["iteration"]
            existing = conn.execute(
                "SELECT status FROM reference_runs WHERE design_id=?", (design_id,)
            ).fetchone()
            if existing is not None:
                self.queue.task_done()
                continue
            lock_fd = None
            try:
                lock_fd = open(self.gpu_lock_path, "w")
                import fcntl
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    # No SKIPPED row: a transient lock collision (e.g. a
                    # calibration holding the GPU) must not permanently burn
                    # this design — it stays eligible and is retried on the
                    # next enqueue. NO task_done() here: the continue exits
                    # through the finally block (line ~190) which calls
                    # task_done() exactly once per consumed item.
                    logger.info(
                        f"gpu.lock busy — keeping design {design_id} eligible "
                        f"for storm retry"
                    )
                    continue
                # Look up the STL path from the DB; skip if the design has no
                # geometry (failed generation) or the file is missing.
                stl_row = conn.execute(
                    "SELECT cad_stl_path FROM designs WHERE id=?", (design_id,)
                ).fetchone()
                stl_path = (stl_row["cad_stl_path"] if stl_row
                            and stl_row["cad_stl_path"]
                            and Path(stl_row["cad_stl_path"]).exists()
                            else None)
                if stl_path is None:
                    # The STL is written to the DB shortly after the design
                    # row (LHS storms enqueue from the batch before geometry
                    # persistence). No SKIPPED row (Bug #154 pattern): poll
                    # within THIS task (never re-queue — that double-counts
                    # task_done and crashes the thread) until the STL lands
                    # or the deadline passes.
                    deadline = time.time() + 300
                    while stl_path is None and time.time() < deadline:
                        time.sleep(15)
                        stl_row = conn.execute(
                            "SELECT cad_stl_path FROM designs WHERE id=?",
                            (design_id,),
                        ).fetchone()
                        stl_path = (stl_row["cad_stl_path"] if stl_row
                                    and stl_row["cad_stl_path"]
                                    and Path(stl_row["cad_stl_path"]).exists()
                                    else None)
                    if stl_path is None:
                        logger.warning(
                            f"design {design_id} still has no STL after 5 "
                            f"min — dropping its storm"
                        )
                        # NO task_done() here: the continue exits through the
                        # finally block which calls task_done() exactly once
                        # per consumed item.
                        continue
                case_dir = self.output_dir / f"ref_ds_{design_id}"
                result = run_reference_storm(
                    design_vector, self.config, str(case_dir), design_id,
                    settings=self.settings,
                    stl_path=stl_path,
                    gpu_lock=self.gpu_lock_path,
                    held_lock_fd=lock_fd,
                )
                conn.execute(
                    """INSERT OR REPLACE INTO reference_runs
                    (design_id, tool, status, max_accel_g, max_pressure_pa,
                     final_orientation, capsized, sim_time_s, wall_time_s, details)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (design_id, result.get("tool","dualsphysics"),
                     result.get("status","FAILED"),
                     result.get("max_accel_g"), result.get("max_pressure_pa"),
                     json.dumps(result.get("final_orientation")),
                     1 if result.get("capsized") else 0,
                     result.get("sim_time_s"), result.get("wall_time_s"),
                     result.get("details",""))
                )
                conn.commit()
                rapid_row = conn.execute(
                    "SELECT rapid_gates FROM designs WHERE id=?", (design_id,)
                ).fetchone()
                if rapid_row and rapid_row["rapid_gates"]:
                    rapid = json.loads(rapid_row["rapid_gates"])
                    gate_accel = rapid.get("storm_peak_accel_g", 0)
                    ref_accel = result.get("max_accel_g", 0)
                    if gate_accel > 0 and ref_accel > 0:
                        ratio = ref_accel / gate_accel
                        old = conn.execute(
                            "SELECT value FROM corrections WHERE key='storm_accel'"
                        ).fetchone()
                        old_val = float(old["value"]) if old else 1.0
                        new_val, _ = update_correction("storm_accel", old_val, ratio)
                        conn.execute(
                            "INSERT OR REPLACE INTO corrections(key, value) VALUES('storm_accel',?)",
                            (new_val,)
                        )
                        conn.commit()
                        log_rapid_summary(design_id, rapid,
                                          extra={"iter": iteration})
                        logger.info(
                            f"[rapid] design_id={design_id} reference_storm "
                            f"ref_accel={ref_accel:.3f}g gate_accel={gate_accel:.3f}g "
                            f"ratio={ratio:.3f} correction={old_val:.3f}->{new_val:.3f}"
                        )
            except Exception as exc:
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO reference_runs(design_id,tool,status,details) VALUES(?,'dualsphysics','FAILED',?)",
                        (design_id, str(exc)[:500])
                    )
                    conn.commit()
                except Exception as inner_exc:
                    logger.warning(f"Failed to record reference run error in DB: {inner_exc}")
            finally:
                if lock_fd is not None:
                    try:
                        import fcntl
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        lock_fd.close()
                    except Exception:
                        pass
                self.queue.task_done()
        conn.close()

    def stop(self):
        self._stop_event.set()


class HullOptimizer:
    def __init__(self, config: Config, db: OptimizationDatabase):
        self.config = config
        self.db = db
        self.bounds = flattened_bounds(self.config)
        self.dim = len(self.bounds)
        self.output_dir = ensure_dir(Path(config.paths.output_dir))
        # Restore the drag factor from the last completed calibration on
        # resume — a fresh 1.0 would silently rescale every FoM mid-run.
        latest_cal = db.get_latest_calibration()
        self.drag_factor = float(latest_cal.get("factor", 1.0)) if latest_cal else 1.0
        # Stored FoMs may have been computed under different factors in
        # earlier segments (or before the factor existed): normalize them
        # all to the current factor so GP training sees one objective.
        try:
            n_rescored = db.rescore_foms(self.drag_factor, self.config.weights.w1)
            if n_rescored:
                logger.info(f"Rescored {n_rescored} stored FoMs under drag factor "
                            f"f={self.drag_factor:.4f}")
                self._sync_history_from_db()
        except Exception as e:
            logger.warning(f"FoM rescore on resume failed: {e}")

        # Config-identity guard: resuming or validating an existing campaign
        # under a different objective config silently corrupts stored FoMs,
        # rankings, and gate thresholds (happened Aug 5 2026: optimization at
        # 4.0kn/0.14m³, validation at 3.2kn/0.145m³).
        try:
            from hull_opt.config import config_signature
            sig = config_signature(self.config)
            matches, prev_sig = db.check_config_signature(sig)
            if not matches:
                if os.environ.get("BOAT_IGNORE_CONFIG_MISMATCH"):
                    logger.error(
                        f"Config MISMATCH vs campaign ({prev_sig[:12]}... != "
                        f"{sig[:12]}...) — continuing because "
                        f"BOAT_IGNORE_CONFIG_MISMATCH is set"
                    )
                else:
                    raise RuntimeError(
                        f"Config signature mismatch: this run's config "
                        f"({sig[:12]}...) differs from the database campaign "
                        f"({prev_sig[:12]}...). Running on this DB would "
                        f"silently mix incompatible objectives. Set "
                        f"BOAT_IGNORE_CONFIG_MISMATCH=1 to force."
                    )
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Config signature check failed: {e}")
        self.best_fom = -float("inf")
        self.best_x = None

        self.gp_model = None
        self.gp_likelihood = None
        self.cls_model = None
        self._best_fom_history = []
        self._fom_history = []
        self.mem = MemoryManager()
        self._reference_queue: queue.Queue = queue.Queue()
        self._reference_thread: Optional[ReferenceRunner] = None

    def run(self):
        logger.info("=" * 60)
        logger.info("Starting hull optimization")
        lhs_max = min(getattr(self.config.optimization, 'lhs_max', 40),
                      self.config.optimization.n_initial)
        n_iter = self.config.optimization.n_iter
        logger.info(f"Plan: up to {lhs_max} LHS samples (plateau may end earlier) "
                    f"+ {n_iter} BO iterations (iterations {lhs_max}+)")
        logger.info("=" * 60)
        self._interrupted = False
        def _sigint_handler(signum, frame):
            logger.warning("Interrupted, shutting down...")
            self._interrupted = True
        original_handler = signal.signal(signal.SIGINT, _sigint_handler)

        n_workers = min(self.config.optimization.n_initial, 4)

        # Start persistent parallel pool
        self._pool = self._start_pool(n_workers)

        ref_config = getattr(self.config, "reference", None)
        ref_tool = getattr(ref_config, "tool", "") if ref_config else ""
        if ref_tool and ref_tool != "none":
            gpu_lock_path = str(self.output_dir / "gpu.lock")
            self._reference_thread = ReferenceRunner(
                self.config, str(self.db.db_path), self.output_dir,
                gpu_lock_path, self._reference_queue
            )
            self._reference_thread.start()
            logger.info("ReferenceRunner thread started")

        try:
            if self._interrupted:
                self._stop_pool()
                signal.signal(signal.SIGINT, original_handler)
                return self.db.get_top_n(3)

            existing_count = self.db.get_iteration_count()
            if existing_count == 0:
                logger.info("Phase 1: LHS initial sampling")
                self._initial_sampling()
                start_iter = self.db.get_iteration_count()
            elif existing_count < self.config.optimization.n_initial:
                logger.info(f"Resuming incomplete initial sampling "
                           f"(found {existing_count}/{self.config.optimization.n_initial})")
                self._resume_initial_sampling(existing_count)
                start_iter = self.db.get_iteration_count()
            else:
                start_iter = existing_count
                logger.info(f"Resuming from iteration {start_iter}")

            if not self._interrupted:
                logger.info("Phase 2: Bayesian optimization")
                self._bo_loop(start_iter)
        finally:
            signal.signal(signal.SIGINT, original_handler)
            self._stop_pool()
            if self._reference_thread is not None:
                self._reference_queue.put(None)
                self._reference_thread.stop()
                self._reference_thread.join(timeout=5)

        logger.info("BO loop complete")

        top_designs = self.db.get_top_n(3)
        logger.info(f"Top 3 designs: {[d['id'] for d in top_designs]}")

        return top_designs

    def _start_pool(self, n_workers):
        """Start a persistent parallel worker pool."""
        self._pool_type = "serial"
        self._process_pool = None
        self._ray_remote = None
        self._ray_initialized = False
        n_workers = self.mem.safe_worker_count(n_workers, per_process_gb=3.5)
        n_gpus = 1 if self.mem.gpu_available else 0
        # Pin BLAS/OpenMP threads per worker: n_workers concurrent BEM solves
        # (each a dense N^3 LU) otherwise thrash all cores via scipy-openblas
        # MAX_THREADS=64 with no affinity, multiplying wall time ~3x.
        n_cpus = os.cpu_count() or 1
        threads_per_worker = max(1, min(4, n_cpus // max(1, n_workers)))
        os.environ["BOAT_BLAS_THREADS"] = str(threads_per_worker)
        logger.info(f"Pinned BLAS threads to {threads_per_worker}/worker "
                    f"({n_workers} workers on {n_cpus} cores)")
        try:
            import ray
            # Kill workers earlier (90% vs Ray's 95% default) so an
            # unexpected memory spike fails a task cleanly (and Ray retries
            # it) instead of pushing the node into a hard OOM that kills
            # idle workers too.
            os.environ.setdefault("RAY_memory_usage_threshold", "0.9")
            ray.init(ignore_reinit_error=True, num_cpus=n_workers, num_gpus=n_gpus)
            self._ray_remote = ray.remote(
                _evaluate_one_wrapper
            ).options(num_cpus=1, num_gpus=0)
            self._pool_type = "ray"
            self._ray_initialized = True
            logger.info(f"Started Ray pool with {n_workers} workers "
                       f"(GPU: {self.mem.gpu_available})")
        except Exception as e:
            logger.warning(f"Ray unavailable ({e}), using ProcessPoolExecutor")
            self._process_pool = ProcessPoolExecutor(max_workers=n_workers)
            self._pool_type = "process"
            logger.info(f"Initialized {n_workers} worker processes")
        return self

    def _stop_pool(self):
        if self._pool_type == "ray" and self._ray_initialized:
            try:
                import ray
                ray.shutdown()
            except Exception:
                pass
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=False)
            self._process_pool = None
        self._pool_type = "serial"
        self._ray_initialized = False

    def _eval_one(self, design_vector, iteration=None, bem_skip=False):
        if self._pool_type == "ray" and self._ray_remote is not None:
            import ray
            return ray.get(self._ray_remote.remote(
                design_vector, self.config, str(self.output_dir), self.drag_factor, iteration, bem_skip
            ))
        if self._pool_type == "process" and self._process_pool is not None:
            future = self._process_pool.submit(
                _evaluate_one_wrapper,
                design_vector, self.config, str(self.output_dir), self.drag_factor, iteration, bem_skip
            )
            return future.result()
        return _evaluate_one_wrapper(
            design_vector, self.config, str(self.output_dir), self.drag_factor, iteration, bem_skip
        )

    @staticmethod
    def _result_db_kwargs(res) -> dict:
        cv = res.constraint_values if res.constraint_values else {}
        return dict(
            righting_energy=cv.get("righting_energy"),
            gm=cv.get("gm"),
            cg_z=cv.get("cg_z"),
            eq_heel_deg=cv.get("eq_heel_feathered_deg"),
            helm_fwd_deg=cv.get("helm_fwd_deg"),
            helm_aft_deg=cv.get("helm_aft_deg"),
            helm_combined_deg=cv.get("helm_combined_deg"),
            lead_pct_lwl=cv.get("lead_pct_lwl"),
            T_over_L=cv.get("T_over_L"),
            T_total_m=cv.get("T_total_m"),
            physical_params=res.physical_params,
        )

    def _log_rapid_summary(self, design_id, result, iteration=None):
        rapid = getattr(result, "rapid", None)
        if result is None or rapid is None:
            return
        log_rapid_summary(design_id, rapid, extra={
            "iter": iteration,
            "fom": result.fom,
            "feasible": result.feasible,
        })

    def _persist_bem_data(self, design_id: int, result) -> None:
        if hasattr(result, 'rao_data') and result.rao_data is not None:
            try:
                bem_hash = f"design_{design_id}"
                omega_arr = np.asarray(result.rao_data["omega"])
                for i, w in enumerate(omega_arr):
                    self.db.store_bem_run(
                        bem_hash, design_id, float(w), 180.0,
                        float(result.rao_data["heave_rao"][i]),
                        float(result.rao_data["pitch_rao"][i]),
                        float(result.rao_data["roll_rao"][i]),
                        float(result.rao_data.get("wall_time_s", 0)),
                    )
            except Exception:
                pass
        if hasattr(result, 'rapid') and result.rapid is not None and result.rapid.storm_rao_data:
            try:
                for hrao in result.rapid.storm_rao_data:
                    h_bem_hash = f"design_{design_id}_storm_h{hrao['heading_deg']}"
                    oa = np.asarray(hrao["omega"])
                    for i, w in enumerate(oa):
                        self.db.store_bem_run(
                            h_bem_hash, design_id, float(w), float(hrao["heading_deg"]),
                            float(hrao["heave_rao"][i]),
                            float(hrao["pitch_rao"][i]),
                            float(hrao["roll_rao"][i]),
                            0.0,
                        )
            except Exception:
                pass

    def _initial_sampling(self):
        lhs_min = getattr(self.config.optimization, 'lhs_min', 20)
        lhs_inc = getattr(self.config.optimization, 'lhs_increment', 10)
        lhs_max = min(getattr(self.config.optimization, 'lhs_max', 40),
                      self.config.optimization.n_initial)
        lhs_seed = getattr(self.config.optimization, 'lhs_seed', 42)

        all_lhs_raw = latin_hypercube_sample(lhs_max, self.dim, seed=lhs_seed)
        all_designs = scale_lhs_to_bounds(all_lhs_raw, self.bounds)

        total_done = 0
        batch_idx = 0
        batch_size = lhs_min

        while total_done < lhs_max and not getattr(self, '_interrupted', False):
            n_this = min(batch_size, lhs_max - total_done)
            logger.info(f"LHS batch {batch_idx}: evaluating {n_this} designs (total {total_done + n_this}/{lhs_max})")

            designs = all_designs[total_done:total_done + n_this]
            self._evaluate_batch(designs, list(range(total_done, total_done + n_this)), bem_skip=False)

            total_done += n_this
            batch_idx += 1
            batch_size = lhs_inc

            if total_done >= lhs_min and self._lhs_plateau_reached():
                logger.info(f"LHS plateau reached at {total_done} samples")
                break

        logger.info(f"LHS complete: {total_done} designs evaluated")

    def _lhs_plateau_reached(self) -> bool:
        all_designs = self.db.get_all_designs()
        if len(all_designs) < 20:
            return False
        prior = all_designs[:-10]
        feasible_ratio = sum(1 for d in prior if d["feasible"]) / max(len(prior), 1)
        recent = all_designs[-10:]
        recent_ratio = sum(1 for d in recent if d["feasible"]) / max(len(recent), 1)
        if feasible_ratio == 0.0 and recent_ratio == 0.0:
            logger.debug(f"LHS plateau: no feasible design seen yet "
                         f"({len(all_designs)} evaluated) — continuing sampling")
            return False
        return abs(recent_ratio - feasible_ratio) < 0.05

    def _reference_enqueue_due(self, iteration: int) -> bool:
        """Storm enqueue policy: every reference_every_n iterations, except
        calibration iterations — a storm enqueued at iteration % 20 == 0
        races the calibration block for gpu.lock in the same iteration and
        starves it (calibration's flock fails -> SKIPPED -> marked failed)."""
        ref_cfg = getattr(self.config, "reference", None)
        ref_every = getattr(ref_cfg, "reference_every_n", 10) if ref_cfg else 10
        cal_freq = getattr(self.config.calibration, "frequency", 20)
        if iteration % ref_every != 0:
            return False
        if not cal_freq:
            return True
        return not (iteration > 0 and iteration % cal_freq == 0)

    def _store_eval_result(self, res, design_vector, iteration):
        """Insert design into DB, log diagnostics, update best-so-far."""
        if res is None:
            return
        design_id = self.db.insert_design(
            iter_num=iteration, design_vector=design_vector,
            feasible=res.feasible, fom=res.fom,
            rt_total=res.rt_total, rt_wave=res.rt_wave,
            rt_friction=res.rt_friction,
            stability_index=res.stability_index,
            roll_period=res.roll_period,
            peak_accel=res.peak_accel,
            constraint_values=res.constraint_values,
            constraint_violations=res.constraint_violations,
            error_code=res.error_code,
            cad_stl_path=res.cad_stl_path,
            cad_sac_path=res.cad_sac_path,
            drag_factor=self.drag_factor,
            **self._result_db_kwargs(res),
        )
        if hasattr(res, 'rapid') and res.rapid is not None:
            try:
                self.db.update_design_rapid_gates(design_id, res.rapid)
            except Exception:
                pass
        if hasattr(res, 'balance') and res.balance is not None:
            try:
                self.db.update_design_balance(design_id, res.balance)
            except Exception:
                pass
        self._log_rapid_summary(design_id, res, iteration=iteration)
        log_design_diagnostics(res, design_id, iteration=iteration)
        self._persist_bem_data(design_id, res)
        self._fom_history.append(res.fom)
        if res.fom > self.best_fom:
            self.best_fom = res.fom
            self.best_x = design_vector.copy()
        self._best_fom_history.append(self.best_fom)
        if self._reference_thread is not None and self._reference_thread.is_alive():
            if self._reference_enqueue_due(iteration):
                self._reference_queue.put({
                    "design_id": design_id,
                    "design_vector": design_vector.tolist() if hasattr(design_vector, 'tolist') else design_vector,
                    "iteration": iteration,
                })

    def _evaluate_batch(self, designs, indices, start_iteration=None, bem_skip=False) -> list:
        if start_iteration is None:
            start_iteration = indices[0] if indices else 0
        n_total = len(designs)
        if self._pool_type == "ray":
            import ray
            futures = {self._ray_remote.remote(
                designs[i], self.config, str(self.output_dir), self.drag_factor,
                start_iteration + i, bem_skip
            ): i for i in range(n_total)}
            pending = set(futures.keys())
            last_log = 0
            while pending:
                done, pending = ray.wait(list(pending), timeout=60, num_returns=1)
                for ref in done:
                    i = futures[ref]
                    try:
                        res = ray.get(ref)
                        self._store_eval_result(res, designs[i], start_iteration + i)
                    except Exception as e:
                        logger.error(f"Worker {i} failed: {e}")
                n_done = n_total - len(pending)
                if n_done - last_log >= max(1, n_total // 20) or n_done == n_total:
                    logger.info(f"LHS batch: {n_done}/{n_total} designs evaluated"
                                f" ({n_done * 100 // n_total}%)")
                    last_log = n_done
        else:
            if n_total == 1 or self._process_pool is None:
                for i in range(n_total):
                    res = self._eval_one(designs[i], iteration=start_iteration, bem_skip=bem_skip)
                    self._store_eval_result(res, designs[i], start_iteration + i)
                return
            futures = {
                self._process_pool.submit(
                    _evaluate_one_wrapper, designs[i], self.config,
                    str(self.output_dir), self.drag_factor,
                    start_iteration + i, bem_skip
                ): i for i in range(n_total)
            }
            last_log = 0
            n_done = 0
            for f in as_completed(futures):
                i = futures[f]
                try:
                    self._store_eval_result(f.result(), designs[i], start_iteration + i)
                except Exception as e:
                    logger.error(f"Worker {i} failed: {e}")
                n_done += 1
                if n_done - last_log >= max(1, n_total // 20) or n_done == n_total:
                    logger.info(f"LHS batch: {n_done}/{n_total} designs evaluated"
                                f" ({n_done * 100 // n_total}%)")
                    last_log = n_done

    def _resume_initial_sampling(self, start_idx: int, bem_skip: bool = False):
        logger.info(f"Resuming LHS initial sampling from index {start_idx}")
        lhs_seed = getattr(self.config.optimization, 'lhs_seed', 42)
        lhs_max = min(getattr(self.config.optimization, 'lhs_max', 40),
                      self.config.optimization.n_initial)
        lhs_raw = latin_hypercube_sample(lhs_max, self.dim, seed=lhs_seed)
        designs = scale_lhs_to_bounds(lhs_raw, self.bounds)

        n_remaining = lhs_max - start_idx
        n_workers = self.mem.safe_worker_count(min(n_remaining, 4), per_process_gb=3.5)
        logger.info(f"Evaluating {n_remaining} remaining LHS designs "
                    f"with {n_workers} parallel workers")

        indices = list(range(start_idx, lhs_max))
        n_total = len(indices)
        if self._pool_type == "ray":
            import ray
            futures = {self._ray_remote.remote(
                designs[i], self.config, str(self.output_dir), self.drag_factor, i, bem_skip
            ): i for i in indices}
            results_by_index = {}
            pending = set(futures.keys())
            last_log = 0
            while pending:
                done, pending = ray.wait(list(pending), timeout=60, num_returns=1)
                for ref in done:
                    idx = futures[ref]
                    try:
                        results_by_index[idx] = ray.get(ref)
                    except Exception as e:
                        logger.error(f"Worker {idx} failed: {e}")
                n_done = len(results_by_index)
                if n_done - last_log >= max(1, n_total // 20) or n_done == n_total:
                    logger.info(f"Resume LHS: {n_done}/{n_total} designs evaluated"
                                f" ({n_done * 100 // n_total}%)")
                    last_log = n_done
            results = [results_by_index.get(i, None) for i in indices]
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _evaluate_one_wrapper, designs[i], self.config,
                        str(self.output_dir), self.drag_factor, i, bem_skip
                    ): i for i in indices
                }
                results_by_index = {}
                last_log = 0
                for f in as_completed(futures):
                    idx = futures[f]
                    try:
                        results_by_index[idx] = f.result()
                    except BrokenProcessPool as e:
                        logger.error(f"Worker {idx} process crashed: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Worker {idx} failed: {e}")
                    n_done = len(results_by_index)
                    if n_done - last_log >= max(1, n_total // 20) or n_done == n_total:
                        logger.info(f"Resume LHS: {n_done}/{n_total} designs evaluated"
                                    f" ({n_done * 100 // n_total}%)")
                        last_log = n_done
                results = [results_by_index.get(i, None) for i in indices]

        for i, res in zip(indices, results):
            if res is None:
                logger.error(f"Worker {i} returned None, skipping")
                continue
            design_id = self.db.insert_design(
                iter_num=i, design_vector=designs[i],
                feasible=res.feasible, fom=res.fom,
                rt_total=res.rt_total, rt_wave=res.rt_wave,
                rt_friction=res.rt_friction,
                stability_index=res.stability_index,
                roll_period=res.roll_period,
                peak_accel=res.peak_accel,
                constraint_values=res.constraint_values,
                constraint_violations=res.constraint_violations,
                error_code=res.error_code,
                cad_stl_path=res.cad_stl_path,
                cad_sac_path=res.cad_sac_path,
                drag_factor=self.drag_factor,
                **self._result_db_kwargs(res),
            )
            if hasattr(res, 'rapid') and res.rapid is not None:
                try:
                    self.db.update_design_rapid_gates(design_id, res.rapid)
                except Exception:
                    pass
            if hasattr(res, 'balance') and res.balance is not None:
                try:
                    self.db.update_design_balance(design_id, res.balance)
                except Exception:
                    pass
            self._log_rapid_summary(design_id, res, iteration=i)
            log_design_diagnostics(res, design_id, iteration=i)
            self._persist_bem_data(design_id, res)
            self._fom_history.append(res.fom)
            if res.fom > self.best_fom:
                self.best_fom = res.fom
                self.best_x = designs[i].copy()
            self._best_fom_history.append(self.best_fom)
            if self._reference_thread is not None and self._reference_thread.is_alive():
                if self._reference_enqueue_due(i):
                    self._reference_queue.put({
                        "design_id": design_id,
                        "design_vector": designs[i].tolist() if hasattr(designs[i], 'tolist') else designs[i],
                        "iteration": i,
                    })

        logger.info(f"Resume LHS complete. Feasible: {sum(1 for r in results if r is not None and r.feasible)}/{len(results)}")

    def _systemic_error_abort_reason(self) -> Optional[str]:
        lhs_max = min(getattr(self.config.optimization, 'lhs_max', 40),
                      self.config.optimization.n_initial)
        rows = self.db.get_all_designs()
        lhs = [r for r in rows if r["iter"] < lhs_max]
        bo = [r for r in rows if r["iter"] >= lhs_max]

        def _env_class(code):
            return (code.startswith("E_RAO_ENV")
                    or "ImportError" in code
                    or "ModuleNotFoundError" in code)

        bo_recent = bo[-10:]
        bo_n = len(bo_recent)
        bo_err_frac = sum(1 for r in bo_recent if r.get("error_code")) / max(bo_n, 1)
        bo_env_frac = sum(1 for r in bo_recent if _env_class(r.get("error_code") or "")) / max(bo_n, 1)
        lhs_err_frac = sum(1 for r in lhs if r.get("error_code")) / max(len(lhs), 1)
        return _abort_reason(bo_n, bo_err_frac, lhs_err_frac, bo_env_frac)

    def _bo_loop(self, start_iter: int):
        n_iter = self.config.optimization.n_iter
        self._bo_iteration_count = 0

        for it in range(start_iter, start_iter + n_iter):
            if getattr(self, '_interrupted', False):
                logger.info("BO loop interrupted")
                break
            logger.info(f"BO iteration {it}/{start_iter + n_iter - 1}")
            self._bo_iteration_count += 1

            if self._bo_iteration_count % 5 == 0:
                reason = self._systemic_error_abort_reason()
                if reason is not None:
                    logger.critical(reason)
                    raise EnvironmentError(reason)

            try:
                candidate = self._propose_candidate()
            except Exception as e:
                logger.error(f"Acquisition optimization failed: {e}")
                candidate = self._random_candidate()

            result = self._eval_one(candidate, iteration=it)
            self._store_eval_result(result, candidate, it)

            # mid-fidelity calibration (frequency=0 disables it entirely)
            if (self.config.calibration.frequency
                    and it > 0 and it % self.config.calibration.frequency == 0):
                logger.info(f"Calibration at iteration {it}")
                retry_failed = getattr(self.config.calibration, 'retry_failed', False)
                best = self.db.get_best_feasible_uncalibrated(
                    retry_failed=retry_failed,
                    fallback_to_least_infeasible=True)
                if best is None:
                    logger.warning(
                        f"Calibration at iteration {it}: no design picked — "
                        f"no feasible design without a calibration attempt "
                        f"(retry_failed={retry_failed})"
                    )
                else:
                    if not best["feasible"]:
                        raw_viol = best.get("constraint_violations")
                        try:
                            n_viol = len(json.loads(raw_viol)) if raw_viol else 0
                        except Exception:
                            n_viol = 0
                        logger.warning(
                            f"Calibration: zero feasible designs — calibrating "
                            f"least-infeasible design {best['id']} "
                            f"(violations: {n_viol})")
                    logger.info(
                        f"Calibration at iteration {it}: picked design "
                        f"{best['id']} (feasible={bool(best['feasible'])})"
                    )
                    try:
                        # Validate design has valid geometry before CFD
                        has_stl = best.get("cad_stl_path") and Path(best["cad_stl_path"]).exists()
                        no_error = best.get("error_code") is None
                        if not has_stl or not no_error:
                            logger.warning(f"Calibration skipped: design {best['id']} has invalid geometry")
                            if self.drag_factor != 1.0:
                                logger.info("Keeping previous drag factor")
                        else:
                            best_x = np.array(json.loads(best["design_vector"]))
                            self.db.mark_calibration_attempt(
                                best["id"], it, "running"
                            )
                            cal = run_mid_fidelity_calibration(
                                best_x, best["id"], it, self.config
                            )
                            self._handle_calibration_result(
                                cal, best["id"], it
                            )
                    except subprocess.TimeoutExpired:
                        self.db.mark_calibration_attempt(
                            best["id"], it, "timeout"
                        )
                        logger.warning("Calibration timed out")
                    except Exception as e:
                        self.db.mark_calibration_attempt(
                            best["id"], it, "failed"
                        )
                        logger.warning(f"Calibration failed: {e}")
                        if self.drag_factor != 1.0:
                            logger.info("Keeping previous drag factor")

            # convergence check
            if self._check_convergence():
                logger.info(f"Converged at iteration {it}")
                break

    def _handle_calibration_result(self, cal, design_id: int, it: int):
        """B2: absorb a mid-fidelity calibration result.

        - None (solver crash/preflight failure) -> mark 'failed'.
        - {'valid': False} (sub-friction Rt or non-steady force trace) ->
          reset drag factor to 1.0, mark 'invalid', and do NOT store/rescore
          from the bad measurement — history stays unpolluted.
        - valid -> smooth the factor, rescore FoMs, store calibration.
        """
        if cal is None:
            self.db.mark_calibration_attempt(design_id, it, "failed")
            return
        if not cal.get("valid", True):
            reason = cal.get("reason", "unknown")
            logger.warning(
                f"Calibration invalid ({reason}): resetting "
                f"drag factor to 1.0"
            )
            self.drag_factor = 1.0
            # Re-score stored FoMs to the reset factor so the DB does not
            # stay stranded on the old factor's scale until the next
            # successful calibration.
            n_rescored = self.db.rescore_foms(
                1.0, self.config.weights.w1
            )
            if n_rescored:
                logger.info(
                    f"Rescored {n_rescored} stored FoMs back to factor 1.0"
                )
                self._sync_history_from_db()
            self.db.mark_calibration_attempt(
                design_id, it, "invalid", factor=1.0
            )
            return
        old_factor = self.drag_factor
        new_factor, was_clipped = smooth_drag_factor(
            old_factor, cal["factor"],
            self.config.calibration.tolerance,
        )
        if was_clipped:
            logger.warning(
                f"Drag factor clipped to tolerance: "
                f"{cal['factor']:.4f} -> {new_factor:.4f}"
            )
        self.drag_factor = new_factor
        n_rescored = self.db.rescore_foms(
            new_factor, self.config.weights.w1
        )
        if n_rescored:
            logger.info(
                f"Rescored {n_rescored} stored FoMs under "
                f"new drag factor f={new_factor:.4f}"
            )
            self._sync_history_from_db()
        logger.info(f"Drag factor updated: f={new_factor:.4f} "
                    f"(raw CFD factor={cal['factor']:.4f}, "
                    f"CFD={cal['rt_cfd']:.4f} N, low-fi={cal['rt_lowfi']:.4f} N)")
        self.db.store_calibration(
            design_id, it,
            cal["rt_lowfi"],
            cal["rt_cfd"],
            cal["delta"],
            new_factor,
        )
        self.db.mark_calibration_attempt(
            design_id, it, "ok", factor=new_factor
        )
        if old_factor != new_factor:
            logger.info(f"Drag factor changed {old_factor:.4f} -> {new_factor:.4f}")

    def _propose_candidate(self) -> np.ndarray:
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf
        from botorch.utils.transforms import normalize, unnormalize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.likelihoods import GaussianLikelihood

        all_designs = self.db.get_all_designs()
        if len(all_designs) < 5:
            return self._random_candidate()

        X_list = []
        y_list = []
        for d in all_designs:
            x_vec = np.array(json.loads(d["design_vector"]), dtype=float)
            if len(x_vec) != self.dim:
                continue
            fom = d["fom"]
            if not np.isfinite(fom) or fom < -1e10:
                fom = -100.0
            X_list.append(x_vec)
            y_list.append(fom)

        if len(X_list) < 5:
            return self._random_candidate()

        device = self.mem.device_for_torch()

        X_raw = torch.tensor(np.array(X_list), dtype=torch.float64)
        y = torch.tensor(np.array(y_list), dtype=torch.float64).unsqueeze(-1)

        bounds_tensor = torch.tensor(
            [[b[0] for b in self.bounds], [b[1] for b in self.bounds]],
            dtype=torch.float64
        )

        X_norm = normalize(X_raw, bounds_tensor)

        if device == "cuda":
            X_norm = X_norm.to(device)
            y = y.to(device)
            bounds_tensor = bounds_tensor.to(device)

        import gpytorch
        noise_bound = max(self.config.optimization.gp_jitter, 1e-6)
        self.gp_likelihood = GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(noise_bound),
        ).to(device)
        self.gp_model = SingleTaskGP(X_norm, y, likelihood=self.gp_likelihood).to(device)
        mll = ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)

        try:
            fit_gpytorch_model(mll)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.warning(f"GPU OOM ({e}), falling back to CPU")
            del mll
            torch.cuda.empty_cache()
            device = "cpu"
            X_norm = X_norm.cpu()
            y = y.cpu()
            bounds_tensor = bounds_tensor.cpu()
            self.gp_likelihood = self.gp_likelihood.cpu()
            self.gp_model = SingleTaskGP(X_norm, y, likelihood=self.gp_likelihood)
            mll = ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)
            fit_gpytorch_model(mll)

        acq = LogExpectedImprovement(self.gp_model, best_f=y.max())

        acq_bounds = torch.tensor(
            [[0.0] * self.dim, [1.0] * self.dim],
            dtype=torch.float64, device=device if device == "cuda" else None
        )

        candidate_norm, _ = optimize_acqf(
            acq, bounds=acq_bounds, q=1,
            num_restarts=self.config.optimization.num_restarts,
            raw_samples=self.config.optimization.raw_samples,
        )

        candidate = unnormalize(candidate_norm, bounds_tensor)
        candidate = candidate.squeeze().detach().cpu().numpy()
        if not np.all(np.isfinite(candidate)):
            logger.warning("BO candidate contains NaN/Inf; falling back")
            return self._random_candidate()
        clipped = np.clip(candidate, -10.0, 10.0)
        if np.any(clipped != candidate):
            logger.warning("BO candidate clipped to [-10, 10]")
        return clipped

    def _random_candidate(self) -> np.ndarray:
        base_seed = getattr(self.config.optimization, 'lhs_seed', 42)
        iter_seed = base_seed + getattr(self, '_bo_iteration_count', 0) * 1000 + 9999
        rng = np.random.default_rng(iter_seed)
        bounds_arr = np.array(self.bounds)
        lo = bounds_arr[:, 0]
        hi = bounds_arr[:, 1]
        return rng.uniform(lo, hi, size=self.dim)

    def _sync_history_from_db(self):
        """Rebuild in-memory FoM history from the DB after a rescore so the
        convergence check never compares foms from different drag factors."""
        all_designs = self.db.get_all_designs()
        self._fom_history = [float(d["fom"]) for d in all_designs if np.isfinite(float(d["fom"] or -100))]
        for f in self._fom_history:
            if f > self.best_fom:
                self.best_fom = f
        if self.best_fom > -float("inf") and self.best_x is None:
            best = self.db.get_best_feasible()
            if best:
                self.best_x = np.array(json.loads(best["design_vector"]))
        self._best_fom_history = []
        running = -float("inf")
        for f in self._fom_history:
            running = max(running, f)
            self._best_fom_history.append(running)

    def _check_convergence(self) -> bool:
        feasible = self.db.get_feasible_designs()
        if len(feasible) < 5:
            return False
        if self.best_fom <= 0:
            return False
        bo_iters = getattr(self, '_bo_iteration_count', 0)
        min_iters = self.config.optimization.min_iterations
        if bo_iters < min_iters:
            return False
        threshold = self.config.optimization.convergence_threshold
        min_check = 10
        if len(self._fom_history) < min_check:
            return False
        n_check = min(10, len(self._fom_history) // 2)
        recent = self._fom_history[-n_check:]
        best_recent = max(recent)
        if len(self._fom_history) > n_check:
            best_prev = max(self._fom_history[:-n_check])
        else:
            best_prev = recent[0]
        best_improvement = (best_recent - best_prev) / max(1e-10, abs(best_prev))
        if not np.isfinite(best_improvement):
            return False
        return best_improvement < threshold


def smooth_drag_factor(old_factor: float, new_factor: float,
                       tolerance: float) -> tuple[float, bool]:
    """50/50-smooth a new calibration factor into the running factor.

    Returns (smoothed_factor, was_clipped). A single bad calibration cannot
    swing the factor by more than ``tolerance`` per step, so the GP never
    sees a 1.10 -> 0.70 (-36%) jump.
    """
    if tolerance <= 0:
        return float(new_factor), False
    cand = 0.5 * old_factor + 0.5 * new_factor
    if abs(cand - old_factor) / old_factor > tolerance:
        clipped = old_factor * (1.0 + np.sign(cand - old_factor) * tolerance)
        return float(clipped), True
    return float(cand), False


def _abort_reason(bo_n, bo_err_frac, lhs_err_frac, bo_env_frac):
    """Return an abort message when BO evaluations are systemically failing
    with environment-class (missing-module) errors, else None.

    Aborts only when: enough BO iterations have run (>=10), environment-class
    errors dominate recent BO results (>=50%), and the BO error rate far
    exceeds the LHS-phase error rate (>=3x, with a floor so a clean LHS phase
    still guards against small early-BO noise).
    """
    if bo_n < 10:
        return None
    if bo_env_frac < 0.5:
        return None
    if bo_err_frac <= 3.0 * max(lhs_err_frac, 0.02):
        return None
    return (
        f"Systemic BEM environment failure: {int(round(bo_env_frac * 100))}% of "
        f"the last {bo_n} BO evaluations failed with environment-class errors "
        f"(BO error rate {bo_err_frac:.2f} vs LHS {lhs_err_frac:.2f}). Aborting: "
        f"check the interpreter/dependencies, e.g. "
        f"/home/anon/apps/boat/venv/bin/pip install fast_simplification"
    )


def _evaluate_one_wrapper(design_vector, config, output_dir, drag_factor, iteration=None, bem_skip=False):
    from threadpoolctl import threadpool_limits
    n_threads = int(os.environ.get("BOAT_BLAS_THREADS", "4"))
    with threadpool_limits(limits=n_threads):
        return evaluate_low_fidelity(
            design_vector, config,
            output_dir=output_dir,
            drag_factor=drag_factor,
            iteration=iteration,
            bem_skip=bem_skip,
        )
