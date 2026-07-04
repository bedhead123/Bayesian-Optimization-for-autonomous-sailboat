"""
Bayesian Optimization surrogate using BoTorch (SingleTaskGP + LogEI).
LHS initial sampling → BO loop with mid-fidelity calibration →
convergence checking. Uses Ray or ProcessPoolExecutor for parallel evaluation.
Key exports: HullOptimizer
Bugs fixed: GP likelihood disconnected from SingleTaskGP (#2)
"""
import json
import numpy as np
import torch
import logging
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from hull_opt.database import OptimizationDatabase
from hull_opt.low_fidelity import evaluate_low_fidelity, EvaluationResult
from hull_opt.config import Config, design_vector_names
from hull_opt.param_layer import flattened_bounds
from hull_opt.utils import (latin_hypercube_sample, scale_lhs_to_bounds,
                             ensure_dir, MemoryManager)
from hull_opt.mid_fidelity import run_mid_fidelity_calibration


class HullOptimizer:
    def __init__(self, config: Config, db: OptimizationDatabase):
        self.config = config
        self.db = db
        self.bounds = flattened_bounds(self.config)
        self.dim = len(self.bounds)
        self.output_dir = ensure_dir(Path(config.paths.output_dir))
        self.drag_correction = 0.0
        self.best_fom = -float("inf")
        self.best_x = None

        self.gp_model = None
        self.gp_likelihood = None
        self.cls_model = None
        self._best_fom_history = []
        self._fom_history = []
        self.mem = MemoryManager()

    def run(self):
        logger.info("Starting hull optimization pipeline")
        self._interrupted = False
        def _sigint_handler(signum, frame):
            logger.warning("Interrupted, shutting down...")
            self._interrupted = True
        original_handler = signal.signal(signal.SIGINT, _sigint_handler)

        n_workers = min(self.config.optimization.n_initial, 4)

        # Start persistent parallel pool
        self._pool = self._start_pool(n_workers)

        try:
            if self._interrupted:
                self._stop_pool()
                signal.signal(signal.SIGINT, original_handler)
                return self.db.get_top_n(3)

            existing_count = self.db.get_iteration_count()
            if existing_count == 0:
                self._initial_sampling()
                start_iter = self.config.optimization.n_initial
            elif existing_count < self.config.optimization.n_initial:
                logger.info(f"Resuming incomplete initial sampling "
                           f"(found {existing_count}/{self.config.optimization.n_initial})")
                self._resume_initial_sampling(existing_count)
                start_iter = self.config.optimization.n_initial
            else:
                start_iter = existing_count
                logger.info(f"Resuming from iteration {start_iter}")

            if not self._interrupted:
                self._bo_loop(start_iter)
        finally:
            signal.signal(signal.SIGINT, original_handler)
            self._stop_pool()

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
        try:
            import ray
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

    def _eval_one(self, design_vector, iteration=None):
        if self._pool_type == "ray" and self._ray_remote is not None:
            import ray
            return ray.get(self._ray_remote.remote(
                design_vector, self.config, str(self.output_dir), self.drag_correction, iteration
            ))
        if self._pool_type == "process" and self._process_pool is not None:
            future = self._process_pool.submit(
                _evaluate_one_wrapper,
                design_vector, self.config, str(self.output_dir), self.drag_correction, iteration
            )
            return future.result()
        return _evaluate_one_wrapper(
            design_vector, self.config, str(self.output_dir), self.drag_correction, iteration
        )

    def _initial_sampling(self):
        lhs_min = getattr(self.config.optimization, 'lhs_min', 20)
        lhs_inc = getattr(self.config.optimization, 'lhs_increment', 10)
        lhs_max = getattr(self.config.optimization, 'lhs_max', 40)
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
            results = self._evaluate_batch(designs, list(range(total_done, total_done + n_this)))

            for i, res in enumerate(results):
                if res is None:
                    continue
                self.db.insert_design(
                    iter_num=total_done + i, design_vector=designs[i],
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
                )
                self._fom_history.append(res.fom)
                if res.fom > self.best_fom:
                    self.best_fom = res.fom
                    self.best_x = designs[i].copy()
                self._best_fom_history.append(self.best_fom)

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
        feasible_ratio = sum(1 for d in all_designs if d["feasible"]) / len(all_designs)
        recent = all_designs[-10:]
        recent_ratio = sum(1 for d in recent if d["feasible"]) / max(len(recent), 1)
        return abs(recent_ratio - feasible_ratio) < 0.05

    def _evaluate_batch(self, designs, indices, start_iteration=None) -> list:
        n_workers = self.mem.safe_worker_count(min(len(designs), 4), per_process_gb=3.5)
        if start_iteration is None:
            start_iteration = indices[0] if indices else 0
        if self._pool_type == "ray":
            import ray
            futures = [self._ray_remote.remote(
                designs[i], self.config, str(self.output_dir), self.drag_correction,
                start_iteration + i
            ) for i in range(len(designs))]
            return ray.get(futures)
        else:
            if len(designs) == 1:
                return [self._eval_one(designs[0], iteration=start_iteration)]
            results_by_index = {}
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _evaluate_one_wrapper, designs[i], self.config,
                        str(self.output_dir), self.drag_correction,
                        start_iteration + i
                    ): i for i in range(len(designs))
                }
                for f in as_completed(futures):
                    idx = futures[f]
                    try:
                        results_by_index[idx] = f.result()
                    except Exception as e:
                        logger.error(f"Worker {idx} failed: {e}")
                return [results_by_index.get(i, None) for i in range(len(designs))]

    def _resume_initial_sampling(self, start_idx: int):
        logger.info(f"Resuming LHS initial sampling from index {start_idx}")
        lhs_seed = getattr(self.config.optimization, 'lhs_seed', 42)
        lhs_raw = latin_hypercube_sample(
            self.config.optimization.n_initial, self.dim, seed=lhs_seed
        )
        designs = scale_lhs_to_bounds(lhs_raw, self.bounds)

        n_workers = self.mem.safe_worker_count(min(self.config.optimization.n_initial - start_idx, 4), per_process_gb=3.5)
        logger.info(f"Evaluating {self.config.optimization.n_initial - start_idx} remaining designs "
                    f"with {n_workers} parallel workers")

        indices = list(range(start_idx, self.config.optimization.n_initial))
        if self._pool_type == "ray":
            import ray
            futures = [self._ray_remote.remote(
                designs[i], self.config, str(self.output_dir), self.drag_correction, i
            ) for i in indices]
            results = ray.get(futures)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _evaluate_one_wrapper, designs[i], self.config,
                        str(self.output_dir), self.drag_correction, i
                    ): i for i in indices
                }
                results_by_index = {}
                for f in as_completed(futures):
                    idx = futures[f]
                    try:
                        results_by_index[idx] = f.result()
                    except BrokenProcessPool as e:
                        logger.error(f"Worker {idx} process crashed: {e}")
                        break
                    except Exception as e:
                        logger.error(f"Worker {idx} failed: {e}")
                results = [results_by_index.get(i, None) for i in indices]

        for i, res in zip(indices, results):
            if res is None:
                logger.error(f"Worker {i} returned None, skipping")
                continue
            self.db.insert_design(
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
            )
            self._fom_history.append(res.fom)
            if res.fom > self.best_fom:
                self.best_fom = res.fom
                self.best_x = designs[i].copy()
            self._best_fom_history.append(self.best_fom)

        logger.info(f"Resume LHS complete. Feasible: {sum(1 for r in results if r is not None and r.feasible)}/{len(results)}")

    def _bo_loop(self, start_iter: int):
        n_iter = self.config.optimization.n_iter
        self._bo_iteration_count = 0

        for it in range(start_iter, start_iter + n_iter):
            if getattr(self, '_interrupted', False):
                logger.info("BO loop interrupted")
                break
            logger.info(f"BO iteration {it}/{start_iter + n_iter - 1}")
            self._bo_iteration_count += 1

            try:
                candidate = self._propose_candidate()
            except Exception as e:
                logger.error(f"Acquisition optimization failed: {e}")
                candidate = self._random_candidate()

            result = self._eval_one(candidate, iteration=it)

            self.db.insert_design(
                iter_num=it, design_vector=candidate,
                feasible=result.feasible, fom=result.fom,
                rt_total=result.rt_total, rt_wave=result.rt_wave,
                rt_friction=result.rt_friction,
                stability_index=result.stability_index,
                roll_period=result.roll_period,
                peak_accel=result.peak_accel,
                constraint_values=result.constraint_values,
                constraint_violations=result.constraint_violations,
                error_code=result.error_code,
                cad_stl_path=result.cad_stl_path,
                cad_sac_path=result.cad_sac_path,
            )

            self._fom_history.append(result.fom)
            if result.fom > self.best_fom:
                self.best_fom = result.fom
                self.best_x = candidate.copy()
            self._best_fom_history.append(self.best_fom)

            # mid-fidelity calibration
            if it > 0 and it % self.config.calibration.frequency == 0:
                logger.info(f"Calibration at iteration {it}")
                best = self.db.get_best_feasible()
                if best is not None:
                    try:
                        # Validate design has valid geometry before CFD
                        has_stl = best.get("cad_stl_path") and Path(best["cad_stl_path"]).exists()
                        no_error = best.get("error_code") is None
                        if not has_stl or not no_error:
                            logger.warning(f"Calibration skipped: design {best['id']} has invalid geometry")
                            if self.drag_correction != 0.0:
                                logger.info("Keeping previous drag correction")
                        else:
                            best_x = np.array(json.loads(best["design_vector"]))
                            new_delta = run_mid_fidelity_calibration(
                                best_x, best["id"], it, self.config
                            )
                            if new_delta is not None:
                                old_delta = self.drag_correction
                                self.drag_correction = new_delta
                                logger.info(f"Drag correction updated: δ={new_delta:.4f} N")
                                rt_michlet = best.get("rt_total", 0.0) - old_delta
                                self.db.store_calibration(
                                    best["id"], it,
                                    rt_michlet,
                                    rt_michlet + new_delta,
                                    new_delta
                                )
                    except Exception as e:
                        logger.warning(f"Calibration failed: {e}")
                        if self.drag_correction != 0.0:
                            logger.info("Keeping previous drag correction")

            # convergence check
            if self._check_convergence():
                logger.info(f"Converged at iteration {it}")
                break

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
        return rng.uniform(-10.0, 10.0, size=self.dim)

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
        feasible_history = [f for f in self._fom_history if f > 0]
        min_check = 10
        if len(feasible_history) < min_check:
            return False
        n_check = min(10, len(feasible_history) // 2)
        recent = feasible_history[-n_check:]
        best_recent = max(recent)
        if len(feasible_history) > n_check:
            best_prev = max(feasible_history[:-n_check])
        else:
            best_prev = recent[0]
        best_improvement = (best_recent - best_prev) / max(1e-10, abs(best_prev))
        if not np.isfinite(best_improvement):
            return False
        return best_improvement < threshold


def _evaluate_one_wrapper(design_vector, config, output_dir, drag_correction, iteration=None):
    result = evaluate_low_fidelity(
        design_vector, config,
        output_dir=output_dir,
        drag_correction=drag_correction,
        iteration=iteration,
    )
    return result
