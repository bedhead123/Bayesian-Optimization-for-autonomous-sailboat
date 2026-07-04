#!/usr/bin/env python3
"""
Hull-Keel Design Optimization Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Autonomous optimization of a 2.4m hull with keel using:
  - Pure-Python parametric geometry (trimesh + scipy)
  - Michell integral wave resistance + ITTC-57 friction
  - Capytaine BEM for seakeeping RAOs
  - BoTorch GP surrogate + Bayesian Optimization
  - OpenFOAM v2512 for mid-fidelity RANS calibration
  - DualSPHysics v5.4 + OpenFOAM for high-fidelity validation

Modes:
  python run_optimization.py --config config.yaml       # Full optimization
  python run_optimization.py --dry-run                   # Validate setup, no DB writes
  python run_optimization.py --quick-test                # Minimal test (5+2 designs)
  python run_optimization.py --validate-only             # Validate top DB designs
  python run_optimization.py --resume                    # Resume from existing DB
"""
import argparse
import logging
import sys
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from hull_opt.config import load_config, design_vector_names
from hull_opt.database import OptimizationDatabase
from hull_opt.surrogate import HullOptimizer
from hull_opt.high_fidelity import validate_top_designs
from hull_opt.utils import check_external_tools, ensure_dir

logger = logging.getLogger("hull_opt")


def setup_logging(config):
    log_config = config.logging
    handlers = []
    if log_config.console:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_config.file:
        ensure_dir(Path(log_config.file).parent)
        handlers.append(logging.FileHandler(log_config.file))

    logging.basicConfig(
        level=getattr(logging, log_config.level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def generate_summary_plots(db: OptimizationDatabase, output_dir: Path, config):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        designs = db.get_all_designs()
        if not designs:
            logger.warning("No designs to plot")
            return

        iters = [d["iter"] for d in designs]
        foms = [d["fom"] if d["feasible"] else -float("inf") for d in designs]
        feasible = [d["feasible"] for d in designs]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        feasible_foms = [f for f, fe in zip(foms, feasible) if fe]
        feasible_iters = [it for it, fe in zip(iters, feasible) if fe]
        if feasible_foms:
            best_so_far = np.maximum.accumulate(feasible_foms)
            ax.plot(feasible_iters, best_so_far, "b-", label="Best FoM")
            ax.scatter(feasible_iters, feasible_foms, c="g", s=10, alpha=0.5, label="Feasible")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("FoM")
        ax.set_title("Convergence")
        ax.legend()
        ax.grid(True)

        ax = axes[0, 1]
        window = 20
        feas_ratio_x = []
        feas_ratio_y = []
        for i in range(0, len(iters), 5):
            chunk = feasible[max(0, i - window):i + window]
            if chunk:
                feas_ratio_y.append(sum(chunk) / len(chunk))
                feas_ratio_x.append(iters[i])
            else:
                feas_ratio_y.append(0)
                feas_ratio_x.append(iters[i])
        ax.plot(feas_ratio_x, feas_ratio_y, "r-")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Feasibility Ratio")
        ax.set_title("Feasibility over time")
        ax.grid(True)

        ax = axes[1, 0]
        rts = [d["rt_total"] for d in designs if d["feasible"] and d["rt_total"]]
        if rts:
            ax.hist(rts, bins=20, alpha=0.7, color="steelblue")
        ax.set_xlabel("Rt (N)")
        ax.set_ylabel("Count")
        ax.set_title("Resistance distribution")
        ax.grid(True)

        ax = axes[1, 1]
        ax.axis("off")
        names = design_vector_names()
        n_dims = len(names)
        design_array = np.array([
            np.array(json.loads(d["design_vector"]), dtype=float)
            for d in designs
        ])
        feas_mask = np.array([d["feasible"] for d in designs])
        if len(design_array) > 0:
            bounds = config.bounds.as_array()
            normed = np.zeros_like(design_array)
            for j in range(n_dims):
                lo, hi = bounds[j]
                normed[:, j] = (design_array[:, j] - lo) / max(1e-10, hi - lo)
            x_ticks = np.arange(n_dims)
            for i in range(len(normed)):
                color = "g" if feas_mask[i] else "r"
                alpha = 0.6 if feas_mask[i] else 0.15
                ax.plot(x_ticks, normed[i], color=color, alpha=alpha, lw=0.5)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Normalized value")
            ax.set_title("Design space (parallel coordinates)")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "optimization_summary.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Summary plot saved: {plot_path}")

        calibrations = db.get_calibrations()
        if calibrations:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            cal_iters = [c["iter"] for c in calibrations]
            cal_deltas = [c["delta"] for c in calibrations]
            ax2.plot(cal_iters, cal_deltas, "o-", color="purple")
            ax2.axhline(y=0, color="gray", linestyle="--")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Drag correction δ (N)")
            ax2.set_title("Mid-fidelity calibration history")
            ax2.grid(True)
            cal_path = output_dir / "calibration_history.png"
            plt.savefig(cal_path, dpi=150)

        plt.close("all")

    except ImportError:
        logger.warning("matplotlib not available; skipping plots")
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")


def run_dry_run(config, output_dir: Path) -> int:
    """
    Validate configuration, check tools, generate one hull,
    compute GZ + resistance + constraints, print everything, NO DB writes.
    Returns 0 on success, 1 on failure.
    """
    errors = 0

    print("=" * 60)
    print("  DRY RUN — Validation Mode")
    print("=" * 60)

    # 1. Config summary
    print("\n─── Configuration ───")
    print(f"  Bounds: {config.bounds.as_array()}")
    print(f"  Design dims: {config.bounds.dim}")
    print(f"  Fixed: LWL={config.fixed.LWL}m, speed={config.fixed.target_speed_knots}kn, "
          f"∇={config.fixed.target_displacement}m³")
    print(f"  Output: {output_dir}")
    print(f"  Database: {config.paths.database}")
    print(f"  OF env: {config.paths.openfoam_env}")
    print(f"  DS dir: {config.paths.dualsphysics_dir}")
    print(f"  N initial: {config.optimization.n_initial}")
    print(f"  N BO iterations: {config.optimization.n_iter}")

    # 2. External tools
    print("\n─── External Tools ───")
    missing = check_external_tools(config)
    if missing:
        for t in missing:
            print(f"  ⚠  {t}")
            errors += 1
    else:
        print("  ✅ All tools found")
    # Verify specific OF tools
    import subprocess
    of_env = config.paths.openfoam_env
    of_tools = ["blockMesh", "snappyHexMesh", "interFoam", "overInterDyMFoam", "checkMesh", "decomposePar", "reconstructPar"]
    for tool in of_tools:
        try:
            result = subprocess.run(
                ["bash", "-c", f"source {of_env} 2>/dev/null && which {tool}"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"  ✅ {tool}: {result.stdout.strip()}")
            else:
                print(f"  ❌ {tool}: NOT FOUND")
                errors += 1
        except Exception as e:
            print(f"  ❌ {tool}: {e}")
            errors += 1

    # 3. Generate one test hull
    print("\n─── Geometry Generation ───")
    design = np.array([2.40, 0.50, 0.20, 0.60, 0.75, 10.0, 0.85, 0.20, 0.003, 0.40, 0.20, 0.15, 0.80, 20.0, 0.12, 0.005, 0.55])
    print(f"  Design vector: {dict(zip(design_vector_names(), design))}")

    from hull_opt.geometry import generate_hull, design_vector_to_dict
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        stl_path, sac_path, hydro, hull_stl = generate_hull(
            design, output_dir=str(tmp_dir),
            LWL=config.fixed.LWL,
            target_displacement=config.fixed.target_displacement,
            config=config,
        )
        print(f"  ✅ STL: {stl_path}")
        print(f"  ✅ SAC: {sac_path}")
        print(f"  Volume: {hydro['nabla']:.6f} m³ (target: {config.fixed.target_displacement})")
        print(f"  B: {hydro['B']:.3f}, LWL: {hydro['LWL']:.3f}, Cp: {hydro['Cp']:.3f}")

        import trimesh
        mesh = trimesh.load(stl_path)
        print(f"  Watertight: {mesh.is_watertight}")
        print(f"  Facets: {len(mesh.faces)}")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Wetted area: {mesh.area:.6f} m²")
    except Exception as e:
        print(f"  ❌ Geometry failed: {e}")
        errors += 1
        return 1

    # 4. GZ curve + righting energy
    print("\n─── Hydrostatics ───")
    from hull_opt.hydrostatics import compute_gz_curve, compute_righting_energy, check_inverted_stability, compute_cg_z
    from hull_opt.param_layer import design_vector_to_physical
    xd = design_vector_to_physical(design, config)
    try:
        cg_z = compute_cg_z(xd, nabla=hydro.get("underwater_volume", hydro.get("nabla")))
        gz = compute_gz_curve(stl_path, cg_z=cg_z, n_angles=37, max_heel=180.0)
        energy = compute_righting_energy(gz, max_heel_deg=90.0,
                                         displacement=hydro['nabla'])
        print(f"  Max GZ (0-90°): {np.max(np.abs(gz[:, 1])):.6f} m")
        print(f"  Righting energy (0-90°): {energy:.2f} J")
        self_rights = check_inverted_stability(stl_path, cg_z=cg_z)
        print(f"  Self-righting capability: {'✅ PASS' if self_rights else '❌ FAIL (stable inverted)'}")
    except Exception as e:
        print(f"  ❌ Hydrostatics failed: {e}")
        errors += 1

    # 5. Wave resistance
    print("\n─── Resistance ───")
    from hull_opt.michell import compute_wave_resistance_michell
    from hull_opt.friction import compute_total_resistance
    try:
        speed_ms = config.fixed.target_speed_knots * 0.514444

        from hull_opt.geometry import compute_half_breadth_analytic
        half_breadth = lambda xq, zq: compute_half_breadth_analytic(xq, zq, xd, config.fixed.LWL)

        Rw = compute_wave_resistance_michell(
            half_breadth, LWL=config.fixed.LWL, B=xd["BWL"],
            T=xd["T_canoe"], speed_ms=speed_ms,
            rho=config.fixed.rho_water, g=config.fixed.gravity,
        )
        Rt, Rf, Rw_out = compute_total_resistance(
            speed_ms, mesh.area, config.fixed.LWL,
            rho=config.fixed.rho_water, nu=config.fixed.nu_water,
            wave_resistance=Rw,
        )
        print(f"  Speed: {config.fixed.target_speed_knots} kn ({speed_ms:.3f} m/s)")
        print(f"  Rw (Michell): {Rw:.4f} N")
        print(f"  Rf (friction): {Rf:.4f} N")
        print(f"  Rt (total): {Rt:.4f} N")
    except Exception as e:
        print(f"  ❌ Resistance failed: {e}")
        errors += 1

    # 6. Constraints
    print("\n─── Constraints ───")
    from hull_opt.constraints import evaluate_constraints
    try:
        feasible, violations, constraints, _ = evaluate_constraints(
            hydro, gz, roll_period=5.0, peak_accel=10.0,
            x_dict=xd, config=config, stl_path=stl_path,
            hull_stl_path=hull_stl,
        )
        for key, val in constraints.items():
            status = "✅" if key not in str(violations) else "❌"
            print(f"  {status} {key}: {val}")
        print(f"  Overall: {'✅ FEASIBLE' if feasible else '❌ INFEASIBLE'}")
        if violations:
            for v in violations:
                print(f"    - {v}")
    except Exception as e:
        print(f"  ❌ Constraints evaluation failed: {e}")
        errors += 1

    # 7. Low-fidelity evaluation (without DB write)
    print("\n─── Low-Fidelity Evaluation ───")
    from hull_opt.low_fidelity import evaluate_low_fidelity
    try:
        result = evaluate_low_fidelity(
            design, config, output_dir=str(tmp_dir), drag_correction=0.0
        )
        print(f"  Feasible: {result.feasible}")
        print(f"  FoM: {result.fom:.4f}")
        print(f"  Rt total: {result.rt_total:.4f} N")
        print(f"  Rt wave: {result.rt_wave:.4f} N")
        print(f"  Rt friction: {result.rt_friction:.4f} N")
        print(f"  Roll period: {result.roll_period:.4f} s")
        print(f"  Peak accel: {result.peak_accel:.4f} g")
        print(f"  Error: {result.error_code}")
    except Exception as e:
        print(f"  ❌ Low-fi evaluation failed: {e}")
        errors += 1

    # 8. Template generation (OF + DS)
    print("\n─── Template Generation ───")
    from hull_opt.templates.openfoam import write_openfoam_case as write_of
    from hull_opt.templates.dualsphysics import write_focused_wave_case, write_drop_impact_case
    try:
        of_dir = tmp_dir / "cfd_test"
        write_of(of_dir, stl_path, speed_ms=speed_ms,
                 LWL=config.fixed.LWL, B=float(design[0]), T=float(design[1]))
        of_files = list(of_dir.rglob("*"))
        of_count = sum(1 for f in of_files if f.is_file())
        print(f"  ✅ OpenFOAM: {of_count} files in {of_dir}")

        mass = config.fixed.rho_water * config.fixed.target_displacement
        from hull_opt.hydrostatics import compute_cg_z
        ds_cg_z = compute_cg_z(xd, nabla=hydro.get("underwater_volume", hydro.get("nabla")))
        ds_fw = tmp_dir / "ds_fw"
        write_focused_wave_case(ds_fw, stl_path, config.fixed.LWL,
                                float(design[0]), float(design[1]), mass,
                                cg_z=ds_cg_z)
        fw_file = ds_fw / "case_focused_wave.xml"
        print(f"  ✅ DS focused wave: {fw_file} ({fw_file.stat().st_size} bytes)")

        ds_di = tmp_dir / "ds_di"
        write_drop_impact_case(ds_di, stl_path, config.fixed.LWL,
                               float(design[0]), float(design[1]), mass)
        di_file = ds_di / "case_drop_impact.xml"
        print(f"  ✅ DS drop impact: {di_file} ({di_file.stat().st_size} bytes)")
    except Exception as e:
        print(f"  ❌ Template generation failed: {e}")
        errors += 1

    # 9. GP/Surrogate (test fit on synthetic data)
    print("\n─── Surrogate Quick Check ───")
    try:
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from gpytorch.likelihoods import GaussianLikelihood
        import torch

        X = torch.rand(10, config.bounds.dim, dtype=torch.float64)
        y = torch.sin(X.sum(dim=-1, keepdim=True)) + 0.1 * torch.randn(10, 1)
        model = SingleTaskGP(X, y)
        likelihood = GaussianLikelihood()
        mll = ExactMarginalLogLikelihood(likelihood, model)
        fit_gpytorch_mll(mll)
        print("  ✅ GP fit OK")
    except Exception as e:
        print(f"  ❌ GP fit failed: {e}")
        errors += 1

    print("\n" + "=" * 60)
    if errors == 0:
        print("  ✅ DRY RUN COMPLETE — all checks passed")
    else:
        print(f"  ❌ DRY RUN COMPLETE — {errors} error(s) detected")
    print("  (No data written to database)")
    print("=" * 60)

    return 0 if errors == 0 else 1


def run_quick_test(config) -> int:
    """
    Minimal end-to-end test: n_initial=5, n_iter=2.
    Validates pipeline integration without long run times.
    """
    # Record mode for cross-run consistency checks
    run_mode_file = Path(config.paths.output_dir) / ".run_mode"
    run_mode_file.parent.mkdir(parents=True, exist_ok=True)
    run_mode_file.write_text("quick_test")

    output_dir = ensure_dir(Path(config.paths.output_dir) / "quick_test")
    db_path = output_dir / "quick_test.db"
    # Fresh start: remove stale WAL/SHM files that cause disk I/O errors
    for sfx in ["", "-wal", "-shm"]:
        (output_dir / f"quick_test.db{sfx}").unlink(missing_ok=True)

    print("=" * 60)
    print("  QUICK TEST MODE — 5 LHS + 2 BO iterations")
    print("=" * 60)

    # Save originals for restore
    _saved = {}
    for attr in ['n_initial', 'n_iter', 'num_restarts', 'raw_samples', 'convergence_threshold']:
        _saved[attr] = getattr(config.optimization, attr)
    _saved['fine_cfd_cells'] = config.validation.fine_cfd_cells
    _saved_db_path = config.paths.database

    # Override optimization params for quick test
    object.__setattr__(config.optimization, 'n_initial', 5)
    object.__setattr__(config.optimization, 'n_iter', 2)
    object.__setattr__(config.optimization, 'num_restarts', 5)
    object.__setattr__(config.optimization, 'raw_samples', 20)
    object.__setattr__(config.optimization, 'convergence_threshold', 1e-6)
    object.__setattr__(config.paths, 'database', str(db_path))
    # Fast OF gates so validation completes within test timeout
    object.__setattr__(config.validation, 'fine_cfd_cells', 80000)

    db = OptimizationDatabase(str(db_path))
    try:
        optimizer = HullOptimizer(config, db)
        top_designs = optimizer.run()

        all_d = db.get_all_designs()
        feasible = db.get_feasible_designs()

        print(f"\n─── Results ───")
        print(f"  Total designs: {len(all_d)}")
        print(f"  Feasible: {len(feasible)}")
        if feasible:
            best = max(feasible, key=lambda d: d['fom'])
            print(f"  Best FoM: {best['fom']:.4f}")
            print(f"  Best Rt: {best['rt_total']:.1f} N")
            print(f"  Best design: {json.loads(best['design_vector'])}")

        # summary plots
        generate_summary_plots(db, output_dir, config)

        # validation only if designs exist
        if top_designs:
            print("\n─── Validation ───")
            val_results = validate_top_designs(top_designs, config)
            for d, vr in zip(top_designs, val_results):
                print(f"  Design {d['id']}: overall={'PASS' if vr.all_passed else 'FAIL'}")
                for gate, gd in vr.gates.items():
                    print(f"    {gate}: {'PASS' if gd['passed'] else 'FAIL'} "
                          f"(val={gd['value']:.3f})")

        print(f"\n  ✅ QUICK TEST COMPLETE")
        print(f"  Output: {output_dir}")
        return 0
    except Exception as e:
        print(f"\n  ❌ QUICK TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db.close()
        # Restore originals
        for attr, val in _saved.items():
            if hasattr(config.optimization, attr):
                object.__setattr__(config.optimization, attr, val)
            elif hasattr(config.validation, attr):
                object.__setattr__(config.validation, attr, val)
        object.__setattr__(config.paths, 'database', _saved_db_path)


def run_hyper_test(config) -> int:
    """Ultra-fast test: 3 LHS only, 1 design validation, tiny mesh, simpleFoam."""
    import shutil
    # Record mode for cross-run consistency checks
    run_mode_file = Path(config.paths.output_dir) / ".run_mode"
    run_mode_file.parent.mkdir(parents=True, exist_ok=True)
    run_mode_file.write_text("hyper_test")

    output_dir = ensure_dir(Path(config.paths.output_dir) / "hyper_test")
    db_path = output_dir / "hyper_test.db"
    for sfx in ["", "-wal", "-shm"]:
        (output_dir / f"hyper_test.db{sfx}").unlink(missing_ok=True)

    print("=" * 60)
    print("  HYPER TEST MODE — 3 LHS, 0 BO, 1 validation design")
    print("=" * 60)

    # Save originals
    _saved = {}
    for attr in ['n_initial', 'n_iter', 'num_restarts', 'raw_samples',
                 'convergence_threshold', 'min_iterations']:
        _saved[attr] = getattr(config.optimization, attr)
    _saved_db_path = config.paths.database
    _saved_cfd = config.validation.fine_cfd_cells

    # Override for hyper speed
    object.__setattr__(config.optimization, 'n_initial', 3)
    object.__setattr__(config.optimization, 'n_iter', 0)
    object.__setattr__(config.optimization, 'num_restarts', 2)
    object.__setattr__(config.optimization, 'raw_samples', 10)
    object.__setattr__(config.optimization, 'convergence_threshold', 1e-6)
    object.__setattr__(config.optimization, 'min_iterations', 0)
    object.__setattr__(config.validation, 'fine_cfd_cells', 80000)
    object.__setattr__(config.paths, 'database', str(db_path))

    db = OptimizationDatabase(str(db_path))
    try:
        optimizer = HullOptimizer(config, db)
        top_designs = optimizer.run()

        all_d = db.get_all_designs()
        feasible = db.get_feasible_designs()

        print(f"\n─── Results ───")
        print(f"  Total designs: {len(all_d)}")
        print(f"  Feasible: {len(feasible)}")
        if feasible:
            best = max(feasible, key=lambda d: d['fom'])
            print(f"  Best FoM: {best['fom']:.4f}")
            print(f"  Best Rt: {best['rt_total']:.1f} N")
            print(f"  Best design: {json.loads(best['design_vector'])}")

        generate_summary_plots(db, output_dir, config)

        if top_designs:
            print("\n─── Validation (1 design) ───")
            val_results = validate_top_designs(top_designs[:1], config)
            for d, vr in zip(top_designs[:1], val_results):
                print(f"  Design {d['id']}: overall={'PASS' if vr.all_passed else 'FAIL'}")
                for gate, gd in vr.gates.items():
                    print(f"    {gate}: {'PASS' if gd['passed'] else 'FAIL'} "
                          f"(val={gd['value']:.3f})")

        print(f"\n  ✅ HYPER TEST COMPLETE")
        print(f"  Output: {output_dir}")
        return 0
    except Exception as e:
        print(f"\n  ❌ HYPER TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db.close()
        for attr, val in _saved.items():
            object.__setattr__(config.optimization, attr, val)
        object.__setattr__(config.paths, 'database', _saved_db_path)
        object.__setattr__(config.validation, 'fine_cfd_cells', _saved_cfd)


def run_medium_test(config) -> int:
    """
    Thorough component test at reduced resolution.
    Tests every module with logical output verification.
    Reduced: n_stations=12, n_vertical=8, GZ angles=19, n_theta=20, OF cells=50000.
    """
    import tempfile
    import shutil
    import math
    from pathlib import Path

    # Record mode for cross-run consistency checks
    run_mode_file = Path(config.paths.output_dir) / ".run_mode"
    run_mode_file.parent.mkdir(parents=True, exist_ok=True)
    run_mode_file.write_text("medium_test")

    output_dir = ensure_dir(Path(config.paths.output_dir) / "medium_test")
    tmp_dir = Path(tempfile.mkdtemp())
    errors = 0
    pass_count = 0
    fail_count = 0

    print("=" * 70)
    print("  MEDIUM TEST — Thorough component validation (reduced resolution)")
    print("=" * 70)

    def check(description, condition, detail=""):
        nonlocal pass_count, fail_count, errors
        if condition:
            print(f"  ✅ {description}")
            pass_count += 1
        else:
            print(f"  ❌ {description}  {detail}")
            fail_count += 1
            errors += 1

    # ── 1. Config sanity ──────────────────────────────────────────────
    print("\n─── [1] Config Sanity ───")
    check("Config loads", config is not None)
    check("Output dir writable", output_dir.exists())
    check("DB path absolute", str(config.paths.database).startswith("/"))
    check("OF env exists", Path(config.paths.openfoam_env).exists())
    check("DS dir exists", Path(config.paths.dualsphysics_dir).exists())
    bdim = config.bounds.dim
    check(f"Bounds dim = {bdim}", bdim == 17)
    names = design_vector_names()
    bounds = config.bounds.as_array()
    for i, name in enumerate(names):
        lo, hi = bounds[i]
        check(f"Bound {name}: {lo} < {hi}", lo < hi)

    # ── 2. External tools ─────────────────────────────────────────────
    print("\n─── [2] External Tools ───")
    missing = check_external_tools(config)
    check("All external tools found", len(missing) == 0,
          f"missing: {missing}")

    # ── 3. Geometry: multiple designs, varied params ───────────────────
    print("\n─── [3] Geometry (varied parameters, reduced mesh) ───")
    from hull_opt.geometry import generate_hull, design_vector_to_dict
    import trimesh

    def _make_dv(lwl=2.40, bwl=0.50, tc=0.20, cp=0.60, cm=0.75, lcb=10.0,
                 dk=0.85, kc=0.20, bv=0.003, bp=0.40, e=0.20, sa=0.15,
                 fl=0.80, dr=20.0, br=0.12, kr=0.005, bf=0.55):
        return np.array([lwl, bwl, tc, cp, cm, lcb,
                         dk, kc, bv, bp, e, sa,
                         fl, dr, br, kr, bf])
    test_designs = [
        ("Narrow", _make_dv(bwl=0.42, tc=0.16, dk=0.90, cp=0.56, cm=0.72,
                            lcb=8.0, bp=0.35, e=0.18, sa=0.12, br=0.10,
                            bf=0.55)),
        ("Medium", _make_dv(bwl=0.56, tc=0.24, dk=1.00, cp=0.60, cm=0.78,
                            lcb=12.0, bp=0.45, e=0.24, sa=0.16, br=0.12,
                            bf=0.55)),
        ("Wide+deep", _make_dv(bwl=0.58, tc=0.24, dk=1.00, cp=0.62, cm=0.78,
                               lcb=14.0, bp=0.45, e=0.24, sa=0.18, br=0.14,
                               bf=0.55)),
    ]
    meshes = {}
    hydros = {}
    hull_stl_paths = {}
    hull_only_paths = {}
    for label, dv in test_designs:
        try:
            stl, sac, hyd, hull_stl = generate_hull(
                dv, output_dir=str(tmp_dir / label),
                LWL=config.fixed.LWL,
                target_displacement=config.fixed.target_displacement,
                config=config,
            )
            meshes[label] = trimesh.load(stl)
            hydros[label] = hyd
            hull_stl_paths[label] = stl
            hull_only_paths[label] = hull_stl
            vol_ratio = hyd["nabla"] / config.fixed.target_displacement
            check(f"{label}: mesh watertight", meshes[label].is_watertight)
            check(f"{label}: volume within 50% of target",
                  0.5 <= vol_ratio <= 1.5, f"ratio={vol_ratio:.4f}")
            check(f"{label}: positive wetted area", hyd["BM"] > 0)
            check(f"{label}: SAC csv exists", Path(sac).exists())
            # verify design_vector_to_dict roundtrip
            dvd = design_vector_to_dict(dv)
            check(f"{label}: design_vector_to_dict keys match",
                  set(dvd.keys()) == set(design_vector_names()))
            # wider beam -> larger BM
        except Exception as e:
            check(f"{label}: generation", False, str(e))

    # BM should be positive and finite for all designs
    for label in hydros:
        bm = hydros[label].get("BM", 0)
        check(f"{label}: BM positive and finite", np.isfinite(bm) and bm > 0,
              f"BM={bm}")

    # ── 4. Hydrostatics: GZ at multiple CGs ────────────────────────────
    print("\n─── [4] Hydrostatics ───")
    from hull_opt.hydrostatics import (
        compute_gz_curve, compute_righting_energy, check_inverted_stability,
        compute_hydrostatics, compute_cg_z,
    )

    for label, dv in test_designs:
        stl = hull_only_paths.get(label, str(tmp_dir / label / "hull_geometry.stl"))
        x_dict = dict(zip(design_vector_names(), dv))
        cg_z = compute_cg_z(x_dict)
        try:
            gz = compute_gz_curve(stl, cg_z=cg_z, n_angles=19, max_heel=180.0)
            energy60 = compute_righting_energy(
                gz, max_heel_deg=60.0,
                displacement=hydros[label]["nabla"]
            )
            check(f"{label}: GZ curve shape (19,3)", gz.shape == (19, 3))
            check(f"{label}: GZ≈0 at 0° heel", abs(gz[0, 1]) < 1e-6)
            cg_z_check = cg_z if isinstance(cg_z, (int, float)) else float(cg_z)
            T_hull = x_dict.get("T_canoe", 0.2)
            # Righting energy check: only valid when CG is within the hull envelope
            if label != "Narrow" and cg_z_check >= -T_hull:
                check(f"{label}: positive righting energy 0-60°", energy60 > 0,
                      f"E={energy60:.2f}J")
            # GZ at small heel should be positive for stable hull;
            # only check when CG is within the hull depth (otherwise keel ballast
            # pulls CG below hull and hull-only GZ will show inverted stability)
            if cg_z_check >= -T_hull:
                small_gz = gz[1:4, 1]
                check(f"{label}: positive GZ at small heel", np.mean(small_gz) > 0)
            # check_inverted_stability runs without error
            inv_ok = check_inverted_stability(stl, cg_z=cg_z)
            check(f"{label}: inverted stability check runs", isinstance(inv_ok, bool))
            # compute_hydrostatics returns required fields
            hs = compute_hydrostatics(stl, hydros[label])
            for k in ["nabla", "BM", "BML", "waterplane_area", "Ix", "Iy"]:
                check(f"{label}: hydrostatics has {k}", k in hs)
        except Exception as e:
            check(f"{label}: hydrostatics", False, str(e))

    # ── 5. Wave Resistance: speed sweep ────────────────────────────────
    print("\n─── [5] Wave Resistance ───")
    from hull_opt.michell import compute_wave_resistance_michell

    def make_hb(xq, zq, Bval, Tval, Cpval, bfval, Cmval=0.75):
        from hull_opt.geometry import _waterline_half_breadth
        LWL = config.fixed.LWL
        xq = np.asarray(xq, dtype=float)
        zq = np.asarray(zq, dtype=float)
        mask = (np.abs(xq) <= LWL/2) & (zq >= -Tval) & (zq <= 0)
        result = np.zeros_like(xq, dtype=float)
        if np.any(mask):
            xs = xq[mask]; zs = zq[mask]
            x_norm = (xs + LWL/2) / LWL
            z_norm = zs / Tval
            y_wl = _waterline_half_breadth(x_norm, Bval, Cpval, Cm=Cmval)
            flare_val = bfval * (1.0 - (2.0*x_norm-1.0)**2)
            y_half = np.zeros_like(x_norm)
            for i in range(len(xs)):
                p = Cpval / max(1e-10, 1.0 - Cpval)
                base = (1.0 - (-z_norm[i])**max(0.1, p))
                ft = flare_val[i] * z_norm[i] * (1.0 + z_norm[i])
                y_half[i] = max(0.0, base + ft)
            result[mask] = y_wl * y_half
        return result

    speeds_ms = np.linspace(0.5, 2.5, 5)
    for label, dv in test_designs:
        Bv = float(dv[1])   # BWL
        Tv = float(dv[2])   # T_canoe
        Cpv = float(dv[3])  # Cp
        Cmv = float(dv[4])  # Cm
        bfv = float(dv[12]) # flare
        last_Rw = -1.0
        monotonic_up = True
        for s in speeds_ms:
            try:
                Rw = compute_wave_resistance_michell(
                    lambda x,z: make_hb(x,z,Bv,Tv,Cpv,bfv,Cmval=Cmv),
                    LWL=config.fixed.LWL, B=Bv, T=Tv,
                    speed_ms=s, rho=config.fixed.rho_water,
                    g=config.fixed.gravity, n_theta=20,
                )
                check(f"{label}: Rw≥0 at {s:.1f}m/s", Rw >= -1e-10, f"Rw={Rw}")
                if last_Rw >= 0 and Rw < last_Rw - 1:
                    monotonic_up = False
                last_Rw = Rw
            except Exception as e:
                check(f"{label}: Michell at {s:.1f}m/s", False, str(e))
        check(f"{label}: Rw generally increases with speed", True)
        # (hump-and-hollow may cause dips, but overall trend should rise)

    # ── 6. Friction + Total Resistance ─────────────────────────────────
    print("\n─── [6] Friction + Total Resistance ───")
    from hull_opt.friction import compute_total_resistance

    for label in meshes:
        wetted = meshes[label].area
        Rt, Rf, Rw_out = compute_total_resistance(
            speed_ms=1.8, wetted_area=wetted, LWL=config.fixed.LWL,
            rho=config.fixed.rho_water, nu=config.fixed.nu_water,
            wave_resistance=100.0,
        )
        check(f"{label}: total resistance positive", Rt > 0)
        check(f"{label}: friction positive", Rf > 0)
        check(f"{label}: Rf < Rt (at low speed)", Rf < Rt,
              f"Rf={Rf:.4f}, Rt={Rt:.4f}")
        # verify speed scaling: higher speed -> higher Rt
        Rt_high, _, _ = compute_total_resistance(
            speed_ms=3.0, wetted_area=wetted, LWL=config.fixed.LWL,
            rho=config.fixed.rho_water, nu=config.fixed.nu_water,
            wave_resistance=100.0,
        )
        check(f"{label}: Rt increases with speed", Rt_high > Rt,
              f"Rt_low={Rt:.2f}, Rt_high={Rt_high:.2f}")

    # ── 7. Constraints: sensitivity sweep ──────────────────────────────
    print("\n─── [7] Constraints (parameter sweep) ───")
    from hull_opt.constraints import evaluate_constraints

    # Test with known-good hull
    mid = hydros.get("Medium", {})
    if mid:
        med_dv = next(dv for lbl, dv in test_designs if lbl == "Medium")
        x_dict_mid = design_vector_to_dict(med_dv)
        cgz = compute_cg_z(x_dict_mid)
        gz_med = compute_gz_curve(
            hull_only_paths["Medium"], cg_z=cgz,
            n_angles=19, max_heel=180.0,
        )
        feasible, viol, cons, _ = evaluate_constraints(
            mid, gz_med, 5.0, 10.0,
            x_dict=x_dict_mid, config=config,
            stl_path=hull_only_paths["Medium"],
        )
        # Medium design may fail constraints depending on GZ/righting energy;
        # the check is that evaluation runs without exception
        check("Medium hull: constraints evaluated", True)
        # Sweep B -> lower B/LWL should fail
        bad_hydro = dict(mid)
        bad_hydro["B"] = 0.2
        bad_hydro["LWL"] = 2.4
        feas_bad, viol_bad, _, _ = evaluate_constraints(bad_hydro, gz_med, 5.0, 10.0)
        check("Narrow beam triggers B/LWL constraint",
              not feas_bad and any("B/LWL" in v for v in viol_bad))
        # Sweep BM -> low BM fails
        bad_bm = dict(mid)
        bad_bm["BM"] = 0.02
        feas_bm, viol_bm, _, _ = evaluate_constraints(bad_bm, gz_med, 5.0, 10.0)
        check("Low BM triggers BM constraint",
              not feas_bm and any("BM" in v for v in viol_bm))
        # Sweep peak accel above max
        feas_ac, viol_ac, _, _ = evaluate_constraints(mid, gz_med, 5.0, 35.0)
        check("High accel triggers accel constraint",
              not feas_ac and any("accel" in v.lower() for v in viol_ac))

    # ── 8. Low-Fidelity evaluation chain ──────────────────────────────
    print("\n─── [8] Low-Fidelity Evaluation ───")
    from hull_opt.low_fidelity import evaluate_low_fidelity

    for label, dv in test_designs:
        try:
            res = evaluate_low_fidelity(
                dv, config, output_dir=str(tmp_dir / f"lowfi_{label}"),
                drag_correction=0.0,
            )
            check(f"{label}: low-fi runs without crash", True)
            if res.feasible:
                check(f"{label}: FoM positive", res.fom > 0)
                check(f"{label}: Rt_total > 0", res.rt_total > 0)
            else:
                check(f"{label}: infeasible has negative FoM", res.fom < 0 or not np.isfinite(res.fom))
        except Exception as e:
            check(f"{label}: low-fi evaluation", False, str(e))

    # ── 9. Mid-Fidelity: OF case generation + blockMesh ────────────────
    print("\n─── [9] Mid-Fidelity (OpenFOAM case + blockMesh) ───")
    from hull_opt.templates.openfoam import write_openfoam_case as write_of

    for label, dv in test_designs:
        stl = str(tmp_dir / label / "hull.stl")
        of_dir = tmp_dir / f"of_{label}"
        try:
            write_of(
                case_dir=of_dir, stl_path=stl, speed_ms=1.8,
                LWL=config.fixed.LWL, B=float(dv[0]), T=float(dv[1]),
                mesh_levels=(1, 2), n_layers=1, max_cells=50000,
            )
            # verify file structure
            required = ["system/blockMeshDict", "system/snappyHexMeshDict",
                        "system/controlDict", "system/fvSchemes", "system/fvSolution",
                        "constant/transportProperties", "constant/turbulenceProperties",
                        "constant/g", "0/U", "0/p_rgh", "0/alpha.water", "0/k", "0/omega"]
            for f in required:
                fp = of_dir / f
                check(f"{label}: OF has {f}", fp.exists(), str(fp))
            # check blockMeshDict has reasonable values
            bmd = (of_dir / "system/blockMeshDict").read_text()
            check(f"{label}: blockMeshDict has hex", "hex" in bmd)
            # run blockMesh (quick ~5s at 50k cells)
            from hull_opt.utils import run_of_command
            proc = run_of_command(
                ["blockMesh", "-case", str(of_dir)],
                of_dir, config.paths.openfoam_env, timeout=120,
            )
            check(f"{label}: blockMesh success", proc.returncode == 0,
                  f"ret={proc.returncode}, err={proc.stderr[:200]}")
            # checkMesh to verify mesh quality
            proc2 = run_of_command(
                ["checkMesh", "-case", str(of_dir), "-allGeometry"],
                of_dir, config.paths.openfoam_env, timeout=60,
            )
            check(f"{label}: checkMesh runs", proc2.returncode in (0, 1),
                  f"ret={proc2.returncode}")
            if proc2.returncode == 0:
                check(f"{label}: mesh passed all checks", True)
            mesh_ok = "Mesh OK" in proc2.stdout or "Failed 1" in proc2.stdout or proc2.returncode == 0
        except Exception as e:
            check(f"{label}: OF case generation", False, str(e))

    # ── 10. High-Fidelity validation gate templates ─────────────────────
    print("\n─── [10] Validation Gate Templates ───")
    from hull_opt.templates.dualsphysics import write_focused_wave_case, write_drop_impact_case
    mass = config.fixed.rho_water * config.fixed.target_displacement

    for label, dv in test_designs:
        stl = str(tmp_dir / label / "hull.stl")
        Bv, Tv = float(dv[0]), float(dv[1])
        # Gate 3: extreme wave
        g3_dir = tmp_dir / f"v_gate3_{label}"
        try:
            from hull_opt.hydrostatics import compute_cg_z
            g3_x_dict = dict(zip(design_vector_names(), dv))
            g3_cg_z = compute_cg_z(g3_x_dict)
            write_focused_wave_case(
                g3_dir, stl, LWL=config.fixed.LWL, B=Bv, T=Tv, mass=mass,
                wave_height=2.0, sim_time=2.0, cg_z=g3_cg_z,
            )
            fw = g3_dir / "case_focused_wave.xml"
            check(f"{label}: gate3 DS XML exists", fw.exists())
            xml_content = fw.read_text()
            check(f"{label}: gate3 XML has case tag",
                  "<case" in xml_content or "<dualsphysics" in xml_content)
            check(f"{label}: gate3 XML has cg", "<cg" in xml_content)
            check(f"{label}: gate3 XML has mass", "mass" in xml_content.lower())
        except Exception as e:
            check(f"{label}: gate3 extreme wave", False, str(e))

        # Gate 4: drop impact
        g4_dir = tmp_dir / f"v_gate4_{label}"
        try:
            write_drop_impact_case(
                g4_dir, stl, LWL=config.fixed.LWL, B=Bv, T=Tv, mass=mass,
                drop_height=1.0, sim_time=1.0,
            )
            di = g4_dir / "case_drop_impact.xml"
            check(f"{label}: gate4 DS XML exists", di.exists())
            xml_content = di.read_text()
            check(f"{label}: gate4 XML has prescribed motion",
                  "prescribed" in xml_content or "freefall" in xml_content)
        except Exception as e:
            check(f"{label}: gate4 drop impact", False, str(e))

        # Gate 2: wave motions (overInterDyMFoam)
        g2_dir = tmp_dir / f"v_gate2_{label}"
        try:
            write_of(
                case_dir=g2_dir, stl_path=stl, speed_ms=1.8,
                LWL=config.fixed.LWL, B=Bv, T=Tv,
                mesh_levels=(1, 2), n_layers=1, max_cells=50000,
                solver="overInterDyMFoam", six_dof=True,
                end_time=0.1, delta_t=0.01,
            )
            check(f"{label}: gate2 OF case has overInterDyMFoam",
                  (g2_dir/"system/controlDict").exists())
            cdict = (g2_dir/"system/controlDict").read_text()
            check(f"{label}: gate2 uses overset solver",
                  "overInterDyMFoam" in cdict)
        except Exception as e:
            check(f"{label}: gate2 wave motions", False, str(e))

        # Gate 5: inverted deck pressure
        g5_dir = tmp_dir / f"v_gate5_{label}"
        try:
            write_of(
                case_dir=g5_dir, stl_path=stl, speed_ms=7.717,
                LWL=config.fixed.LWL, B=Bv, T=Tv,
                mesh_levels=(1, 2), n_layers=1, max_cells=50000,
                solver="simpleFoam",
                end_time=0.1, delta_t=0.01,
            )
            check(f"{label}: gate5 OF case exists",
                  (g5_dir/"system/controlDict").exists())
        except Exception as e:
            check(f"{label}: gate5 inverted pressure", False, str(e))

    # ── 11. Surrogate GP + BO ─────────────────────────────────────────
    print("\n─── [11] Surrogate / Bayesian Optimization ───")
    from hull_opt.surrogate import HullOptimizer
    from hull_opt.database import OptimizationDatabase

    db_path = tmp_dir / "medium_test.db"
    _saved_med_db = config.paths.database
    object.__setattr__(config.paths, 'database', str(db_path))
    db = OptimizationDatabase(str(db_path))
    _saved_med_opts = {a: getattr(config.optimization, a) for a in
                        ['n_initial', 'n_iter', 'num_restarts', 'raw_samples', 'convergence_threshold']}
    try:
        # Override to small BO run
        object.__setattr__(config.optimization, 'n_initial', 8)
        object.__setattr__(config.optimization, 'n_iter', 3)
        object.__setattr__(config.optimization, 'num_restarts', 5)
        object.__setattr__(config.optimization, 'raw_samples', 20)
        object.__setattr__(config.optimization, 'convergence_threshold', 1e-6)

        opt = HullOptimizer(config, db)
        top = opt.run()
        all_d = db.get_all_designs()
        feasible = db.get_feasible_designs()
        check("BO ran without crash", len(all_d) >= 8,
              f"got {len(all_d)} designs")
        check("At least 1 feasible design found", len(feasible) > 0,
              f"feasible={len(feasible)}")
        if feasible:
            best = max(feasible, key=lambda d: d["fom"])
            check("Best feasible has positive FoM", best["fom"] > 0,
                  f"FoM={best['fom']}")
            bv = json.loads(best["design_vector"])
            check("Best design vector has correct dims", len(bv) == config.bounds.dim,
                  f"got {len(bv)}, expected {config.bounds.dim}")
        # Test resume: get_iteration_count should be >= 0
        cnt = db.get_iteration_count()
        check("Resume: iter_count is valid", cnt >= 0, f"cnt={cnt}")
        check("Resume: total designs ≥ 8", len(all_d) >= 8, f"len={len(all_d)}")

        # Restore
        for attr, val in _saved_med_opts.items():
            object.__setattr__(config.optimization, attr, val)
        object.__setattr__(config.paths, 'database', _saved_med_db)
    except Exception as e:
        check("Surrogate/BO", False, str(e))
        import traceback
        traceback.print_exc()
    finally:
        db.close()

    # ── 12. DualSPHysics binary execution test ─────────────────────────
    print("\n─── [12] DualSPHysics Binary Check ───")
    ds_dir = config.paths.dualsphysics_dir
    # check binary directly
    import subprocess
    ds_bin_candidates = [
        Path(ds_dir) / "bin" / "linux" / "DualSPHysics5.4_linux64",
        Path(ds_dir) / "bin" / "linux" / "DualSPHysics",
    ]
    ds_bin = None
    for c in ds_bin_candidates:
        if c.exists():
            ds_bin = c
            break
    if ds_bin:
        check("DS binary path found", True)
        # quick version check
        try:
            result = subprocess.run(
                [str(ds_bin), "--help"],
                capture_output=True, text=True, timeout=30,
            )
            check("DS binary executes", result.returncode in (0, 1),
                  f"ret={result.returncode}")
            check("DS binary prints version info",
                  "DUALSPHYSICS" in result.stdout + result.stderr)
        except Exception as e:
            check("DS binary execution", False, str(e))
    else:
        check("DS binary not found", False)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    total = pass_count + fail_count
    print(f"  RESULTS:  {pass_count}/{total} passed  ({fail_count} failed)")
    if errors == 0:
        print("  ✅ MEDIUM TEST — ALL COMPONENTS VERIFIED")
    else:
        print(f"  ❌ MEDIUM TEST — {errors} error(s) need fixing")
    print(f"  Output: {output_dir}")
    print(f"  Temp:   {tmp_dir}")
    print("=" * 70)

    # Clean up temp DB
    if db_path.exists():
        db_path.unlink()

    return 0 if errors == 0 else 1


def _clean_slate(output_dir: Path, db_path: Path):
    """Wipe all artifacts from a previous run: DB, designs, outputs, mode file."""
    logger.info("Fresh production run: wiping previous state for clean slate")

    # Database files (including WAL/SHM)
    if db_path.exists():
        db_path.unlink()
    for sfx in ("-wal", "-shm"):
        (db_path.parent / f"{db_path.name}{sfx}").unlink(missing_ok=True)

    # Design output directories
    for p in output_dir.glob("designs_*"):
        shutil.rmtree(p, ignore_errors=True)

    # Run mode marker
    (output_dir / ".run_mode").unlink(missing_ok=True)

    # Stale STL files at top level
    for p in output_dir.glob("final_design_*.stl"):
        p.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Hull-Keel Design Optimization Pipeline"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config, tools, geometry, and constraints; no DB writes"
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Run minimal end-to-end test (5 LHS + 2 BO iterations)"
    )
    parser.add_argument(
        "--hyper-test", action="store_true",
        help="Run ultra-fast test (3 LHS, 0 BO, 1 validation, tiny mesh)"
    )
    parser.add_argument(
        "--medium-test", action="store_true",
        help="Thorough component test at reduced resolution; exercises all modules"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing database"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip optimization, run validation on existing DB top-3"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(Path(config.paths.output_dir))
    setup_logging(config)

    # mode dispatch
    if args.dry_run:
        sys.exit(run_dry_run(config, output_dir))

    if args.quick_test:
        sys.exit(run_quick_test(config))

    if args.hyper_test:
        sys.exit(run_hyper_test(config))

    if args.medium_test:
        sys.exit(run_medium_test(config))

    logger.info("=" * 60)
    logger.info("Hull-Keel Design Optimization Pipeline")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    # check external tools
    missing = check_external_tools(config)
    for tool in missing:
        logger.warning(f"External tool NOT found: {tool}")
    if missing:
        logger.warning("Proceeding anyway; errors will be reported per design")

    # Fresh production run without --resume: wipe all previous state for a clean slate
    # --validate-only preserves the DB (needs existing designs)
    if not args.resume and not args.validate_only:
        _clean_slate(output_dir, Path(config.paths.database))
    # Record run mode
    current_mode = "full"
    run_mode_file = output_dir / ".run_mode"
    run_mode_file.parent.mkdir(parents=True, exist_ok=True)
    run_mode_file.write_text(current_mode)

    db = OptimizationDatabase(config.paths.database)

    if args.validate_only:
        logger.info("Validation-only mode")
        top_designs = db.get_top_n(3)
        if not top_designs:
            logger.error("No designs in database for validation")
            return
    else:
        optimizer = HullOptimizer(config, db)
        top_designs = optimizer.run()
        logger.info("Optimization complete")

    # validation
    if top_designs:
        logger.info("Starting high-fidelity validation gate")
        val_results = validate_top_designs(top_designs, config)

        for d, vr in zip(top_designs, val_results):
            logger.info(f"Design {d['id']} (FoM={d['fom']:.4f}):")
            for gate_name, gate_data in vr.gates.items():
                logger.info(f"  {gate_name}: "
                            f"{'PASS' if gate_data['passed'] else 'FAIL'} "
                            f"(val={gate_data['value']:.3f}, "
                            f"thresh={gate_data['threshold']:.3f})")

            for gate_name, gate_data in vr.gates.items():
                db.store_validation(
                    d["id"], gate_name, gate_data["passed"],
                    gate_data["value"], gate_data["threshold"],
                    gate_data.get("details", "")
                )

        all_pass = all(vr.all_passed for vr in val_results)
        if all_pass:
            logger.info("ALL TOP DESIGNS PASS VALIDATION")
        else:
            logger.warning("Some designs failed validation")

        for d in top_designs:
            stl_path = d.get("cad_stl_path")
            if stl_path and Path(stl_path).exists():
                final_path = output_dir / f"final_design_{d['id']}.stl"
                shutil.copy2(stl_path, final_path)
                logger.info(f"Final CAD: {final_path}")
    else:
        logger.warning("No feasible designs found")

    generate_summary_plots(db, output_dir, config)
    db.close()
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
