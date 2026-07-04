# Module Reference

## `hull_opt/config.py`
**Role:** YAML configuration → frozen dataclass tree.

| Export | Type | Description |
|--------|------|-------------|
| `load_config(path)` | function | Reads YAML, resolves paths, returns `Config` |
| `Config` | dataclass | Root config with 9 sub-configs |
| `BoundsConfig` | dataclass | 17 parameter bounds as `(lo, hi)` tuples |
| `FixedConfig` | dataclass | LWL, speed, displacement, gravity, rho, nu |
| `OptimizationConfig` | dataclass | n_initial=80, n_iter=300, GP settings |
| `design_vector_names()` | function | Returns list of 17 parameter names |

## `hull_opt/geometry.py`
**Role:** Parametric hull/keel/bulb → watertight STL mesh.

| Export | Description |
|--------|-------------|
| `generate_hull(design_vector, ...)` | Main: returns `(stl_path, sac_path, hydro_dict, hull_stl_path)` |
| `compute_half_breadth_analytic(xq, zq, x_dict, LWL, sac_scale)` | Analytic y(x,z) for Michell integral |
| `design_vector_to_dict(design_vector)` | 17-float array → dict with named keys |
| `_waterline_half_breadth(x_norm, BWL, Cp, Cm)` | Waterline shape function |
| `_section_curve(z_norm, y_wl, T, deadrise, bilge_r, flare)` | Transverse section profile |
| `_sac_form(x_norm, Cp, LCB)` | Sectional Area Curve |
| `_make_keel(chord, depth, ...)` | NACA foil keel mesh |
| `_make_bulb(chord, depth, volume, ...)` | Ellipsoid bulb mesh |
| `_compute_hydrostatics(mesh, ...)` | Volume, CB, BM, CG from mesh |

**Bugs fixed:** Cm parameter now passed to `_waterline_half_breadth` in `compute_half_breadth_analytic` for Michell consistency.

## `hull_opt/hydrostatics.py`
**Role:** Hydrostatic calculations on trimesh hull.

| Export | Description |
|--------|-------------|
| `compute_gz_curve(mesh_path, cg_z, n_angles, max_heel)` | Returns `(angles, gz, volumes)` array |
| `compute_righting_energy(gz_curve, max_heel_deg, ...)` | Area under GZ curve × ρg∇ |
| `compute_cg_z(x_dict)` | CG z-coordinate from ballast fraction |
| `compute_downflooding_angle(mesh_path, cg_z)` | Heel angle where deck submerges |
| `compute_reserve_buoyancy(mesh_path, x_dict)` | Above-water volume ratio |
| `compute_wind_heeling_arm(heel_deg, ...)` | Wind heeling moment arm |
| `check_inverted_stability(mesh_path, cg_z)` | Positive GZ at 140-180° heel |
| `compute_hydrostatics(mesh_path, x_dict)` | Full hydrostatics dict |
| `_waterplane_properties(mesh)` | Waterplane Ix, Iy, area |

**Bugs fixed:** Downflooding angle now checks rotated deck height against z=0 instead of comparing against upright max_z.

## `hull_opt/michell.py`
**Role:** Wave resistance via Michell integral with sigma-transformation.

| Export | Description |
|--------|-------------|
| `compute_wave_resistance_michell(half_breadth_func, LWL, B, T, speed_ms, ...)` | Returns Rw in Newtons |

**Algorithm:** Uses σ = 1/cos(θ) substitution to avoid cos³θ singularity. Integrates over σ ∈ [1, 4] with `n_theta=60`. Double integral over hull surface uses `np.gradient` for ∂f/∂x.

**Bugs fixed:** Prefactor changed from `(4ρg)/(πV²)` to `(4ρg²)/(πV²)` for correct units (Newtons). Zero-speed guard added.

## `hull_opt/friction.py`
**Role:** ITTC-1957 skin friction line.

| Export | Description |
|--------|-------------|
| `compute_frictional_resistance(speed_ms, wetted_area, LWL, ...)` | Rf in Newtons |
| `compute_total_resistance(speed_ms, wetted_area, LWL, ..., wave_resistance)` | Returns `(Rt, Rf, Rw)` tuple |

Uses form factor = 0.1 (10% increase for 3D effects).

## `hull_opt/constraints.py`
**Role:** Feasibility constraint evaluation.

| Export | Description |
|--------|-------------|
| `evaluate_constraints(hydro, gz_curve, roll_period, peak_accel, x_dict, config, stl_path)` | Returns `(feasible, violations_list, constraints_dict)` |

**Constraints checked:**
- B/LWL ∈ [0.15, 0.30]
- Cp ∈ [0.55, 0.65]
- BM > 0.03 m
- Volume error < 35%
- Minimum displacement > 0.02 m³ and > 50% of target
- SAC scale factor ∈ [0.5, 2.8]
- Righting energy > 0 J
- Self-righting at 150-180°
- Peak acceleration < 30g
- Keel aspect ratio ∈ [1.5, 8.0]
- Ballast moment < 0.75
- Ballast ratio > min_ballast_ratio
- Reserve buoyancy > min_reserve_buoyancy
- Downflooding angle > min_downflooding_angle
- Wind heeling equilibrium < 45°

## `hull_opt/low_fidelity.py`
**Role:** Full analytic evaluation pipeline orchestration.

| Export | Description |
|--------|-------------|
| `evaluate_low_fidelity(design_vector, config, output_dir, drag_correction)` | Returns `EvaluationResult` |
| `EvaluationResult` | Class with feasible, fom, rt_*, stability, constraints |
| `_compute_raos_capytaine(stl_path, config, x_dict, hydro, speed_ms)` | Capytaine BEM for RAOs |
| `_compute_peak_accel(...)` | JONSWAP-based peak acceleration |
| `_estimate_roll_period(hydro, x_dict)` | Simplified roll period if Capytaine unavailable |

**Evaluation order:**
1. Geometry generation → validate
2. GZ curve (hull-only mesh, system CG)
3. Michell wave resistance + ITTC-57 friction
4. Capytaine RAOs (heave, pitch, roll)
5. Constraints
6. FoM = w1/Rt + w2·SI + w3·SR + light_wind_bonus - w4·accel_penalty - disp_penalty

**Bugs fixed:** GM formula corrected from `BM - |cg_z|` to `BM + CB_z - cg_z` for roll period.

## `hull_opt/mid_fidelity.py`
**Role:** OpenFOAM RANS calibration.

| Export | Description |
|--------|-------------|
| `run_mid_fidelity_calibration(design_vector, design_id, iter, config)` | Returns drag correction delta in Newtons |

Generates geometry → writes OF case → blockMesh → snappyHexMesh → interFoam → extracts forces → compares with low-fi prediction.

## `hull_opt/high_fidelity.py`
**Role:** 6-gate validation suite for top designs.

| Export | Description |
|--------|-------------|
| `validate_top_designs(top_designs, config)` | Returns list of `ValidationResult` |
| `ValidationResult` | Class with `all_passed: bool` and `gates: dict` |

**Gates:**
1. **calm_water_rt**: Fine CFD calm-water resistance (interFoam)
2. **wave_motions_accel**: Regular wave 6-DOF (overInterDyMFoam)
3. **extreme_wave_self_right**: Extreme wave self-righting (hydrostatic)
4. **drop_impact_accel**: Drop impact 6-DOF (DualSPHysics + interFoam)
5. **inverted_pressure**: Inverted deck pressure (simpleFoam)
6. **downflooding**: Downflooding angle + reserve buoyancy

## `hull_opt/surrogate.py`
**Role:** Bayesian Optimization loop.

| Export | Description |
|--------|-------------|
| `HullOptimizer` | Main optimizer class |
| `HullOptimizer.run()` | Returns top 3 designs |
| `HullOptimizer._propose_candidate()` | BoTorch acquisition optimization |
| `HullOptimizer._initial_sampling()` | LHS with parallel eval |
| `_evaluate_one_wrapper(design_vector, config, output_dir, drag_correction)` | Worker function |

**GP setup:** SingleTaskGP with GaussianLikelihood, LogExpectedImprovement acquisition. Falls back from GPU to CPU on OOM.

**Bugs fixed:** Likelihood now passed to SingleTaskGP constructor (was creating orphan likelihood that never trained model predictions).

## `hull_opt/database.py`
**Role:** SQLite database for persistent optimization state.

| Export | Description |
|--------|-------------|
| `OptimizationDatabase(db_path)` | Main class |
| `.insert_design(...)` | Store iteration result |
| `.get_feasible_designs()` | All designs passing constraints |
| `.get_best_feasible()` | Highest FoM feasible design |
| `.get_top_n(n)` | Top n by FoM |
| `.store_calibration(...)` | Mid-fi calibration result |
| `.store_validation(...)` | Validation gate result |

## `hull_opt/utils.py`
**Role:** Shared utilities.

| Export | Description |
|--------|-------------|
| `check_external_tools(config)` | Verify OF and DS availability |
| `run_of_command(cmd, case_dir, env_path, timeout)` | Run OF command with sourced environment |
| `extract_openfoam_force(case_dir)` | Parse drag from postProcessing |
| `knots_to_ms(knots)` / `ms_to_knots(ms)` | Unit conversion |
| `latin_hypercube_sample(n, d)` | LHS design matrix |
| `scale_lhs_to_bounds(lhs, bounds)` | Map [0,1]^d → parameter space |
| `ensure_dir(path)` | mkdir -p |
| `MemoryManager` | GPU detection, worker throttling |

## `hull_opt/geometry_validator.py`
**Role:** Mesh quality checks.

| Export | Description |
|--------|-------------|
| `validate_hull_geometry(stl_path)` | Returns `(is_valid, message)` |
| `validate_design_vector(design_vector, config)` | Returns `(is_valid, message)` |

Checks: watertightness, volume bounds, aspect ratio, body count, finite values, parameter ranges.

## `hull_opt/check_system.py`
**Role:** Standalone system validation script.

Run from project root:
```bash
python -m hull_opt.check_system
```

Checks: config sanity, Python dependencies, OpenFOAM env file, DualSPHysics GenCase binary, database connection.
