# Architecture

## Data Flow

```
                          ┌─────────────┐
                          │  config.yaml │
                          └──────┬──────┘
                                 │ load_config()
                                 ▼
                          ┌─────────────┐
                          │   Config    │  ← Frozen dataclass tree
                          │  (immutable)│
                          └──────┬──────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌─────────────────┐       ┌─────────────────┐
          │  LHS Sampling   │       │   BO Proposal   │
          │ (n_initial=80)  │       │  (LogEI acq.)   │
          └────────┬────────┘       └────────┬────────┘
                   │                         │
                   └──────────┬──────────────┘
                              │ design_vector (17 floats)
                              ▼
                    ┌─────────────────┐
                    │  evaluate_low_  │
                    │  fidelity()     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌────────────┐ ┌────────────┐ ┌────────────┐
     │  geometry  │ │hydrostatics│ │   wave     │
     │ .py        │ │ .py        │ │ resistance │
     │ STL mesh   │ │ GZ curve   │ │ michell.py │
     └────────────┘ └────────────┘ └────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  constraints.py │
                    │  feasible?      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     FoM =      │
                    │  w1/Rt + w2*SI │
                    │  + w3*SR - ... │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Database      │
                    │   (SQLite)      │
                    └─────────────────┘
```

## Class Hierarchy

```
Config (frozen dataclass)
├── BoundsConfig        - 17 parameter bounds
├── FixedConfig         - LWL, speed, displacement, gravity, etc.
├── WeightConfig        - FoM weights
├── OptimizationConfig  - n_initial, n_iter, GP settings
├── CalibrationConfig   - OF calibration frequency, cells
├── ValidationConfig    - Gate thresholds
├── WaveSpectrumConfig  - JONSWAP params
├── PathConfig          - File paths
└── LoggingConfig       - Log levels

HullOptimizer
├── __init__(config, db)
├── run() → top_designs
├── _initial_sampling()
├── _bo_loop(start_iter)
├── _propose_candidate() → design_vector
├── _eval_one() → EvaluationResult
└── _check_convergence() → bool

EvaluationResult
├── feasible: bool
├── fom: float
├── rt_total, rt_wave, rt_friction: float
├── stability_index, roll_period, peak_accel: float
├── constraint_values: dict
├── constraint_violations: list[str]
└── error_code: str | None

OptimizationDatabase
├── insert_design(), get_design(), get_all_designs()
├── get_feasible_designs(), get_best_feasible(), get_top_n()
├── store_calibration(), get_calibrations(), get_latest_calibration()
└── store_validation(), get_validation_results()

ValidationResult
├── all_passed: bool
├── gates: dict[str, GateResult]
└── per-gate: value, threshold, passed, details
```

## Module Dependencies

```
run_optimization.py
├── hull_opt.config
├── hull_opt.database
├── hull_opt.surrogate
├── hull_opt.high_fidelity
└── hull_opt.utils

low_fidelity.py
├── hull_opt.geometry
├── hull_opt.geometry_validator
├── hull_opt.hydrostatics
├── hull_opt.constraints
├── hull_opt.michell
└── hull_opt.friction

surrogate.py
├── hull_opt.database
├── hull_opt.low_fidelity
├── hull_opt.config
├── hull_opt.utils
├── hull_opt.mid_fidelity
├── botorch.models.SingleTaskGP
└── botorch.acquisition.LogExpectedImprovement

geometry.py
├── hull_opt.config
├── numpy
└── trimesh

hydrostatics.py
├── numpy
└── trimesh

constraints.py
├── hull_opt.hydrostatics
├── numpy
└── trimesh
```

## Execution Modes

### Full Optimization
```
python run_optimization.py
1. Load config.yaml
2. LHS: 80 designs (parallel)
3. BO loop: up to 300 iterations
   - Every 20 iters: mid-fidelity calibration
   - Convergence check after min_iterations
4. Return top 3 designs
5. Run 6 validation gates on each
```

### Dry Run
```
python run_optimization.py --dry-run
1. Validate config
2. Check external tools (OF, DS)
3. Generate one test hull
4. Compute GZ, resistance, constraints
5. Test GP fit on synthetic data
6. NO database writes
```

### Quick Test
```
python run_optimization.py --quick-test
1. Override: n_initial=5, n_iter=2
2. Full pipeline with minimal sampling
3. Validation on tiny OF mesh (80k cells)
```

## Database Schema

```sql
CREATE TABLE designs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iter INTEGER, design_vector TEXT, feasible INTEGER,
    fom REAL, rt_total REAL, rt_wave REAL, rt_friction REAL,
    stability_index REAL, roll_period REAL, peak_accel REAL,
    constraint_values TEXT, constraint_violations TEXT,
    error_code TEXT, cad_stl_path TEXT, cad_sac_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    design_id INTEGER, iter INTEGER,
    rt_low REAL, rt_cfd REAL, delta REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE validation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    design_id INTEGER, gate_name TEXT, passed INTEGER,
    value REAL, threshold REAL, details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
