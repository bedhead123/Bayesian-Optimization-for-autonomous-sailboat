# Agent Guide: Hull-Keel Optimization Pipeline

## File Map

```
run_optimization.py          → CLI entry: --dry-run, --quick-test, --hyper-test, etc.
hull_opt/config.py           → Config dataclasses + YAML loader
hull_opt/geometry.py         → STL mesh generation (17 params)
hull_opt/hydrostatics.py     → GZ curve, righting energy, CG, downflooding
hull_opt/michell.py          → Michell wave resistance integral
hull_opt/friction.py         → ITTC-57 skin friction
hull_opt/constraints.py      → Feasibility constraint evaluation
hull_opt/low_fidelity.py     → Full analytic evaluation orchestration
hull_opt/mid_fidelity.py     → OpenFOAM RANS calibration
hull_opt/high_fidelity.py    → 6 validation gates
hull_opt/surrogate.py        → BoTorch BO loop: HullOptimizer class
hull_opt/database.py         → SQLite: OptimizationDatabase class
hull_opt/utils.py            → LHS, OF runner, memory mgmt, tool checks
hull_opt/check_system.py     → Standalone system validation
hull_opt/templates/          → Jinja2 templates for OF + DS cases
tests/test_*.py              → Per-module pytest tests
webui/app.py                 → FastAPI dashboard
```

## Where to Look for Specific Things

| What you need | File |
|---------------|------|
| How is FoM computed? | `low_fidelity.py` lines 153-210 |
| How does BO propose candidates? | `surrogate.py` lines 227-316 |
| What constraints exist? | `constraints.py` lines 58-151 |
| How is the hull mesh built? | `geometry.py` lines 93-398 |
| How is wave resistance computed? | `michell.py` lines 1-50 |
| How does mid-fi calibration work? | `mid_fidelity.py` |
| Database schema | `database.py` lines 14-40 |
| How does the GZ curve work? | `hydrostatics.py` lines 79-121 |
| Validation gate definitions | `high_fidelity.py` |

## Bug Fix History

See `docs/BUGS_FIXED.md` for a complete log of every bug found and fixed.

Key fixed bugs to be aware of:
- **Michell g² factor** (`michell.py:47`): Prefactor was missing a factor of g. Wave resistance under-predicted by 9.81×.
- **GP likelihood disconnect** (`surrogate.py:272-276`): Likelihood not passed to SingleTaskGP; predictions used untrained noise.
- **GM sign error** (`low_fidelity.py:355`): `GM = BM - \|cg_z\|` → `GM = BM + CB_z - cg_z`. All deep-keel designs got zero roll period.
- **Cm omission** (`geometry.py:513`): Cm defaulted to 0.75 in analytic half-breadth; Michell computed on wrong hull shape.
- **shutil import** (`run_optimization.py`): Missing import crashed DB wipe on mode-switch.
- **Downflooding angle** (`hydrostatics.py:153`): Logic compared against upright max_z instead of rotated deck height; never triggered.
- **Zero-speed division** (`michell.py`): No guard for speed_ms ≤ 0.
- **LWL variable shadowing** (`low_fidelity.py:104`): `half_breadth_func` captured `config.fixed.LWL` instead of the design vector's LWL. Michell integral computed on wrong hull length.
- **Undefined LWL in mid_fidelity** (`mid_fidelity.py:63`): `LWL=LWL` → `LWL=hull_lwl`. Would NameError on first calibration run.
- **min_righting_energy unenforced** (`constraints.py:99-102`): Config threshold was never checked; only `<0` was enforced.
- **Downflooding max_z bug** (`hydrostatics.py:179-182`): Checked all vertices (hull bottom rotating up) instead of deck vertices only.
- **Capytaine peak_accel=0** (`low_fidelity.py:144,150`): Fallback set 0 (passes constraint) instead of 60 (conservative penalty).
- **Rectangular BM overestimate** (`geometry.py:553-554`): BM assumed rectangular waterplane; now uses Cp-dependent waterplane coefficient.
- **Bulb volume scaling** (`geometry.py:484-490`): Non-uniform xz-scaling changed volume by 6.7%; now compensated.
- **Config thresholds** (`config.yaml`): 200J/120° were unrealistic for hull-only GZ; lowered to 75J/85°.
- **Gunwale strip** (`geometry.py:207-219`): Connected keel→sheer instead of waterline→sheer, overlapping hull side faces by 60-80%. Fixed to connect waterline→sheer; also fixed bottom section and bow/stern closures for watertightness.
- **SAC volume override** (`geometry.py:547-552`): Volume reported SAC-integrated value (gamed by scaling), not mesh volume. Fixed to use mesh volume.
- **Self-righting heuristic** (`constraints.py:60`): Allowed `mean_gz_high > -0.01` bypassing positive-GZ requirement. Fixed to require `> 0.005`.
- **Cp constraint** (`geometry.py`, `constraints.py`): Used design vector Cp ignoring SAC scaling. Now computes actual Cp from station areas.
- **Reserve buoyancy high-fi** (`high_fidelity.py:157`): Used hull-only mesh instead of full mesh. Fixed.
- **Convergence check** (`surrogate.py:398-418`): Used best-so-far values instead of actual FoMs, causing premature convergence. Fixed with `_fom_history`.
- **CG_z mismatch** (`hydrostatics.py:164-175`): Ballast split equally but masses are geometry-dependent. Now uses actual mass distribution.
- **SAC cap 3.0→2.5** (`geometry.py`): Mismatched with constraint boundary. Aligned.
- **Resume LHS non-deterministic** (`surrogate.py`): No seed caused different designs on crash-resume. Fixed with fixed seed.
- **Missing UNIQUE on iter** (`database.py`): Duplicate entries from crash recovery. Fixed with UNIQUE index + INSERT OR REPLACE.
- **Downflooding threshold** (`hydrostatics.py:200`): Was already `0.0`, not `0.01` (bug report was stale).
- **B/LWL SAC scaling** (`constraints.py`): Beam constraint didn't account for SAC scaling. Fixed.
- **Normal fixer threshold** (`geometry.py:284`): `vol < 0` missed degenerate `vol == 0` faces. Fixed to `vol <= 0`.
- **Flat validation tolerance** (`geometry_validator.py:180`): ±0.01 tolerance allowed keel_rake=-0.009 (52% below bound). Fixed to relative tolerance.
- **7 missing param checks** (`geometry.py:742-760`): LCB, E, SA, flare, bulb_pos, keel_rake, ballast_frac unvalidated in generate_hull. Added.
- **Cross-param validation** (`geometry_validator.py:169-181`): D_keel>LWL, T_canoe+D_keel>LWL not checked. Added.
- **Exception swallowing** (`geometry.py:153,338,348,389`): 5 validators used `except Exception: pass`. Fixed to propagate errors.
- **Control net angle 135°→100°** (`geometry.py:229`): Was letting 45° kinks through. Tightened.
- **Convexity 0.25→0.35** (`geometry.py:334`): Was letting severely wrinkled hulls through.
- **SAC station ratio 3.0→2.5** (`geometry.py:209`): Was letting fluted hulls through.
- **STL export before check** (`geometry.py:976-979`): Corrupt STL written before gradient check. Moved check before export.
- **Combined mesh missing checks** (`geometry.py:1052-1065`): No spike/convexity check on hull+keel+bulb. Added.
- **Min face count** (`geometry.py:921`): No minimum after mesh fixing. Added 500-face minimum.
- **SAC clip 2.5→5.0** (`geometry.py:813`): Clip prevented volume convergence. Widened.
- **Downflooding 40°→85°** (`config.yaml:69`): Bug #13 fix partially applied. Fixed.
- **Wind heeling 2% dead code** (`constraints.py:350`): 2% sail area never triggered. Changed to 15%.
- **Self-righting heuristic** (`constraints.py:220`): Could bypass GZ data. Restricted to poor-resolution cases.
- **Volume error 35%→25%** (`constraints.py:260`): Threshold too lenient. Tightened.
- **Hard-coded Cp/AR bounds** (`constraints.py:281,311`): Not config-derived. Fixed.
- **NaN from acquisition** (`surrogate.py:416-423`): Not sanitized before use. Added fallback.
- **High-fi skips bounds check** (`high_fidelity.py:60`): No pre-validation. Added.
- **bulb_pos fallback bounds** (`geometry_validator.py:193`): (0.0,1.0) vs config (0.30,0.50). Matched.

## Conventions

- Frozen dataclasses for config (immutable after loading)
- numpy arrays for design vectors, torch tensors for GP training
- All geometry functions in `geometry.py` use `float64`
- SAC scaling capped at 3.0 to prevent balloon sections
- GZ curves computed on hull-only mesh (keel/bulb excluded)
- CG_z from `compute_cg_z()` includes keel and bulb mass via ballast fraction
- Tests use `pytest` (no test runner config needed)
- OpenFOAM case files generated via Jinja2 templates in `templates/`

## Running Tests

```bash
cd /home/anon/apps/boat
python -m pytest tests/ -v                      # All unit tests
python -m pytest tests/test_geometry.py -v      # Single test file
python -m pytest tests/ -k "not quick_test"     # Skip long-running tests
```

## Optimization Loop Lifecycle

1. `HullOptimizer.run()` → `_initial_sampling()` (LHS)
2. → `_bo_loop()` (BO iterations)
3. Each iteration: `_propose_candidate()` (GP acquisition) → `_eval_one()` (Ray/process pool)
4. Every N iterations: `run_mid_fidelity_calibration()` (OpenFOAM)
5. On convergence or exhaustion: return top 3 designs
6. → `validate_top_designs()` (high-fidelity gates)
