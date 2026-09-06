# Agent Guide: Hull-Keel Optimization Pipeline

## File Map

```
run_optimization.py          → CLI entry: --dry-run, --quick-test, --hyper-test, etc.
hull_opt/config.py           → Config dataclasses + YAML loader
hull_opt/param_layer.py      → Raw GP vector ↔ physical params (17-dim) decode
hull_opt/rig.py              → Dual-sail rig: CE, helm angle, helm arm
hull_opt/geometry.py         → STL mesh generation (19 params), NURBS patches
hull_opt/hydrostatics.py     → GZ curve, righting energy, CG, downflooding, NURBS GZ
hull_opt/michell.py          → Michell wave resistance integral
hull_opt/friction.py         → ITTC-57 skin friction
hull_opt/constraints.py      → Feasibility constraint evaluation
hull_opt/low_fidelity.py     → Full analytic evaluation orchestration
hull_opt/mid_fidelity.py     → DualSPHysics calibration (was OpenFOAM RANS)
hull_opt/high_fidelity.py    → 5 validation gates (SPH + rapid-gate)
hull_opt/surrogate.py        → BoTorch BO loop: HullOptimizer class
hull_opt/database.py         → SQLite: OptimizationDatabase class
hull_opt/utils.py            → LHS, OF runner (deprecated), memory mgmt, tool checks
hull_opt/check_system.py     → Standalone system validation
hull_opt/preflight.py        → SPH preflight (binary checks, STL, case validation)
hull_opt/sph_resistance.py   → DualSPHysics towing / inverted-pressure wrapper
hull_opt/geometry_validator.py → Design vector and mesh quality validation
hull_opt/rapid_gates.py      → Rapid gates: R/E/S/P cheap + W storm BEM sweep
hull_opt/templates/          → Jinja2 templates for SPH (dualsphysics.py) + deprecated OF
tests/test_*.py              → Per-module pytest tests
```

## Deleted Files (Phase 1)

The following files were deleted when OpenFOAM was replaced by DualSPHysics:
- `hull_opt/wave_fields.py` — OF-specific wave dispersion
- `hull_opt/monitor.py` — OF force monitoring
- `hull_opt/run_calibration72.py` — Design-72-specific calibration script

## Where to Look for Specific Things

| What you need | File |
|---------------|------|
| How is FoM computed? | `low_fidelity.py` lines 153-210 |
| How is the wingsail rig modeled? | `rig.py` (build_rig, helm_angle_deg, helm_arm) |
| How does BO propose candidates? | `surrogate.py` lines 227-316 |
| What constraints exist? | `constraints.py` lines 58-151 |
| Helm balance / CE-CLR gates | `constraints.py` lines 708-732 |
| How is the hull mesh built? | `geometry.py` lines 93-398 |
| How is wave resistance computed? | `michell.py` lines 1-50 |
| How does mid-fi calibration work? | `mid_fidelity.py` |
| Database schema | `database.py` lines 14-40 |
| How does the GZ curve work? | `hydrostatics.py` lines 79-121 |
| How does NURBS GZ work? | `hydrostatics.py` lines 433-670 (nurbs_gz_curve) |
| Validation gate definitions | `high_fidelity.py` |
| Rapid gate definitions | `rapid_gates.py` |
| Per-design results report | `run_optimization.py` (`generate_results_report`) → output/results.md |
| DualSPHysics towing setup | `sph_resistance.py` (`run_towing_resistance`) |
| NURBS geometry kernel | `geometry.py` (NURBSPatch, _build_nurbs_patches, _tessellate_patches) |

## Bug Fix History

See `docs/BUGS_FIXED.md` for a complete log of every bug found and fixed.

Key fixed bugs to be aware of:
- **Zero-fluid wave cases** (`templates/dualsphysics.py` #139): Storm/focused-wave/drop fillboxes were anchored at `-zdomain` (crest height), protruding below the definition-box floor → 0 fluid. Fillbox now spans the water column `[-zmin, SWL]`; `zmax` clamped ≥ SWL+0.5. All 5 SPH cases verified via GenCase.
- **Empty towing outlet zone** (`templates/dualsphysics.py` #140): `<boxfill>right</boxfill>` face landed at `2·x_out − x_in` (outside definition box) → "No particles data with mkfluid=2" crash. Now `zone_o_x0 = x_out − zone_dx`.
- **Missing keel + drawmove** (`geometry.py`, `templates/dualsphysics.py` #141): Storm/focused-wave hulls had no `drawmove` (floated 1.3–2 m below SWL); all 5 SPH cases used hull-only STL (no keel/bulb). Full-mesh `hull_full.stl` exported; all writers resolve through `_sph_stl_path()` and compute placement from STL bounds. Keel tip at design draft (storm/focused) or `drop_height` above surface (drop).
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
- **E_FOM `w` ordering** (`low_fidelity.py:289-297`): Helm w5 term referenced `w` before `config.weights` assignment. Uses `config.weights` directly.
- **Mast CG in hydrostatics** (`hydrostatics.py` `compute_cg_z`): mast masses (fwd/aft, from `config.fixed`) added above WL; `compute_cg_x` is new (mesh frame bow=0).
- **Mesh frame** (`rig.py`): bow at x=0, stern at x=+LWL (NOT ±LWL/2). `sail_pos_x(pos_frac) = pos_frac * LWL`; `LCB` is %LWL from bow.
- **Slow BO iterations** (`low_fidelity.py`, `surrogate.py`): 5-7 min/iteration → ~17s with BLAS pinning + BEM decimation.
- **Validation integrity** (`high_fidelity.py`, `utils.py`): Stale-data passes from crash reuse fixed; per-validation cleanup; data-quality guards.

## Conventions

- Frozen dataclasses for config (immutable after loading)
- numpy arrays for design vectors, torch tensors for GP training
- All geometry functions in `geometry.py` use `float64`
- SAC scaling capped at 3.0 to prevent balloon sections
- GZ curves computed on hull-only mesh (keel/bulb excluded)
- **Topside flare** (2026-08-30): the wall above the waterline leans inboard at `flare·TOPSIDE_BOW_AMP` (0.5× at the bow, 0.4× at the stern, capped at `TOPSIDE_MAX_DEG` 45°), so the deck edge is wider than the waterline and tapers into a graceful curve at BOTH bow and stern points (`_topside_wall_angle`, used by BOTH `_build_nurbs_control_net` and `compute_half_breadth_analytic`; net and analytic MUST stay in sync or Michell/BEM compute on a different hull than the mesh — `tests/test_geometry.py::test_topside_control_net_matches_analytic` pins this). `_bow_deck_floor` keeps a rounded stem (30% half-beam) instead of a knife blade. The below-waterline section curve (`_section_curve` flare term, bow_factor=1.0) is UNCHANGED — topside flare never alters displacement/waterplane/GZ at a given flare value (`test_topside_underwater_rows_use_section_curve`). Flare bounds are 10–24° (config.yaml + dataclass + config.fast/phase0 — all three must stay aligned); changing bounds re-maps stored raw GP vectors, so resume of an old campaign decodes differently (wipe DB for a fresh campaign).
- **2026-09-02 spec overhaul** (3-paper audit: PYD 5th ed., JFR hull review, Fanhai-T2 SIMP): `target_displacement` 0.10 m³ (LDR 5.2), `target_speed_knots` 3.5 (Fn 0.37); bounds `D_keel` [0.45, 2.50] (INSANE MODE user directive: T/L ≤ 0.30 soft gate + drag price depth; RealityCheck caps T_total at LWL), `keel_chord` [0.18, 0.26], `bulb_vol` [0.002, 0.0065] (sized to hold ballast_frac lead as lead, Bug #163), `ballast_frac` [0.35, 0.55] (PYD fleet 0.25-0.50), `keel_rake` [15, 30]° aft-swept (USNA debris-shedding; geometry builds +x aft, matching `rig.extended_keel_clr_x`), `LCB` [45, 56]; wingsail AR 3.5 (`wingsail_cr_frac` 0.34), tail 35% (JMSE optimum); `min_avs_deg` 110; calibration `frequency: 0` (SPH tow audit-only via --validate-only).
- **Signed lead hard gate** (constraints.py): hydrodynamic CLR = PYD extended-keel rule (25%-chord line at 45% draft, sweep-corrected) via `rig.extended_keel_clr_x`; lead = (CE_x − CLR_x)/LWL must be ≥ −0.5% (lee helm → hard infeasible) with a soft ramp above +8.5%; `lead_pct_lwl`/`T_over_L` persisted in DB + results.md.
- **Michell capped to the Fn-dependent Delft residuary envelope** (`michell.delft_cap_frac`: 0.4%@Fn0.30 → 5%@Fn0.45 from PYD printed bands, last-band above validity; `capped_wave_resistance` is a TRUE `min(Rw,cap)` — Bug #171 killed the `0.05·Rw` leak and the flat-0.035/0.02 pair) at every Rt site; appendage wetted area (keel both sides + bulb) + `fouling_cf_mult` 1.25 enter ITTC friction; FoM drag term = `w1·(fom_drag_reference_n 45 / Rt_ocean)` where Rt_ocean adds keel induced + Hs² added-wave resistance at the balance leeway. JONSWAP spectra self-normalize to m0=(Hs/4)² (Bug #171: neither 5/16 nor 0.0081 fits the Hs² form).
- **Mass model single source of truth**: `hydrostatics._ballast_struct_split` (cap-fit lead IN the bulb at −(T+D), excess mid-fin at −(T+D/2), DERIVED carbon cantilever `m_struct` from root moment M=tip·g·D·safety at −(T+2D/3), Bug #168) is used by `compute_cg_z/x`, `compute_lumped_inertia` and geometry's `_compute_hydrostatics` reporting;
- **Depth is two-sided, never taxed** (Bug #166 pricing, #168 de-tuning, #169 mission): benefits via GM/RE/AVS/drive as before; costs via light-wind Michell at full draft, keel induced + added-wave drag in Rt_ocean, derived root structure mass+VCG, and a priced draft-logistics inconvenience (`draft_logistics_cost` = rate × draft over `draft_free_m` 1.0 m — user directive: inconvenience, not a wall). T/L ≤ 0.30 soft gate + physical cap T_total > LWL. BEM inertia = `compute_lumped_inertia` 6×6 with rotation dofs created AFTER `center_of_mass` is set (mesh-centroid axes produced spurious roll-yaw hydrostatic coupling and ~280° storm roll sigma); beam-seas roll RAO + keel-vortex damping (ζ = 0.3 + 0.25·D/LWL) anchored to the RAO's own resonance.
- **Mission FoM** (Bug #169: ocean crossing, mostly reach/run, gust-ready): balance polar scored in 3 wind bands (5/10/22 kt × 0.25/0.45/0.30) through the same solver — reach-drive per band vs Rt (`w_mission_drive`), heavy-band heel-margin gust reward (`w_gust`), linear leeway penalty past 4° (`w_leeway`), upwind VMG kept for shifts. Per-band boat speeds (user max 5 kt): 2.0 / 3.5 / 5.0 kt. Stored `result.balance` is the 10 kt band (gates + DB keys unchanged); `reach_drive_ops_N` added to the polar dict. New weights live in config.yaml + dataclass + config.fast/phase0 — all four must stay aligned (same rule as bounds).
- **Wipe hygiene** (Bug #170): `_clean_slate` backs a campaign DB holding designs up to `output/.backup_<ts>/`, never unlinks. Tests must use tmp_path DBs (they do); never point a fresh run at a live campaign dir without --resume.
- `nabla`/`underwater_volume` in the hydro dict = `geometry.mesh_displacement(hull_mesh, 0.0)` (trimesh slice below the waterplane on the same hull-only mesh the GZ curve uses — NOT the SAC integral); if final SAC volume < 75% of `target_displacement` → `ValueError("E_DISP: SAC volume ... far from target ...")` (surfaces as `E_GEOM:<msg>` in low-fi)
- CG_z/CG_x from `compute_cg_z/x()` is the SINGLE mass model (payload 15 kg @ deck +0.3 m, ballast IN the bulb, hull structural floor 20 kg, mast masses); geometry's `_compute_hydrostatics` delegates to it when x_dict is passed
- Design vectors are 21-dim: 16 hull params + wingsail_pos (single wingsail mast, %LWL from bow; rig geometry — cr/ct/b, tail, mast mass — is LWL-scaled from `config.fixed.wingsail_*`) + `sheer_bow/sheer_stern/stem_rake_deg` (PINNED [0,0] flat-deck directive, Bug #164) + `forefoot_cut` [0,0.6] (fleet-standard transoceanic cutaway: fraction of local T removed at stem, fixed 0.20 LWL extent, default anchor 0.35; 17/20-dim legacy vectors pad flat + full). Bounds live in config.yaml + dataclass + config.fast/phase0 — all four must stay aligned.
- **Bulb honesty** (Bug #163): full-ellipsoid bulb seated overlapping the aft-swept fin tip; ballast must fit in lead (`bulb_vol·11340`), enforced as downstream violation (geometry stays buildable); keel t/c 12% chord-based; inertia floored + CB_x/electronics-bay consistent with CG.
- Downflooding angle is info-only in low-fi constraints (fully watertight vessel); the high-fi downflooding gate was removed (unpassable for this hull class, see docs/BUGS_FIXED.md)
- Tests use `pytest` (no test runner config needed)
- DualSPHysics case files generated via Jinja2 templates in `templates/dualsphysics.py`
- OpenFOAM templates in `templates/openfoam.py` are deprecated (importable, nothing executes them)

## DualSPHysics Calibration Backend (Default)

- **Calibration is an AUDIT by design** (`calibration.frequency: 0` current — SPH tow audit-only via --validate-only top-3). The tow is a **moving-hull carriage test** (2026-08-25 rewrite): hull = moving mkbound (mvrect constant velocity, `<motion>` lives under `<casedef>`, hull `objreal ref` = geometry creation index 1 — NOT mkbound+offset) through still water, closed tank, damping zones at both ends, NO InOut zones. Both fixed-hull flume variants were contaminated (Bug #157): imposed outlet trapped waves (Fz drift +405 N/s), extrapolated outlet let forced inlet flux pile water up (Fz → 11.6 kN). A moving hull has no global mass balance to violate: Fz bounded ~2 kN.
- **Factor can no longer corrupt the FoM**: `calibration.factor_bounds: [0.5, 2.0]` (config) — a raw SPH/low-fi drag factor outside the band returns `{"valid": False}` → factor stays 1.0, attempt 'invalid', stored FoMs rescored to 1.0 (`surrogate._handle_calibration_result`). The old clip-to-5.0 path (which reshaped every FoM at iter 110 of the 2026-08-24 campaign) is gone; `smooth_drag_factor` no longer short-circuits the first calibration (no instant 1.0→X jumps).
- **Why SPH can't correct drag here**: at dp=0.05 under `max_particles: 400000`, staircase-DBC form drag reads ~16× the CFD-validated low-fi Rt (effective Cf = 78× ITTC-57; keel chord = 3.8 dp; resistance-grade needs L/dp 200-400 = 10-33M particles, Tagliafierro 2021 used 33M for the same-size hull). The tow is a health audit (steady Fx, bounded Fz, towable geometry), not a drag instrument.
- Guards in `run_towing_resistance`: `window_s` (tail averaging, config `convergence.window_s: 1.2`), `min_sim_time` (config `convergence.min_sim_time: 2.0` — 2.2 false-failed design_116 whose dt_out-rounded traces end at 2.10 s), `max_fz_n` = 2×ρg·target_displacement (Fz sanity; uses design displacement NOT hydro nabla — nabla is hull-only and made the bound falsely tight). `read_force_tail_stats` parses Fx+Fz; `calibration_factor` enforces factor_bounds.
- Moving-tow geometry (`write_towing_case` v3): tank = 0.6·LWL front + hull + travel + 0.5·LWL back, width 2.0·B, floor −(T_total + 1.5·keel_chord) capped [−3.2, −0.4], ceiling = deck + 1dp (deck used to get clipped), sim_time auto-capped to the particle budget (~2.9 s at design dims → trace ends ~2.4 s). NO inlet velocity ramp — fluid is not pre-initialized; ramping against pre-initialized motion collapsed dt 4× (2.4e-4 → 5.8e-5). Hull motion ref pitfall: `objreal ref` is the creation index (0=tank box, 1=hull STL); ref=mkbound+offset silently binds nothing (CaseNmoving=0, Fx≡0).
- Runtime: ~5 min per moving tow (364k particles, 2.9 s sim) + ~1 min GenCase/ComputeForces. `frequency: 0` — tows are invalid at dp=0.05 (staircase-DBC ~16× low-fi); audit only via --validate-only top-3.
- Default calibration solver is `"sph"` (config: `calibration.solver`).
- SPH towing: `sph_resistance.py:run_towing_resistance()` builds GenCase + runs solver + ComputeForces.
- SPH inverted pressure: `sph_resistance.py:run_inverted_pressure()`.
- **Gate 1 PASS/FAIL uses the OF-validated low-fi Rt** (`_gate_fine_cfd` in `high_fidelity.py`); the SPH tow runs only as an audit when `validation.sph_towing_audit: true`, and its Rt is logged with the ratio but never decides the gate.
- **Gate 5 (inverted pressure)** needs the envelope tank in `write_inverted_case`: walls above the keel tip (`z_wall_top = zsurf + z_sheer_max + 0.5`, else the solver domain — the initial particle bbox ± (KernelH·0.05 + dp/2), NOT the def box — excludes the bobbing keel tip), `ymax ≥ z_sheer_max + 0.3` for the roll swing, floor at `-1.5·T_total`, hull at equilibrium draft `dz_eq`, CG in tank frame (`-cg_z + dz_eq`, `hull_x + 0.4·LWL`), closed tank (`boxfill` includes `left | right`), `sph_inverted_dp: 0.075` (0.06 gave 650-750k GenCase particles > the 400k cap → unverified fallback; 0.075 ≈ 360k; 0.03 → ~2M). MeasureTool v5.4 writes `{stem}_Press.csv` (not `{stem}.csv`) — `_run_measuretool` (both copies) returns the `_Press.csv`; `_parse_measuretool_csv` accepts `Part;Time [s];Press_k [Pa]` headers.
- GPU dependency: DualSPHysics requires a CUDA-capable GPU (RTX 3050 6GB verified). Solver binaries live in `bin/dualsphysics/5.4/bin/linux/DSGcc7/` (GPU, needs `LD_LIBRARY_PATH=<same dir>` — `find_solver` sets it) with CPU fallback `DualSPHysics5.4CPU_linux64`; full install sourced from `/home/anon/apps/DualSPHysics_v5.4`. GenCase CLI takes the case name WITHOUT the .xml extension.
- **Storm paddle pocket** (Bug #167): `storm_domain` sizes the back pocket from Hs (`max(0.35, Hs+4dp)`) — the deterministic JONSWAP piston return stroke (~Hs in shallow transfer) used to carry paddle particles past the particle-derived domain face (AbortBoundOut, −X) on narrow (B=0.55) hulls. Estimator stays in sync via the single source.
- **Roll sigma capped at 180°** (`_sigma_deg_capped`, Bug #167): near-undamped RAO resonance integrated to 1455–2021° garbage; downstream tanh/gates already neutralized it, now the DB does too.
- dp scaling: particles ∝ 1/dp³. At dp=0.03 the tow tank holds ~1M particles — over `calibration.max_particles` (400k), rejected before the solver runs.
- gpu_lock contract: the main process holds `<output_dir>/gpu.lock` ONLY during the DualSPHysics solver subprocess (GenCase/ComputeForces run outside the lock). Low-fidelity BEM workers (Ray) flock-poll the same lock every 5 s via `_wait_spf_lock(config)` and park for up to 3600 s, so 3 Capytaine dense-LU jobs can't OOM the host while SPH runs. Before solver launch an nvidia-smi VRAM guard checks the estimate (particles × 1.5 kB, gpu tool only); a particle-count cap is enforced right after GenCase parses "Total particles:" from `gencase.out` (abort FAILED if > `calibration.max_particles`). Solver progress goes to `<case_dir>/solver.log`; failures tail only ~250 chars. Result dicts carry `np_particles`. Calibration results log compact `<calib FAILED|INVALID design=.. iter=.. reason=..>` lines.
- DB hygiene (2026-08-25): `calibration_attempts` PK is `(design_id, iter)` (was `design_id` — history overwrites), stale `'running'` rows are scrubbed to `'failed'` at DB open (a crash mid-calibration used to block the design forever).

## Rapid Gates (Tier-1)

- Gate order in `evaluate_rapid_gates`: R (GZ stability) → E (storm wind heel) → S (slam) → P (inverted pressure) — all cheap — then W (storm BEM sweep, 2-4 min), skipped when any cheap HARD gate (avs/wind_heel/righting_energy/max_gz_m/gz_area_30/gz_area_40_90/self_right) already fails: `details["gate_w"] = "storm BEM skipped: cheap hard gate failed (<name>)"`.
- Slam gate is a capped initial-impact model: `p_eff = min(p_wagner, slam_max_pressure_pa)` (rapid_validation, 0.5 MPa), impact force `F = _SLAM_FORCE_FACTOR (0.85) · 0.5·ρ·v²·A_slam·k(β)` with `A_slam = min(A_wet, BWL·T_canoe)`, `k(β) = (π/2)²·max(0.3, cot β)/2`; mass base = design displacement (`target_displacement · ρ`), NOT hydro nabla. Results monotonic in deadrise: 16.6 / 10.1 / 5.8 g at 5°/15°/25°, 3 m drop.
- Soft margins (capsize/storm_accel/slam_pressure/slam_accel/inverted_pressure) no longer inflate `violation_magnitude` — only hard gates do (`_accumulate_margin_violations` in low_fidelity.py); the w6-w9 bonus uses `_margin_bonus_clip` = np.tanh smooth saturation instead of a hard ±1 clip.
- `RapidGateResult.worst_hard` / `worst_hard_value` = most-negative non-soft margin name/value; carried into rapid log lines (`utils.py: format_rapid_summary`) and results.md/CSV + interactive table (run_optimization.py).
- `_storm_bem_sweep` logs an INFO line when the surrogate anchor error threshold is exceeded → full-BEM fallback (no silent slow paths).

## NURBS Geometry Kernel

- `config.fixed.use_nurbs_geometry: false` — uses analytic mesh generation (default, backward-compatible).
- `config.fixed.use_nurbs_gz: false` — uses STL-based GZ (default); true uses NURBS hydrostatics for GZ curve.
- NURBS mode: 4 patches (port hull, starboard hull, keel, bulb) built from control net.
- NURBS GZ: `hydrostatics.py:nurbs_gz_curve()` rotates NURBS control nets analytically (no mesh intersection).
- Patches are serialized as JSON in `nurbs_patches_file` within the hydro dict.

## Running Tests

```bash
cd /home/anon/apps/boat
venv/bin/python -m pytest tests/ -v                # All unit tests (slow tests excluded by default)
venv/bin/python -m pytest tests/test_geometry.py -v   # Single test file
venv/bin/python -m pytest tests/ -k "not quick_test"  # Skip long-running tests
venv/bin/python -m pytest tests/ -m slow              # Long-running end-to-end tests
```

## Optimization Loop Lifecycle

1. `HullOptimizer.run()` → `_initial_sampling()` (LHS)
2. → `_bo_loop()` (BO iterations)
3. Each iteration: `_propose_candidate()` (GP acquisition) → `_eval_one()` (Ray/process pool)
4. Every N iterations: `run_mid_fidelity_calibration()` (DualSPHysics, was OpenFOAM)
5. On convergence or exhaustion: return top 3 designs
6. → `validate_top_designs()` (high-fidelity gates, SPH + rapid-gate)
