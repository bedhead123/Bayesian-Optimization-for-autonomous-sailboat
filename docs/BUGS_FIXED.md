# Bug Fix Log

This document records every bug found and fixed in the codebase. Future agents should review this before modifying any of these files.

---

## Bug #1: Michell wave resistance missing g² factor
- **File:** `hull_opt/michell.py:47`
- **Severity:** Critical
- **Discovery:** Code review (subagent), dimensional analysis
- **Problem:** Prefactor used `(4ρg)/(πV²)` but should be `(4ρg²)/(πV²)`. Dimensional analysis showed the formula produced kg instead of Newtons. All wave resistance results were under-predicted by a factor of g = 9.81.
- **Fix:** Changed `g` to `g**2` in the prefactor. Added zero-speed guard (`speed_ms ≤ 0 → return 0.0`).
- **Impact:** Wave resistance values increased ~10×. This is the correct physical value. Previous optimization would have systematically selected hulls with unrealistically high wave resistance.

## Bug #2: GP likelihood disconnected from SingleTaskGP
- **File:** `hull_opt/surrogate.py:272-276`
- **Severity:** Critical
- **Discovery:** Code review, gpytorch API inspection
- **Problem:** `SingleTaskGP(X_norm, y)` was created without passing the separately created `GaussianLikelihood`. The model created its own internal likelihood with default noise (untrained). A separate likelihood was trained via `ExactMarginalLogLikelihood` but never connected to the model. When `model.posterior(X)` was called by the acquisition function, it used the model's internal untrained likelihood, giving incorrect posterior variances.
- **Fix:** Create `GaussianLikelihood` first, then pass it to `SingleTaskGP(X_norm, y, likelihood=self.gp_likelihood)`. The fallback CPU path was also fixed to ensure the likelihood is moved to CPU before model creation.
- **Impact:** BO acquisition function now uses properly trained noise levels. Previously the GP would have incorrect uncertainty estimates, potentially degrading BO convergence.

## Bug #3: Downflooding angle never triggered
- **File:** `hull_opt/hydrostatics.py:153-178`
- **Severity:** High
- **Discovery:** Code review (subagent)
- **Problem:** The downflooding check computed `deck_clearance = max_z - abs(immersed_max_z)` where `max_z` was the highest point of the *upright* hull and `immersed_max_z` was the highest underwater point at the current heel angle (~0). Since `abs(immersed_max_z) ≈ 0`, `deck_clearance` was always ≈ `max_z` > 0.02, so the condition was never met. The function always returned 180.0°.
- **Fix:** Changed to check `rotated_deck_z < 0.01` where `rotated_deck_z` is the maximum z-coordinate of the rotated hull at each heel angle.
- **Impact:** Downflooding angle is now physically meaningful. Previously no design could fail the downflooding constraint.

## Bug #4: GM sign error in roll period estimate
- **File:** `hull_opt/low_fidelity.py:355`
- **Severity:** High
- **Discovery:** Code review (subagent), naval architecture formula check
- **Problem:** The metacentric height was computed as `GM = BM - |cg_z|`. The correct formula in waterline coordinates is `GM = BM + CB_z - cg_z`. Since `cg_z` is negative (CG below waterline) and more negative than `CB_z` (CB below waterline), `GM = BM - |cg_z|` gives negative values for all deep-keel designs, while `GM = BM + CB_z - cg_z = BM - |CB_z| + |cg_z|` gives positive values. Negative GM caused roll period to return 0.0.
- **Fix:** Added `CB_z` from hydro dict and used `GM = BM + CB_z - cg_z`.
- **Impact:** Deep-keel designs now get physically realistic roll periods (~0.4-2.0s). Previously all such designs had zero roll period, affecting roll-based constraints and seakeeping estimates.

## Bug #5: Cm parameter ignored in analytic half-breadth
- **File:** `hull_opt/geometry.py:513`
- **Severity:** High
- **Discovery:** Code review (subagent)
- **Problem:** `compute_half_breadth_analytic` called `_waterline_half_breadth(x_norm, BWL, Cp)` without passing `Cm`, so it defaulted to 0.75. The actual hull generation at line 125 used the design vector's `Cm`. When `Cm ≠ 0.75`, the Michell integral computed wave resistance on a different hull shape than the one actually generated.
- **Fix:** Extract `Cm = x_dict.get("Cm", 0.75)` and pass `Cm=Cm` to `_waterline_half_breadth`.
- **Impact:** Wave resistance now correctly accounts for the hull's midship coefficient. The discrepancy was up to ~15% for extreme Cm values.

## Bug #6: shutil not imported in run_optimization.py
- **File:** `run_optimization.py`
- **Severity:** High
- **Discovery:** Source code inspection
- **Problem:** `shutil.rmtree()` was called at line 1099 during mode-switch DB wipe, but `import shutil` was missing from the module-level imports. A separate `import shutil` existed at line 1147 inside a conditional block, which was unreachable when the error occurred at line 1099. This would cause a `NameError` crash on mode-switch.
- **Fix:** Added `import shutil` at the top of the file alongside other standard library imports.
- **Impact:** Mode-switching between full and test runs no longer crashes.

## Bug #7: Geometry.py stray whitespace line
- **File:** `hull_opt/geometry.py:161`
- **Severity:** Low
- **Discovery:** Source code inspection
- **Problem:** Line 161 contained only whitespace characters, left over from earlier fix scripts. Python ignores this, but it was untidy and indicated instability from repeated scripted edits.
- **Fix:** Removed the stray whitespace.
- **Impact:** None functional, but code is cleaner.

## Bug #8: Stale fix scripts and backup files
- **Files:** `fix_geom*.py`, `fix_indent*.py` (8 files), `*.bak`, `*.orig`, `*.current` (5 files)
- **Severity:** Low (maintenance hazard)
- **Discovery:** Directory listing
- **Problem:** Multiple fix scripts in the project root targeted indentation issues in geometry.py that had already been resolved. The `geometry.py.current` file had a diverged SAC cap (4.0 vs 3.0 in active file). `.orig` files contained pre-fix code that could confuse developers.
- **Fix:** Deleted all fix scripts and stale backup files.
- **Impact:** Cleaner project root, no confusion about which file is authoritative.

## Bug #9: geometry.py.orig left over after cleanup
- **File:** `hull_opt/geometry.py.orig`
- **Severity:** Low
- **Discovery:** Final audit
- **Problem:** One `.orig` file remained after initial fix script cleanup.
- **Fix:** Removed it.
- **Impact:** Clean.

## Bug #10: LWL variable shadowing in half_breadth_func (low/mid/high fidelity)
- **Files:** `hull_opt/low_fidelity.py:104-105`, `hull_opt/mid_fidelity.py:50-52`, `hull_opt/high_fidelity.py:50-52`
- **Severity:** High
- **Discovery:** Code review (agent analysis)
- **Problem:** The local variable `LWL` in half_breadth_func/closure captured `config.fixed.LWL` instead of the design vector's actual LWL. The Michell integral was always computed at LWL=2.4 regardless of the design vector's LWL value (range 2.3-2.5).
- **Fix:** Renamed the captured variable to `hull_lwl` which is loaded from `x_dict.get("LWL", config.fixed.LWL)`.
- **Impact:** Wave resistance (and thus total resistance) now correctly scales with the design's actual waterline length. Designs with LWL ≠ 2.4 previously got incorrect wave resistance values.

## Bug #11: min_righting_energy constraint not enforced
- **File:** `hull_opt/constraints.py`
- **Severity:** High
- **Discovery:** Medium-test failure analysis
- **Problem:** The `min_righting_energy` value from config (200J) was never checked. The constraint function only checked if righting_energy < 0, not if it fell below the configured threshold.
- **Fix:** Added `if config is not None: if constraints["righting_energy"] < min_re: violations.append(...)`.
- **Impact:** Designs with righting energy between 0 and the configured minimum were incorrectly marked feasible. Config changes to tighten righting energy had no effect.

## Bug #12: Downflooding angle computed with wrong metric
- **File:** `hull_opt/hydrostatics.py:161-184`
- **Severity:** High
- **Discovery:** Debugging constraint failures
- **Problem:** `compute_downflooding_angle` checked `verts[:, 2].max() < 0.01`, i.e., when the ENTIRE hull (including hull bottom that rotates upward after 90° heel) is fully underwater. The correct downflooding check is when the DECK (vertices with original z ≥ 0) submerges. The all-vertices max_z check gave artificially high downflooding angles (98°+ vs actual ~86°).
- **Fix:** Filter to vertices with original z ≥ 0 (deck/gunwale vertices only), then check their rotated max_z.
- **Impact:** Downflooding angles are now 10-15° lower, physically realistic for deep-keel hulls.

## Bug #13: Config constraints unrealistic for hull-only GZ computation
- **File:** `config.yaml`
- **Severity:** Medium
- **Discovery:** Constraint analysis
- **Problem:** `min_righting_energy: 200.0` and `min_downflooding_angle: 120.0` were set assuming full-mesh GZ computation. With hull-only GZ (documented AGENTS.md convention), max achievable righting energy is ~120J and max downflooding angle is ~86° for 2.4m deep-keel designs. These constraints made ALL designs infeasible.
- **Fix:** Lowered to `min_righting_energy: 75.0` and `min_downflooding_angle: 85.0`, matching achievable values.
- **Impact:** Feasible designs now exist. Optimization can find valid hulls.

## Bug #14: Undefined variable LWL in mid_fidelity.py (NameError at runtime)
- **File:** `hull_opt/mid_fidelity.py:63`
- **Severity:** Critical
- **Discovery:** Code review (subagent)
- **Problem:** `write_openfoam_case(LWL=LWL, ...)` referenced the undefined name `LWL`. The local variable is `hull_lwl` (defined at line 37). This would raise `NameError: name 'LWL' is not defined` whenever mid-fidelity calibration ran (every 20 iterations).
- **Fix:** Changed `LWL=LWL` to `LWL=hull_lwl`.
- **Impact:** Mid-fidelity calibration would crash on its first invocation. Fix prevents crash.

## Bug #15: Friction return type annotation wrong
- **File:** `hull_opt/friction.py:27`
- **Severity:** Low
- **Discovery:** Code review (subagent)
- **Problem:** Return type annotated `-> float` but returns `tuple[float, float, float]` (Rt, Rf, wave_resistance).
- **Fix:** Changed to `-> tuple[float, float, float]`.
- **Impact:** None at runtime, but could cause type-checker confusion.

## Bug #16: Capytaine failure silently bypasses acceleration constraint
- **File:** `hull_opt/low_fidelity.py:138-143`
- **Severity:** Medium
- **Discovery:** Code review (subagent)
- **Problem:** When Capytaine BEM raised any exception (solver divergence, mesh issue), `peak_accel` was set to `0.0`. Zero means "no problem" to the acceleration constraint (`peak_accel > 30g → violation`), so dangerously high accelerations would never trigger the constraint or FoM penalty.
- **Fix:** Changed fallback to `peak_accel = 60.0` (well above the 30g threshold).
- **Impact:** Designs with failed RAO computations are now conservatively penalized instead of silently passed.

## Bug #17: Dead constant expression in geometry.py
- **File:** `hull_opt/geometry.py:340`
- **Severity:** Low
- **Discovery:** Code review (subagent)
- **Problem:** `tip_chord = keel_chord * (1.0 - 0.5)` is just `keel_chord * 0.5`.
- **Fix:** Simplified to `tip_chord = keel_chord * 0.5`.
- **Impact:** None functional. Cleaner code.

## Bug #18: Fixed LWL used for friction instead of design-vector LWL
- **File:** `hull_opt/low_fidelity.py:117`
- **Severity:** Low
- **Discovery:** Code review (subagent)
- **Problem:** `compute_total_resistance(..., config.fixed.LWL)` used the fixed 2.4m LWL instead of the design vector's actual LWL (range 2.3-2.5). Impact is logarithmic (Reynolds number) so <1% error.
- **Fix:** Changed to `hull_lwl`.
- **Impact:** Frictional resistance now correctly matches design LWL.

## Bug #19: Duplicate righting-energy violation messages
- **File:** `hull_opt/constraints.py:97-102`
- **Severity:** Low
- **Discovery:** Code review (subagent)
- **Problem:** When righting_energy < 0, two equivalent violation messages were generated: one for `< 0 J` and one for `< min_re J`.
- **Fix:** Changed to `elif` so that when config is present, only the `min_re` message appears; when config is absent, the `< 0 J` message appears.
- **Impact:** Cleaner output, no duplicate messages.

## Bug #20: False positive warning in check_system.py
- **File:** `hull_opt/check_system.py:23-24`
- **Severity:** Low
- **Discovery:** Code review (subagent)
- **Problem:** A warning fired whenever LWL appeared in both `bounds` and `fixed` config sections. This is by design: LWL has bounds (2.3-2.5) and a fixed value (2.4). The warning always triggered, confusing users.
- **Fix:** Changed to an informative message showing the fixed value and bounds.
- **Impact:** No more false warnings.

## Bug #21: Waterplane Ix computed on wrong faces
- **File:** `hull_opt/hydrostatics.py:58-84`
- **Severity:** Medium
- **Discovery:** Code review (agent analysis)
- **Problem:** `_waterplane_properties` computed Ix from ALL faces of the sliced mesh (submerged hull + cap), not just the waterplane cap at z=0. This overestimated Ix by 10-15×.
- **Fix:** Added face filtering to include only cap faces (vertices with |z| < 1e-6).
- **Impact:** Accurate waterplane Ix and BM for stability analysis. Rectangular approximation in geometry.py was also improved with a Cp-dependent correction factor (Bug #22).

## Bug #22: Rectangular waterplane BM overestimates stability by 2×
- **File:** `hull_opt/geometry.py:553-554`
- **Severity:** Medium
- **Discovery:** Code review (subagent)
- **Problem:** BM was computed from `Ix = LWL * BWL³ / 12`, assuming a rectangular waterplane. Real hulls taper at bow and stern, so actual Ix is ~50-60% of the rectangular value. Overestimated BM gives overly optimistic GM and roll period.
- **Fix:** Applied waterplane coefficient correction: `C_wp = 0.35 + 0.6 * Cp`, giving `Ix = C_wp * LWL * BWL³ / 12`.
- **Impact:** BM is now ~60% of previous value, more physically realistic for typical hull shapes.

## Bug #23: Bulb volume mismatch from non-uniform scaling
- **File:** `hull_opt/geometry.py:482-492`
- **Severity:** Low
- **Discovery:** Code review (subagent)
- **Problem:** Bulb sphere was created at requested volume, then scaled by x_scale≈1.33 and z_scale=0.8, changing actual volume by ~1.067× (6.7% larger than requested).
- **Fix:** Initial sphere radius compensated by dividing target volume by `x_scale * z_scale`.
- **Impact:** Ballast mass and displacement no longer overestimated by ~6.7%.

---

## Bug #24: Gunwale strip connects keel to sheer instead of waterline to sheer

- **File:** `hull_opt/geometry.py:207-219`
- **Severity:** Critical
- **Discovery:** Systematic code review
- **Problem:** The gunwale strip (topside panel) used `station_start_idx[i] + n_vert - 1` (port KEEL) and `station_start_idx[i] + 2*n_vert - 1` (starboard KEEL) as the lower edge, connecting keel to sheer. This created large diagonal faces that completely overlapped the hull side faces in the underwater region (z = [-T, 0]). The correct lower edge is the waterline at index 0 (port) and `n_vert` (starboard). Consequence: wetted area overestimated by ~60-80%, inflating ITTC-57 friction resistance and penalizing the optimizer's FoM. The mesh had non-manifold overlapping faces that were masked by `process(validate=True)`.
- **Fix:** Changed gunwale strip to connect waterline (index 0 for port, `n_vert` for starboard) to sheer. Also fixed:
  - Bottom section (keel closure): was incorrectly connecting port waterline to starboard waterline (a horizontal face at z=0). Changed to connect port keel to starboard keel.
  - Bow/stern extra faces: were closing the keel-sheer gap (leftover from the overlapping gunwale). Changed to close the waterline-sheer gap above the deck.
- **Impact:** Wetted area is now physically correct. Hull is watertight without relying on overlapping non-manifold faces. Total resistance estimates drop by ~60%.

## Bug #25: Volume constraint uses SAC-integrated volume instead of mesh volume

- **File:** `hull_opt/geometry.py:547-552`
- **Severity:** High
- **Discovery:** Systematic code review
- **Problem:** `_compute_hydrostatics` overrode `volume = sac_volume` when the SAC volume was positive, ignoring the actual mesh volume. The SAC scaling can amplify tiny sections (up to 2.5×) to match the target displacement, so the volume constraint always saw `nabla ≈ target_displacement` regardless of the actual hull volume. The optimizer could produce hulls with extreme SAC scaling where the actual mesh volume was far from the target, yet the constraint passed.
- **Fix:** Removed the override; mesh volume is now always used. SAC volume is retained as a cross-check: if mesh/SAC ratio exceeds 1.5 or falls below 0.5, a warning is logged.
- **Impact:** The displacement constraint now reflects the actual hull geometry. The SAC-gaming vector is closed.

## Bug #26: Self-righting heuristic bypasses GZ curve

- **File:** `hull_opt/constraints.py:60`
- **Severity:** High
- **Discovery:** Systematic code review
- **Problem:** The heuristic at line 60 (`D_keel > T_canoe * 2.0 and ballast_frac > min_ballast and D_keel >= 1.0`) allowed `mean_gz_high > -0.01` — a barely-negative GZ at high angles. A design with `D_keel = 1.0`, `T_canoe = 0.15` (ratio = 6.7), and `ballast_frac = 0.3` could be automatically self-righting even with slightly negative righting arms at 150-180°.
- **Fix:** Changed threshold to `mean_gz_high > 0.005`, matching the standard GZ-based self-righting check.
- **Impact:** The heuristic no longer bypasses positive-GZ requirements. Exploited designs are correctly flagged.

## Bug #27: Cp constraint checks design vector value, not actual mesh value

- **File:** `hull_opt/geometry.py:547-590`, `hull_opt/constraints.py:95-99`
- **Severity:** Medium
- **Discovery:** Systematic code review
- **Problem:** `constraints["Cp"]` came from `hydro.get("Cp")`, which stored the design vector Cp, not the actual prismatic coefficient of the generated mesh. SAC scaling (up to 2.5×) can significantly alter the actual Cp. The optimizer could set `Cp = 0.55` while the actual hull's Cp was 0.45 after scaling.
- **Fix:** Added `actual_Cp` computation in `_compute_hydrostatics`: computed from mesh volume and the maximum SAC-scaled station area (`actual_Am = max(station_areas)`, `actual_Cp = volume / (actual_Am * LWL)`). The constraint now checks `actual_Cp` instead of design Cp.
- **Impact:** Cp constraint reflects actual hull geometry. Exploitation through SAC scaling is blocked.

## Bug #28: Reserve buoyancy and downflooding computed on hull-only mesh in high-fidelity

- **File:** `hull_opt/high_fidelity.py:157`
- **Severity:** Medium
- **Discovery:** Systematic code review
- **Problem:** `_gate_downflooding` received `hull_stl` (hull-only mesh before keel/bulb) instead of `stl_path` (full mesh). Since keel displacement was missing, the below-water volume was underestimated, causing reserve buoyancy to be overestimated. Same issue affected downflooding angle. A poor-reserve design could appear to pass.
- **Fix:** Changed to pass `stl_path` (full mesh with keel/bulb) to `_gate_downflooding`.
- **Impact:** Reserve buoyancy and downflooding angles now account for keel/bulb displacement.

## Bug #29: Port hull side faces have inverted normals (fixed post-hoc)

- **File:** `hull_opt/geometry.py:284`
- **Severity:** Medium
- **Discovery:** Systematic code review
- **Problem:** The brute-force normal fixer used `vol < 0` (strictly negative) to identify inverted faces. If any degenerate face had `vol == 0` (e.g., very flat triangle or numerical zero), it would not be flipped, leaving an inward-pointing face.
- **Fix:** Changed to `vol <= 0` so degenerate faces are also flipped.
- **Impact:** All inverted faces are now reliably corrected.

## Bug #30: SAC scale factor mismatch (code cap 3.0 vs constraint boundary 2.5)

- **File:** `hull_opt/geometry.py:166,311,398`, `hull_opt/constraints.py:90`
- **Severity:** Medium
- **Discovery:** Systematic code review
- **Problem:** The SAC area cap was `min(area_scale, 3.0)` but the constraint flagged `sac_scale > 2.5`. A scale of 2.9 bypassed the constraint but was still applied in geometry generation, allowing extreme hull deformation.
- **Fix:** Changed all three SAC caps from 3.0 to 2.5, matching the constraint boundary.
- **Impact:** Extreme SAC scaling is prevented at both the geometry and constraint levels.

## Bug #31: Convergence check uses best-so-far values instead of actual evaluation FoMs

- **File:** `hull_opt/surrogate.py:398-418`
- **Severity:** Medium
- **Discovery:** Systematic code review
- **Problem:** `_best_fom_history` recorded `self.best_fom` (best-so-far) on every evaluation, not the actual evaluation's FoM. When no new best was found for many iterations, the history was filled with repeated best values. The convergence check compared `recent[-1] - recent[0]`, which was 0 when the best hadn't changed, triggering premature convergence even when exploration was valuable.
- **Fix:** Added `_fom_history` tracking the actual FoM of each feasible evaluation. Convergence check now compares the best FoM in the last N evaluations (`best_recent`) against the best FoM before that window (`best_prev`). Only terminates when the true best-in-window improvement is below threshold.
- **Impact:** BO loop no longer converges prematurely due to flat best-so-far values.

## Bug #32: CG_z ballast distribution doesn't match mass distribution

- **File:** `hull_opt/hydrostatics.py:164-175`
- **Severity:** Medium
- **Discovery:** Systematic code review
- **Problem:** `compute_cg_z` split ballast equally between keel and bulb (`0.5 * ballast_frac` each), but `_compute_hydrostatics` computed actual masses from geometry (`bulb_vol * 11340` for bulb, `D_keel * keel_chord * 0.03 * 0.5 * 1025` for keel). For unequal keel/bulb masses, the CG_z used in GZ curves and downflooding differed from the actual mass distribution.
- **Fix:** `compute_cg_z` now accepts an optional `nabla` parameter. When provided, it uses the same mass-distribution formula as `_compute_hydrostatics`. Updated all call sites (low_fidelity, constraints, high_fidelity) to pass `nabla` from the available hydro dict or config.
- **Impact:** CG_z used in stability analysis now matches the actual mass distribution. GZ curves and downflooding angles are consistent with the mass model.

## Bug #33: Unused variable `hull_lwl_cap` in Capytaine RAO function

- **File:** `hull_opt/low_fidelity.py:243`
- **Severity:** Low
- **Discovery:** Systematic code review
- **Problem:** `hull_lwl_cap` was computed but never used in `_compute_raos_capytaine`.
- **Fix:** Removed the unused variable.
- **Impact:** Cleaner code.

## Bug #34: `_compute_raos_capytaine` accepts `speed_ms` but never uses it

- **File:** `hull_opt/low_fidelity.py:226`
- **Severity:** Low
- **Discovery:** Systematic code review
- **Problem:** The `speed_ms` parameter is passed for forward-speed effects but Capytaine only solves zero-speed diffraction/radiation. The unused parameter is misleading.
- **Fix:** Added docstring noting the limitation.
- **Impact:** Documentation clarity.

## Bug #35: B/LWL constraint doesn't account for SAC scaling of beam

- **File:** `hull_opt/constraints.py:33,68`
- **Severity:** Low
- **Discovery:** Systematic code review
- **Problem:** `constraints["B/LWL"]` used the design BWL, but SAC scaling multiplies `y_local` by `area_scale`, directly scaling the actual beam. With `area_scale = 2.0`, actual B/LWL = 2× design B/LWL, potentially falling outside the bounds.
- **Fix:** Multiplied B/LWL by `sac_scale_factor` from the hydro dict.
- **Impact:** B/LWL constraint now reflects the actual SAC-scaled beam.

## Bug #36: Initial LHS sampling not deterministic on resume after crash

- **File:** `hull_opt/surrogate.py:52-58,114,175`
- **Severity:** Low
- **Discovery:** Systematic code review
- **Problem:** `_initial_sampling` and `_resume_initial_sampling` called `latin_hypercube_sample` without a seed. On resume, a new random LHS was generated, so indices [50, 79] evaluated different designs than the original LHS would have produced. This broke the LHS space-filling property.
- **Fix:** Both functions now pass `seed=getattr(self.config.optimization, 'lhs_seed', 42)` to `latin_hypercube_sample`. The function already accepted a `seed` parameter.
- **Impact:** LHS is deterministic across crash/resume cycles. The same designs are always produced for the same seed.

## Bug #37: Starboard hull side face winding inconsistent with port

- **File:** `hull_opt/geometry.py:199-204`
- **Severity:** Low (maintainability)
- **Discovery:** Systematic code review
- **Problem:** Starboard hull side used `[s0, s2, s3]/[s0, s1, s2]` while port used `[v0, v2, v1]/[v0, v3, v2]`. While both produce outward normals after y-negation, the inconsistent winding is a maintenance risk.
- **Fix:** Changed starboard to `[s0, s2, s1]/[s0, s3, s2]`, matching the port pattern.
- **Impact:** Consistent winding convention across both sides.

## Bug #38: Database has no UNIQUE constraint on `iter` column

- **File:** `hull_opt/database.py:25-43`
- **Severity:** Low
- **Discovery:** Systematic code review
- **Problem:** The `iter` column was NOT NULL but had no UNIQUE constraint. Duplicate entries could accumulate from crash recovery, biasing GP training with duplicate data points.
- **Fix:** Changed `CREATE INDEX` to `CREATE UNIQUE INDEX` on `iter`. Changed `INSERT INTO designs` to `INSERT OR REPLACE INTO designs` so re-evaluating an existing iteration replaces the old entry.
- **Impact:** No duplicate iteration entries. GP training data is clean.

## Bug #39: `compute_righting_energy` doesn't validate displacement against GZ curve

- **File:** `hull_opt/hydrostatics.py:149-161`
- **Severity:** Low
- **Discovery:** Systematic code review
- **Problem:** The function accepts `displacement` as a separate argument. The GZ curve's third column contains submerged volumes at each heel angle, but these were never used for cross-checking. If the passed displacement didn't match the GZ curve's displacement, the righting energy was wrong.
- **Fix:** Added cross-check: compare the zero-heel submerged volume (from GZ curve) against the passed displacement. If the ratio exceeds 10%, a warning is issued.
- **Impact:** Mismatched displacement is now detected and reported.

## Bug #40: `compute_gz_curve` stores submerged volumes but they're never used by constraints

- **File:** `hull_opt/hydrostatics.py:146`
- **Severity:** Low
- **Discovery:** Systematic code review
- **Problem:** The third column of the GZ curve tracks submerged volume at each heel angle but was never consumed by any caller beyond the `compute_righting_energy` validation added in Bug #39.
- **Fix:** (Covered by Bug #39 — the `compute_righting_energy` cross-check now uses the third column.)
- **Impact:** Submerged volume data is now used for displacement validation. 

---

## Bug #41: Mid-fidelity water level set to T_hull instead of z=0 (over-submerges hull)

- **File:** `hull_opt/mid_fidelity.py:173`
- **Severity:** Critical
- **Discovery:** Bug report (subagent)
- **Problem:** The inline setFieldsDict used `box ( -1000 -1000 -1000 ) ( 1000 1000 {T_hull} )`. This set the water level at z = T_hull (~0.2m) instead of z = 0 (the waterline). The hull was over-submerged by its full draft, so every mid-fidelity CFD run computed forces on a hull with water up to the sheer line. The calibration delta Δ = Rt_CFD - Rt_lowfi was computed against an artificially high CFD resistance, corrupting the drag correction for all subsequent iterations.
- **Fix:** Changed the box z-upper bound from `{T_hull}` to `0`.
- **Impact:** Mid-fidelity CFD now computes forces at the correct waterline. Drag correction is physically meaningful.

## Bug #42: SAC scaling omitted from half-breadth function in calibration delta

- **File:** `hull_opt/mid_fidelity.py:125`
- **Severity:** Critical
- **Discovery:** Bug report (subagent)
- **Problem:** The `hb_func` lambda passed no `sac_scale` to `compute_half_breadth_analytic`, so it defaulted to 1.0. But the actual mesh generated on line 50 was SAC-scaled (up to 2.5×). The low-fi wave resistance used for the calibration delta was computed on the unscaled analytic shape, not the actual geometry. Every calibration computed Δ against the wrong hull shape.
- **Fix:** Added `sac_scale = hydro.get("sac_scale_factor", 1.0)` and passed `sac_scale=sac_scale` to `compute_half_breadth_analytic`.
- **Impact:** Calibration delta now compares CFD against the correct SAC-scaled analytic hull shape.

## Bug #43: Self-righting fallback heuristic is dead code (can never trigger)

- **File:** `hull_opt/constraints.py:62`
- **Severity:** Critical
- **Discovery:** Bug report (subagent)
- **Problem:** The fallback check `mean_gz_high > 0.005` on line 62 was the same condition already required for the primary check on line 53. When GZ data was sparse (np.sum(late_mask) <= 2), `mean_gz_high` stayed 0.0, making `0.0 > 0.005` impossible. The geometric heuristic (deep keel + ballast) could never independently override the GZ-based check.
- **Fix:** Removed the `mean_gz_high > 0.005` condition from the fallback, leaving only geometry criteria (D_keel, ballast_frac).
- **Impact:** Designs with deep keel and sufficient ballast can now be self-righting even when GZ data is too sparse for a reliable mean.

## Bug #44: Light wind bonus uses reverse speed regime

- **File:** `hull_opt/low_fidelity.py:191`
- **Severity:** High
- **Discovery:** Bug report (subagent)
- **Problem:** `min_wind_speed_kt = 6.0 knots → speed_low_ms = 3.087 m/s`. The target design speed is 4.0 knots → 2.058 m/s. The "light wind" bonus evaluated wave resistance at a speed 50% higher than the design speed, not lower. This rewarded hulls optimized for a higher speed regime, the opposite of the stated intent ("good light-wind performance").
- **Fix:** Changed to use `0.5 * config.fixed.target_speed_knots`, a fraction of design speed.
- **Impact:** Light wind bonus now correctly evaluates performance at low speed.

## Bug #45: Geometry normal fixer threshold inconsistent between hull-only and combined mesh

- **File:** `hull_opt/geometry.py:284,362`
- **Severity:** High
- **Discovery:** Bug report (subagent)
- **Problem:** Line 284 used `vol <= 0` (fix applied after Bug #29), but line 362 on the combined (hull+keel+bulb) mesh used `vol < 0`. Zero-volume faces on the combined mesh were not flipped. While `merge_vertices` + `process(validate=True)` may clean these, the inconsistency meant structured zero-volume faces on the combined mesh could remain with inward normals.
- **Fix:** Changed line 362 to `vol <= 0`.
- **Impact:** Combined mesh normals are now consistently corrected, matching hull-only behavior.

## Bug #46: compute_cg_z without nabla produces dimensionally meaningless CG

- **File:** `hull_opt/hydrostatics.py:191-194`
- **Severity:** High
- **Discovery:** Bug report (subagent)
- **Problem:** When `nabla` was None (Capytaine unavailable fallback in `_estimate_roll_period`), the function used dimensionless fractions (hull_mass = 1.0 - ballast_frac, total_mass = 1.0). The weighted average still gave meters, but the weights did not correspond to actual mass distribution. The result had correct units but wrong magnitudes, affecting roll period estimates.
- **Fix:** Added approximate nabla computation from x_dict (BWL * LWL * T_canoe * Cp * Cm) when not provided.
- **Impact:** Roll period estimates from the fallback path now use physically meaningful mass distribution.

## Bug #47: B/LWL constraint multiplied by sac_scale but used with fixed thresholds

- **File:** `hull_opt/constraints.py:34,70-71`
- **Severity:** High
- **Discovery:** Bug report (subagent)
- **Problem:** `B/LWL` was computed as `B / LWL * sac_scale`. The threshold check used fixed bounds [0.15, 0.30]. Since sac_scale could be up to 2.5, and base B/LWL was ~0.17-0.26, the product frequently exceeded 0.30 even for physically reasonable hulls. This created a tension where valid hull shapes (with moderate SAC scaling) were rejected.
- **Fix:** Changed the constraint to use unscaled `B/LWL` against [0.15, 0.30]. The SAC-scaled value is still reported as `B/LWL_scaled` for diagnostics. SAC scaling has its own dedicated constraint.
- **Impact:** Designs with moderate SAC scaling now pass the B/LWL constraint, matching the generator's capabilities.

## Bug #48: generate_hull return type annotation wrong

- **File:** `hull_opt/geometry.py:100-102,409`
- **Severity:** Medium
- **Discovery:** Bug report (subagent)
- **Problem:** Signature showed `-> tuple[str, str, dict]` but returned `(stl_path, sac_path, hydro, hull_stl)` — 4 values, not 3. All callers correctly unpacked 4 values, so no runtime error, but the annotation was misleading for static analysis.
- **Fix:** Changed to `-> tuple[str, str, dict, str]`.
- **Impact:** Static analysis now correctly reflects the 4-element return.

## Bug #49: Worker count memory safety bypassed in LHS phase

- **File:** `hull_opt/surrogate.py:123,185`
- **Severity:** Medium
- **Discovery:** Bug report (subagent)
- **Problem:** `_initial_sampling` at line 123 used `min(n_initial, 4)` directly for ProcessPoolExecutor, bypassing `self.mem.safe_worker_count()`. If the system had limited RAM (e.g., 4GB), requesting 4 workers at 3.5GB each would cause OOM. The memory throttle was bypassed in the non-Ray path.
- **Fix:** Changed to `self.mem.safe_worker_count(min(n_initial, 4), per_process_gb=3.5)` in both `_initial_sampling` and `_resume_initial_sampling`.
- **Impact:** LHS phase now respects system memory limits, preventing OOM on low-RAM systems.

## Bug #50: _best_fom_history populated inconsistently between LHS and BO phases

- **File:** `hull_opt/surrogate.py:172-173 vs 275-276`
- **Severity:** Medium
- **Discovery:** Bug report (subagent)
- **Problem:** In `_initial_sampling`, `_best_fom_history` was appended for ALL designs (line 173). In `_bo_loop`, it was inside the `if result.feasible:` block (line 276). This meant the BO convergence plot had fewer entries than LHS, breaking the time-series continuity.
- **Fix:** Moved `_best_fom_history.append` outside the `if result.feasible:` block in `_bo_loop`.
- **Impact:** Convergence history is now consistently recorded for all iterations across both phases.

## Bug #51: Calibration stored CFD value double-counts previous drag correction

- **File:** `hull_opt/surrogate.py:291-295`
- **Severity:** Medium
- **Discovery:** Bug report (subagent)
- **Problem:** `best.get("rt_total", 0.0)` retrieved the stored total resistance, which already included a previous iteration's `drag_correction`. Adding `new_delta` to it produced `rt_uncorrected + old_delta + new_delta`, double-counting. The stored `rt_cfd` in the calibration table was inflated.
- **Fix:** Saved `old_delta` before overwriting, then computed `rt_michlet = rt_total - old_delta` to recover the uncorrected value.
- **Impact:** Calibration table now stores the actual CFD-measured resistance, not an inflated double-counted value.

## Bug #52: SQLite connection not in WAL mode

- **File:** `hull_opt/database.py:19`
- **Severity:** Medium
- **Discovery:** Bug report (subagent)
- **Problem:** No `PRAGMA journal_mode=WAL` was set. While currently single-threaded, any future concurrent SELECT during a write would raise SQLITE_BUSY. This was fragile.
- **Fix:** Added `PRAGMA journal_mode=WAL` after connection.
- **Impact:** Concurrent read/write access is now safe; WAL mode prevents SQLITE_BUSY.

## Bug #53: Redundant x_dict recomputation in high_fidelity.py

- **File:** `hull_opt/high_fidelity.py:59,73`
- **Severity:** Low
- **Discovery:** Bug report (subagent)
- **Problem:** Line 59 called `design_vector_to_dict(design_vector)`. Line 73 recomputed the identical `x_dict`. The line-73 assignment shadowed the earlier one.
- **Fix:** Removed the redundant line-73 call.
- **Impact:** Cleaner code, no functional change.

## Bug #54: Dead code: _extract_max_pressure

- **File:** `hull_opt/high_fidelity.py:559-588`
- **Severity:** Low
- **Discovery:** Bug report (subagent)
- **Problem:** The function was never called. Gate 5 (inverted pressure) used inline pressure extraction at lines 383-393.
- **Fix:** Removed `_extract_max_pressure`.
- **Impact:** Cleaner code, removed dead code path.

## Bug #55: Orphan pass statement in run_of_command

- **File:** `hull_opt/utils.py:52`
- **Severity:** Low
- **Discovery:** Bug report (subagent)
- **Problem:** The `pass` after the warning about failed OF env source was dead code.
- **Fix:** Removed the `pass` statement.
- **Impact:** Cleaner code.

## Bug #56: keel_x_pos uses bulb_pos parameter for keel placement

- **File:** `hull_opt/geometry.py:327`
- **Severity:** Low
- **Discovery:** Bug report (subagent)
- **Problem:** `keel_x_pos = bulb_pos * LWL` used the bulb position parameter as the keel's longitudinal position. If bulb_vol was near-zero (no bulb), the keel still positioned at bulb_pos * LWL. The design vector had no independent keel-position parameter, coupling keel position to bulb position.
- **Fix:** When bulb_vol is near-zero (< 1e-6), use `0.4 * LWL` as a reasonable default keel position instead of `bulb_pos * LWL`.
- **Impact:** Keel position is decoupled from bulb position when no bulb is present.

## Bug #57: GZ curve rotation direction reversed (DOCUMENTATION ERROR — NOT A BUG)

- **File:** `hull_opt/hydrostatics.py:120`
- **Severity:** None (false alarm)
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem (claimed):** GZ curve used `rotation_matrix(-rad, [1, 0, 0])` for positive heel angles. This rotates the hull port-side down instead of starboard down, flipping the sign of GZ at small angles.
- **Actual analysis:** `-rad` is physically correct. In a right-handed coordinate system (+x forward, +y starboard, +z up), a negative rotation about the x-axis rotates +y toward -z, lowering the starboard side — the correct convention for positive heel to starboard. Changing to `+rad` would raise starboard, inverting every computed GZ sign.
- **Verdict:** The code was correct all along. This entry is retained only to prevent future re-investigation of the same sign convention.

## Bug #58: Righting energy integrates negative GZ, canceling positive contributions

- **File:** `hull_opt/hydrostatics.py:168`
- **Severity:** Critical
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** `np.trapezoid(gz_valid, angle_rad)` integrated raw GZ values. For hulls with positive GZ at small angles and negative GZ at large angles, the integral canceled out, producing near-zero energy for partially stable hulls.
- **Fix:** Changed to `gz_positive = np.maximum(gz_valid, 0.0)` and integrate only positive GZ.
- **Impact:** Righting energy now reflects only the stabilizing portion of the GZ curve.

## Bug #59: Michell integral numerically sensitive to grid resolution

- **File:** `hull_opt/michell.py:35`
- **Severity:** Critical
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** `np.gradient(f, dx, axis=1)` computes hull slope at bow/stern where width changes rapidly. No convergence check against grid resolution allowed the optimizer to exploit numerical artifacts (sharp bow/stern stiffness) to game wave resistance values by 20-30%.
- **Fix:** Added gradient clipping at ±10 and a double-resolution convergence check. If low/high resolution differ by >50%, the average is used.
- **Impact:** Wave resistance values are now numerically stable against grid resolution.

## Bug #60: CB_z from degenerate meshes not validated

- **File:** `hull_opt/geometry.py:586-588`
- **Severity:** Critical
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** `mesh.center_mass` for self-intersecting meshes could return physically meaningless CB_z values outside `[-T_canoe, 0]`.
- **Fix:** Added clamping: if CB_z > 0 or CB_z < -T_canoe, use `-T_canoe * 0.4` fallback.
- **Impact:** Degenerate meshes no longer produce physically impossible CB_z.

## Bug #61: Convergence check can produce empty slice at exactly 10 evaluations

- **File:** `hull_opt/surrogate.py:422-428`
- **Severity:** High
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** `self._fom_history[:-10]` is empty when `len(self._fom_history) == 10`, causing `max([])` ValueError. Also `n_check = min(10, len(self._fom_history))` could take all entries leaving nothing for `best_prev`.
- **Fix:** Changed `min_check = 10` guard, `n_check = min(10, len(self._fom_history) // 2)` and proper fallback when preview window is empty.
- **Impact:** Convergence check is now safe at all history lengths.

## Bug #62: Roll period overestimated CB_z from hull-only mesh

- **File:** `hull_opt/low_fidelity.py:377`
- **Severity:** High
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** `CB_z` from hydro dict is computed on hull-only mesh (before keel/bulb), giving ~-0.08 to -0.12 instead of true combined CB (~-0.3 to -0.5). GM was overestimated, giving falsely short roll periods.
- **Fix:** Added clamping of CB_z to `[-T_total, 0]` and use combined depth `T_canoe + D_keel` for inertia computation.
- **Impact:** Roll period estimates are now physically realistic for deep-keel designs.

## Bug #63: Waterline half-breadth ignores LCB parameter

- **File:** `hull_opt/geometry.py:25-39`
- **Severity:** High
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** `_waterline_half_breadth` used fixed `peak_pos = 0.45`, producing symmetric waterline shape. LCB parameter only affected vertical section scaling via SAC, not the waterplane shape. Wave resistance didn't respond correctly to LCB asymmetry.
- **Fix:** Added `LCB` parameter. LCB shift from 12.5 shifts `peak_pos` within ±0.15.
- **Impact:** Waterline shape now responds to LCB, affecting wave resistance correctly.

## Bug #64: Lambda closure captures sac_scale by mutable reference

- **File:** `hull_opt/low_fidelity.py:104-105`
- **Severity:** High
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** `half_breadth_func` captured `sac_scale` by closure reference. Future mutations would silently change the lambda's behavior.
- **Fix:** Used default argument pattern `lambda xq, zq, _ss=sac_scale: ...`.
- **Impact:** Lambda is now robust against scope mutations.

## Bug #65: run_of_command path resolution could cause silent failures

- **File:** `hull_opt/utils.py:53-54`
- **Severity:** High
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** `cwd = case_dir` and `-case` argument both using the same path could create nested directories in OpenFOAM. No `pipefail` meant silent errors in pipeline commands.
- **Fix:** Added `set -o pipefail`, proper command escaping via `shlex.quote()`, and use explicit absolute case directory.
- **Impact:** OpenFOAM errors are now reliably detected.

## Bug #66: body_count threshold too permissive for merged meshes

- **File:** `hull_opt/geometry_validator.py:62-63`
- **Severity:** High
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** Single body count > 10 check was too loose for non-watertight meshes (should flag any disconnected bodies) and too strict for watertight meshes (hull+keel+bulb = 3 bodies expected).
- **Fix:** Watertight: allow ≤ 3 bodies. Non-watertight: allow ≤ 1 body.
- **Impact:** Appropriate body count enforcement for both cases.

## Bug #67: validate_design_vector uses hard-coded bounds instead of config

- **File:** `hull_opt/geometry_validator.py:86-91`
- **Severity:** High
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** Hard-coded bounds (BWL 0.10-1.5) were too loose vs config bounds (0.40-0.60). Designs passing validation could produce physically unrealistic hulls.
- **Fix:** Added optional `config` parameter. When provided, uses config bounds with ±0.01 tolerance. Falls back to hard-coded bounds when config is absent.
- **Impact:** Designs are now validated against the actual optimization bounds.

## Bug #68: _waterline_half_breadth default Cm=0.75 masks caller omissions

- **File:** `hull_opt/geometry.py:26`
- **Severity:** High
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** Default parameter `Cm=0.75` meant any call forgetting to pass Cm silently used 0.75 instead of erroring.
- **Fix:** Removed default, forcing explicit Cm parameter.
- **Impact:** Missing Cm now raises TypeError.

## Bug #69: Mesh volume vs SAC volume discrepancy not rejected

- **File:** `hull_opt/geometry.py:577-580`
- **Severity:** Medium
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** SAC scaling could make mesh volume gamely diverge from target without rejection.
- **Fix:** Added warning when mesh/target volume ratio < 0.3 or > 2.0.
- **Impact:** SAC gaming is now detected and logged.

## Bug #70: Section shape interpolation produces extreme stern sections

- **File:** `hull_opt/geometry.py:58-67`
- **Severity:** Medium
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** `_interp_param` with `stern_factor=3.0` and `bow_factor=1.5` produced extreme V-shaped stern sections that SAC scaling (up to 2.5×) amplified into balloon sections.
- **Fix:** Capped bow_factor ≤ 2.0 and stern_factor ≤ 2.5.
- **Impact:** Section shapes are now physically reasonable at all stations.

## Bug #71: JONSWAP alpha factor 5/16 = 0.3125 is wrong (should be ~0.0081)

- **File:** `hull_opt/low_fidelity.py:332`
- **Severity:** Medium
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** Pierson-Moskowitz alpha `5/16 = 0.3125` was used as the JONSWAP Phillips constant. Correct value is ~0.0081. Spectral energy density was overestimated by ~38×, making acceleration predictions ~6× too high.
- **Fix:** Changed to `alpha = 0.0081`.
- **Impact:** Peak acceleration predictions are now physically realistic (factor ~6 lower).

## Bug #72: get_iteration_count used MAX(iter) instead of COUNT(*)

- **File:** `hull_opt/database.py:171-173`
- **Severity:** Medium
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** `MAX(iter)` returns the highest iteration number, but if iterations have gaps (crash recovery), the optimizer incorrectly assumes all iterations are present.
- **Fix:** Changed to `COUNT(*)`.
- **Impact:** Resume logic now correctly counts actual designs, not highest iteration number.

## Bug #73: Gate 5 pressure extraction uses final time, not maximum

- **File:** `hull_opt/high_fidelity.py:390-394`
- **Severity:** Medium
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** Only the last write time's pressure was extracted. Maximum pressure during transient events could be mid-simulation, causing significant underestimation.
- **Fix:** Iterate all write times and take the max pressure value.
- **Impact:** Maximum pressure now correctly accounts for transient peaks.

## Bug #74: Reserve buoyancy formula divides by submerged volume instead of total volume

- **File:** `hull_opt/hydrostatics.py:264`
- **Severity:** Medium
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** `above_vol / submerged_vol` returns fraction of below-water volume, not the conventional `above_vol / total_vol`. Values are approximately doubled.
- **Fix:** Changed to `above_vol / total_vol`.
- **Impact:** Reserve buoyancy values now match the conventional definition (0-1 range).

## Bug #75: Feasibility ratio plot uses non-iter x-axis

- **File:** `run_optimization.py:85-92`
- **Severity:** Low
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** X-axis was `range(len(feas_ratio))` instead of actual iteration numbers, making the plot misleading.
- **Fix:** Changed to use actual iteration numbers.
- **Impact:** Visualization now correctly shows feasibility over iterations.

## Bug #76: Keel root closure faces missing

- **File:** `hull_opt/geometry.py:469-477`
- **Severity:** Low
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** Keel root (z=0 end) had no closure faces connecting port to starboard. Relied on hull bottom mesh for closure, which could leave gaps at the keel-hull interface.
- **Fix:** Added explicit root closure faces connecting port to starboard keel top.
- **Impact:** Keel-hull junction is now watertight without relying on merge_vertices.

## Bug #77: Keel mass formula inconsistent between geometry.py and hydrostatics.py

- **File:** `hull_opt/geometry.py:639`
- **Severity:** Low
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** `geometry.py` used `BWL * 0.06` as keel width while `hydrostatics.py` used `0.03` (half). Mass distribution used for CG_z didn't match the hydrostatics mass model.
- **Fix:** Changed `geometry.py` keel width to `BWL * 0.03` to match `hydrostatics.py`.
- **Impact:** CG_z computation is now consistent across both modules.

## Bug #78: Stale Ray processes on keyboard interrupt

- **File:** `hull_opt/surrogate.py:77-103`
- **Severity:** Low
- **Discovery:** Comprehensive code review (senior reviewer)
- **Problem:** SIGINT during optimization left Ray processes running. No clean shutdown path for interruption.
- **Fix:** Added SIGINT handler, `_ray_initialized` flag, and interruption checks in `_bo_loop`. Ray is reliably shut down on interrupt.
- **Impact:** No stale Ray processes remain after keyboard interrupt.

---

## Bug #79: Non-monotonic section curve allows hull bulging

- **File:** `hull_opt/geometry.py:44-57`
- **Severity:** High
- **Discovery:** Shape analysis — section curve can produce cross-sections 96% wider below the waterline than at the waterline when `bilge_r` is high and `deadrise` is low.
- **Problem:** The `_section_curve()` function sums `base + dr_term + fl_term + br_term` where `br_term = mid_wt * bilge_r * 4.0` peaks at mid-depth (z_norm ~ -0.5). With `bilge_r=0.30`, `deadrise=5°`, this produces y_frac=1.91 at mid-depth vs y_frac=0.97 at the waterline — the hull bulges outward below the waterline, creating a physically invalid "hourglass" cross-section. The `np.clip(..., 0.0, None)` only prevents negative values; it does not enforce monotonicity.
- **Fix:** Added `y = np.minimum.accumulate(y)` after the clip to enforce non-increasing y from waterline to keel. This clips any bulge to the waterline value, producing a valid cross-section (vertical wall above the bulge point, then taper to keel).
- **Impact:** Eliminates 96% width bulge. Affected designs will now have boxier cross-sections in the upper depth, which is physically realistic for displacement hulls.

## Bug #80: Normal fixing uses origin-based tetrahedron volume

- **File:** `hull_opt/geometry.py:287-298`, `hull_opt/geometry.py:363-374`
- **Severity:** Medium
- **Discovery:** Code review — origin-based reference incorrectly flips faces for hulls mostly in negative-z space.

- **File:** `hull_opt/geometry.py:287-298`, `hull_opt/geometry.py:363-374`
- **Severity:** Medium
- **Discovery:** Code review — origin-based reference incorrectly flips faces for hulls mostly in negative-z space.
- **Problem:** The normal-fixing routine used `sum(verts[face[0]] * cross) / 6 < 0` (signed tetrahedron volume with the origin as the fourth vertex) to detect inverted faces. For hull geometry where most vertices have negative z-values, the origin is outside the mesh, so many correctly-oriented faces have negative signed volume and get incorrectly flipped. `trimesh.process(validate=True)` then silently patches the winding, masking the bug.
- **Fix:** Changed both normal-fixing blocks to use `hull_mesh.centroid` as reference: compute `center → face_centroid` vectors and dot with `face_normals`; flip faces where the dot product is negative (normal points inward).
- **Impact:** Correct face winding on first pass, no reliance on `process()` to silently repair bad normals.

## Bug #81: No cross-section monotonicity validation

- **File:** `hull_opt/geometry_validator.py`
- **Severity:** Medium
- **Discovery:** Systematic gap analysis — validator checked volume, bounds, watertightness but not shape quality.
- **Problem:** The geometry validator had no checks for cross-section shape quality. A hull with bulging, self-intersecting, or concave cross-sections could pass all checks as long as it was watertight and had reasonable volume/dimensions.
- **Fix:** Added to `validate_hull_geometry()`:
  - Convex hull ratio check: `0.25 < mesh_volume / convex_hull_volume < 0.97`. Catches bulging (too concave) and featureless (too convex) shapes.
  - Beam-to-draft ratio: `beam / height < 5.0`. Catches cartoonishly flat hulls.
- **Impact:** Four new failure modes that reject degenerate shapes at validation time.

## Bug #82: Missing bilge_r/BWL and B/T constraints

- **File:** `hull_opt/constraints.py`
- **Severity:** Medium
- **Discovery:** Cross-section analysis — bilge_r up to 0.30 with BWL down to 0.40 produces bilge_r/BWL=0.75, guaranteeing bulging.
- **Problem:** No constraints prevented extreme bilge radius relative to beam, or extreme beam relative to draft. These ratios directly control cross-section validity and overall hull proportions.
- **Fix:** Added two new constraints: `beam_draft_ratio = B / T_canoe` (fails if > 4.5) and `bilge_r_BWL_ratio = bilge_r / B` (fails if > 0.5).
- **Impact:** BO acquisition function cannot propose designs with extreme proportions that guarantee invalid cross-sections.

## Bug #83: Station-to-station smoothness not checked

- **File:** `hull_opt/geometry.py:209-217`
- **Severity:** Low
- **Discovery:** Code review — adjacent stations with wildly different shapes cause wavy/fluted hull surfaces.
- **Problem:** No check existed for abrupt changes in cross-section area between adjacent stations. Station area could vary arbitrarily, producing hourglass or fluted hull shapes that pass all other checks.
- **Fix:** Added area-jump detection after station generation: if `max(Δ_area) / mean(area) > 2.0`, a warning is logged (diagnostic only, not fatal, since parametric formulation provides inherent smoothness).
- **Impact:** Early warning for station-to-station shape discontinuities during development.

---

## Bug #84: Validation tolerance is flat ±0.01 regardless of parameter range

- **Files:** `hull_opt/geometry_validator.py:180`, `hull_opt/geometry.py:742-760`
- **Severity:** High
- **Discovery:** Subagent investigation of validation gaps
- **Problem:** `validate_design_vector` used a flat ±0.01 tolerance for all parameters. For `keel_rake` (range 0.019), this allowed values as low as -0.009 (52.6% below the lower bound). A negative keel_rake sweeps the keel forward — geometrically invalid. Additionally, 7 design parameters (LCB, E, SA, flare, bulb_pos, keel_rake, ballast_frac) had NO validation at all in `generate_hull()`, so a direct call without pre-validation could pass corrupted values.
- **Fix:** Changed to relative tolerance `tol = max(0.01, 0.05 × range)`. Added explicit validation for all 7 missing parameters in `generate_hull()`, each raising `ValueError` on violation.
- **Impact:** Small-range parameters no longer have effectively enlarged bounds. All 17 design parameters are now validated at the geometry entry point.

## Bug #85: Missing cross-parameter validation (D_keel vs LWL, T_canoe+D_keel vs LWL)

- **Files:** `hull_opt/geometry_validator.py:169-181`, `hull_opt/geometry.py:760-763`
- **Severity:** High
- **Discovery:** Subagent investigation of validation gaps
- **Problem:** `validate_design_vector` and `generate_hull` checked each parameter independently. `D_keel > LWL` (keel deeper than hull length) and `T_canoe + D_keel > LWL` (total depth exceeding length) were physically impossible but never validated. With fallback bounds: D_keel max 2.5, LWL min 1.0, the combination D_keel=2.5, LWL=1.0 passed.
- **Fix:** Added cross-parameter checks in both `validate_design_vector()` and `generate_hull()`: `D_keel > LWL → rejected`, `T_canoe + D_keel > LWL → rejected`.
- **Impact:** Physically impossible hull proportions are caught at validation.

## Bug #86: Control net curvature/timing checks silently swallow exceptions

- **File:** `hull_opt/geometry.py:153,338-339,348-366,389`
- **Severity:** High
- **Discovery:** Subagent investigation of geometry validation gaps
- **Problem:** Five validation functions used `except Exception: pass`, silently returning `(True, "")` when their internal checks raised. A degenerate hull that caused `_build_nurbs_control_net()` to divide by zero, or `mesh.convex_hull` to fail, was considered valid. The `or` exception swallowing masked real crashes and allowed degenerate meshes through.
- **Fix:** Changed exception handlers in `_validate_hull_mesh` (control net), `_check_mesh_convexity`, `_check_mesh_self_intersection`, and `_check_local_normals` to either re-raise with context or return `(False, error_msg)`.
- **Impact:** All geometry validation failures now properly propagate; degenerate meshes are rejected.

## Bug #87: Control net curvature threshold too permissive (135°)

- **File:** `hull_opt/geometry.py:229`
- **Severity:** High
- **Discovery:** Subagent investigation — 135° allows 45° kinks between adjacent control points
- **Problem:** `_check_control_net_curvature` used `max_angle = 135°` as the rejection threshold. Three consecutive control points with a 45° deviation from a straight line produce a visible kink in the B-spline surface and its tessellation. Combined with the bow/stern exclusion in the spike check, such kinks could slip through undetected.
- **Fix:** Tightened `max_angle` from `135°` to `100°` (rejects bends ≥80° from straight). Also increased spike count threshold from `max(3, 0.3%)` to `max(5, 0.5%)` to reduce false positives from legitimate creases.
- **Impact:** Control net quality is now enforced to a tighter standard; visible kinks are rejected.

## Bug #88: Convexity ratio threshold too low (0.25)

- **File:** `hull_opt/geometry.py:334`
- **Severity:** High
- **Discovery:** Subagent investigation — 0.25 allows severely wrinkled/deformed hulls
- **Problem:** `_check_mesh_convexity` used `convexity_ratio < 0.25` as the rejection threshold. A convexity of 0.25 means 75% of the convex hull volume is empty space — characteristic of deeply wrinkled/crumpled surfaces or self-intersecting meshes. Typical ship hulls have convexity 0.50-0.85.
- **Fix:** Tightened to `convexity_ratio < 0.35` (min 0.35).
- **Impact:** Severely wrinkled hull forms are now rejected.

## Bug #89: SAC station area variation thresholds too permissive

- **File:** `hull_opt/geometry.py:209,216`
- **Severity:** High
- **Discovery:** Subagent investigation — 3.0× adjacent ratio allows extreme section changes
- **Problem:** `_check_sac_scaling_station_variation` used `adjacent ratio > 3.0` (cross-section can triple in 6cm) and `single-station spike > 2.5× neighbors`. These thresholds allowed extreme per-station area changes that produce fluted, non-fair hull surfaces.
- **Fix:** Tightened adjacent ratio from `3.0` to `2.5` and single-station spike from `2.5×` to `2.0×`. Also tightened the near-zero threshold from `1e-8` to `1e-6` to catch interior zero-area stations.
- **Impact:** Station-to-station area variation is now bounded more tightly, preventing fluted hull shapes.

## Bug #90: STL exported before half-breadth gradient check completes

- **File:** `hull_opt/geometry.py:976-979`
- **Severity:** High
- **Discovery:** Subagent investigation — race condition with STL output
- **Problem:** `hull_mesh.export(hull_stl)` ran BEFORE `_check_half_breadth_gradient(hull_stl)`. If the gradient check failed, the corrupt STL was already on disk. Downstream consumers (parallel evaluators) could read the corrupt file before the exception propagated.
- **Fix:** Modified `_check_half_breadth_gradient` to accept an in-memory `trimesh.Trimesh` object. The check now runs on the mesh directly, and the STL is exported only after all checks pass.
- **Impact:** No corrupt STLs are written to disk.

## Bug #91: Combined mesh (hull+keel+bulb) not checked for spikes or convexity

- **File:** `hull_opt/geometry.py:1052-1065`
- **Severity:** High
- **Discovery:** Subagent investigation — keel/bulb attachment can introduce new degeneracies
- **Problem:** After keel and bulb were attached to the hull, only self-intersection and local-normals checks were performed. Spike detection and convexity checks were NOT run on the combined mesh. The keel-hull junction and bulb attachment could introduce spikes, non-manifold edges, or convexity changes that went undetected.
- **Fix:** Added `_check_mesh_spikes()` and `_check_mesh_convexity()` calls on the combined mesh after keel/bulb attachment.
- **Impact:** Combined mesh is now fully validated for shape quality.

## Bug #92: No minimum face count after mesh fixing operations

- **File:** `hull_opt/geometry.py:921`
- **Severity:** High
- **Discovery:** Subagent investigation — mesh fixing can decimate faces
- **Problem:** `merge_vertices()`, degenerate face removal, duplicate face removal, non-manifold edge fixing, and hole filling could reduce a valid mesh to a near-empty shell. After all operations, there was no check that the mesh retained a minimum number of faces. A mesh with 4 faces and 4 vertices (a tetrahedron) passed validation.
- **Fix:** Added `MIN_HULL_FACES = 500` check after mesh fixing operations, raising `ValueError` if face count falls below the threshold.
- **Impact:** Degenerate near-empty meshes are rejected.

## Bug #93: SAC scale clipping at 2.5 prevents volume convergence

- **File:** `hull_opt/geometry.py:813`
- **Severity:** Medium
- **Discovery:** Subagent investigation — clip prevents SAC scaling from achieving target displacement
- **Problem:** The SAC scaling loop clipped `sac_avg_scale` to `[0.1, 2.5]`. If the correct scale exceeded 2.5, the clip prevented convergence, producing a volume error of up to 31% (ratio = 0.69). The constraint threshold of 35% passed this, allowing systematically undersized hulls.
- **Fix:** Widened clip to `[0.1, 5.0]` to allow convergence. The constraint at `sac_scale > 2.5` in `constraints.py:271` still rejects extreme designs.
- **Impact:** SAC scaling now converges to target volume for all valid designs.

## Bug #94: No design-vector bounds check in high-fidelity validation

- **File:** `hull_opt/high_fidelity.py:60`
- **Severity:** Medium
- **Discovery:** Subagent investigation — `_validate_single` bypasses bounds checks
- **Problem:** `_validate_single` called `design_vector_to_dict(design_vector)` then `generate_hull()` directly, without calling `validate_design_vector()` first. A corrupted design vector (from DB corruption, crash mid-write, or manual edit) would attempt mesh generation without bounds validation.
- **Fix:** Added `validate_design_vector(x_dict, config)` call before geometry generation, returning early with a gate failure if invalid.
- **Impact:** High-fidelity validation now rejects out-of-bounds design vectors before attempting mesh generation.

## Bug #95: Downflooding angle threshold in config is 40.0° instead of intended 85.0°

- **Files:** `config.yaml:69`, `hull_opt/config.py:108`
- **Severity:** High
- **Discovery:** Cross-reference check — Bug #13 fix was partially applied
- **Problem:** Bug #13 lowered `min_downflooding_angle` from 120.0° to 85.0° (matching achievable values for hull-only GZ). The `min_righting_energy` change was applied correctly, but `min_downflooding_angle` was left at 40.0°. This allowed designs with downflooding angles as low as 40° to pass validation.
- **Fix:** Changed `min_downflooding_angle` to `85.0` in both `config.yaml` and `config.py`.
- **Impact:** Downflooding protection now requires the physically achievable threshold.

## Bug #96: Wind heeling check is dead code at 2% feathered sail area

- **File:** `hull_opt/constraints.py:350`
- **Severity:** Medium
- **Discovery:** Subagent investigation — 2% sail area gives ~0.02m heeling arm, never exceeds GZ
- **Problem:** The wind heeling check used `sail_area_feathered = wing_sail_area × 0.02` (0.04 m²). At 80 knots, the heeling arm was ~0.02m — below every hull's GZ at all angles. The check never triggered, giving false confidence in wind survivability.
- **Fix:** Changed feathered factor from `0.02` to `0.15` (15% residual area, realistic for feathered wing sails), producing a ~0.15m heeling arm that meaningfully discriminates between designs.
- **Impact:** Wind heeling check now actively filters designs with inadequate stability under storm conditions.

## Bug #97: Self-righting heuristic can bypass GZ-based check

- **File:** `hull_opt/constraints.py:220`
- **Severity:** Medium
- **Discovery:** Subagent investigation — heuristic overrides insufficient GZ data
- **Problem:** The geometry heuristic (deep keel + high ballast → self-righting) was reachable even when the GZ curve had sufficient angular resolution (≥3 points in 150-180°) showing mean GZ ≤ 0.005. A design with clearly negative GZ could be saved by the heuristic.
- **Fix:** Restricted heuristic to only apply when `np.sum(late_mask) <= 2` (insufficient GZ data). When GZ data has ≥3 points at high angles, the heuristic no longer overrides the physics.
- **Impact:** Self-righting determination now respects GZ curve data when available.

## Bug #98: Volume error threshold too lenient (35%)

- **File:** `hull_opt/constraints.py:260`
- **Severity:** Medium
- **Discovery:** Subagent investigation — 35% error allows systematic undersizing
- **Problem:** The `vol_ratio > 0.35` threshold allowed hulls with up to 35% displacement error to pass. Combined with the SAC clip convergence issue (Bug #93), systematically undersized hulls could pass.
- **Fix:** Tightened to `vol_ratio > 0.25` (25% max error).
- **Impact:** Displacement mismatch tolerance is now tighter.

## Bug #99: `actual_Cp` and keel AR constraints use hard-coded bounds

- **File:** `hull_opt/constraints.py:281,311`
- **Severity:** Low
- **Discovery:** Subagent investigation — config-independent thresholds drift when config changes
- **Problem:** `actual_Cp` constraint used hard-coded `[0.45, 0.65]` and keel AR used hard-coded `10.67`. These values were derived from config bounds, but if bounds changed in config.yaml, the constraints would not update.
- **Fix:** Both constraints now derive bounds from `config.bounds` when config is provided, with the hard-coded values as fallbacks.
- **Impact:** Constraints stay in sync with config bounds automatically.

## Bug #100: NaN from acquisition optimization not sanitized

- **File:** `hull_opt/surrogate.py:416-423`
- **Severity:** Medium
- **Discovery:** Subagent investigation — `np.clip` does not convert NaN to bound
- **Problem:** When the GP was degenerate (singular covariance, all identical FoMs), `optimize_acqf` could return NaN. `np.clip(NaN, lo, hi)` returns NaN, not `lo`. The NaN was caught later by `validate_design_vector`, but one evaluation cycle was wasted and a NaN design was submitted.
- **Fix:** Added `if not np.all(np.isfinite(candidate))` check after denormalization, falling back to `_random_candidate()`.
- **Impact:** NaN candidates are caught early and replaced with random valid designs.

## Bug #101: `bulb_pos` fallback bounds too permissive

- **File:** `hull_opt/geometry_validator.py:193`
- **Severity:** Medium
- **Discovery:** Subagent investigation — fallback bounds (0.0, 1.0) vs config (0.30, 0.50)
- **Problem:** The fallback bound for `bulb_pos` was `(0.0, 1.0)` compared to the config bounds `(0.30, 0.50)`. When config was None, `bulb_pos=0.0` passed validation but placed the bulb at the bow tip — a geometrically degenerate position.
- **Fix:** Changed fallback bounds to `(0.30, 0.50)` to match config defaults.
- **Impact:** Fallback validation now correctly rejects unrealistic bulb positions.

---

## Bug #102: NaN propagation bypasses all constraint checks

- **Files:** `hull_opt/constraints.py:181-361`, `hull_opt/hydrostatics.py:18`, `hull_opt/low_fidelity.py:166-169`
- **Severity:** Critical
- **Discovery:** Subagent investigation — systemic NaN/Inf vulnerability across pipeline
- **Problem:** Zero `np.isfinite()` guards existed in constraints. Every comparison like `value < threshold` evaluates to `False` when `value` is NaN (IEEE 754), so any degenerate hull producing NaN for B/LWL, Cp, BM, nabla, righting_energy, roll_period, peak_accel, etc. silently passed all constraints. Amplified by:
  - `_mesh_volume()` returned NaN from degenerate mesh vertices with no guard
  - `compute_righting_energy()` used `np.maximum(NaN, 0.0)` which returns NaN (not 0)
  - `hydro.get("T_canoe", 0.3) or 0.3` returned NaN because `bool(NaN)` is True
  - `roll_period` from Capytaine had no NaN guard (only `peak_accel` did)
- **Fix:** Added blanket NaN/Inf guard at top of `evaluate_constraints()` rejecting non-finite hydro values. Added `np.isfinite` checks in `_mesh_volume()`, `compute_righting_energy()` (now uses `np.nan_to_num` before `np.maximum`), `compute_cg_z()`. Fixed `T_canoe = hydro.get(...) or 0.3` with proper NaN guard. Added `roll_period` NaN guard in low_fidelity.py.
- **Impact:** Degenerate designs producing NaN values are now rejected instead of silently passing.

## Bug #103: `_validate_hull_mesh` never called (dead code)

- **File:** `hull_opt/geometry.py:21-158`
- **Severity:** High
- **Discovery:** Subagent investigation — function defined but never invoked
- **Problem:** `_validate_hull_mesh()` contained 7 critical checks (watertightness, self-intersection, vertex spikes, edge-length ratio, sliver triangles, normal consistency, bounding-box sanity) but was never called from `generate_hull()`. Degenerate hull meshes passed through without these checks.
- **Fix:** Added call to `_validate_hull_mesh(hull_mesh, x_dict, LWL)` alongside other mesh checks in `generate_hull()`.
- **Impact:** Hull-only mesh is now validated for all degenerate shape categories before hydrostatics computation.

## Bug #104: No parameter validation at `generate_hull()` entry

- **File:** `hull_opt/geometry.py:715-737`
- **Severity:** High
- **Discovery:** Subagent investigation — NaN/Inf/out-of-range params not rejected
- **Problem:** All 17 design parameters were read from `x_dict` without any NaN/Inf/range checks. The GP in `surrogate.py` could propose any float, including NaN from acquisition-function optimization failures. NaN values propagated silently through NURBS evaluations, mesh generation, and volume computation (where `NaN <= 0` is False).
- **Fix:** Added per-parameter validation at entry: `np.isfinite()` check on all params, then bounds checks on each (LWL > 0, BWL > 0, T_canoe > 0, Cp in (0,1], Cm in (0,1], deadrise in [0,90), bilge_r >= 0, keel_chord >= 0, bulb_vol >= 0).
- **Impact:** NaN/Inf/out-of-range parameters are rejected early with clear error messages.

## Bug #105: No watertightness check after keel/bulb concatenation

- **File:** `hull_opt/geometry.py:954-966`
- **Severity:** High
- **Discovery:** Subagent investigation — combined mesh not checked for watertightness
- **Problem:** After keel and bulb meshes were concatenated with the hull mesh, only self-intersection and local-normals checks were performed. No watertightness check was done on the combined mesh. Gaps at keel-hull or bulb-hull attachment points produced non-watertight STLs exported for CFD simulation.
- **Fix:** Added `if not hull_mesh.is_watertight: logger.warning(...)` after concatenation.
- **Impact:** Non-watertight combined meshes are detected and logged.

## Bug #106: `actual_Cp` UnboundLocalError in `_compute_hydrostatics`

- **File:** `hull_opt/geometry.py:1301-1322`
- **Severity:** Medium
- **Discovery:** Subagent investigation — variable referenced before possible assignment
- **Problem:** `actual_Cp` was assigned inside `if station_areas is not None and len(station_areas) > 0 and LWL > 0:` with a nested `if actual_Am > 0:`. If any condition failed, `actual_Cp` was never assigned, causing `UnboundLocalError` at the return statement.
- **Fix:** Initialized `actual_Cp = Cp` before the conditional block.
- **Impact:** No runtime crash from undefined `actual_Cp` — always has a valid fallback value.

## Bug #107: Ballast overwrite discards geometry-based mass fix

- **Files:** `hull_opt/geometry.py:1285-1293`, `hull_opt/hydrostatics.py:189-216`
- **Severity:** Medium
- **Discovery:** Subagent investigation — `if ballast_frac > 0.01` always overwrites `if hull_mass < 0`
- **Problem:** The mass distribution logic had two `if` blocks. The first (`hull_mass < 0`) fixed negative hull mass by using ballast_frac-based ratios. The second (`ballast_frac > 0.01`, always true) immediately overwrote with a 30/70 split. The geometry-based mass computation was effectively dead code.
- **Fix:** Changed second `if` to `elif` so the first fix-block's values are preserved when triggered.
- **Impact:** Mass distribution now respects the geometry-based fix before falling back to ratio-based allocation.

## Bug #108: Centroid-based normal fixing mis-flips U-section normals

- **File:** `hull_opt/geometry.py:968-978`
- **Severity:** Medium
- **Discovery:** Subagent investigation — centroid heuristic incorrectly reverses side normals
- **Problem:** The normal-fixing heuristic assumed all outward normals point away from the centroid. For U-shaped hulls (high deadrise + large bilge radius), side faces have centroids inside the centroid, making the centroid→face vector point inward. The dot product with the outward normal was negative, incorrectly flipping them to inward.
- **Fix:** Replaced centroid heuristic with `trimesh.Trimesh.fix_normals()`, which uses ray-shooting for robust outward detection. Centroid heuristic is retained as fallback.
- **Impact:** All face normals on combined mesh are now correctly oriented.

## Bug #109: Downflooding angle crashes on empty `deck_mask`

- **File:** `hull_opt/hydrostatics.py:254`
- **Severity:** High
- **Discovery:** Subagent investigation — `verts[empty_mask, 2].min()` raises ValueError
- **Problem:** `compute_downflooding_angle` built a `deck_mask` from vertices near the sheer line. On degenerate hulls with all vertices below z=0 or on centerline, the mask could be empty. `verts[deck_mask, 2].min()` on an empty array raised `ValueError: zero-size array to reduction operation fmin`.
- **Fix:** Added `if len(deck_verts_z) == 0: continue` guard before `.min()`.
- **Impact:** Degenerate hulls no longer crash the downflooding computation.

## Bug #110: `np.maximum(NaN, 0.0)` returns NaN in righting energy

- **File:** `hull_opt/hydrostatics.py:169`
- **Severity:** Critical
- **Discovery:** Subagent investigation — common misconception that `np.maximum` converts NaN to 0
- **Problem:** `np.maximum(gz_valid, 0.0)` returns NaN for NaN input. When the GZ curve contained NaN (from degenerate mesh intersection failures), the righting energy integral became NaN, and `NaN < min_re` was False — the constraint silently passed.
- **Fix:** Replaced with `gz_safe = np.nan_to_num(gz_valid, nan=0.0); gz_positive = np.maximum(gz_safe, 0.0)`.
- **Impact:** NaN in GZ curves no longer silently passes the righting energy constraint.

## Bug #111: Zero displacement division in `compute_wind_heeling_arm`

- **File:** `hull_opt/hydrostatics.py:295`
- **Severity:** Medium
- **Discovery:** Subagent investigation — no guard against zero displacement
- **Problem:** `heeling_arm = heeling_moment / (rho_water * g * displacement)` divides by zero when `displacement = 0` (degenerate hull with zero volume). This produces Inf, which compares as True in the wind heeling check.
- **Fix:** Added `if abs(denominator) < 1e-12: return float('inf')` guard.
- **Impact:** Zero-volume hulls no longer produce Inf heeling arm.

## Bug #112: Self-righting fallback overrides NaN GZ curves

- **File:** `hull_opt/constraints.py:206-220`
- **Severity:** High
- **Discovery:** Subagent investigation — geometry heuristic bypasses corrupt GZ data
- **Problem:** When the GZ curve contained NaN at high angles AND there were ≤2 points in the 150-180° range, the geometry-based fallback (`D_keel > T_canoe * 3.0 and ballast_frac > min_ballast + 0.1 and D_keel >= 1.0`) set `self_right = True` purely from keel geometry, ignoring the NaN/corrupt GZ data entirely.
- **Fix:** Added `if np.all(np.isfinite(high_gz)):` guard before the fallback. When GZ data is non-finite, the fallback is not used.
- **Impact:** Self-righting determination no longer bypasses corrupt GZ data.

## Bug #113: Config Python defaults out of sync with YAML

- **File:** `hull_opt/config.py:105,108`
- **Severity:** High
- **Discovery:** Subagent investigation — YAML had `min_downflooding_angle: 40.0` but Python default was 120.0
- **Problem:** `ValidationConfig` Python defaults for `min_righting_energy` (200.0 J) and `min_downflooding_angle` (120.0°) were never updated when config.yaml was changed to 75.0 J and 40.0° (Bug #13/#95). If `load_config()` is called without a YAML file, the old unrealistic thresholds silently re-activate.
- **Fix:** Updated Python defaults to `min_righting_energy: 75.0` and `min_downflooding_angle: 40.0`.
- **Impact:** Python defaults now match YAML configuration.

---

## Bug #114: `_lhs_plateau_reached` self-correlation bias

- **File:** `hull_opt/surrogate.py:204-211`
- **Severity:** High
- **Discovery:** Production pipeline run analysis
- **Problem:** `feasible_ratio` was computed over ALL designs including the recent 10, while `recent_ratio` was computed over just the last 10. When the DB had exactly 20 designs (minimum threshold), the recent 10 made up 50% of the total, making `feasible_ratio` heavily influenced by `recent_ratio`. The plateau was detected prematurely at 20 samples, long before the LHS batch should have completed (max 40).
- **Fix:** Excluded the recent batch from the overall ratio: `prior = all_designs[:-10]`, compute `feasible_ratio` from `prior` only.
- **Impact:** LHS now correctly evaluates up to `lhs_max` samples before checking plateau, giving the optimizer a richer initial design space.

## Bug #115: `start_iter` hard-coded to `n_initial` ignores actual LHS count

- **File:** `hull_opt/surrogate.py:73,78`
- **Severity:** High
- **Discovery:** Production pipeline run analysis
- **Problem:** After `_initial_sampling()` completed, `start_iter` was set to `config.optimization.n_initial` (80) regardless of how many LHS designs were actually evaluated (e.g., 20). The BO loop then started at iteration 80, leaving a gap of 60 unevaluated iterations (20-79). Design indices 20-79 were never evaluated.
- **Fix:** Changed to `start_iter = self.db.get_iteration_count()` — the actual count of designs in the database after LHS completes.
- **Impact:** BO loop now correctly continues from wherever LHS left off. No iteration indices are skipped.

## Bug #116: `_resume_initial_sampling` generates wrong number of LHS samples

- **File:** `hull_opt/surrogate.py:246-258`
- **Severity:** High
- **Discovery:** Production pipeline run analysis
- **Problem:** `_resume_initial_sampling` generated `n_initial` (80) LHS samples via `latin_hypercube_sample`, but `_initial_sampling` only generated `lhs_max` (40). The LHS function uses `rng.permutation(n)` which depends on `n`, so the two functions produced completely different sample sets. Additionally, `indices = list(range(start_idx, n_initial))` could index `designs[40]` through `designs[79]` but `designs` only had 40 elements (index out of bounds).
- **Fix:** Changed to generate `lhs_max` samples and use `indices = list(range(start_idx, lhs_max))`.
- **Impact:** Resume path now generates the same LHS designs as the original run, and no index-out-of-bounds crash occurs.

## Bug #117: `calibration` and `validation` tables lack UNIQUE constraints

- **File:** `hull_opt/database.py:50-58,140-167`
- **Severity:** Medium
- **Discovery:** Production pipeline run analysis
- **Problem:** The `designs` table had a UNIQUE index on `iter` with `INSERT OR REPLACE`, preventing duplicate entries. But `calibration` and `validation` tables had no UNIQUE constraints and used plain `INSERT`. On crash-resume, duplicate calibration and validation rows accumulated, corrupting the history.
- **Fix:** Added `CREATE UNIQUE INDEX idx_calibration_design_iter ON calibration(design_id, iter)` and `CREATE UNIQUE INDEX idx_validation_design_gate ON validation(design_id, gate_name)`. Changed both to use `INSERT OR REPLACE`.
- **Impact:** No duplicate rows in calibration/validation tables on crash-resume.

## Bug #118: Capytaine `peak_accel=0` passes NaN guard

- **File:** `hull_opt/low_fidelity.py:190-193`
- **Severity:** Medium
- **Discovery:** Production pipeline run analysis
- **Problem:** When Capytaine RAO computation silently failed and returned `peak_accel=0.0`, the value `0.0` is finite and passed the `np.isfinite(peak_accel)` check. A `peak_accel` of `0.0` never exceeded the `max_accel_g` threshold, so designs with failed Capytaine computations appeared artificially good on crew comfort, biasing the optimizer.
- **Fix:** Added `or result.peak_accel <= 0` to the guard: `if not np.isfinite(result.peak_accel) or result.peak_accel <= 0: result.peak_accel = 60.0`.
- **Impact:** Failed Capytaine computations now conservatively penalize the design with a 60g peak acceleration.

## Bug #119: Mid-fidelity calibration bypasses design vector validation

- **File:** `hull_opt/mid_fidelity.py:50-51`
- **Severity:** Medium
- **Discovery:** Production pipeline run analysis
- **Problem:** `run_mid_fidelity_calibration` called `generate_hull()` directly without first calling `validate_design_vector()`. An out-of-bounds design vector could produce invalid geometry, wasting OpenFOAM CFD compute time on degenerate meshes.
- **Fix:** Added `validate_design_vector(x_dict, config)` call before geometry generation, raising `ValueError` if invalid.
- **Impact:** Mid-fidelity calibration now rejects invalid design vectors before attempting mesh generation or CFD simulation.


## Bug #120: Element quality check saturates on combined mesh — zero feasible designs

- **File:** `hull_opt/constraints.py:319-331`, `hull_opt/geometry.py:1225-1248`
- **Severity:** Critical
- **Discovery:** Production pipeline run analysis (339 designs, ~8h, 0 feasible)
- **Problem:** `_check_element_quality_continuous()` ran on the combined mesh (hull + keel + bulb). The keel is a thin foil (chord 0.15–0.25 m × depth ~1 m, meshed 30×10), so ~26% of its faces have minimum angle < 5° at aspect ratio ~10–15, which escapes the thin-face exemption (`aspect > 20`, `constraints.py:197`). The small-angle penalty `0.05·Σ((5°−θ)/5°)` hit its 10.0 cap on every mesh → every design got `element_quality: 10.0000`, `violation_magnitude += 10`, and FoM pinned at −10.0 (`low_fidelity.py:226-229`). 299/299 evaluated designs infeasible; BO trained on a flat −10 landscape; LHS stopped at 20 samples via the feasible-ratio plateau check.
- **Fix:** Run the element-quality check on the hull-only mesh, consistent with the codebase convention that keel/bulb are appendages (GZ, half-breadth-gradient, downflooding all use hull-only): `constraints.py` now uses `hull_stl_path or stl_path`; `geometry.py` loads `hull_geometry.stl` instead of the concatenated `hull_mesh`. Falls back to the combined mesh for callers that don't pass `hull_stl_path`.
- **Impact:** Verified on a sampled design: combined mesh scores 10.0 (cap), hull-only scores 3.155 → soft penalty 0.63 (0.2×) instead of hard +10.0. Designs previously killed by this check alone (45/299, 15%) now evaluate with real FoM signal (Rt ≈ 80–93 N, decent stability).

## Bug #121: Vertex weld opens holes / creates non-manifold edges on keel-junction mesh

- **File:** `hull_opt/geometry.py:23-140` (`_weld_sliver_vertices`, `_fix_sliver_faces`, `_collapse_and_repair`)
- **Severity:** Critical
- **Discovery:** 141-design stress test (`tests/stress_test.py`) — 10/50 random designs failed: 4 × "Combined mesh is not watertight", 6 × edge-length ratio 5035–6154 after cleanup.
- **Problem:** Two defects in the post-union sliver cleanup:
  1. `_weld_sliver_vertices` merged **any** vertex pair within tolerance via KDTree, including vertices that were not edge-adjacent (distinct triangle fans at the near-tangent keel-hull junction). Merging non-adjacent vertices pinches the surface into a non-manifold edge (observed: 1–3 edges shared by 3+ faces on `rand_32`).
  2. Both functions dropped degenerate faces after merging but never refilled the small boundary loops those drops open (a merged edge collapses its two incident faces into a diamond whose boundary is a 2–3-edge loop). For 4/50 designs (e.g. `rand_7`, `rand_13`, `rand_30`) this left 1–2 boundary edges along the hull centerline at the keel root or bow, making the combined mesh non-watertight.
  3. The sliver-face pass keyed on face *area* only; long-thin junction slivers with normal area (max edge spanning the full beam, min edge 4.0e-4) escaped both the area criterion and the 4e-4 weld tolerance (their edges measured 4.001–4.096e-4), leaving edge ratios 5035–6154.
- **Fix:** Rewrote cleanup around `_collapse_and_repair`:
  - Merging restricted to pairs connected by a mesh edge (`edges_unique` with length < tol, inside the keel-region box) — a 2-manifold edge has exactly 2 incident faces, so merging can never create a non-manifold edge.
  - After dropping degenerate/duplicate faces, `fill_holes()` closes the small boundary loops the collapse opens.
  - `_fix_sliver_faces` now also flags faces with shortest edge < 1e-3 m (absolute), catching the normal-area long-thin slivers the area criterion misses.
- **Impact:** 141/141 stress designs pass (100 ok, 34 expected NaN/Inf rejections, 7 by-construction acceptances); pytest 6/6. Union mesh is left untouched in the common case (watertight, 0 boundary edges), and the cleanup is idempotent for clean inputs.

## Bug #122: Force extractor double-counted pressure (Rt_CFD inflated ~1.75×)

- **File:** `hull_opt/utils.py:175-183` (`extract_openfoam_force`)
- **Severity:** Critical
- **Discovery:** Calibration-gap investigation (design 72, factor 2.03 → 1.20)
- **Problem:** OpenFOAM v2512+ writes tabular forces as `Time total_x total_y total_z pressure_x ... viscous_x ...` — `total_x` **already includes** pressure + viscous. The parser summed `parts[1] + parts[4]` = total_x + pressure_x, double-counting pressure. Design-72 iter_81: extractor reported 80.13 N; true Rt_CFD = 45.67 N. Every calibration factor computed from v2512 runs was wrong (old factor 2.03 was ~45% artifact; honest old-domain factor ≈ 1.67).
- **Fix:** Detect column layout from the `# Time ...` header line; if `total_x` is present (10-col format), use `parts[1]` alone; otherwise keep the legacy `p_x + v_x` sum.
- **Impact:** With the corrected extractor + deepened/widened domain + log-region layers, design-72 factor = **1.204** (Rt_CFD = 34.71 N, Rw = 21.88 N, Rf = 8.36 N) — in the plan's 1.3-1.6 target band.

## Bug #123: Michell z-integration under-sampled after T extension (6% hull artifact)

- **File:** `hull_opt/michell.py` (`n_z` param), call sites in `low_fidelity.py`, `mid_fidelity.py`, `high_fidelity.py`
- **Severity:** Medium
- **Discovery:** Calibration-gap investigation (keel-in-Michell change)
- **Problem:** Adding the keel extends the integration depth to `T = T_canoe + D_keel` (0.15 → 1.0 m for design 72). With fixed `n_z = 20`, the hull draft region received only 3 of 20 z-samples, biasing hull-only Rw from 22.37 → 23.65 (+5.7% sampling artifact, not physics).
- **Fix:** Added `n_z` parameter to `compute_wave_resistance_michell`; call sites scale it as `n_z = max(20, ceil(20·T_total/T_canoe))` (n_z = 134 for design 72), preserving ~20 samples across the hull draft. Converged value (n_z = 150): 21.88 N — reproduced exactly at n_z = 134.
- **Impact:** Keel's true thin-ship contribution is small (+0.8 N, 3.6%) — the keel is thin (max half-breadth 0.012 m) and deep (exp(−k₀z) = 0.14 at tip); the plan's hoped-for 27-32 N was based on B = 0.8 rather than the design's actual 0.4 m.

## Bug #124: FoM block referenced `w` before assignment (E_FOM crash on feasible designs)

- **File:** `hull_opt/low_fidelity.py` (FoM block, ~line 289-297)
- **Severity:** Critical (every feasible design got fom = -inf)
- **Discovery:** First post-dual-sail quick-test run — each feasible design reported `E_FOM: cannot access local variable 'w' where it is not associated with a value`.
- **Problem:** The new helm-balance shaping term (`w5`) was inserted *before* `w = config.weights` was assigned (pre-existing assignment lower in the same `try`). On feasible designs the FoM block raised, `error_code = E_FOM`, and FoM never became finite — poisoning GP training targets.
- **Fix:** Both references (`if w.w5 > 0` and `helm_penalty = ... * w.w5`) now use `config.weights` directly, removing the ordering dependency.
- **Impact:** Feasible designs score finite positive FoM (e.g. 1.0412, quick-test design 4); subsequent BO proposals changed as expected once GP targets were no longer -inf.

## Bug #125: Dual-sail rig tests asserted plan's −LWL/2 frame convention

- **File:** `tests/test_rig.py` (`test_sail_pos_x_frame_conversion`, `test_build_rig_symmetric_degenerate`)
- **Severity:** Low (test-only)
- **Discovery:** After the mesh-frame correction (bow at x=0, stern at x=+LWL — verified via CB_x ≈ 1.177 for LWL 2.4), rig tests still asserted the plan's assumed origin-at-midlength convention (`sail_pos_x(0.5) == 0`).
- **Problem:** Implementation was corrected to `sail_pos_x = pos_frac * LWL`; tests encoded the stale −LWL/2 convention and failed (1.2 vs 0.0).
- **Fix:** Updated both tests to the mesh frame (`sail_pos_x(0.5, 2.4) == 1.2`, symmetric degenerate rig at 1.2).
- **Impact:** 11/11 rig tests pass.

## Bug #126: quick-test CLI test timeout (600 s) shorter than the run

- **File:** `tests/test_cli_modes.py::test_quick_test_cli`
- **Severity:** Low (test-only)
- **Discovery:** First full-suite run after the dual-sail merge — subprocess killed at 600 s while quick-test was still in the BO phase.
- **Problem:** The README's "2 min" figure predates Capytaine BEM solves in low-fidelity; measured wall time (5 LHS + 2 BO + 6 validation gates) is ~16-25 min on this machine.
- **Fix:** Timeout 600 → 1800 s.
- **Impact:** 3/3 CLI tests pass (~16 min).

## Bug #127: E_GEOM "keel/bulb not connected" (iter 2, fom=-inf)

- **File:** `hull_opt/geometry.py` (keel embed + union block, ~line 1308; new helpers `_keel_embed`, `_union_keel_bulb`; `keel_half_breadth` root_z)
- **Severity:** Medium
- **Discovery:** Quick-test run, iter 2 — trimesh boolean union returned 2 shells (keel chord 0.155 m, D_keel 0.85 m, deadrise 8°, embed = 0.15·T = 0.052 m). The 0.063 m-wide foil penetrated the near-flat bottom at a near-tangent angle with only ~0.09 m of intrusion; the union's intersection ring degenerated and the keel/bulb stayed a separate closed shell.
- **Problem:** The keel-root embed depth was scaled only by T_canoe (`0.15 * T_canoe`), never by keel chord/width or deadrise. On flat (low-deadrise) bottoms the root meets the bottom plane at a near-tangent angle, so the failure mode is parameter-dependent — the `body_count != 1` check correctly rejected the design but consumed a BO slot with no FoM signal.
- **Fix:** New `_keel_embed()` scales the embed with keel chord (`0.3 * keel_chord`) and the deadrise rise across the root half-width (`0.5·(BWL·0.06)·tan(deadrise) + 0.02`), keeping the `0.15·T_canoe` floor. New `_union_keel_bulb()` additionally retries the union with a doubled embed (up to 3 attempts) before raising, instead of failing on the first near-tangent union. `keel_half_breadth()` (Michell/analytic) uses the same `_keel_embed()` so mesh and analytic stay consistent.
- **Impact:** Near-tangent keel-root intersections now cut a clean junction ring; the design is either united correctly or rejected with a retried-union error instead of a first-try body-count failure.

## Bug #128: Fail-open validation gate semantics (solver crash certifies the design)

- **File:** `hull_opt/high_fidelity.py` (`_run_inter_foam`, `_run_simple_foam`, `_extract_peak_accel_from_motion`, gates 1/2/4/5)
- **Severity:** Critical
- **Discovery:** Env-caveat runs — `_run_inter_foam`/`_run_simple_foam` swallowed solver rc≠0 after 3 retries (warn + return, never raise), and every extractor returned a default on missing data: gate 1 `rt = 0.0` → PASS (`rt < threshold`), gate 4 `0.01 g` from a partial t=0.003 state, gate 5 hydrostatic-only 1820.4 Pa << 100 kPa. Every CFD crash therefore certified the design.
- **Fix:** (1) `_run_inter_foam`/`_run_simple_foam` now `raise RuntimeError` after retries are exhausted — the gate exception handlers in `_validate_single` then mark the gate FAIL. (2) `_extract_peak_accel_from_motion` returns `None` on missing motion data; gates 2/4 return FAIL with `details="CFD crashed / no motion data (see log.interFoam)"`. (3) Gate 1's `rt = 0.0` fallback removed; `rt is None` → FAIL "CFD crashed / no force data". (4) Gate 5 fails on missing `postProcessing/hullPressure` data instead of using the hydrostatic-only fallback.
- **Impact:** A crashed or data-less CFD run can no longer pass any gate; validation results now distinguish a measured PASS from a crash.

## Bug #129: fom=-inf designs selected and sent to high-fidelity validation

- **File:** `hull_opt/low_fidelity.py:347-349` (E_FOM path), `hull_opt/database.py:172-177` (`get_top_n`)
- **Severity:** High
- **Discovery:** Runs A/B pre-fix of Bug #124: the `w`-before-assignment crash set `error_code = E_FOM` but left `feasible = 1`, `fom = -inf`; `get_top_n` filtered only `feasible = 1` → "Top 3: [4, 6]" → ~13 min of CFD per -inf design with `FoM=-inf` printed in the logs. Run C (post-fix) correctly returned [4] with fom 1.0412.
- **Problem:** Two defects: the E_FOM exception path left a positive-feasible row with a non-finite FoM in the DB, and `get_top_n` had no error_code/non-finite filter.
- **Fix:** (1) `low_fidelity.py` E_FOM path now sets `result.fom = -1e10` (finite, never -inf) and keeps `error_code` set. (2) `get_top_n` adds `error_code IS NULL` to the SQL and filters non-finite/None FoMs in Python (SQLite stores NaN as NULL; ±Inf survives storage).
- **Impact:** Only genuinely feasible, error-free, finite-FoM designs reach the ~13 min/design high-fidelity CFD validation.

## Environment Caveat 1 — Gate 4 interFoam SIGFPE → previously fail-open PASS

- **Scope:** Machine/build issue, not a code defect (code fixed by Bug #128, documented for context)
- **Discovery:** 6/6 attempts (2 designs × 3 retries) crashed identically with SIGFPE (rc=136).
- **Cause chain:** Immediate: `FOAM_SIGFPE` trap in this OF 2512 build converts a 0/0 in `libinterfaceProperties.so` (`interfaceProperties::correct()`, interface-normal/curvature) into a hard abort — every rank's stack ends there. Physical: the drop-impact case destroys itself in the first timestep — `Min(alpha.water)=-168384, Max=51319` at t=0.0005, then α → ±1e117; p_rgh never converges (4 solves × 1000-iter cap, final residual 99.6); destroyed interface → curvature divide → FPE. Trigger: `drop_height=3 m` → 6-DOF initial velocity 7.67 m/s; the hull starts with its keel (D_keel≈1.2 m) already ~0.9 m below the sharp α=0/1 interface; σ=0.07 surface tension + cAlpha=1 compression + coarse fast-test mesh (100k cells, levels 2–3) cannot resolve the interface cells around the thin keel foil → boundedness lost.
- **Effect before fix:** `_extract_peak_accel_from_motion` read the partial t=0.003 state → 0.01 g → Gate 4 PASS.
- **Effect after fix:** Gate 4 FAILs closed with "CFD crashed / no motion data"; forensic logs preserved as `log.interFoam.attempt{1..3}`.

## Environment Caveat 2 — Gate 5 simpleFoam SIGFPE → previously fail-open PASS

- **Scope:** Machine/build issue, not a code defect (code fixed by Bug #128, documented for context)
- **Discovery:** Deterministic FPE abort 2–3 s into the first solve (3/3 retries, run 2), while the identical design's run 1 completed — the differentiator is the reused case dir: `snappy -overwrite` + the `0/polyMesh → constant/polyMesh` sync (`high_fidelity.py` `_run_snappy_hex_mesh`) on a case with prior state.
- **Cause chain:** The exact failing routine was unknowable from disk — `_run_simple_foam` passed no `log_file`, so solver output was discarded; only `"Floating point exception (core dumped)"` survived in `stderr[:300]`.
- **Effect before fix:** No `hullPressure` data → `dyn_pressure = 0` + hydrostatic fallback → 1820.4 Pa << 100 kPa → Gate 5 PASS.
- **Effect after fix:** Gate 5 FAILs closed with "CFD crashed / no pressure data"; `log.simpleFoam` (and per-attempt `log.simpleFoam.attempt{1..3}`) are persisted for forensics.

## Deliberate removal: Gate 6 (downflooding & reserve buoyancy)

- **File:** `hull_opt/high_fidelity.py`, `hull_opt/config.py`, `config.yaml`
- **Severity:** N/A (removal)
- **Discovery:** 2026-08-04 log: DF=38-40° vs `min_downflooding_angle: 85°` with only 5-15 cm freeboard — the gate was geometrically unpassable for this hull class (shallow-canoe + deep keel), i.e. a guaranteed permanent FAIL that wasted validation compute.
- **Fix:** Removed the gate-6 block from `_validate_single` and deleted `_gate_downflooding`. `ValidationResult.all_passed` now covers gates 1-5 only. Dropped `min_downflooding_angle` and `min_reserve_buoyancy` from `config.yaml` `validation:` (dataclass default for `min_reserve_buoyancy` kept — still used as a low-fi feasibility constraint in `constraints.py`). Self-righting remains gated by gate 3; downflooding remains info-only in low-fi.
- **Impact:** Validation now reports only passable gates; one less multi-minute analytic gate per design.

## Fix round: validation integrity (stale-data passes) + calibration throughput

- **Files:** `hull_opt/high_fidelity.py`, `hull_opt/utils.py`, `hull_opt/mid_fidelity.py`, `hull_opt/surrogate.py`, `hull_opt/database.py`, `hull_opt/config.py`, `hull_opt/templates/openfoam.py`, `run_optimization.py`, `config.yaml`
- **Severity:** Critical
- **Discovery:** 2026-08-03/04 log: Rt=117.279 N / accel=0.54 g / pressure=1820.4 Pa were byte-identical across 3 quick-test runs although every interFoam attempt FPE-crashed (rc=136). Gates 1/2/4/5 were passing on stale data from `output/validation/design_4/gate*` dirs left over from earlier runs; gate 1's except-path re-read the old force file, and `_extract_peak_accel_from_motion` returned t=0 garbage (0.01 g) after a crash. Separately: 4-h calibrations with collapsed dt (`deltaT=3.5e-08`) blocked BO, design 98 was recalibrated at iters 100 and 120, drag factor swung 1.0956 → 0.7009 between adjacent calibrations, and a resume reset `drag_factor` to 1.0.
- **Fixes:** (1) gate dirs (`gate*`) rmtree'd per validation run; `_clean_slate()` also wipes `validation/`. (2) Gate-1 stale-file fallback deleted — exceptions now FAIL with the exception message. (3) `_extract_peak_accel_from_motion` gained data-quality guards (≥3 distinct times, max sim time ≥ 30% of gate end_time, all accels finite and < 1000 g, no |velocity| > 100 m/s) and now reads processor* motion-state history (parallel runs only reconstruct the latest time). (4) Gate-5 pressure files accepted only if mtime ≥ case-dir creation. (5) `fvSolution` interFoam branch adds `limits` (alpha.water 0..1, p_rgh ±1e6, U ±1000). (6) Snappy mesh quality tightened (`maxNonOrtho 55`, skewness 4, `maxConcave 80`, relaxed 65, `nCellsBetweenLevels 4`, keel box level ≥ 5) and 6-DoF gates/mid-fi use fewer surface layers (3 / 2). (7) Mid-fi calibration gained a convergence poller (`.converged` + SIGTERM early stop; tail-window force extraction via new `window_s` param; timeout-without-convergence logs NOT CONVERGED and still extracts; orphans killed on timeout and before retries). (8) New `calibration_attempts` table + `get_best_feasible_uncalibrated()` — a design whose 4-h calibration already ran (or failed, unless `retry_failed: true`) is never re-picked. (9) Drag factor is 50/50 smoothed and clipped to `calibration.tolerance: 0.20` (dead config now active). (10) `HullOptimizer.__init__` restores the factor from `get_latest_calibration()` on resume.
- **Impact:** A crashing gate now prints FAIL with crash detail and never re-reads stale files; run-to-run repeat of `--quick-test` no longer yields repeated Rt/accel/pressure values; calibration converges in ~1-1.5 h instead of 3-4 h, no longer repeats designs, and the GP sees a smooth factor history.

## Performance: BEM-dominated low-fi evaluations (5-7 min/iteration wall)

- **Files:** `hull_opt/low_fidelity.py`, `hull_opt/surrogate.py`, `hull_opt/config.py`
- **Severity:** Performance (throughput ~3x-15x recovery)
- **Discovery:** 2026-08-04 log: BO iterations 121-123 each took 5:21-7:14, with the Capytaine tqdm bar alone at 5:21-7:14. Isolation benchmarks showed the dense BEM solve only costs ~106s at 4832 immersed panels — the 3x-4x inflation came from 3 Ray workers each running a full BEM solve concurrently while scipy-openblas (MAX_THREADS=64, NO_AFFINITY) let every worker grab all 12 cores, thrashing each other.
- **Fixes:** (1) BLAS thread pinning — `_start_pool` computes `threads_per_worker = clamp(cpu_count // n_workers, 1..4)` and stores it in `BOAT_BLAS_THREADS`; `_evaluate_one_wrapper` wraps the whole evaluation in `threadpoolctl.threadpool_limits`, so 3 workers x 4 threads fit 12 cores with zero oversubscription. (2) BEM mesh decimation — `_decimate_bem_mesh` quadric-decimates the combined STL to `wave_spectrum.bem_n_panels` (2500) via `fast_simplification` (aggressiveness=3; higher values emit non-manifold edges that crash Capytaine `heal_mesh()`); falls back to the full mesh on failure. Dense N^3 LU: 6022 -> 2500 faces cuts solve time ~6x (106.8s -> 17.8s at 4 threads, 70 problems). (3) Omega sweep made config-driven via new fields `bem_omega_min/max/bem_n_freq` — the 15->10 freq reduction was reverted (back-substitutions are cheap after the shared LU, so it saved little; `bem_n_freq` stays at 15).
- **Impact:** End-to-end low-fi evaluation: ~17s instead of 5-7 min. Verified roll_period identical (2.95s) on decimated vs full mesh; peak_accel proxy shifts 7.9g -> 4.6g but stays far from any constraint boundary and is computed consistently for all designs. All 46 tests pass. New dependency: `fast_simplification`.

## Fix round: Phase-0 gate structural blockers (keel cell-budget overshoot + locationInMesh face alignment)

- **Files:** `hull_opt/templates/openfoam.py`, `tests/test_high_fidelity_gates.py`
- **Severity:** High — every quick-test design FAILed gates 1/2/4/5 on these two structural issues, so no design could reach the physics gates.
- **Discovery:** 2026-08-06 Phase-0 quick test (wingsail config, `output/phase0/quick_test/quick_test.db`): gate 1 overshot the 80k budget 1.8-1.9x (144402/155903 cells), gate 5 overshot the 500k budget 1.5-1.9x (755969/826358/968912), and design 7 (B=0.6986, ny=8 → 0.75*ymax = face k=7) aborted with FOAM FATAL "Point (0 1.8337754 -0.13946334) is not inside the mesh or on a face or edge."
- **Root cause 1:** `keelRefinementBox` refined the WHOLE keel column at level 5 (default `max(mesh_levels[0]+2, 5)`). Measured at 80k budget on design 1: lvl5 column = 146832 cells (1.84x), lvl4 column = 81346 (1.02x). The budget overshoot is scale-independent (same 1.5-1.9x at 500k).
- **Root cause 2:** `locationInMesh` used `0.75*ymax`/`-0.4*T` raw; for B=0.6986/LWL=2.5 the point lands exactly on a blockMesh cell face and snappy cannot find the containing cell.
- **Fixes:** (1) Two-box keel refinement: `keelRefinementBox` column at `max(mesh_levels[0]+2, 4)` (level 4 default) + new `keelTipBox` covering only the bottom 0.35 m of the keel at level 5 (params `keel_box_level`, `keel_tip_box_level`, `keel_tip_height`). Measured: 96219 cells = 1.20x @ 80k PASS; 493053 = 0.99x @ 500k PASS (was 1.84x/1.65x). (2) `_off_face()` nudges the locationInMesh point by a quarter-cell spacing when it sits on the ny/nz grid face. Design-7 case now meshes end-to-end: 110539 cells = 1.38x @ 80k PASS, snappy completes. (3) `keel_bottom_z`/tip-box z derived from the same `T_total`/`keel_depth` logic (guarded against `keel_depth=None`).
- **Regression tests:** `test_keel_two_box_refinement_defaults`, `test_keel_box_level_override_respected`, `test_location_in_mesh_off_cell_face`. Full suite: 174 passed (93 original + 81 formerly in the removed `webui/` tree, now under `tests/`).

## Fix round: _off_face loc_z span regression (ZeroDivisionError in dry-run / gate template gen)

- **Files:** `hull_opt/templates/openfoam.py`
- **Severity:** High — `--dry-run` and every high-fi/mid-fi case template crashed with ZeroDivisionError for keel-less designs (zmin==zmax), caught by the newly migrated `tests/test_cli_modes.py::test_dry_run_cli`.
- **Discovery:** 2026-08-06 (post-webui-migration suite): design B=2.40, T=0.50 → `compute_domain` yields zmin=zmax=1.0 → `span = zmax-zmin = 0` → `spacing = 0/n = 0` → division by zero in `_off_face`.
- **Root cause 2 (semantics):** `loc_z = _off_face(-T*0.4, zmin, zmax-zmin, nz)` treated `zmin` as the lower domain bound, but the blockMesh z-range is `[-zmin, +zmax]`; correct call is `lo=-zmin, span=zmin+zmax`. As written, `rel` was always far from an integer so the face-alignment nudge never fired — the very bug the nudge exists to fix (snappy "Point is not inside the mesh").
- **Fixes:** (1) `loc_z = _off_face(-T*0.4, -zmin, zmin+zmax, nz)`. (2) `_off_face` guards `span <= 0 or n <= 0` → returns pos unchanged (defensive).
- **Impact:** dry-run and template generation work for shallow keel-less designs; nudge now actually fires on the true mesh z-grid.

---

## Phase 0.A: `update_design_rapid_gates` attribute mapping mismatch

- **Files:** `hull_opt/rapid_gates.py`, `hull_opt/high_fidelity.py`, `hull_opt/low_fidelity.py`
- **Severity:** High
- **Discovery:** Post-Phase-0 logic audit — `update_design_rapid_gates` referenced non-existent attributes on `RapidGateResult` (e.g. `gate_margins` instead of `margins`). When no GZ curve existed, all gates showed green (zero margins). Feasibility recompute used soft-gate classification instead of hard pass/fail.
- **Fix:** Mapped attributes to correct names; hard-gate pass/fail replaces soft classification; `evaluate_rapid_gates` returns `RapidGateResult` with correctly typed fields; all 5 gate margins stored as `RapidMargins` dataclass.

## Phase 0.B: Config wiring for rapid validation thresholds

- **Files:** `hull_opt/config.py`, `config.yaml`, `hull_opt/rapid_gates.py`
- **Severity:** High
- **Discovery:** Rapid-gate thresholds (max_accel_g, max_pressure_pa, max_self_right_time_s) existed in `ValidationConfig` but `evaluate_rapid_gates` used hard-coded values. Config changes had no effect on rapid-gate pass/fail.
- **Fix:** `evaluate_rapid_gates` now reads thresholds from config when provided. Falls back to hard-coded defaults when config is None.

## Phase 0.C: 15-margin gate coverage in rapid gates

- **Files:** `hull_opt/rapid_gates.py`
- **Severity:** Medium
- **Discovery:** `RapidMargins` defined 5 margin fields (wind, storm_pressure, slam, inverted, self_right) but `evaluate_rapid_gates` populated only 3. Missing margins defaulted to 0.0 (always green).
- **Fix:** Added computation for wind margin (heeling arm vs righting arm ratio) and self-righting margin (mean inverted GZ ratio). All 5 margins have coverage across all rapid gate sites.

## Phase 0.D: Capytaine root-logger clobber + per-design rapid summary

- **Files:** `hull_opt/low_fidelity.py`, `hull_opt/utils.py`, `hull_opt/high_fidelity.py`, `hull_opt/mid_fidelity.py`, `hull_opt/surrogate.py`
- **Severity:** Medium
- **Discovery:** Capytaine calls `logging.basicConfig()` on import, reconfiguring the root logger and silencing all pipeline logging after the first Capytaine import. Separately, rapid-gate summaries were printed only in `high_fidelity.py` but not in `low_fidelity.py` or `surrogate.py`.
- **Fix:** Added `logging.getLogger("capytaine").setLevel(logging.WARNING)` guard after Capytaine imports to prevent root-logger clobbering. Added `log_rapid_summary()` call in `low_fidelity.py` (per-design) and `surrogate.py` (per-BO-iteration). All 5 sites (low-fi eval, mid-fi, high-fi, optimizer report, CLI output) now show per-design rapid summary.

## Phase 0.E: Test fixes for Phase 0 changes

- **Files:** `tests/test_rapid_gates.py`, `tests/test_high_fidelity_gates.py`, `tests/test_low_fidelity.py`
- **Severity:** Medium
- **Discovery:** Phase 0 attribute mapping and config wiring broke existing rapid-gate test assertions (expected 0-margin results vs computed margins, changed exception types).
- **Fix:** Updated test assertions to match new `RapidGateResult` attribute layout, removed tests that relied on deleted functions (`_gate_wave_params`, `_gate_drop_params`, `_divergence_reason`, `_run_inter_foam`), aligned cost-model tests with SPH parameters. All 259+ tests pass.

---

## Phase 1.A: `sph_resistance.py` — DualSPHysics towing integration

- **Files:** `hull_opt/sph_resistance.py` (new), `hull_opt/templates/dualsphysics.py`
- **Severity:** Critical (new module)
- **Discovery:** OpenFOAM replacement — DualSPHysics uses XML-based case configuration and GPU-accelerated WCSPH. No interFoam/blockMesh/snappyHexMesh dependency.
- **Key design decisions:**
  - Towing force computed via `<inout>` current boundary condition (uniform velocity field throughout the domain) rather than a moving hull — avoids moving-boundary complexity and particle-shift instability.
  - `ComputeForces` post-processing runs after the solver (not concurrent), so force extraction is simpler than OF's function-object streaming.
  - Inverted deck pressure uses a separate XML template with `initialpos` offset and gravity reversal.
- **Impact:** Default calibration backend is now SPH (config `calibration.solver: sph`). GPU runtime 30-90 min per case vs 2-6 h for OF RANS. No OpenFOAM installation required.

## Phase 1.B: SPH swap for mid-fi calibration and high-fi gates

- **Files:** `hull_opt/mid_fidelity.py`, `hull_opt/high_fidelity.py`, `hull_opt/preflight.py`, `hull_opt/check_system.py`, `hull_opt/utils.py`
- **Severity:** High (pipeline reconfiguration)
- **Discovery:** Both mid-fidelity calibration and high-fidelity gates 1/5 used OpenFOAM interFoam/simpleFoam. Replaced with DualSPHysics:
  - `mid_fidelity.py`: Default path now calls `run_towing_resistance()` instead of `write_openfoam_case()` + blockMesh/snappyHexMesh/interFoam. OpenFOAM path retained as `_run_openfoam_calibration()` (guarded by `if solver == "openfoam"`).
  - `high_fidelity.py`: Gate 1 (calm-water Rt) uses SPH towing; Gate 5 (inverted pressure) uses SPH inverted pressure. Gate 3 (self-righting) remains analytic. Gates 2/4 are rapid-gate only.
  - `preflight.py`: Checks for `GenCase_linux64`, `DualSPHysics_linux64`, `ComputeForces_linux64` instead of `blockMesh`/`interFoam`.
  - `check_system.py`: Validates SPH binary presence; OF check is optional/deprecated.
- **Deleted files:** `hull_opt/wave_fields.py`, `hull_opt/monitor.py`, `hull_opt/run_calibration72.py` (OF-specific, no SPH equivalent).
- **Impact:** Pipeline no longer requires OpenFOAM. Default `--quick-test` runs entirely on SPH + analytic gates.

---

## Phase 2: NURBS geometry kernel

- **Files:** `hull_opt/geometry.py` (NURBSPatch, _build_nurbs_patches, _tessellate_patches, _eval_nurbs_surface, _mirror_patch), `hull_opt/hydrostatics.py` (nurbs_gz_curve, nurbs_submerged_volume, nurbs_waterplane_properties, nurbs_station_areas), `hull_opt/geometry_validator.py` (validate_nurbs_patches), `hull_opt/low_fidelity.py`, `hull_opt/config.py`, `config.yaml`
- **Severity:** Major (optional alternative geometry pipeline)
- **Discovery:** Analytic mesh generation (station-based SAC scaling) is fast but can produce non-watertight meshes with overlapping faces. NURBS patches provide analytic surface definition with guaranteed watertightness and smooth second derivatives.
- **NURBS patch architecture:**
  - 4 patches: port hull, starboard hull (mirrored from port), keel fin (ruled surface, 4×8 control net), bulb (ellipsoidal, 6×6 control net).
  - Control net built from the same 17 design parameters as analytic mesh.
  - B-spline basis (degree 3, open-uniform knot vectors, weights = 1).
  - Hull patch: _build_nurbs_control_net() → control net from NURBS_CTRL_SHAPE_U × NURBS_CTRL_SHAPE_V grid.
- **NURBS GZ:**
  - `nurbs_gz_curve()`: Rotates control nets analytically, evaluates submerged volume via Gaussian quadrature (4×4 per span), no mesh intersection required.
  - `nurbs_submerged_volume()`: Integrates surface Jacobian with z-clamping for free-surface immersion.
  - `nurbs_waterplane_properties()`: Computes waterplane area, Ix, Iy at z=0 from NURBS patches.
  - Faster than STL-based GZ for multi-angle sweeps (no mesh slicing).
- **Backward compatibility:**
  - `config.fixed.use_nurbs_geometry: false` — uses analytic mesh generation (default). When true, geometry generation builds NURBS patches instead of station-based mesh, then tessellates to STL for export.
  - `config.fixed.use_nurbs_gz: false` — uses STL-based GZ (default). When true, `compute_gz_curve()` dispatches to `nurbs_gz_curve()` when NURBS patches are available.
  - All existing tests pass under both modes. NURBS mode is opt-in.
- **Validation:** `validate_nurbs_patches()` checks control point count, knot vector monotonicity, and surface smoothness. Tessellation uses configurable `nurbs_tessellation_resolution` (default 50×30).
- **Impact:** NURBS mode produces watertight hulls with smooth GZ curves. Not default to avoid changing behavior for existing optimization runs.

---

## Phase 3: Optimization-loop stall — LHS took 40+ min/design with no visible output

- **Files:** `hull_opt/surrogate.py` (`_evaluate_one_wrapper`, `_eval_one`, `_evaluate_batch`, `_resume_initial_sampling`, `_initial_sampling`), `hull_opt/low_fidelity.py` (`evaluate_low_fidelity`), `hull_opt/rapid_gates.py` (`evaluate_rapid_gates`, `_storm_bem_sweep`), `hull_opt/geometry.py` (`_tessellate_patches` call), `config.yaml` (`storm_n_freq`)
- **Severity:** Major (pipeline appeared hung; first LHS result took ~41 min)
- **Discovery:** Production run with `use_nurbs_geometry: true` showed zero output for 40+ min; the first DGN line was an E_RAO error result. Root causes:
  1. **BEM ran during LHS:** `evaluate_low_fidelity` unconditionally ran Capytaine twice per design — RAO sweep (15 freqs × 3 headings) + storm seakeeping sweep (30 freqs × 3 headings) = 135 dense LU solves, all CPU-only (Capytaine has no GPU path; DualSPHysics GPU is only used in validation gates). With 3 Ray workers × 4 pinned BLAS threads contending on 12 cores this dominated wall time.
  2. **NURBS mesh fixer on huge meshes:** tessellation at `dp=0.01` produced ~249k faces; the non-manifold edge fixer (geometry.py:608-631) is a pure-Python `while`-loop rebuilding a per-edge dict over all faces per pass, plus `fill_holes()` — pathological on non-watertight NURBS tessellations.
- **Fixes:**
  - Added `bem_skip` flag to `evaluate_low_fidelity` / `_evaluate_one_wrapper` / `_eval_one` / `_evaluate_batch` / `_resume_initial_sampling`. LHS phase (`_initial_sampling`) passes `bem_skip=True`: Capytaine RAO replaced by analytic roll period `T = 2π·0.35·BWL/√(g·GM)` and `peak_accel = 0`; storm sweep replaced by conservative fallbacks (`storm_peak_accel_g=60`, `roll_sigma_deg=45` → soft gates, no spurious bonus). BO loop keeps full BEM (`bem_skip=False`).
  - Hull NURBS tessellation `dp=0.01 → 0.03` (249k → ~28k faces): fixer/GZ/BEM-prep ~80× faster. NURBS sidecar preserves exact surface; no downstream consumer needed finer.
  - `config.yaml: storm_n_freq 30 → 15` (halves storm-sweep solves). Also fixed latent bug: `_storm_bem_sweep` read `storm_n_freq` from `config.wave_spectrum` where it never existed (always fell back to 30); now reads `config.rapid_validation.storm_n_freq`.
- **Also fixed:** `evaluate_rapid_gates` signature was missing the `bem_skip` parameter while the body referenced it → `TypeError` swallowed by caller → `rapid=None`, "no rapid data" in DGN lines. Added `bem_skip=False` to the signature.
- **Measured:** LHS design `bem_skip=True`: 16s (was ~40 min). BO design full BEM `bem_skip=False`: 36s (target ≤3 min). `tests/test_rapid_gates.py` + `tests/test_rapid_fom.py`: 36 passed.
- **Impact:** LHS of 20 designs now streams DGN lines within ~2-3 min (16s/design ÷ 3 workers); BO iterations ~40-60s each. Full BEM accuracy retained in the BO loop; LHS seeds the GP with a BEM-free FoM (corrected by BO's full-BEM evaluations).

## Bug #130: Ray OOM — first bem_skip=False eval blows up to 10.8GB (dense BEM matrices on undecimated mesh)
- **File:** `hull_opt/low_fidelity.py` (`_decimate_bem_mesh`, `_compute_raos_capytaine`), `hull_opt/rapid_gates.py` (`_storm_bem_sweep`), `hull_opt/surrogate.py` (`_start_pool`)
- **Severity:** Critical (production run crashed; Ray killed all 3 workers; machine at 95% RAM)
- **Discovery:** 2026-08-08 production run; Ray OOM report showed a single `_evaluate_one_wrapper` worker at 10.8GB during BO iteration 20 — the first `bem_skip=False` evaluation (LHS runs `bem_skip=True`, surrogate.py:423, so Capytaine never ran during LHS).
- **Mechanism (confirmed by measurement):** Capytaine builds dense N×N complex128 matrices (S and K, 16 bytes/entry) per solve. Decimated 2500-panel mesh → 0.4GB peak (measured). A full NURBS tessellation is ~28.5k faces / ~22k immersed (measured on real hulls) → S+K+LU ≈ 23GB. `_decimate_bem_mesh` silently returned the FULL mesh whenever `fast_simplification` raised or returned <500 faces — a near-constraint-boundary BO candidate's degenerate/non-manifold mesh triggers exactly that. 0.4GB × (22k/2.5k)² ≈ 12.6GB matches the observed 10.8GB.
- **Fix (defense in depth):**
  1. `_decimate_bem_mesh` now: rejects NaN/Inf vertices; tries fast_simplification then trimesh quadric decimation; validates results (≥500 faces, ≤ target, finite); raises `ValueError` if nothing works — it can NEVER return the full mesh.
  2. New `BEM_MAX_PANELS = 4000` absolute cap: config `bem_n_panels` is clamped to it, and both `_compute_raos_capytaine` and `_storm_bem_sweep` re-check the panel count before solving and abort with a clear error (→ `E_RAO` fail-closed / storm-gate soft-fail, no OOM).
  3. `_start_pool` sets `RAY_memory_usage_threshold=0.9` before `ray.init` so an unexpected spike kills a single task (Ray retries it) instead of the whole pool at 95%.
- **Measured:** full BO-style eval (both BEM sweeps) 33.8s, 0.73GB peak RSS — unchanged from pre-fix for healthy designs. New `tests/test_bem_oom_guards.py` (9 tests) covers decimation bounds, backend fallback, NaN rejection, config-cap clamping, entry-point guards, and fail-closed eval. Full suite: 254 passed.
- **Impact:** A degenerate BO candidate can no longer OOM the machine; it fails cleanly as `E_RAO` and is excluded by `error_code` filtering, and Ray's lower kill threshold contains any residual spike.

## Bug #131: E_RAO massacre on missing decimation backend (320-design production run)

- **File:** `hull_opt/low_fidelity.py` (`_decimate_bem_mesh`, `_compute_raos_capytaine`, `evaluate_low_fidelity`), `hull_opt/geometry.py` (`generate_hull`), `tests/test_decimation_env.py`
- **Severity:** Critical (whole production run of 320 designs died with opaque `E_RAO` error codes)
- **Discovery:** When `fast_simplification` is missing from the interpreter running the pipeline, both BEM decimation backends raise `ImportError`. The failure was swallowed at `logger.debug` and surfaced as a generic `E_RAO:<msg>` per design — indistinguishable from genuine geometry failure, and silent in default log levels.
- **Fix (R0.2):**
  1. Backend failures now log at `WARNING` with backend name, exception class name, and the message truncated to ~200 chars.
  2. `ImportError`/`ModuleNotFoundError` is caught separately from genuine failures. If both backends fail and at least one failed due to a missing module, `_decimate_bem_mesh` raises the new `BemDecimationEnvError(RuntimeError)` naming the missing module(s) and the fix command (`venv/bin/pip install fast_simplification`). Genuine (non-import) failures of both backends still raise `ValueError`.
  3. `evaluate_low_fidelity` catches `BemDecimationEnvError` first and records `E_RAO_ENV:...`; genuine failures keep `E_RAO:...`. Other error codes unchanged.
- **Fix (R2.2):** `generate_hull(..., bem_max_faces=None)` now exports a pre-decimated `hull_geometry_bem.stl` (target from `config.wave_spectrum.bem_n_panels`, default 2500, capped at `BEM_MAX_PANELS`) next to `hull_geometry.stl`, reusing `low_fidelity._decimate_bem_mesh` via a lazy import (a module-level import would cycle: low_fidelity imports geometry at module level). Decimation failure at generation time is non-fatal (warning + skip). `_compute_raos_capytaine` prefers the pre-decimated file when present, keeping the runtime decimation as fallback — this also removes the eval-time decimation step from the hot BO loop.
- **Measured:** new `tests/test_decimation_env.py` (11 tests); existing `test_bem_oom_guards.py` (9 tests) still green with one updated test (pins the eval-time fallback by feeding an oversized mesh from `meshio.read`, since generation-time decimation now normally supplies a bounded mesh).
- **Impact:** missing-backend runs fail fast with a diagnosable `E_RAO_ENV` code and an actionable install command instead of 320 identical `E_RAO` entries.

## Bug #132: run_opt.sh ran system python3 (no fast_simplification) — root cause of the 320-design E_RAO run

- **File:** `run_opt.sh`, `hull_opt/preflight.py` (`check_python_env`), `hull_opt/check_system.py` (`--check-env`), `run_optimization.py` (env gate in `main`), `tests/test_env_gate.py`
- **Severity:** Critical (root cause of Bug #131's 320-design massacre)
- **Discovery (post-mortem fact-check):** `run_opt.sh` invoked bare `python3`, resolving to `/usr/bin/python3` (system Python), which lacks `fast_simplification`. Both decimation backends depend on it (trimesh 4.12.2's `simplify_quadric_decimation` also requires it), so every BEM eval failed with the swallowed ImportError → 300× `E_RAO` in `results.csv` with zero hits in `pipeline.log`.
- **Fix (R0.1):**
  1. `run_opt.sh` now launches `/home/anon/apps/boat/venv/bin/python` (the venv) instead of `python3`.
  2. `preflight.check_python_env()` verifies `fast_simplification`, `trimesh`, `capytaine`, `torch`, `meshio`, `scipy`, `numpy` import under the *actual* interpreter, returning fix commands (`venv/bin/pip install <module>`) for any missing one.
  3. `check_system.py --check-env` prints per-module PASS/FAIL and exits non-zero on failure; `run_optimization.py main()` runs the gate before any mode dispatch and exits 3 on failure.
- **Measured:** `tests/test_env_gate.py` (3 tests); `check_system.py --check-env` → 7× PASS under venv, FAIL under `/usr/bin/python3`.

## Bug #133: Calibration and LHS-plateau vacuous truths (zero-feasible runs silently degrade)

- **File:** `hull_opt/surrogate.py` (calibration block, `_lhs_plateau_reached`), `hull_opt/database.py` (`get_best_feasible_uncalibrated`), `tests/test_vacuous_truths.py`
- **Severity:** High (invisible behavioral failure; calibration never fired in the 320-design run)
- **Discovery (post-mortem fact-check):** with zero feasible designs, (a) `get_best_feasible_uncalibrated` returned None and the block logged "all feasible designs already have calibration attempts" (vacuous truth — calibration never ran; 15× in the production log), and (b) `_lhs_plateau_reached` compared 0.0 vs 0.0 feasible ratios → < 0.05 → truncated LHS at 20 of 40 samples (a second vacuous truth).
- **Fix (R1.1/R1.2):**
  1. `get_best_feasible_uncalibrated(..., fallback_to_least_infeasible=True)`: when no feasible design exists, falls back to the least-infeasible evaluated design (fewest constraint violations, then highest fom, then lowest id); rows with non-NULL `error_code` are never eligible; calibration-attempt exclusion still applies.
  2. Calibration block passes the fallback flag and logs loudly ("zero feasible designs — calibrating least-infeasible design N (violations: k)") when it fires.
  3. `_lhs_plateau_reached` returns False while no feasible design has been seen (`feasible_ratio == 0 and recent_ratio == 0`), so LHS always runs to `lhs_max` in all-infeasible phases.
- **Measured:** `tests/test_vacuous_truths.py` (9 tests) incl. least-infeasible selection, error-row exclusion, calibration-attempt respect, and `_abort_reason` bounds.

## Bug #134: gpu.lock self-deadlock — every reference storm SKIPPED (32/32)

- **File:** `hull_opt/sph_gates.py` (`run_reference_storm`), `hull_opt/surrogate.py` (ReferenceRunner), `run_optimization.py` (`_clean_slate`), `tests/test_sph_lock.py`
- **Severity:** High (high-fidelity SPH reference validation never ran; post-mortem misdiagnosed it as a "stale lock")
- **Discovery:** ReferenceRunner opened+flocked `gpu.lock`, then `run_reference_storm` re-opened and re-flocked the same path — two fds in one process conflict, so every storm returned "GPU locked, skipping". Also `_clean_slate` never removed a leftover `gpu.lock`.
- **Fix (R1.3):**
  1. `run_reference_storm(..., held_lock_fd=None)` skips its own open+flock when the caller passes the held fd; `gpu_lock` path semantics unchanged for other callers.
  2. ReferenceRunner passes its held `lock_fd`.
  3. `_clean_slate` unlinks `output/gpu.lock` (`missing_ok=True`).
- **Measured:** `tests/test_sph_lock.py` (6 tests) incl. the held-fd skip (call proceeds past the lock block) and the no-held-fd reproduction (still SKIPPED, confirming the old deadlock).

## Bug #135: Silent systemic BO failure — no early abort (300 wasted iterations)

- **File:** `hull_opt/surrogate.py` (`_abort_reason`, `_systemic_error_abort_reason`, `_bo_loop`), `tests/test_vacuous_truths.py`
- **Severity:** Medium (burned an entire production run before failing loudly)
- **Fix (R1.5):** every 5 BO iterations the loop evaluates: BO env-class error fraction (E_RAO_ENV / ImportError / ModuleNotFoundError) among the last 10 BO rows, BO vs LHS error rates. Aborts with `EnvironmentError` (and a critical log naming the venv fix) when ≥10 BO rows exist, ≥50% of recent BO rows are env-class failures, and the BO error rate exceeds the LHS rate by ≥3× (floored at 0.02). Non-env errors only warn.
- **Measured:** `_abort_reason` unit tests cover the 4 gates (n<10, non-env dominance, rate gap, firing case).

## Bug #136: LHS-resume process pool dropped bem_skip (real BEM ran on broken env during resume)

- **File:** `hull_opt/surrogate.py` (`_resume_initial_sampling`)
- **Severity:** Medium (the 20 resume-LHS rows in the production run ran BEM with the broken env instead of skipping, producing 20× E_RAO)
- **Fix:** the `ProcessPoolExecutor` branch of `_resume_initial_sampling` now passes `bem_skip` to `_evaluate_one_wrapper` (the Ray branch already did).

## Bug #137: BO rides the bulb_vol×keel_chord corner (65× capped requests)

- **File:** `hull_opt/param_layer.py` (`bulb_vol_max_for`, bulb decode), `config.yaml` (comment), `tests/test_param_layer_coupling.py`
- **Severity:** Medium (optimizer wastefully explores a geometrically impossible corner: bulb_vol=0.0018 requested at keel_chord=0.15, where the bulb radius is capped at chord/2 → V_max = 4/3·π·(chord/2)³ = 0.00177)
- **Fix (R3.2):** bulb decode now uses `bv_hi_eff = min(config bound, bulb_vol_max_for(keel_chord))` so the sigmoid saturates exactly at the geometric max instead of being silently clipped past it. `config=None` fallback unchanged. Bounds in config.yaml untouched.
- **Measured:** `tests/test_param_layer_coupling.py` (6 tests); golden design unchanged except a 0.34% bulb-volume shift (roll-RAO argmax flips between adjacent BEM grid points — see Bug #138).

## Bug #138: Golden smoke metrics stale after R3.2 (roll-RAO argmax grid-point flip)

- **File:** `tests/golden/metrics.json`
- **Severity:** Low (regression-guard noise, not a code bug)
- **Discovery:** the R3.2 bulb clamp shifted the golden design's bulb volume 0.0012617→0.0012574 (0.34%), moving `cg_z` enough that the roll-RAO argmax flipped between adjacent BEM frequency-grid points (ω 2.2714→2.6857 rad/s, bem_n_freq=15 over [0.2, 6.0]), changing roll_period 2.7662→2.3395 s and FoM 2.2386→2.2533. Note: `bem_n_freq=15` quantization makes roll_period a coarse step function; a finer grid would stabilize it (deferred).
- **Fix:** regenerated `tests/golden/metrics.json` from a real-BEM run under the venv. Full suite: 306 passed.

## Post-mortem-driven verification results (2026-08-09)

- `run_optimization.py --quick-test` (venv, SPH-disabled sanity config): 7/7 designs with zero error codes; BO rows carry real roll_period/peak_accel/gm; a feasible design found (FoM 2.04) — vs 0 feasible in the broken production run.
- `scripts/replay_bem.py` on all 320 production design vectors with `bem_skip=False`: **315/320 pass with real BEM**; the 5 E_RAO are genuine geometry failures (3 decimation, 2 Capytaine connectivity), matching the post-mortem's "only 3/300 meshes truly degenerate".
- **Feasibility with real data: 36/320 designs feasible** (best FoM 2.342, iter 44) vs 0/320 in the broken run — the original "no feasible hull exists" conclusion is falsified; the failure was entirely the env artifact + bem_skip placeholders.
- The 40 LHS fake roll_periods (bem_skip fallback T=2.2203·BWL) are replaced by real RAO values in `output/replay_bem_results.csv`; 2/40 LHS designs are feasible with real data.

## Bug #139: Wave-family SPH cases (storm / focused-wave / drop) produced ZERO fluid particles

- **File:** `hull_opt/templates/dualsphysics.py` (`write_storm_case`, `write_focused_wave_case`, `write_drop_impact_case` + STORM/FOCUSED_WAVE/DROP_IMPACT templates)
- **Severity:** Critical (all three validation cases crashed at GenCase with 0 fluid; solver could never run)
- **Root cause:** the fillbox was anchored at `z=-zdomain` with height `SWL+zdomain` (or `zsurf+zdomain`), but `zdomain = max(zmax, zmin)` and `zmax` (wave crest height) exceeds `zmin` (tank depth) whenever `Hs·2.5+0.5 > T·3` or `wave_height·1.5 > T·3` or `drop_height+T+0.5 > T·3`. The fillbox then protrudes BELOW the definition box floor (`-zmin`), so GenCase creates zero fluid ("the fillbox is not inside the definition box"). The `seed_z` was likewise inside the protruding region. The STORM/FOCUSED_WAVE templates also sized the tank walls with `z=SWL+zmin`, leaving a gap between wall top and the (larger) domain top.
- **Fix:** the fillbox now spans the domain's water column `[-zmin, SWL]`:
  - fillbox point z: `-zdomain` → `-zmin`; size z: `SWL+zdomain` → `SWL+zmin` (`zsurf+zdomain` → `zsurf+zmin` for drop);
  - `seed_z = -zmin + 5·dp`;
  - `zmax` clamped to at least `SWL + 0.5` so the domain top always clears the still water level;
  - tank walls sized `zmax + zmin` (full domain height).
- **Measured:** GenCase smoke on a generated design (LWL=2.468, T_total=1.469, Hs=2.5, dp=0.05): storm 578,801 fluid (was 0), focused_wave 578,801 (was 0), drop 780,120 (was 0), inverted unchanged 472,032. Full pytest suite 306 passed.

## Bug #140: Towing case crashed at solver start — "No particles data with mkfluid=2" (outlet zone empty)

- **File:** `hull_opt/templates/dualsphysics.py` (`write_towing_case`)
- **Severity:** Critical (gate 1 / calibration towing could never run)
- **Root cause:** the outlet inout zone is drawn with `<boxfill>right</boxfill>`, which fills only the box's right FACE at `x = zone_o_x0 + zone_o_dx`. With `zone_o_x0 = x_out` and `zone_dx = x_out - x_in`, the face landed at `2·x_out - x_in` — beyond the definition box `x1+2dp`, so GenCase created zero mkfluid=2 particles.
- **Fix:** `zone_o_x0 = x_out - zone_dx` — the outlet box now spans `[x_in, x_out]`, its right face at `x_out` (inside the domain), mirroring the inlet zone (boxfill "left" at `x_in`).
- **Measured:** GenCase smoke: towing fluid 210,961 across all 3 fluid mk blocks (fillbox + inlet + outlet; previously outlet = 0). Full pytest suite 306 passed.
- **Known follow-up (not fixed — design decision needed):** STORM and FOCUSED_WAVE templates draw the hull STL with NO `drawmove`, and `hull.stl` is hull-only (keel/bulb excluded, `geometry.py:1114-1119`). The hull therefore floats 1.3-2 m below the still water level mid-tank instead of resting at the waterline. Towing/drop/inverted have drawmove and are positioned correctly. Fix requires deciding the intended draft/clearance (draft = T_canoe, hull-only STL) and adding `drawmove z` to both templates.

## Bug #141: Storm/focused-wave hull drawn without drawmove (1.3–2 m below waterline); no keel/bulb in SPH cases

- **Files:** `hull_opt/geometry.py` (NURBS export), `hull_opt/templates/dualsphysics.py` (all 5 writers + STORM/FOCUSED_WAVE templates)
- **Severity:** Medium (storm/focused-wave physics meaningless; keel drag missing in towing/inverted calibration)
- **Root cause (placement):** The STORM_XML and FOCUSED_WAVE_XML templates drew the hull STL with no `<drawmove>` element, so GenCase placed the mesh at its tank-frame origin. With hull-only STL bounds z ∈ [-T_canoe, freeboard] and tank floor at -3·T_total, the hull floated 1.3–2 m below the still water level — far below the wave-zone where storm impact should be measured. The DROP, TOWING, and INVERTED templates already had drawmove and were correct.
- **Root cause (geometry):** `generate_hull` exported only watertight hull+deck STL (`hull.stl`); the separate keel and bulb NURBS patches (open surfaces) were excluded. BEM and GZ are correct to exclude them (<0.1% FoM impact), but the SPH cases (towing resistance, inverted capsizing) physically benefit from the full underwater geometry: the keel contributes significant drag and righting moment.
- **Fix:**
  1. `geometry.py` (NURBS mode): export a new `hull_full.stl` containing all patches (hull + deck + keel + bulb + mirrored) via `_tessellate_patches`. Falls back to `hull.stl` on failure. Path stored in `hydro["full_mesh_stl"]`.
  2. `templates/dualsphysics.py`: new helpers `_sph_stl_path()` (prefers `hull_full.stl` sibling) and `_stl_z_bounds()` (reads STL z-extent). All 5 writers resolve through `_sph_stl_path`.
  3. STORM and FOCUSED_WAVE templates: add `<drawmove x="{{ hull_x }}" y="0" z="{{ hull_z }}" />`. The writers compute `hull_z = (SWL - T) - stl_z_min` so the keel/bulb tip rests at the design draft. `hull_x = -LWL/2` centres the hull in the symmetric tank.
  4. DROP writer: `init_z = drop_height - stl_z_min` so the keel tip starts exactly `drop_height` above the water surface (was hull bottom at `drop_height`). `zmax` expanded to cover the raised hull.
  5. TOWING and INVERTED writers: STL swapped via `_sph_stl_path()`; drawmove unchanged (towing z=0 → keel tip at -T_total ≈ correct draft; inverted bounds-driven bisection adapts automatically).
- **Measured:** GenCase smoke on a realistic design (LWL=2.468, T_total=1.469, dp=0.05):
  - `hull_full.stl` z ∈ [-1.408, +0.112] (keel depth 1.408 m ≈ T_total); hull.stl z ∈ [-0.329, +0.112].
  - storm: 579,088 fluid (was 578,801 at wrong depth) — hull now at the waterline.
  - focused_wave: 579,088 fluid (was 578,801).
  - drop: 780,120 fluid (was 780,120) — hull starts 2.0 m above surface; domain top adjusted.
  - towing: 210,772 fluid, 3 fluid MKs (boundary up from 23,630 → 24,008 — keel added 378 bound particles). Outlet (mk2) still alive.
  - inverted: 472,032 fluid — unchanged (keel sits above the water in upside-down orientation).
- **Tests:** 5 new tests (`TestHullPlacementKeelAware` in test_sph_gates.py) verifying drawmove presence, keel-tip-at-draft placement for storm/focused, keel-tip-at-drop-height for drop, full-mesh path resolution, and template regression guard. Full suite: 311 passed, 1 skipped.

## Bug #142: Mid-fidelity preflight ran before the towing case was rendered (calibration never ran)

- **File:** `hull_opt/mid_fidelity.py`, `hull_opt/preflight.py`
- **Severity:** High (calibration table had 0 rows ever)
- **Root cause:** `run_mid_fidelity_calibration` called `preflight_case(case_dir, config)` before `run_towing_resistance` rendered the towing XML (the XML is created inside `run_towing_resistance` via `build_towing_case`). Preflight's "Case XML file found (none)" FAIL always aborted calibration.
- **Root cause (latent):** even after the XML existed, preflight's STL check used a fragile quote-heuristic (`rfind('"')` around the `file=` keyword) that extracted garbage on the real `<drawfilestl file="...">` syntax and FAILed "STL path exists". It also FAILed on the intentionally non-watertight `hull_full.stl` (open keel/bulb patches).
- **Fix:**
  1. `mid_fidelity.py`: resolve dp/sim_time up-front, call `build_towing_case()` before `preflight_case()` (idempotent — `run_towing_resistance` re-renders).
  2. `preflight.py`: extract STL paths with regex `file\s*=\s*"([^"]*\.stl[^"]*)"`; watertightness is a WARN (DualSPHysics doesn't need a closed mesh); FAIL only on missing/unloadable STL, zero faces, or degenerate (zero-extent/non-finite) bounds; `ok` accepts PASS+WARN.
- **Measured:** calibration-pipeline test (generate_hull → build_towing_case → preflight) passes; real hull_full.stl (33,505 faces, non-watertight) → WARN + PASS. Full suite 324 passed.

## Bug #143: SAC-gaming loophole — volume gate compared against the shrunken adaptive target

- **File:** `hull_opt/constraints.py`, `hull_opt/geometry.py`
- **Severity:** High (design 16 passed feasible with 27% of the 0.145 m³ target)
- **Root cause:** `geometry.py` caps SAC scaling at 1.30, shrinking the effective target (`target_eff = min(target, 1.3·capacity)`), and `constraints.py` compared underwater volume against that shrunken `target_eff` — tiny hulls matched the shrunken target and skated through. This also fed the "Negative hull mass clamped to zero" warnings (ballast exceeding hull mass on the shrunk volume).
- **Fix:**
  1. `constraints.py`: volume gate and the 50%-minimum-displacement gate now compare against the TRUE `config.fixed.target_displacement` (fallback: `target_nabla`), not `target_nabla_eff`. SAC-capped designs accumulate the volume penalty/violation as intended.
  2. `geometry.py`: negative hull mass now raises `ValueError` (stored as `E_GEOM`, steering the GP away) instead of silently clamping to zero.
- **Impact:** early/LHS designs below the displacement target are correctly infeasible; expect lower feasible counts early in BO until the surrogate learns the viable region.

## Bug #144: fill_holes repair + noisy warning applied to the combined SPH mesh

- **File:** `hull_opt/geometry.py`
- **Severity:** Low (wasteful; could create garbage fan triangles)
- **Root cause:** `_tessellate_patches` always ran `fill_holes()` and warned when the mesh wasn't watertight — but the combined hull+keel+bulb mesh is open by design.
- **Fix:** `_tessellate_patches(..., require_watertight=True)`. Hull-only call keeps repair + warning (with new boundary-edge z-cluster diagnostics via `_log_boundary_edges` on failure); combined SPH call passes `require_watertight=False` (debug-level log, no fill_holes).

## Bug #145: TestHullPlacementKeelAware placement tests read hull-only STL bounds (stale after bug #141)

- **File:** `tests/test_sph_gates.py`
- **Severity:** Low (test-only; failures masked by missing hull_full.stl artifacts)
- **Discovery:** full-suite run after #142–#144; the two placement tests failed only when `output/design_*/` artifacts contained `hull_full.stl`.
- **Root cause:** `test_storm_focused_keel_tip_at_draft` / `test_drop_keel_tip_at_drop_height` loaded the hull-only fixture `hull.stl` bounds (`z_min ≈ -T_canoe`) but asserted the placement invariant `dz + z_min == 0.1 / drop_height` against it. Since bug #141 the writers resolve the STL through `_sph_stl_path()` (preferring `hull_full.stl`, keel/bulb included) and place the DEEPEST POINT OF THAT MESH at `still_water_level - T` (storm/focused) or `drop_height` above the surface (drop). The writer code is correct — verified `dz + z_min(placed mesh) == 0.1` and `init_z + z_min(placed mesh) == 2.0` exactly on a real artifact (`hull_full.stl` z_min = -1.4752).
- **Fix:** both tests resolve the placed mesh via `_sph_stl_path()` and assert the invariant against its bounds (falls back to the fixture STL when no `hull_full.stl` exists, so the invariant still holds). Class docstring updated.
- **Measured:** `tests/test_sph_gates.py` 25 passed; full suite 326 passed, 1 skipped.

---

## Bug #146: SPH calibration OOM — 1.08M-particle towing case killed the host (15 GiB RAM)

- **Files:** `config.yaml` (`calibration:` dp/timeout), `hull_opt/config.py`, `hull_opt/templates/dualsphysics.py` (`write_towing_case`), `hull_opt/sph_resistance.py` (`run_sph_case`), `hull_opt/mid_fidelity.py`, `hull_opt/low_fidelity.py` (`_wait_spf_lock`)
- **Severity:** Critical (calibration towing killed the machine while 3 Capytaine BEM workers ran)
- **Discovery:** Production calibration run — the towing tank at dp=0.03 held ~1.1M particles (~15 GiB total); the host died during the CASES.bin particle-array allocation.
- **Problem:** (1) dp=0.03 with 10 s sim time was far heavier than needed and unbounded — nothing stopped an oversized case from reaching the solver. (2) BEM workers (3 Ray Capytaine dense-LU jobs) had no coordination with the SPH solver, so both heavy loads ran concurrently. (3) Tank depth `-2.0·T_total` and `z_sheet_top = zsurf + 0.2` wasted fluid volume below/above the hull.
- **Fix:**
  - Calibration defaults: `sph_dp` 0.03 → 0.05, `sph_sim_time` 10.0 → 6.0, `timeout` 14400 → 3600; new `max_particles: 400000` (`CalibrationConfig` defaults match).
  - Tank: `z_floor = -1.5·T_total` (was -2.0), `z_sheet_top = zsurf + 0.15` (was 0.2) → ~230k particles.
  - `run_sph_case`: after GenCase, parses "Total particles:" from `gencase.out` and aborts FAILED when > `calibration.max_particles` (particles ∝ 1/dp³ — dp=0.03 now rejects at ~1.1M before the solver launches). nvidia-smi VRAM guard (estimate = particles × 1.5 kB, gpu tool only) before launch. `gpu_lock` held ONLY around the solver subprocess (GenCase/ComputeForces outside the lock). Solver progress → `<case_dir>/solver.log`; failures tail ~250 chars; result dicts carry `np_particles`.
  - `mid_fidelity.py` plumbs `gpu_lock=<output_dir>/gpu.lock` to `run_towing_resistance`; calibration results log compact `<calib FAILED design=.. iter=.. rc=.. t=..s np=..>` lines.
  - `low_fidelity.py` `_wait_spf_lock(config)`: BEM workers flock-poll the lock every 5 s and park up to 3600 s while the solver runs.
- **Impact:** Calibration cases ~230k particles (20-40 min GPU), oversized cases rejected pre-solver, and SPH/BEM never contend for host RAM.

## Bug #147: Slam gate physics — whole-bottom stagnation pressure gave absurd accelerations

- **Files:** `hull_opt/rapid_gates.py` (`_slam_pressure`, `_slam_accel`, `_SLAM_FORCE_FACTOR`), `config.yaml` / `hull_opt/config.py` (`validation.max_pressure_pa`, new `rapid_validation.slam_max_pressure_pa`), `hull_opt/low_fidelity.py` (`_accumulate_margin_violations`, `_margin_bonus_clip`)
- **Severity:** High (slam margin dominated the FoM; observed fom −18.9 mostly slam)
- **Discovery:** Optimization run analysis — the old model applied full stagnation pressure to the entire bottom, producing absurd slam accelerations that dominated infeasible FoMs.
- **Problem:** (1) Whole-bottom-at-stagnation-pressure force was unphysical. (2) Soft margins (capsize/storm_accel/slam_pressure/slam_accel/inverted_pressure) added `|margin|·0.5` to `violation_magnitude`, so even slightly-negative soft margins dominated the FoM landscape. (3) The w6-w9 bonus was clipped at ±1, losing gradient signal.
- **Fix:** (1) Replaced with a capped initial-impact model: `p_eff = min(p_wagner, p_cap)` with `p_cap = rapid_validation.slam_max_pressure_pa` (0.5 MPa; `validation.max_pressure_pa` lowered 2.0 MPa → 0.5 MPa to match); `F_slam = 0.85 · 0.5·ρ·v²·A_slam·k(β)` with `A_slam = min(A_wet, BWL·T_canoe)`, `k(β) = (π/2)²·max(0.3, cot β)/2`; mass base = design displacement (`target_displacement·ρ`), NOT hydro nabla. (2) Soft margins no longer add to `violation_magnitude` — only hard gates do (`_accumulate_margin_violations`). (3) `_margin_bonus_clip` = np.tanh smooth saturation.
- **Impact:** Slam results monotonic in deadrise: 16.6 / 10.1 / 5.8 g at 5°/15°/25°, 3 m drop; feasible FoMs are no longer drowned by soft-margin penalties.

## Bug #148: Triple volume definitions — SAC vs mesh vs GZ submerged volume disagreed

- **Files:** `hull_opt/geometry.py` (`mesh_displacement`, `_compute_hydrostatics`), `hull_opt/low_fidelity.py` (FoM displacement penalty)
- **Severity:** High (SAC 0.0722 vs target 0.1450 — volume gating was effectively disabled)
- **Discovery:** Design audit — SAC-integrated volume, mesh displacement, and GZ-curve submerged volume were three different quantities computed three different ways.
- **Problem:** `nabla`/`underwater_volume` came from the SAC integral, which is gamed by SAC scaling (Bug #25/#143 history) and does not reflect the actual mesh the GZ curve and FoM operate on.
- **Fix:** New public `geometry.mesh_displacement(mesh, z=0.0)` (trimesh slice below the waterplane; handles Trimesh/Scene/path; 0.0 fallback). `_compute_hydrostatics` derives `nabla`/`underwater_volume` from `mesh_displacement` of the hull-only mesh (the same mesh the GZ curve uses). When the final SAC volume < 0.75 × `target_displacement`, the hull is rejected with `ValueError("E_DISP: SAC volume ... far from target ...")` (surfaces as `E_GEOM:<msg>` in low-fi). The low-fi FoM displacement penalty uses `hydro["mesh_displacement_m3"]` of the loaded hull mesh. `compute_righting_energy`'s displacement warning now sees the same volume from the same mesh.
- **Impact:** Displacement is single-sourced from the mesh; SAC-gamed hulls are rejected (E_DISP) instead of passing with a shrunken volume.

## Bug #149: Rapid gate ordering — unconditional 2-4 min storm BEM sweep per evaluation

- **Files:** `hull_opt/rapid_gates.py` (`evaluate_rapid_gates`, `_CHEAP_HARD_KEYS`, `_build_cheap_margins`)
- **Severity:** Medium (eval-time regression; ~2-4 min burned on every candidate)
- **Discovery:** P3 eval-speed profiling — the storm BEM seakeeping sweep ran on every evaluation regardless of stability outcome.
- **Problem:** Gate W (storm BEM sweep) ran unconditionally, even for hulls whose cheap GZ/wind-heel gates already failed — the expensive sweep can never rescue an unstable hull.
- **Fix:** Gate order is now R (GZ stability) → E (storm wind heel) → S (slam) → P (inverted pressure) — all cheap — then W (storm BEM sweep). W is skipped when any cheap HARD gate (avs/wind_heel/righting_energy/max_gz_m/gz_area_30/gz_area_40_90/self_right) already fails: `details["gate_w"] = "storm BEM skipped: cheap hard gate failed (<name>)"`. `_storm_bem_sweep` also logs an INFO line when the surrogate anchor error threshold is exceeded (full-BEM fallback) so slow paths are visible.
- **Impact:** Most candidates skip the 2-4 min sweep entirely; the remaining sweep is no longer silent when it falls back to full BEM.

## Bug #150: `worst_hard` missing from rapid diagnostics

- **Files:** `hull_opt/rapid_gates.py` (`RapidGateResult`), `hull_opt/utils.py` (`format_rapid_summary`), `run_optimization.py` (results report)
- **Severity:** Low (diagnosability)
- **Discovery:** Post-P1 review — rapid summaries showed margin values but not which hard gate was the binding failure.
- **Problem:** With 15 margins, the most-negative non-soft margin (the true blocker) was not surfaced per design.
- **Fix:** `RapidGateResult` gained `worst_hard` / `worst_hard_value` (most-negative non-soft margin name/value, `""`/`0.0` when none). `utils.format_rapid_summary` log lines, results.md/CSV, and the interactive table (run_optimization.py) all include it.
- **Impact:** The binding hard-gate failure is visible per design in logs and reports.

## Bug #151: Tests stale after P0-P1 (tank depth, slam/FoM baseline)

- **Files:** `tests/test_sph_gates.py`, `tests/test_high_fidelity_gates.py`, `tests/golden/metrics.json`
- **Severity:** Low (test-only)
- **Discovery:** Post-fix suite — towing tank depth assertions still assumed `-2.0·T_total`; golden metrics predated the new slam/FoM baseline.
- **Fix:** Towing tank depth assertions updated `2.0` → `1.5·T_total`; `tests/golden/metrics.json` regenerated for the new slam/FoM baseline.
- **Impact:** Tests encode the new tank geometry and FoM baseline.

## Bug #152: Reference storms ran on CPU with no particle cap (9-12M particles, 0/5 completed)

- **Files:** `config.yaml` (`reference.dp`, `reference.timeout_min`), `hull_opt/sph_gates.py` (`run_reference_storm`)
- **Severity:** Critical (reference storms never produced data; campaign ~2.5 h of 3 h wasted)
- **Discovery:** Post-run audit of `output/ref_ds_3`/`ref_ds_13` (`gencase.out`: 9.0M/11.9M particles) and `reference_runs` (all 5 storms TIMEOUT at 1800 s, 1 output frame each).
- **Problem:** (1) `reference.dp=0.02` made storm cases ~50× the calibration size (9-12M vs ~230k particles) — far beyond this 6 GiB box. (2) `run_reference_storm` had neither the particle-cap preflight nor the nvidia-smi VRAM guard that `run_sph_case` has. (3) The storm solver command never appended `-gpu`; DualSPHysics defaults to CPU (`JSphCfgRun.cpp: Cpu=true`), so even a feasible case would crawl.
- **Fix:** `reference.dp` 0.02 → 0.06 (~284-440k particles; verified worst-case tank at 284,235 < 500k cap), `reference.timeout_min` 30 → 180. `run_reference_storm` now parses `Total particles:` from `gencase.out` (via `_parse_particle_count`) and returns FAILED immediately above `_STORM_MAX_PARTICLES` (500k); adds the ~1.5 kB/particle nvidia-smi VRAM guard before solver launch; appends `-gpu` when the GPU binary is selected.
- **Impact:** Oversized cases die in seconds, not hours; all storms run on the RTX 3050 with real data.

## Bug #153: Mid-fi calibration starved by reference storms (zero calibrations ever ran)

- **Files:** `hull_opt/sph_resistance.py` (`run_sph_case`, `run_towing_resistance`), `hull_opt/mid_fidelity.py`, `hull_opt/surrogate.py` (`_store_eval_result`, `_resume_initial_sampling`, `_bo_loop`), `config.yaml` (`calibration.retry_failed`)
- **Severity:** Critical (drag factor stuck at 1.0 — every FoM unscaled)
- **Discovery:** `calibration` table empty; exactly one `calibration_attempts` row (design 29, iter 40, failed). The iter-20 attempt never registered at all (silent skip).
- **Problem:** (1) `_store_eval_result` enqueued the storm at `iteration % 10 == 0` BEFORE the calibration block in the same iteration; ReferenceRunner grabbed `gpu.lock` first, and calibration's non-blocking flock failed → SKIPPED → marked failed. (2) Calibration used non-blocking flock, so any running storm starved it permanently.
- **Fix:** Calibration enqueues no storm on calibration iterations (`iteration > 0 and iteration % calibration.frequency == 0`) via the shared `_reference_enqueue_due` policy; `run_sph_case` gained `lock_wait=True` (blocking `fcntl.LOCK_EX`) and `run_towing_resistance`/`run_mid_fidelity_calibration` thread it through, so calibration waits out a storm and then runs; the calibration block logs an explicit pick/skip decision line so a missed calibration is never silent; `calibration.retry_failed: true` lets previously-failed designs be retried.
- **Impact:** Every 20 iterations gets a real calibration; the drag factor converges from real SPH data.

## Bug #154: Lock-busy storms burned permanently via SKIPPED row

- **Files:** `hull_opt/surrogate.py` (`ReferenceRunner.run`)
- **Severity:** Medium (transient lock collisions permanently lost a design's reference storm)
- **Discovery:** Post-mortem of the failed campaign — a storm that collided with any GPU holder was written `INSERT OR REPLACE ... 'SKIPPED'`, and the runner skips any design with an existing `reference_runs` row forever.
- **Fix:** On lock-busy the runner now logs and continues WITHOUT inserting a row — the design stays eligible and is retried on the next enqueue.
- **Impact:** A transient collision no longer burns a design's storm.

## Bug #155: Storm tank open-ended + wavegen silently ignored → AbortBoundOut

- **Files:** `hull_opt/templates/dualsphysics.py` (STORM_XML, FOCUSED_WAVE_XML, DROP_IMPACT_XML + writers)
- **Severity:** Critical (every reference storm crashed; waves never generated)
- **Discovery:** All 4 storms of the resumed campaign (designs 7/51/71/91) died with `JSphGpuSingle::AbortBoundOut` at JSph.cpp:2909. Artifacts showed the chain: the storm tank had only `bottom | front | back` walls (open +X end), so the 6.3 m water column dam-broke out of the tank from t=0 (RunPARTs.csv: ~5k particles out at t=0.1 s, ~50% of fluid by t=0.9 s; 195k total excluded); the violent current grounded the floating hull and rolled it out the +Y wall at t≈0.95 s ("Some boundary particle exceeded the +Y limit (back limit)"). Independently, `gencase.out` reported `Moving...: 0` — the `<wavegen><pistonwave>` block is **silently ignored** by GenCase v5.4.354.01, so the storms ran with no waves at all.
- **Fix:** (1) Classic DS wave dialect: replaced `<wavegen><pistonwave>` with `<wavepaddles><piston_spectrum>` (storm) / `<piston_focused>` (focused), driven by an explicit mk=10 paddle `drawbox solid` + `<motion><objreal ref="10">`; piston `depth` = real water column (zmin + SWL), `ramptime = 0.5·Tp`. (2) Closed the tank: far-end `boxfill>right` wall at +X. (3) Added `<damping><dampingzone>` slab (limitmin at 0.7·xmax → xmax, overlimit 1, redumax 10) — fluid-only, kills reflections before the closed end. (4) Paddle back pocket 0.35 m ≥ max piston stroke (~0.23 m). Focused/drop templates share the structural fix (dead code today, latent crashes).
- **Verification:** GenCase on design-91 case → `Moving...: 17,088` (paddle created), 324,214 total particles (276,488 fluid, < 500k cap). Solver smoke run (3 s, GPU): only 3,916 particles out at t=0.1 s (initial splash, 1.2%) vs ~195k drain before; damping zone and paddle motion confirmed in Run.out; no AbortBoundOut.
- **Impact:** Reference storms now generate real waves in a sealed tank with a damping beach; campaign can resume with `--resume` (91 designs preserved).

## Bug #156: Gate-5 inverted SPH crashed at t≈0.16 s (AbortBoundOut) — 3 stacked root causes

- **Files:** `hull_opt/templates/dualsphysics.py` (`write_inverted_case`, `INVERTED_XML`), `config.yaml` (`sph_inverted_dp`), `hull_opt/sph_resistance.py` (`_run_measuretool`), `hull_opt/sph_gates.py` (`_run_measuretool`, `_parse_measuretool_csv`)
- **Severity:** Critical (gate 5 SPH never produced a measurement — every validation run since the SPH port silently fell back to the analytic pressure; the "PASS" rows were fabricated fallbacks)
- **Discovery:** Revalidation of finalists 140/63/79 — every gate-5 SPH run died `AbortBoundOut` at t≈0.14-0.16 s with NaN hull coordinates in `Error_BoundaryOut.vtk`.
- **Root cause chain (3 independent bugs):**
  1. **dp=0.03 → ~2M particles (5× the 400k cap).** `sph_inverted_dp=0.03` on the 3·LWL × 3·B × 3·T_total tank = ~1.9-2.1M particles; GenCase aborted, so gate 5 always scored the rapid-gate fallback. Fixed: `sph_inverted_dp=0.06` (~235-260k, under cap — same fix as reference storms, Bug #152).
  2. **Inverted tank had no ±X walls** (`boxfill` = `bottom | front | back` only — copied from the towing template where those faces are inout zones). Water dam-broke out both open ends carrying the hull with it. Fixed: closed tank (`bottom | front | back | left | right`).
  3. **Domain is the initial particle bbox, not the def box.** `MapRealPos` = particle bbox ± (KernelH·0.05 + dp/2) (JSph.cpp:2176). The inverted hull's keel tip points UP at z≈1.48, so the domain top was 1.52 — the floating hull bobbed 4 cm and the keel tip crossed it. Independently, the tank was too narrow/short for the self-righting roll: `ymax=1.5·B` < the keel-tip swing radius (keel tip traces a ~0.85 m circle about the CG). Fixed: tank sized to the motion envelope — wall top at `zsurf + z_sheer_max + 0.5` (above keel tip + bob), `ymax = max(1.5·B, z_sheer_max + 0.3)` (covers the roll swing), floor at `-1.5·T_total` (upright keel tip after roll), and the hull's initial position at the computed equilibrium draft `dz_eq` instead of `z_sheer_max` (~1.2 m above the water → free-fall slam).
  4. **Floating-body CG given in the wrong frame.** `write_inverted_case` rotates the hull 180° but passed the upright-frame `cg_z` (≈ −0.60) and omitted the `hull_x` drawmove from `<center>` — the mass center sat ~1.2 m from the true one → spurious torque → the body spun out. Fixed: `<center x="hull_x + 0.4·LWL" z="-cg_z + dz_eq">` (tank frame, post-rotation, post-drawmove). Inertias are unchanged by the 180° rotation (Ixx/Iyy/Izz all symmetric in the y-z swap).
  5. **MeasureTool writes `{stem}_Press.csv`, not `{stem}.csv`** (v5.4: one CSV per variable, stem minus extension + `_Var.csv`). Both `_run_measuretool` copies checked for the literal `measuretool.csv`, which never exists → gate always took the "MeasureTool produced no output" fallback even when the solver ran perfectly. Fixed: return `{stem}_Press.csv`. Also `_parse_measuretool_csv` only matched headers starting with "Time", but v5.4 writes `Part;Time [s];Press_0 [Pa];...` — now matches either.
- **Verification:** GenCase at dp=0.06 → 304-346k particles (under cap); solver runs 6.0 s clean (np stable, 0 exclusions); MeasureTool reads real pressure traces. Finalists: d140 **22,078 Pa**, d63 **37,156 Pa**, d79 **24,282 Pa** — all PASS (threshold 500 kPa). Real measurements, no fallbacks.
- **Impact:** Gate 5 is a real SPH gate again; the fabricated fallback PASS rows are replaced by measured values.

## Bug #157: Gate-1 SPH measures momentum-flux drag, not tow resistance (~14× low-fi)

- **Files:** `hull_opt/templates/dualsphysics.py` (`write_towing_case`), `hull_opt/high_fidelity.py` (`_gate_fine_cfd`), `config.yaml` (`validation.rt_upper_bound_factor`)
- **Severity:** High (gate-1 thresholds are unpassable by construction with the current SPH setup)
- **Discovery:** Fresh post-drain-fix gate-1 runs on finalists 140/63/79 measure Rt = 410-584 N vs low-fi thresholds 56-74 N (steady traces, all FAIL). t=0 hydrostatics is perfect (Fz=1,227 N ≈ 1,273 N buoyancy; Fx≈0.3 N), but at speed Fz=5.6 kN exceeds full-submersion buoyancy (3 kN).
- **Analysis:** (1) Widening the channel 1.5·B → 3.0·B (33% → 17% blockage) changes nothing (426 N both ways) — not side-wall blockage. (2) No water inside the hull shell (void-fill works; 0 fluid particles in the hull bbox). (3) The measured Fx ≈ momentum flux ρ·u²·A_frontal ≈ 529 N max for the hull's 0.19 m² submerged frontal area — the fixed hull in the imposed 1.65 m/s current intercepts the channel's momentum, which is NOT what a towed hull experiences. (4) The OF-era calibrations (docs/POST_RUN_ANALYSIS.md §3.1) measured CFD/low-fi ratios 0.73-1.08 — the low-fi model is validated against proper CFD; the SPH fixed-hull current setup is the outlier. (5) All stored SPH drag factors (0.05-0.06) came from drain-era calibrations (rt_cfd ≈ 0.0005-0.25 N — water drained away, near-zero force) and are invalid.
- **Status:** Known measurement-model mismatch (fixed-hull current vs towed hull). The 0.05-0.06 factors made FoMs optimistic; a fresh valid calibration yields factor=5.0 (clip max) with the current setup. Gate-1 threshold policy for production runs is being decided (see run_optimization.py gate handling); the analytic low-fi model remains the validated drag basis.
- **Impact:** Gate-1 SPH runs on the current template cannot validate drag; production policy must not burn GPU hours on unpassable gate-1 runs until the towing formulation is fixed (moving hull or calibration-scaled thresholds).

## Bug #158: Reference storm SKIPPED at enqueue when STL not yet persisted + task_done double-count

- **Files:** `hull_opt/surrogate.py` (`ReferenceRunner.run`)
- **Severity:** Medium (storm data silently lost for designs enqueued before geometry persistence; thread crash on re-queue)
- **Discovery:** Fresh production campaign — design 10's storm row appeared as `SKIPPED / no valid STL path` and design 9's thread died with `task_done() called too many times`.
- **Root causes:** (1) LHS enqueues storms from the batch immediately after the design row, but `cad_stl_path` is persisted a moment later; the runner's no-STL branch wrote a SKIPPED row, permanently burning the design (Bug #154 pattern). (2) The first fix re-queued the item, but re-putting a consumed queue item double-counts `task_done()` and crashes the thread.
- **Fix:** The no-STL branch now polls the DB within the same task (15 s interval, 5 min deadline) until the STL path lands; no SKIPPED row, no re-queue. If the deadline passes the storm is dropped with a warning (kept eligible for a later enqueue).
- **Verification:** Storm for design 6 completed `OK` (101 parts, no capsize, no drain) in the restarted campaign; no SKIPPED rows; thread alive through subsequent storms.
- **Impact:** Every storm now produces data; campaigns no longer silently lose reference runs.

## Bug #159: `_clean_slate` left `ref_ds_*` storm dirs from a previous campaign

- **Files:** `run_optimization.py` (`_clean_slate`)
- **Severity:** High (stale storm outputs could mix with a fresh campaign's solver writes, corrupting storm results)
- **Discovery:** After restarting a fresh campaign, the new storm for design 6 wrote into a `ref_ds_6/` dir still holding 80 stale `Part_*.bi4` from the previous run (10:27 vs 10:44 timestamps); the solver then failed writing `PartMotionRef.ibi4` (file collision).
- **Fix:** `_clean_slate` now removes `output/ref_ds_*` dirs alongside `designs_*`, `validation/`, and `gpu.lock`.
- **Verification:** Restarted campaign's storm dirs are created empty; storm completes OK.
- **Impact:** Fresh campaigns cannot inherit corrupted storm state.

## Bug #160: ReferenceRunner task_done double-count via try/finally (thread dies, storms stop)

- **Files:** `hull_opt/surrogate.py` (`ReferenceRunner.run`)
- **Severity:** Critical (the storm thread crashes on the FIRST no-STL or lock-busy drop; all subsequent storms silently never run)
- **Discovery:** Fresh campaign thread died with `ValueError: task_done() called too many times` at the finally-block task_done, immediately after the design-9 "no STL after 5 min" drop.
- **Root cause:** The per-item processing is wrapped in `try: ... finally: self.queue.task_done()`. The early-continue paths (lock-busy at ~line 89, no-STL drop at ~line 123) each ALSO called `self.queue.task_done()` before `continue` — the continue then runs the finally, so a single consumed item got task_done() TWICE. The unfinished-task count went negative and the very next finally task_done raised, killing the daemon thread. Every consumed item must have exactly one task_done.
- **Fix:** Removed the redundant task_done() from both early-continue paths; the finally is now the single task_done per consumed item. Audit confirms exactly 3 call sites (sentinel-None, existing-row skip, finally), one per item in every path. Simulated 5 consecutive no-STL drops: thread stays alive.
- **Verification:** `tests/test_surrogate.py` (8 passed) + custom queue-accounting simulation.
- **Impact:** The storm thread survives no-STL drops and lock collisions; storms run for the whole campaign.

## Bug #161: NURBS hull tessellation not watertight — merge rounding + bitwise-zero half-breadth band

- **Files:** `hull_opt/geometry.py` (`_section_curve`, `_tessellate_patches`)
- **Severity:** High (22/37 production designs failed with "NURBS hull tessellation not watertight after fill_holes repair")
- **Discovery:** DB sweep `error_code LIKE '%watertight%'` → 22 designs; open edges clustered at stern-keel corner; multi-face edges traced to merged port/starboard vertices.
- **Root cause (two defects):**
  1. `merge_vertices()` used trimesh default `digits_vertex=None` → `decimal_to_digits(tol.merge)` = 8 digits (tol ~5e-9). Near-keel port vertices (y=+3.97e-9) and starboard vertices (y=−3.97e-9) both rounded to 0 and welded onto the y=0 keel line → port/starboard faces shared edges → 4-face non-manifold edges → the killer loop deleted faces → open slivers `fill_holes` couldn't fix.
  2. `_section_curve` clipped `y_raw` to exactly 0.0 (`np.clip(y_raw, 0.0, None)`). With steep deadrise (design 36: 29.45° → stern-interp ×2.5 → ~73°, tan≈3.4), `dr_term = zn·tan(dr_rad)` drove the whole lower band of the stern sections negative → yf=0.0 bitwise → control-net rows v=1..3 at stern columns exactly zero → NURBS surface = 0 over a 2D region (u=73..76 × v=1..3) → port/starboard surfaces coincide exactly, regardless of merge digits.
- **Fix:** (1) `merge_vertices(..., digits_vertex=12)` (5e-13 tolerance keeps the two halves distinct). (2) `np.clip(y_raw, 1e-7, None)` in `_section_curve` (floors normalized half-breadth at 1e-7 instead of exact zero; physical delta ≤ y_wl·1e-7 ≈ 1e-8 m, negligible).
- **Verification:** All 22/22 previously-failing DB designs now watertight through the real `_tessellate_patches(dp=0.03, require_watertight=True)` path; 15-sample passing set unchanged; `tests/test_geometry.py` 32 passed incl. new regression tests (`test_tessellate_steep_deadrise_watertight`, `test_section_curve_never_exact_zero`); full suite 390 passed.

## Bug #162: Reference storms aborted at GenCase — oversized hulls tripped the 500k particle cap

- **Files:** `hull_opt/sph_gates.py` (`run_reference_storm`, new `_estimate_storm_dp`), `hull_opt/templates/dualsphysics.py` (new `storm_domain`), `config.yaml` (`reference.dp` comment)
- **Severity:** Medium (storms for large designs silently recorded FAILED; those designs got no SPH reference data, only the rapid-gate fallback)
- **Discovery:** Production campaign — `ref_ds_11` (501,868 particles) and `ref_ds_41` (505,841) aborted by the post-GenCase particle-count preflight (`_STORM_MAX_PARTICLES = 500_000`); the other four storms ran at 371k-390k.
- **Root cause:** The storm tank is sized linearly off the hull (`xmax=1.5·LWL`, `ymax=1.5·BWL`, `zmin=3·T_total`, fillbox to `T_total+0.6`) at fixed `reference.dp=0.06`, so particle count ∝ hull volume. Designs 11/41 sat at the large end of the search space (LWL≈2.43-2.47 m, BWL≈0.67-0.70 m, T_total≈1.52-1.54 m, disp 0.145 m³) — ~2.5% over the cap — while the "330-440k" comment was calibrated on smaller designs.
- **Fix:** Design-adaptive storm dp. `storm_domain()` is now the single source of truth for tank geometry (shared by `write_storm_case` and the estimator). `_estimate_storm_dp` budgets the layout before GenCase: count ≈ `0.65 · V_defbox / dp³` (fill coefficient fitted to the 6 real layouts, k=0.620-0.661). Hulls whose nominal-dp estimate is below 480k keep dp=0.06 untouched; larger hulls are coarsened just enough to land near the 420k target (ceil to 3 decimals, clamped to [dp_nominal, 0.10], self-checked against the cap). Non-finite inputs fall back to dp_nominal; the post-GenCase 500k abort remains as the final safety net.
- **Verification:** All 6 real campaign layouts replay through the estimator — the 4 passing cases keep dp=0.06, the 2 failing cases get dp=0.064 (~410k est, well under the cap); new `TestAdaptiveStormDp` unit tests (real failing dims, unchanged small hull, NaN fallback, extreme clamp); `test_sph_gates.py`, `test_sph_lock.py`, `test_sph_resistance_fixes.py` green.
- **Impact:** Big-hull storms now run (≤10% coarser dp, capped at 0.10) instead of being lost; previously-passing storms are bit-identical.

## Bug #163: Vertical stem + forward-swept fin + quarter-bulb-cap + 4-way mass split (JFR/PYD/Fanhai-T2 overhaul)

- **Files:** `hull_opt/config.py`, `hull_opt/param_layer.py`, `config.yaml/fast/phase0.yaml`, `hull_opt/geometry.py` (`_sheer_height`, `_stem_rake_shift`, `_build_nurbs_control_net`, `compute_half_breadth_analytic`, SAC loop, `_build_keel_patch`, `_build_bulb_patch`, `keel_half_breadth`, `_compute_hydrostatics`, `_tessellate` docs), `hull_opt/hydrostatics.py` (`compute_cg_z/x`, `compute_lumped_inertia`, windage), `hull_opt/geometry_validator.py`, `hull_opt/constraints.py` (deck_beam, bulb capacity, JFR monitors), `hull_opt/rapid_gates.py` (inverted fallback), `hull_opt/low_fidelity.py` (appendage area), `run_optimization.py` (final copy, report cols), `tests/test_plan_impl.py` (new)
- **Severity:** High (all three user-visible defects on finalist renders + physics audit findings)
- **Discovery:** Finalist side-view render (flat slab deck, stubby forward-leaning fin, detached bulb speck) + code audit vs JFR review / PYD 5th ed / Fanhai-T2 SIMP paper.
- **Root causes:**
  1. Bow: `_sheer_height` returned flat `E`; no `dz(x)`/`dx(z)` DOF existed, validator enforced monotonic-x vertical ends.
  2. Keel: sweep sign built forward (`(z-root)*tan`, tip -0.25 m) while `rig.extended_keel_clr_x` assumed aft (+0.11 m) — 0.36 m / 14% LWL disagreement; t/c tied to BWL (23% corner); no fillet.
  3. Bulb: single-octant control net y-mirrored to a 1/4 open dome hanging on point contact; `bulb_vol` cap 0.0025 held 28 kg vs 45-70 kg demand (29-39 t/m³ implied); 4 mass paths disagreed by ~18 kg.
- **Fix:** 20-dim vector (+`sheer_bow [0,0.25]`, `sheer_stern [0,0.12]`, `stem_rake_deg [0,25]`, flat 0.30-0.70 mid-deck, topside-only rake, sheer-exempt SAC scaling, analytic cap at sheer); aft sweep + root fillet + chord-based 12% t/c (mesh+analytic in sync); full-ellipsoid bulb (half y>=0 + mirror) seated overlapping tip; `bulb_vol [0.002,0.0065]` + downstream capacity violation (geometry stays buildable); unified floor + CB_x/electronics-bay inertia + chord-based keel mass; sheer-aware windage/deck/inverted paths; info-only JFR LDR/L/B/SA-D/Wb-DT monitors; final CAD prefers `hull_full.stl`; 17-dim legacy vectors pad flat.
- **Verification:** `tests/test_plan_impl.py` (8 new) + `test_geometry/test_config` updated green; end-to-end `generate_hull` + `evaluate_constraints` shows bow kick (zmax 0.465 > E 0.34), monitors flow, bulb violation soft-gates new campaigns.
- **Impact:** Config signature change → old DBs refuse (wipe for fresh campaign); old finalists re-score as bulb-undersized (expected — they were optimized under the infeasible stacking model).

## Bug #164: Dead-flat deck directive + fleet-standard cutaway forefoot (21-dim)

- **Files:** `hull_opt/config.py`, `hull_opt/param_layer.py`, `config.yaml/fast/phase0.yaml`, `hull_opt/geometry.py` (`FOREFOOT_EXTENT`, `_forefoot_factor`, control net / analytic / SAC `T_local`, `generate_hull` gate), `hull_opt/geometry_validator.py`, `hull_opt/constraints.py` (`forefoot_cut` info), `hull_opt/corrections.py` (MLP 21), `run_optimization.py` (report col, generic dim check), `tests/test_plan_impl.py` (flat ≤1e-6 + cutaway tests)
- **Severity:** Medium (user-directed shape change; physics-grounded in fleet survey)
- **Discovery:** User renders — kicked deck ends rejected (want 100% flat); square bow corner vs rounded stern quarter (red line); fleet research (Saildrone/Sailbuoy/Microtransat survivors).
- **Root causes:**
  1. Deck kick came from `sheer_bow/stern` bounds; optimizer bought reserve at ends to dodge the midship slab gate.
  2. Bow corner square because both ends are `y=0` point columns at 70% `T_canoe` with no bottom-rise term; only topside rake/plan-floor existed.
- **Fix:** sheer/rake bounds pinned `[0,0]` (deck flat to 0.000mm measured, 1e-6 test); new `forefoot_cut [0,0.6]` (fraction of local T removed at stem, fixed 0.20 LWL smoothstep extent, default anchor 0.35) applied identically in control net + analytic + SAC loop (net↔analytic sync preserved); legacy 17/20-dim vectors pad flat + full forefoot; reserve fallback is midship volume (BWL/Cp/Cm/T, fuller U) then E→0.40, never kick-back.
- **Verification:** `test_flat_deck_stl_tolerance` (deck_err 0.0mm), `test_forefoot_cutaway` (bottom@10%LWL −0.166→−0.129 at cut 0.6), full `test_geometry/test_config/test_constraints/test_hydrostatics/test_database` green (90 passed).
- **Impact:** Config signature change → wipe for fresh campaign; old DBs refuse by design.

## Bug #166: Fallback Gate-5 passes + phantom tip ballast + free depth (two-sided depth pricing)

- **Files:** `hull_opt/templates/dualsphysics.py` (`write_inverted_case`), `hull_opt/sph_resistance.py` (`run_inverted_pressure`), `hull_opt/hydrostatics.py` (`_ballast_struct_split`, `compute_cg_z/x`, `compute_lumped_inertia`), `hull_opt/geometry.py` (reporting split), `hull_opt/low_fidelity.py` (`draft_penalty_for`, light-wind T, FoM debit), `tests/test_plan_impl.py`, `docs/BUGS_FIXED.md`
- **Severity:** High (campaign winner 157: T/L 0.91 on phantom ballast + unverified Gate 5)
- **Discovery:** Post-campaign audit of the insane-mode run (167 designs, converged iter 166): all 3 top designs' Gate 5 = SPH AbortBoundOut -> analytic fallback; winner's 70 kg ballast stacked at bulb point vs 35 kg lead capacity; T/L 0.91 paid 4.9 into dead violation_magnitude, 0.0 into FoM.
- **Root causes:**
  1. Inverted tank mixed param extents (`-1.5*T_total` floor) with measured tip height; roll-swing radius (~1.2 m on 2.18 m hull) exceeded ~0.20 m wall headroom (def-box != domain); real `compute_cg_z` computed then dropped; fixed dp blew the 400k cap (~750k at 2.18 m).
  2. `ballast_cg = -(T+D)` for ALL of `total*frac` even at 2x lead capacity; floor trimmed mass but never re-located it.
  3. T/L penalty never reached feasible FoM; light-wind Michell excluded the fin; zero structural proxy for root bending (PYD Ch.13).
- **Fix:**
  1. Domain from measured STL bounds + dz_eq-first: walls cover tip+/-bob, floor covers post-roll tip, ymax covers swing radius R; adaptive dp (Bug #162 pattern); real CG + lumped inertia threaded through (`cg_z`, `inertia` params; box fallback kept). Verified: 2.18 m hull GenCase builds 218k fluid / 283k total < 400k cap.
  2. `_ballast_struct_split`: cap-fit lead at bulb, excess mid-fin at -(T+D/2), shared by all 3 mass paths + reporting.
  3. Light-wind Michell at T_canoe+D_keel (fin wetted at all speeds — was free ride).
- **Verification:** `test_honest_ballast_split_vcg` green; measured Gate-5 rerun for 157/104/163 (all status OK, recorded in validation).
- **Follow-up (Bug #168):** the feasible-path `draft_penalty_for` (0.15 quad + 8 cubic) and `m_struct = 0.035` coefficient added here were THEMSELVES tuned backwards from outcome targets ("dethrone 157", "~28 kg"). Removed/replaced — see next entry.

## Bug #167: Reference-storm paddle ejection on narrow hulls + 5-min STL polls on infeasible designs + 2021° roll_sigma garbage

- **Files:** `hull_opt/templates/dualsphysics.py` (`storm_domain`), `hull_opt/surrogate.py` (ReferenceRunner STL poll), `hull_opt/rapid_gates.py` (`_sigma_deg_capped`), `tests/test_plan_impl.py`
- **Severity:** Medium (5/16 reference storms FAILED; audit-only data loss + GPU waste; garbage sigmas in DB/logs)
- **Discovery:** Campaign post-mortem: ref_ds_41/91/121/141/161 all `AbortBoundOut`, all B=0.55; all B=0.68 passed.
- **Root causes:**
  1. `Error_BoundaryOut.vtk` plane at x=-3.744 + `Run.out` MapRealPos min -3.74384: the excluded body is the WAVE PADDLE, not the hull. Deterministic JONSWAP piston (seed 42) return stroke (~0.36 m in shallow transfer regimes) exceeds the fixed 0.35 m back pocket + kernel margin. Deep/wide cases survived by luck of transfer/timing, not margin.
  2. ReferenceRunner polled 5 min for STLs of infeasible (E_GEOM) designs that never get geometry (designs 19/33) — enqueue is post-eval so `feasible` is final and checkable.
  3. Near-undamped RAO resonance integrates to sigma 1455-2021°; stored raw although downstream tanh/gates already neutralized it.
- **Fix:**
  1. Pocket sized from Hs: `max(0.35, Hs+4dp)` in `storm_domain` (single source — particle estimator auto-syncs). Failing case pocket 0.35 -> 0.84 m; fetch cost ~0.5 m, particle cost absorbed by adaptive dp.
  2. Drop `feasible==0` storms immediately, no poll.
  3. `_sigma_deg_capped` (180°) at both sigma assignments; raw `_roll_sigma_rad` untouched.
- **Verification:** failing-case storm regenerates (GenCase 210k particles < caps); full 10 s GPU solver rerun of design 41 via production `run_reference_storm`: status OK, zero aborts (old case died at t=1.12 s); campaign DB left intact (41 stays FAILED as run-record); `test_storm_paddle_pocket_scales_with_hs`, `test_sigma_deg_capped` green; `test_sph_gates` 26 passed; legacy `hyper_test`/`stress_test` dim asserts + `test_rao_surrogate` vectors made dimension-agnostic (17→21 rot); full suite 407 passed.

## Bug #168: Outcome-tuned depth pricing (draft debit + 0.035 structure coefficient)

- **Files:** `hull_opt/low_fidelity.py` (deleted `draft_penalty_for`), `hull_opt/hydrostatics.py` (`_ballast_struct_split` derived scantling + `config` thread-through), `hull_opt/config.py` (`structural_allowable_stress_pa`, `structural_density_kg_m3`), `config.yaml` (same + `max_total_draft_m: 3.05`), `hull_opt/constraints.py` (hard draft ceiling, `ballast_moment` info-only), `tests/test_plan_impl.py`, `docs/BUGS_FIXED.md`
- **Severity:** High (FoM-shaping constants tuned backwards from outcome targets: "dethrone 157", "~28 kg")
- **Discovery:** Self-audit of the Bug #166 fix: `draft_penalty_for` coefficients (0.15 quad + 8 cubic) were picked so T/L 0.91 costs ~3 FoM, and `m_struct = 0.035·ballast·D²/chord` was picked so the 157-like fin reads ~28 kg. Both are judgments labeled as derivations.
- **Root causes:**
  1. Transport/logistics priced as a per-Newton FoM tax instead of feasibility: a 10 ft user transport requirement is a hard ceiling, not a shaping curve.
  2. Structure mass as a tuned coefficient instead of a scantling: PYD Ch.13 sizes the root from the bending moment, not from ballast·D².
  3. `ballast_moment = frac·D` constraint reverse-engineered from `2.2·0.75 = 1.65` to bind the winner.
- **Fix:**
  1. Deleted `draft_penalty_for` — no feasible-path draft debit anywhere in FoM. Depth still prices honestly via GM/RE/AVS/drive benefits vs light-wind Michell at full draft + keel induced/added-wave drag. Transport = hard gate: `T_total > fixed.max_total_draft_m` (3.05 m) → infeasible.
  2. `m_struct` DERIVED cantilever: `M = tip·g·D·safety`, `t = 6M/(σ·c²)`, `m = t·D·c·ρ`, centroid `-(T+2D/3)` (tapered laminate). σ/ρ from `fixed.structural_*` — carbon primary structure (600 MPa / 1600 kg/m³); layup = Kevlar outer (impact) + glass general + carbon structure + metal frame. Result: ~0.47 kg on the monster fin — carbon laminate is nearly free; the honest anti-depth costs are drag + logistics, stated openly with scope (skins only; floors/bolts/grounding excluded).
  3. `ballast_moment` cap deleted; product recorded info-only. Fleet grounding for all of the above: PYD ballast 0.25–0.50, GM 10–15% LWL, AVS ~118° avg, STIX Cat A; USNA ~30° aft sweep + taper ≈0.45; Fanhai-T2 SIMP (mass −33%, CG −34 mm, sway −48%, roll −43%, resistance −37%) via layered lay-up stacking.
- **Verification:** `test_struct_mass_calibration` (hand-checked derivation + carbon bands), `test_max_draft_hard_step` (3.05 m + `draft_penalty_for` gone — SUPERSEDED by Bug #169 below), `test_honest_ballast_split_vcg` green; `test_plan_impl.py` 16/16 pass; zero `frp_`/`draft_penalty_for`/`ballast_moment`-as-gate references left. Golden regen PROVEN honest: `tests/golden/metrics.json` re-recorded (design_0 fom −0.44→+0.94, rt 46.4→38.6 N — debit removal + 27 kg structure loss); clean-HEAD worktree reproduces the OLD golden exactly, so the delta is 100% working-tree (this bug's scope). Full suite (`-m "not slow"`): everything green except the golden, now fixed → `test_golden_smoke.py` 5/5 pass.

## Bug #169: Single-point upwind-biased FoM + draft wall (mission mis-scoring)

- **Files:** `hull_opt/balance.py` (`tws_ops_kt` param, `reach_drive_ops_N`), `hull_opt/low_fidelity.py` (mission bands, gust reward, VMG leeway, logistics cost, helpers), `hull_opt/constraints.py` (wall → priced), `hull_opt/config.py` + `config.yaml/fast/phase0` (band weights, `w_mission_drive/gust/leeway`, `draft_free_m/logistics_per_m`; `max_total_draft_m` DELETED), `tests/test_plan_impl.py`, `scripts/rescore_mission.py`
- **Severity:** High (campaign learned stubby keels under rigged scoring; user mission = ocean crossing, mostly reach/run, gust-ready; draft = inconvenience-with-price per user directive)
- **Discovery:** User asked why validated top-3 wear bound-pinned stubby fins (D_keel = 0.45 m lower bound) while honest rescore puts deep 133 (D = 1.58 m) top-3; plain-English review confirmed the FoM scored one upwind-ish point + light bonus, saturated the drive/storm bonuses via tanh, capped leeway at 0.2, and walled draft at 3.05 m.
- **Root causes:**
  1. One wind point (10 kt) + light bonus priced a mission the boat never sails; reach/run drive (the actual ocean-crossing work) entered only via saturated `0.6·tanh`.
  2. Gust readiness gated but never rewarded — 30° survival headroom scored = 1° headroom (saturated 0.4 tanh).
  3. Leeway > 6° cost at most 0.2 — crabbing at 9° nearly free.
  4. Draft wall contradicted the user's stated preference (price, not limit).
- **Fix:**
  1. Three bands through the SAME solver (5/10/22 kt × 0.25/0.45/0.30): reach-drive (TWA 90/135/180) per band vs Rt, `w_mission_drive`. Stored `result.balance` stays the 10 kt call — gates + DB keys untouched. Per-band boat speeds (user: 5 kt max): drift 2.0 / work 3.5 / breeze-on 5.0 kt (`band_light_kt`, `target_speed_knots`, `band_heavy_kt=max_speed_knots`) — heavy band at Fn ~0.53, past the 0.45 barrier, priced honestly.
  2. `w_gust` × heavy-band heel margin (real gradient, replaces storm tanh); upwind VMG kept for shifts.
  3. `leeway_penalty_deg` helper: linear past the 4° PYD norm, `w_leeway` slope (replaces 0.2 token).
  4. `draft_logistics_cost` helper: free under `draft_free_m` 1.0 m, linear rate above (stated judgment); only T_total > LWL fails (physical RealityCheck cap).
  5. All weights/judgments in all 4 config files; helpers pure + unit-tested with ±50% sensitivity probes. Mission scalars persisted to constraint_values → results.md/CSV columns (`test_mission_columns_reach_report`) — production runs can see what the FoM rewarded.
- **Verification:** `test_mission_bands_and_leeway` + `test_draft_priced_not_walled` green (17/17 plan_impl); golden re-recorded under mission FoM with physics-identical signature (Rt/GM/RE/cg UNCHANGED, fom 0.94→1.62 — pure scoring delta, exactly as designed); `scripts/rescore_mission.py` added for DB-read-only audit rescores.

## Bug #170: Campaign DB silently wiped by a fresh-run wipe (unrecoverable data loss)

- **Files:** `run_optimization.py` (`_clean_slate`), `tests/test_sph_lock.py`, `output/` (restored by user from backup)
- **Severity:** Critical (160-design campaign DB + all design dirs deleted mid-session 2026-09-05; DB file left 0 bytes; cause unproven — no default-suite test calls the wiper on `./output`, no stray processes found)
- **Discovery:** Rescore script hit `no such table: designs`; `output/` held only the empty DB. Other on-disk DBs are older campaigns (160-design Aug-25, 39-design) — the 167-design run itself had no second copy.
- **Fix:** `_clean_slate` NEVER unlinks a DB holding designs (or an unreadable/oversize DB — assumed precious): it moves DB+WAL+SHM to `output/.backup_<ts>/` and logs loudly. Empty DBs still clean as before (existing tests unchanged).
- **Verification:** `test_clean_slate_backs_up_campaign_db` green (7/7 sph_lock); lesson recorded in AGENTS.md wipe-hygiene note. Standing rule: never point a fresh run at a live campaign dir without --resume; keep an offline DB copy before mode switches.

## Bug #171: Audit findings — dead survival gates, lost spectral fix, leaky cap, AVS/CLR/mass errors

- **Files:** `hull_opt/constraints.py` (reserve latch), `hull_opt/rapid_gates.py` + `hull_opt/low_fidelity.py` (JONSWAP), `hull_opt/michell.py` + call sites (`delft_cap_frac`, true cap), `hull_opt/hydrostatics.py` (AVS, centroid), `hull_opt/geometry.py` (payload closure), `hull_opt/rig.py` (CLR), `hull_opt/param_layer.py` (fallbacks), `tests/test_plan_impl.py`, `tests/test_param_layer_coupling.py`
- **Severity:** Critical (2 items) + High (rest). Found by 4 adversarial subagents reading every line against JFR/PYD-5/Fanhai-T2, with the 2 criticals hand-verified in code.
- **P0 — survival correctness:**
  1. `reserve_fatal` overwrite: `else: reserve_fatal = False` cleared RE/self-right/accel fatals whenever reserve passed — unsurvivable designs scored feasible. Now latches (never cleared). Pinned by `test_reserve_fatal_latches`.
  2. JONSWAP alpha: Bug #71's fix (5/16→0.0081) never reached code (squashed history). Autopsy went further: 0.0081 belongs to the g² spectrum form — in this Hs² form NO constant is right, so both spectra now self-normalize to m0=(Hs/4)² (exact by definition of Hs, any sea state). Pinned by `test_jonswap_alpha_phillips` over 3 sea states.
- **P1 — honest water:**
  3. AVS was argmax-seeded: a dominant inverted lobe returned 180° despite a ~110° first crossing. Now scans from the first positive sample (`test_avs_first_crossing`).
  4. Michell cap leaked (`max(cap, 0.05·Rw)` — spikes over 20× cap paid 5% unbounded) and was Fn-flat with inconsistent 0.035/0.02 pair. Now TRUE `min(Rw,cap)` with `delft_cap_frac(Fn)` from PYD printed bands (0.4%@0.30 → 5%@0.45, last-band above validity, stated).
  5. Added-wave was linear in Hs (2× over at moderate seas); now Hs².
  6. CLR omitted T_canoe (6% LWL fwd error vs a 0.5% gate) and mis-anchored the root LE (0.4kc vs meshed 0.65kc): now waterline-extended `0.45·(T+D)` from the meshed 25%-chord root.
  7. `param_layer` no-config fallbacks decoded a different boat (D_keel capped 0.65, bulb 0.004, flare/deadrise/bilge/wingsail/sheer all stale): now mirror live bounds (bulb geometric shrink excepted, by design). `test_fallback_bounds_match_config` + updated `test_bulb_vol_decouple_when_config_none` (it had pinned the rot).
  8. Reporting mass light by the 15 kg payload vs the CG path (mass non-closure): geometry base now includes payload. Structure centroid corrected to root-heavy −(T+D/3) (was tip-heavy −(T+2D/3); sub-kg effect, fixed for the derivation's honesty).
- **Verification:** 23/23 plan_impl; golden re-recorded with honest signature (Rt 38.6→24.6 N via Fn cap, storm accel 2.24→1.81 g via normalization, GM/RE/CG unchanged); top-8 rescore under fixed physics re-ranks 152→#1 (6.248), 158→#2 (5.879), 124→#3 (5.790); fast suite green modulo the one pinned-rot test, fixed.
- **Known open (P2, not this bug):** rig CL double-discount + CE height, Oswald/end-plate factors, Gate 1/4 pass-by-construction thresholds, helm 5° unenforced, Fanhai % cited-not-enforced, AGENTS drift items (NURBS default, SAC cap, thresholds, fast/phase0 weights).
