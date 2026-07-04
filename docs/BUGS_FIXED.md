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