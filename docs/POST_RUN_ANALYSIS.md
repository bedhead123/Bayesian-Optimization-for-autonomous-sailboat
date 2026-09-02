# Post-Run Failure Analysis — Aug 3–6, 2026

Full analysis of the completed production optimization campaign. Evidence sources:
`output/pipeline.log` (5029 lines), `output/optimization.db`, `output/quick_test/quick_test.db`,
`output/results.md`/`results.csv`, validation case dirs under `output/validation/`.

---

## 1. Executive summary

The pipeline "finished" and reported a converged optimum, but the result is not trustworthy:

1. **The optimization objective was non-stationary.** A global drag-calibration factor
   changed 7 times during the run (1.0956 → 0.7009 → 0.7704 → 0.8181 → 0.7967 → 0.9434 →
   0.9856). Stored FoMs were computed at the factor active at evaluation time and never
   rescored, so the GP trained on data produced under mutually inconsistent objectives.
   The final top-3 (designs 98, 102, 117) were all evaluated under f=0.70 and were never
   re-scored under the final f=0.99.
2. **~87% of wall time (38.5 h of 44.5 h) was spent on mid-fi calibration**, of which
   3 attempts hard-timed-out (4 h each), 3 "completed" by extracting a drag value from a
   **non-converged** simulation tail, 1 was killed by the user, and only 4 converged cleanly.
3. **The config changed between phases.** The optimization ran at 4.0 kn / 0.145→0.14 m³
   (Aug 3–5); the final validation-only run loaded **3.2 kn / 0.145 m³** (Aug 5 20:58).
   All gate thresholds and the regenerated results.md reflect the new operating point.
4. **All 6 wave/drop validation gates failed — not because the designs are bad, but
   because interFoam crashed** (FPE, rc=136) in every one of them (54 crashes total,
   38 retries). Every analytic gate passed. The gate code records a crash as
   `measured=0.0, FAIL`, conflating "solver failure" with "design failed".
5. **The BO plateaued early and wasted ~40% of its budget.** Best FoM found at iter 97;
   iters 98–223 (126 evals) found nothing better, and after iter 195 zero designs were
   feasible. Only 32/224 designs (14%) were ever feasible.
6. **Operational chaos around the finish:** three processes wrote the same
   `output/results.md` concurrently (validation-only run + 2 quick tests), the production
   run was killed mid-calibration, validation executed three times total, and the
   `calibration_attempts` table lost data (rebuilt with wrong timestamps on resume).

---

## 2. Run timeline

| # | Start (local) | Mode | Config | Notes |
|---|---------------|------|--------|-------|
| 1 | Aug 3 15:39 | quick test | fast | 5 LHS + 2 BO; validated top-3 |
| 2 | Aug 3 15:51 | quick test | fast | |
| 3 | Aug 3 16:29 | quick test | fast | validation of design 4 (reconstructPar rc=1 ×9) |
| 4 | Aug 3 17:12 | **production** | **4.0 kn / 0.14 m³** | 30 LHS + BO; ran to Aug 4 14:10, killed by user |
| 5 | Aug 4 14:10 | quick test | fast | killed iter-120 calibration (1 h 43 m in) |
| 6 | Aug 4 14:43 | quick test | fast | |
| 7 | Aug 4 15:28 | **production resume** | 4.0 kn / 0.14 m³ | ran 1.2 h, killed |
| 8 | Aug 4 16:43 | **production resume** | 4.0 kn / 0.14 m³ | iter 124→223; converged Aug 5 13:45; then validation tail (gate1 4 h timeout, gate2 running) — killed by user at 20:58 |
| 9 | Aug 5 20:58 | validation-only | **3.2 kn / 0.145 m³** | gates for 98/117/102; finished Aug 6 00:10 |
| 10 | Aug 5 21:40 | quick test | fast | **ran concurrently with #9**; wrote output/results.md |
| 11 | Aug 5 21:53 | quick test | fast | **ran concurrently with #9**; wrote output/results.md; validated its own top-3 [1,7,6] |

Production BO history (run #4+7+8): LHS 30 samples (Aug 3 17:12 → ~18:20), then BO
iterations 1→223 with calibrations at iters 40, 60, 80, 100, 120, 140, 160, 180, 200, 220.

---

## 3. Failure deep-dives

### 3.1 Mid-fi calibration: objective corruption + 87% of wall time

**Mechanism.** `smooth_drag_factor` (surrogate.py:584) updates a single global factor;
every design evaluation applies the factor active at eval time
(surrogate.py:156, 254, 400–421); stored FoMs are **never rescaled** when the factor
changes. `_propose_candidate` trains the GP on all stored FoMs (surrogate.py:468–482),
i.e., on a non-stationary objective.

**Raw CFD/low-fi ratios measured per calibration** (calibration picks the best feasible
design at each scheduled iteration):

| Iter | Design | CFD Rt (N) | Low-fi Rt (N) | Raw ratio | Smoothed factor | Outcome |
|------|--------|-----------|--------------|-----------|-----------------|---------|
| 40 | (design 40) | — | — | — | — | **4 h timeout** (no update) |
| 60 | 60 | 92.87 | 86.00 | 1.080 | 1.0956 | converged |
| 80 | 80 | 100.72 | 138.06 | 0.730 | 0.7009 | converged |
| 100 | 98 | — | — | — | — | **4 h timeout** (no update) |
| 120 | 98 | — | — | — | — | killed at 1 h 43 m |
| 140 | 117 | 105.92 | 123.49 | 0.858 | 0.7704 | converged |
| 160 | 102 | 134.90 | 153.77 | 0.877 | 0.8181 | **not converged**, tail extracted |
| 180 | 130 | 98.65 | 123.25 | 0.800 | 0.7967 | **not converged**, tail extracted |
| 200 | 143 | 146.02 | 135.09 | 1.081 | 0.9434 | converged |
| 220 | 171 | 86.93 | 84.96 | 1.023 | 0.9856 | **not converged**, tail extracted |

Facts:

- Raw ratios vary **0.73–1.08** across designs → low-fi model error is design-dependent
  (±30 %), so a single global factor cannot correct it. It is a *moving average of
  per-design error*, and designs are never re-calibrated, so it is systematically wrong
  for any design far from the calibrated ones.
- Calibration wall time: ~38.5 h of the ~44.5 h campaign (87 %). Each attempt is 4 h
  (14400 s budget, `end_time=20 s` sim time on 8 procs).
- Three 4 h timeouts (iters 40, 100; and 120 killed at 1 h 43 m) produced **no factor
  update**, i.e. the objective silently stayed on the stale value.
- Three "updates" (iters 160, 180, 220) came from **non-converged** runs: the 20 s sim
  end time was never reached; the code extracted a tail-window average
  (`mid_fidelity.py`) and updated the factor anyway.
- **Impact:** designs 98/102/117 (the final top-3) were scored at f=0.7009; at the end of
  the run the factor was 0.9856. Design 171 (rt=72.1 N — the lowest drag of all feasible
  designs) scored 1.3531 *under the inflated-by-0.70 drag reduction*, and the run
  converged under a factor that would re-rank the leaderboard had FoMs been rescored.

### 3.2 Config changed between phases

- Production (runs 4/7/8): `Config loaded: LWL=2.4m, B=0.5m, Target=0.14m³, Speed=4.0kn`
- Validation-only (run 9): `Config loaded: LWL=2.4m, B=0.6m, Target=0.145m³, Speed=3.2kn`

Consequences:
- Gate thresholds (calm_water_rt) were computed for 3.2 kn designs with 0.145 m³
  displacement, not for the 4.0 kn / 0.14 m³ objective that was optimized.
- The drag factor was calibrated at 4.0 kn but gate-1 thresholds used low-fi at 3.2 kn.
  Measured ratio for design 98: Rt_CFD=25.67 N vs low-fi basis 40.03 N → 0.64, vs the
  calibrated 0.99. The calibration is speed-specific; nothing in the pipeline
  tracks that.
- results.md / results.csv were regenerated under the new config, so the "final" report
  describes a different operating point than the one optimized.

### 3.3 Validation gate CFD crashes (rc=136 FPE) — all 6 wave/drop gates

Final stored results (validation-only run, all three designs):

| Gate | 98 | 117 | 102 |
|------|----|----|----|
| 1 calm_water_rt | PASS 25.67 N | PASS 50.81 N | PASS 46.53 N |
| 2 wave_motions_accel | **FAIL** (crash) | **FAIL** (crash) | **FAIL** (crash) |
| 3 extreme_wave_self_right | PASS | PASS | PASS |
| 4 drop_impact_accel | **FAIL** (crash) | **FAIL** (crash) | **FAIL** (crash) |
| 5 inverted_pressure | PASS | PASS | PASS |

- 54 `interFoam (parallel) failed (rc=136)` errors in the log (18 on Aug 3 — quick-test
  era; 6 on Aug 4; 25 on Aug 5; 5 on Aug 6), 38 `retrying (N/3)` warnings.
- Both crash signatures are divergence-driven:
  - `libinterfaceProperties.so` FPE (surface-tension/curvature) — consistent with
    alpha.water going unbounded (seen down to −1.34e+24), Courant exploding (4.18e+51),
    deltaT collapsing (7.48e−88), forces 1e+121 (design_102/gate4_drop).
  - PCG `scalarSolve` FPE in the p_rgh pressure solve (design_98/gate2_wave).
- Gate code records a crash as `measured_value=0.0, passed=0` with detail
  `interFoam returned 136...` — i.e. **a solver failure is indistinguishable from a
  design failure** in the DB and in the report.
- Even when interFoam did not crash, the calm-water gate for design 98 in the production
  tail **hung**: estimated 64 min for 206 k cells, but never converged in 4 h
  (deltaT collapse → stuck at ~1e−88); recorded as `Gate 1: FAIL (Rt=0.000 N)` — a
  timeout mislabeled as a failed gate with a fabricated 0.0 N measurement.

### 3.4 Low feasibility (14 %) and BO plateau

- 224 designs: **32 feasible (14.3 %)**, 192 infeasible; of those, only 13 carry an
  `error_code` (8× convexity 0.196–0.248 vs min 0.25, 5× STL edge ratio 1030–1130,
  1× not watertight); the other 179 are "infeasible" via constraint penalties (FoM < 0).
- Dominant constraint violations: `B/LWL near boundary` (~60 designs, penalty 0.11–0.18),
  `sac_scale_std > 0.5` (per-station SAC scaling), `freeboard/draft > 2.0`,
  `reserve_buoyancy < 0.30`, `bilge_r/BWL > 0.5–0.62`.
- Infeasible error codes cluster *right at the constraint edge* (convexity 0.196–0.248 vs
  the 0.25 threshold) — the BO pushes proposals into the boundary region and the
  geometry generator produces borderline hulls there; 100+ evals were spent on hulls that
  are invalid.
- **Plateau:** best FoM 1.3639 at iter 97; #2 (1.3605) at iter 116; #3 (1.3588) at
  iter 101. Iters 98–223 (126 evals, 56 % of BO budget) produced nothing better; the last
  feasible design was iter 195. The run then "converged" at iter 223
  (no-improvement window — a *spurious-looking* plateau signal given the non-stationary
  objective and 14 % feasibility).
- Note: design 171 (iter 170) has the lowest Rt (72.1 N) of all feasible designs yet ranks
  #7 — FoM is drag-dominated but not drag-only; still, under the final factor it would
  move up.

### 3.5 Operational issues

1. **Concurrent writers.** Runs 9, 10, 11 overlapped (20:58 → 00:10). Quick tests 10/11
   wrote their results report into the **main** `output/results.md`/`results.csv`
   (lines 2864–2866, 2949–2951), clobbering the validation-only run's files mid-run.
   Final state is the validation run's 00:10 write — by luck, not by design.
2. **Killed calibration.** Run 8's iter-120 calibration (1 h 43 m in, 4 h budget) was
   killed when the user started a quick test at 14:10.
3. **Validation ran 3×** with different configs/DBs: production tail (98 only: gate1
   timeout + gate2 crash, killed before storing), quick-test validations (designs 4, then
   1/7/6 — colliding design ids in shared `output/validation/design_*` dirs), and the
   final validation-only run.
4. **calibration_attempts table lost data.** All rows carry created_at 2026-08-04
   22:15–22:17 (the resume moment), attempts for iters 40 and 100 (both timeouts) are
   missing, and iter 120 is recorded as a `timeout` for design 98 although it was killed
   by the user mid-run. The table was evidently rebuilt with `INSERT OR REPLACE` on
   resume and its history is unreliable.
5. **Memory throttle** reduced workers 4 → 3 on every start (10.3–12.8 GB free, 3.5
   GB/worker needed) — minor throughput loss only.
6. **9 reconstructPar failures** (rc=1, reconstructPar.C:285) on Aug 3 during
   quick-test validation of design 4 — inconsistent processor dirs; force extraction
   fell back to processor-local data.

### 3.6 Bug-fix context (already fixed during the campaign)

The run inherited the full `docs/BUGS_FIXED.md` history (Michell g² factor, GP likelihood
disconnect, GM sign, downflooding logic, gunwale watertightness, SAC volume override,
stale-gate-data bug in validation, etc.). The stale-data fix (deleting gate dirs before
re-validation, high_fidelity.py:86–92) is present in the current code and the final
validation-only run benefited from it — but the production tail (which ran before the
fix) did not.

---

## 4. Severity matrix

| # | Issue | Severity | Evidence |
|---|-------|----------|----------|
| 1 | Drag factor changes make the objective non-stationary; stored FoMs never rescored | **Critical** | surrogate.py:400–421 vs 468–482; factor history 1.0956→0.9856 |
| 2 | Calibration consumes 87 % wall time; 6/10 attempts unusable (timeout/killed/non-converged) | **High** | pipeline.log calibration intervals; mid_fidelity.py tail-extraction |
| 3 | Config mismatch between optimization (4 kn/0.14) and validation (3.2 kn/0.145) | **High** | log lines 1584, 2821 |
| 4 | All 6 wave/drop gates fail via solver crash; crash misrecorded as gate FAIL with val=0 | **High** | validation table rows 2,4,7,9,12,14; 54× rc=136 |
| 5 | Gate-1 4 h timeout recorded as Rt=0.000 FAIL | **Medium** | log 2811–2812 |
| 6 | 14 % feasibility; last 126 BO iters worthless; convexity failures at 0.25 edge | **Medium** | DB designs table; results.md error codes |
| 7 | Concurrent processes clobber shared results.md / validation dirs; quick tests write main dir | **Medium** | log 2864, 2949 |
| 8 | calibration_attempts data integrity (timestamps, missing rows) | **Low** | DB table |
| 9 | Worker throttle 4→3, reconstructPar failures, killed calibrations | **Low** | log |

---

## 5. Root causes

1. **Global scalar calibration is the wrong model.** Low-fi vs CFD error varies 0.73–1.08
   per design; a single multiplicative factor is a biased moving average. Combined with
   non-rescoring of stored FoMs, it actively corrupts the GP's training data.
2. **The 20 s / 4 h calibration budget is too tight for this mesh class.** 6/10 attempts
   never reached end_time; the pipeline then *fabricated* a factor from an unconverged
   tail. Timeouts silently leave the objective stale.
3. **No config-identity enforcement.** The DB does not record the config under which
   designs were evaluated; resuming/validating under a different config silently mixes
   incompatible data.
4. **Gate failure taxonomy is missing.** Crash / timeout / genuine-fail all collapse to
   `passed=0, measured=0.0`.
5. **interFoam wave/drop cases are numerically fragile** (FPE in interfaceProperties and
   PCG after alpha unboundedness; deltaT collapse). Root cause analysis in
   `docs/CFD_GATE_CRASH_ANALYSIS.md` (see §6).
6. **Design space is mostly infeasible** and the BO pushes proposals to the constraint
   boundary, where the geometry generator produces borderline hulls (convexity 0.19–0.25,
   edge ratios ~1100) — most of the later budget was spent on invalid hulls.

---

## 6. Recommendations (prioritized)

**Critical (do before next run):**
1. **Rescore FoMs under the current factor.** Store the factor-independent raw drag at
   eval time (or store `factor` per design) and rescale stored FoMs in GP training /
   ranking whenever the factor changes. [FIXED — see §7]
2. **Freeze config per campaign; hash it into the DB.** Refuse to resume/validate if the
   loaded config differs from the campaign config. [FIXED — see §7]
3. **Gate classification.** Record `PASS/FAIL/CRASH/TIMEOUT` per gate; never invent a
   measured value (Rt=0.0, accel=0.0) for a failed run; add optional stop-on-crash.
   [FIXED — see §7]

**High:**
4. Fix the wave/drop case numerics (root cause and concrete fixes:
   `docs/CFD_GATE_CRASH_ANALYSIS.md`): initial alpha field bounded, MULES limiter
   settings, deltaT control, mesh-quality lower bounds (edge ratio) in geometry
   validation, drop-case initial placement.
5. Replace global factor with per-design (or per-region) correction, or at minimum skip
   updates from non-converged tails and cap the 4 h budget lower (e.g. 2 h) with a
   clear "no update" path. Consider running calibration **asynchronously** (it blocks the
   BO loop for 87 % of wall time).
6. Reduce boundary-waste: tighten proposal sampling near constraints or add a constraint
   model to acquisition; raise convexity margin so borderline hulls fail geometrically
   earlier.

**Low:**
7. Isolate quick-test outputs (`output/quick_test/`) including results.md/csv; add a
   per-process run-id / lock for shared files.
8. Fix `calibration_attempts` logging (append-only, real timestamps).
9. Re-run the campaign end-to-end under a frozen config before trusting any final design.

---

## 7. Fixes applied (post-run)

| Fix | File(s) | Behavior |
|-----|---------|----------|
| FoM rescoring on drag-factor change | `surrogate.py`, `low_fidelity.py`, `database.py` | `rt_wave`/`rt_friction` are stored unscaled, so stored FoMs are rescaled exactly under the current factor (`rescore_foms`) before GP training/ranking, whenever the calibration factor changes or on resume; per-design `drag_factor` audit column; in-memory history re-synced so the convergence check never compares foms from different factors |
| Config-hash guard | `surrogate.py`, `run_optimization.py`, `database.py` | campaign config (bounds, fixed, weights, validation) hashed into DB at campaign start; optimize/resume/**validate-only** runs refuse to proceed on a mismatched DB unless `BOAT_IGNORE_CONFIG_MISMATCH=1` |
| Gate crash/timeout classification | `high_fidelity.py`, `database.py`, `run_optimization.py` | `status` column: PASS / FAIL / CRASH / TIMEOUT; solver crashes and wall-time timeouts are recorded as such (`GateCrashError`/`GateTimeoutError`), never as a fabricated `FAIL val=0.0` |
| Gate 2/4 CFD crash root cause | `high_fidelity.py` | removed `potentialFoam` from gates 2 and 4 — it overwrote 0/U with a garbage potential field (±3 m/s vertical velocities from an invalid atmosphere BC), deterministically exploding a single cell at t=0.12 s (see `docs/CFD_GATE_CRASH_ANALYSIS.md`) |
