# CFD Gate Crash Analysis — interFoam rc=136 FPE (Aug 2026)

Root-cause analysis of why all 6 wave/drop validation gates failed with
`interFoam returned 136` (floating-point exception) while every calm-water,
self-righting, and pressure gate passed. Evidence: `output/validation/`
case dirs and `output/pipeline.log` from the Aug 3–6 2026 campaign.

## 1. Crash signatures

| Gate | Design | Crash time (sim) | Collapse signature | FPE site |
|------|--------|------------------|--------------------|----------|
| gate2 wave | 98/117/102 | **t = 0.1226 s** (all 3 attempts, identical) | deltaT 1e-3 → 1e-12 → 1e-105; max Co 68 in 1 cell (mean 3e-4); alpha → [−156, +65] | `PCG::scalarSolve` (p_rgh) |
| gate4 drop | 98/117/102 | **t = 0.0022 s** | alpha → ±1e110; deltaT → 2e-17 | `libinterfaceProperties` (surface tension / curvature) |
| gate1 calm | 98/117/102 | no crash | — | — (PASS) |

In both cases alpha.water (volume fraction, must stay in [0,1]) goes wildly
out of bounds → negative densities → NaN/Inf in the pressure solve →
FPE. The unboundedness is the *disease*; the crashes are the *symptom*.

## 2. Root cause — gate 2 (wave motions): potentialFoam corrupts 0/U

The case write (`write_openfoam_case`) initializes U to a clean uniform
free stream: `internalField uniform (1.6462208 0 0)` (3.2 kn) with a flat
free surface (alpha = 0/1 step at z=0) and `p_rgh = 0`.

Gate 2 then ran `_run_potential_foam()` (potentialFoam) before interFoam.
potentialFoam solves a single-phase velocity potential and **overwrites
0/U**. With the atmosphere patch typed `pressureInletOutletVelocity` — an
invalid BC for a potential solve — the resulting "potential" field is
garbage:

```
0/U internalField (after potentialFoam, mtime 21:41:11):
  u ∈ [−0.02, 1.59] m/s
  w ∈ [−3.00, +3.00] m/s      ← ±3 m/s vertical velocities for a
                                 1.65 m/s free stream
```

Evidence that this field is the trigger:

- `gate1_calm` 0/U is uniform (`grep internalField`) — gate 1 never runs
  potentialFoam and never crashes, with the identical mesh and solver
  settings.
- Gate 2's 0/U mtime (21:41:11) matches the potentialFoam step, minutes
  after the template write and before interFoam (first attempt log 21:46).
- The 6-DoF motion state at t=0.1 s shows the hull being slammed by the
  unbalanced field: `acceleration = (0.035, −0.0007, −5.28) m/s²`,
  `torque = (−0.04, −665, −0.21) N·m`. The ±3 m/s water column loads the
  hull with a 0.54 g downward acceleration and a 665 N·m pitch torque.
- At t=0.1226 s a single cell's velocity explodes: max Courant jumps to
  68.2 while the mean is 3e-4. PIMPLE stops converging → `adjustTimeStep`
  collapses deltaT to 7e-12 → the solver never advances past
  t=0.12262473 s → death spiral (deltaT → 1e-105, velocities → 1e104,
  alpha → [−156, +65]) → FPE in the p_rgh PCG solve.
- **All three interFoam attempts crash at the identical sim time
  (t=0.1226 s) — fully deterministic.** The retry mechanism cannot help.

### Bonus finding: the "regular wave" is never generated

`config.validation.regular_wave_H` (2.0 m) and `regular_wave_T` (6.0 s)
are **dead config** — grep across the codebase: nothing consumes them.
The gate-2 case contains no wave: flat free surface, uniform current,
`p_rgh=0`. The gate as configured does not test wave motions at all; it
tests "hull with 6-DoF in a uniform current" — and the only wave-like
signal in the case was the accidental potentialFoam artifact.

## 3. Root cause — gate 4 (drop impact): violent slam at deltaT 5e-4

Gate 4 starts the hull at the free surface with a 6-DoF initial velocity
`v_drop = sqrt(2·g·drop_height) = 7.67 m/s` (drop_height 3.0 m) into
still water, deltaT=5e-4, maxCo=maxAlphaCo=0.7, end_time=1.0 s.

- The 7.67 m/s water entry of a hull with a 1.2 m keel is an extreme
  slamming event; the interface is shredded within **t=0.0022 s** (4
  steps) — alpha reaches ±1e110, deltaT collapses, surface-tension
  curvature NaNs → FPE in `libinterfaceProperties`.
- potentialFoam also ran here but produced the trivial U=0 field (no
  forcing), so it is not the gate-4 trigger — the slam itself is.
- Note the gate is not calibrated against any impact reference; the
  analytic threshold (30 g) was never reachable because the solver dies
  first.

## 4. Contributing factors (both gates)

1. **No crash taxonomy.** A crashed gate was stored as `FAIL, measured=0.0`
   — indistinguishable from a genuine design failure. Fixed: gates are now
   classified `PASS/FAIL/CRASH/TIMEOUT` (see `high_fidelity.py`).
2. **Mesh quality headroom.** `log.checkMesh` for design 98 shows
   `Max skewness = 1.93` — acceptable to OpenFOAM, but slamming/6-DoF
   cases are where marginal cells die first.
3. **6-DoF mass/moment check.** The hull mass is `rho × target_displacement`
   with inertia from a box model; no hydrostatic-equilibrium verification
   of the initial float position exists. A sinking/pitching hull (gate-2
   state file) amplifies mesh distortion.
4. **No wave initialization framework.** A physically valid wave gate needs
   alpha + U + p_rgh all initialized consistently (waves2Foam-style);
   initializing only U is exactly what produced the corrupt state here.

## 5. Fixes applied

| Fix | File | Effect |
|-----|------|--------|
| Removed `_run_potential_foam` from gates 2 and 4 | `high_fidelity.py` | Gate 2 now starts from the clean uniform field exactly like the stable gate 1 — removes the corrupt-U trigger. Gate 4 keeps its (trivial) potential step removed; the slam starts from quiescent water. |
| Removed dead `_run_potential_foam` helper | `high_fidelity.py` | no callers remain |
| Gate crash/timeout classification | `high_fidelity.py`, `database.py`, `run_optimization.py` | crashes/timouts stored with `status=CRASH/TIMEOUT`, never as fabricated `FAIL val=0.0` |

## 6. Recommended follow-ups (not yet implemented)

1. **Real wave generation for gate 2** — implement a proper Airy/Stokes
   initialization (alpha, U, p_rgh, consistent) or integrate waves2Foam;
   only then does the gate test what it claims. With H=2.0 m on a 0.3 m
   draft hull, sanity-check the wave height against hull size first.
2. **Gate-4 stability** — reduce `max_alpha_co` to 0.5, drop deltaT to
   2.5e-4 for the first 0.1 s (or use `adjustTimeStep` with a hard
   `maxCo=0.5`), and add a free-surface refinement box at z=0 around the
   impact line. Re-verify with a short run before trusting the 30 g
   threshold.
3. **Float-position check** — verify the 6-DoF hull starts at its
   hydrostatic equilibrium (displaced volume = mass/rho) before running;
   a sinking hull is a setup bug, not a wave response.
4. **Mesh-quality gates** — reject designs with max skewness > ~1.5 or
   max aspect ratio > 100 before CFD (the geometry validator already
   rejects edge ratios > ~1000; tighten the pipeline margin).
