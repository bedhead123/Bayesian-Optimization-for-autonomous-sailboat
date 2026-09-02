# Rapid Validation Benchmark — Tier-1 Gate Suite

## 1. BEM Storm Benchmark (P0a)

### Baseline 1×15

| Metric | Value |
|--------|-------|
| Headings | 1 (180°) |
| Frequencies | 15 |
| Panels | 2500 |
| Wall time | ~13 s (idle) |
| Problems/design | 105 (15 diff + 6×15 rad) |

### Storm Sweep — Naïve (3×30)

| Mode | Headings | n_freq | Problems | Wall time |
|------|----------|--------|----------|-----------|
| `sweep30` (naïve) | 180/135/90 | 30 | 630 | 77–193 s |
| `one30` | 180 | 30 | 210 | ~46 s |
| `sweep15` | 180/135/90 | 15 | 315 | ~26 s |

### Radiation-once optimisation (smart)

| Mode | Headings | n_freq | Problems | Wall time |
|------|----------|--------|----------|-----------|
| `smart30` | 180/135/90 | 30 | 270 (180 rad + 90 diff) | ~1.4 min |

- **−57% problems** vs naïve: radiation solved once, diffraction per heading.
- **15→30 freq peak-accel discrepancy**: up to ±70% between 15 and 30-line BEM sweeps at identical panels. The 30-freq storm sweep is the production choice.
- **Panel scaling**: N^1.85 (2500 → 4000 panels costs ~3× wall time).
- **Chosen defaults**: headings [180, 135, 90], `storm_n_freq=30`, panels 2500, budget ≤2.5 min/design (production ~1.4 min with radiation reuse).

### BEM tunables (config defaults via `getattr`)

- `storm_n_freq` → 30
- `storm_headings_deg` → [180.0, 135.0, 90.0]
- `storm_hs_factor` → 4.0
- `storm_tp_factor` → 1.3

## 2. GPU Tool Decision (P0b)

### FluidX3D v3.7 — REJECTED
- Prebuilt binary only; no input-file interface, no free-surface, no 6-DoF, no wave initialisation.
- Benchmark-only tool; cannot be used for validation case generation.

### DualSPHysics 5.4 — ACCEPTED
- Location: `/home/anon/apps/DualSPHysics_v5.4`
- GPU solver binary: `bin/linux/DSGcc7/DualSPHysics54.linux64` (verified runs OK).
- Needs `LD_LIBRARY_PATH` pointing to the solver directory.
- Chosen as Tier-2 reference tool (sph_gates.py, ≤30 min GPU per design).

## 3. Tier Taxonomy

| Tier | Module | Solver | Budget/design | Purpose |
|------|--------|--------|---------------|---------|
| 1 | `rapid_gates.py` | Capytaine BEM + analytic | ≤3 min CPU | Pre-filter BO candidates; 5 gates (R, W, S, P, E) |
| 2 | `sph_gates.py` | DualSPHysics 5.4 | ≤30 min GPU | Wave impact / slamming reference for ~top-30 candidates |
| 3 | `high_fidelity.py` | OpenFOAM interFoam | ≤4 h CPU | Finalist validation (6 gates) |

## 4. Gate S Formula (Wagner Wedge Impact)

The slam pressure is estimated via the analytic 2D Wagner wedge model:

- **Drop velocity**: `v = sqrt(2·g·h)` with h = 3.0 m → v ≈ 7.67 m/s
- **Wagner peak pressure**: `p_max = 0.5·ρ·(0.5·π·v·cot(β))² / 2`
  - `β` = deadrise angle (rad), minimum 2° to avoid singularity
  - Factor 1/2 in denominator accounts for 3D spreading vs 2D wedge
- **Slam acceleration**: `a_g = (p_max · A_wet) / (m_hull · g)`
  - `A_wet` ≈ 0.5 · BWL · LWL (approximate bottom area)
  - `m_hull` = displacement × ρ_water

This is a conservative estimate: real 3D pressure is lower than 2D wedge; the factor `/2` is defensible per Chuang (1966) and Johnsen (1968). The 7.67 m/s entry matches the gate-4 crash analysis (docs/CFD_GATE_CRASH_ANALYSIS.md §2). Maximum pressure upstream is clipped by `config.validation.max_pressure_pa` (100 kPa default).

## 5. Cost Model Per Campaign

| Stage | Designs | Unit cost | Total core-minutes |
|-------|---------|-----------|-------------------|
| BO initial sampling (LHS) | 20–40 | ~17 s (low-fi) | ~10 |
| BO iterations | 300 | ~17 s (low-fi) | ~85 |
| Tier-1 rapid gates | top-50 from BO | ~1.4 min | ~70 |
| Tier-2 SPH | top-30 | ~30 min | ~900 |
| Tier-3 OpenFOAM | 6 finalists | ~240 min | ~1440 |
| **Total** | — | — | ~2505 |
