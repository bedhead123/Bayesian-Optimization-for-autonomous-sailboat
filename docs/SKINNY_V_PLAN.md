# Skinny V Deep Narrow - Plan - Fast + Stable + 50ft Drop

## Goal
Keep L/B ~4 skinny fast, survive 50ft drop, stable, hit displacement without balloon.

## What was wrong (root cause)
1. Dry-run fed physical 2.4 as raw -> sigmoid 0.92 near-max (run_optimization.py:451)
2. FABRIC check used box *0.7, real V hull *0.55 (config.py:343 vs geometry.py:852)
3. Biggest skinny boat 0.105 even at 1.3x only 130kg, never 170kg needs 1.98x (cube)
4. Display bug Cp in actual_Cp (run_optimization.py:539)

## Fixes pushed (b407ca7, 6d47452, 87e0b40)
- BWL 0.55-0.68 stays skinny
- T 0.26-0.32 -> 0.28-0.36 deep not wide
- E 0.25-0.45 -> 0.35-0.45 high reserve 0.78
- deadrise 26-30 -> 28-30 sharp V for 50ft slam
- bilge 0.14-0.28 -> 0.14-0.22 tight
- D_keel 0.90-1.95 -> 0.90-1.90 fits T+D<=LWL
- target 0.17 -> 0.14 (140kg) hits without balloon
- run_optimization.py:451 raw zeros true median
- run_optimization.py:538 exact Cp match

## Validation now
- dry-run FEASIBLE vol 17% (was 37%) Cp OK actual_Cp soft
- median 0.112/0.14 80% max 0.131/0.14 93%
- GM 0.87-1.03 reserve 0.78 energy 1206J >>40J
- Rt 125-149N, helm 16deg, slam 0.51 pass

## Next
1. ./run.sh --hyper-test (~1 min smoke)
2. ./run.sh --quick-test (5+2 BO, ~2 min)
3. Full 80+300 if feasible >60%
4. Optional: tune Cp/Cm mapping to realize 0.58 if penalty still 11
