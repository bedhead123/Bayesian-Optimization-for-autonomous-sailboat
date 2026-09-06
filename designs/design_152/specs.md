# Design 152 — 2.5 m autonomous ocean-crossing sailboat (Bayesian optimization winner)

Photos: `152_profile.png` (broadside) · `152_bow_quarter.png` · `152_stern_quarter.png` · `152_top.png` · `152_underbody.png` — all rendered from `output/validation/design_152/hull_full.stl` (32,072 faces, hull + fin + bulb).

> **Provenance (read this before comparing files):** renders carry vertex hash
> `8df7e8809515`, byte-identical to `output/final_design_152.stl` and the
> validation mesh (max vertex diff 0.0). The validation folder also contains
> `hull.stl` (fin amputated — stability math only), a decimated BEM copy, and
> an upside-down `inverted_hull.stl` (deliberate test setup). Those are NOT
> the boat. See `validation/design_152/MANIFEST.md`.

## Headline numbers

| Item | Value |
|---|---|
| LWL / Beam / Hull draft | 2.50 m / 0.55 m / 0.30 m |
| Keel: depth / chord / rake / bulb | 0.82 m / 0.18 m / 15° aft / 3.1 L lead bulb |
| Total draft / T/L | 1.12 m / 0.45 |
| Displacement (target 100 kg) | ~100 kg, LDR 6.0 |
| Ballast 35% (2.2 kg tip lead, fits bulb) | GM 0.53 m, AVS 180°, righting energy 128 J |
| Rig | wingsail AR 3.5, NACA0018, 35% tail |
| Campaign score 3.893 → **recalculated 6.248 (#1 of 160)** after physics fixes |
| Mission breakdown | reach-drive 0.94, gust margin 0.77, heavy-air leeway 4.5°, draft logistics 0.04 |
| Validation (measured) | 5/5 PASS: calm-water Rt, storm accel 2.2 g, self-right 0.4 s, slam + inverted pressure |

## Why this one won

Scored over 3 wind bands (5/10/22 kt, reach/run weighted) for an ocean crossing: it pulls hard on a reach (drive 0.94 vs Rt 24 N), holds 4.5° leeway in a breeze (designer norm is 3–5°), keeps 0.77 gust heel margin, and self-rights in 0.4 s. Mid-depth fin: shallower than a racing yacht, deeper than a daysailer. This one pays 0.04 in logistics.

## Known limitations (be brutal — that's why I'm here)

1. Structures use a derived carbon root scantling with safety factor 2 — floors, bolts, grounding shoe excluded (stated lower bound).
2. Added-wave resistance and Delft caps are empirical regressions, capped past Fn 0.45 where our 5 kt band sails (Fn ~0.53).
3. No crew, no reefing model — feathered-storm survival only.
4. Single monohull study; no multihull comparison.
5. Scores assume smooth antifouled bottom ×1.25 fouling factor for long deployments.

Built with: Michell thin-ship + ITTC-57 + lifting-line appendages + Capytaine BEM + DualSPHysics SPH audit. Happy to share mesh/params for independent checks.
