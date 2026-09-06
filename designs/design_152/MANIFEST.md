# validation/design_152 — file manifest

Authoritative boat: **`hull_full.stl`** (32072 faces, vertex hash `8df7e8809515`).

| File | What it is |
|---|---|
| `hull_full.stl` | THE boat (hull + fin + bulb, bow x=0). Renders, CAD, CFD/SPH. |
| `hull.stl` / `hull_geometry.stl` | Hull shell ONLY — GZ/stability math, NOT the boat. |
| `hull_geometry_bem.stl` | Decimated hull-only — BEM speed mesh. |
| `gate5_inverted/inverted_hull.stl` | Upside-down copy — deliberate test setup. |
| `*.nurbs`, `sac.csv` | build/audit data. |

Do not compare renders across rows of this table — compare hashes.
