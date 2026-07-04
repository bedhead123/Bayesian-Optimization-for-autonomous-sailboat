# Hull-Keel Design Optimization Pipeline

Autonomous optimization of a **2.4m sailing hull with keel and bulbous bow** using Bayesian Optimization with multi-fidelity analysis (analytic → RANS CFD → SPH validation).

## Quick Start

```bash
pip install -r requirements.txt
python run_optimization.py --dry-run         # Validate setup
python run_optimization.py --quick-test      # 5 LHS + 2 BO (2 min)
python run_optimization.py --hyper-test      # 3 LHS + 0 BO (1 min)
python run_optimization.py --medium-test     # Full component test
python run_optimization.py                   # Full optimization (80+300 iters)
```

## CLI Reference

| Flag | Description |
|------|-------------|
| `--config PATH` | YAML config file (default: `config.yaml`) |
| `--dry-run` | Validate config, tools, geometry; no DB writes |
| `--quick-test` | 5 LHS + 2 BO iterations |
| `--hyper-test` | 3 LHS, 0 BO, 1 validation design |
| `--medium-test` | Thorough component test at reduced resolution |
| `--resume` | Resume from existing database |
| `--validate-only` | Run validation gates on DB top-3 designs |

## Directory Layout

```
├── run_optimization.py      # CLI entry point
├── config.yaml              # Full optimization config
├── config.fast.yaml         # Fast-test config variant
├── requirements.txt         # Python dependencies
├── hull_opt/                # Core library
│   ├── geometry.py          # Parametric hull generation
│   ├── hydrostatics.py      # GZ curves, righting energy
│   ├── michell.py           # Wave resistance (Michell integral)
│   ├── friction.py          # ITTC-57 friction line
│   ├── constraints.py       # Design feasibility checks
│   ├── low_fidelity.py      # Full analytic evaluation pipeline
│   ├── mid_fidelity.py      # OpenFOAM RANS calibration
│   ├── high_fidelity.py     # 6-gate validation suite
│   ├── surrogate.py         # BoTorch Bayesian Optimization
│   ├── config.py            # YAML → frozen dataclass
│   ├── database.py          # SQLite storage
│   ├── utils.py             # LHS, OF runner, memory mgmt
│   └── check_system.py      # System validation script
├── tests/                   # pytest unit tests (11 files)
├── webui/                   # FastAPI web dashboard
├── scripts/                 # Installation scripts
├── docs/                    # Detailed documentation
└── bin/                     # DualSPHysics binaries
```

## 17 Design Parameters

| # | Param | Range | Description |
|---|-------|-------|-------------|
| 1 | LWL | 2.30–2.50 m | Waterline length |
| 2 | BWL | 0.40–0.60 m | Waterline beam |
| 3 | T_canoe | 0.15–0.35 m | Hull depth (canoe body) |
| 4 | Cp | 0.55–0.65 | Prismatic coefficient |
| 5 | Cm | 0.60–0.90 | Midship coefficient |
| 6 | LCB | 5.0–20.0 % | Longitudinal CB position |
| 7 | D_keel | 0.85–1.20 m | Keel depth |
| 8 | keel_chord | 0.15–0.25 m | Keel chord length |
| 9 | bulb_vol | 0.001–0.05 m³ | Bulb volume |
| 10 | bulb_pos | 0.30–0.50 | Bulb position (fraction of LWL) |
| 11 | E | 0.15–0.30 m | Sheer height at midship |
| 12 | SA | 0.05–0.25 | Sheer angle factor |
| 13 | flare | 0.50–1.20 | Flare angle (deg) multiplier |
| 14 | deadrise | 5.0–25.0° | Deadrise angle |
| 15 | bilge_r | 0.05–0.30 m | Bilge radius |
| 16 | keel_rake | 0.001–0.02 | Keel rake (sweep) |
| 17 | ballast_frac | 0.30–0.70 | Ballast fraction |

## Multi-Fidelity Pipeline

```
Design Vector (17 params)
    │
    ▼
┌─────────────────────┐
│  Geometry Generation │  ← generate_hull() → STL mesh + SAC
│  (trimesh)          │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Low-Fidelity       │  ← Michell wave res. + ITTC-57 friction
│  (analytic)         │     + GZ curve + Capytaine RAOs
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  GP Surrogate       │  ← BoTorch SingleTaskGP + LogEI
│  (Bayesian Opt)     │     LHS → BO loop → convergence
└─────────┬───────────┘
          │ (every N iters)
          ▼
┌─────────────────────┐
│  Mid-Fidelity       │  ← OpenFOAM RANS calibration
│  (OpenFOAM)         │     drag correction Δ
└─────────┬───────────┘
          │ (top 3 designs)
          ▼
┌─────────────────────┐
│  High-Fidelity      │  ← 6 validation gates
│  (Validation)       │     CFD + SPH + hydrostatics
└─────────────────────┘
```

## Dependencies

- **Python**: numpy, scipy, trimesh, pyyaml, torch, botorch, gpytorch, capytaine, ray
- **External**: OpenFOAM v2512, DualSPHysics v5.4
- See `requirements.txt` for exact versions
