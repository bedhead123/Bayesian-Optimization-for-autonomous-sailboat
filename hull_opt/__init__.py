"""
Hull-Keel Design Optimization Pipeline.
Parametric hull generation → multi-fidelity analysis → Bayesian Optimization.

Submodules:
  config.py        - YAML config → frozen dataclass
  geometry.py      - 17-param hull → STL mesh
  hydrostatics.py  - GZ curves, righting energy, CG
  michell.py       - Michell integral wave resistance
  friction.py      - ITTC-57 friction line
  constraints.py   - Design feasibility constraints
  low_fidelity.py  - Full analytic evaluation pipeline
  mid_fidelity.py  - OpenFOAM RANS calibration
  high_fidelity.py - 6-gate validation suite
  surrogate.py     - BoTorch Bayesian Optimization
  database.py      - SQLite results storage
  utils.py         - LHS sampling, OF runner, memory mgmt
  check_system.py  - System validation script
"""
from hull_opt.config import load_config, Config
from hull_opt.database import OptimizationDatabase
from hull_opt.geometry import generate_hull, design_vector_to_dict
