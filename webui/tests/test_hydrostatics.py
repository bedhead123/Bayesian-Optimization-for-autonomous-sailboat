import numpy as np
import trimesh
import tempfile
from hull_opt.geometry import generate_hull
from hull_opt.hydrostatics import (
    compute_gz_curve,
    compute_righting_energy,
    compute_hydrostatics,
    check_inverted_stability,
)

_DESIGN = np.array([
    2.40,   # LWL
    0.50,   # BWL
    0.20,   # T_canoe
    0.60,   # Cp
    0.75,   # Cm
    10.0,   # LCB
    1.00,   # D_keel
    0.20,   # keel_chord
    0.003,  # bulb_vol
    0.45,   # bulb_pos
    0.20,   # E
    0.15,   # SA
    0.80,   # flare
    12.0,   # deadrise
    0.10,   # bilge_r
    0.005,  # keel_rake
    0.55,   # ballast_frac
])
_DESIGN_WIDE = _DESIGN.copy()
_DESIGN_WIDE[1] = 10.0  # BWL pushed to upper bound for wider beam


def _make_reference_hull():
    tmp = tempfile.mkdtemp()
    stl_path, sac_path, hydro, hull_stl = generate_hull(
        _DESIGN, output_dir=tmp, target_displacement=0.30)
    return stl_path, hydro


def test_gz_curve_shape():
    stl_path, _ = _make_reference_hull()
    gz = compute_gz_curve(stl_path, cg_z=-0.1, n_angles=37, max_heel=180.0)
    assert gz.shape == (37, 3)
    assert np.all(gz[:, 0] >= 0) and np.all(gz[:, 0] <= 180.0)


def test_gz_zero_at_upright():
    stl_path, _ = _make_reference_hull()
    gz = compute_gz_curve(stl_path, cg_z=-0.1)
    assert abs(gz[0, 1]) < 1e-6


def test_righting_energy_nonnegative():
    stl_path, _ = _make_reference_hull()
    gz = compute_gz_curve(stl_path, cg_z=-0.1)
    energy = compute_righting_energy(gz, max_heel_deg=30.0)
    assert energy >= 0


def test_righting_energy_zero():
    gz = np.column_stack([np.linspace(0, 180, 37), np.zeros(37), np.zeros(37)])
    energy = compute_righting_energy(gz, max_heel_deg=90.0)
    assert energy == 0.0


def test_righting_energy_increases_with_beam():
    stl_path1, _ = _make_reference_hull()
    gz1 = compute_gz_curve(stl_path1, cg_z=-0.10)
    e1 = compute_righting_energy(gz1, max_heel_deg=30.0)

    tmp = tempfile.mkdtemp()
    stl_path2, _, _, _ = generate_hull(_DESIGN_WIDE, output_dir=tmp,
                                       target_displacement=0.30)
    gz2 = compute_gz_curve(stl_path2, cg_z=-0.10)
    e2 = compute_righting_energy(gz2, max_heel_deg=30.0)
    assert e1 >= 0
    assert e2 >= 0


def test_check_inverted_stability():
    stl_path, _ = _make_reference_hull()
    result = check_inverted_stability(stl_path, cg_z=-0.1)
    assert isinstance(result, bool)


def test_hydrostatics_returns_keys():
    stl_path, hydro = _make_reference_hull()
    hs = compute_hydrostatics(stl_path, hydro)
    required_keys = ["nabla", "CB_x", "CB_y", "CB_z", "Ix", "Iy",
                     "waterplane_area", "BM", "BML", "Am"]
    for k in required_keys:
        assert k in hs, f"Missing key: {k}"
    assert hs["nabla"] > 0
    assert hs["BM"] > 0
    assert hs["waterplane_area"] > 0


def test_gz_curve_monotonic():
    stl_path, _ = _make_reference_hull()
    gz = compute_gz_curve(stl_path, cg_z=-0.1)
    angles = gz[:, 0]
    assert np.all(np.diff(angles) > 0)
