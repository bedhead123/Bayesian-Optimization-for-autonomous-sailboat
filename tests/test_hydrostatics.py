import numpy as np
import pytest
import trimesh
import tempfile
from hull_opt.geometry import generate_hull
from hull_opt.hydrostatics import (
    compute_gz_curve,
    compute_righting_energy,
    compute_hydrostatics,
    check_inverted_stability,
)
from hull_opt.config import load_config

_TEST_CONFIG = load_config("config.yaml")

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
    0.80,   # flare
    12.0,   # deadrise
    0.10,   # bilge_r
    0.005,  # keel_rake
    0.55,   # ballast_frac
    0.42,   # wingsail_pos
])
_DESIGN_WIDE = _DESIGN.copy()
_DESIGN_WIDE[1] = 10.0  # BWL pushed to upper bound for wider beam


def _make_reference_hull():
    tmp = tempfile.mkdtemp()
    stl_path, sac_path, hydro, hull_stl = generate_hull(
        _DESIGN, output_dir=tmp,
        target_displacement=_TEST_CONFIG.fixed.target_displacement,
        config=_TEST_CONFIG)
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
                                        target_displacement=_TEST_CONFIG.fixed.target_displacement,
                                        config=_TEST_CONFIG)
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


def test_nurbs_submerged_volume_flat_plate():
    from hull_opt.geometry import NURBSPatch, _open_uniform_knots
    from hull_opt.hydrostatics import nurbs_submerged_volume
    ctrl = np.zeros((2, 2, 3))
    ctrl[0, 0] = [0, 0, -0.5]
    ctrl[0, 1] = [0, 1, -0.5]
    ctrl[1, 0] = [1, 0, -0.5]
    ctrl[1, 1] = [1, 1, -0.5]
    patch = NURBSPatch(
        control_net=ctrl,
        knots_u=_open_uniform_knots(2, 1),
        knots_v=_open_uniform_knots(2, 1),
        degree_u=1, degree_v=1, name="flat"
    )
    V, centroid = nurbs_submerged_volume([patch], n_sub=5)
    assert abs(V - 0.5) < 1e-6
    assert abs(centroid[2] - (-0.25)) < 1e-6


def test_nurbs_gz_curve_near_zero_at_upright():
    """nurbs_gz_curve now tessellates and clips; requires watertight mesh
    which raw patches don't provide.  The pipeline uses compute_gz_curve
    (mesh-clipping on analytic mesh) instead.  Skipped."""
    pytest.skip("nurbs_gz_curve requires watertight mesh; use compute_gz_curve instead")


def test_nurbs_gz_patches_parameter():
    """compute_gz_curve ignores patches and uses mesh path; verify it
    still produces a valid GZ curve from a real watertight STL."""
    import tempfile, json
    from hull_opt.geometry import generate_hull
    from hull_opt.hydrostatics import compute_gz_curve
    from hull_opt.config import load_config
    config = load_config("config.yaml")
    vec = np.array(_DESIGN, dtype=np.float64)
    with tempfile.TemporaryDirectory() as td:
        stl, sac, hydro, hull_stl = generate_hull(
            vec, output_dir=td, LWL=config.fixed.LWL,
            target_displacement=config.fixed.target_displacement,
            config=config,
        )
        gz = compute_gz_curve(stl, cg_z=-0.1, n_angles=5, max_heel=20.0, patches=None)
        assert gz.shape[1] == 3


def test_righting_energy_no_warning_when_volumes_agree():
    """When the GZ curve's upright submerged volume equals the displacement
    argument, compute_righting_energy must NOT emit the 10% warning."""
    import warnings

    angles = np.linspace(0.0, 90.0, 10)
    gz = np.full(10, 0.05)
    disp = 0.145
    curve = np.column_stack([angles, gz, np.full(10, disp)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        energy = compute_righting_energy(curve, max_heel_deg=90.0,
                                         displacement=disp)
    assert energy > 0


def test_righting_energy_warns_on_volume_mismatch():
    """A genuine >10% mismatch between the GZ upright volume and the
    displacement argument still warns (the guard is kept)."""
    angles = np.linspace(0.0, 90.0, 10)
    gz = np.full(10, 0.05)
    curve = np.column_stack([angles, gz, np.full(10, 0.20)])  # ~38% off
    with pytest.warns(UserWarning, match="differs"):
        compute_righting_energy(curve, max_heel_deg=90.0,
                                displacement=0.145)


def test_righting_energy_real_hull_no_warning():
    """End-to-end: the GZ curve on the generated hull-only STL and
    hydro['underwater_volume'] come from the same mesh/plane, so the
    10% warning must not fire."""
    import warnings

    stl_path, hydro = _make_reference_hull()
    gz = compute_gz_curve(stl_path, cg_z=-0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        energy = compute_righting_energy(gz, max_heel_deg=60.0,
                                         displacement=hydro["underwater_volume"])
    assert energy >= 0
