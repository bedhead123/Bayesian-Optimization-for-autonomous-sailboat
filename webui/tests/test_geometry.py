import numpy as np
import trimesh
import tempfile
from pathlib import Path
from hull_opt.geometry import (
    generate_hull,
    design_vector_to_dict,
    _waterline_half_breadth,
    _section_curve,
    _sac_form,
    _make_keel,
    _make_bulb,
    _interp_param,
    _interp_bilge,
    _compute_hydrostatics,
)

DESIGN = np.array([
    2.40,   # LWL
    0.50,   # BWL (within config bounds [0.40, 0.60])
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

# Raw-space design vector — DESIGN values are already valid raw-space [-10, +10]
RAW_DESIGN = DESIGN.copy()


def test_design_vector_to_dict():
    d = design_vector_to_dict(DESIGN)
    assert d["LWL"] == 2.40
    assert d["BWL"] == 0.50
    assert d["Cp"] == 0.60
    assert d["ballast_frac"] == 0.55
    assert len(d) == 17


def test_waterline_half_breadth():
    x_norm = np.linspace(0, 1, 21)
    y = _waterline_half_breadth(x_norm, BWL=0.5, Cp=0.60, Cm=0.75)
    assert np.all(y >= 0)
    assert np.all(y <= 0.25)
    assert np.isclose(y.argmax() / 20, 0.45, atol=0.06)


def test_sac_form():
    x_norm = np.linspace(-1, 1, 31)
    sac = _sac_form(x_norm, Cp=0.55)
    assert np.all(sac >= 0)
    assert np.all(sac <= 1 + 1e-10)
    assert sac[len(sac) // 2] == 1.0


def test_section_curve():
    z_norm = np.linspace(-1, 0, 11)
    section = _section_curve(z_norm, y_wl=0.25, T=0.2, deadrise=10.0, bilge_r=0.05, flare=15.0)
    assert np.all(section >= 0)
    assert section[0] >= 0  # keel has non-zero half-breadth (deadrise + flare)
    assert abs(section[-1] - 1.0) < 0.1  # waterline ~ full beam


def test_interp_param():
    v = _interp_param(0.3, mid_val=10.0, bow_factor=1.5, stern_factor=3.0)
    assert v > 0
    v_mid = _interp_param(0.5, mid_val=10.0)
    assert abs(v_mid - 10.0) < 1e-6


def test_interp_bilge():
    v = _interp_bilge(0.3, mid_val=0.05, end_factor=0.3)
    assert v > 0
    v_mid = _interp_bilge(0.5, mid_val=0.05)
    assert abs(v_mid - 0.05) < 1e-6


def test_generate_hull_produces_valid_mesh():
    with tempfile.TemporaryDirectory() as tmp:
        stl_path, sac_path, hydro, hull_stl = generate_hull(RAW_DESIGN, output_dir=tmp)
        mesh = trimesh.load(hull_stl)
        assert mesh.volume > 0
        assert mesh.body_count == 1
        assert hydro["B"] > 0.4
        assert hydro["Cp"] > 0.55


def test_generate_hull_with_keel():
    with tempfile.TemporaryDirectory() as tmp:
        stl_path, sac_path, hydro, hull_stl = generate_hull(RAW_DESIGN, output_dir=tmp)
        mesh = trimesh.load(hull_stl)
        assert mesh.volume > 0
        assert mesh.body_count == 1


def test_generate_hull_no_keel():
    with tempfile.TemporaryDirectory() as tmp:
        stl_path, sac_path, hydro, hull_stl = generate_hull(RAW_DESIGN, output_dir=tmp)
        mesh = trimesh.load(hull_stl)
        assert mesh.volume > 0
        assert mesh.body_count == 1


def test_generate_hull_sac_csv():
    with tempfile.TemporaryDirectory() as tmp:
        stl_path, sac_path, hydro, _ = generate_hull(RAW_DESIGN, output_dir=tmp)
        sac_data = np.loadtxt(sac_path, delimiter=",", skiprows=1)
        assert sac_data.shape[1] == 2
        x_vals = sac_data[:, 0]
        assert np.all(x_vals >= 0)
        assert np.all(x_vals <= 3.0)


def test_make_keel():
    keel = _make_keel(chord=0.2, depth=1.0, thickness=0.03,
                      x_pos=0.8, sweep_deg=12.0, tc=0.12)
    assert keel.volume > 0
    assert keel.is_watertight


def test_make_bulb():
    bulb = _make_bulb(chord=0.2, depth=1.0, volume=0.003,
                      ar=4.0, sweep_deg=12.0, x_pos=0.8)
    assert bulb.volume > 0


def test_nan_vertices_removed():
    with tempfile.TemporaryDirectory() as tmp:
        stl_path, _, _, _ = generate_hull(RAW_DESIGN, output_dir=tmp)
        mesh = trimesh.load(stl_path)
        assert not np.any(np.isnan(mesh.vertices))


def test_hydrostatics_returns_all_keys():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, hydro, _ = generate_hull(RAW_DESIGN, output_dir=tmp)
        for key in ["nabla", "CB_x", "CB_y", "CB_z", "BM", "Cp", "B",
                     "LWL", "cg_z", "D_keel", "total_mass_kg", "bulb_mass_kg"]:
            assert key in hydro, f"Missing key: {key}"
        assert hydro["total_mass_kg"] > 0


def test_compute_half_breadth_analytic():
    from hull_opt.geometry import compute_half_breadth_analytic
    x_dict = design_vector_to_dict(DESIGN)
    xq = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    zq = np.array([-0.05, -0.10, -0.15, -0.10, -0.05])
    y = compute_half_breadth_analytic(xq, zq, x_dict, LWL=2.4)
    assert y.shape == (5,)
    assert np.all(y >= 0)


def test_compute_half_breadth_analytic_outside_domain():
    from hull_opt.geometry import compute_half_breadth_analytic
    x_dict = design_vector_to_dict(DESIGN)
    xq = np.array([[-2.0], [3.0]])
    zq = np.array([[0.5], [0.5]])
    y = compute_half_breadth_analytic(xq, zq, x_dict, LWL=2.4)
    assert np.all(y == 0.0)


def test_compute_half_breadth_analytic_vectorized():
    from hull_opt.geometry import compute_half_breadth_analytic
    x_dict = design_vector_to_dict(DESIGN)
    xq = np.linspace(0, 2.4, 10)
    zq = np.full_like(xq, -0.1)
    y = compute_half_breadth_analytic(xq, zq, x_dict, LWL=2.4)
    assert y.shape == (10,)
    assert np.all(y >= 0)
    assert np.any(y > 0)  # at least some points on the hull
