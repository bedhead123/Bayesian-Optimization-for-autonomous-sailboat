import numpy as np
import pytest
import trimesh
import tempfile
from pathlib import Path
from hull_opt.geometry import (
    generate_hull,
    design_vector_to_dict,
    _waterline_half_breadth,
    _section_curve,
    _sac_form,
    _interp_param,
    _interp_bilge,
    _compute_hydrostatics,
)
from hull_opt.config import load_config

_TEST_CONFIG = load_config("config.yaml")

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
    0.80,   # flare
    12.0,   # deadrise
    0.10,   # bilge_r
    0.005,  # keel_rake
    0.55,   # ballast_frac
    0.42,   # wingsail_pos
])

# Raw-space design vector — DESIGN values are already valid raw-space [-10, +10]
RAW_DESIGN = DESIGN.copy()


def test_design_vector_to_dict():
    d = design_vector_to_dict(DESIGN)
    assert d["LWL"] == 2.40
    assert d["BWL"] == 0.50
    assert d["Cp"] == 0.60
    assert d["ballast_frac"] == 0.55
    assert d["wingsail_pos"] == 0.42
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
        stl_path, sac_path, hydro, hull_stl = generate_hull(RAW_DESIGN, output_dir=tmp, config=_TEST_CONFIG)
        mesh = trimesh.load(hull_stl)
        assert mesh.volume > 0
        assert mesh.body_count == 1
        assert hydro["B"] > 0.4
        assert hydro["Cp"] > 0.55


def test_generate_hull_with_keel():
    with tempfile.TemporaryDirectory() as tmp:
        stl_path, sac_path, hydro, hull_stl = generate_hull(RAW_DESIGN, output_dir=tmp, config=_TEST_CONFIG)
        mesh = trimesh.load(hull_stl)
        assert mesh.volume > 0
        assert mesh.body_count == 1


def test_generate_hull_no_keel():
    with tempfile.TemporaryDirectory() as tmp:
        stl_path, sac_path, hydro, hull_stl = generate_hull(RAW_DESIGN, output_dir=tmp, config=_TEST_CONFIG)
        mesh = trimesh.load(hull_stl)
        assert mesh.volume > 0
        assert mesh.body_count == 1


def test_generate_hull_sac_csv():
    with tempfile.TemporaryDirectory() as tmp:
        stl_path, sac_path, hydro, _ = generate_hull(RAW_DESIGN, output_dir=tmp, config=_TEST_CONFIG)
        sac_data = np.loadtxt(sac_path, delimiter=",", skiprows=1)
        assert sac_data.shape[1] == 2
        x_vals = sac_data[:, 0]
        assert np.all(x_vals >= 0)
        assert np.all(x_vals <= 3.0)


def test_nan_vertices_removed():
    with tempfile.TemporaryDirectory() as tmp:
        stl_path, _, _, _ = generate_hull(RAW_DESIGN, output_dir=tmp, config=_TEST_CONFIG)
        mesh = trimesh.load(stl_path)
        assert not np.any(np.isnan(mesh.vertices))


def test_hydrostatics_returns_all_keys():
    with tempfile.TemporaryDirectory() as tmp:
        _, _, hydro, _ = generate_hull(RAW_DESIGN, output_dir=tmp, config=_TEST_CONFIG)
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


def test_nurbs_patch_dataclass():
    from hull_opt.geometry import NURBSPatch, _open_uniform_knots
    ctrl = np.zeros((4, 4, 3))
    ctrl[0, 0] = [0, 0, 0]
    ctrl[0, -1] = [0, 1, 0]
    ctrl[-1, 0] = [1, 0, 0]
    ctrl[-1, -1] = [1, 1, 0]
    patch = NURBSPatch(
        control_net=ctrl,
        knots_u=_open_uniform_knots(4, 3),
        knots_v=_open_uniform_knots(4, 3),
        name="test"
    )
    assert patch.control_net.shape == (4, 4, 3)
    assert not patch.is_mirrored


def test_nurbs_mirror_patch():
    from hull_opt.geometry import NURBSPatch, _open_uniform_knots, _mirror_patch
    ctrl = np.zeros((4, 4, 3))
    ctrl[:, :, 1] = 0.5
    patch = NURBSPatch(
        control_net=ctrl,
        knots_u=_open_uniform_knots(4, 3),
        knots_v=_open_uniform_knots(4, 3),
        name="port"
    )
    mirror = _mirror_patch(patch)
    assert np.all(mirror.control_net[:, :, 1] == -0.5)
    assert mirror.is_mirrored


def test_tessellate_simple_patch():
    from hull_opt.geometry import NURBSPatch, _open_uniform_knots, _tessellate_patches
    ctrl = np.zeros((2, 2, 3))
    ctrl[0, 0] = [0, 0, 0]
    ctrl[0, 1] = [0, 1, 0]
    ctrl[1, 0] = [1, 0, -0.5]
    ctrl[1, 1] = [1, 1, -0.5]
    patch = NURBSPatch(
        control_net=ctrl,
        knots_u=_open_uniform_knots(2, 1),
        knots_v=_open_uniform_knots(2, 1),
        degree_u=1, degree_v=1,
        name="test"
    )
    mesh = _tessellate_patches([patch], dp=0.1)
    assert mesh is not None
    assert len(mesh.vertices) >= 4


def test_build_nurbs_patches():
    from hull_opt.geometry import _build_nurbs_patches
    from hull_opt.param_layer import design_vector_to_physical
    from hull_opt.config import load_config
    config = load_config("config.yaml")
    x_dict = design_vector_to_physical(RAW_DESIGN, config)
    patches = _build_nurbs_patches(x_dict)
    assert len(patches) >= 2
    names = [p.name for p in patches]
    assert "hull_port" in names
    assert "hull_port_mirror" in names


def test_tessellate_steep_deadrise_watertight():
    """Regression: design-36-style hull (steep deadrise) collapsed its lower
    stern band to bitwise-zero half-breadth, making the NURBS tessellation
    non-watertight regardless of vertex-merge digits.  The section-curve
    floor + digits_vertex=12 keep port/starboard distinct."""
    from hull_opt.geometry import _build_nurbs_patches, _tessellate_patches
    x_dict = {
        "LWL": 2.30, "BWL": 0.5057, "T_canoe": 0.2503,
        "Cp": 0.6433, "Cm": 0.8999, "LCB": 35.21,
        "D_keel": 0.9123, "keel_chord": 0.15, "bulb_vol": 0.001,
        "bulb_pos": 0.3172, "E": 0.0505,
        "flare": 8.01, "deadrise": 29.45, "bilge_r": 0.0526,
        "keel_rake": 0.0197, "ballast_frac": 0.3897,
    }
    patches = _build_nurbs_patches(x_dict)
    hull_patches = [p for p in patches if ("hull" in p.name or p.name == "deck") and not p.is_mirrored]
    mesh = _tessellate_patches(hull_patches, dp=0.03, require_watertight=True)
    assert mesh is not None
    assert mesh.is_watertight


def test_section_curve_never_exact_zero():
    """Regression: steep deadrise drove y_raw negative over the whole lower
    band of a section; clipping to exactly 0.0 collapsed that band to zero
    half-breadth in the NURBS control net."""
    z_norm = np.linspace(-1.0, 0.0, 11)
    section = _section_curve(z_norm, y_wl=0.25, T=0.2, deadrise=30.0, bilge_r=0.05, flare=8.0)
    assert np.all(section >= 1e-7)


def test_negative_hull_mass_raises():
    """Designs whose keel+bulb+mast alone exceed the displacement mass
    (residual budget < 0) must raise ValueError (stored as E_GEOM in the
    optimizer), not silently clamp hull mass to zero — clamping let
    SAC-gamed tiny hulls pass."""
    from hull_opt.geometry import _compute_hydrostatics

    mesh = trimesh.creation.box(extents=(2.4, 0.5, 0.3))
    mesh.apply_translation([0.0, 0.0, -0.15])
    with pytest.raises(ValueError, match="Negative hull mass"):
        _compute_hydrostatics(
            mesh, LWL=2.4, BWL=0.5, Cp=0.6, T_canoe=0.2, D_keel=1.0,
            bulb_vol=0.035, ballast_frac=0.30, keel_chord=0.2,
            target_displacement=0.145, bulb_density=11340.0,
        )


def test_ballast_no_longer_double_counts_against_hull():
    """ballast_frac must be a fraction of the residual budget
    (total - bulb - keel - mast), not of the total: a high ballast_frac with
    small appendages must NOT force a negative hull mass (double-count bug —
    over-ballasted small hulls were capsized into E_GEOM)."""
    from hull_opt.geometry import _compute_hydrostatics

    mesh = trimesh.creation.box(extents=(2.4, 0.5, 0.3))
    mesh.apply_translation([0.0, 0.0, -0.15])
    hydro = _compute_hydrostatics(
        mesh, LWL=2.4, BWL=0.5, Cp=0.6, T_canoe=0.2, D_keel=1.0,
        bulb_vol=0.0018, ballast_frac=0.95, keel_chord=0.2,
        target_displacement=0.145, bulb_density=11340.0,
    )
    assert hydro is not None
    assert hydro["ballast_mass_kg"] > 0.0
    assert hydro["hull_mass_kg"] > 0.0


def test_balanced_hull_mass_no_raise():
    """A normally-ballasted design computes a positive hull mass."""
    from hull_opt.geometry import _compute_hydrostatics

    mesh = trimesh.creation.box(extents=(2.4, 0.5, 0.3))
    mesh.apply_translation([0.0, 0.0, -0.15])
    hydro = _compute_hydrostatics(
        mesh, LWL=2.4, BWL=0.5, Cp=0.6, T_canoe=0.2, D_keel=1.0,
        bulb_vol=0.0005, ballast_frac=0.30, keel_chord=0.2,
        target_displacement=0.145, bulb_density=11340.0,
    )
    assert hydro is not None


def test_mesh_displacement_box():
    """A centered 1x1x1 box (z in [-0.5, 0.5]) displaces 0.5 below the
    waterplane at z=0; a waterplane above the box keeps the full volume,
    one below the box keeps none."""
    from hull_opt.geometry import mesh_displacement

    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    assert mesh_displacement(box) == pytest.approx(0.5, abs=1e-4)
    assert mesh_displacement(box, 0.5) == pytest.approx(1.0, abs=1e-4)
    assert mesh_displacement(box, 1.0) == pytest.approx(1.0, abs=1e-4)
    assert mesh_displacement(box, -0.5) == pytest.approx(0.0, abs=1e-4)
    assert mesh_displacement(box, -1.0) == pytest.approx(0.0, abs=1e-4)


def test_mesh_displacement_path_input(tmp_path):
    """mesh_displacement accepts an STL path string or Path."""
    from hull_opt.geometry import mesh_displacement

    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    stl = tmp_path / "box.stl"
    box.export(str(stl))
    assert mesh_displacement(str(stl)) == pytest.approx(0.5, abs=1e-4)
    assert mesh_displacement(Path(stl)) == pytest.approx(0.5, abs=1e-4)


def test_mesh_displacement_scene():
    """mesh_displacement accepts a trimesh.Scene (dumps to a single mesh)."""
    from hull_opt.geometry import mesh_displacement

    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    scene = trimesh.Scene([box])
    assert mesh_displacement(scene) == pytest.approx(0.5, abs=1e-4)


def test_mesh_displacement_degenerate():
    """Empty/degenerate inputs return 0.0 (never raise)."""
    from hull_opt.geometry import mesh_displacement

    assert mesh_displacement(None) == 0.0
    flat = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                           faces=[[0, 1, 2]])
    assert mesh_displacement(flat) == 0.0
    empty = trimesh.creation.box(extents=[1, 1, 1])
    empty.vertices = np.zeros((0, 3))
    assert mesh_displacement(empty) == 0.0


def test_mesh_displacement_nonzero_waterplane():
    """The waterplane z is honored: raising/lowering it shifts the
    displaced volume continuously."""
    from hull_opt.geometry import mesh_displacement

    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    assert mesh_displacement(box, 0.25) == pytest.approx(0.75, abs=1e-4)
    assert mesh_displacement(box, -0.25) == pytest.approx(0.25, abs=1e-4)


def test_sac_volume_far_below_target_raises_edisp():
    """A hull whose SAC capacity can never reach the target (raw capacity
    far below target; the 1.30 SAC cap cannot make up the difference) must
    be rejected with the E_DISP token, not merely warned."""
    tiny = np.full(17, -10.0)  # all params at lower bounds → tiny hull
    # Rebalanced survive-not-thrive: lower E/deadrise bounds shift the
    # smallest hull to ~0.156 m³ (cap 1.30), so 0.18 no longer forces
    # E_DISP (ratio 0.87 > 0.75). Use 0.25 to guarantee far-below rejection.
    with pytest.raises(ValueError, match="E_DISP"):
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(tiny, output_dir=tmp, config=_TEST_CONFIG,
                          target_displacement=0.25)


def test_sac_volume_near_miss_warns_not_raises( caplog):
    """Near-miss designs (>= 75% of target but still below) keep the plain
    warning and are NOT rejected."""
    from hull_opt.geometry import _compute_hydrostatics
    import logging

    mesh = trimesh.creation.box(extents=(2.4, 0.5, 0.3))
    mesh.apply_translation([0.0, 0.0, -0.15])
    with caplog.at_level(logging.WARNING, logger="hull_opt.geometry"):
        hydro = _compute_hydrostatics(
            mesh, LWL=2.4, BWL=0.5, Cp=0.6, T_canoe=0.2, D_keel=1.0,
            bulb_vol=0.0005, ballast_frac=0.30, keel_chord=0.2,
            sac_volume=0.13, target_displacement=0.145, bulb_density=11340.0,
        )
    assert hydro is not None
    assert hydro["nabla"] == pytest.approx(0.36, rel=1e-3)
    assert any("near-miss" in r.message for r in caplog.records)


def test_hydro_nabla_matches_mesh_displacement():
    """hydro['nabla'] and hydro['underwater_volume'] must equal the
    canonical mesh_displacement of the hull-only mesh at z=0."""
    from hull_opt.geometry import mesh_displacement

    with tempfile.TemporaryDirectory() as tmp:
        _, _, hydro, hull_stl = generate_hull(RAW_DESIGN, output_dir=tmp, config=_TEST_CONFIG)
        mesh = trimesh.load(hull_stl)
        expected = mesh_displacement(mesh, 0.0)
        assert hydro["nabla"] == pytest.approx(expected, rel=1e-6)
        assert hydro["underwater_volume"] == pytest.approx(expected, rel=1e-6)


# ── Topside flare (above-waterline) tests ──────────────────────────────
# The topside wall between the waterline edge and the deck edge leans
# inboard at flare·TOPSIDE_BOW_AMP (strongest at the bow, capped at
# TOPSIDE_MAX_DEG), with a rounded bow deck floor.  These tests pin the
# shape and the net↔analytic consistency so underwater physics can't be
# silently disturbed by future flare edits.

_FLARE_X_DICT = {
    "LWL": 2.4, "BWL": 0.60, "T_canoe": 0.29, "Cp": 0.58, "Cm": 0.82,
    "LCB": 45.0, "D_keel": 1.2, "keel_chord": 0.20, "bulb_vol": 0.001,
    "bulb_pos": 0.40, "E": 0.35, "flare": 25.0, "deadrise": 28.0,
    "bilge_r": 0.20, "keel_rake": 0.01, "ballast_frac": 0.65,
    "wingsail_pos": 0.42,
}


def test_topside_flare_slopes_inward():
    """Deck edge must be wider than the waterline at bow, mid, and stern
    stations — the hull topsides slope inboard-down to the waterline."""
    from hull_opt.geometry import _build_nurbs_control_net, _bow_deck_floor
    ctrl = _build_nurbs_control_net(_FLARE_X_DICT)
    for xn in (1 / 12, 0.25, 0.5, 0.75):
        i = int(round(xn * 12))
        zs = ctrl[i, :, 2]
        y_uw = ctrl[i, :, 1][zs <= 0].max()
        y_deck = ctrl[i, -1, 1]
        assert y_deck > y_uw, f"station {xn}: deck {y_deck:.4f} <= waterline {y_uw:.4f}"
    # Rounded stem: the bow deck edge keeps the deck floor, not a knife blade
    assert ctrl[1, -1, 1] >= float(_bow_deck_floor(1 / 12, 0.60))
    # Deck edge sanity: no wild jumps (stem point excluded — it closes to
    # y=0 by construction), deck never balloons beyond 2x half-beam
    deck_ys = ctrl[:, -1, 1]
    assert np.all(deck_ys <= 2.0 * (_FLARE_X_DICT["BWL"] / 2.0) + 1e-12)
    assert np.all(np.abs(np.diff(deck_ys[1:-1])) < 0.10)


def test_topside_control_net_matches_analytic():
    """The analytic half-breadth (used by Michell/BEM/SAC) must reproduce
    the control-net topside exactly, or drag is computed on a different
    hull than the mesh."""
    from hull_opt.geometry import _build_nurbs_control_net, compute_half_breadth_analytic
    ctrl = _build_nurbs_control_net(_FLARE_X_DICT)
    for i in (1, 3, 6, 9):
        for j in range(1, 9):
            if ctrl[i, j, 2] <= 0.0:
                continue
            xq = ctrl[i, j, 0] - 2.4 / 2  # analytic frame is centered
            ya = compute_half_breadth_analytic(
                np.array([xq]), np.array([ctrl[i, j, 2]]),
                _FLARE_X_DICT, LWL=2.4)[0]
            assert abs(ya - ctrl[i, j, 1]) < 1e-9, \
                f"net ({ctrl[i, j, 1]:.6f}) vs analytic ({ya:.6f}) at i={i}, j={j}"


def test_topside_underwater_rows_use_section_curve():
    """Below-waterline rows must stay on the section curve (deadrise/bilge/
    flare) — the flare change is strictly above the waterline, so
    displacement, waterplane, and GZ physics at a given flare are unchanged."""
    from hull_opt.geometry import _build_nurbs_control_net, _section_curve
    ctrl = _build_nurbs_control_net(_FLARE_X_DICT)
    BWL = _FLARE_X_DICT["BWL"]
    for i, xn in enumerate(np.linspace(0.0, 1.0, 13)):
        if xn <= 0.0 or xn >= 1.0 - 1e-12:
            continue
        T_local = _FLARE_X_DICT["T_canoe"] * (1.0 - 0.3 * (1.0 - xn))
        for j in range(1, 9):
            if ctrl[i, j, 2] > 0.0:
                continue
            y_wl = _waterline_half_breadth(np.array([xn]), BWL,
                                           _FLARE_X_DICT["Cp"],
                                           Cm=_FLARE_X_DICT["Cm"],
                                           LCB=_FLARE_X_DICT["LCB"])[0]
            dr = _interp_param(xn, _FLARE_X_DICT["deadrise"])
            br = _interp_bilge(xn, _FLARE_X_DICT["bilge_r"])
            fl = _interp_param(xn, _FLARE_X_DICT["flare"], bow_factor=1.0,
                               stern_factor=0.4)
            yf = _section_curve(np.array([ctrl[i, j, 2] / T_local]),
                                y_wl, T_local, dr, br, fl)
            assert abs(ctrl[i, j, 1] - y_wl * float(yf[0])) < 1e-9


def test_extreme_flare_hull_valid_and_watertight():
    """Max flare (35°) with max bow amplification must pass the control-net
    curvature check and produce a watertight STL."""
    from hull_opt.geometry import _build_nurbs_control_net, _check_control_net_curvature
    xd = dict(_FLARE_X_DICT)
    xd["flare"] = 35.0
    ctrl = _build_nurbs_control_net(xd)
    ok, msg = _check_control_net_curvature(ctrl)
    assert ok, msg
    raw = RAW_DESIGN.copy()
    raw[11] = 10.0  # flare → ~upper bound (35°)
    with tempfile.TemporaryDirectory() as tmp:
        stl_path, _, _, _ = generate_hull(raw, output_dir=tmp, config=_TEST_CONFIG)
        mesh = trimesh.load(stl_path)
        assert mesh.is_watertight
        assert len(mesh.faces) > 500
