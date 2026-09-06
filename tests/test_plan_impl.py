"""Plan-implementation regression: sheer/rake, aft keel, full bulb, honest mass, JFR."""
import numpy as np

from hull_opt.config import load_config, design_vector_names
from hull_opt.geometry import (
    _sheer_height, _stem_rake_shift, _build_nurbs_control_net,
    compute_half_breadth_analytic, _build_bulb_patch, _tessellate_patches,
    _build_nurbs_patches,
)
from hull_opt.param_layer import flattened_bounds, design_vector_to_physical


def _xdict():
    return {"LWL": 2.5, "BWL": 0.55, "T_canoe": 0.36, "Cp": 0.6, "Cm": 0.78,
            "LCB": 55.0, "D_keel": 0.47, "keel_chord": 0.26, "bulb_vol": 0.006,
            "bulb_pos": 0.5, "E": 0.3, "flare": 10.0, "deadrise": 16.0,
            "bilge_r": 0.18, "keel_rake": 28.0, "ballast_frac": 0.45,
            "wingsail_pos": 0.3, "sheer_bow": 0.0, "sheer_stern": 0.0,
            "stem_rake_deg": 0.0, "forefoot_cut": 0.35}


def test_dim_21():
    assert len(design_vector_names()) == 21
    assert len(flattened_bounds()) == 21
    cfg = load_config("config.yaml")
    assert cfg.bounds.dim == 21
    raw = np.zeros(21)
    d = design_vector_to_physical(raw, config=cfg)
    assert set(d) == set(design_vector_names())
    # flat-deck directive: pinned bounds decode to exactly 0
    assert d["sheer_bow"] == 0.0
    assert d["sheer_stern"] == 0.0
    assert d["stem_rake_deg"] == 0.0
    # legacy vectors still decode (flat deck, full forefoot)
    d17 = design_vector_to_physical(np.zeros(17), config=cfg)
    assert d17["sheer_bow"] == 0.0 and abs(d17["forefoot_cut"]) < 1e-3
    d20 = design_vector_to_physical(np.zeros(20), config=cfg)
    assert abs(d20["forefoot_cut"]) < 1e-3


def test_sheer_flat_mid_kicked_ends():
    # pinned flat deck: all stations exactly E (legacy helper still supports kick if passed)
    assert _sheer_height(0.5, 0.3, 0.0, 0.0) == 0.3
    assert _sheer_height(0.0, 0.3, 0.0, 0.0) == 0.3
    assert _sheer_height(1.0, 0.3, 0.0, 0.0) == 0.3
    assert _sheer_height(0.0, 0.3, 0.08, 0.04) > 0.37


def test_flat_deck_stl_tolerance():
    # 0.001 mm flatness: deck edge z == E + sinkage at every station when
    # pinned (Bug #172-A: re-float shifts the rigid mesh; flatness is the
    # directive, absolute height follows buoyancy).
    import tempfile
    from hull_opt.geometry import generate_hull
    cfg = load_config("config.yaml")
    raw = np.zeros(21)
    stl, _, hydro, _ = generate_hull(raw, output_dir=tempfile.mkdtemp(), config=cfg,
                                     target_displacement=0.10)
    import trimesh
    m = trimesh.load(stl)
    d = design_vector_to_physical(raw, config=cfg)
    deck_z = m.bounds[1, 2]
    assert abs(float(deck_z) - (d["E"] + hydro.get("sinkage_m", 0.0))) <= 1e-6


def test_forefoot_cutaway():
    from hull_opt.geometry import _forefoot_factor, FOREFOOT_EXTENT
    assert FOREFOOT_EXTENT == 0.20
    assert _forefoot_factor(0.0, 0.35) == 0.65
    assert _forefoot_factor(0.20, 0.35) == 1.0
    assert _forefoot_factor(0.5, 0.35) == 1.0
    assert _forefoot_factor(0.0, 0.0) == 1.0
    # net vs analytic stay in sync with cut
    ctrl = _build_nurbs_control_net(_xdict())
    assert np.all(np.isfinite(ctrl))
    LWL = 2.5
    y_net_bow = ctrl[1, 4, 1]
    y_an = compute_half_breadth_analytic(np.array([(-0.5 + 1.0 / 13) * LWL]), np.array([-0.1]),
                                         _xdict(), LWL)
    assert y_an[0] > 0.0 and np.isfinite(y_an[0])


def test_stem_rake_topside_only():
    assert _stem_rake_shift(-0.1, 10.0) == 0.0
    assert _stem_rake_shift(0.0, 10.0) == 0.0
    assert _stem_rake_shift(0.3, 10.0) > 0.05


def test_net_analytic_agree_with_sheer():
    xd = _xdict()
    ctrl = _build_nurbs_control_net(xd)
    # flat deck: bow deck edge z == E exactly
    assert abs(ctrl[0, -1, 2] - xd["E"]) <= 1e-12
    # analytic capped above sheer
    LWL = xd["LWL"]
    y = compute_half_breadth_analytic(np.array([0.0]), np.array([5.0]), xd, LWL)
    assert y[0] == 0.0


def test_bulb_full_ellipsoid_closed():
    xd = _xdict()
    patch = _build_bulb_patch(xd)
    assert patch is not None
    xs = patch.control_net[:, :, 0]
    zs = patch.control_net[:, :, 2]
    # fore AND aft of center, top AND bottom of center
    assert xs.min() < xs.mean() < xs.max()
    assert zs.min() < zs.mean() < zs.max()
    # y>=0 half; tessellator mirrors to full
    assert patch.control_net[:, :, 1].min() >= -1e-12


def test_keel_aft_sweep():
    patches = _build_nurbs_patches(_xdict())
    keel = next(p for p in patches if p.name == "keel_port")
    # tip (vf=1) x should be AFT (+x) of root (vf=0) for +rake
    assert keel.control_net[:, -1, 0].mean() > keel.control_net[:, 0, 0].mean()


def test_ballast_fits_in_bulb():
    xd = _xdict()
    need = 127.5 * xd["ballast_frac"]
    cap = xd["bulb_vol"] * 11340
    assert cap >= need


def test_jfr_monitors_present():
    from hull_opt import constraints as C
    import inspect
    src = inspect.getsource(C.evaluate_constraints)
    for k in ["LDR", "L_over_B", "SA_over_D", "Wb_over_DT"]:
        assert k in src


def test_bulb_capacity_true_displacement_basis(caplog):
    # Bug #165: box-basis demand (~360 kg) phantom-flagged every design.
    # True basis: 0.10 m3 -> ~127.5 kg total; frac 0.45 -> ~57 kg <= 68 kg cap.
    import logging
    from hull_opt.geometry import generate_hull
    import tempfile
    cfg = load_config("config.yaml")
    raw = np.zeros(21)
    raw[8] = 8.0    # bulb_vol -> upper bound 0.0065
    raw[15] = -1.0  # ballast_frac -> ~0.40
    d = design_vector_to_physical(raw, config=cfg)
    need = d["ballast_frac"] * (0.10 * 1025.0 + 10.0 + 15.0)
    assert need <= d["bulb_vol"] * 11340 + 1e-9
    assert need < 75.0  # never the 117-200 kg box-basis phantom
    with caplog.at_level(logging.WARNING, logger="hull_opt.geometry"):
        generate_hull(raw, output_dir=tempfile.mkdtemp(), config=cfg,
                      target_displacement=0.10)
    assert "Bulb undersized" not in caplog.text


def test_honest_ballast_split_vcg():
    # Bug #166: over-capacity lead must ride mid-fin, not at the bulb point.
    from hull_opt.hydrostatics import _ballast_struct_split
    sp = _ballast_struct_split(127.5, 0.0031 * 11340, 2.0, 10.0, 15.0,
                               0.55, 0.0031, 1.823, 0.18, 20.0)
    assert sp["m_tip"] <= 0.0031 * 11340 + 1e-9
    assert sp["m_fin"] > 5.0  # 157-like: ~8.6 kg homeless at mid-fin
    assert sp["ballast"] == sp["m_tip"] + sp["m_fin"]
    # honest twin with max bulb holds everything at the tip
    sp2 = _ballast_struct_split(127.5, 0.0065 * 11340, 2.0, 10.0, 15.0,
                                0.35, 0.0065, 0.45, 0.26, 20.0)
    assert sp2["m_fin"] == 0.0


def test_struct_mass_calibration():
    # Bug #168: DERIVED cantilever (PYD Ch.13), carbon primary structure —
    # not the old tuned 0.035. Hand-check: M=tip*g*D*safety,
    # t=6M/(sigma*c^2), m=t*D*c*rho.
    from hull_opt.hydrostatics import _ballast_struct_split
    deep = _ballast_struct_split(127.5, 35.0, 2.0, 10.0, 15.0,
                                 0.55, 0.0031, 1.823, 0.18, 20.0, None)
    M = (35.0 + deep["ballast"]) * 9.81 * 1.823 * 2.0
    expect = 6.0 * M / (600e6 * 0.18 ** 2) * 1.823 * 0.18 * 1600.0
    assert abs(deep["m_struct"] - expect) < 1e-6
    assert 0.2 < deep["m_struct"] < 2.0  # carbon laminate is light; floors/bolts excluded (stated scope)
    shallow = _ballast_struct_split(127.5, 70.0, 2.0, 10.0, 15.0,
                                    0.35, 0.0065, 0.45, 0.26, 20.0, None)
    assert shallow["m_struct"] < 0.5


def test_draft_priced_not_walled():
    # Bug #169 (user directive): draft is inconvenience-with-price, not a
    # wall. No max_total_draft_m anywhere; logistics helper prices draft
    # over the ramp-free allowance linearly; shallow draft is free.
    from hull_opt.config import load_config
    from hull_opt.low_fidelity import draft_logistics_cost
    cfg = load_config("config.yaml")
    assert not hasattr(cfg.fixed, "max_total_draft_m")
    assert cfg.fixed.draft_free_m == 1.0
    deep = {"T_canoe": 0.36, "D_keel": 1.80}
    assert abs(draft_logistics_cost(deep, cfg) - 0.3 * (2.16 - 1.0)) < 1e-9
    shallow = {"T_canoe": 0.28, "D_keel": 0.45}
    assert draft_logistics_cost(shallow, cfg) == 0.0
    # ±50% rate sensitivity: monotonic, bounded (fair-weight probe)
    import copy
    hi = copy.deepcopy(cfg)
    object.__setattr__(hi.fixed, "draft_logistics_per_m", 0.45)
    lo = copy.deepcopy(cfg)
    object.__setattr__(lo.fixed, "draft_logistics_per_m", 0.15)
    assert draft_logistics_cost(deep, lo) < draft_logistics_cost(deep, cfg) < draft_logistics_cost(deep, hi)


def test_mission_bands_and_leeway():
    # Bug #169: band weights sum to 1; polar carries reach-drive for the
    # reach/run mission; leeway helper is zero at/under the 4° norm and
    # linear past it with ±50% sensitivity.
    import copy
    from hull_opt.config import load_config
    from hull_opt.low_fidelity import leeway_penalty_deg
    from hull_opt.balance import evaluate_balance_polar
    cfg = load_config("config.yaml")
    s = cfg.weights.band_light_wt + cfg.weights.band_medium_wt + cfg.weights.band_heavy_wt
    assert abs(s - 1.0) < 1e-9
    bal = evaluate_balance_polar(_xdict(), cfg, None, {}, tws_ops_kt=10.0)
    assert "reach_drive_ops_N" in bal
    assert np.isfinite(bal["reach_drive_ops_N"]) and bal["reach_drive_ops_N"] >= 0.0
    # 5 kt user max: band speeds ordered light < target < heavy=max.
    assert cfg.fixed.band_light_kt < cfg.fixed.target_speed_knots < cfg.fixed.band_heavy_kt
    assert cfg.fixed.band_heavy_kt == cfg.fixed.max_speed_knots == 5.0
    # Heavy band at max speed still solves (Fn ~0.53, past the barrier).
    heavy = evaluate_balance_polar(_xdict(), cfg, None, {}, tws_ops_kt=22.0,
                                   boat_speed_ms=5.0 * 0.514444)
    assert np.isfinite(heavy["worst_leeway_ops_deg"])
    assert leeway_penalty_deg(3.9, cfg) == 0.0
    assert leeway_penalty_deg(4.0, cfg) == 0.0
    assert abs(leeway_penalty_deg(8.0, cfg) - 0.5) < 1e-9
    hi = copy.deepcopy(cfg)
    object.__setattr__(hi.weights, "w_leeway", 0.75)
    lo = copy.deepcopy(cfg)
    object.__setattr__(lo.weights, "w_leeway", 0.25)
    assert leeway_penalty_deg(8.0, lo) < leeway_penalty_deg(8.0, cfg) < leeway_penalty_deg(8.0, hi)


def test_measured_beam_and_eff_draft():
    # Bug #172-B/A: physics prices the built mesh, re-floated draft.
    from hull_opt.hydrostatics import measured_beam, eff_draft
    xd = {"BWL": 0.55, "T_canoe": 0.30}
    assert measured_beam(xd, None) == 0.55
    assert measured_beam(xd, {}) == 0.55
    assert measured_beam(xd, {"BWL_measured": 0.41}) == 0.41
    assert measured_beam(xd, {"BWL_measured": -1.0}) == 0.55  # bad data: fallback
    assert eff_draft(xd, None) == 0.30
    assert abs(eff_draft(xd, {"sinkage_m": 0.06}) - 0.24) < 1e-9
    assert eff_draft(xd, {}) == 0.30


def test_mission_columns_reach_report(tmp_path):    # Bug #169 wiring: mission scalars persisted via constraint_values must
    # surface in results.md/CSV, or production runs fly blind on what the
    # FoM actually rewarded.
    import json
    from run_optimization import generate_results_report
    from hull_opt.config import load_config
    cfg = load_config("config.yaml")
    cv = {"mission_drive": 1.23, "gust_margin": 0.4, "heavy_leeway_deg": 5.0,
          "draft_logistics_cost": 0.35, "T_over_L": 0.32, "mission_fom": 6.25,
          "BWL_measured": 0.41}
    row = {"iter": 0, "feasible": 1, "fom": 2.5, "rt_total": 38.0,
           "constraint_values": json.dumps(cv), "constraint_violations": "[]",
           "physical_params": json.dumps(_xdict()), "design_vector": "[]",
           "gate_margins": json.dumps({})}

    class _DB:
        def get_all_designs(self):
            return [row]

    generate_results_report(_DB(), cfg, tmp_path)
    md = (tmp_path / "results.md").read_text()
    csv_text = (tmp_path / "results.csv").read_text()
    for col in ("mission_drive", "gust_margin", "heavy_leeway_deg",
                "draft_logistics_cost", "mission_fom", "BWL_measured"):
        assert col in md
        assert col in csv_text.splitlines()[0]
    assert "1.2300" in md  # value flows through, not just the header
    assert "6.2500" in md  # mission_fom from cv fallback


def test_sigma_deg_capped():
    # Bug #167: 2021 deg garbage -> 180 cap; sane values pass through.
    import math
    from hull_opt.rapid_gates import _sigma_deg_capped
    assert _sigma_deg_capped(math.radians(2021.6)) == 180.0
    assert _sigma_deg_capped(math.radians(45.0)) == 45.0
    assert _sigma_deg_capped(0.0) == 0.0


def test_storm_paddle_pocket_scales_with_hs():
    # Bug #167: back pocket must cover ~Hs piston travel, not fixed 0.35.
    from hull_opt.templates.dualsphysics import storm_domain
    d = storm_domain(2.49, 0.55, 0.74, 0.6, 0.06)
    pocket = -(-3.735) + d["x_paddle_lo"]
    assert d["x_paddle_lo"] > -3.735 + 0.6  # >= Hs clearance, was 0.35
    d2 = storm_domain(2.49, 0.55, 0.74, 0.1, 0.06)
    assert abs(d2["x_paddle_lo"] - (-3.735 + 0.35)) < 1e-9  # legacy pocket


def test_reserve_fatal_latches(monkeypatch):
    # Bug #171: an `else: reserve_fatal = False` wiped RE/self-right/accel
    # fatals whenever reserve passed. Fatal must latch: low-RE design with
    # passing reserve stays infeasible.
    import numpy as np
    from hull_opt.config import load_config
    from hull_opt import constraints as C
    cfg = load_config("config.yaml")
    angles = np.linspace(0, 180, 37)
    gz = np.full(37, 0.01)  # RE ~16 J < 40 J floor
    gz[angles >= 140] = 0.02  # self-righting passes in isolation
    vol = np.full(37, 0.10)
    curve = np.column_stack([angles, gz, vol])
    hydro = {"B": 0.55, "LWL": 2.4, "Cp": 0.58, "nabla": 0.10,
             "BM": 0.15, "target_nabla": 0.10, "rho": 1025.0}
    monkeypatch.setattr(C, "compute_reserve_buoyancy", lambda *a, **k: 0.50)
    from hull_opt.param_layer import design_vector_to_physical
    xd = {k: float(v) for k, v in
          design_vector_to_physical(np.zeros(21), config=cfg).items()}
    feasible, violations, _, _ = C.evaluate_constraints(
        hydro, curve, roll_period=5.0, peak_accel=10.0,
        x_dict=xd, config=cfg, stl_path="dummy.stl")
    assert not feasible
    assert any("righting_energy" in v for v in violations)


def test_jonswap_alpha_phillips():
    # Bug #171 (Bug #71 autopsy): NEITHER 5/16 nor 0.0081 is right for the
    # Hs^2 spectrum form (0.0081 belongs to the g^2 form) — so the spectrum
    # self-normalizes: variance must equal (Hs/4)^2 by definition of Hs.
    import numpy as np
    from hull_opt.rapid_gates import _jonswap_spectrum
    for Hs, Tp, gamma in ((2.5, 9.0, 3.3), (1.2, 7.0, 3.3), (10.0, 9.0, 1.0)):
        S, w = _jonswap_spectrum(Hs, Tp, gamma, 400, 0.05, 5.0)
        m0 = float(np.trapezoid(S, w))
        assert abs(m0 - (Hs / 4.0) ** 2) / (Hs / 4.0) ** 2 < 0.05


def test_avs_first_crossing():
    # Bug #171: dominant inverted lobe must not drag AVS to 180.
    import numpy as np
    from hull_opt.hydrostatics import compute_avs
    angles = np.linspace(0, 180, 37)
    gz = -0.05 * np.ones(37)
    gz[(angles >= 10) & (angles <= 110)] = 0.10 * np.sin(
        np.deg2rad((angles[(angles >= 10) & (angles <= 110)] - 10) / 100 * 180))
    gz[angles > 120] = 0.30  # huge upside-down lobe
    curve = np.column_stack([angles, gz, np.full(37, 0.10)])
    avs = compute_avs(curve)
    assert 105.0 < avs < 120.0, avs


def test_delft_cap_curve_and_true_cap():
    # Bug #171: flat 0.035 + 0.05*Rw leak replaced by PYD-band Fn curve.
    from hull_opt.michell import delft_cap_frac, capped_wave_resistance
    assert abs(delft_cap_frac(0.30) - 0.004) < 1e-9
    assert abs(delft_cap_frac(0.45) - 0.05) < 1e-9
    assert abs(delft_cap_frac(0.60) - 0.05) < 1e-9  # beyond validity: last band
    cap5 = 0.05 * 1025.0 * 9.81 * 0.10
    assert abs(capped_wave_resistance(1000.0, 0.10, cap_frac=0.05) - cap5) < 1e-6
    assert capped_wave_resistance(10.0, 0.10, cap_frac=0.05) == 10.0
    capd = 0.035 * 1025.0 * 9.81 * 0.10
    assert abs(capped_wave_resistance(1000.0, 0.10) - capd) < 1e-6


def test_fallback_bounds_match_config():
    # Bug #171: config=None path decoded a different boat. Only the
    # bulb geometric shrink may differ (documented, by design).
    import numpy as np
    from hull_opt.config import load_config
    from hull_opt.param_layer import design_vector_to_physical
    cfg = load_config("config.yaml")
    a = design_vector_to_physical(np.zeros(21), config=None)
    b = design_vector_to_physical(np.zeros(21), config=cfg)
    bad = [k for k in a if k != "bulb_vol" and abs(a[k] - b[k]) > 1e-9]
    assert bad == [], bad
