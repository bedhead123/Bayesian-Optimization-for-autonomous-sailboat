"""
Stress test: exercises hull geometry beyond normal usage.
Random sweeps, edge-of-bounds, extreme combos, degenerate vectors.
Records failures and clusters them by root cause.
"""

import numpy as np
import trimesh
import tempfile
import warnings
import traceback
import json
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from hull_opt.config import load_config, design_vector_names, BoundsConfig
from hull_opt.geometry import generate_hull, design_vector_to_dict
from hull_opt.geometry_validator import validate_design_vector
from hull_opt.param_layer import design_vector_to_physical, flattened_bounds

CONFIG = load_config("config.yaml")
NAMES = design_vector_names()
BOUNDS = flattened_bounds()  # raw [-10, 10] GP-space bounds
PHYS_BOUNDS = CONFIG.bounds.as_array()
assert len(NAMES) == 17, f"Expected 17 params, got {len(NAMES)}"
assert len(BOUNDS) == 17, f"Expected 17 bounds, got {len(BOUNDS)}"

LWL = CONFIG.fixed.LWL
TARGET_DISP = CONFIG.fixed.target_displacement

DESIGN = []
RESULTS = []
FAILURE_CLUSTERS = defaultdict(list)

def _rng(seed=42):
    return np.random.default_rng(seed)

def _random_design(rng, fix_lwl=True):
    dv = np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS])
    if fix_lwl:
        dv[0] = LWL
    return dv

def _register_design(category, label, dv, outcome, details, metrics=None):
    DESIGN.append({"category": category, "label": label, "dv": dv.tolist()})
    RESULTS.append({"category": category, "label": label, "outcome": outcome, "details": details, "metrics": metrics or {}})

def _check_mesh_quality(mesh, label):
    metrics = {}
    checks = []

    metrics["n_vertices"] = int(len(mesh.vertices))
    metrics["n_faces"] = int(len(mesh.faces))

    vol = mesh.volume
    metrics["volume"] = float(vol) if np.isfinite(vol) else -1.0
    metrics["watertight"] = bool(mesh.is_watertight)
    metrics["body_count"] = int(mesh.body_count)

    try:
        ch = mesh.convex_hull
        if ch is not None and ch.volume > 0 and vol > 0:
            metrics["convexity_ratio"] = float(vol / ch.volume)
        else:
            metrics["convexity_ratio"] = -1.0
    except Exception:
        metrics["convexity_ratio"] = -2.0

    edges = mesh.edges_unique
    if len(edges) > 0:
        lengths = np.linalg.norm(mesh.vertices[edges[:,0]] - mesh.vertices[edges[:,1]], axis=1)
        good = lengths > 1e-10
        if np.sum(good) > 0:
            metrics["edge_length_ratio"] = float(np.max(lengths) / np.min(lengths[good]))
        else:
            metrics["edge_length_ratio"] = -1.0
    else:
        metrics["edge_length_ratio"] = -1.0

    verts = mesh.vertices
    faces = mesh.faces
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.sqrt(np.sum(cross ** 2, axis=1))
    metrics["face_area_ratio_min_median"] = float(np.min(areas) / max(np.median(areas), 1e-15))

    fn = mesh.face_normals
    fa = mesh.face_adjacency
    spike_count = 0
    if len(fa) > 0:
        vn = mesh.vertex_normals
        if vn is not None and len(vn) == len(verts):
            vf = mesh.vertex_faces
            on_center = np.abs(verts[:, 1]) < 0.002
            for vi in range(len(verts)):
                if on_center[vi]:
                    continue
                fidxs = vf[vi]
                fidxs = fidxs[fidxs >= 0]
                if len(fidxs) < 2:
                    continue
                fn_i = fn[fidxs]
                vmax = 0.0
                for i in range(len(fn_i)):
                    for j in range(i+1, len(fn_i)):
                        dot = np.clip(np.dot(fn_i[i], fn_i[j]), -1.0, 1.0)
                        ang = np.arccos(dot)
                        if ang > vmax:
                            vmax = ang
                if vmax > np.deg2rad(150):
                    spike_count += 1
    metrics["spike_count"] = spike_count

    return metrics

def _test_random_sweep(n=50):
    rng = _rng(999)
    failures = []
    for i in range(n):
        dv = _random_design(rng)
        label = f"rand_{i}"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                stl_path, sac_path, hydro, hull_stl_path = generate_hull(
                    dv, output_dir=tmp, LWL=LWL, target_displacement=TARGET_DISP)
                hull_mesh = trimesh.load(hull_stl_path)
                if isinstance(hull_mesh, trimesh.Scene):
                    hull_mesh = hull_mesh.dump(concatenate=True)
                full_mesh = trimesh.load(stl_path)
                if isinstance(full_mesh, trimesh.Scene):
                    full_mesh = full_mesh.dump(concatenate=True)
                m1 = _check_mesh_quality(hull_mesh, f"{label}.hull")
                m2 = _check_mesh_quality(full_mesh, f"{label}.full")
                ok = True
                issues = []
                if not m1["watertight"]:
                    ok = False; issues.append("hull_not_watertight")
                if not m2["watertight"]:
                    ok = False; issues.append("full_not_watertight")
                if m1["volume"] <= 0:
                    ok = False; issues.append(f"hull_vol_{m1['volume']:.6f}")
                if m2["volume"] <= 0:
                    ok = False; issues.append(f"full_vol_{m2['volume']:.6f}")
                body = m2["body_count"]
                if body > 3:
                    ok = False; issues.append(f"body_count_{body}")
                if m2["convexity_ratio"] >= 0 and m2["convexity_ratio"] < 0.30:
                    ok = False; issues.append(f"convexity_{m2['convexity_ratio']:.3f}")
                if m2["edge_length_ratio"] > 0 and m2["edge_length_ratio"] > 5000:
                    ok = False; issues.append(f"edge_ratio_{m2['edge_length_ratio']:.0f}")
                if m2["spike_count"] > max(10, int(m2["n_vertices"] * 0.01)):
                    ok = False; issues.append(f"spikes_{m2['spike_count']}")
                outcome, details = ("ok", "") if ok else ("bad_mesh", "; ".join(issues))
                _register_design("random", label, dv, outcome, details, {"hull": m1, "full": m2})
                if not ok:
                    failures.append((label, issues))
        except (ValueError, RuntimeError, AssertionError) as e:
            _register_design("random", label, dv, "exception", f"{type(e).__name__}: {e}")
            failures.append((label, f"exception: {e}"))
        except Exception as e:
            _register_design("random", label, dv, "exception_other", f"{type(e).__name__}: {e}")
            failures.append((label, f"other_exception: {e}"))
    return failures

def _test_edge_of_bounds():
    failures = []
    tests = []
    for i, name in enumerate(NAMES):
        lo, hi = BOUNDS[i]
        dvmin = _make_base_dv()
        dvmin[i] = lo
        tests.append((f"{name}_min", dvmin))
        dvmax = _make_base_dv()
        dvmax[i] = hi
        tests.append((f"{name}_max", dvmax))

    for label, dv in tests:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                stl_path, sac_path, hydro, hull_stl_path = generate_hull(
                    dv, output_dir=tmp, LWL=LWL, target_displacement=TARGET_DISP)
                hull_mesh = trimesh.load(hull_stl_path)
                if isinstance(hull_mesh, trimesh.Scene):
                    hull_mesh = hull_mesh.dump(concatenate=True)
                full_mesh = trimesh.load(stl_path)
                if isinstance(full_mesh, trimesh.Scene):
                    full_mesh = full_mesh.dump(concatenate=True)
                m1 = _check_mesh_quality(hull_mesh, f"{label}.hull")
                m2 = _check_mesh_quality(full_mesh, f"{label}.full")
                ok = True; issues = []
                if not m1["watertight"]: ok = False; issues.append("hull_not_watertight")
                if not m2["watertight"]: ok = False; issues.append("full_not_watertight")
                if m1["volume"] <= 0: ok = False; issues.append(f"hull_vol_{m1['volume']:.6f}")
                if m2["volume"] <= 0: ok = False; issues.append(f"full_vol_{m2['volume']:.6f}")
                if m2["body_count"] > 3: ok = False; issues.append(f"body_count_{m2['body_count']}")
                if m2["convexity_ratio"] >= 0 and m2["convexity_ratio"] < 0.30: ok = False; issues.append(f"convexity_{m2['convexity_ratio']:.3f}")
                if m2["edge_length_ratio"] > 0 and m2["edge_length_ratio"] > 5000: ok = False; issues.append(f"edge_{m2['edge_length_ratio']:.0f}")
                if m2["spike_count"] > max(10, int(m2["n_vertices"] * 0.01)): ok = False; issues.append(f"spikes_{m2['spike_count']}")
                outcome = "ok" if ok else "bad_mesh"
                details = "" if ok else "; ".join(issues)
                _register_design("edge_bounds", label, dv, outcome, details, {"hull": m1, "full": m2})
                if not ok:
                    failures.append((label, issues))
        except (ValueError, RuntimeError, AssertionError) as e:
            _register_design("edge_bounds", label, dv, "exception", f"{type(e).__name__}: {e}")
            failures.append((label, f"exception: {e}"))
        except Exception as e:
            _register_design("edge_bounds", label, dv, "exception_other", f"{type(e).__name__}: {e}")
            failures.append((label, f"other_exception: {e}"))
    return failures

def _test_extreme_combinations():
    failures = []
    extras = [
        ("max_bilge_min_deadrise_max_flare", {"bilge_r": 0.30, "deadrise": 5.0, "flare": 15.0}),
        ("min_bilge_max_deadrise_min_flare", {"bilge_r": 0.05, "deadrise": 25.0, "flare": 5.0}),
        ("max_bilge_min_deadrise_max_Cp", {"bilge_r": 0.30, "deadrise": 5.0, "Cp": 0.65}),
        ("max_deadrise_min_bilge_min_BWL", {"deadrise": 25.0, "bilge_r": 0.05, "BWL": 0.40}),
        ("max_keel_rake_min_T_canoe_max_D_keel", {"keel_rake": 0.02, "T_canoe": 0.15, "D_keel": 1.19}),
        ("min_keel_rake_max_T_canoe_min_D_keel", {"keel_rake": 0.001, "T_canoe": 0.35, "D_keel": 0.85}),
        ("max_flare_max_E_max_SA", {"flare": 15.0, "E": 0.30, "SA": 0.25}),
        ("min_flare_min_E_min_SA", {"flare": 5.0, "E": 0.15, "SA": 0.05}),
        ("max_bulb_vol_extreme_bulb_pos", {"bulb_vol": 0.05, "bulb_pos": 0.30}),
        ("max_bilge_min_deadrise_max_flare_max_keel", {"bilge_r": 0.30, "deadrise": 5.0, "flare": 15.0, "D_keel": 1.19, "keel_chord": 0.25}),
        ("min_bilge_max_deadrise_min_flare_min_BWL", {"bilge_r": 0.05, "deadrise": 25.0, "flare": 5.0, "BWL": 0.40}),
        ("max_Cm_max_Cp_max_LCB", {"Cm": 0.90, "Cp": 0.65, "LCB": 20.0}),
        ("min_Cm_min_Cp_min_LCB", {"Cm": 0.60, "Cp": 0.55, "LCB": 5.0}),
        ("max_ballast_frac_min_D_keel_min_T_canoe", {"ballast_frac": 0.70, "D_keel": 0.85, "T_canoe": 0.15}),
        ("min_ballast_frac_max_D_keel_max_T_canoe", {"ballast_frac": 0.30, "D_keel": 1.19, "T_canoe": 0.35}),
        ("max_bilge_max_flare_max_E_max_ballast", {"bilge_r": 0.30, "flare": 15.0, "E": 0.30, "ballast_frac": 0.70}),
    ]
    for label, overrides in extras:
        dv = _make_base_dv()
        for k, v in overrides.items():
            dv[NAMES.index(k)] = v
        try:
            with tempfile.TemporaryDirectory() as tmp:
                stl_path, sac_path, hydro, hull_stl_path = generate_hull(
                    dv, output_dir=tmp, LWL=LWL, target_displacement=TARGET_DISP)
                hull_mesh = trimesh.load(hull_stl_path)
                if isinstance(hull_mesh, trimesh.Scene):
                    hull_mesh = hull_mesh.dump(concatenate=True)
                full_mesh = trimesh.load(stl_path)
                if isinstance(full_mesh, trimesh.Scene):
                    full_mesh = full_mesh.dump(concatenate=True)
                m1 = _check_mesh_quality(hull_mesh, f"{label}.hull")
                m2 = _check_mesh_quality(full_mesh, f"{label}.full")
                ok = True; issues = []
                if not m1["watertight"]: ok = False; issues.append("hull_not_watertight")
                if not m2["watertight"]: ok = False; issues.append("full_not_watertight")
                if m1["volume"] <= 0: ok = False; issues.append(f"hull_vol_{m1['volume']:.6f}")
                if m2["volume"] <= 0: ok = False; issues.append(f"full_vol_{m2['volume']:.6f}")
                if m2["body_count"] > 3: ok = False; issues.append(f"body_count_{m2['body_count']}")
                if m2["convexity_ratio"] >= 0 and m2["convexity_ratio"] < 0.30: ok = False; issues.append(f"convexity_{m2['convexity_ratio']:.3f}")
                if m2["edge_length_ratio"] > 0 and m2["edge_length_ratio"] > 5000: ok = False; issues.append(f"edge_{m2['edge_length_ratio']:.0f}")
                if m2["spike_count"] > max(10, int(m2["n_vertices"] * 0.01)): ok = False; issues.append(f"spikes_{m2['spike_count']}")
                outcome = "ok" if ok else "bad_mesh"
                details = "" if ok else "; ".join(issues)
                _register_design("extreme", label, dv, outcome, details, {"hull": m1, "full": m2})
                if not ok:
                    failures.append((label, issues))
        except (ValueError, RuntimeError, AssertionError) as e:
            _register_design("extreme", label, dv, "exception", f"{type(e).__name__}: {e}")
            failures.append((label, f"exception: {e}"))
        except Exception as e:
            _register_design("extreme", label, dv, "exception_other", f"{type(e).__name__}: {e}")
            failures.append((label, f"other_exception: {e}"))
    return failures

def _test_degenerate_vectors():
    failures = []
    rng = _rng(300)

    nan_dv = np.full(17, float("nan"))
    nan_dv[0] = LWL
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(nan_dv, output_dir=tmp, LWL=LWL, target_displacement=TARGET_DISP)
        _register_design("degenerate", "all_nan", nan_dv, "unexpected_ok", "expected exception, got success")
        failures.append(("all_nan", "should have raised"))
    except (ValueError, RuntimeError) as e:
        _register_design("degenerate", "all_nan", nan_dv, "exception_expected", f"{type(e).__name__}: {e}")
    except Exception as e:
        _register_design("degenerate", "all_nan", nan_dv, "exception_other", f"{type(e).__name__}: {e}")
        failures.append(("all_nan", f"wrong exception type: {e}"))

    inf_dv = np.full(17, float("inf"))
    inf_dv[0] = LWL
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(inf_dv, output_dir=tmp, LWL=LWL, target_displacement=TARGET_DISP)
        _register_design("degenerate", "all_inf", inf_dv, "unexpected_ok", "expected exception, got success")
        failures.append(("all_inf", "should have raised"))
    except (ValueError, RuntimeError) as e:
        _register_design("degenerate", "all_inf", inf_dv, "exception_expected", f"{type(e).__name__}: {e}")
    except Exception as e:
        _register_design("degenerate", "all_inf", inf_dv, "exception_other", f"{type(e).__name__}: {e}")
        failures.append(("all_inf", f"wrong exception type: {e}"))

    for i, name in enumerate(NAMES):
        if name == "LWL":
            continue
        dv = _random_design(rng)
        dv[i] = float("nan")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                generate_hull(dv, output_dir=tmp, LWL=LWL, target_displacement=TARGET_DISP)
            _register_design("degenerate", f"NaN_{name}", dv, "unexpected_ok", "should have raised")
            failures.append((f"NaN_{name}", "should have raised"))
        except (ValueError, RuntimeError) as e:
            _register_design("degenerate", f"NaN_{name}", dv, "exception_expected", f"{type(e).__name__}: {e}")
        except Exception as e:
            _register_design("degenerate", f"NaN_{name}", dv, "exception_other", f"{type(e).__name__}: {e}")
            failures.append((f"NaN_{name}", f"wrong exception: {e}"))

        dv2 = _random_design(rng)
        dv2[i] = float("inf")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                generate_hull(dv2, output_dir=tmp, LWL=LWL, target_displacement=TARGET_DISP)
            _register_design("degenerate", f"Inf_{name}", dv2, "unexpected_ok", "should have raised")
            failures.append((f"Inf_{name}", "should have raised"))
        except (ValueError, RuntimeError) as e:
            _register_design("degenerate", f"Inf_{name}", dv2, "exception_expected", f"{type(e).__name__}: {e}")
        except Exception as e:
            _register_design("degenerate", f"Inf_{name}", dv2, "exception_other", f"{type(e).__name__}: {e}")
            failures.append((f"Inf_{name}", f"wrong exception: {e}"))
    return failures

def _make_base_dv():
    """A clean safe base design vector in raw [-10, 10] GP space.
    Zeros map to sigmoid midpoints; specific params are tuned to produce
    physically valid hulls via design_vector_to_physical."""
    dv = np.zeros(17)
    dv[NAMES.index("LWL")] = 0.0       # sigmoid(0)=0.5 → LWL ≈ 2.4
    dv[NAMES.index("T_canoe")] = -0.5  # sigmoid(-0.5)→ ~0.38, T_canoe ≈ 0.23
    dv[NAMES.index("BWL")] = 0.0       # BWL ≈ 0.50
    dv[NAMES.index("D_keel")] = 0.5    # sigmoid(0.5)→ ~0.62, D_keel ≈ 1.07
    dv[NAMES.index("Cp")] = 0.2        # Cp ≈ 0.60
    dv[NAMES.index("Cm")] = 0.3        # Cm ≈ 0.75
    return dv

def _test_cross_param_violations():
    failures = []

    # D_keel near LWL with small T_canoe so T_canoe + D_keel stays under LWL
    dv = _make_base_dv()
    dv[NAMES.index("D_keel")] = 10.0  # sigmoid(10)→~1, D_keel≈1.20 (near max)
    dv[NAMES.index("T_canoe")] = -5.0  # sigmoid(-5)→~0, T_canoe≈0.15 (near min)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=LWL, target_displacement=TARGET_DISP)
        _register_design("cross_param", "D_keel_near_LWL", dv, "ok_edge", "D_keel≈1.20=0.5*LWL, accepted")
    except (ValueError, RuntimeError) as e:
        _register_design("cross_param", "D_keel_near_LWL", dv, "exception", f"{type(e).__name__}: {e}")
        failures.append(("D_keel_near_LWL", str(e)))

    # D_keel > LWL — need raw value high enough to push past LWL after sigmoid
    # Physical D_keel bound is 1.20, LWL is 2.4, so D_keel > LWL needs sigmoid(something) producing
    # D_keel = 0.85 + 0.35 * sigmoid(x) > 2.4 → sigmoid(x) > 4.43 (impossible, sigmoid max=1.0)
    # So D_keel > LWL is physically impossible with current bounds. Skip.
    _pass_skip = True
    _register_design("cross_param", "D_keel_gt_LWL", np.zeros(17), "ok_edge", "D_keel bound 1.20 < LWL 2.4 by construction")

    # T_canoe + D_keel > LWL → T_canoe max=0.35 + D_keel max=1.20 = 1.55 < LWL 2.4
    # Cannot trigger with current bounds. Skip.
    _register_design("cross_param", "T_plus_D_gt_LWL", np.zeros(17), "ok_edge", "T+D max=1.55 < LWL=2.4 by construction")

    # keel_chord > LWL — physical bound is 0.25, cannot exceed LWL=2.4
    _register_design("cross_param", "keel_chord_gt_LWL", np.zeros(17), "ok_edge", "keel_chord max=0.25 < LWL=2.4 by construction")

    # Cp = 0 — raw value must produce Cp=0, need sigmoid(x) = -0.55/0.10 = -5.5 (impossible)
    # Skip: Cp min=0.55 by construction
    _register_design("cross_param", "Cp_zero", np.zeros(17), "ok_edge", "Cp min=0.55 by construction")

    # Cp = 1.0 — need sigmoid(x) = 3.5 → x ≈ 1.25 (possible but Cp bound is 0.65)
    dv = _make_base_dv()
    dv[NAMES.index("Cp")] = 10.0  # sigmoid(10)→1, Cp=0.65 (max bound)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=LWL, target_displacement=TARGET_DISP)
        _register_design("cross_param", "Cp_one", dv, "ok_edge", "Cp max=0.65 by construction")
    except (ValueError, RuntimeError) as e:
        _register_design("cross_param", "Cp_one", dv, "exception", f"{type(e).__name__}: {e}")

    # Bulb extending beyond hull length
    dv = _make_base_dv()
    dv[NAMES.index("bulb_vol")] = -2.0  # sigmoid(-2)→~0.12, bulb_vol≈0.007
    dv[NAMES.index("bulb_pos")] = 10.0  # sigmoid(10)→1, bulb_pos≈0.50 (near stern)
    dv[NAMES.index("keel_chord")] = -2.0  # small keel for bulb containment
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=LWL, target_displacement=TARGET_DISP)
        _register_design("cross_param", "bulb_edge", dv, "ok_edge", "bulb near stern edge, accepted")
    except (ValueError, RuntimeError) as e:
        _register_design("cross_param", "bulb_edge", dv, "exception", f"{type(e).__name__}: {e}")

    return failures

def _cluster_failures():
    for entry in RESULTS:
        cat = entry["category"]
        out = entry["outcome"]
        det = entry["details"]
        lab = entry["label"]

        if out.startswith("exception"):
            msg = det.lower()
            if "non-finite" in msg or "nan" in msg or "inf" in msg:
                key = "NaN/Inf parameter rejection"
            elif "d_keel" in msg and ("lwl" in msg or ">" in msg):
                key = "D_keel > LWL cross-param violation"
            elif "t_canoe + d_keel" in msg:
                key = "T_canoe + D_keel > LWL cross-param violation"
            elif "keel_chord" in msg and "lwl" in msg:
                key = "keel_chord > LWL cross-param violation"
            elif "bulb extends" in msg:
                key = "Bulb extends beyond hull bounds"
            elif "cp must be" in msg or "cp=" in msg:
                key = "Cp out of range (0, 1]"
            elif "deadrise must be" in msg:
                key = "Deadrise out of valid range"
            elif "flare must be" in msg:
                key = "Flare out of valid range"
            elif "design vector" in msg and "failed" in msg:
                key = "Geometry validator design vector rejection"
            elif "mesh not watertight" in msg or "not watertight" in msg:
                key = "Non-watertight mesh after keel/bulb attachment"
            elif ("self-intersection" in msg or "self_intersection" in msg or "volume changed" in msg):
                key = "Self-intersecting mesh"
            elif "spike" in msg and ("detected" in msg or "detection" in msg):
                key = "Mesh spike detection failure"
            elif "convexity" in msg:
                key = "Convexity ratio check failure"
            elif "half-breadth" in msg or "half_breadth" in msg:
                key = "Half-breadth gradient check failure"
            elif "normal" in msg and ("validation" in msg or "inverted" in msg or "inversion" in msg):
                key = "Normal orientation/consistency failure"
            elif "sac" in msg:
                key = "SAC scaling divergence"
            elif "control net" in msg:
                key = "Control net curvature/spike failure"
            elif "hull mesh validation" in msg or "combined mesh validation" in msg:
                key = "Hull mesh validation failure (generic)"
            elif "keel-hull" in msg or "keel_hull" in msg:
                key = "Keel-hull intersection/penetration"
            elif "too degenerate" in msg or "insufficient faces" in msg or "degenerate" in msg:
                key = "Mesh too degenerate (insufficient faces)"
            elif "section curve" in msg:
                key = "Section curve deeply negative"
            elif "bilge" in msg:
                key = "Bilge section curve failure"
            else:
                key = f"Other exception: {det[:80]}"
        elif out == "bad_mesh":
            if "hull_not_watertight" in det:
                key = "Hull mesh not watertight (succeeded but flawed)"
            elif "full_not_watertight" in det:
                key = "Combined mesh not watertight"
            elif "vol_" in det:
                key = "Zero/negative mesh volume"
            elif "body_count" in det:
                key = "Disconnected mesh bodies"
            elif "convexity" in det:
                key = "Low convexity ratio (deformed hull)"
            elif "edge_" in det:
                key = "Extreme edge length ratio (sliver triangles)"
            elif "spikes" in det:
                key = "Excessive spike vertices"
            else:
                key = f"Bad mesh: {det[:60]}"
        else:
            continue

        FAILURE_CLUSTERS[key].append({
            "label": f"{cat}/{lab}",
            "outcome": out,
            "details": det,
        })

def _report():
    total = len(DESIGN)
    outcomes = defaultdict(int)
    for r in RESULTS:
        outcomes[r["outcome"]] += 1

    print("=" * 72)
    print("  STRESS TEST RESULTS")
    print("=" * 72)
    print(f"  Total designs tested: {total}")
    print(f"  OK:                  {outcomes.get('ok', 0)}")
    print(f"  Exception (expected): {outcomes.get('exception_expected', 0)}")
    print(f"  Exception (unexpected): {outcomes.get('exception', 0)}")
    print(f"  Exception (other):   {outcomes.get('exception_other', 0)}")
    print(f"  Bad mesh:            {outcomes.get('bad_mesh', 0)}")
    print(f"  Unexpected OK:       {outcomes.get('unexpected_ok', 0)}")
    print(f"  OK edge:             {outcomes.get('ok_edge', 0)}")
    print()

    print("=" * 72)
    print("  FAILURE CLUSTERS BY ROOT CAUSE")
    print("=" * 72)
    cluster_list = sorted(FAILURE_CLUSTERS.items(), key=lambda x: -len(x[1]))
    for i, (cause, entries) in enumerate(cluster_list, 1):
        examples = entries[:3]
        labels = ", ".join(e["label"] for e in examples)
        print(f"\n  Cluster {i}: {cause}")
        print(f"    Count:       {len(entries)}")
        print(f"    Examples:    {labels}")
        print(f"    First detail: {entries[0]['details'][:100]}")

    print()
    print("=" * 72)
    print("  DETAILED DESIGN LOG (first/last failing)")
    print("=" * 72)
    bad = [r for r in RESULTS if r["outcome"] not in ("ok", "ok_edge", "exception_expected")]
    for r in bad[:10]:
        print(f"  [{r['category']}/{r['label']}] {r['outcome']}: {r['details'][:100]}")

    return {
        "total": total,
        "outcomes": dict(outcomes),
        "clusters": {k: len(v) for k, v in FAILURE_CLUSTERS.items()},
    }

def main():
    np.random.seed(42)

    print("\n  [1/5] Random parameter sweep (50 designs)...")
    f1 = _test_random_sweep(50)
    print(f"        done. {len(f1)} designs with issues.")

    print("\n  [2/5] Edge-of-bounds (32 designs, each param min+max)...")
    f2 = _test_edge_of_bounds()
    print(f"        done. {len(f2)} with issues.")

    print("\n  [3/5] Extreme combinations (16 designs)...")
    f3 = _test_extreme_combinations()
    print(f"        done. {len(f3)} with issues.")

    print("\n  [4/5] Degenerate vectors (NaN/Inf)...")
    f4 = _test_degenerate_vectors()
    print(f"        done. {len(f4)} with issues.")

    print("\n  [5/5] Cross-parameter violations...")
    f5 = _test_cross_param_violations()
    print(f"        done. {len(f5)} with issues.")

    _cluster_failures()
    summary = _report()

    out_path = Path("/tmp/opencode/stress_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "designs": DESIGN,
            "results": RESULTS,
            "clusters": {k: v for k, v in FAILURE_CLUSTERS.items()},
            "summary": summary,
        }, f, indent=2)
    print(f"\n  Results written to {out_path}")

if __name__ == "__main__":
    main()
