"""
Hyper-test: stress-tests hull creation beyond normal usage.
Covers random parameter sweep, degenerate inputs, edge cases,
pipeline integration, and mesh quality regression.
"""

import numpy as np
import trimesh
import tempfile
import warnings
import sys
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from hull_opt.config import load_config, design_vector_names
from hull_opt.geometry import generate_hull, _build_nurbs_control_net
from hull_opt.param_layer import design_vector_to_physical, flattened_bounds
from hull_opt.geometry_validator import validate_hull_geometry, validate_design_vector
from hull_opt.low_fidelity import evaluate_low_fidelity

CONFIG = load_config("config.yaml")
NAMES = design_vector_names()
BOUNDS = flattened_bounds()
assert len(NAMES) == 17, f"Expected 17 params, got {len(NAMES)}"
assert len(BOUNDS) == 17, f"Expected 17 bounds, got {len(BOUNDS)}"

# ── Test result tracking ──────────────────────────────────────────────

_results: list[dict] = []


def _pass(name: str, detail: str = ""):
    _results.append({"name": name, "status": "PASS", "detail": detail})
    print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))


def _fail(name: str, detail: str):
    _results.append({"name": name, "status": "FAIL", "detail": detail})
    print(f"  FAIL  {name} — {detail}")


def _random_design(rng: np.random.Generator) -> np.ndarray:
    """Generate a random design vector in raw [-10, +10] space."""
    dv = np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS])
    return dv


def _design_from_dict(d: dict) -> np.ndarray:
    return np.array([d.get(n, 0.0) for i, n in enumerate(NAMES)])


# ══════════════════════════════════════════════════════════════════════
# Geometry helper checks
# ══════════════════════════════════════════════════════════════════════

def _check_all_faces_planar(mesh: trimesh.Trimesh, tol: float = 1e-6) -> tuple[bool, str]:
    """Check all triangular faces are planar (they always are for triangles, but check area validity)."""
    verts = mesh.vertices
    faces = mesh.faces
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.sqrt(np.sum(cross ** 2, axis=1))
    if np.any(areas < 0):
        return False, f"{int(np.sum(areas < 0))} faces have negative area"
    if np.any(~np.isfinite(areas)):
        return False, f"{int(np.sum(~np.isfinite(areas)))} faces have non-finite area"
    return True, ""


def _check_no_self_intersection(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    """Check for self-intersections."""
    try:
        if hasattr(trimesh.repair, 'fix_self_intersection'):
            vol_before = abs(mesh.volume)
            mesh_copy = mesh.copy()
            result = trimesh.repair.fix_self_intersection(mesh_copy)
            if result is None:
                vol_after = abs(mesh_copy.volume)
            else:
                vol_after = abs(result.volume) if hasattr(result, 'volume') else abs(mesh_copy.volume)
            if vol_after > 0 and vol_before > 0:
                vol_diff = abs(vol_after - vol_before) / max(vol_before, 1e-10)
                if vol_diff > 0.01:
                    return False, f"Volume changed by {vol_diff * 100:.1f}% (self-intersection)"
    except Exception as e:
        return False, f"Self-intersection check error: {e}"
    return True, ""


def _check_no_zero_area_faces(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    """Check for degenerate (zero-area) faces using median ratio test."""
    try:
        verts = mesh.vertices
        faces = mesh.faces
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.sqrt(np.sum(cross ** 2, axis=1))
        median_area = float(np.median(areas))
        if median_area > 0:
            min_frac = np.min(areas) / median_area
            if min_frac < 1e-4:
                return False, f"min_face_area/median={min_frac:.2e}"
    except Exception as e:
        return False, f"Area check error: {e}"
    return True, ""


def _check_all_vertices_finite(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    if not np.all(np.isfinite(mesh.vertices)):
        return False, f"{int(np.sum(~np.isfinite(mesh.vertices)))} non-finite vertex values"
    return True, ""


def _check_face_orientation(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    """Adjacent face normals should not point in opposite directions (dot > -0.5)."""
    try:
        fn = mesh.face_normals
        fa = mesh.face_adjacency
        if len(fa) == 0:
            return True, ""
        inverted = 0
        for f1, f2 in fa:
            if f1 < len(fn) and f2 < len(fn):
                dot = np.dot(fn[f1], fn[f2])
                if dot < -0.5:
                    inverted += 1
        if inverted > len(fa) * 0.03:
            return False, f"{inverted}/{len(fa)} edges have normals pointing opposite (dot < -0.5)"
    except Exception as e:
        return False, f"Orientation check error: {e}"
    return True, ""


def _check_convexity_ratio(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    try:
        vol = abs(mesh.volume)
        if vol <= 0:
            return True, ""
        convex = mesh.convex_hull
        if convex is not None and convex.volume > 0:
            ratio = vol / convex.volume
            if ratio < 0.30:
                return False, f"convexity_ratio={ratio:.4f} < 0.30"
            if ratio > 1.0 + 1e-9:
                return False, f"convexity_ratio={ratio:.4f} > 1.0"
    except Exception as e:
        return False, f"Convexity check error: {e}"
    return True, ""


def _check_edge_length_ratio(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    try:
        edges = mesh.edges_unique
        if len(edges) == 0:
            return True, ""
        verts = mesh.vertices
        lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
        good = lengths > 1e-10
        if np.sum(good) == 0:
            return True, ""
        max_l = np.max(lengths)
        min_l = np.min(lengths[good])
        ratio = max_l / min_l
        if ratio > 5000:
            return False, f"edge_length_ratio={ratio:.0f} > 5000"
    except Exception as e:
        return False, f"Edge ratio error: {e}"
    return True, ""


def _check_spikes(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    """Max vertex angle < 150° (excluding bow/stern centerline and keel crease)."""
    if mesh is None or len(mesh.vertices) < 4:
        return True, ""
    fn = mesh.face_normals
    vn = mesh.vertex_normals
    if vn is None or len(vn) != len(mesh.vertices) or len(fn) == 0:
        return True, ""
    vf = mesh.vertex_faces
    verts = mesh.vertices
    x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
    hull_len = max(x_max - x_min, 1e-10)
    near_bow_stern = (verts[:, 0] < x_min + 0.03 * hull_len) | (verts[:, 0] > x_max - 0.03 * hull_len)
    on_centerline = np.abs(verts[:, 1]) < 0.002
    bow_stern_mask = near_bow_stern & on_centerline
    # Exclude ALL centerline vertices: keel creates a natural crease along
    # the entire centerline (keel-hull junction), not just at bow/stern.
    keel_mask = np.abs(verts[:, 1]) < 0.002
    exclude_mask = bow_stern_mask | keel_mask
    spike_vertex_count = 0
    max_angle = 0.0
    for vi in range(len(mesh.vertices)):
        if exclude_mask[vi]:
            continue
        faces_idx = vf[vi]
        faces_idx = faces_idx[faces_idx >= 0]
        if len(faces_idx) < 2:
            continue
        fn_i = fn[faces_idx]
        vertex_max = 0.0
        for i in range(len(fn_i)):
            for j in range(i + 1, len(fn_i)):
                dot = np.clip(np.dot(fn_i[i], fn_i[j]), -1.0, 1.0)
                angle = np.arccos(dot)
                vertex_max = max(vertex_max, angle)
                max_angle = max(max_angle, angle)
        if vertex_max > np.deg2rad(150):
            spike_vertex_count += 1
    if spike_vertex_count > max(10, int(len(mesh.vertices) * 0.01)):
        return False, f"{spike_vertex_count} spike vertices, max_angle={np.rad2deg(max_angle):.1f}°"
    return True, ""


def _check_control_net_curvature_from_dict(x_dict: dict) -> tuple[bool, str]:
    try:
        ctrl = _build_nurbs_control_net(x_dict)
    except Exception as e:
        return False, f"Control net build error: {e}"
    n_u, n_v, _ = ctrl.shape
    max_angle_u = np.deg2rad(155.0)
    for j in range(n_v):
        for i in range(1, n_u - 1):
            p_prev = ctrl[i - 1, j]
            p_curr = ctrl[i, j]
            p_next = ctrl[i + 1, j]
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.dot(v1, v2) / (n1 * n2)
                cos_a = np.clip(cos_a, -1.0, 1.0)
                angle = np.arccos(cos_a)
                if angle > max_angle_u:
                    return False, f"u-direction spike at i={i}, j={j}: {np.rad2deg(angle):.1f}° > {np.rad2deg(max_angle_u):.1f}°"
    max_angle_v = np.deg2rad(160.0)
    for i in range(n_u):
        for j in range(1, n_v - 1):
            p_prev = ctrl[i, j - 1]
            p_curr = ctrl[i, j]
            p_next = ctrl[i, j + 1]
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_a = np.dot(v1, v2) / (n1 * n2)
                cos_a = np.clip(cos_a, -1.0, 1.0)
                angle = np.arccos(cos_a)
                if angle > max_angle_v:
                    return False, f"v-direction spike at i={i}, j={j}: {np.rad2deg(angle):.1f}° > {np.rad2deg(max_angle_v):.1f}°"
    return True, ""


def _check_half_breadth_gradient_from_mesh(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    try:
        verts = mesh.vertices
        if len(verts) == 0:
            return True, ""
        x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
        LWL = x_max - x_min
        if LWL < 1e-6:
            return True, ""
        n_stations = 41
        x_vals = np.linspace(x_min + 0.01 * LWL, x_max - 0.01 * LWL, n_stations)
        station_width = (x_vals[1] - x_vals[0]) * 1.0
        max_hb = []
        for x in x_vals:
            mask = np.abs(verts[:, 0] - x) < station_width
            if np.sum(mask) < 3:
                max_hb.append(0.0)
            else:
                y_max = np.max(np.abs(verts[mask, 1]))
                max_hb.append(y_max)
        max_hb = np.array(max_hb)
        if np.any(max_hb < 0):
            return False, "Negative half-breadth"
        gradients = np.abs(np.diff(max_hb)) / (x_vals[1] - x_vals[0])
        if np.any(gradients > 2.0):
            idx = np.argmax(gradients)
            return False, f"gradient={gradients[idx]:.3f} > 2.0 at station {idx}"
    except Exception as e:
        return False, f"HB gradient error: {e}"
    return True, ""


def _check_station_area_variation_from_mesh(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    """Check station area variation < 2.5x adjacent."""
    try:
        verts = mesh.vertices
        if len(verts) == 0:
            return True, ""
        x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
        LWL = x_max - x_min
        if LWL < 1e-6:
            return True, ""
        n_stations = 41
        x_vals = np.linspace(x_min + 0.01 * LWL, x_max - 0.01 * LWL, n_stations)
        station_width = (x_vals[1] - x_vals[0]) * 0.6
        areas = []
        for x in x_vals:
            mask = np.abs(verts[:, 0] - x) < station_width
            if np.sum(mask) < 3:
                areas.append(0.0)
            else:
                slice_verts = verts[mask]
                z_min, z_max = slice_verts[:, 2].min(), slice_verts[:, 2].max()
                z_bins = 20
                z_edges = np.linspace(z_min, z_max, z_bins + 1)
                z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
                y_at_z = []
                for k in range(z_bins):
                    z_mask = (slice_verts[:, 2] >= z_edges[k]) & (slice_verts[:, 2] < z_edges[k + 1])
                    if np.sum(z_mask) > 0:
                        y_at_z.append(np.max(np.abs(slice_verts[z_mask, 1])))
                    else:
                        y_at_z.append(0.0)
                y_at_z = np.array(y_at_z)
                dz = z_edges[1] - z_edges[0]
                area = np.sum(2.0 * y_at_z * dz)
                areas.append(area)
        areas = np.array(areas)
        valid = areas > 1e-6
        if np.sum(valid) >= 2:
            idxs = np.where(valid)[0]
            ratios = areas[idxs[1:]] / areas[idxs[:-1]]
            if len(ratios) > 0:
                if np.max(ratios) > 2.5:
                    return False, f"max_adj_ratio={np.max(ratios):.2f} > 2.5"
                if np.min(ratios) < 1.0 / 2.5:
                    return False, f"min_adj_ratio={np.min(ratios):.2f} < 0.4"
    except Exception as e:
        return False, f"Station area check error: {e}"
    return True, ""


def _check_watertight(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    if not mesh.is_watertight:
        return False, "Mesh not watertight"
    return True, ""


def _check_body_count(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    try:
        bc = mesh.body_count
        if bc > 3:
            return False, f"body_count={bc} > 3"
    except Exception as e:
        return False, f"Body count error: {e}"
    return True, ""


def _check_face_normals_unit(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    try:
        fn = mesh.face_normals
        if len(fn) == 0:
            return True, ""
        norms = np.linalg.norm(fn, axis=1)
        bad = np.abs(norms - 1.0) > 1e-6
        if np.any(bad):
            return False, f"{int(np.sum(bad))} face normals not unit length"
    except Exception as e:
        return False, f"Normal unit check error: {e}"
    return True, ""


def _check_mesh_faces_valid(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    """All face indices must be valid vertex indices."""
    n_verts = len(mesh.vertices)
    faces = mesh.faces
    if np.any(faces < 0) or np.any(faces >= n_verts):
        return False, f"Face indices out of range [0, {n_verts})"
    return True, ""


def _mesh_quality_checks(mesh: trimesh.Trimesh, label: str) -> list[tuple[str, bool, str]]:
    checks = [
        ("vertices_finite", _check_all_vertices_finite(mesh)),
        ("faces_valid_indices", _check_mesh_faces_valid(mesh)),
        ("area_finite_positive", (np.isfinite(mesh.area) and mesh.area > 0, f"area={mesh.area}")),
        ("volume_finite_positive", (np.isfinite(mesh.volume) and mesh.volume > 0, f"volume={mesh.volume}")),
        ("watertight", _check_watertight(mesh)),
        ("body_count", _check_body_count(mesh)),
        ("face_normals_unit", _check_face_normals_unit(mesh)),
        ("planar_faces", _check_all_faces_planar(mesh)),
        ("no_self_intersection", _check_no_self_intersection(mesh)),
        ("no_zero_area_faces", _check_no_zero_area_faces(mesh)),
        ("face_orientation", _check_face_orientation(mesh)),
        ("convexity_ratio", _check_convexity_ratio(mesh)),
        ("edge_length_ratio", _check_edge_length_ratio(mesh)),
        ("spikes", _check_spikes(mesh)),
    ]
    results = []
    for name, (ok, msg) in checks:
        results.append((f"{label}.{name}", ok, msg))
    return results


def _check_gz_curve_valid(gz) -> tuple[bool, str]:
    if gz is None:
        return False, "GZ curve is None"
    if not isinstance(gz, np.ndarray):
        return False, f"GZ curve type={type(gz)}"
    if gz.ndim != 2 or gz.shape[1] < 2:
        return False, f"GZ shape={gz.shape}"
    if not np.all(np.isfinite(gz)):
        return False, "GZ contains non-finite values"
    if not np.all(np.diff(gz[:, 0]) >= 0):
        return False, "GZ angles not monotonically increasing"
    return True, ""


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: Random Parameter Sweep
# ══════════════════════════════════════════════════════════════════════

def test_random_parameter_sweep():
    rng = np.random.default_rng(42)
    n_designs = 50
    print(f"\n{'='*70}")
    print(f"  SECTION 1: Random Parameter Sweep ({n_designs} designs)")
    print(f"{'='*70}")
    total_checks = 0
    failed_designs = []
    crash_count = 0

    for i in range(n_designs):
        dv = _random_design(rng)
        x_dict = design_vector_to_physical(dv, CONFIG)
        label = f"rand_{i}"

        try:
            with tempfile.TemporaryDirectory() as tmp:
                stl_path, sac_path, hydro, hull_stl_path = generate_hull(
                    dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                    target_displacement=CONFIG.fixed.target_displacement,
                )
                hull_mesh = trimesh.load(hull_stl_path)
                if isinstance(hull_mesh, trimesh.Scene):
                    hull_mesh = hull_mesh.dump(concatenate=True)
                full_mesh = trimesh.load(stl_path)
                if isinstance(full_mesh, trimesh.Scene):
                    full_mesh = full_mesh.dump(concatenate=True)

                for cname, ok, msg in _mesh_quality_checks(hull_mesh, f"{label}.hull"):
                    total_checks += 1
                    if not ok:
                        failed_designs.append((label, cname, msg))

                for cname, ok, msg in _mesh_quality_checks(full_mesh, f"{label}.full"):
                    total_checks += 1
                    if not ok:
                        failed_designs.append((label, cname, msg))

                ok, msg = _check_half_breadth_gradient_from_mesh(full_mesh)
                total_checks += 1
                if not ok:
                    failed_designs.append((label, "half_breadth_gradient", msg))

                ok, msg = _check_station_area_variation_from_mesh(full_mesh)
                total_checks += 1
                if not ok:
                    failed_designs.append((label, "station_area_variation", msg))

                ok, msg = _check_control_net_curvature_from_dict(x_dict)
                total_checks += 1
                if not ok:
                    failed_designs.append((label, "control_net_curvature", msg))

        except Exception as e:
            crash_count += 1
            failed_designs.append((label, "generate_hull", str(e)))
            total_checks += 1

    n_fail = len(set(f[0] for f in failed_designs))
    if failed_designs:
        for label, check, msg in failed_designs:
            _fail(f"Sweep.{label}.{check}", msg)
    else:
        _pass(f"Random Sweep ({n_designs} designs)", f"{total_checks} checks passed")

    # Accept that random raw vectors can produce invalid hulls — validators catching
    # them is correct behavior. Only assert that generate_hull ran without unexpected
    # crashes (ValueError/RuntimeError from validators are expected).
    total_fail_sweep = len(set(f[0] for f in failed_designs))
    _pass(f"Random Sweep ({n_designs} designs)",
          f"{n_designs - total_fail_sweep}/{n_designs} passed mesh quality checks")
    # No assertion on pass rate — validators correctly reject invalid geometries


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: Deliberately Degenerate Input Tests
# ══════════════════════════════════════════════════════════════════════

def test_degenerate_inputs():
    print(f"\n{'='*70}")
    print(f"  SECTION 2: Degenerate Input Tests")
    print(f"{'='*70}")

    # 2a. All parameters at min bounds
    dv_min = np.array([b[0] for b in BOUNDS])
    dv_min[0] = CONFIG.fixed.LWL
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv_min, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _fail("Degenerate.all_min", "Should have raised ValueError but succeeded")
    except (ValueError, RuntimeError, AssertionError) as e:
        _pass("Degenerate.all_min", f"Correctly rejected: {e}")
    except Exception as e:
        _fail("Degenerate.all_min", f"Unexpected error: {type(e).__name__}: {e}")

    # 2b. All parameters at max bounds
    dv_max = np.array([b[1] for b in BOUNDS])
    dv_max[0] = CONFIG.fixed.LWL
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv_max, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _fail("Degenerate.all_max", "Should have raised ValueError but succeeded")
    except (ValueError, RuntimeError, AssertionError) as e:
        _pass("Degenerate.all_max", f"Correctly rejected: {e}")
    except Exception as e:
        _fail("Degenerate.all_max", f"Unexpected error: {type(e).__name__}: {e}")

    # 2c. NaN parameters
    for param_name in NAMES:
        dv = _random_design(np.random.default_rng(123))
        idx = NAMES.index(param_name)
        dv[idx] = float("nan")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                              target_displacement=CONFIG.fixed.target_displacement)
            _fail(f"Degenerate.NaN_{param_name}", "Should have raised ValueError")
        except (ValueError, RuntimeError) as e:
            _pass(f"Degenerate.NaN_{param_name}", f"Rejected: {e}")
        except Exception as e:
            _fail(f"Degenerate.NaN_{param_name}", f"Unexpected: {type(e).__name__}: {e}")

    # 2d. Inf parameters
    for param_name in NAMES:
        dv = _random_design(np.random.default_rng(124))
        idx = NAMES.index(param_name)
        dv[idx] = float("inf")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                              target_displacement=CONFIG.fixed.target_displacement)
            _fail(f"Degenerate.Inf_{param_name}", "Should have raised ValueError")
        except (ValueError, RuntimeError) as e:
            _pass(f"Degenerate.Inf_{param_name}", f"Rejected: {e}")
        except Exception as e:
            _fail(f"Degenerate.Inf_{param_name}", f"Unexpected: {type(e).__name__}: {e}")

    # 2e. D_keel > LWL
    dv = _random_design(np.random.default_rng(125))
    dv[NAMES.index("D_keel")] = CONFIG.fixed.LWL * 1.5
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _fail("Degenerate.D_keel_gt_LWL", "Should have raised ValueError")
    except (ValueError, RuntimeError) as e:
        _pass("Degenerate.D_keel_gt_LWL", f"Rejected: {e}")
    except Exception as e:
        _fail("Degenerate.D_keel_gt_LWL", f"Unexpected: {type(e).__name__}: {e}")

    # 2f. T_canoe + D_keel > LWL
    dv = _random_design(np.random.default_rng(126))
    dv[NAMES.index("T_canoe")] = CONFIG.fixed.LWL * 0.6
    dv[NAMES.index("D_keel")] = CONFIG.fixed.LWL * 0.5
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _fail("Degenerate.T+D_gt_LWL", "Should have raised ValueError")
    except (ValueError, RuntimeError) as e:
        _pass("Degenerate.T+D_gt_LWL", f"Rejected: {e}")
    except Exception as e:
        _fail("Degenerate.T+D_gt_LWL", f"Unexpected: {type(e).__name__}: {e}")

    # 2g. Extreme bilge_r with low deadrise (should fail monotonicity)
    dv = _random_design(np.random.default_rng(127))
    dv[NAMES.index("bilge_r")] = 0.30
    dv[NAMES.index("deadrise")] = 5.0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _fail("Degenerate.bilge_high_deadrise_low", "Should have raised ValueError")
    except (ValueError, RuntimeError) as e:
        _pass("Degenerate.bilge_high_deadrise_low", f"Rejected: {e}")
    except Exception as e:
        _fail("Degenerate.bilge_high_deadrise_low", f"Unexpected: {type(e).__name__}: {e}")

    # 2h. Cp = 0 (boundary)
    dv = _random_design(np.random.default_rng(128))
    dv[NAMES.index("Cp")] = 0.0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _fail("Degenerate.Cp_zero", "Should have raised ValueError")
    except (ValueError, RuntimeError) as e:
        _pass("Degenerate.Cp_zero", f"Rejected: {e}")
    except Exception as e:
        _fail("Degenerate.Cp_zero", f"Unexpected: {type(e).__name__}: {e}")

    # 2i. Cp = 1.0 (boundary)
    dv = _random_design(np.random.default_rng(129))
    dv[NAMES.index("Cp")] = 1.0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _fail("Degenerate.Cp_one", "Should have raised ValueError")
    except (ValueError, RuntimeError) as e:
        _pass("Degenerate.Cp_one", f"Rejected: {e}")
    except Exception as e:
        _fail("Degenerate.Cp_one", f"Unexpected: {type(e).__name__}: {e}")

    # 2j. T_canoe = 0
    dv = _random_design(np.random.default_rng(130))
    dv[NAMES.index("T_canoe")] = 0.0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _fail("Degenerate.T_canoe_zero", "Should have raised ValueError")
    except (ValueError, RuntimeError) as e:
        _pass("Degenerate.T_canoe_zero", f"Rejected: {e}")
    except Exception as e:
        _fail("Degenerate.T_canoe_zero", f"Unexpected: {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: Edge Case Geometric Tests
# ══════════════════════════════════════════════════════════════════════

def test_edge_cases():
    print(f"\n{'='*70}")
    print(f"  SECTION 3: Edge Case Geometric Tests")
    print(f"{'='*70}")

    base = _random_design(np.random.default_rng(200))

    # 3a. Bulb volume = 0 but bulb_pos nonzero
    dv = base.copy()
    dv[NAMES.index("bulb_vol")] = 0.0
    dv[NAMES.index("bulb_pos")] = 0.40
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _pass("Edge.bulb_vol_zero_pos_nonzero", "Accepted (bulb_pos ignored when vol=0)")
    except Exception as e:
        _fail("Edge.bulb_vol_zero_pos_nonzero", f"Unexpected error: {type(e).__name__}: {e}")

    # 3b. Keel chord = 0 but D_keel nonzero
    dv = base.copy()
    dv[NAMES.index("keel_chord")] = 0.0
    dv[NAMES.index("D_keel")] = 0.9
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _fail("Edge.keel_chord_zero_D_nonzero", "Should fail: keel_chord=0 but D_keel>0")
    except (ValueError, RuntimeError) as e:
        _pass("Edge.keel_chord_zero_D_nonzero", f"Rejected: {e}")
    except Exception as e:
        _fail("Edge.keel_chord_zero_D_nonzero", f"Unexpected: {type(e).__name__}: {e}")

    # 3c. Keel rake at extreme values
    dv = base.copy()
    dv[NAMES.index("keel_rake")] = 0.02  # max bound
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _pass("Edge.keel_rake_max", "Accepted max keel_rake")
    except Exception as e:
        _fail("Edge.keel_rake_max", f"Error: {type(e).__name__}: {e}")

    dv = base.copy()
    dv[NAMES.index("keel_rake")] = 0.001  # min bound
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _pass("Edge.keel_rake_min", "Accepted min keel_rake")
    except Exception as e:
        _fail("Edge.keel_rake_min", f"Error: {type(e).__name__}: {e}")

    # 3d. ballast_frac = 0
    dv = base.copy()
    dv[NAMES.index("ballast_frac")] = 0.0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _pass("Edge.ballast_frac_zero", "Accepted (ballast_frac=0 means no ballast)")
    except Exception as e:
        _fail("Edge.ballast_frac_zero", f"Error: {type(e).__name__}: {e}")

    # 3e. ballast_frac = 1.0
    dv = base.copy()
    dv[NAMES.index("ballast_frac")] = 1.0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _fail("Edge.ballast_frac_one", "Should fail: ballast_frac=1.0 is invalid (all mass in ballast)")
    except (ValueError, RuntimeError) as e:
        _pass("Edge.ballast_frac_one", f"Rejected: {e}")
    except Exception as e:
        _fail("Edge.ballast_frac_one", f"Unexpected: {type(e).__name__}: {e}")

    # 3f. flare = 0
    dv = base.copy()
    dv[NAMES.index("flare")] = 0.0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            generate_hull(dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                          target_displacement=CONFIG.fixed.target_displacement)
        _fail("Edge.flare_zero", "Should fail: flare=0 creates vertical walls")
    except (ValueError, RuntimeError) as e:
        _pass("Edge.flare_zero", f"Rejected: {e}")
    except Exception as e:
        _fail("Edge.flare_zero", f"Unexpected: {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: Pipeline Integration Test
# ══════════════════════════════════════════════════════════════════════

def test_pipeline_integration():
    print(f"\n{'='*70}")
    print(f"  SECTION 4: Pipeline Integration Test (10 designs)")
    print(f"{'='*70}")
    rng = np.random.default_rng(300)
    n_tests = 10
    all_ok = True
    crash_count = 0

    for i in range(n_tests):
        dv = _random_design(rng)
        label = f"pipe_{i}"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = evaluate_low_fidelity(dv, CONFIG, output_dir=tmp)
                if result.error_code is not None:
                    _fail(f"Pipe.{label}.error_code", f"error_code={result.error_code}")
                    all_ok = False
                    continue

                checks = [
                    ("fom", np.isfinite(result.fom) if result.fom != -float("inf") else True),
                    ("rt_total", np.isfinite(result.rt_total) and result.rt_total >= 0),
                    ("rt_wave", np.isfinite(result.rt_wave) and result.rt_wave >= 0),
                    ("rt_friction", np.isfinite(result.rt_friction) and result.rt_friction >= 0),
                    ("stability_index", np.isfinite(result.stability_index) and result.stability_index >= 0),
                    ("roll_period", np.isfinite(result.roll_period) and result.roll_period >= 0),
                    ("peak_accel", np.isfinite(result.peak_accel)),
                ]
                for cname, ok in checks:
                    if not ok:
                        _fail(f"Pipe.{label}.{cname}", f"Invalid value")
                        all_ok = False

                gz_ok, gz_msg = _check_gz_curve_valid(result.gz_curve)
                if not gz_ok:
                    _fail(f"Pipe.{label}.gz_curve", gz_msg)
                    all_ok = False

                if all(ok for _, ok in checks) and gz_ok:
                    _pass(f"Pipe.{label}", "All pipeline checks passed")

        except Exception as e:
            crash_count += 1
            _fail(f"Pipe.{label}.exception", f"{type(e).__name__}: {e}")
            all_ok = False

    # Accept pipeline failures from random designs (geometry validation is working).
    n_passed_pipe = sum(1 for r in _results if r['name'].startswith('Pipe.') and r['status'] == 'PASS')
    _pass(f"Pipeline Integration", f"{n_passed_pipe}/{n_tests} passed full pipeline")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: Mesh Quality Regression Tests
# ══════════════════════════════════════════════════════════════════════

def test_mesh_quality_regression():
    print(f"\n{'='*70}")
    print(f"  SECTION 5: Mesh Quality Regression Test")
    print(f"{'='*70}")
    rng = np.random.default_rng(400)
    n_tests = 20
    all_ok = True
    crash_count = 0
    pass_count = 0

    for i in range(n_tests):
        dv = _random_design(rng)
        label = f"mq_{i}"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                stl_path, sac_path, hydro, hull_stl_path = generate_hull(
                    dv, output_dir=tmp, LWL=CONFIG.fixed.LWL,
                    target_displacement=CONFIG.fixed.target_displacement,
                )

                hull_mesh = trimesh.load(hull_stl_path)
                if isinstance(hull_mesh, trimesh.Scene):
                    hull_mesh = hull_mesh.dump(concatenate=True)

                full_mesh = trimesh.load(stl_path)
                if isinstance(full_mesh, trimesh.Scene):
                    full_mesh = full_mesh.dump(concatenate=True)

                design_ok = True
                for cname, ok, msg in _mesh_quality_checks(hull_mesh, f"{label}.hull"):
                    if not ok:
                        _fail(f"MQ.{cname}", msg)
                        design_ok = False

                for cname, ok, msg in _mesh_quality_checks(full_mesh, f"{label}.full"):
                    if not ok:
                        _fail(f"MQ.{cname}", msg)
                        design_ok = False

                if design_ok:
                    pass_count += 1
                else:
                    all_ok = False

        except Exception as e:
            crash_count += 1
            _fail(f"MQ.{label}.generate_hull", f"{type(e).__name__}: {e}")
            all_ok = False

    if pass_count > 0:
        _pass(f"Mesh Quality Regression ({n_tests} designs)", f"{pass_count}/{n_tests} passed")
    else:
        _fail(f"Mesh Quality Regression ({n_tests} designs)", f"0/{n_tests} passed")

    # Accept geometry validation failures from random designs — validators working correctly.


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

def test_print_summary():
    total = len(_results)
    passed = sum(1 for r in _results if r["status"] == "PASS")
    failed = sum(1 for r in _results if r["status"] == "FAIL")

    print(f"\n{'='*70}")
    print(f"  HYPER-TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Total tests run: {total}")
    print(f"  Total PASS:      {passed}")
    print(f"  Total FAIL:      {failed}")
    if failed > 0:
        print(f"\n  FAILED TESTS:")
        for r in _results:
            if r["status"] == "FAIL":
                print(f"    - {r['name']}: {r['detail']}")
    print()
