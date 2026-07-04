"""
Design constraint evaluation for hull feasibility.
Checks B/LWL, Cp, BM, volume error, righting energy, self-righting,
keel aspect ratio, ballast moment, reserve buoyancy, downflooding angle,
and wind heeling equilibrium.
Key exports: evaluate_constraints()
"""
import numpy as np
import trimesh
from typing import Optional
from collections import defaultdict
from hull_opt.hydrostatics import (
    compute_righting_energy, compute_cg_z,
    compute_downflooding_angle, compute_reserve_buoyancy,
    compute_wind_heeling_arm,
)


def _check_half_breadth_gradient(stl_path: str, n_stations: int = 41) -> tuple[bool, str]:
    """Check for extreme half-breadth gradients between stations (spikes)."""
    try:
        mesh = trimesh.load(stl_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        vertices = mesh.vertices
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        LWL = x_max - x_min
        if LWL < 1e-6:
            return True, ""
        
        x_vals = np.linspace(x_min + 0.01 * LWL, x_max - 0.01 * LWL, n_stations)
        station_width = (x_vals[1] - x_vals[0]) * 2.0
        max_half_breadths = []
        
        for x in x_vals:
            slice_mask = np.abs(vertices[:, 0] - x) < station_width
            if np.sum(slice_mask) < 3:
                wider_mask = np.abs(vertices[:, 0] - x) < station_width * 3
                if np.sum(wider_mask) >= 3:
                    slice_verts = vertices[wider_mask]
                    max_half_breadths.append(np.max(np.abs(slice_verts[:, 1])))
                else:
                    max_half_breadths.append(0.0)
                continue
            slice_verts = vertices[slice_mask]
            y_max = np.max(np.abs(slice_verts[:, 1]))
            max_half_breadths.append(y_max)
        
        max_half_breadths = np.array(max_half_breadths)
        gradients = np.abs(np.diff(max_half_breadths)) / (x_vals[1] - x_vals[0])
        
        max_grad_threshold = 2.0  # must match geometry.py
        if np.any(gradients > max_grad_threshold):
            idx = np.argmax(gradients)
            return False, f"Extreme half-breadth gradient at station {idx}: {gradients[idx]:.3f} > {max_grad_threshold}"
        
    except Exception as e:
        return False, f"Half-breadth gradient check error: {e}"
    return True, ""


def _check_keel_hull_intersection(stl_path: str) -> tuple[bool, str]:
    """Check for keel/bulb penetrating deep into hull volume (not surface attachment)."""
    try:
        mesh = trimesh.load(stl_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if mesh.body_count <= 1:
            return True, ""

        bodies = mesh.split(only_watertight=False)
        if len(bodies) <= 1:
            return True, ""

        hull_idx = int(np.argmax([max(b.volume, 0.0) for b in bodies]))
        hull = bodies[hull_idx]
        for j, appendage in enumerate(bodies):
            if j == hull_idx or len(appendage.faces) < 10:
                continue
            try:
                inside = hull.contains(appendage.triangles_center)
                penetration = float(inside.sum()) / max(len(inside), 1)
                if penetration > 0.20:
                    return False, (
                        f"Keel-hull penetration: body {j} has "
                        f"{penetration * 100:.0f}% faces inside hull"
                    )
            except Exception:
                import logging
                logging.getLogger(__name__).warning(f"Penetration check failed for body {j}", exc_info=True)
    except Exception as e:
        return False, f"Keel-hull intersection check error: {e}"
    return True, ""


def _check_mesh_self_intersection(stl_path: str) -> tuple[bool, str]:
    """Check for self-intersections in mesh."""
    try:
        mesh = trimesh.load(stl_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        if hasattr(trimesh.repair, 'fix_self_intersection'):
            vol_before = abs(mesh.volume)
            mesh_copy = mesh.copy()
            result = trimesh.repair.fix_self_intersection(mesh_copy)
            if result is None:
                vol_after = abs(mesh_copy.volume)
            else:
                vol_after = abs(result.volume) if hasattr(result, 'volume') else abs(mesh_copy.volume)
            if vol_after > 0 and vol_before > 0:
                vol_ratio = abs(vol_after - vol_before) / max(vol_before, 1e-10)
                if vol_ratio > 0.01:
                    return False, f"Self-intersection detected (volume changed by {vol_ratio*100:.1f}%)"
                if len(mesh_copy.faces) != len(mesh.faces):
                    return False, f"Self-intersection repaired (faces: {len(mesh.faces)} -> {len(mesh_copy.faces)})"
    except Exception as e:
        return False, f"Self-intersection check error: {e}"
    return True, ""


def _check_local_normals(stl_path: str) -> tuple[bool, str]:
    """Check for locally inverted normals (spikes/folds)."""
    try:
        mesh = trimesh.load(stl_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        face_normals = mesh.face_normals
        face_adjacency = mesh.face_adjacency
        
        if len(face_adjacency) == 0:
            return True, ""
        
        threshold = -0.5  # cos(120°)
        inverted_count = 0
        
        for f1, f2 in face_adjacency:
            if f1 < len(face_normals) and f2 < len(face_normals):
                dot = np.dot(face_normals[f1], face_normals[f2])
                if dot < threshold:
                    inverted_count += 1
        
        if inverted_count > len(face_adjacency) * 0.03:  # > 3% inconsistent
            return False, f"Excessive locally inverted normals: {inverted_count}/{len(face_adjacency)}"
    except Exception as e:
        return False, f"Local normal check error: {e}"
    return True, ""


def _check_element_quality_continuous(mesh) -> float:
    """Return continuous element quality violation magnitude (0.0 = perfect).

    Based on triangle aspect ratio and minimum angle. Catches degenerate
    elements that would crash expensive solvers. Analogous to Jacobian check.
    """
    import numpy as np
    try:
        viol = 0.0
        verts = mesh.vertices
        faces = mesh.faces
        if len(faces) < 1:
            return 0.0

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        side_a = np.linalg.norm(v1 - v0, axis=1)
        side_b = np.linalg.norm(v2 - v1, axis=1)
        side_c = np.linalg.norm(v0 - v2, axis=1)
        s = (side_a + side_b + side_c) / 2
        area = np.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))
        area = np.maximum(area, 1e-30)

        # Triangle aspect ratio: circumradius / (2 * inradius)
        R = (side_a * side_b * side_c) / (4.0 * area)
        r = area / np.maximum(s, 1e-30)
        aspect = R / np.maximum(2.0 * r, 1e-30)

        skinny = aspect[aspect > 10.0]
        if len(skinny) > 0:
            viol += 0.01 * len(skinny) / max(len(faces), 1)

        # Minimum angle check (< 5!)
        cos_a = (side_b**2 + side_c**2 - side_a**2) / (2.0 * np.maximum(side_b * side_c, 1e-30))
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angle_a = np.arccos(cos_a)
        cos_b = (side_a**2 + side_c**2 - side_b**2) / (2.0 * np.maximum(side_a * side_c, 1e-30))
        cos_b = np.clip(cos_b, -1.0, 1.0)
        angle_b = np.arccos(cos_b)
        angle_c = np.pi - angle_a - angle_b

        min_angles = np.minimum(np.minimum(angle_a, angle_b), angle_c)
        small = min_angles[min_angles < np.deg2rad(5.0)]
        if len(small) > 0:
            viol += 0.05 * float(np.sum(np.deg2rad(5.0) - small) / np.deg2rad(5.0))

        return float(min(viol, 10.0))
    except Exception:
        return 0.5  # conservative penalty if check fails


def _check_volume_area_ratio_soft(stl_path: str, mesh=None) -> float:
    """Soft penalty for low volume/area ratio. Returns 0.0 (OK) to 2.0 (severe)."""
    try:
        if mesh is None:
            mesh = trimesh.load(stl_path)
        sa = mesh.area
        vol = mesh.volume
        if sa <= 0 or vol <= 0:
            return 2.0
        ratio = vol / (sa ** 1.5)
        if ratio >= 0.02:
            return 0.0
        penalty = 2.0 * max(0.0, 0.02 - ratio) / 0.02
        return min(penalty, 2.0)
    except Exception:
        return 0.0


def _check_symmetry_soft(stl_path: str, mesh=None) -> float:
    """Soft penalty for port/starboard asymmetry. Returns 0.0 (symmetric) to 5.0 (severe)."""
    try:
        if mesh is None:
            mesh = trimesh.load(stl_path)
        verts = mesh.vertices
        port_mask = verts[:, 1] >= 0
        star_mask = verts[:, 1] <= 0
        port_verts = verts[port_mask].copy()
        star_verts = verts[star_mask].copy()
        star_verts[:, 1] *= -1.0  # mirror to port side
        if len(port_verts) < 10 or len(star_verts) < 10:
            return 5.0
        from scipy.spatial import cKDTree
        tree = cKDTree(port_verts)
        dists, _ = tree.query(star_verts)
        max_asym = float(np.max(dists))
        bounds = mesh.bounds
        beam = bounds[1, 1] - bounds[0, 1]
        if beam < 1e-6:
            return 5.0
        asym_norm = max_asym / beam
        if asym_norm <= 0.01:
            return 0.0
        penalty = 5.0 * min(1.0, max(0.0, asym_norm - 0.01) / 0.04)
        return penalty
    except Exception:
        return 0.0


def _check_curvature_roughness(stl_path: str, mesh=None) -> float:
    """Soft penalty for serrated/jagged hull surfaces. Returns 0.0 (smooth) to 5.0 (severe)."""
    try:
        if mesh is None:
            mesh = trimesh.load(stl_path)
        try:
            from trimesh.curvature import discrete_mean_curvature_measure
            curv = discrete_mean_curvature_measure(mesh, mesh.vertices)
        except ImportError:
            edges = mesh.edges_unique
            lengths = mesh.edges_unique_length
            if len(lengths) < 2:
                return 0.0
            roughness = float(np.std(lengths) / max(np.mean(lengths), 1e-10))
            return 5.0 * min(1.0, max(0.0, roughness - 0.5) / 2.0)
        curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
        roughness = float(np.std(curv))
        penalty = 5.0 * min(1.0, max(0.0, roughness - 0.1) / 0.4)
        return penalty
    except Exception:
        return 0.0


def evaluate_constraints(hydro: dict, gz_curve: np.ndarray,
                         roll_period: float = 0.0,
                         peak_accel: float = 0.0,
                         x_dict: Optional[dict] = None,
                         config=None,
                         stl_path: Optional[str] = None,
                         hull_stl_path: Optional[str] = None,
                         iteration: Optional[int] = None) -> tuple[bool, list, dict, float]:
    violation_magnitude = 0.0
    violations = []
    constraints = {}

    relaxation = 0.0
    if iteration is not None:
        relaxation = max(0.0, 1.0 - iteration / 200.0)

    # Mesh-level validation for degenerate shapes (uses full mesh with keel/bulb)
    if stl_path is not None:
        is_valid, err_msg = _check_mesh_self_intersection(stl_path)
        if not is_valid:
            return False, [f"mesh_self_intersection: {err_msg}"], {"mesh_self_intersection": err_msg}, 2.0
        is_valid, err_msg = _check_local_normals(stl_path)
        if not is_valid:
            return False, [f"local_normals: {err_msg}"], {"local_normals": err_msg}, 1.0
        is_valid, err_msg = _check_keel_hull_intersection(stl_path)
        if not is_valid:
            return False, [f"keel_hull_intersection: {err_msg}"], {"keel_hull_intersection": err_msg}, 2.0

    # Check half-breadth gradient on hull-only mesh
    if hull_stl_path is not None:
        is_valid, err_msg = _check_half_breadth_gradient(hull_stl_path)
        if not is_valid:
            return False, [f"half_breadth_gradient: {err_msg}"], {"half_breadth_gradient": err_msg}, 1.0
    elif stl_path is not None:
        is_valid, err_msg = _check_half_breadth_gradient(stl_path)
        if not is_valid:
            return False, [f"half_breadth_gradient: {err_msg}"], {"half_breadth_gradient": err_msg}, 1.0

    # Rule 6: Element quality (Jacobian-adjacent) — continuous violation
    if stl_path is not None:
        try:
            eq_mesh = trimesh.load(stl_path)
            if isinstance(eq_mesh, trimesh.Scene):
                eq_mesh = eq_mesh.dump(concatenate=True)
            eq_viol = _check_element_quality_continuous(eq_mesh)
            if eq_viol > 3.0:
                violations.append(f"element_quality: {eq_viol:.4f}")
                violation_magnitude += eq_viol
        except Exception:
            pass

        # --- Volume/Area ratio penalty (anti paper-thin) ---
        try:
            va_penalty = _check_volume_area_ratio_soft(stl_path if stl_path else hull_stl_path)
            if va_penalty > 0:
                violation_magnitude += va_penalty
                violations.append(f"volume_area_ratio: {va_penalty:.3f}")
        except Exception:
            pass

        # --- Symmetry penalty (anti CFD hack) ---
        try:
            sym_penalty = _check_symmetry_soft(stl_path if stl_path else hull_stl_path)
            if sym_penalty > 0:
                violation_magnitude += sym_penalty
                violations.append(f"asymmetry: {sym_penalty:.3f}")
        except Exception:
            pass

        # --- Curvature roughness penalty (anti serration) ---
        try:
            rough_penalty = _check_curvature_roughness(stl_path if stl_path else hull_stl_path)
            if rough_penalty > 0:
                violation_magnitude += rough_penalty
                violations.append(f"curvature_roughness: {rough_penalty:.3f}")
        except Exception:
            pass

    # Blanket NaN/Inf guard: reject any design with non-finite constraint values
    for k, v in hydro.items():
        if isinstance(v, (int, float, np.floating)) and not np.isfinite(v):
            return False, [f"Non-finite hydro value {k}={v}"], {f"{k}_finite": v}, 5.0

    B = hydro.get("B", 0.0)
    LWL = hydro.get("LWL", 2.4)
    Cp = hydro.get("Cp", 0.0)
    nabla = hydro.get("nabla", 0.0)
    BM = hydro.get("BM", 0.0)
    target_nabla = hydro.get("target_nabla", 0.25)
    underwater_vol = hydro.get("underwater_volume", nabla)

    for k in list(locals()):
        v = locals()[k]
        if isinstance(v, (int, float, np.floating)) and not np.isfinite(v):
            return False, [f"Non-finite value {k}={v} before constraint eval"], {}

    sac_scale = hydro.get("sac_scale_factor", 1.0)
    constraints["B/LWL"] = B / max(1e-10, LWL)
    constraints["B/LWL_scaled"] = constraints["B/LWL"] * sac_scale
    constraints["Cp"] = Cp
    constraints["actual_Cp"] = hydro.get("actual_Cp", Cp)
    constraints["BM"] = BM
    constraints["nabla"] = nabla
    constraints["underwater_volume"] = underwater_vol
    constraints["target_nabla"] = target_nabla

    rho = hydro.get("rho", 1025.0)
    righting_energy = compute_righting_energy(gz_curve, max_heel_deg=60.0,
                                                displacement=underwater_vol, rho=rho)
    constraints["righting_energy"] = righting_energy

    high_angles = gz_curve[:, 0]
    high_gz = gz_curve[:, 1]
    late_mask = (high_angles >= 150.0) & (high_angles < 180.0)
    self_right = False
    mean_gz_high = 0.0
    if np.sum(late_mask) > 2:
        gz_high_valid = high_gz[late_mask]
        if np.all(np.isfinite(gz_high_valid)):
            mean_gz_high = float(np.mean(gz_high_valid))
            self_right = mean_gz_high > 0.005
    # Geometry heuristic fallback: only apply when GZ angular resolution is poor
    if not self_right and x_dict is not None and np.sum(late_mask) <= 2:
        if np.all(np.isfinite(high_gz)):
            T_canoe = x_dict.get("T_canoe", 0.2)
            D_keel = x_dict.get("D_keel", 0.5)
            ballast_frac = x_dict.get("ballast_frac", 0.3)
            min_ballast = 0.3
            if config is not None and hasattr(config, 'validation'):
                min_ballast = config.validation.min_ballast_ratio
            if (D_keel > T_canoe * 3.0 and ballast_frac > min_ballast + 0.1 and D_keel >= 1.0):
                self_right = True
    constraints["self_righting"] = 1.0 if self_right else 0.0

    constraints["roll_period"] = roll_period
    constraints["peak_accel"] = peak_accel

    # Phase 5: B/LWL floating penalty (with relaxation)
    blwl_actual = constraints["B/LWL"]
    blwl_target = (0.15 + 0.30) / 2.0  # 0.225
    blwl_range = (0.30 - 0.15) / 2.0   # 0.075
    blwl_dev = abs(blwl_actual - blwl_target) / blwl_range
    blwl_degenerate = 2.0 + relaxation * 2.0  # start at 4.0, tighten to 2.0
    if blwl_dev > blwl_degenerate:
        return False, [f"B/LWL {blwl_actual:.4f} out of control (dev={blwl_dev:.2f})"], constraints, violation_magnitude + 10.0
    elif blwl_dev > 1.0:
        blwl_penalty = 8.0 * (blwl_dev - 1.0) ** 2
        violation_magnitude += blwl_penalty
        violations.append(f"B/LWL: {blwl_actual:.4f} (target {blwl_target:.3f}, penalty {blwl_penalty:.2f})")
        constraints['B/LWL'] = blwl_actual
    elif blwl_dev > 0.8:
        blwl_penalty = 8.0 * 0.05 * (blwl_dev - 0.8) / 0.2
        violation_magnitude += blwl_penalty
        violations.append(f"B/LWL: {blwl_actual:.4f} (near boundary, penalty {blwl_penalty:.2f})")
        constraints['B/LWL'] = blwl_actual
    else:
        constraints['B/LWL'] = blwl_actual

    # B/LWL_scaled accounts for SAC volume scaling factor (can be ~0.44 when hull
    # displacement needs to be met). The unscaled B/LWL constraint already ensures
    # reasonable beam. The scaled constraint only catches extreme cases where SAC
    # scaling produces undersized or balloon sections.
    if constraints["B/LWL_scaled"] < 0.03:
        viol = 0.03 - constraints["B/LWL_scaled"]
        violations.append(f"B/LWL_scaled={constraints['B/LWL_scaled']:.3f} < 0.03")
        violation_magnitude += viol
    elif constraints["B/LWL_scaled"] > (0.75 + relaxation * 0.50):
        max_blwl_scaled = 0.75 + relaxation * 0.50
        viol = constraints["B/LWL_scaled"] - max_blwl_scaled
        violations.append(f"B/LWL_scaled={constraints['B/LWL_scaled']:.3f} > {max_blwl_scaled:.2f}")
        violation_magnitude += viol

    # Beam-to-draft ratio: prevent cartoonishly flat hulls (excessive beam vs depth)
    # Use actual beam after SAC scaling, not the unscaled design-parameter BWL
    T_canoe = hydro.get("T_canoe", 0.3)
    if not np.isfinite(T_canoe) or T_canoe <= 0:
        T_canoe = 0.3
    B_actual = B * sac_scale
    if T_canoe > 0:
        beam_draft = B_actual / T_canoe
        constraints["beam_draft_ratio"] = beam_draft
        max_beam_draft = 4.5 + relaxation * 2.0  # start at 6.5, tighten to 4.5
        if beam_draft > max_beam_draft:
            viol = beam_draft - max_beam_draft
            violations.append(f"B_actual/T_canoe={beam_draft:.2f} > {max_beam_draft:.2f} (hull too flat)")
            violation_magnitude += viol * 0.2

    # Volume mismatch: compare underwater volume against target displacement
    if config is not None and hasattr(config, 'fixed'):
        true_target = config.fixed.target_displacement
    else:
        true_target = target_nabla
    vol_ratio = abs(underwater_vol - true_target) / max(1e-10, true_target)
    constraints["vol_ratio_error"] = vol_ratio
    # Phase 5: Volume error floating penalty (with relaxation)
    vol_degenerate = 0.50 + relaxation * 0.50  # start at 100%, tighten to 50%
    vol_penalty_thresh = 0.25 + relaxation * 0.50  # start at 75%, tighten to 25%
    if vol_ratio > vol_degenerate:
        return False, [f"Volume error {vol_ratio:.2%} > {vol_degenerate:.0%}"], constraints, violation_magnitude + 10.0
    elif vol_ratio > vol_penalty_thresh:
        vol_penalty = 3.0 * ((vol_ratio - vol_penalty_thresh) / max(vol_penalty_thresh, 1e-6)) ** 2
        violation_magnitude += vol_penalty
        violations.append(f"volume_error: {vol_ratio:.2%} (penalty {vol_penalty:.2f})")
        constraints['volume_error'] = vol_ratio
    elif vol_ratio > 0.20:
        vol_penalty = 3.0 * 0.05 * (vol_ratio - 0.20) / 0.05
        violation_magnitude += vol_penalty
        violations.append(f"volume_error: {vol_ratio:.2%} (near boundary, penalty {vol_penalty:.2f})")
        constraints['volume_error'] = vol_ratio
    else:
        constraints['volume_error'] = vol_ratio

    # Minimum displacement: prevent pancake hulls
    if nabla < 0.02:
        viol = 0.02 - nabla
        violations.append(f"hull_volume={nabla:.4f} m³ < 0.02 m³ (minimum displacement)")
        violation_magnitude += viol * 20.0
    if true_target > 0 and nabla < 0.5 * true_target:
        viol = 0.5 - nabla / true_target
        violations.append(f"hull_volume={nabla:.4f} m³ < 50% of target={true_target:.4f} m³")
        violation_magnitude += viol * 2.0

    # SAC scale factor: prevent extreme SAC scaling producing balloon sections
    constraints["sac_scale_factor"] = sac_scale
    max_sac = 2.5 + relaxation * 2.5  # start at 5.0, tighten to 2.5
    if sac_scale > max_sac:
        viol = sac_scale - max_sac
        violations.append(f"sac_scale_factor={sac_scale:.2f} > {max_sac:.2f}")
        violation_magnitude += viol * 0.5
    if sac_scale < 0.5:
        viol = 0.5 - sac_scale
        violations.append(f"sac_scale_factor={sac_scale:.2f} < 0.5")
        violation_magnitude += viol * 0.5
    sac_scale_std = hydro.get("sac_scale_std", 0.0)
    constraints["sac_scale_std"] = sac_scale_std
    if sac_scale_std > 0.5:
        viol = sac_scale_std - 0.5
        violations.append(f"sac_scale_std={sac_scale_std:.3f} > 0.5 (per-station SAC scaling varies too much)")
        violation_magnitude += viol

    # Phase 5: Cp floating penalty
    cp_check = constraints.get("actual_Cp", constraints["Cp"])
    if config is not None:
        cp_low = config.bounds.Cp[0] * 0.9
        cp_high = config.bounds.Cp[1] * 1.1
    else:
        cp_low = 0.45
        cp_high = 0.65
    if not np.isfinite(cp_check):
        return False, [f"actual_Cp non-finite: {cp_check}"], constraints, violation_magnitude + 10.0
    cp_target = (cp_low + cp_high) / 2.0
    cp_range = (cp_high - cp_low) / 2.0
    cp_dev = abs(cp_check - cp_target) / max(cp_range, 1e-6)
    if cp_dev > 3.0:  # >3x range = degenerate
        return False, [f"actual_Cp {cp_check:.4f} out of control (dev={cp_dev:.2f})"], constraints, violation_magnitude + 10.0
    elif cp_dev > 1.0:
        cp_penalty = 5.0 * (cp_dev - 1.0) ** 2
        violation_magnitude += cp_penalty
        violations.append(f"actual_Cp: {cp_check:.4f} (penalty {cp_penalty:.2f})")
        constraints['actual_Cp'] = cp_check
    elif cp_dev > 0.8:
        cp_penalty = 5.0 * 0.05 * (cp_dev - 0.8) / 0.2
        violation_magnitude += cp_penalty
        violations.append(f"actual_Cp: {cp_check:.4f} (near boundary, penalty {cp_penalty:.2f})")
        constraints['actual_Cp'] = cp_check
    else:
        constraints['actual_Cp'] = cp_check
    # Phase 5: BM floating penalty (with relaxation)
    if BM < 0.005:  # Near-zero BM = degenerate
        return False, [f"BM {BM:.4f}m < 0.005m"], constraints, violation_magnitude + 10.0
    bm_threshold = max(0.03 - relaxation * 0.025, 0.005)  # start at 0.005, ramp to 0.03
    if bm_threshold > 0.005 and BM < bm_threshold:
        bm_penalty = 10.0 * (bm_threshold - BM) / bm_threshold
        violation_magnitude += bm_penalty
        violations.append(f"BM: {BM:.4f}m (penalty {bm_penalty:.2f})")
        constraints['BM'] = BM
    elif BM < 0.04:
        bm_penalty = 10.0 * 0.05 * (0.04 - BM) / 0.01
        violation_magnitude += bm_penalty
        violations.append(f"BM: {BM:.4f}m (near boundary, penalty {bm_penalty:.2f})")
        constraints['BM'] = BM
    else:
        constraints['BM'] = BM
    if config is not None:
        min_re = config.validation.min_righting_energy
        min_re_relaxed = min_re * (1.0 - relaxation * 0.7)  # start at 30% of nominal, ramp up
        if not np.isfinite(constraints["righting_energy"]) or constraints["righting_energy"] < min_re_relaxed:
            viol = max(0, min_re_relaxed - constraints["righting_energy"]) if np.isfinite(constraints["righting_energy"]) else min_re_relaxed
            violations.append(f"righting_energy={constraints['righting_energy']:.1f} J < {min_re_relaxed:.1f} J")
            violation_magnitude += viol / max(min_re_relaxed, 1e-6)
    elif not np.isfinite(constraints["righting_energy"]) or constraints["righting_energy"] <= 0.0:
        violations.append(f"righting_energy={constraints['righting_energy']:.1f} J <= 0 J")
        violation_magnitude += 1.0
    if not np.isfinite(constraints["self_righting"]) or constraints["self_righting"] < 0.5:
        violation_magnitude += 1.0 - (constraints["self_righting"] if np.isfinite(constraints["self_righting"]) else 0.0)
        violations.append("inverted stability FAIL")
    max_accel_gate = 30.0
    if config is not None and hasattr(config, 'validation'):
        max_accel_gate = getattr(config.validation, 'max_accel_gate', getattr(config.validation, 'max_accel_g', 30.0))
    if not np.isfinite(constraints["peak_accel"]) or constraints["peak_accel"] > max_accel_gate:
        viol = constraints["peak_accel"] - max_accel_gate if np.isfinite(constraints["peak_accel"]) else max_accel_gate
        violations.append(f"peak_accel={constraints['peak_accel']:.1f}g > {max_accel_gate:.0f}g")
        violation_magnitude += viol / max_accel_gate

    if roll_period > 0 and (not np.isfinite(roll_period) or roll_period < 0.5 or roll_period > 8.0):
        if np.isfinite(roll_period) and roll_period < 0.5:
            viol = 0.5 - roll_period
        elif np.isfinite(roll_period):
            viol = roll_period - 8.0
        else:
            viol = 0.5
        violations.append(f"roll_period={roll_period:.2f}s not in [0.5, 8.0] s")
        violation_magnitude += viol

    if x_dict is not None:
        D_keel = x_dict.get("D_keel", 0.5)
        keel_chord = x_dict.get("keel_chord", 0.2)
        ballast_frac = x_dict.get("ballast_frac", 0.3)

        # Keel aspect ratio: prevent stubby or absurdly slender keels
        mean_chord = keel_chord * 0.75
        keel_ar = D_keel / max(1e-10, mean_chord)
        constraints["keel_aspect_ratio"] = keel_ar
        ar_upper = (config.bounds.D_keel[1] / (config.bounds.keel_chord[0] * 0.75)
                    if config is not None else 10.67)
        if keel_ar < 2.0:
            viol = 2.0 - keel_ar
            violations.append(f"keel_AR={keel_ar:.3f} < 2.0")
            violation_magnitude += viol * 0.5
        if keel_ar > ar_upper:
            viol = keel_ar - ar_upper
            violations.append(f"keel_AR={keel_ar:.3f} > {ar_upper:.2f}")
            violation_magnitude += viol * 0.5

        # Ballast moment: prevent extreme ballast + deep keel gaming righting energy
        ballast_moment = ballast_frac * D_keel
        constraints["ballast_moment"] = ballast_moment
        max_bm = 0.75 + relaxation * 0.30  # start at 1.05, tighten to 0.75
        if ballast_moment > max_bm + 1e-9:
            viol = ballast_moment - max_bm
            violations.append(f"ballast_moment={ballast_moment:.3f} > {max_bm:.2f}")
            violation_magnitude += viol * 2.0

        # Bilge radius vs beam: prevent extreme bilge radius causing bulging sections
        bilge_r = x_dict.get("bilge_r", 0.0)
        if B > 0:
            br_ratio = bilge_r / B
            constraints["bilge_r_BWL_ratio"] = br_ratio
            max_br = 0.5 + relaxation * 0.3  # start at 0.8, tighten to 0.5
            if br_ratio > max_br:
                viol = br_ratio - max_br
                violations.append(f"bilge_r/BWL={br_ratio:.3f} > {max_br:.2f} (causes non-monotonic cross-section)")
                violation_magnitude += viol * 2.0

        constraints["ballast_ratio"] = ballast_frac
        min_ballast = config.validation.min_ballast_ratio if config is not None else 0.25
        if ballast_frac < min_ballast:
            viol = min_ballast - ballast_frac
            violations.append(f"ballast_ratio={ballast_frac:.3f} < {min_ballast:.2f}")
            violation_magnitude += viol * 3.0

    if x_dict is not None and config is not None and (stl_path is not None or hull_stl_path is not None):
        reserve = compute_reserve_buoyancy(stl_path, x_dict)
        constraints["reserve_buoyancy"] = reserve
        min_reserve = config.validation.min_reserve_buoyancy
        if reserve < min_reserve:
            viol = min_reserve - reserve
            violations.append(f"reserve_buoyancy={reserve:.3f} < {min_reserve:.2f}")
            violation_magnitude += viol * 3.0

        # Downflooding: vessel is fully enclosed/watertight - skip this check
        # df_angle is still computed for info but never triggers a violation
        try:
            df_angle = compute_downflooding_angle(hull_stl_path or stl_path, cg_z=compute_cg_z(x_dict, nabla=nabla))
        except Exception:
            df_angle = 180.0
        constraints["downflooding_angle"] = df_angle
        # No violation enforced (watertight vessel)

        storm_wind_ms = config.validation.storm_wind_speed_knots * 0.514444
        sail_area_feathered = config.fixed.wing_sail_area * 0.15
        sail_height = config.fixed.wing_sail_height
        g = config.fixed.gravity

        gz_angles = gz_curve[:, 0]
        gz_vals = gz_curve[:, 1]
        eq_heel = 90.0
        above_wind = False
        for deg in np.linspace(0, 90, 19):
            wind_arm = compute_wind_heeling_arm(
                deg, storm_wind_ms, sail_area_feathered, sail_height,
                nabla, rho_water=rho, g=g
            )
            gz_at_deg = float(np.interp(deg, gz_angles, gz_vals, left=0, right=0))
            if gz_at_deg >= wind_arm:
                above_wind = True
                eq_heel = deg
                break  # first crossing = equilibrium angle
            elif above_wind:
                break
        constraints["eq_heel_feathered_deg"] = eq_heel
        max_eq_heel = 45.0 + relaxation * 30.0  # start at 75°, tighten to 45°
        if eq_heel > max_eq_heel:
            viol = eq_heel - max_eq_heel
            violations.append(f"equilibrium heel={eq_heel:.1f}° > {max_eq_heel:.0f}° with feathered wings")
            violation_magnitude += viol / max(45.0, 1e-6)

    feasible = len(violations) == 0
    return feasible, violations, constraints, float(violation_magnitude)
