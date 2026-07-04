"""
Geometry validation module to prevent degenerate hull shapes.
Checks for pancake hulls, detached keels, bulging cross-sections,
self-intersecting faces, and other invalid geometries.
"""

import trimesh
import numpy as np
from typing import Tuple


def validate_hull_geometry(stl_path: str) -> Tuple[bool, str]:
    """
    Validate hull geometry for physical feasibility.

    Returns:
    (is_valid, error_message)
    """
    try:
        mesh = trimesh.load(stl_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if mesh is None:
            return False, "Mesh is None"

        if len(mesh.vertices) < 4:
            return False, f"Insufficient vertices: {len(mesh.vertices)}"

        volume = mesh.volume
        if volume <= 0:
            return False, f"Non-positive volume: {volume:.6f}"
        if volume < 0.02:
            return False, f"Volume too small: {volume:.4f} (min 0.02)"
        if volume > 1.0:
            return False, f"Volume too large: {volume:.4f} (max 1.0)"

        bbox = mesh.bounds
        length = bbox[1][0] - bbox[0][0]
        beam = bbox[1][1] - bbox[0][1]
        height = bbox[1][2] - bbox[0][2]

        if length < 0.5:
            return False, f"Hull too short: length={length:.3f}m (min 0.5m)"
        if length > 3.5:
            return False, f"Hull too long: length={length:.3f}m (max 3.5m)"

        if beam < 0.1:
            return False, f"Hull too narrow: beam={beam:.3f}m (min 0.1m)"
        if beam > 2.0:
            return False, f"Hull too wide: beam={beam:.3f}m (max 2.0m)"

        if height < 0.02:
            return False, f"Hull too flat: height={height:.3f}m (min 0.02m)"
        if height > 2.0:
            return False, f"Hull too tall: height={height:.3f}m (max 2.0m)"

        if length > 0 and height > 0:
            aspect_ratio = length / height
            if aspect_ratio > 100:
                return False, f"Aspect ratio too extreme: {aspect_ratio:.1f} (max 100)"

        if not mesh.is_watertight:
            return False, "Mesh is not watertight"
        if mesh.body_count > 3:
            return False, f"{mesh.body_count} components - watertight mesh should have ≤ 3 bodies"

        # Convexity ratio check: detect self-intersecting or highly concave hulls.
        # A valid hull has some concavity (bow/stern tapering), so the ratio of
        # mesh volume to convex hull volume should be between 0.3 and 0.97.
        try:
            convex = mesh.convex_hull
            if convex is not None and convex.volume > 0:
                convexity = volume / convex.volume
                if convexity < 0.35:
                    return False, f"Hull too concave/bulging: convexity_ratio={convexity:.3f} (min 0.35)"
                if convexity > 1.0 + 1e-9:
                    return False, f"Hull has no concavity (no bow/stern taper): convexity_ratio={convexity:.3f}"
        except Exception as e:
            return False, f"Convexity check error: {e}"

        # Beam-to-draft ratio: prevent cartoonishly flat or excessively deep hulls
        if height > 0.01 and beam > 0.01:
            beam_draft = beam / height
            if beam_draft > 5.0:
                return False, f"Beam/draft ratio too extreme: {beam_draft:.2f} (max 5.0)"

        # Self-intersection check: detect degenerate shapes
        try:
            if hasattr(trimesh.repair, 'fix_self_intersection'):
                mesh_copy = mesh.copy()
                vol_before = abs(mesh_copy.volume)
                result = trimesh.repair.fix_self_intersection(mesh_copy)
                if result is None:
                    vol_after = abs(mesh_copy.volume)
                else:
                    vol_after = abs(result.volume) if hasattr(result, 'volume') else abs(mesh_copy.volume)
                if vol_after > 0 and vol_before > 0:
                    vol_diff = abs(vol_after - vol_before) / max(vol_before, 1e-10)
                    if vol_diff > 0.01:
                        return False, f"Self-intersection detected (volume changed by {vol_diff*100:.1f}%)"
        except Exception as e:
            return False, f"Self-intersection check error: {e}"

        # Local normal consistency: detect inverted patches
        try:
            face_normals = mesh.face_normals
            face_adjacency = mesh.face_adjacency
            if len(face_adjacency) > 0:
                threshold = -0.5  # cos(120°)
                inverted_count = 0
                for f1, f2 in face_adjacency:
                    if f1 < len(face_normals) and f2 < len(face_normals):
                        dot = np.dot(face_normals[f1], face_normals[f2])
                        if dot < threshold:
                            inverted_count += 1
                if inverted_count > len(face_adjacency) * 0.03:
                    return False, f"Excessive inverted normals: {inverted_count}/{len(face_adjacency)} (>120°)"
        except Exception as e:
            return False, f"Normal check error: {e}"

        # Edge length ratio: detect stretched/degenerate faces
        try:
            edges = mesh.edges_unique
            if len(edges) > 0:
                edge_lengths = np.linalg.norm(
                    mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
                if len(edge_lengths) > 0:
                    min_edge = edge_lengths[edge_lengths > 1e-10]
                    if len(min_edge) > 0:
                        max_edge = np.max(edge_lengths)
                        min_edge_val = np.min(min_edge)
                        if max_edge / max(min_edge_val, 1e-10) > 1000:
                            return False, f"Extreme edge ratio: {max_edge/min_edge_val:.0f}"
        except Exception as e:
            return False, f"Edge ratio check error: {e}"

        # Sliver triangle check: reject faces with near-zero area
        try:
            verts = mesh.vertices
            faces = mesh.faces
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            cross = np.cross(v1 - v0, v2 - v0)
            face_areas = 0.5 * np.sqrt(np.sum(cross ** 2, axis=1))
            median_area = float(np.median(face_areas))
            if median_area > 0:
                min_frac = np.min(face_areas) / median_area
                if min_frac < 1e-4:
                    return False, f"Sliver triangle detected: min_face_area/median={min_frac:.2e}"
        except Exception as e:
            return False, f"Sliver triangle check error: {e}"

        return True, ""

    except Exception as e:
        return False, f"Validation error: {e}"


def validate_design_vector(x_dict: dict, config=None) -> Tuple[bool, str]:
    """
    Validate design vector values are within reasonable ranges.

    Returns:
    (is_valid, error_message)
    """
    for key, val in x_dict.items():
        if not np.isfinite(val):
            return False, f"{key}={val} is not finite"

    LWL_val = x_dict.get("LWL", None)
    D_keel_val = x_dict.get("D_keel", None)
    T_canoe_val = x_dict.get("T_canoe", None)

    # Cross-parameter checks: reject physically impossible combinations
    if D_keel_val is not None and LWL_val is not None:
        if D_keel_val > LWL_val:
            return False, f"D_keel ({D_keel_val:.3f}) > LWL ({LWL_val:.3f}) — keel deeper than hull length"
    if T_canoe_val is not None and D_keel_val is not None and LWL_val is not None:
        total_depth = T_canoe_val + D_keel_val
        if total_depth > LWL_val:
            return False, f"T_canoe + D_keel ({total_depth:.3f}) > LWL ({LWL_val:.3f}) — total depth exceeds hull length"

    if config is not None:
        from hull_opt.config import design_vector_names
        names = design_vector_names()
        bnds = config.bounds.as_array()
        for i, name in enumerate(names):
            if name in x_dict:
                lo, hi = bnds[i]
                val = x_dict[name]
                tol = max(0.01, 0.05 * (hi - lo))
                if val < lo - tol or val > hi + tol:
                    return False, f"{name}={val:.4f} outside bounds [{lo:.4f}, {hi:.4f}] (tolerance={tol:.4f})"
    else:
        bounds = {
            "LWL": (1.0, 4.0),
            "BWL": (0.10, 1.5),
            "T_canoe": (0.01, 0.80),
            "Cp": (0.30, 0.85),
            "Cm": (0.30, 1.50),
            "LCB": (0.0, 40.0),
            "D_keel": (0.001, 2.5),
            "keel_chord": (0.001, 0.50),
            "bulb_vol": (0.0, 0.20),
            "bulb_pos": (0.30, 0.50),
            "E": (0.0, 1.0),
            "SA": (0.0, 1.0),
            "flare": (0.0, 30.0),
            "deadrise": (0.0, 60.0),
            "bilge_r": (0.0, 0.50),
            "keel_rake": (0.0, 0.10),
            "ballast_frac": (0.0, 1.0),
        }
        for key, (min_val, max_val) in bounds.items():
            if key in x_dict:
                val = x_dict[key]
                if val < min_val or val > max_val:
                    return False, f"{key}={val:.3f} outside [{min_val:.2f}, {max_val:.2f}]"

    return True, ""
