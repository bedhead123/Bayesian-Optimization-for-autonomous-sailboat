"""
NURBS-based hull, keel, and bulbous bow geometry generation.
17-parameter design vector → watertight STL mesh via B-spline surface,
NACA foil keel, and ellipsoid bulb.
Key exports: generate_hull(), compute_half_breadth_analytic(), design_vector_to_dict()
"""
import numpy as np
import trimesh
from pathlib import Path
from typing import Optional
import warnings
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


from hull_opt.config import design_vector_names
from hull_opt.geometry_validator import validate_design_vector
from hull_opt.param_layer import design_vector_to_physical


def _check_volume_area_ratio(mesh, min_ratio=0.008, warn_ratio=0.02):
    """Hard kill for paper-thin hulls: V / (SA^1.5) below min_ratio.
    
    A brick is ~0.08, a typical hull is 0.02-0.06, a paper-thin shell is ~0.001.
    Returns (passed: bool, message: str).
    """
    sa = mesh.area
    vol = mesh.volume
    if sa <= 0 or vol <= 0:
        return False, f"Non-positive SA ({sa:.6f}) or volume ({vol:.6f})"
    ratio = vol / (sa ** 1.5)
    if ratio < min_ratio:
        return False, f"V/SA^1.5 ratio {ratio:.6f} < {min_ratio} (min={min_ratio})"
    return True, ""


def _validate_hull_mesh(mesh: trimesh.Trimesh, x_dict: dict, LWL: float) -> tuple[bool, str]:
    """
    Comprehensive hull mesh validation for degenerate shapes.
    
    Returns: (is_valid, error_message)
    """
    try:
        # 1. Basic mesh validity
        if mesh is None or len(mesh.vertices) < 4:
            return False, "Mesh is None or has insufficient vertices"
        
        if not mesh.is_watertight:
            return False, "Mesh is not watertight"
        
        volume = mesh.volume
        if not np.isfinite(volume) or volume <= 0:
            return False, f"Invalid volume: {volume}"
        
        # Volume/area ratio: catch paper-thin membranes before expensive checks
        passed, msg = _check_volume_area_ratio(mesh)
        if not passed:
            return False, msg
        
        # 2. Self-intersection check
        try:
            if hasattr(trimesh.repair, 'fix_self_intersection'):
                fixed = trimesh.repair.fix_self_intersection(mesh.copy())
                if fixed is not None and abs(fixed.volume - volume) / max(abs(volume), 1e-10) > 0.01:
                    return False, f"Self-intersection detected (volume changed by {abs(fixed.volume - volume) / max(abs(volume), 1e-10) * 100:.1f}%)"
        except Exception as e:
            return False, f"Self-intersection check error: {e}"
        
        # 3. Local curvature / spike detection
        # Check for extreme vertex angles (spikes), excluding bow/stern centerline
        # where a natural knife-edge crease occurs (port/starboard sides meet).
        verts = mesh.vertices
        # Exclude centerline vertices (keel crease) where port/starboard
        # sides meet at naturally sharp angles.
        on_centerline = np.abs(verts[:, 1]) < 0.002
        face_normals = mesh.face_normals
        if len(face_normals) > 0:
            # For each vertex, compute angle between adjacent face normals
            vertex_normals = mesh.vertex_normals
            if vertex_normals is not None and len(vertex_normals) == len(mesh.vertices):
                # Compute angle between vertex normal and face normals of incident faces
                # This catches sharp spikes where face normals diverge sharply
                try:
                    vf_adj = mesh.vertex_faces
                    max_angle_diff = 0.0
                    spike_vertex_count = 0
                    for vi in range(len(mesh.vertices)):
                        if on_centerline[vi]:
                            continue
                        faces_idx = vf_adj[vi]
                        faces_idx = faces_idx[faces_idx >= 0]
                        if len(faces_idx) >= 2:
                            fn = face_normals[faces_idx]
                            vertex_max = 0.0
                            for i in range(len(fn)):
                                for j in range(i+1, len(fn)):
                                    dot = np.clip(np.dot(fn[i], fn[j]), -1.0, 1.0)
                                    angle = np.arccos(dot)
                                    vertex_max = max(vertex_max, angle)
                            if vertex_max > np.deg2rad(150):
                                spike_vertex_count += 1
                            max_angle_diff = max(max_angle_diff, vertex_max)
                    if spike_vertex_count > max(10, int(len(mesh.vertices) * 0.01)):
                        return False, f"Sharp spikes detected: {spike_vertex_count} vertices with angle > 150°, max={np.rad2deg(max_angle_diff):.1f}°"
                except Exception as e:
                    logger.warning(f"Spike detection error in _validate_hull_mesh: {e}")
        
        # 4. Edge length ratio check (detects stretched/tiny faces)
        edges = mesh.edges_unique
        if len(edges) > 0:
            edge_lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
            if len(edge_lengths) > 0:
                max_edge = np.max(edge_lengths)
                min_edge = np.min(edge_lengths[edge_lengths > 1e-10])
                if max_edge / max(min_edge, 1e-10) > 10000:
                    return False, f"Extreme edge length ratio: {max_edge/min_edge:.0f} (max 10000)"
        
        # 5. Sliver triangle check (near-zero area faces)
        try:
            verts = mesh.vertices
            faces = mesh.faces
            if len(faces) > 0:
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
            logger.debug(f"Sliver triangle check error: {e}")

        # 6. Local normal consistency (detect inverted patches)
        # Sample face normals and check for local inversions
        try:
            f_normals = mesh.face_normals
            ff_adj = mesh.face_adjacency
            if len(ff_adj) > 0:
                inconsistent = 0
                for fa, fb in ff_adj:
                    if fa < len(f_normals) and fb < len(f_normals):
                        dot = np.dot(f_normals[fa], f_normals[fb])
                        if dot < -0.5:  # > 120° between adjacent faces
                            inconsistent += 1
                if inconsistent > len(ff_adj) * 0.03:  # > 3% inconsistent edges
                    return False, f"Local normal inconsistency: {inconsistent}/{len(ff_adj)} edges (>120°)"
        except Exception as e:
            logger.debug(f"Normal consistency check error in _validate_hull_mesh: {e}")
            return False, f"Normal consistency check failed: {e}"
        
        # 6. Bounding box sanity (no NaN/Inf, reasonable dimensions)
        bbox = mesh.bounds
        if not np.all(np.isfinite(bbox)):
            return False, "Mesh bounds contain NaN/Inf"
        dims = bbox[1] - bbox[0]
        if dims[0] < 0.5 or dims[0] > 3.5:
            return False, f"Hull length {dims[0]:.3f} outside bounds [0.5, 3.5]"
        if dims[1] < 0.1 or dims[1] > 3.5:
            return False, f"Hull beam {dims[1]:.3f} outside bounds [0.1, 3.5]"
        if dims[2] < 0.02 or dims[2] > 2.0:
            return False, f"Hull height {dims[2]:.3f} outside bounds [0.02, 2.0]"
        
        # 7. NURBS control net curvature check (if we have x_dict)
        if x_dict:
            try:
                ctrl = _build_nurbs_control_net(x_dict)
                # Check control point spacing - no extreme local curvature
                n_u, n_v, _ = ctrl.shape
                max_ctrl_dist = 0.0
                min_ctrl_dist = float('inf')
                for i in range(n_u - 1):
                    for j in range(n_v - 1):
                        # Check four edges of control net quad
                        d1 = np.linalg.norm(ctrl[i+1, j] - ctrl[i, j])
                        d2 = np.linalg.norm(ctrl[i, j+1] - ctrl[i, j])
                        d3 = np.linalg.norm(ctrl[i+1, j+1] - ctrl[i, j+1])
                        d4 = np.linalg.norm(ctrl[i+1, j+1] - ctrl[i+1, j])
                        for d in [d1, d2, d3, d4]:
                            if d > 1e-6:
                                max_ctrl_dist = max(max_ctrl_dist, d)
                                min_ctrl_dist = min(min_ctrl_dist, d)
                if min_ctrl_dist > 0 and max_ctrl_dist / min_ctrl_dist > 50:
                    return False, f"Control net extreme spacing ratio: {max_ctrl_dist/min_ctrl_dist:.1f}"
            except Exception as e:
                return False, f"Control net check error: {e}"
        
        return True, ""
    except Exception as e:
        return False, f"Validation error: {e}"


def _check_sac_scaling_station_variation(x_dict: dict, LWL: float, sac_scale: float) -> tuple[bool, str]:
    """
    Check that SAC scaling doesn't create extreme per-station variation.
    Returns (is_valid, error_message).
    """
    try:
        n_stations = 41
        u_vals = np.linspace(0.0, 1.0, n_stations)
        BWL = x_dict["BWL"]
        T_canoe = x_dict["T_canoe"]
        Cp = x_dict["Cp"]
        Cm = x_dict.get("Cm", 0.75)
        LCB_val = x_dict.get("LCB", 12.5)
        
        station_areas = np.zeros(n_stations)
        for i in range(n_stations):
            xs = u_vals[i] * LWL
            x_norm = u_vals[i]
            if x_norm <= 0.0:
                T_local = 1e-6
            else:
                T_local = T_canoe * max(0.05, 1.0 - 0.3 * (1.0 - x_norm))
            z_sheer = _sheer_height(x_norm, x_dict.get("E", 0.20), x_dict.get("SA", 0.15))
            n_z = 20
            z_vals = np.linspace(-T_local, z_sheer, n_z)
            y_vals = np.zeros(n_z)
            if i == 0 or i == n_stations - 1:
                continue
            for k in range(n_z):
                xq = xs - LWL / 2
                y_vals[k] = compute_half_breadth_analytic(
                    np.array([xq]), np.array([z_vals[k]]),
                    x_dict, LWL, sac_scale=sac_scale).item(0)
            ds = np.diff(z_vals)
            y_mid = 0.5 * (y_vals[:-1] + y_vals[1:])
            station_areas[i] = float(np.sum(2.0 * y_mid * np.abs(ds)))
        
        # Check for extreme variation in adjacent stations (spikes).
        # Skip bow/stern transitions where area is intentionally zero.
        valid = station_areas > 1e-6
        if np.sum(valid) >= 2:
            ratios = []
            for i in range(len(station_areas) - 1):
                if valid[i] and valid[i + 1]:
                    ratios.append(station_areas[i + 1] / station_areas[i])
            if ratios:
                max_r = max(ratios)
                min_r = min(ratios)
                if max_r > 2.5 or min_r < 1.0 / 2.5:
                    return False, f"Extreme station area variation: adjacent ratio {max_r:.2f}/{min_r:.2f}"

            # Check for single-station spike
            for i in range(1, n_stations - 1):
                if station_areas[i] > 0:
                    neighbors = (station_areas[i-1] + station_areas[i+1]) / 2
                    if neighbors > 1e-10:
                        if station_areas[i] / neighbors > 2.0:
                            return False, f"Single-station spike at station {i}: {station_areas[i]/neighbors:.2f}x neighbors"
                    elif station_areas[i] > np.max(station_areas) * 0.5:
                        # Isolated station with zero neighbors but significant area — still a spike
                        return False, f"Isolated station spike at {i}: area={station_areas[i]:.6f} with zero neighbors"
        
        return True, ""
    except Exception as e:
        return False, f"SAC scaling check error: {e}"

def _check_control_net_curvature(ctrl: np.ndarray) -> tuple[bool, str]:
    """
    Check NURBS control net for extreme local curvature that would create spikes.
    Returns (is_valid, error_message).
    """
    n_u, n_v, _ = ctrl.shape

    # Check u-direction (longitudinal) curvature
    # u-direction threshold: 155° — longitudinal control net naturally has higher
    # curvature at bow/stern taper and with extreme sheer/flare combinations.
    # v-direction threshold: 160° — vertical curvature allows extreme sections.
    max_angle_u = np.deg2rad(155.0)
    for j in range(n_v):
        for i in range(1, n_u - 1):
            p_prev = ctrl[i - 1, j]
            p_curr = ctrl[i, j]
            p_next = ctrl[i + 1, j]
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-10 and norm2 > 1e-10:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                if angle > max_angle_u:
                    return False, f"Control net spike at u={i}, v={j}: angle={np.rad2deg(angle):.1f}° > {np.rad2deg(max_angle_u):.1f}°"

    # Check v-direction (vertical) curvature
    max_angle_v = np.deg2rad(160.0)
    for i in range(n_u):
        for j in range(1, n_v - 1):
            p_prev = ctrl[i, j - 1]
            p_curr = ctrl[i, j]
            p_next = ctrl[i, j + 1]
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-10 and norm2 > 1e-10:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                if angle > max_angle_v:
                    return False, f"Control net spike at u={i}, v={j}: angle={np.rad2deg(angle):.1f}° > {np.rad2deg(max_angle_v):.1f}°"

    return True, ""


def _check_mesh_spikes(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    """
    Check for sharp spikes by measuring max angle between face normals at each vertex.
    Excludes bow/stern regions where a knife-edge crease is normal.
    Returns (is_valid, error_message).
    """
    if mesh is None or len(mesh.vertices) < 4:
        return True, ""
    face_normals = mesh.face_normals
    if len(face_normals) == 0:
        return True, ""
    vertex_normals = mesh.vertex_normals
    if vertex_normals is None or len(vertex_normals) != len(mesh.vertices):
        return True, ""
    vf_adj = mesh.vertex_faces
    verts = mesh.vertices
    x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
    hull_len = max(x_max - x_min, 1e-10)
    # Exclude bow/stern vertices from spike check where the natural knife-edge
    # crease occurs (port and starboard sides meet at the centerline).
    # Use a minimal bow/stern exclusion (3% of length) combined with centerline
    # proximity (y ≈ 0) to avoid false positives from the keel crease.
    near_bow_stern = (verts[:, 0] < x_min + 0.03 * hull_len) | (verts[:, 0] > x_max - 0.03 * hull_len)
    on_centerline = np.abs(verts[:, 1]) < 0.002
    bow_stern_mask = near_bow_stern & on_centerline
    # Exclude ALL centerline vertices: keel creates a natural crease along
    # the entire centerline (keel-hull junction), not just at bow/stern.
    keel_mask = np.abs(verts[:, 1]) < 0.002
    exclude_mask = bow_stern_mask | keel_mask
    max_angle_diff = 0.0
    spike_vertex_count = 0
    for vi in range(len(mesh.vertices)):
        if exclude_mask[vi]:
            continue
        faces_idx = vf_adj[vi]
        faces_idx = faces_idx[faces_idx >= 0]
        if len(faces_idx) < 2:
            continue
        fn = face_normals[faces_idx]
        vertex_max = 0.0
        for i in range(len(fn)):
            for j in range(i + 1, len(fn)):
                dot = np.clip(np.dot(fn[i], fn[j]), -1.0, 1.0)
                angle = np.arccos(dot)
                vertex_max = max(vertex_max, angle)
        if vertex_max > np.deg2rad(150):
            spike_vertex_count += 1
        max_angle_diff = max(max_angle_diff, vertex_max)
    if spike_vertex_count > max(10, int(len(mesh.vertices) * 0.01)):
        return False, f"Sharp spikes detected: {spike_vertex_count} non-bow/stern vertices with angle > 150°, max={np.rad2deg(max_angle_diff):.1f}°"
    return True, ""


def _check_mesh_convexity(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    """
    Check that the hull mesh has reasonable convexity ratio (not deformed/inverted).
    A valid hull has some concavity (bow/stern tapering). Combined meshes
    (hull+keel+bulb) naturally have lower ratios due to keel protrusion.
    Returns (is_valid, error_message).
    """
    try:
        volume = abs(mesh.volume)
        if volume <= 0:
            return True, ""
        convex = mesh.convex_hull
        if convex is not None and convex.volume > 0:
            convexity = volume / convex.volume
            if convexity < 0.35:
                return False, f"Hull too concave/deformed: convexity_ratio={convexity:.3f} (min 0.35)"
            if convexity > 1.0 + 1e-9:
                return False, f"Hull has no concavity (no bow/stern taper): convexity_ratio={convexity:.3f}"
    except Exception as e:
        logger.warning(f"Convexity check failed: {e}")
        return False, f"Convexity check error: {e}"
    return True, ""


def _check_mesh_self_intersection(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    """
    Check for self-intersections in the mesh.
    Returns (is_valid, error_message).
    """
    try:
        if hasattr(trimesh, 'repair') and hasattr(trimesh.repair, 'fix_self_intersection'):
            vol_before = abs(mesh.volume)
            mesh_copy = mesh.copy()
            result = trimesh.repair.fix_self_intersection(mesh_copy)
            if result is None:
                vol_after = abs(mesh_copy.volume)
                if abs(vol_after - vol_before) / max(vol_before, 1e-10) > 0.01:
                    return False, f"Self-intersection suspected: fix returned None but volume changed {abs(vol_after - vol_before) / max(vol_before, 1e-10) * 100:.1f}%"
            else:
                vol_after = abs(result.volume) if hasattr(result, 'volume') else abs(mesh_copy.volume)
                vol_ratio = abs(vol_after - vol_before) / max(vol_before, 1e-10)
                if vol_ratio > 0.001:
                    return False, f"Self-intersection detected: volume changed by {vol_ratio * 100:.2f}%"
                check_faces = result.faces if hasattr(result, 'faces') else mesh_copy.faces
                if len(check_faces) != len(mesh.faces):
                    return False, f"Self-intersection detected: face count changed from {len(mesh.faces)} to {len(check_faces)}"
        else:
            logger.debug("trimesh.repair.fix_self_intersection not available; skipping self-intersection check")
    except Exception as e:
        logger.error(f"Self-intersection check failed: {e}")
        return False, f"Self-intersection check error: {e}"
    return True, ""


def _check_local_normals(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    """
    Check for locally inverted normals (spikes/folds).
    Returns (is_valid, error_message).
    """
    if len(mesh.faces) < 4:
        return True, ""
    
    try:
        face_normals = mesh.face_normals
        face_adjacency = mesh.face_adjacency
    except Exception as e:
        return False, f"Local normal adjacency error: {e}"
    
    if len(face_adjacency) == 0:
        return True, ""
    
    # Check dot product of adjacent face normals
    # For a smooth hull, adjacent faces should have normals pointing roughly same direction
    # (dot product > cos(threshold)). Threshold ~ 120° = -0.5
    threshold = -0.5
    inverted_count = 0

    for f1, f2 in face_adjacency:
        if f1 < len(face_normals) and f2 < len(face_normals):
            dot = np.dot(face_normals[f1], face_normals[f2])
            if dot < threshold:
                inverted_count += 1

    if inverted_count > len(face_adjacency) * 0.03:
        return False, f"Excessive locally inverted normals: {inverted_count}/{len(face_adjacency)} adjacent pairs have dot < {threshold}"
    
    return True, ""


def _check_half_breadth_gradient(hull_stl: str = "", n_stations: int = 41,
                                  mesh: Optional[trimesh.Trimesh] = None) -> tuple[bool, str]:
    """
    Check for extreme half-breadth gradients between adjacent stations (spikes).
    Accepts either a path (hull_stl) or a Trimesh object (mesh).
    Returns (is_valid, error_message).
    """
    try:
        if mesh is not None:
            pass  # use provided mesh
        elif hull_stl:
            mesh = trimesh.load(hull_stl)
        else:
            return True, ""
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        vertices = mesh.vertices
        if len(vertices) == 0:
            return True, ""
        
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        LWL = x_max - x_min
        if LWL < 1e-6:
            return True, ""
        
        x_vals = np.linspace(x_min + 0.01 * LWL, x_max - 0.01 * LWL, n_stations)
        station_width = (x_vals[1] - x_vals[0]) * 2.0  # doubled from 1.0
        max_half_breadths = []
        
        for x in x_vals:
            slice_mask = np.abs(vertices[:, 0] - x) < station_width
            if np.sum(slice_mask) < 3:
                # Try wider slice
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
        
        # Max reasonable gradient: half-breadth can't change faster than ~2.0 per meter
        # (which would be a 45-degree angle)
        max_grad_threshold = 2.0
        if np.any(gradients > max_grad_threshold):
            idx = np.argmax(gradients)
            return False, f"Extreme half-breadth gradient at station {idx}: {gradients[idx]:.3f} > {max_grad_threshold}"
        
    except Exception as e:
        logger.warning(f"Half-breadth gradient check failed: {e}")
        return False, f"Half-breadth gradient check: {e}"
    return True, ""


def _check_keel_hull_intersection(stl_path: str = "",
                                   mesh: Optional[trimesh.Trimesh] = None) -> tuple[bool, str]:
    """
    Check for keel/bulb penetrating deep into hull volume.
    Surface attachment at the hull root is expected; bbox overlap alone is not a failure.
    Accepts either a path (stl_path) or a Trimesh object (mesh).
    """
    try:
        if mesh is not None:
            pass
        elif stl_path:
            mesh = trimesh.load(stl_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
        else:
            return True, ""

        if mesh.body_count <= 1:
            # Single body after merge — check z-extent as penetration proxy
            verts = mesh.vertices
            z_min = float(verts[:, 2].min())
            if not np.isfinite(z_min):
                return False, "Combined mesh has non-finite vertices"
            # Check that minimum z is physically reasonable (not negative volume)
            bbox = mesh.bounds
            height = bbox[1][2] - bbox[0][2]
            vol = abs(mesh.volume)
            # A healthy merged mesh has vol consistent with bbox
            expected_vol_min = 0.5 * height * bbox[1][0] * bbox[1][1]
            if vol < 0.1 * max(expected_vol_min, 1e-10):
                return False, f"Combined mesh volume {vol:.6f} too small for its bbox"
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
            except Exception as e:
                logger.warning(f"Penetration check failed for body {j}: {e}")
    except Exception as e:
        return False, f"Keel-hull intersection check error: {e}"
    return True, ""


def design_vector_to_dict(x: np.ndarray) -> dict:
    names = design_vector_names()
    return {n: float(x[i]) for i, n in enumerate(names)}


# ──────────────────────────────────────────────
# NURBS surface evaluation (B-spline, weights=1)
# ──────────────────────────────────────────────

_nurbs_cache: dict = {}


def _open_uniform_knots(n: int, p: int) -> np.ndarray:
    """Open uniform knot vector for n control points, degree p (clamped)."""
    nk = n + p + 1
    knots = np.zeros(nk)
    knots[:p+1] = 0.0
    knots[-(p+1):] = 1.0
    n_inner = n - p - 1
    if n_inner > 0:
        inner = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
        knots[p+1:p+1+n_inner] = inner
    return knots


def _bspline_basis(i: int, p: int, u: float, knots: np.ndarray) -> float:
    """Cox–de Boor: i-th B-spline basis function of degree p at parameter u."""
    if p == 0:
        return 1.0 if knots[i] <= u < knots[i+1] else (1.0 if np.isclose(u, knots[i+1]) and u >= knots[-1] else 0.0)
    left = 0.0
    d = knots[i+p] - knots[i]
    if d > 1e-15:
        left = (u - knots[i]) / d * _bspline_basis(i, p-1, u, knots)
    right = 0.0
    d = knots[i+p+1] - knots[i+1]
    if d > 1e-15:
        right = (knots[i+p+1] - u) / d * _bspline_basis(i+1, p-1, u, knots)
    return left + right


def _eval_nurbs_surface(ctrl: np.ndarray,
                         knots_u: np.ndarray, knots_v: np.ndarray,
                         u_vals: np.ndarray, v_vals: np.ndarray,
                         p: int, q: int) -> np.ndarray:
    """Evaluate a (weight=1) B-spline surface at every (u,v) pair.
    ctrl: (n_u, n_v, 3) control net.
    Returns: (len(u_vals), len(v_vals), 3) evaluation grid.
    """
    n_u, n_v, _ = ctrl.shape
    mu, mv = len(u_vals), len(v_vals)
    # Precompute basis-function tables
    Nu = np.zeros((n_u, mu))
    for i in range(n_u):
        for k, u in enumerate(u_vals):
            Nu[i, k] = _bspline_basis(i, p, u, knots_u)
    Nv = np.zeros((n_v, mv))
    for j in range(n_v):
        for k, v in enumerate(v_vals):
            Nv[j, k] = _bspline_basis(j, q, v, knots_v)
    # Evaluate tensor product
    out = np.zeros((mu, mv, 3))
    for i in range(n_u):
        Ni = Nu[i]           # (mu,)
        if np.all(Ni == 0):
            continue
        for j in range(n_v):
            Nij = Ni[:, None] * Nv[j][None, :]  # (mu, mv)
            out += Nij[:, :, None] * ctrl[i, j][None, None, :]
    return out


def _waterline_half_breadth(x_norm: np.ndarray, BWL: float,
                            Cp: float, Cm: float,
                            LCB: float = 12.5) -> np.ndarray:
    n = 0.5 + 2.0 * (1.0 - Cp)
    cm_factor = Cm / 0.75
    n = np.clip(n / cm_factor, 0.3, 3.0)
    lcb_shift = np.clip((LCB - 12.5) / 30.0, -0.15, 0.15)
    peak_pos = 0.45 + lcb_shift
    result = np.zeros_like(x_norm)
    for i, x in enumerate(x_norm):
        if x < peak_pos:
            z = min(0.9999, (peak_pos - x) / max(peak_pos, 1e-10))
            result[i] = (BWL / 2) * (1 - z ** n) * (1.0 - z * 0.06)
        else:
            z = min(0.9999, (x - peak_pos) / max(1 - peak_pos, 1e-10))
            result[i] = (BWL / 2) * (1 - z ** n) * (1.0 - z * 0.06)
    return np.clip(result, 0.0, None)


def _section_curve(z_norm: np.ndarray, y_wl: float, T: float,
                   deadrise: float, bilge_r: float, flare: float) -> np.ndarray:
    dr_rad = np.deg2rad(deadrise)
    fl_rad = np.deg2rad(flare)
    p = 1.6
    base = 1.0 - (-z_norm) ** p
    zn = np.clip(z_norm, -1.0, 0.0)
    wl_profile = -zn
    mid_wt = 4 * (-zn) * (1 + zn)
    dr_term = zn * np.tan(dr_rad)
    fl_term = wl_profile * (-zn) * np.tan(fl_rad) * 0.4
    br_term = mid_wt * bilge_r * 1.0
    y_raw = base + dr_term + fl_term + br_term
    if np.any(y_raw < -0.1):
        logger.debug(f"Section curve negative at keel (expected for V-hull): min={np.min(y_raw):.4f}")
    # Check if bilge creates a bulge below waterline (wider than waterline half-breadth)
    wl_val = max(y_raw[0], y_raw[-1]) if len(y_raw) > 0 else 1.0
    if wl_val > 1e-6 and np.any(y_raw > wl_val * 1.001):
        # Bilge+deadrise combination creates bulge below waterline, which would
        # require a hard clip that produces a crease. Instead, proportionally
        # reduce bilge effect to create a smooth, physically valid section.
        bulge = y_raw - wl_val
        excess = np.max(bulge)
        scale = max(0.0, 1.0 - excess * 4.0)
        logger.warning(f"Section curve bulge: excess={excess:.3f}, bilge_scaled={scale:.3f}")
        y_raw = base + dr_term + fl_term + br_term * scale
        if np.any(y_raw < -0.001):
            logger.warning(f"Section curve still deeply negative after bilge scale: min={np.min(y_raw):.4f}")
    y = np.clip(y_raw, 0.0, None)
    # Enforce monotonicity: y must be non-increasing from waterline to keel.
    # Process from waterline (z_norm=0, last element) to keel (z_norm=-1, first)
    y_fixed = np.minimum.accumulate(y[::-1])[::-1]
    return y_fixed


def _interp_param(x_norm: float, mid_val: float,
                  bow_factor: float = 1.5, stern_factor: float = 3.0) -> float:
    bow_factor = min(bow_factor, 2.0)
    stern_factor = min(stern_factor, 2.5)
    if x_norm < 0.5:
        t = x_norm / 0.5
        bow_val = mid_val * bow_factor
        return bow_val + t * (mid_val - bow_val)
    else:
        t = (x_norm - 0.5) / 0.5
        stern_val = mid_val * stern_factor
        return mid_val + t * (stern_val - mid_val)


def _interp_bilge(x_norm: float, mid_val: float,
                  end_factor: float = 0.3) -> float:
    if x_norm < 0.5:
        t = x_norm / 0.5
        end_val = mid_val * end_factor
        return end_val + t * (mid_val - end_val)
    else:
        t = (x_norm - 0.5) / 0.5
        end_val = mid_val * end_factor
        return mid_val + t * (end_val - mid_val)


def _sheer_height(x_norm: float, E: float = 0.20, SA: float = 0.15) -> float:
    bow = E * 1.3
    stern = E + SA * E * 0.8
    if x_norm < 0.5:
        t = x_norm / 0.5
        return bow + t * (E - bow)
    else:
        t = (x_norm - 0.5) / 0.5
        return E + t * (stern - E)


def _sac_form(x_norm: np.ndarray, Cp: float, LCB: float = 12.5) -> np.ndarray:
    n = Cp / max(1e-10, 1.0 - Cp)
    lcb_shift = (LCB - 12.5) / 60.0
    x_shifted = np.clip(x_norm - lcb_shift, -0.999, 0.999)
    return (1.0 - np.abs(x_shifted) ** n)


def _build_nurbs_control_net(x_dict: dict) -> np.ndarray:
    """Build a 7×5×3 B-spline control net for the port half-hull.

    u-direction (7 pts, 0=bow → 1=stern):
      [0.0, 0.15, 0.35, 0.50, 0.65, 0.85, 1.0]
    v-direction (5 pts, 0=keel → 1=sheer):
      [0.0, 0.25, 0.50, 0.75, 1.0]

    Returns: (7, 5, 3) array of (x, y, z) control points for y ≥ 0.
    """
    LWL = float(x_dict.get("LWL", 2.4))
    BWL = x_dict["BWL"]
    T_canoe = x_dict["T_canoe"]
    Cp = x_dict["Cp"]
    Cm = x_dict.get("Cm", 0.75)
    LCB_val = x_dict.get("LCB", 12.5)
    deadrise_val = x_dict["deadrise"]
    bilge_r = x_dict["bilge_r"]
    flare_param = x_dict["flare"]
    E_val = x_dict.get("E", 0.20)
    SA_val = x_dict.get("SA", 0.15)

    u_pos = np.array([0.0, 0.15, 0.35, 0.50, 0.65, 0.85, 1.0])
    v_lev = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    n_u, n_v = len(u_pos), len(v_lev)

    # Waterline half-breadth at each u control point
    y_wl = _waterline_half_breadth(u_pos, BWL, Cp, Cm, LCB=LCB_val)

    # Enforce y=0 at bow and stern
    y_wl[0] = 0.0
    y_wl[-1] = 0.0

    ctrl = np.zeros((n_u, n_v, 3))

    for i, xn in enumerate(u_pos):
        xs = xn * LWL
        if xn <= 0.0:
            T_local = 1e-6
        else:
            T_local = T_canoe * (1.0 - 0.3 * (1.0 - xn))
            T_local = max(T_local, 1e-6)

        dr = _interp_param(xn, deadrise_val)
        br = _interp_bilge(xn, bilge_r)
        fl = _interp_param(xn, flare_param, bow_factor=0.6, stern_factor=0.4)
        z_sheer = _sheer_height(xn, E_val, SA_val)

        for j, vf in enumerate(v_lev):
            z_pos = -T_local * (1.0 - vf) + z_sheer * vf

            if xn <= 0.0 or xn >= 1.0 - 1e-12:
                y_val = 0.0
            elif vf <= 1e-12:
                y_val = 0.0
            elif vf >= 0.999:
                y_val = y_wl[i] + z_sheer * np.tan(np.deg2rad(fl))
            else:
                zn = np.clip(z_pos / T_local, -1.0, 0.0)
                yf = _section_curve(np.array([zn]), y_wl[i], T_local, dr, br, fl)
                y_val = y_wl[i] * float(yf[0])

            ctrl[i, j] = [xs, y_val, z_pos]

    return ctrl


def _reality_check(x_dict: dict) -> None:
    """Rule 3: Fast geometry pre-check — runs in <1ms, rejects impossible shapes.
    
    Returns None (passes) or raises ValueError (fails). Returns binary result:
    no penalty, no warnings — just pass or fail.
    """
    LWL = x_dict["LWL"]
    BWL = x_dict["BWL"]
    T_canoe = x_dict["T_canoe"]
    D_keel = x_dict["D_keel"]
    keel_chord = x_dict["keel_chord"]
    Cp = x_dict["Cp"]
    Cm = x_dict.get("Cm", 0.75)

    # 1. Self-consistency: cross-parameter checks
    if D_keel > LWL:
        raise ValueError(f"RealityCheck: D_keel ({D_keel}) > LWL ({LWL})")
    if T_canoe + D_keel > LWL:
        raise ValueError(f"RealityCheck: T_canoe + D_keel ({T_canoe + D_keel}) > LWL ({LWL})")

    # 2. Minimum wall thickness: keel chord and draft must allow 1mm minimum
    min_thickness = 0.001
    if keel_chord < min_thickness and D_keel > min_thickness:
        raise ValueError(f"RealityCheck: keel_chord ({keel_chord}) < 1mm with D_keel > 1mm")
    if T_canoe < min_thickness:
        raise ValueError(f"RealityCheck: T_canoe ({T_canoe}) < 1mm")

    # 3. Non-zero volume condition: displacement estimate > 0
    est_vol = BWL * LWL * T_canoe * Cp * Cm
    if est_vol < 1e-10:
        raise ValueError(f"RealityCheck: estimated volume {est_vol} < 1e-10")

    # 4. Bulb containment: bulb must fit within hull envelope
    bulb_vol = x_dict["bulb_vol"]
    bulb_pos = x_dict["bulb_pos"]
    if bulb_vol > 1e-10:
        bulb_r = (3 * bulb_vol / (4 * np.pi)) ** (1/3)
        bulb_fwd = bulb_pos * LWL - 1.5 * bulb_r
        bulb_aft = bulb_pos * LWL + 1.5 * bulb_r
        if bulb_fwd < 0 or bulb_aft > LWL:
            raise ValueError(f"RealityCheck: bulb extends beyond hull [0, {LWL}]")


def generate_hull(design_vector: np.ndarray,
                  output_dir: Optional[str] = None,
                  LWL: float = 2.4,
                  target_displacement: Optional[float] = None,
                  config=None) -> tuple[str, str, dict, str]:
    # Rule 0: Raw design vector must be finite before any transformation
    if not np.all(np.isfinite(design_vector)):
        bad_idx = np.where(~np.isfinite(design_vector))[0]
        bad_vals = design_vector[bad_idx]
        raise ValueError(f"Non-finite values in raw design vector at indices {bad_idx.tolist()}: {bad_vals.tolist()}")
    # Rule 1+2: Convert raw GP vector to physical parameters via ratio + squashing
    x_dict = design_vector_to_physical(design_vector, config=config)
    # Sanity check: all parameters must be finite (baked-in by construction, but defend)
    for k, v in x_dict.items():
        if not np.isfinite(v):
            raise ValueError(f"Non-finite parameter {k}: {v}")
    LWL = float(x_dict["LWL"])
    BWL = x_dict["BWL"]
    T_canoe = x_dict["T_canoe"]
    D_keel = x_dict["D_keel"]
    Cp = x_dict["Cp"]
    Cm = x_dict.get("Cm", 0.75)
    LCB_val = x_dict.get("LCB", 12.5)
    E_val = x_dict.get("E", 0.20)
    SA_val = x_dict.get("SA", 0.15)
    deadrise_val = x_dict["deadrise"]
    bilge_r = x_dict["bilge_r"]
    flare_param = x_dict["flare"]
    keel_chord = x_dict["keel_chord"]
    bulb_vol = x_dict["bulb_vol"]
    bulb_pos = x_dict["bulb_pos"]
    keel_rake_val = x_dict["keel_rake"]
    ballast_frac = x_dict["ballast_frac"]

    # Rule 3: Reality Check — fast geometry pre-check (milliseconds)
    _reality_check(x_dict)

    # Cap bulb volume: bulb radius limited by keel chord
    max_bulb_r = 1.0 * (keel_chord * 0.5)
    max_bulb_vol = 4.0 / 3.0 * np.pi * max_bulb_r ** 3
    if bulb_vol > max_bulb_vol:
        logger.warning(f"Bulb volume capped: requested {x_dict['bulb_vol']:.6f}, max possible {max_bulb_vol:.6f} (keel_chord={keel_chord:.3f})")
    else:
        logger.debug(f"Bulb volume OK: {bulb_vol:.6f} <= {max_bulb_vol:.6f} (keel_chord={keel_chord:.3f})")
    bulb_vol = min(bulb_vol, max_bulb_vol)

    if output_dir is None:
        output_dir = "/tmp/hull_opt"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Iterative SAC scaling: compute actual SAC from analytic half-breadths,
    # then scale uniformly to match target volume.
    n_stations = 41
    u_vals = np.linspace(0.0, 1.0, n_stations)
    sac_avg_scale = 1.0
    for iteration in range(12):
        station_areas = np.zeros(n_stations)
        uw_station_areas = np.zeros(n_stations)
        for i in range(n_stations):
            xs = u_vals[i] * LWL
            x_norm = u_vals[i]
            if x_norm <= 0.0:
                T_local = 1e-6
            else:
                T_local = T_canoe * max(0.05, 1.0 - 0.3 * (1.0 - x_norm))
            z_sheer = _sheer_height(x_norm, E_val, SA_val)
            n_z = 20
            z_vals = np.linspace(-T_local, z_sheer, n_z)
            y_vals = np.zeros(n_z)
            if i == 0 or i == n_stations - 1:
                station_areas[i] = 0.0
                uw_station_areas[i] = 0.0
                continue
            for k in range(n_z):
                xq = xs - LWL / 2
                y_vals[k] = compute_half_breadth_analytic(
                    np.array([xq]), np.array([z_vals[k]]),
                    x_dict, LWL, sac_scale=sac_avg_scale).item(0)
                y_vals[k] = max(y_vals[k], 0.0)
            ds = np.diff(z_vals)
            y_mid = 0.5 * (y_vals[:-1] + y_vals[1:])
            station_areas[i] = float(np.sum(2.0 * y_mid * np.abs(ds)))
            if not np.isfinite(station_areas[i]):
                station_areas[i] = 0.0
            uw_mask = (z_vals[:-1] + z_vals[1:]) / 2 <= 0
            uw_station_areas[i] = float(np.sum(2.0 * y_mid[uw_mask] * np.abs(ds[uw_mask])))
            if not np.isfinite(uw_station_areas[i]):
                uw_station_areas[i] = 0.0
        if target_displacement is not None and target_displacement > 0:
            sac_volume_uw = float(np.trapezoid(uw_station_areas, u_vals * LWL))
            if not np.isfinite(sac_volume_uw) or sac_volume_uw <= 0:
                raise ValueError(f"Non-finite or non-positive SAC underwater volume: {sac_volume_uw}")
            vol_scale = target_displacement / max(1e-10, sac_volume_uw)
            if not np.isfinite(vol_scale) or vol_scale <= 0:
                raise ValueError(f"Non-finite or non-positive volume scale: {vol_scale}")
            sac_avg_scale *= vol_scale
            sac_avg_scale = np.clip(sac_avg_scale, 0.2, 5.0)
            if not np.isfinite(sac_avg_scale):
                raise ValueError(f"SAC scale became non-finite: {sac_avg_scale}")
            if abs(vol_scale - 1.0) < 0.01:
                break
        else:
            break
    if target_displacement is not None and target_displacement > 0:
        logger.info(f"SAC scaling: final scale={sac_avg_scale:.6f}, "
                    f"SAC volume={sac_volume_uw:.6f}, target={target_displacement:.6f}, "
                    f"n_iters={iteration+1}")
    else:
        logger.info(f"SAC scaling: final scale={sac_avg_scale:.6f} (no target displacement)")

    positive_areas = station_areas[station_areas > 1e-6]
    sac_scale_std = float(np.std(positive_areas) / max(np.mean(positive_areas), 1e-10)) if len(positive_areas) > 3 else 0.0

    # ── Generate hull mesh: port + starboard + deck strip ──
    n_z_stations = 16
    n_ports = n_stations * n_z_stations
    verts = np.empty((2 * n_ports, 3), dtype=np.float64)

    z_offsets = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_z_stations)))
    z_offsets[0] = 0.0
    z_offsets[-1] = 1.0

    for i in range(n_stations):
        xs = u_vals[i] * LWL
        x_norm = u_vals[i]
        if x_norm <= 0.0:
            T_local = 1e-6
        else:
            T_local = T_canoe * max(0.05, 1.0 - 0.3 * (1.0 - x_norm))
        z_sheer = _sheer_height(x_norm, E_val, SA_val)
        for j in range(n_z_stations):
            z_pos = -T_local * (1.0 - z_offsets[j]) + z_sheer * z_offsets[j]
            if i == 0 or i == n_stations - 1 or j == 0:
                y_val = 0.0
            else:
                xq = xs - LWL / 2
                y_val = compute_half_breadth_analytic(
                    np.array([xq]), np.array([z_pos]),
                    x_dict, LWL, sac_scale=sac_avg_scale).item(0)
                y_val = max(y_val, 0.0)
            idx = i * n_z_stations + j
            verts[idx] = [xs, y_val, z_pos]
            verts[n_ports + idx] = [xs, -y_val, z_pos]

    # Phase 4: Force bow (first) and stern (last) stations to zero beam at all depths.
    # This guarantees pointed ends and prevents the optimizer from exploiting
    # near-zero-but-nonzero endpoint sections to game volume.
    n_bow = n_z_stations
    bow_slice = slice(0, n_bow)
    stern_slice = slice((n_stations - 1) * n_z_stations, n_stations * n_z_stations)
    for s in [bow_slice, stern_slice]:
        verts[s, 1] = 0.0
        verts[slice(s.start + n_ports, s.stop + n_ports), 1] = 0.0

    hull_faces = []
    for i in range(n_stations - 1):
        # Port side
        for j in range(n_z_stations - 1):
            v0 = i * n_z_stations + j
            v1 = i * n_z_stations + j + 1
            v2 = (i + 1) * n_z_stations + j + 1
            v3 = (i + 1) * n_z_stations + j
            if not (abs(verts[v0, 1]) < 1e-9 and abs(verts[v1, 1]) < 1e-9 and abs(verts[v2, 1]) < 1e-9):
                hull_faces.append([v0, v2, v1])
            if not (abs(verts[v0, 1]) < 1e-9 and abs(verts[v3, 1]) < 1e-9 and abs(verts[v2, 1]) < 1e-9):
                hull_faces.append([v0, v3, v2])
        # Starboard side
        for j in range(n_z_stations - 1):
            v0 = n_ports + i * n_z_stations + j
            v1 = n_ports + i * n_z_stations + j + 1
            v2 = n_ports + (i + 1) * n_z_stations + j + 1
            v3 = n_ports + (i + 1) * n_z_stations + j
            if not (abs(verts[v0, 1]) < 1e-9 and abs(verts[v1, 1]) < 1e-9 and abs(verts[v2, 1]) < 1e-9):
                hull_faces.append([v0, v2, v1])
            if not (abs(verts[v0, 1]) < 1e-9 and abs(verts[v3, 1]) < 1e-9 and abs(verts[v2, 1]) < 1e-9):
                hull_faces.append([v0, v3, v2])
        # Deck strip (port sheer to starboard sheer)
        p0 = i * n_z_stations + (n_z_stations - 1)
        p3 = (i + 1) * n_z_stations + (n_z_stations - 1)
        s0 = n_ports + i * n_z_stations + (n_z_stations - 1)
        s2 = n_ports + (i + 1) * n_z_stations + (n_z_stations - 1)
        if abs(verts[p0, 1]) > 1e-10 or abs(verts[p3, 1]) > 1e-10:
            hull_faces.append([p0, p3, s2])
        if abs(verts[s2, 1]) > 1e-10 or abs(verts[s0, 1]) > 1e-10:
            hull_faces.append([p0, s2, s0])

    hull_mesh = trimesh.Trimesh(
        vertices=verts, faces=np.array(hull_faces, dtype=np.int64), process=False)

    # Merge coincident vertices (port/starboard at y=0)
    hull_mesh.merge_vertices()

    # Remove degenerate faces created by merging
    keep = ~((hull_mesh.faces[:, 0] == hull_mesh.faces[:, 1]) |
             (hull_mesh.faces[:, 1] == hull_mesh.faces[:, 2]) |
             (hull_mesh.faces[:, 0] == hull_mesh.faces[:, 2]))
    if not keep.all():
        hull_mesh.update_faces(keep)

    # Remove duplicate faces
    sorted_faces = np.sort(hull_mesh.faces, axis=1)
    _, uniq_idx = np.unique(sorted_faces, axis=0, return_index=True)
    if len(uniq_idx) < len(hull_mesh.faces):
        hull_mesh.update_faces(uniq_idx)

    # Minimum face count: prevent degenerate/near-empty meshes after fixing
    MIN_HULL_FACES = 500
    if len(hull_mesh.faces) < MIN_HULL_FACES:
        raise ValueError(
            f"Hull mesh has only {len(hull_mesh.faces)} faces "
            f"(minimum {MIN_HULL_FACES}) — mesh too degenerate"
        )

    # Fix non-manifold edges at bow/stern (port+starboard keel collapse)
    # Remove one face from each triple-shared edge
    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(hull_mesh.faces):
        for pi in range(3):
            e = tuple(sorted([face[pi], face[(pi+1)%3]]))
            edge_to_faces[e].append(fi)
    non_manifold_edges = {e: fs for e, fs in edge_to_faces.items() if len(fs) >= 3}
    if non_manifold_edges:
        faces_to_kill = set()
        for e, fs in non_manifold_edges.items():
            for fi in fs[2:]:
                faces_to_kill.add(fi)
        if faces_to_kill:
            hull_mesh.update_faces(
                [fi for fi in range(len(hull_mesh.faces)) if fi not in faces_to_kill])

    # Fill remaining holes
    if not hull_mesh.is_watertight:
        hull_mesh.fill_holes()

    # Export hull-only STL (for hydrostatics / GZ — keel excluded)
    hull_stl = str(out / "hull_geometry.stl")
    hull_mesh = hull_mesh.process(validate=True)
    if hull_mesh.volume < 0:
        hull_mesh.invert()
    
    # Validate hull-only mesh for degenerate shapes
    is_valid, err_msg = _validate_hull_mesh(hull_mesh, x_dict, LWL)
    if not is_valid:
        raise ValueError(f"Hull mesh validation failed: {err_msg}")

    is_valid, err_msg = _check_control_net_curvature(
        _build_nurbs_control_net(x_dict))
    if not is_valid:
        raise ValueError(f"Control net validation failed: {err_msg}")
    
    is_valid, err_msg = _check_mesh_self_intersection(hull_mesh)
    if not is_valid:
        raise ValueError(f"Self-intersection check failed: {err_msg}")
    
    is_valid, err_msg = _check_local_normals(hull_mesh)
    if not is_valid:
        raise ValueError(f"Local normal validation failed: {err_msg}")

    is_valid, err_msg = _check_mesh_spikes(hull_mesh)
    if not is_valid:
        raise ValueError(f"Mesh spike detection failed: {err_msg}")

    is_valid, err_msg = _check_mesh_convexity(hull_mesh)
    if not is_valid:
        raise ValueError(f"Mesh convexity check failed: {err_msg}")

    is_valid, err_msg = _check_half_breadth_gradient(mesh=hull_mesh)
    if not is_valid:
        raise ValueError(f"Half-breadth gradient check failed: {err_msg}")

    hull_mesh.export(hull_stl)

    # Hydrostatics from hull-only mesh
    hydro = _compute_hydrostatics(
        hull_mesh, LWL, BWL, Cp, T_canoe, D_keel,
        bulb_vol, ballast_frac, target_displacement,
        sac_volume=float(np.trapezoid(uw_station_areas, u_vals * LWL)),
        keel_chord=keel_chord,
        sac_scale_factor=sac_avg_scale,
        sac_scale_std=sac_scale_std,
        station_areas=uw_station_areas)

    # Now add keel + bulb to the hull mesh
    if D_keel > 0.01 and keel_chord > 0.01:
        keel_sweep = keel_rake_val
        keel_tc = 0.12
        keel_x_pos = (0.4 * LWL if bulb_vol < 1e-6 else bulb_pos * LWL)
        bulb_AR = 4.0
        keel_mesh = _make_keel(keel_chord, D_keel, BWL * 0.06,
                               keel_x_pos, keel_sweep, keel_tc)
        keel_x_norm = keel_x_pos / LWL
        T_keel = T_canoe * max(0.05, 1.0 - 0.3 * (1.0 - keel_x_norm))
        keel_mesh.apply_translation([0, 0, -T_keel])
        try:
            hull_mesh = trimesh.util.concatenate(hull_mesh, keel_mesh)
        except Exception as e:
            logger.error(f"Keel concatenation failed: {e}")
            raise
        if bulb_vol > 1e-6:
            bulb_mesh = _make_bulb(keel_chord, D_keel, bulb_vol, bulb_AR, keel_sweep, keel_x_pos)
            bulb_mesh.apply_translation([0, 0, -T_keel])
            try:
                hull_mesh = trimesh.util.concatenate(hull_mesh, bulb_mesh)
            except Exception as e:
                logger.error(f"Bulb concatenation failed: {e}")
                raise

    if not hull_mesh.is_watertight:
        hull_mesh.fill_holes()
        if not hull_mesh.is_watertight:
            raise ValueError("Combined mesh not watertight after keel/bulb attachment")

    # Check keel-hull penetration on pre-process mesh (before process merges bodies)
    is_valid, err_msg = _check_keel_hull_intersection(mesh=hull_mesh)
    if not is_valid:
        raise ValueError(f"Keel-hull intersection pre-check: {err_msg}")

    # Fix normals on combined mesh: use trimesh's fix_normals
    try:
        hull_mesh.fix_normals()
    except Exception as e:
        logger.warning(f"fix_normals() failed: {e}, using multi-pass normal fix")
        # Multi-pass: compute total volume sign from face contributions
        verts = hull_mesh.vertices
        faces = hull_mesh.faces
        cross = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                         verts[faces[:, 2]] - verts[faces[:, 0]])
        dot = np.sum(cross * hull_mesh.face_normals, axis=1)
        neg = dot < 0
        if neg.sum() > 0:
            # Flip faces with inward normals (face normal points opposite to edge cross product)
            faces_fixed = hull_mesh.faces.copy()
            faces_fixed[neg] = faces_fixed[neg][:, [0, 2, 1]]
            hull_mesh = trimesh.Trimesh(
                vertices=hull_mesh.vertices.copy(), faces=faces_fixed, process=False)

    hull_mesh = hull_mesh.process(validate=True)
    # B3: If process merged all bodies, the penetration check is skipped. Warn and verify.
    if hull_mesh.body_count <= 1:
        logger.warning("Combined mesh body_count=1 after process() — keel-hull junction merged")
    verts = hull_mesh.vertices
    faces = hull_mesh.faces
    cross = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                     verts[faces[:, 2]] - verts[faces[:, 0]])
    total_vol = np.sum(verts[faces[:, 0]] * cross, axis=1).sum() / 6.0
    if total_vol < 0:
        hull_mesh.invert()

    # Validate final combined mesh
    is_valid, err_msg = _check_mesh_self_intersection(hull_mesh)
    if not is_valid:
        raise ValueError(f"Combined mesh self-intersection: {err_msg}")
    
    is_valid, err_msg = _check_local_normals(hull_mesh)
    if not is_valid:
        raise ValueError(f"Combined mesh normal validation failed: {err_msg}")

    is_valid, err_msg = _check_mesh_spikes(hull_mesh)
    if not is_valid:
        raise ValueError(f"Combined mesh spike detection failed: {err_msg}")

    is_valid, err_msg = _check_mesh_convexity(hull_mesh)
    if not is_valid:
        raise ValueError(f"Combined mesh convexity check failed: {err_msg}")
    
    is_valid, err_msg = _check_sac_scaling_station_variation(x_dict, LWL, sac_avg_scale)
    if not is_valid:
        raise ValueError(f"SAC scaling variation: {err_msg}")

    is_valid, err_msg = _validate_hull_mesh(hull_mesh, x_dict, LWL)
    if not is_valid:
        raise ValueError(f"Combined mesh validation failed: {err_msg}")

    # Rule 6: Element quality — log warning (constraints.py handles penalty)
    try:
        from hull_opt.constraints import _check_element_quality_continuous
        # Remove sliver faces before quality check to avoid false positives
        # from NACA foil trailing edge degenerate triangles
        v = hull_mesh.vertices
        f = hull_mesh.faces
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.sqrt(np.sum(cross ** 2, axis=1))
        median_area = float(np.median(areas))
        if median_area > 0:
            min_area_thresh = max(median_area * 0.02, 1e-12)
            good = areas >= min_area_thresh
            if good.sum() >= 3 and good.sum() < len(f):
                hull_mesh.update_faces(good)
                hull_mesh.remove_unreferenced_vertices()
        eq_viol = _check_element_quality_continuous(hull_mesh)
        if eq_viol > 0.5:
            logger.warning(f"Mesh element quality: {eq_viol:.4f} (may affect solver)")
    except Exception as e:
        logger.debug(f"Element quality check failed: {e}")

    stl_path = str(out / "hull.stl")
    hull_mesh.export(stl_path)

    is_valid, err_msg = _check_keel_hull_intersection(stl_path)
    if not is_valid:
        raise ValueError(f"Keel-hull intersection: {err_msg}")

    # #region agent log
    import json as _json, time as _time
    try:
        with open("/home/anon/apps/boat/.cursor/debug-b990d8.log", "a") as _lf:
            _lf.write(_json.dumps({"sessionId": "b990d8", "hypothesisId": "A", "location": "geometry.py:generate_hull", "message": "hull exported", "data": {"stl_path": stl_path, "hull_stl": hull_stl, "sac_scale": float(sac_avg_scale), "underwater_vol": float(hydro.get("underwater_volume", 0))}, "runId": "pre-fix", "timestamp": int(_time.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    # SAC csv from station areas
    sac_actual = [[u_vals[i] * LWL, station_areas[i]] for i in range(n_stations)]
    sac_path = str(out / "sac.csv")
    np.savetxt(sac_path, np.array(sac_actual), delimiter=",", header="x,area", comments="")

    return stl_path, sac_path, hydro, hull_stl


def _make_keel(chord: float, depth: float, thickness: float,
               x_pos: float, sweep_deg: float, tc: float) -> trimesh.Trimesh:
    sweep_rad = np.deg2rad(sweep_deg)
    n_span = 10
    n_chord = 20
    verts = []
    faces = []
    naca_max = _naca_thickness(0.3, tc)
    for i, z in enumerate(np.linspace(0, -depth, n_span)):
        frac = z / -depth if depth > 0 else 0
        local_chord = chord * (1.0 - 0.5 * frac)
        taper = 1.0 - 0.5 * frac
        half_thick = 0.5 * thickness * taper
        scale = half_thick / max(1e-10, naca_max)
        sweep_shift = z * np.tan(sweep_rad)
        x_start = -local_chord / 2 + sweep_shift + x_pos
        for j in range(n_chord):
            xi = j / (n_chord - 1)
            x_pos_vert = x_start + xi * local_chord
            y_half = _naca_thickness(xi, tc) * scale
            if j == 0 or j == n_chord - 1:
                y_half = max(y_half, local_chord * 0.01)
            verts.append([x_pos_vert, y_half, z])
        for j in range(n_chord - 1):
            a0 = i * n_chord + j
            a1 = i * n_chord + j + 1
            b0 = (i + 1) * n_chord + j
            b1 = (i + 1) * n_chord + j + 1
            if z > -depth + 0.001:
                faces.append([a0, b1, a1])
                faces.append([a0, b0, b1])
    n0 = len(verts)
    for i in range(n0):
        xv, yv, zv = verts[i]
        verts.append([xv, -yv, zv])
    all_faces = faces + [[a + n0, c + n0, b + n0] for a, b, c in faces]
    tip0 = (n_span - 1) * n_chord
    for j in range(n_chord - 1):
        a = tip0 + j
        b = tip0 + j + 1
        sa = a + n0
        sb = b + n0
        all_faces.append([a, b, sa])
        all_faces.append([b, sb, sa])
    for i_span in range(n_span - 1):
        a0 = i_span * n_chord
        a1 = (i_span + 1) * n_chord
        sa0 = a0 + n0
        sa1 = a1 + n0
        all_faces.append([a0, a1, sa0])
        all_faces.append([a1, sa1, sa0])
    for i_span in range(n_span - 1):
        a0 = i_span * n_chord + n_chord - 1
        a1 = (i_span + 1) * n_chord + n_chord - 1
        sa0 = a0 + n0
        sa1 = a1 + n0
        all_faces.append([a0, sa0, a1])
        all_faces.append([a1, sa0, sa1])
    # Root closure (z=0 end) - connect port root to starboard root
    for j in range(n_chord - 1):
        a0 = j
        a1 = j + 1
        sa0 = a0 + n0
        sa1 = a1 + n0
        all_faces.append([a0, sa0, a1])
        all_faces.append([a1, sa0, sa1])
    keel = trimesh.Trimesh(vertices=np.array(verts, dtype=np.float64),
                           faces=np.array(all_faces, dtype=np.int64))
    keel.process(validate=True)
    # Check for degenerate sliver faces from NACA trailing edge
    faces = keel.faces
    verts = keel.vertices
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.sqrt(np.sum(cross ** 2, axis=1))
    median_area = float(np.median(areas))
    if median_area > 0:
        min_area_thresh = max(median_area * 0.01, 1e-12)
        good = areas >= min_area_thresh
        if good.sum() < 3:
            raise ValueError("Keel mesh has insufficient valid faces after sliver removal")
        keel.update_faces(good)
        keel.remove_unreferenced_vertices()
    return keel


def _naca_thickness(x: float, t: float = 0.12) -> float:
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x
                    - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)


def _make_bulb(chord: float, depth: float, volume: float,
               ar: float, sweep_deg: float, x_pos: float) -> trimesh.Trimesh:
    x_scale = max(ar / 3, 0.5) if ar > 0 else 1.5
    z_scale = 0.8
    vol_compensated = volume / (x_scale * z_scale)
    r = (3 * vol_compensated / (4 * np.pi)) ** (1 / 3)
    sweep_rad = np.deg2rad(sweep_deg)
    x_c = -depth * np.tan(sweep_rad) + x_pos
    z_c = -depth
    bulb = trimesh.creation.icosphere(subdivisions=3, radius=r)
    bulb.vertices[:, 0] *= x_scale
    bulb.vertices[:, 2] *= z_scale
    bulb.apply_translation([x_c, 0, z_c])
    return bulb


def compute_half_breadth_analytic(xq: np.ndarray, zq: np.ndarray,
                                    x_dict: dict, LWL: float,
                                    sac_scale: float = 1.0) -> np.ndarray:
    BWL = x_dict["BWL"]
    T_canoe = x_dict["T_canoe"]
    deadrise = x_dict["deadrise"]
    bilge = x_dict["bilge_r"]
    flare = x_dict["flare"]
    Cp = x_dict["Cp"]
    Cm = x_dict.get("Cm", 0.75)

    xq = np.asarray(xq, dtype=float)
    zq = np.asarray(zq, dtype=float)
    result = np.zeros_like(xq)

    # Below waterline (z <= 0): use section curve with deadrise, bilge, flare
    mask_below = (np.abs(xq) <= LWL / 2) & (zq >= -T_canoe) & (zq <= 0)
    if np.any(mask_below):
        xs = xq[mask_below]
        zs = zq[mask_below]
        x_norm = (xs + LWL / 2) / LWL

        x_dict_lcb = x_dict.get("LCB", 12.5)
        y_wl = _waterline_half_breadth(x_norm, BWL, Cp, Cm=Cm, LCB=x_dict_lcb)
        T_local = T_canoe * (1.0 - 0.3 * (1.0 - x_norm))
        z_norm = np.clip(zs / T_local, -1.0, 0.0)

        dr = np.array([_interp_param(xn, deadrise) for xn in x_norm])
        br = np.array([_interp_bilge(xn, bilge) for xn in x_norm])
        fl = np.array([_interp_param(xn, flare, bow_factor=0.6, stern_factor=0.4) for xn in x_norm])

        y_local = np.zeros_like(xs)
        for i in range(len(xs)):
            yf = _section_curve(np.array([z_norm[i]]), y_wl[i], T_local[i], dr[i], br[i], fl[i])
            y_local[i] = y_wl[i] * yf[0]

        result[mask_below] = y_local * sac_scale

    # Above waterline (z > 0): flare from waterline half-breadth
    mask_above = (np.abs(xq) <= LWL / 2) & (zq > 0) & (zq <= 0.5)
    if np.any(mask_above):
        xs = xq[mask_above]
        zs = zq[mask_above]
        x_norm = (xs + LWL / 2) / LWL
        x_dict_lcb = x_dict.get("LCB", 12.5)
        y_wl = _waterline_half_breadth(x_norm, BWL, Cp, Cm=Cm, LCB=x_dict_lcb)
        fl = np.array([_interp_param(xn, flare, bow_factor=0.6, stern_factor=0.4) for xn in x_norm])
        y_flare = y_wl + zs * np.tan(np.deg2rad(fl))
        result[mask_above] = y_flare * sac_scale
    return result


def _compute_hydrostatics(mesh: trimesh.Trimesh, LWL: float,
                          BWL: float, Cp: float, T_canoe: float,
                          D_keel: float, bulb_vol: float,
                          ballast_frac: float,
                          target_displacement: Optional[float] = None,
                          sac_volume: Optional[float] = None,
                          keel_chord: float = 0.2,
                          sac_scale_factor: float = 1.0,
                          sac_scale_std: float = 0.0,
                          station_areas: Optional[np.ndarray] = None) -> dict:
    total_mesh_vol = max(abs(mesh.volume), 0.0)
    total_mesh_vol = total_mesh_vol if np.isfinite(total_mesh_vol) and total_mesh_vol > 0 else None
    # Compute underwater volume from mesh (slice below z=0)
    underwater_mesh = None
    submerged_centroid = None
    try:
        plane_normal = np.array([0, 0, -1])
        plane_origin = np.array([0, 0, 0])
        submerged = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal, plane_origin, cap=True
        )
        if submerged is not None and submerged.vertices.shape[0] >= 4:
            uv = abs(submerged.volume)
            if np.isfinite(uv) and uv > 0:
                underwater_mesh = uv
                submerged_centroid = submerged.center_mass
    except Exception as e:
        logger.warning(f"Underwater volume slice failed: {e}")
    if underwater_mesh is not None:
        volume = underwater_mesh
        if sac_volume is not None and sac_volume > 0:
            keel_vol_est = D_keel * keel_chord * (BWL * 0.06) * 0.5
            sac_vol_hull_only = sac_volume
            mesh_hull_vol_est = volume - keel_vol_est - bulb_vol
            vol_ratio = mesh_hull_vol_est / max(1e-10, sac_vol_hull_only)
            if vol_ratio > 1.5 or vol_ratio < 0.5:
                logger.warning(f"Mesh hull volume ~{mesh_hull_vol_est:.4f} vs SAC volume {sac_vol_hull_only:.4f} ratio={vol_ratio:.3f}")
    else:
        volume = max(total_mesh_vol, 1e-10) if total_mesh_vol is not None else 1e-10
    if sac_volume is not None and sac_volume > 0 and target_displacement is not None and target_displacement > 0:
        ratio = sac_volume / target_displacement
        if ratio > 1.5 or ratio < 0.5:
            logger.warning(f"SAC volume {sac_volume:.4f} far from target {target_displacement:.4f}")
    if volume > 0 and target_displacement is not None and target_displacement > 0:
        mesh_target_ratio = volume / target_displacement
        if mesh_target_ratio < 0.3 or mesh_target_ratio > 2.0:
            logger.warning(f"Mesh underwater volume {volume:.4f} vs target {target_displacement:.4f} ratio={mesh_target_ratio:.3f} - possible SAC gaming")
    if total_mesh_vol is not None and volume > 0:
        total_ratio = total_mesh_vol / volume
        if total_ratio > 2.5 or total_ratio < 0.8:
            logger.warning(f"Total mesh volume {total_mesh_vol:.4f} vs underwater volume {volume:.4f} ratio={total_ratio:.3f}")
    if submerged_centroid is not None and np.all(np.isfinite(submerged_centroid)):
        cb_z = float(submerged_centroid[2])
        if cb_z > 0.0 or cb_z < -T_canoe:
            center_mass = np.array([submerged_centroid[0], submerged_centroid[1], -T_canoe * 0.4])
        else:
            center_mass = submerged_centroid
    else:
        center_mass = mesh.center_mass
        if not np.all(np.isfinite(center_mass)):
            center_mass = np.array([0.0, 0.0, -T_canoe * 0.4])
        else:
            cb_z = float(center_mass[2])
            if cb_z > 0.0 or cb_z < -T_canoe:
                center_mass[2] = -T_canoe * 0.4

    # Compute actual prismatic coefficient from station areas (before waterplane fallback needs it)
    actual_Cp = Cp
    if station_areas is not None and len(station_areas) > 0 and LWL > 0:
        max_station_idx = int(np.argmax(station_areas))
        actual_Am = float(station_areas[max_station_idx])
        if actual_Am > 0:
            actual_Cp = volume / (actual_Am * LWL)
            actual_Cp = float(actual_Cp)

    # Compute waterplane properties from mesh when possible
    wp_area = 0.0
    use_mesh_wp = True
    try:
        plane_normal = np.array([0, 0, -1])
        plane_origin = np.array([0, 0, 0])
        wp_mesh = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal, plane_origin, cap=True
        )
        if wp_mesh is None or wp_mesh.vertices.shape[0] < 4:
            use_mesh_wp = False
        else:
            verts_wp = wp_mesh.vertices
            faces_wp = wp_mesh.faces
            cap_mask = np.abs(verts_wp[:, 2]) < 1e-6
            if cap_mask.sum() < 3:
                use_mesh_wp = False
            else:
                cap_faces = faces_wp[np.all(cap_mask[faces_wp], axis=1)]
                if cap_faces.shape[0] < 1:
                    use_mesh_wp = False
    except Exception as e:
        logger.warning(f"Waterplane mesh processing failed: {e}")
        use_mesh_wp = False

    if use_mesh_wp:
        verts_2d = verts_wp.copy()
        verts_2d[:, 2] = 0.0
        Ix = 0.0
        Iy = 0.0
        for tri in cap_faces:
            v = verts_2d[tri]
            x1, y1 = v[0, 0], v[0, 1]
            x2, y2 = v[1, 0], v[1, 1]
            x3, y3 = v[2, 0], v[2, 1]
            cross = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            area_tri = 0.5 * abs(cross)
            Ix += area_tri * (y1**2 + y2**2 + y3**2 + y1*y2 + y2*y3 + y3*y1) / 6.0
            Iy += area_tri * (x1**2 + x2**2 + x3**2 + x1*x2 + x2*x3 + x3*x1) / 6.0
            wp_area += area_tri
    else:
        C_wp = 0.35 + 0.6 * max(actual_Cp, 0.5)
        B_eff = BWL * max(sac_scale_factor, 1.0)
        Ix = C_wp * (1.0 / 12.0) * LWL * B_eff ** 3
        Iy = C_wp * (1.0 / 12.0) * B_eff * LWL ** 3

    BM = Ix / max(1e-10, volume)
    BML = Iy / max(1e-10, volume)
    Am = volume / max(1e-10, Cp * LWL)
    rho = 1025.0
    total_mass = volume * rho
    bulb_mass = bulb_vol * 11340  # Always from actual bulb geometry
    keel_mass = max(0, D_keel * keel_chord * (BWL * 0.06) * 0.5 * 1025)  # Always from actual keel geometry
    ballast_mass = total_mass * ballast_frac  # Ballast is additional internal mass
    hull_mass = total_mass - bulb_mass - keel_mass - ballast_mass
    if hull_mass < 0:
        hull_mass = 0
    bulb_cg_z = -(T_canoe + D_keel)
    keel_cg_z = -(T_canoe + D_keel * 0.5)
    ballast_cg_z = -(T_canoe + D_keel * 0.5)  # ballast distributed in keel region
    hull_cg_z = float(center_mass[2]) if np.isfinite(center_mass[2]) and abs(center_mass[2]) < 1 else -T_canoe * 0.4
    cg_z = (hull_mass * hull_cg_z + keel_mass * keel_cg_z + bulb_mass * bulb_cg_z + ballast_mass * ballast_cg_z) / max(1e-10, total_mass)

    underwater_vol = volume
    return {
        "nabla": volume,
        "underwater_volume": float(underwater_vol),
        "CB_x": float(center_mass[0]),
        "CB_y": float(center_mass[1]),
        "CB_z": float(center_mass[2]),
        "Ix": float(Ix),
        "Iy": float(Iy),
        "waterplane_area": float(wp_area),
        "BM": float(BM),
        "BML": float(BML),
        "Am": float(Am),
        "Cp": Cp,
        "actual_Cp": actual_Cp,
        "B": BWL,
        "LWL": LWL,
        "T_canoe": T_canoe,
        "D_keel": D_keel,
        "target_nabla": target_displacement if target_displacement is not None else volume,
        "rho": rho,
        "cg_z": float(cg_z),
        "total_mass_kg": float(total_mass),
        "bulb_mass_kg": float(bulb_mass),
        "ballast_frac": ballast_frac,
        "sac_scale_factor": sac_scale_factor,
        "sac_scale_std": sac_scale_std,
    }
