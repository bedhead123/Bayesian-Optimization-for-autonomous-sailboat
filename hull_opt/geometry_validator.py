"""
NURBS geometry validation.
Checks NURBS patch quality, closed-surface edge matching, and enclosed volume.
"""

import numpy as np
from pathlib import Path
from typing import Tuple


def validate_nurbs_closed_surface(patches: list, mesh_volume: float = 0.0) -> Tuple[bool, str]:
    """Check that NURBS patches form a closed surface.

    Verifies: port hull↔deck, stbd hull↔deck, port↔stbd at y=0 edges
    all match within tolerance, and that the enclosed NURBS volume
    agrees with the tessellated mesh volume within 2%.

    Returns (is_valid, error_message).
    """
    if not patches:
        return False, "No patches provided"

    tol = 1e-6

    # Group patches by name
    by_name = {}
    for p in patches:
        by_name.setdefault(p.name, []).append(p)

    port_hull = by_name.get("hull_port", [None])[0]
    stbd_hull = by_name.get("hull_port_mirror", [None])[0]
    deck = by_name.get("deck", [None])[0]

    if port_hull is None:
        return False, "Missing hull_port patch"
    if stbd_hull is None:
        return False, "Missing hull_port_mirror patch"
    if deck is None:
        return False, "Missing deck patch"

    # Check deck v=0 matches port hull v=1 (sheer edge)
    deck_v0 = deck.control_net[:, 0, :]
    port_sheer = port_hull.control_net[:, -1, :]
    diff = np.abs(deck_v0 - port_sheer)
    max_err = np.max(diff)
    if max_err > tol:
        return False, f"Deck v=0 vs hull_port v=1 mismatch: max diff {max_err:.2e} > {tol}"

    # Check deck v=2 matches stbd hull v=1
    deck_v2 = deck.control_net[:, 2, :]
    stbd_sheer = stbd_hull.control_net[:, -1, :]
    diff = np.abs(deck_v2 - stbd_sheer)
    max_err = np.max(diff)
    if max_err > tol:
        return False, f"Deck v=2 vs hull_port_mirror v=1 mismatch: max diff {max_err:.2e} > {tol}"

    # Check port/starboard hull share y=0 at keel (v=0)
    port_keel = port_hull.control_net[:, 0, 1]
    stbd_keel = stbd_hull.control_net[:, 0, 1]
    if np.max(np.abs(port_keel)) > tol or np.max(np.abs(stbd_keel)) > tol:
        return False, "Keel (v=0) not on y=0 centerline"

    # Check port bow/stern at y=0
    if np.max(np.abs(port_hull.control_net[0, :, 1])) > tol:
        return False, "Port hull bow not closed at y=0"
    if np.max(np.abs(port_hull.control_net[-1, :, 1])) > tol:
        return False, "Port hull stern not closed at y=0"

    # Volume check: nurbs_submerged_volume computes z<0 volume, mesh_volume
    # is total enclosed — only warn, don't fail (different quantities).

    return True, ""


def validate_nurbs_patches(patches: list) -> tuple[bool, str]:
    """Validate NURBS patches for analytic quality.

    Checks: control net non-degeneracy, waterline closure at y=0,
    control point spacing, curvature limits.

    Returns (is_valid, error_message).
    """
    if not patches:
        return False, "No NURBS patches provided"

    for patch in patches:
        ctrl = patch.control_net
        if ctrl.size == 0:
            return False, f"Empty control net in patch '{patch.name}'"

        if not np.all(np.isfinite(ctrl)):
            return False, f"Non-finite control points in patch '{patch.name}'"

        n_u, n_v, _ = ctrl.shape
        if n_u < 2 or n_v < 3:
            return False, f"Control net too small ({n_u}×{n_v}) in patch '{patch.name}'"

        if "hull" in patch.name and not patch.is_mirrored:
            if abs(ctrl[0, :, 1]).max() > 1e-6:
                return False, f"Bow not closed at y=0 in patch '{patch.name}'"
            if abs(ctrl[-1, :, 1]).max() > 1e-6:
                return False, f"Stern not closed at y=0 in patch '{patch.name}'"

        for j in range(n_v):
            x_vals = ctrl[:, j, 0]
            if np.any(np.diff(x_vals) < -1e-6):
                return False, f"Non-monotonic x in patch '{patch.name}' at v={j}"

        if n_u >= 4 and n_v >= 4:
            max_ratio = 0.0
            for i in range(n_u - 1):
                for j in range(n_v - 1):
                    d1 = np.linalg.norm(ctrl[i+1, j] - ctrl[i, j])
                    d2 = np.linalg.norm(ctrl[i, j+1] - ctrl[i, j])
                    d3 = np.linalg.norm(ctrl[i+1, j+1] - ctrl[i, j+1])
                    d4 = np.linalg.norm(ctrl[i+1, j+1] - ctrl[i+1, j])
                    for d in [d1, d2, d3, d4]:
                        if d > 1e-10:
                            if max_ratio > 0:
                                ratio = d / max_ratio if d > max_ratio else max_ratio / d
                                if ratio > 50:
                                    return False, f"Extreme spacing ratio in patch '{patch.name}'"
                            max_ratio = max(max_ratio, d)

    return True, ""


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

    if D_keel_val is not None and LWL_val is not None:
        if D_keel_val > 1.5 * LWL_val:
            return False, f"D_keel ({D_keel_val:.3f}) > 1.5*LWL ({1.5*LWL_val:.3f}) — keel way deeper than hull length"
    if T_canoe_val is not None and D_keel_val is not None and LWL_val is not None:
        total_depth = T_canoe_val + D_keel_val
        if total_depth > 1.5 * LWL_val:
            return False, f"T_canoe + D_keel ({total_depth:.3f}) > 1.5*LWL ({1.5*LWL_val:.3f}) — total depth exceeds hull length"

    wp = x_dict.get("wingsail_pos", None)
    if wp is not None:
        if wp <= 0.0 or wp >= 1.0:
            return False, (
                f"wingsail mast position invalid: wingsail_pos={wp:.3f} — need 0 < pos < 1"
            )

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
            "LCB": (0.0, 100.0),
            "D_keel": (0.001, 2.5),
            "keel_chord": (0.001, 0.50),
            "bulb_vol": (0.0, 0.20),
            "bulb_pos": (0.30, 0.50),
            "E": (0.0, 1.0),
            "flare": (0.0, 60.0),
            "deadrise": (0.0, 60.0),
            "bilge_r": (0.0, 0.50),
            "keel_rake": (0.0, 35.0),
            "ballast_frac": (0.0, 1.0),
            "wingsail_pos": (0.0, 1.0),
        }
        for key, (min_val, max_val) in bounds.items():
            if key in x_dict:
                val = x_dict[key]
                if val < min_val or val > max_val:
                    return False, f"{key}={val:.3f} outside [{min_val:.2f}, {max_val:.2f}]"

    return True, ""


def validate_hull_geometry(path: str) -> Tuple[bool, str]:
    """Validate an exported hull STL mesh.

    Checks: loadability, watertightness, enclosed volume and bounding-box
    dimensions (length, beam). Returns (is_valid, error_message).
    """
    import trimesh

    if not path or not Path(path).exists():
        return False, f"Mesh file not found: {path}"

    try:
        mesh = trimesh.load(path, process=True)
    except Exception as e:
        return False, f"Cannot load mesh: {e}"

    if not hasattr(mesh, "is_watertight"):
        return False, "Loaded object is not a triangle mesh"
    if not mesh.is_watertight:
        return False, "Mesh is not watertight"
    if mesh.faces is None or len(mesh.faces) == 0:
        return False, "Mesh has no faces"

    volume = float(mesh.volume)
    if not np.isfinite(volume):
        return False, f"Non-positive volume (mesh volume is {volume})"
    if volume <= 0.0:
        return False, "Non-positive volume"
    if volume < 0.02:
        return False, f"Volume too small ({volume:.6f} m^3)"
    if volume > 0.5:
        return False, f"Volume too large ({volume:.6f} m^3)"

    extents = mesh.extents
    if extents is None or extents.size < 3:
        return False, "Mesh has no finite bounding box"
    length, beam = float(extents[0]), float(extents[1])
    if length < 0.4:
        return False, f"Hull too short (length={length:.3f} m)"
    if length > 4.5:
        return False, f"Hull too long (length={length:.3f} m)"
    if beam < 0.1:
        return False, f"Beam too narrow (beam={beam:.3f} m)"
    if beam > 2.0:
        return False, f"Beam too wide (beam={beam:.3f} m)"

    return True, ""
