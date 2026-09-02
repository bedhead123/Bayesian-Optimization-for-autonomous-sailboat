"""
NURBS-based hull, keel, and bulbous bow geometry generation.
19-parameter design vector → watertight STL mesh via B-spline surface,
NACA foil keel, and ellipsoid bulb.
Key exports: generate_hull(), compute_half_breadth_analytic(), design_vector_to_dict()
"""
import pickle
import numpy as np
import trimesh
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings
import logging

logger = logging.getLogger(__name__)


def _trapz(y, x):
    return np.trapezoid(y, x)


from hull_opt.config import design_vector_names
from hull_opt.geometry_validator import validate_design_vector
from hull_opt.param_layer import design_vector_to_physical
# sail_pos_x imported lazily in generate_hull to avoid circular import risk


@dataclass
class NURBSPatch:
    """NURBS surface patch with control net, knot vectors, and degree."""
    control_net: np.ndarray
    knots_u: np.ndarray
    knots_v: np.ndarray
    degree_u: int = 3
    degree_v: int = 3
    is_mirrored: bool = False
    name: str = ""


# _collapse_and_repair deleted (NURBS-only pipeline)
# _fix_sliver_faces deleted
# _weld_sliver_vertices deleted
# _check_volume_area_ratio deleted
# _validate_hull_mesh deleted


# _check_sac_scaling_station_variation deleted (NURBS-only pipeline)

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


# _check_mesh_spikes deleted (NURBS-only pipeline)
# _check_mesh_convexity deleted
# _check_mesh_self_intersection deleted
# _check_local_normals deleted
# _check_half_breadth_gradient deleted
# _check_keel_hull_intersection deleted


def design_vector_to_dict(x: np.ndarray) -> dict:
    names = design_vector_names()
    return {n: float(x[i]) for i, n in enumerate(names)}


def mesh_displacement(mesh, z: float = 0.0) -> float:
    """Displaced volume (m³) of the hull below the waterplane z.
    Slices the mesh at plane z with trimesh (keep-below, cap=False) and
    returns the volume of the part below z; falls back to the total mesh
    volume if slicing fails. Returns 0.0 for empty/degenerate input."""
    try:
        if isinstance(mesh, (str, Path)):
            mesh = trimesh.load(str(mesh))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if mesh is None:
            return 0.0
        vertices = getattr(mesh, "vertices", None)
        if vertices is None or len(vertices) < 4:
            return 0.0
        sliced = trimesh.intersections.slice_mesh_plane(
            mesh, np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, float(z)]),
            cap=False,
        )
        if sliced is None or len(getattr(sliced, "vertices", [])) < 4:
            raise ValueError("empty slice")
        vol = float(sliced.volume)
        if not np.isfinite(vol) or vol < 0.0:
            raise ValueError("non-finite slice volume")
        return vol
    except Exception:
        if z == 0.0:
            try:
                vol = float(mesh.volume)
            except Exception:
                return 0.0
            if np.isfinite(vol) and vol > 0.0:
                return vol
            return 0.0
        return 0.0


# ──────────────────────────────────────────────
# NURBS surface evaluation (B-spline, weights=1)
# ──────────────────────────────────────────────

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


def _find_span(n: int, p: int, u: float, knots: np.ndarray) -> int:
    """Find knot span index for parameter u using binary search.
    n = number of control points - 1 (= len(knots) - p - 2); p = degree.
    Returns integer span such that u ∈ [knots[span], knots[span+1]).
    Implements Algorithm A2.1 from Piegl & Tiller."""
    if u >= knots[n + 1]:
        return n
    if u <= knots[p]:
        return p
    lo, hi = p, n + 1
    mid = (lo + hi) // 2
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


def _basis_funs(span: int, u: float, p: int, knots: np.ndarray) -> np.ndarray:
    """Compute all non-zero B-spline basis functions (degree p) at parameter u
    for knot span index `span`. Returns array N[0..p] where
    N[k] = N_{span-p+k, p}(u).  Uses standard Cox-de Boor (Algorithm A2.2,
    Piegl & Tiller). O(p²)."""
    N = np.zeros(p + 1)
    N[0] = 1.0
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    for j in range(1, p + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = N[r] / denom if denom > 1e-15 else 0.0
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N


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
    # Precompute basis-function tables using fast span-based algorithm
    Nu = np.zeros((n_u, mu))
    for k, u in enumerate(u_vals):
        span = _find_span(n_u - 1, p, u, knots_u)
        Nvals = _basis_funs(span, u, p, knots_u)
        for i in range(p + 1):
            Nu[span - p + i, k] = Nvals[i]
    Nv = np.zeros((n_v, mv))
    for k, v in enumerate(v_vals):
        span = _find_span(n_v - 1, q, v, knots_v)
        Nvals = _basis_funs(span, v, q, knots_v)
        for j in range(q + 1):
            Nv[span - q + j, k] = Nvals[j]
    # Evaluate tensor product
    out = np.zeros((mu, mv, 3))
    for i in range(n_u):
        Ni = Nu[i]
        if np.all(Ni == 0):
            continue
        for j in range(n_v):
            Nij = Ni[:, None] * Nv[j][None, :]
            out += Nij[:, :, None] * ctrl[i, j][None, None, :]
    return out


def _nurbs_surface_point(ctrl: np.ndarray, knots_u: np.ndarray,
                          knots_v: np.ndarray, deg_u: int, deg_v: int,
                          u: float, v: float) -> np.ndarray:
    """Evaluate NURBS at a single (u,v) pair using O(p²) algorithm.
    Returns (x,y,z)."""
    n_u = ctrl.shape[0]
    span_u = _find_span(n_u - 1, deg_u, u, knots_u)
    Nu = _basis_funs(span_u, u, deg_u, knots_u)
    n_v = ctrl.shape[1]
    span_v = _find_span(n_v - 1, deg_v, v, knots_v)
    Nv = _basis_funs(span_v, v, deg_v, knots_v)
    pt = np.zeros(3)
    for i in range(deg_u + 1):
        i_ctrl = span_u - deg_u + i
        Ni = Nu[i]
        if Ni == 0:
            continue
        for j in range(deg_v + 1):
            j_ctrl = span_v - deg_v + j
            Nvj = Nv[j]
            if Nvj == 0:
                continue
            pt += Ni * Nvj * ctrl[i_ctrl, j_ctrl]
    return pt


def _nurbs_partial_deriv(ctrl: np.ndarray, knots_u: np.ndarray,
                          knots_v: np.ndarray, deg_u: int, deg_v: int,
                          u: float, v: float, eps: float = 1e-6) -> tuple:
    """Numerical partial derivatives dS/du, dS/dv at (u,v)."""
    u_clamped = np.clip(u, eps, 1.0 - eps)
    v_clamped = np.clip(v, eps, 1.0 - eps)
    Su = (_nurbs_surface_point(ctrl, knots_u, knots_v, deg_u, deg_v,
                                u_clamped + eps, v_clamped)
          - _nurbs_surface_point(ctrl, knots_u, knots_v, deg_u, deg_v,
                                  u_clamped - eps, v_clamped)) / (2.0 * eps)
    Sv = (_nurbs_surface_point(ctrl, knots_u, knots_v, deg_u, deg_v,
                                u_clamped, v_clamped + eps)
          - _nurbs_surface_point(ctrl, knots_u, knots_v, deg_u, deg_v,
                                  u_clamped, v_clamped - eps)) / (2.0 * eps)
    return Su, Sv


def _nurbs_jacobian_z(ctrl, knots_u, knots_v, deg_u, deg_v,
                       u: float, v: float) -> float:
    """z-component of ∂S/∂u × ∂S/∂v (scalar)."""
    Su, Sv = _nurbs_partial_deriv(ctrl, knots_u, knots_v, deg_u, deg_v, u, v)
    cross = np.cross(Su, Sv)
    return float(cross[2])


def _mirror_patch(patch: NURBSPatch) -> NURBSPatch:
    """Mirror a hull patch across y=0 for the starboard side."""
    ctrl = patch.control_net.copy()
    ctrl[:, :, 1] *= -1.0
    return NURBSPatch(
        control_net=ctrl,
        knots_u=patch.knots_u.copy(),
        knots_v=patch.knots_v.copy(),
        degree_u=patch.degree_u,
        degree_v=patch.degree_v,
        is_mirrored=True,
        name=patch.name + "_mirror",
    )


def _build_hull_patch(x_dict: dict) -> NURBSPatch:
    """Build port half-hull NURBS patch from the control net."""
    ctrl = _build_nurbs_control_net(x_dict)
    knots_u = _open_uniform_knots(ctrl.shape[0], 3)
    knots_v = _open_uniform_knots(ctrl.shape[1], 3)
    return NURBSPatch(
        control_net=ctrl,
        knots_u=knots_u,
        knots_v=knots_v,
        degree_u=3,
        degree_v=3,
        is_mirrored=False,
        name="hull_port",
    )


def _build_deck_patch(port_patch: NURBSPatch, x_dict: dict) -> NURBSPatch:
    """Build a deck NURBS patch closing the hull from port sheer to starboard sheer.

    Control net: (13, 3, 3) where v=0 is the port sheer line, v=1 is the
    centerline (y=0, z = z_sheer — 100% flat, no camber), and v=2 is the
    starboard sheer line (mirror of port).  Degree_u=3 (cubic along hull),
    degree_v=2 (quadratic across deck — reproduces the exact plane spanned
    by three z-equal rows).

    Watertightness at the bow/stern columns (where all 3 rows coincide once
    the deck is flat) is handled by collapse-aware emission in
    _tessellate_patches, not by geometry hacks.
    """
    port_ctrl = port_patch.control_net  # (13, 9, 3)
    sheer_ctrl = port_ctrl[:, -1, :]    # (13, 3) — sheer line
    n_u = port_ctrl.shape[0]

    stbd_sheer = sheer_ctrl.copy()
    stbd_sheer[:, 1] *= -1.0

    camber = sheer_ctrl.copy()
    camber[:, 1] = 0.0
    camber[:, 2] = sheer_ctrl[:, 2]  # flat: center row exactly at sheer height

    ctrl = np.zeros((n_u, 3, 3))
    ctrl[:, 1, :] = camber
    ctrl[:, 0, :] = sheer_ctrl   # port sheer
    ctrl[:, 2, :] = stbd_sheer   # starboard sheer

    knots_u = _open_uniform_knots(n_u, 3)
    knots_v = _open_uniform_knots(3, 2)

    return NURBSPatch(
        control_net=ctrl,
        knots_u=knots_u,
        knots_v=knots_v,
        degree_u=3,
        degree_v=2,
        is_mirrored=False,
        name="deck",
    )


def _build_keel_patch(x_dict: dict) -> Optional[NURBSPatch]:
    """Build keel fin as a ruled NURBS surface (4×8 control net).

    Returns None if the keel is effectively disabled.
    """
    D_keel = x_dict.get("D_keel", 0.0)
    keel_chord = x_dict.get("keel_chord", 0.0)
    if D_keel <= 0.01 or keel_chord <= 0.01:
        return None

    LWL = x_dict["LWL"]
    BWL = x_dict["BWL"]
    T_canoe = x_dict["T_canoe"]
    keel_rake = x_dict.get("keel_rake", 0.0)
    keel_tc = 0.12
    keel_x_pos = x_dict.get("bulb_pos", 0.4) * LWL
    keel_x_norm = keel_x_pos / LWL
    hull_draft_at_keel = T_canoe * max(0.05, 1.0 - 0.3 * (1.0 - keel_x_norm))
    embed = _keel_embed(x_dict, T_canoe, BWL, keel_chord)
    root_z = -hull_draft_at_keel + embed
    tip_z = root_z - D_keel
    sweep_rad = np.deg2rad(keel_rake)

    n_u = 4
    n_v = 8
    ctrl = np.zeros((n_u, n_v, 3))
    v_positions = np.linspace(0.0, 1.0, n_v)
    thickness = BWL * 0.06
    for j, vf in enumerate(v_positions):
        z_pos = root_z + vf * (tip_z - root_z)
        frac = vf
        local_chord = keel_chord * (1.0 - 0.5 * frac)
        taper = 1.0 - 0.5 * frac
        half_thick = 0.5 * thickness * taper
        sweep_shift = (z_pos - root_z) * np.tan(sweep_rad)
        x_start = -local_chord / 2.0 + sweep_shift + keel_x_pos
        # 4 chordwise control points: LE, max-thickness region, mid, TE
        xi = np.array([0.0, 0.3, 0.6, 1.0])
        naca_max = max(1e-10, _naca_thickness(0.3, keel_tc))
        scale = half_thick / naca_max
        for i, xf in enumerate(xi):
            ctrl[i, j, 0] = x_start + xf * local_chord
            ctrl[i, j, 1] = _naca_thickness(xf, keel_tc) * scale
            ctrl[i, j, 2] = z_pos

    knots_u = _open_uniform_knots(n_u, 3)
    knots_v = _open_uniform_knots(n_v, 3)
    return NURBSPatch(
        control_net=ctrl,
        knots_u=knots_u,
        knots_v=knots_v,
        degree_u=3,
        degree_v=3,
        is_mirrored=False,
        name="keel_port",
    )


def _build_bulb_patch(x_dict: dict) -> Optional[NURBSPatch]:
    """Build bulb as an ellipsoidal NURBS patch (6×6 control net).

    Returns None if the bulb is effectively disabled.
    """
    bulb_vol = x_dict.get("bulb_vol", 0.0)
    if bulb_vol < 1e-6:
        return None
    LWL = x_dict["LWL"]
    BWL = x_dict["BWL"]
    T_canoe = x_dict["T_canoe"]
    D_keel = x_dict["D_keel"]
    keel_chord = x_dict.get("keel_chord", 0.2)
    bulb_pos = x_dict.get("bulb_pos", 0.4)
    keel_rake = x_dict.get("keel_rake", 0.0)
    keel_x_pos = bulb_pos * LWL
    keel_x_norm = keel_x_pos / LWL
    hull_draft_at_keel = T_canoe * max(0.05, 1.0 - 0.3 * (1.0 - keel_x_norm))
    bulb_AR = x_dict.get("bulb_AR", 4.0)
    x_scale = max(bulb_AR / 3, 0.5) if bulb_AR > 0 else 1.5
    z_scale = 0.8
    vol_compensated = bulb_vol / (x_scale * z_scale)
    r = (3 * vol_compensated / (4 * np.pi)) ** (1.0 / 3.0)
    sweep_rad = np.deg2rad(keel_rake)
    x_c = -D_keel * np.tan(sweep_rad) + keel_x_pos
    z_c = -hull_draft_at_keel - D_keel

    n_u = 6
    n_v = 6
    ctrl = np.zeros((n_u, n_v, 3))
    # Spherical control net (quadrant), then scale
    u_pos = np.linspace(0.0, 1.0, n_u)
    v_pos = np.linspace(0.0, 1.0, n_v)
    for i, uf in enumerate(u_pos):
        theta = np.pi * uf * 0.5
        for j, vf in enumerate(v_pos):
            phi = np.pi * vf * 0.5
            xb = r * np.sin(theta) * np.cos(phi)
            yb = r * np.sin(theta) * np.sin(phi)
            zb = r * np.cos(theta)
            ctrl[i, j, 0] = x_c + xb * x_scale
            ctrl[i, j, 1] = yb
            ctrl[i, j, 2] = z_c + zb * z_scale

    knots_u = _open_uniform_knots(n_u, 3)
    knots_v = _open_uniform_knots(n_v, 3)
    return NURBSPatch(
        control_net=ctrl,
        knots_u=knots_u,
        knots_v=knots_v,
        degree_u=3,
        degree_v=3,
        is_mirrored=False,
        name="bulb",
    )


def _build_nurbs_patches(x_dict: dict) -> list[NURBSPatch]:
    """Build all NURBS patches: port hull, starboard hull (mirrored), deck,
    keel fin (if active), bulb (if active).

    Returns a list of NURBSPatch objects.
    """
    patches = []
    port_hull = _build_hull_patch(x_dict)
    starboard_hull = _mirror_patch(port_hull)
    patches.append(port_hull)
    patches.append(starboard_hull)

    deck = _build_deck_patch(port_hull, x_dict)
    patches.append(deck)

    keel = _build_keel_patch(x_dict)
    if keel is not None:
        keel_mirror = _mirror_patch(keel)
        patches.append(keel)
        patches.append(keel_mirror)
    bulb = _build_bulb_patch(x_dict)
    if bulb is not None:
        patches.append(bulb)
    return patches


def _log_boundary_edges(mesh, logger, n_quantiles: int = 4) -> None:
    """Diagnose a mesh that failed watertight repair: boundary-edge count and
    vertical (z) distribution, so the offending patch region is identifiable.

    Boundary edges = sorted edges appearing exactly once (trimesh 5.x has no
    `mesh.boundaries` attribute; the old call silently raised and was swallowed)."""
    try:
        edges_sorted = np.sort(mesh.edges, axis=1)
        uniq_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
        boundary = uniq_edges[counts == 1]
        if len(boundary) == 0:
            logger.warning("  boundary edges: none detected (mesh.is_watertight=False)")
            return
        z_centers = np.mean(mesh.vertices[boundary], axis=1)[:, 2]
        z_pcts = np.percentile(z_centers, np.linspace(0.0, 100.0, n_quantiles + 2)[1:-1])
        logger.warning(
            "  boundary edges: %d open edges; z-centile clusters (m): %s",
            len(boundary),
            ", ".join(f"{z:.4f}" for z in z_pcts),
        )
    except Exception:
        pass


def _tessellate_patches(patches: list[NURBSPatch],
                         dp: float = 0.01,
                         require_watertight: bool = True) -> trimesh.Trimesh:
    """Evaluate NURBS patches on a uniform grid and connect into triangles.

    Uses uniform evaluation resolution for hull patches so that shared
    edges produce matching vertices.  The deck patch uses the same
    u-resolution so its sheer edges match the hull patches exactly.
    After all patches are tessellated, ``merge_vertices`` welds the
    coincident boundary vertices into a watertight mesh.

    Port-side patches (hull, keel) are evaluated once and mirrored
    to produce the starboard side, avoiding non-manifold centerline
    edges that would arise from two coincident patch boundaries.

    Returns a single Trimesh object with all patches merged.
    """
    all_verts = []
    all_faces = []
    vert_offset = 0

    hull_n_ue = None
    hull_n_ve = None
    for patch in patches:
        if "hull" not in patch.name or patch.is_mirrored:
            continue
        ctrl = patch.control_net
        n_u = ctrl.shape[0]
        n_v = ctrl.shape[1]
        bb_min = ctrl.min(axis=(0, 1))
        bb_max = ctrl.max(axis=(0, 1))
        extent = np.max(bb_max - bb_min)
        n_ue = max(4, int(np.ceil(extent / dp)))
        n_ve = max(4, int(np.ceil(extent / dp)))
        n_ue = max(n_ue, n_u * 2)
        n_ve = max(n_ve, n_v * 2)
        if hull_n_ue is None or n_ue > hull_n_ue:
            hull_n_ue = n_ue
        if hull_n_ve is None or n_ve > hull_n_ve:
            hull_n_ve = n_ve

    for patch in patches:
        if patch.is_mirrored:
            continue

        ctrl = patch.control_net
        n_u = ctrl.shape[0]
        n_v = ctrl.shape[1]
        is_hull = "hull" in patch.name
        is_deck = patch.name == "deck"

        if is_hull and hull_n_ue is not None:
            n_ue = hull_n_ue
            n_ve = hull_n_ve
        elif is_deck and hull_n_ue is not None:
            n_ue = hull_n_ue
            n_ve = max(3, min(7, hull_n_ve // 2))
        else:
            bb_min = ctrl.min(axis=(0, 1))
            bb_max = ctrl.max(axis=(0, 1))
            extent = np.max(bb_max - bb_min)
            n_ue = max(4, int(np.ceil(extent / dp)))
            n_ve = max(4, int(np.ceil(extent / dp)))
            n_ue = max(n_ue, n_u * 2)
            n_ve = max(n_ve, n_v * 2)

        u_vals = np.linspace(0.0, 1.0, n_ue)
        v_vals = np.linspace(0.0, 1.0, n_ve)

        surf = _eval_nurbs_surface(
            ctrl, patch.knots_u, patch.knots_v,
            u_vals, v_vals, patch.degree_u, patch.degree_v,
        )
        verts = surf.reshape(-1, 3)

        if is_deck:
            # Deck spans port→starboard; no mirror needed.  With a 100% flat
            # deck the control columns at bow/stern collapse to a single point
            # (y=0 forced at the ends, all rows z-equal): quads adjacent to a
            # collapsed column would emit one zero-area triangle each, which
            # after vertex welding become repeated-index faces that poison the
            # non-manifold cleanup (it kills the healthy fan triangles).  Emit
            # a single fan triangle instead — topologically identical closure.
            grid = verts.reshape(n_ue, n_ve, 3)
            collapsed = np.all(np.ptp(grid, axis=1) < 1e-12, axis=1)  # (n_ue,)
            faces = []
            for i in range(n_ue - 1):
                for j in range(n_ve - 1):
                    v0 = i * n_ve + j
                    v1 = i * n_ve + j + 1
                    v2 = (i + 1) * n_ve + j + 1
                    v3 = (i + 1) * n_ve + j
                    if collapsed[i]:
                        faces.append([v0 + vert_offset, v3 + vert_offset, v2 + vert_offset])
                    elif collapsed[i + 1]:
                        faces.append([v0 + vert_offset, v2 + vert_offset, v1 + vert_offset])
                    else:
                        faces.append([v0 + vert_offset, v2 + vert_offset, v1 + vert_offset])
                        faces.append([v0 + vert_offset, v3 + vert_offset, v2 + vert_offset])
            all_verts.append(verts)
            all_faces.extend(faces)
            vert_offset += len(verts)
        else:
            # Port-side patch: generate port faces, mirror vertices for starboard
            n_verts = len(verts)
            stbd_verts = verts.copy()
            stbd_verts[:, 1] *= -1.0
            for i in range(n_ue - 1):
                for j in range(n_ve - 1):
                    v0 = i * n_ve + j
                    v1 = i * n_ve + j + 1
                    v2 = (i + 1) * n_ve + j + 1
                    v3 = (i + 1) * n_ve + j
                    # Port side
                    all_faces.append([v0 + vert_offset, v2 + vert_offset, v1 + vert_offset])
                    all_faces.append([v0 + vert_offset, v3 + vert_offset, v2 + vert_offset])
                    # Starboard side (mirrored).  Use the OTHER diagonal
                    # for quads where port vertices are on the centerline
                    # (y≈0), so that port-starboard diagonal edges don't
                    # coincide and create 4-face non-manifold edges.
                    s0 = v0 + vert_offset + n_verts
                    s1 = v1 + vert_offset + n_verts
                    s2 = v2 + vert_offset + n_verts
                    s3 = v3 + vert_offset + n_verts
                    if (abs(verts[v0, 1]) < 1e-9 or abs(verts[v2, 1]) < 1e-9):
                        all_faces.append([s0, s1, s3])
                        all_faces.append([s1, s2, s3])
                    else:
                        all_faces.append([s0, s1, s2])
                        all_faces.append([s0, s2, s3])
            all_verts.append(verts)
            all_verts.append(stbd_verts)
            vert_offset += 2 * n_verts

    if len(all_verts) == 0:
        return trimesh.Trimesh()

    combined_verts = np.vstack(all_verts)
    combined_faces = np.array(all_faces, dtype=np.int64)

    mesh = trimesh.Trimesh(
        vertices=combined_verts, faces=combined_faces,
        process=False
    )
    mesh.merge_vertices(merge_tex=False, merge_norm=False, digits_vertex=12)

    # Drop repeated-index (zero-area) faces BEFORE dedup and the non-manifold
    # pass: a degenerate face contributes its shared edge twice, and the
    # >2-faces cleanup would then kill healthy neighbor triangles instead.
    if len(mesh.faces):
        f = np.asarray(mesh.faces)
        degen = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
        if degen.any():
            mesh.update_faces(np.where(~degen)[0])
            mesh.remove_unreferenced_vertices()

    # Remove duplicate faces (same sorted vertices).  The starboard
    # mirror creates exact face copies at the centerline (bow/stern/keel).
    sorted_faces = np.sort(mesh.faces, axis=1)
    _, uniq_idx = np.unique(sorted_faces, axis=0, return_index=True)
    if len(uniq_idx) < len(mesh.faces):
        mesh.update_faces(uniq_idx)
        mesh.remove_unreferenced_vertices()

    # Resolve non-manifold edges (>2 faces) by keeping only 2 faces
    # per edge.  Iterate since removing one face can change counts
    # for other edges.
    changed = True
    while changed:
        changed = False
        edge_to_faces = {}
        for fi in range(len(mesh.faces)):
            f = mesh.faces[fi]
            for pi in range(3):
                e = tuple(sorted([int(f[pi]), int(f[(pi + 1) % 3])]))
                edge_to_faces.setdefault(e, []).append(fi)
        bad = {e: fs for e, fs in edge_to_faces.items() if len(fs) > 2}
        if not bad:
            break
        kill = set()
        for e, fs in bad.items():
            for fi in fs[2:]:
                kill.add(fi)
        if kill:
            keep = [fi for fi in range(len(mesh.faces)) if fi not in kill]
            mesh.update_faces(keep)
            mesh.remove_unreferenced_vertices()
            changed = True

    if not mesh.is_watertight and require_watertight:
        try:
            mesh.fill_holes()
        except Exception:
            pass
        if not mesh.is_watertight:
            logger.warning("Mesh still not watertight after fill_holes")
            _log_boundary_edges(mesh, logger)
    elif not mesh.is_watertight:
        logger.debug("Combined mesh not watertight (open keel/bulb patches); skipping fill_holes")

    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _compute_nurbs_station_areas(patches: list[NURBSPatch],
                                  LWL: float,
                                  n_stations: int = 41) -> tuple:
    """Compute station areas from NURBS patches at specified x stations.

    For each hull/keel patch, evaluates the NURBS on a (u,v) grid where
    u=const slices are cross-sections (bow→stern) and v runs keel→sheer.
    The y(z) curve at each u station is extracted from the full v-direction
    evaluation, then integrated as 2*|y|*dz.

    Skips the deck patch (above waterline, contributes 0 to submerged areas)
    and bulb patch (not a hull/keel section).

    Returns (x_stations, area_stations, uw_area_stations) where:
      x_stations: array of x positions (m)
      area_stations: total cross-sectional area at each station
      uw_area_stations: underwater (z ≤ 0) cross-sectional area
    """
    x_stations = np.linspace(0.0, LWL, n_stations)
    area_stations = np.zeros(n_stations)
    uw_area_stations = np.zeros(n_stations)
    n_ve = 41

    for patch in patches:
        if "hull" not in patch.name and "keel" not in patch.name:
            continue
        ctrl = patch.control_net

        u_vals = np.linspace(0.0, 1.0, n_stations)
        v_vals = np.linspace(0.0, 1.0, n_ve)
        surf = _eval_nurbs_surface(
            ctrl, patch.knots_u, patch.knots_v,
            u_vals, v_vals, patch.degree_u, patch.degree_v,
        )
        # For mirrored (starboard) patches, flip y sign
        if patch.is_mirrored:
            surf[:, :, 1] *= -1.0

        for si in range(n_stations):
            # Extract the cross-section curve at this u station (all v)
            zs = surf[si, :, 2]
            ys = surf[si, :, 1]
            order = np.argsort(zs)
            zs = zs[order]
            ys = ys[order]

            # Integrate 2*|y|*dz for the full cross-section
            dz = np.diff(zs)
            y_mid = 0.5 * (np.abs(ys[:-1]) + np.abs(ys[1:]))
            area_stations[si] += float(np.sum(y_mid * np.abs(dz)))

            # Underwater portion (z ≤ 0)
            uw_mask = (zs[:-1] + zs[1:]) / 2 <= 0
            if np.any(uw_mask):
                uw_area_stations[si] += float(np.sum(y_mid[uw_mask] * np.abs(dz[uw_mask])))

    return x_stations, area_stations, uw_area_stations


def _save_nurbs_patches(patches: list[NURBSPatch], path: str) -> None:
    """Save NURBS patches to a pickle sidecar file."""
    data = []
    for p in patches:
        data.append({
            "control_net": p.control_net,
            "knots_u": p.knots_u,
            "knots_v": p.knots_v,
            "degree_u": p.degree_u,
            "degree_v": p.degree_v,
            "is_mirrored": p.is_mirrored,
            "name": p.name,
        })
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _load_nurbs_patches(path: str) -> list[NURBSPatch]:
    """Load NURBS patches from a pickle sidecar file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    patches = []
    for d in data:
        patches.append(NURBSPatch(
            control_net=d["control_net"],
            knots_u=d["knots_u"],
            knots_v=d["knots_v"],
            degree_u=d["degree_u"],
            degree_v=d["degree_v"],
            is_mirrored=d["is_mirrored"],
            name=d["name"],
        ))
    return patches


def _waterline_half_breadth(x_norm: np.ndarray, BWL: float,
                            Cp: float, Cm: float,
                            LCB: float = 45.0) -> np.ndarray:
    taper = max(0.02, min(0.15, BWL * 0.1))
    n = 0.5 + 2.0 * (1.0 - Cp)
    cm_factor = Cm / 0.75
    n = np.clip(n / cm_factor, 0.3, 3.0)
    lcb_shift = np.clip((LCB - 45.0) / 100.0, -0.15, 0.15)
    peak_pos = 0.45 + lcb_shift
    result = np.zeros_like(x_norm)
    for i, x in enumerate(x_norm):
        if x < peak_pos:
            z = min(0.9999, (peak_pos - x) / max(peak_pos, 1e-10))
            result[i] = (BWL / 2) * (1 - z ** n) * (1.0 - z * taper)
        else:
            z = min(0.9999, (x - peak_pos) / max(1 - peak_pos, 1e-10))
            result[i] = (BWL / 2) * (1 - z ** n) * (1.0 - z * taper)
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
    br_term = mid_wt * bilge_r
    y_raw = base + dr_term + fl_term + br_term
    # Floor at 1e-7 (not 0.0): steep deadrise can drive y_raw negative over the
    # whole lower band of a section, and clipping to exactly 0.0 collapses that
    # band to bitwise-zero half-breadth in the NURBS control net, making port
    # and starboard surfaces coincide exactly (non-watertight tessellation).
    # Minimum keel half-width: prevent knife-edge V where the whole
    # lower section collapses to zero.  Keep at least 2% of the local
    # waterline beam (or 5mm absolute) so the bottom is a small flat,
    # not a triangle, without affecting the upper V shape.
    keel_min = max(0.005, 0.02 * float(y_wl))
    y = np.clip(y_raw, keel_min, None)
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


def _bow_deck_floor(x_norm: np.ndarray, BWL: float) -> np.ndarray:
    """Minimum deck half-breadth at the bow: the stem keeps a real rounded
    width (30% of half-beam) instead of collapsing to the waterline point,
    tapering into the flare line by 18% of LWL. Without it the bow topsides
    are a vertical knife blade (y_wl → 0 at the stem), so there is no
    inward slope from deck to waterline."""
    stem_w = 0.30 * (BWL / 2.0)
    return stem_w * np.clip(1.0 - x_norm / 0.18, 0.0, 1.0)


# Topside (above-waterline) flare tuning.  The wall between the waterline
# edge and the deck edge leans inboard at flare·TOPSIDE_BOW_AMP; the flare
# angle tapers toward both ends (bow 0.5×, stern 0.4×) so the deck edge
# curves gracefully into the bow and stern points — the same taper at both
# ends, not a blunt bow.  Capped at TOPSIDE_MAX_DEG as a backstop, with
# _bow_deck_floor guaranteeing a rounded stem instead of a knife blade.
# TOPSIDE_BOW_AMP must stay ≤ 2.0: _interp_param silently caps bow_factor.
TOPSIDE_BOW_AMP = 0.5
TOPSIDE_MAX_DEG = 45.0


def _topside_wall_angle(x_norm: float, flare_deg: float, BWL: float,
                        E: float, y_wl: float) -> float:
    """Effective topside wall angle (rad) above the waterline at station x_norm.

    The deck edge sits at max(waterline edge + E·tan(flare·bow_amp),
    bow deck floor); the wall runs straight from (z=0, y_wl) to
    (z=E, y_deck), so the effective angle folds the bow floor in.  Used by
    both the control net and the analytic half-breadth so the mesh and the
    Michell/BEM hull always agree.
    """
    fl_ts = min(
        _interp_param(x_norm, float(flare_deg), bow_factor=TOPSIDE_BOW_AMP,
                      stern_factor=0.4),
        TOPSIDE_MAX_DEG,
    )
    y_deck = max(y_wl + E * np.tan(np.deg2rad(fl_ts)),
                 _bow_deck_floor(x_norm, BWL))
    return np.arctan2(max(y_deck - y_wl, 0.0), E)


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


def _sheer_height(x_norm: float, E: float = 0.20) -> float:
    """100% flat main deck at freeboard E.

    No bow rise, no stern counter-sheer, no midship dip: the deck edge is
    exactly level along the whole hull.  (History: piecewise-linear
    bow-1.3E/midship-E/stern-E+0.8*SA*E profile made the deck a warped
    banana; a quintic bow rise of 0.15*SA followed; both removed — the deck
    is now planar and SA was dropped from the design vector entirely.)
    """
    return E


def _sac_form(x_norm: np.ndarray, Cp: float, LCB: float = 45.0) -> np.ndarray:
    n = Cp / max(1e-10, 1.0 - Cp)
    lcb_shift = (LCB - 45.0) / 120.0
    x_shifted = np.clip(x_norm - lcb_shift, -0.999, 0.999)
    return (1.0 - np.abs(x_shifted) ** n)


def _build_nurbs_control_net(x_dict: dict) -> np.ndarray:
    """Build a 13×9×3 B-spline control net for the port half-hull.

    u-direction (13 pts, 0=bow → 1=stern): uniform spacing
    v-direction (9 pts, 0=keel → 1=sheer): uniform spacing

    Returns: (13, 9, 3) array of (x, y, z) control points for y ≥ 0.
    """
    LWL = float(x_dict.get("LWL", 2.4))
    BWL = x_dict["BWL"]
    T_canoe = x_dict["T_canoe"]
    Cp = x_dict["Cp"]
    Cm = x_dict.get("Cm", 0.75)
    LCB_val = x_dict.get("LCB", 45.0)
    deadrise_val = x_dict["deadrise"]
    bilge_r = x_dict["bilge_r"]
    flare_param = x_dict["flare"]
    E_val = x_dict.get("E", 0.20)

    u_pos = np.linspace(0.0, 1.0, 13)
    v_lev = np.linspace(0.0, 1.0, 9)
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
        fl = _interp_param(xn, flare_param, bow_factor=1.0, stern_factor=0.4)
        fl_ts = _topside_wall_angle(xn, flare_param, BWL, E_val, y_wl[i])
        z_sheer = _sheer_height(xn, E_val)

        for j, vf in enumerate(v_lev):
            z_pos = -T_local * (1.0 - vf) + z_sheer * vf

            if xn <= 0.0 or xn >= 1.0 - 1e-12:
                y_val = 0.0
            elif vf <= 1e-12:
                y_val = 0.0
            elif z_pos > 0.0:
                y_val = y_wl[i] + z_pos * np.tan(fl_ts)
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

    # 5. Mast sanity: wingsail mast position within (0, 1) × LWL from bow
    wingsail_pos = x_dict.get("wingsail_pos", 0.42)
    if wingsail_pos <= 0 or wingsail_pos >= 1:
        raise ValueError(
            f"RealityCheck: wingsail mast position invalid "
            f"(wingsail_pos={wingsail_pos}) — need 0 < pos < 1"
        )


def generate_hull(design_vector: np.ndarray,
                  output_dir: Optional[str] = None,
                  LWL: float = 2.4,
                  target_displacement: Optional[float] = None,
                  config=None,
                  bem_max_faces: Optional[int] = None) -> tuple[str, str, dict, str]:
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
    LCB_val = x_dict.get("LCB", 45.0)
    E_val = x_dict.get("E", 0.20)
    deadrise_val = x_dict["deadrise"]
    bilge_r = x_dict["bilge_r"]
    flare_param = x_dict["flare"]
    keel_chord = x_dict["keel_chord"]
    bulb_vol = x_dict["bulb_vol"]
    bulb_pos = x_dict["bulb_pos"]
    keel_rake_val = x_dict["keel_rake"]
    ballast_frac = x_dict["ballast_frac"]
    bulb_density = x_dict.get("bulb_density", 11340)

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

    # Generation-time BEM decimation target (R2.2): config wins over the
    # default; an explicit argument wins over config.
    if bem_max_faces is None:
        bem_max_faces = int(getattr(config.wave_spectrum, "bem_n_panels", 2500)) if config is not None else 2500
    bem_max_faces = int(bem_max_faces)
    if bem_max_faces <= 0:
        bem_max_faces = 2500

    # Iterative SAC scaling: compute actual SAC from analytic half-breadths,
    # then scale uniformly to match target volume. The scale is capped so a
    # thin hull cannot be inflated into a balloon: volume scales linearly with
    # the breadth scale, so hitting an unreachable target (0.14 m³ on a
    # 0.046 m³ hull) used to require scale ~3.2, tripling the beam. The
    # adaptive target keeps scale in [0.5, 1.30].
    n_stations = 41
    u_vals = np.linspace(0.0, 1.0, n_stations)
    sac_avg_scale = 1.0
    sac_max_scale = 1.30
    target_eff = None
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
            z_sheer = _sheer_height(x_norm, E_val)
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
            sac_volume_uw = float(_trapz(uw_station_areas, u_vals * LWL))
            if not np.isfinite(sac_volume_uw) or sac_volume_uw <= 0:
                raise ValueError(f"Non-finite or non-positive SAC underwater volume: {sac_volume_uw}")
            if target_eff is None:
                target_eff = min(float(target_displacement), sac_max_scale * sac_volume_uw)
            vol_scale = target_eff / max(1e-10, sac_volume_uw)
            if not np.isfinite(vol_scale) or vol_scale <= 0:
                raise ValueError(f"Non-finite or non-positive volume scale: {vol_scale}")
            sac_avg_scale *= vol_scale
            sac_avg_scale = np.clip(sac_avg_scale, 0.5, sac_max_scale)
            if not np.isfinite(sac_avg_scale):
                raise ValueError(f"SAC scale became non-finite: {sac_avg_scale}")
            if abs(vol_scale - 1.0) < 0.01:
                break
        else:
            break
    if target_displacement is not None and target_displacement > 0:
        logger.info(f"SAC scaling: final scale={sac_avg_scale:.6f}, "
                    f"SAC volume={sac_volume_uw:.6f}, target={target_eff:.6f} "
                    f"(raw capacity={sac_volume_uw / max(sac_avg_scale, 1e-10):.6f}), "
                    f"n_iters={iteration+1}")
    else:
        logger.info(f"SAC scaling: final scale={sac_avg_scale:.6f} (no target displacement)")

    positive_areas = station_areas[station_areas > 1e-6]
    sac_scale_std = float(np.std(positive_areas) / max(np.mean(positive_areas), 1e-10)) if len(positive_areas) > 3 else 0.0

    use_nurbs = config is not None and config.fixed.use_nurbs_geometry

    if use_nurbs:
        # ── NURBS path: build patches (with deck closure), tessellate, export ──
        sac_scale = sac_avg_scale
        for _sac_iter in range(6):
            patches = _build_nurbs_patches(x_dict)
            # Volumetric SAC scaling: scale y AND z together (sqrt split)
            # so volume ~ y*z scales by sac_scale without distorting L/B.
            # Previously y-only scaling turned every under-volume hull into
            # a sideways-stretched pizza slice.
            yz_scale = float(np.sqrt(max(sac_scale, 1e-10)))
            for p in patches:
                if "hull" in p.name:
                    p.control_net[:, :, 1] *= yz_scale
                    p.control_net[:, :, 2] *= yz_scale
            if target_eff is not None and target_eff > 0:
                from hull_opt.hydrostatics import nurbs_submerged_volume
                nurbs_vol, _ = nurbs_submerged_volume(patches, n_sub=5)
                vol_ratio = target_eff / max(nurbs_vol, 1e-10)
                if abs(vol_ratio - 1.0) < 0.03:
                    break
                sac_scale = np.clip(sac_scale * vol_ratio, 0.5, 1.30)
            else:
                break

        # Rebuild deck patch from the (now scaled) port hull, since the
        # deck was built from the unscaled port hull in _build_nurbs_patches.
        port_hull = next(p for p in patches if p.name == "hull_port")
        new_deck = _build_deck_patch(port_hull, x_dict)
        patches = [p for p in patches if p.name != "deck"] + [new_deck]

        # Check control net curvature before tessellating
        for p in patches:
            if "hull" in p.name:
                is_valid, msg = _check_control_net_curvature(p.control_net)
                if not is_valid:
                    raise ValueError(f"NURBS validation: {p.name}: {msg}")

        # Build and tessellate hull-only mesh (hull + deck patches).  This is
        # used by the GZ curve and BEM, so require_watertight=True (repair via
        # fill_holes + boundary-edge diagnostics on failure).
        hull_patches = [p for p in patches if ("hull" in p.name or p.name == "deck") and not p.is_mirrored]
        hull_mesh = _tessellate_patches(hull_patches, dp=0.03, require_watertight=True)
        if hull_mesh is None or len(hull_mesh.vertices) < 4:
            raise ValueError("NURBS hull tessellation produced degenerate mesh")
        if not hull_mesh.is_watertight:
            raise ValueError(
                "NURBS hull tessellation not watertight after fill_holes repair "
                f"(boundary edges remain) — infeasible design (stored as E_GEOM)"
            )

        # Combined STL uses hull+deck only (the keel and bulb NURBS patches
        # are separate open surfaces that don't share boundaries with the
        # hull — they contribute to SAC/volume via _compute_nurbs_station_areas
        # but are not included in the export STL. The BEM and SPH resistance
        # see hull-only geometry; the FoM impact is < 0.1% relative.
        full_mesh = hull_mesh

        # Compute SAC from NURBS station areas
        x_sta, sta_areas, uw_sta_areas = _compute_nurbs_station_areas(
            patches, LWL, n_stations=41)
        station_areas = sta_areas
        uw_station_areas = uw_sta_areas

        # Save NURBS patches sidecar
        nurbs_pickle = str(out / "hull_patches.nurbs")
        _save_nurbs_patches(patches, nurbs_pickle)

        # Hull-only STL (for GZ curve)
        hull_stl = str(out / "hull_geometry.stl")
        hull_mesh.export(hull_stl)

        # Pre-decimated BEM copy (R2.2): low_fidelity reads this when present,
        # skipping per-eval decimation. Non-fatal on failure — the runtime
        # fallback decimates hull.stl at eval time. Import is lazy to avoid a
        # cycle: low_fidelity imports geometry at module level.
        try:
            from hull_opt.low_fidelity import BEM_MAX_PANELS, _decimate_bem_mesh
            import meshio as _meshio
            bem_msh = _meshio.Mesh(
                points=np.asarray(hull_mesh.vertices, dtype=np.float64),
                cells=[("triangle", np.asarray(hull_mesh.faces, dtype=np.int32))],
            )
            decimated = _decimate_bem_mesh(
                bem_msh, config, n_target=min(bem_max_faces, BEM_MAX_PANELS)
            )
            if decimated is not bem_msh:
                _tm = trimesh.Trimesh(
                    vertices=np.asarray(decimated.points, dtype=np.float64),
                    faces=np.asarray(decimated.cells_dict["triangle"], dtype=np.int32),
                    process=False,
                )
                _tm.export(str(out / "hull_geometry_bem.stl"))
        except Exception as e:
            _msg = str(e).replace("\n", " ").strip()
            logger.warning(f"BEM decimated export skipped: {_msg[:200]}")

        # Combined STL (for Capytaine / BEM)
        stl_path = str(out / "hull.stl")
        full_mesh.export(stl_path)

        # Hydrostatics from NURBS hull mesh
        if config is not None:
            from hull_opt.rig import build_rig, sail_pos_x
            rig = build_rig(x_dict, config)
            mast_cg_frac = config.fixed.mast_cg_height_frac
            mast_masses = (config.fixed.wingsail_mast_mass, config.fixed.wingsail_nose_pod_mass)
            mast_ce_z = rig["single"]["z"]
            mast_cg_z = (mast_cg_frac * mast_ce_z, 0.0)
            mast_cg_x = (sail_pos_x(x_dict["wingsail_pos"], LWL),
                         sail_pos_x(x_dict["wingsail_pos"], LWL))
        else:
            mast_masses = (0.0, 0.0)
            mast_cg_z = (0.0, 0.0)
            mast_cg_x = (0.0, 0.0)
        hydro = _compute_hydrostatics(
            hull_mesh, LWL, BWL, Cp, T_canoe, D_keel,
            bulb_vol, ballast_frac, target_displacement,
            sac_volume=float(_trapz(uw_station_areas, u_vals * LWL)),
            keel_chord=keel_chord,
            sac_scale_factor=sac_avg_scale,
            sac_scale_std=sac_scale_std,
            target_nabla_eff=target_eff,
            station_areas=uw_station_areas,
            x_dict=x_dict, config=config,
            bulb_pos=bulb_pos,
            bulb_density=bulb_density,
            mast_masses=mast_masses,
            mast_cg_z=mast_cg_z,
            mast_cg_x=mast_cg_x)
        hydro["nurbs_patches_file"] = nurbs_pickle

        # Validate NURBS patches
        from hull_opt.geometry_validator import validate_nurbs_patches, validate_nurbs_closed_surface
        is_valid, val_msg = validate_nurbs_patches(patches)
        if not is_valid:
            raise ValueError(f"NURBS patch validation: {val_msg}")
        is_valid, val_msg = validate_nurbs_closed_surface(patches, mesh_volume=hull_mesh.volume)
        if not is_valid:
            raise ValueError(f"NURBS closed surface validation: {val_msg}")

        # SAC csv
        sac_actual = [[u_vals[i] * LWL, station_areas[i]] for i in range(n_stations)]
        sac_path = str(out / "sac.csv")
        np.savetxt(sac_path, np.array(sac_actual), delimiter=",", header="x,area", comments="")

        # Combined STL with keel+bulb for the SPH cases (towing/storm/
        # focused-wave/drop/inverted).  DualSPHysics does not need a watertight
        # mesh, so the keel and bulb patches (open surfaces that don't share
        # boundaries with the hull) are tessellated together.  BEM and the GZ
        # curve keep the hull-only STL (stl_path / hull_geometry.stl) — the
        # keel's contribution there is < 0.1% relative (see the hull_mesh
        # comment above).  Falls back to the hull-only STL on failure so the
        # SPH writers can never receive a missing path.
        sph_stl_path = stl_path
        try:
            sph_mesh = _tessellate_patches(patches, dp=0.03, require_watertight=False)
            if sph_mesh is None or len(sph_mesh.vertices) < 500:
                raise ValueError("SPH combined tessellation degenerate")
            sph_stl_path = str(out / "hull_full.stl")
            sph_mesh.export(sph_stl_path)
        except Exception as _e:
            logger.warning("hull_full.stl export skipped (%s); SPH uses hull-only STL",
                           str(_e)[:160])
        hydro["full_mesh_stl"] = sph_stl_path

        return stl_path, sac_path, hydro, hull_stl

    # ── Analytic triangle-mesh path (deprecated — unreachable with NURBS default) ──
    # This path references functions that have been removed for the NURBS-only
    # pipeline.  It is kept only for reference and will raise NameError if
    # reached (use_nurbs=False is no longer supported).
    raise NotImplementedError(
        "Analytic triangle-mesh path is deprecated. "
        "Set use_nurbs_geometry=True in config."
    )


def _naca_thickness(x: float, t: float = 0.12) -> float:
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x
                    - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)


def _keel_embed(x_dict: dict, T_canoe: float, BWL: float,
                keel_chord: float) -> float:
    """Vertical overlap of the keel root above the hull bottom plane.

    On flat (low-deadrise) bottoms the keel root meets the bottom at a
    near-tangent angle: a draft-only embed (0.15*T_canoe) lets the root
    barely poke through, degenerating the boolean-union junction ring so
    the keel/bulb stays a separate closed shell (E_GEOM "keel/bulb not
    connected").  Scale the embed with the keel chord (the root ring is
    ~1 chord long) and the deadrise rise across the root half-width so
    the whole root sits inside the hull volume.
    """
    deadrise_rad = np.deg2rad(float(x_dict.get("deadrise", 10.0)))
    half_width = 0.5 * BWL * 0.06
    return max(0.15 * T_canoe,
               0.3 * keel_chord,
               half_width * np.tan(deadrise_rad) + 0.02)


# _union_keel_bulb deleted (NURBS-only pipeline)


# _make_bulb deleted (NURBS-only pipeline — bulb built as NURBS patch)


def keel_half_breadth(xq: np.ndarray, zq: np.ndarray,
                      x_dict: dict, LWL: float) -> np.ndarray:
    """Analytic half-breadth of the NACA keel foil, mirroring _make_keel.

    Root sits at z = -T_keel + _keel_embed (embed scales with keel chord
    and deadrise, see _keel_embed), depth D_keel, chord tapers to 50% at
    the tip, and the chord sweeps aft with depth by keel_rake. Returns 0
    outside the keel span and when the keel is disabled (D_keel <= 0.01
    or keel_chord <= 0.01 — matches the mesh gate in generate_hull).

    NOT composed into compute_half_breadth_analytic: the Michell call
    sites add it explicitly so SAC/hydrostatics never double-count the
    keel volume.
    """
    D_keel = x_dict.get("D_keel", 0.0)
    keel_chord = x_dict.get("keel_chord", 0.0)
    xq = np.asarray(xq, dtype=float)
    zq = np.asarray(zq, dtype=float)
    if D_keel <= 0.01 or keel_chord <= 0.01:
        return np.zeros_like(xq)

    BWL = x_dict["BWL"]
    T_canoe = x_dict["T_canoe"]
    keel_rake = x_dict.get("keel_rake", 0.0)
    keel_tc = 0.12
    keel_x_pos = x_dict.get("bulb_pos", 0.4) * LWL
    keel_x_norm = keel_x_pos / LWL
    hull_draft_at_keel = T_canoe * max(0.05, 1.0 - 0.3 * (1.0 - keel_x_norm))
    root_z = -hull_draft_at_keel + _keel_embed(x_dict, T_canoe, BWL, keel_chord)
    tip_z = root_z - D_keel

    out = np.zeros_like(xq)
    span_mask = (zq <= root_z) & (zq >= tip_z)
    if not np.any(span_mask):
        return out

    frac = (np.where(span_mask, zq, root_z) - root_z) / -D_keel
    taper = 1.0 - 0.5 * frac
    local_chord = keel_chord * taper
    naca_max = max(1e-10, _naca_thickness(0.3, keel_tc))
    scale = 0.5 * (BWL * 0.06) * taper / naca_max
    sweep_shift = (np.where(span_mask, zq, root_z) - root_z) \
        * np.tan(np.deg2rad(keel_rake))
    x_start = -local_chord / 2.0 + sweep_shift + keel_x_pos
    xi = (xq - x_start) / local_chord
    le_frac = 0.004
    te_frac = 0.97
    # Clip before the thickness eval: the mesh clamps y at the blunt LE/TE
    # caps, and _naca_thickness is only valid on [0,1] (sqrt of negative x).
    xi_clipped = np.clip(xi, le_frac, te_frac)
    y_half = _naca_thickness(xi_clipped, keel_tc) * scale
    chord_mask = (xi >= 0.0) & (xi <= 1.0)
    return np.where(span_mask & chord_mask, y_half, 0.0)


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

        x_dict_lcb = x_dict.get("LCB", 45.0)
        y_wl = _waterline_half_breadth(x_norm, BWL, Cp, Cm=Cm, LCB=x_dict_lcb)
        T_local = T_canoe * (1.0 - 0.3 * (1.0 - x_norm))
        z_norm = np.clip(zs / T_local, -1.0, 0.0)

        dr = np.array([_interp_param(xn, deadrise) for xn in x_norm])
        br = np.array([_interp_bilge(xn, bilge) for xn in x_norm])
        fl = np.array([_interp_param(xn, flare, bow_factor=1.0, stern_factor=0.4) for xn in x_norm])

        y_local = np.zeros_like(xs)
        for i in range(len(xs)):
            yf = _section_curve(np.array([z_norm[i]]), y_wl[i], T_local[i], dr[i], br[i], fl[i])
            y_local[i] = y_wl[i] * yf[0]

        result[mask_below] = y_local * sac_scale

    # Above waterline (z > 0): flare from waterline half-breadth
    mask_above = (np.abs(xq) <= LWL / 2) & (zq > 0) & (zq <= max(0.5, x_dict.get('E', 0.2) * 3))
    if np.any(mask_above):
        xs = xq[mask_above]
        zs = zq[mask_above]
        x_norm = (xs + LWL / 2) / LWL
        x_dict_lcb = x_dict.get("LCB", 45.0)
        y_wl = _waterline_half_breadth(x_norm, BWL, Cp, Cm=Cm, LCB=x_dict_lcb)
        E_val = x_dict.get("E", 0.2)
        fl_ts = np.array([_topside_wall_angle(xn, flare, BWL, E_val, yw)
                          for xn, yw in zip(x_norm, y_wl)])
        y_flare = y_wl + zs * np.tan(fl_ts)
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
                          target_nabla_eff: Optional[float] = None,
                          station_areas: Optional[np.ndarray] = None,
                          bulb_pos: float = 0.4,
                          bulb_density: float = 11340,
                          mast_masses: tuple = (0.0, 0.0),
                          mast_cg_z: tuple = (0.0, 0.0),
                          mast_cg_x: tuple = (0.0, 0.0),
                          x_dict: Optional[dict] = None,
                          config=None) -> dict:
    total_mesh_vol = max(abs(mesh.volume), 0.0)
    total_mesh_vol = total_mesh_vol if np.isfinite(total_mesh_vol) and total_mesh_vol > 0 else None
    # Canonical displacement below the waterplane z=0.  This is the single
    # source of truth for nabla/underwater_volume: the GZ curve machinery
    # (hydrostatics._gz_curve_mesh_clip) slices the same hull-only mesh with
    # the same plane, so the upright GZ volume agrees with the hydro
    # displacement and the 10% righting-energy warning stays silent.
    volume = mesh_displacement(mesh, 0.0)
    if not (volume > 0.0):
        volume = max(total_mesh_vol, 1e-10) if total_mesh_vol is not None else 1e-10
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
                submerged_centroid = submerged.center_mass
    except Exception as e:
        logger.warning(f"Underwater volume slice failed: {e}")
    if sac_volume is not None and sac_volume > 0 and volume > 0:
        # The mesh is hull-only (keel/bulb excluded, see generate_hull)
        # and sac_volume is the hull-only underwater SAC integral, so
        # compare directly — subtracting keel/bulb would double-count.
        sac_vol_hull_only = sac_volume
        mesh_hull_vol_est = volume
        vol_ratio = mesh_hull_vol_est / max(1e-10, sac_vol_hull_only)
        if vol_ratio > 1.5 or vol_ratio < 0.5:
            logger.warning(f"Mesh hull volume ~{mesh_hull_vol_est:.4f} vs SAC volume {sac_vol_hull_only:.4f} ratio={vol_ratio:.3f}")
    if sac_volume is not None and sac_volume > 0 and target_displacement is not None and target_displacement > 0:
        ratio = sac_volume / target_displacement
        # Hard-reject when the SAC cap (1.30) cannot reach the target: a hull
        # whose scaled capacity is below 75% of the design displacement can
        # never be inflated there, so all downstream metrics (BM, GM, CG, GZ)
        # would be computed on the wrong hull.  Fires as E_DISP so the
        # optimizer records E_GEOM/E_DISP rejects.
        if ratio < 0.75:
            raise ValueError(
                f"E_DISP: SAC volume {sac_volume:.4f} far from target "
                f"{target_displacement:.4f} (raw hull capacity too small)"
            )
        if ratio > 1.5:
            logger.warning(f"SAC volume {sac_volume:.4f} far above target {target_displacement:.4f} ratio={ratio:.3f}")
        elif ratio < 1.0:
            logger.warning(f"SAC volume {sac_volume:.4f} below target {target_displacement:.4f} ratio={ratio:.3f} (near-miss, not rejected)")
    if volume > 0 and target_displacement is not None and target_displacement > 0:
        mesh_target_ratio = volume / target_displacement
        # SAC-gamed hulls: the 1.30 SAC cap means a hull whose raw capacity is
        # < ~0.77x of the target can never reach the design displacement — all
        # downstream metrics (BM, GM, CG, GZ) are computed on the wrong hull.
        # Hard-reject below 50% (aligned with constraints.py: hull_volume < 50%
        # of target is a violation); the upper side stays a warning.
        if mesh_target_ratio < 0.5:
            raise ValueError(
                f"Mesh underwater volume {volume:.4f} m³ is {mesh_target_ratio:.2%} "
                f"of target displacement {target_displacement:.4f} m³ — SAC-cap "
                f"cannot reach the target (SAC gaming) — infeasible design "
                f"(stored as E_GEOM)"
            )
        if mesh_target_ratio > 2.0:
            logger.warning(f"Mesh underwater volume {volume:.4f} vs target {target_displacement:.4f} ratio={mesh_target_ratio:.3f} - possible SAC gaming")
    if total_mesh_vol is not None and volume > 0:
        total_ratio = total_mesh_vol / volume
        # High-freeboard/flared hulls legitimately exceed a fixed 2.5 ratio;
        # scale the upper warn threshold with sheer height vs draft.
        sheer_est = max(float(mesh.bounds[1, 2]), 0.0)
        adaptive_limit = max(2.5, 1.0 + 3.0 * sheer_est / max(T_canoe, 1e-3))
        if total_ratio > adaptive_limit or total_ratio < 0.8:
            logger.warning(f"Total mesh volume {total_mesh_vol:.4f} vs underwater volume {volume:.4f} ratio={total_ratio:.3f} (limit {adaptive_limit:.2f})")
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
    mast_mass_total = float(mast_masses[0]) + float(mast_masses[1])
    total_mass = volume * rho + mast_mass_total
    bulb_mass = bulb_vol * bulb_density  # Always from actual bulb geometry
    keel_mass = max(0, D_keel * keel_chord * (BWL * 0.06) * 0.5 * 1025)  # Always from actual keel geometry
    # Single source of truth for the mass/CG model: hydrostatics.compute_cg_z/x
    # (payload at deck height, ballast IN the bulb, hull structural floor —
    # mirrors the audit fixes). Kept in sync with the FoM/constraint path.
    if x_dict is not None:
        from hull_opt.hydrostatics import compute_cg_z as _h_cg_z, compute_cg_x as _h_cg_x
        try:
            cg_z = _h_cg_z(x_dict, nabla=volume, config=config)
            cg_x = _h_cg_x(x_dict, config, cb_x=float(center_mass[0]))
            if not np.isfinite(cg_z) or not np.isfinite(cg_x):
                raise ValueError(f"Non-finite unified CG: z={cg_z}, x={cg_x}")
        except Exception:
            cg_z = None
            cg_x = None
        # Reporting-only split (CG itself comes from the unified model).
        payload_mass = float(getattr(config.fixed, "payload_mass_kg", 0.0)) if config is not None else 0.0
        ballast_mass = max(0.0, total_mass * ballast_frac)
        hull_mass = max(0.0, total_mass - bulb_mass - keel_mass - ballast_mass
                        - mast_mass_total - payload_mass)
    else:
        cg_z = None
        cg_x = None
    if cg_z is None:
        # Fallback: legacy residual split (only when no design vector available)
        residual_budget = total_mass - bulb_mass - keel_mass - mast_mass_total
        ballast_mass = residual_budget * ballast_frac
        hull_mass = residual_budget - ballast_mass
        if hull_mass < 0:
            raise ValueError(
                f"Negative hull mass ({hull_mass:.4f} kg): ballast {ballast_mass:.4f} kg + "
                f"bulb {bulb_mass:.4f} kg + keel {keel_mass:.4f} kg + mast "
                f"{mast_mass_total:.4f} kg exceed total mass "
                f"{total_mass:.4f} kg — infeasible design (stored as E_GEOM)"
            )
        bulb_cg_z = -(T_canoe + D_keel)
        keel_cg_z = -(T_canoe + D_keel * 0.5)
        ballast_cg_z = -(T_canoe + D_keel * 0.5)  # ballast distributed in keel region
        hull_cg_z = float(center_mass[2]) if np.isfinite(center_mass[2]) and abs(center_mass[2]) < T_canoe * 2 else -T_canoe * 0.4
        cg_z = (hull_mass * hull_cg_z + keel_mass * keel_cg_z + bulb_mass * bulb_cg_z
                + ballast_mass * ballast_cg_z
                + mast_masses[0] * mast_cg_z[0] + mast_masses[1] * mast_cg_z[1]) / max(1e-10, total_mass)

        # Longitudinal CG (mesh frame, bow=0, stern=LWL): hull at CB_x,
        # keel+ballast ~ midship, bulb at its position, masts at their positions.
        hull_cg_x = float(center_mass[0]) if np.isfinite(center_mass[0]) and 0 <= center_mass[0] <= LWL else LWL * 0.4
        keel_ballast_cg_x = LWL / 2.0
        bulb_cg_x = bulb_pos * LWL
        cg_x = (hull_mass * hull_cg_x + keel_mass * keel_ballast_cg_x
                + bulb_mass * bulb_cg_x + ballast_mass * keel_ballast_cg_x
                + mast_masses[0] * mast_cg_x[0] + mast_masses[1] * mast_cg_x[1]) / max(1e-10, total_mass)

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
        "target_nabla_eff": target_nabla_eff if target_nabla_eff is not None else volume,
        "rho": rho,
        "cg_z": float(cg_z),
        "cg_x": float(cg_x),
        "total_mass_kg": float(total_mass),
        "mast_mass_total_kg": float(mast_mass_total),
        "bulb_mass_kg": float(bulb_mass),
        "keel_mass_kg": float(keel_mass),
        "ballast_mass_kg": float(ballast_mass),
        "hull_mass_kg": float(hull_mass),
        "ballast_frac": ballast_frac,
        "sac_scale_factor": sac_scale_factor,
        "sac_scale_std": sac_scale_std,
    }
