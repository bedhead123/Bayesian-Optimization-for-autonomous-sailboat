"""
Hydrostatic computations: GZ righting arm curve, righting energy,
CG position, downflooding angle, reserve buoyancy, wind heeling.
Uses trimesh slicing for submerged volume at each heel angle.
Key exports: compute_gz_curve(), compute_righting_energy(), compute_cg_z(),
             compute_hydrostatics(), check_inverted_stability()
Bugs fixed: downflooding angle logic (#3)
"""
import logging
import numpy as np
import trimesh
from typing import Optional

logger = logging.getLogger(__name__)


def _mesh_volume(mesh: trimesh.Trimesh) -> float:
    v = mesh.vertices
    f = mesh.faces
    cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vol = float(abs(np.sum(v[f[:, 0]] * cross, axis=1).sum() / 6.0))
    if not np.isfinite(vol):
        raise ValueError(f"Non-finite mesh volume computed: {vol}")
    return vol


def compute_hydrostatics(mesh_path: str, x_dict: Optional[dict] = None) -> dict:
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    volume = _mesh_volume(mesh)
    if not np.isfinite(volume) or volume < 0:
        volume = 0.0
    center_mass = mesh.center_mass

    Ix, Iy, wp_area = _waterplane_properties(mesh)

    B = x_dict.get("B", 1.0) if x_dict else 1.0
    LWL = x_dict.get("LWL", 2.4) if x_dict else 2.4
    Cp = x_dict.get("Cp", 0.55) if x_dict else 0.55
    target_nabla = x_dict.get("target_nabla", x_dict.get("target_displacement", 0.25)) if x_dict else 0.25

    BM = Ix / max(1e-10, volume)
    BML = Iy / max(1e-10, volume)
    Am = target_nabla / max(1e-10, Cp * LWL)

    return {
        "nabla": volume,
        "CB_x": float(center_mass[0]),
        "CB_y": float(center_mass[1]),
        "CB_z": float(center_mass[2]),
        "Ix": Ix,
        "Iy": Iy,
        "waterplane_area": wp_area,
        "BM": BM,
        "BML": BML,
        "Am": Am,
        "Cp": Cp,
        "B": B,
        "LWL": LWL,
        "target_nabla": target_nabla,
    }


def _waterplane_properties(mesh: trimesh.Trimesh) -> tuple[float, float, float]:
    try:
        plane_normal = np.array([0, 0, -1])
        plane_origin = np.array([0, 0, 0])
        wp_mesh = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal, plane_origin, cap=True
        )
        if wp_mesh is None or wp_mesh.vertices.shape[0] < 4:
            return 0.0, 0.0, 0.0

        verts = wp_mesh.vertices
        faces = wp_mesh.faces
        verts_2d = verts.copy()
        verts_2d[:, 2] = 0.0

        cap_mask = np.abs(verts[:, 2]) < 1e-6
        if cap_mask.sum() < 3:
            return 0.0, 0.0, 0.0

        cap_faces = faces[np.all(cap_mask[faces], axis=1)]
        if cap_faces.shape[0] < 1:
            return 0.0, 0.0, 0.0

        Ix = 0.0
        Iy = 0.0
        for tri in cap_faces:
            v = verts_2d[tri]
            x1, y1 = v[0, 0], v[0, 1]
            x2, y2 = v[1, 0], v[1, 1]
            x3, y3 = v[2, 0], v[2, 1]
            cross = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            area_tri = 0.5 * abs(cross)
            Ix += area_tri * (y1 ** 2 + y2 ** 2 + y3 ** 2 + y1 * y2 + y2 * y3 + y3 * y1) / 6.0
            Iy += area_tri * (x1 ** 2 + x2 ** 2 + x3 ** 2 + x1 * x2 + x2 * x3 + x3 * x1) / 6.0

        wp_area = 0.5 * abs(np.sum(
            verts_2d[cap_faces[:, 0], 0] * (verts_2d[cap_faces[:, 1], 1] - verts_2d[cap_faces[:, 2], 1])
            + verts_2d[cap_faces[:, 1], 0] * (verts_2d[cap_faces[:, 2], 1] - verts_2d[cap_faces[:, 0], 1])
            + verts_2d[cap_faces[:, 2], 0] * (verts_2d[cap_faces[:, 0], 1] - verts_2d[cap_faces[:, 1], 1])
        ))

        return Ix, Iy, wp_area
    except Exception as e:
        logger.debug(f"Waterplane properties computation failed: {e}")
        return 0.0, 0.0, 0.0


def compute_gz_curve(mesh_path: str, cg_z: float = -0.05,
                     n_angles: int = 37, max_heel: float = 180.0) -> np.ndarray:
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    angles = np.linspace(0, max_heel, n_angles)
    gz = np.zeros(n_angles)
    submerged_volumes = np.zeros(n_angles)

    cg = np.array([0.0, 0.0, cg_z])

    for i, deg in enumerate(angles):
        rad = np.deg2rad(deg)
        rot_mesh = mesh.copy()
        rot_mesh.apply_translation(-cg)
        # Heel to starboard: CB shifts starboard (positive y), GZ = CB_y positive for stable hull
        rot_matrix = trimesh.transformations.rotation_matrix(-rad, [1, 0, 0])
        rot_mesh.apply_transform(rot_matrix)
        rot_mesh.apply_translation(cg)

        try:
            plane_normal = np.array([0, 0, -1])
            plane_origin = np.array([0, 0, 0])
            submerged = trimesh.intersections.slice_mesh_plane(
                rot_mesh, plane_normal, plane_origin, cap=True
            )
            if submerged is None or submerged.vertices.shape[0] < 4:
                gz[i] = 0.0
                submerged_volumes[i] = 0.0
                continue

            vol = _mesh_volume(submerged)
            cb = submerged.center_mass
            submerged_volumes[i] = vol

            if vol > 1e-10:
                gz_rot = cb[1]
                gz[i] = gz_rot
        except Exception as e:
            logger.warning(f"GZ curve slice at angle {deg} failed: {e}")
            gz[i] = float('nan')
            submerged_volumes[i] = float('nan')

    nan_count = np.sum(~np.isfinite(gz))
    if nan_count > 0:
        logger.warning(f"{nan_count}/{len(gz)} GZ values are NaN (slice failures)")
        gz = np.nan_to_num(gz, nan=0.0)
        submerged_volumes = np.nan_to_num(submerged_volumes, nan=0.0)

    return np.column_stack([angles, gz, submerged_volumes])


def compute_righting_energy(gz_curve: np.ndarray, max_heel_deg: float = 90.0,
                            rho: float = 1025.0, g: float = 9.81,
                            displacement: float = 0.25) -> float:
    angles = gz_curve[:, 0]
    gz = gz_curve[:, 1]
    submerged_volumes = gz_curve[:, 2] if gz_curve.shape[1] > 2 else None
    if submerged_volumes is not None and len(submerged_volumes) > 0:
        upright_vol = submerged_volumes[0]
        if upright_vol > 0 and displacement > 0:
            vol_ratio = abs(upright_vol - displacement) / displacement
            if vol_ratio > 0.1:
                import warnings
                warnings.warn(f"GZ curve submerged volume ({upright_vol:.4f}) differs "
                              f"from displacement ({displacement:.4f}) by {vol_ratio*100:.1f}%")
    mask = angles <= max_heel_deg
    if np.sum(mask) < 2:
        return 0.0
    if not np.all(np.isfinite(gz[mask])):
        return 0.0
    angle_rad = np.deg2rad(angles[mask])
    gz_valid = gz[mask]
    gz_safe = np.nan_to_num(gz_valid, nan=0.0)
    gz_positive = np.maximum(gz_safe, 0.0)
    area_under_curve = float(np.trapezoid(gz_positive, angle_rad))
    energy = rho * g * displacement * area_under_curve
    return energy


def compute_cg_z(x_dict: dict, nabla: Optional[float] = None) -> float:
    T_hull = x_dict.get("T_canoe", 0.3)
    D_keel = x_dict.get("D_keel", 1.0)
    ballast_frac = x_dict.get("ballast_frac", 0.30)
    bulb_vol = x_dict.get("bulb_vol", 0.0)
    keel_chord = x_dict.get("keel_chord", 0.2)
    BWL = x_dict.get("BWL", 0.5)

    rho = 1025.0
    if nabla is not None and nabla > 0:
        total_mass = nabla * rho
        bulb_mass = bulb_vol * 11340  # Always from actual bulb geometry
        keel_mass = max(0, D_keel * keel_chord * (BWL * 0.06) * 0.5 * 1025)  # Always from actual keel geometry
        ballast_mass = total_mass * ballast_frac  # Ballast is additional internal mass
        hull_mass = total_mass - bulb_mass - keel_mass - ballast_mass
        if hull_mass < 0:
            hull_mass = 0
    else:
        LWL = x_dict.get("LWL", 2.4)
        T_canoe_hull = x_dict.get("T_canoe", 0.3)
        Cp_hull = x_dict.get("Cp", 0.55)
        Cm_hull = x_dict.get("Cm", 0.75)
        nabla_approx = BWL * LWL * T_canoe_hull * Cp_hull * Cm_hull
        total_mass = nabla_approx * rho
        bulb_mass = bulb_vol * 11340  # Always from actual bulb geometry
        keel_mass = max(0, D_keel * keel_chord * (BWL * 0.06) * 0.5 * 1025)  # Always from actual keel geometry
        ballast_mass = total_mass * ballast_frac  # Ballast is additional internal mass
        hull_mass = total_mass - bulb_mass - keel_mass - ballast_mass
        if hull_mass < 0:
            hull_mass = 0

    hull_cg_z = -T_hull * 0.4
    keel_cg_z = -(T_hull + D_keel * 0.5)
    bulb_cg_z = -(T_hull + D_keel)
    ballast_cg_z = -(T_hull + D_keel * 0.5)  # ballast distributed in keel region
    cg_z = (hull_mass * hull_cg_z + keel_mass * keel_cg_z + bulb_mass * bulb_cg_z + ballast_mass * ballast_cg_z) / max(1e-10, total_mass)
    return cg_z


def compute_downflooding_angle(mesh_path: str, cg_z: float = 0.0) -> float:
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    orig_verts = mesh.vertices
    # Gunwale/sheer edge: highest point per longitudinal station on each side.
    deck_mask = np.zeros(len(orig_verts), dtype=bool)
    for x in np.unique(np.round(orig_verts[:, 0], 3)):
        at_x = np.abs(orig_verts[:, 0] - x) < 0.02
        side = at_x & (np.abs(orig_verts[:, 1]) > 0.05) & (orig_verts[:, 2] > 0.0)
        if side.any():
            idx = int(np.where(side)[0][np.argmax(orig_verts[side, 2])])
            deck_mask[idx] = True
    if deck_mask.sum() < 3:
        deck_mask = (orig_verts[:, 2] > 0.05) & (np.abs(orig_verts[:, 1]) > 1e-4)

    cg = np.array([0.0, 0.0, cg_z])
    angles = np.linspace(0, 180.0, 91)

    for deg in angles:
        rad = np.deg2rad(deg)
        rot_mesh = mesh.copy()
        rot_mesh.apply_translation(-cg)
        rot_matrix = trimesh.transformations.rotation_matrix(rad, [1, 0, 0])
        rot_mesh.apply_transform(rot_matrix)
        rot_mesh.apply_translation(cg)

        verts = rot_mesh.vertices
        deck_verts_z = verts[deck_mask, 2]
        if len(deck_verts_z) == 0:
            continue
        rotated_deck_z = float(deck_verts_z.min())
        if rotated_deck_z < 0.0:
            return float(deg)

    return 180.0


def compute_reserve_buoyancy(mesh_path: str, x_dict: dict) -> float:
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    total_vol = _mesh_volume(mesh)
    if total_vol < 1e-10:
        return 0.0

    plane_normal = np.array([0, 0, -1])
    plane_origin = np.array([0, 0, 0])
    submerged = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal, plane_origin, cap=True
    )
    if submerged is None or submerged.vertices.shape[0] < 4:
        return 1.0

    submerged_vol = _mesh_volume(submerged)
    above_vol = total_vol - submerged_vol
    # Guard against mesh slicing artifacts (cap can add volume)
    if submerged_vol > total_vol:
        submerged_vol = total_vol * 0.95
        above_vol = total_vol - submerged_vol
    return above_vol / max(1e-10, total_vol)


def compute_wind_heeling_arm(heel_deg: float, wind_speed_ms: float,
                             sail_area: float, sail_height: float,
                             displacement: float, rho_water: float = 1025.0,
                             g: float = 9.81, rho_air: float = 1.225) -> float:
    rad = np.deg2rad(heel_deg)
    effective_area = sail_area * np.cos(rad)
    heeling_force = 0.5 * rho_air * wind_speed_ms ** 2 * effective_area
    heeling_moment = heeling_force * sail_height * np.cos(rad)
    denominator = rho_water * g * displacement
    if abs(denominator) < 1e-12:
        return float('inf')
    heeling_arm = heeling_moment / denominator
    return heeling_arm


def check_inverted_stability(mesh_path: str, cg_z: float = -0.05) -> bool:
    gz_curve = compute_gz_curve(mesh_path, cg_z, n_angles=37, max_heel=180.0)
    angles = gz_curve[:, 0]
    gz = gz_curve[:, 1]

    near_inv_mask = (angles >= 140.0) & (angles < 180.0)
    if np.sum(near_inv_mask) < 3:
        return False

    gz_near = gz[near_inv_mask]

    gz_clean = gz_near[np.isfinite(gz_near)]
    if len(gz_clean) < 3:
        return False
    return float(np.mean(gz_clean)) > 0.005
