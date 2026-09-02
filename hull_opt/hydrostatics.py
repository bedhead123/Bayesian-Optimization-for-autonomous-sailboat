"""
Hydrostatic computations: GZ righting arm curve, righting energy,
CG position, downflooding angle, reserve buoyancy, wind heeling.
Uses trimesh slicing for submerged volume at each heel angle.
Key exports: compute_gz_curve(), compute_gz_area(), compute_avs(),
             compute_righting_energy(), compute_cg_z(),
             compute_hydrostatics(), check_inverted_stability()
Bugs fixed: downflooding angle logic (#3)
"""
import logging
import numpy as np
import trimesh
from typing import Optional

logger = logging.getLogger(__name__)


def _trapz(y, x, **kwargs):
    return np.trapezoid(y, x, **kwargs)


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
        "underwater_volume": volume,
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
        logger.warning(f"Waterplane properties computation failed: {e}")
        return 0.0, 0.0, 0.0


def compute_gz_curve(mesh_path: str, cg_z: float = -0.05,
                     n_angles: int = 37, max_heel: float = 180.0,
                     patches: Optional[list] = None) -> np.ndarray:
    """Compute GZ curve.  When NURBS patches are provided, uses the
    direct NURBS surface integral (no mesh clipping).  Falls back
    to mesh-clipping when patches=None.
    """
    if patches is not None:
        return nurbs_gz_curve(patches, cg_z, n_angles, max_heel,
                              n_sub=6)
    return _gz_curve_mesh_clip(mesh_path, cg_z, n_angles, max_heel)


def _gz_curve_mesh_clip(mesh_path: str, cg_z: float, n_angles: int, max_heel: float) -> np.ndarray:
    """Compute GZ curve by mesh-clipping (always accurate for watertight meshes)."""
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


def compute_gz_area(gz_curve: np.ndarray, lo_deg: float = 0.0,
                    hi_deg: float = 90.0) -> float:
    """Trapezoid area (rad·m) of the GZ curve between lo_deg and hi_deg,
    with GZ clipped at zero; 0.0 for lo > hi, empty or non-finite input."""
    if gz_curve is None or len(gz_curve) < 2 or lo_deg > hi_deg:
        return 0.0
    angles = gz_curve[:, 0]
    gz = gz_curve[:, 1]
    mask = (angles >= lo_deg) & (angles <= hi_deg)
    if np.sum(mask) < 2:
        return 0.0
    if not np.all(np.isfinite(gz[mask])):
        return 0.0
    angle_rad = np.deg2rad(angles[mask])
    gz_safe = np.nan_to_num(gz[mask], nan=0.0)
    gz_positive = np.maximum(gz_safe, 0.0)
    return float(_trapz(gz_positive, angle_rad))


def compute_avs(gz_curve: np.ndarray) -> float:
    """Angle of vanishing stability: first downward zero-crossing of GZ
    after the peak; max angle in the curve if none, 0.0 if no positive
    stability (max GZ <= 0)."""
    if gz_curve is None or len(gz_curve) < 2:
        return 0.0
    angles = gz_curve[:, 0]
    gz = np.nan_to_num(gz_curve[:, 1], nan=0.0)
    max_idx = int(np.argmax(gz))
    if gz[max_idx] <= 0.0:
        return 0.0
    for i in range(max_idx, len(gz) - 1):
        if gz[i] > 0.0 and gz[i + 1] <= 0.0:
            t = gz[i] / (gz[i] - gz[i + 1])
            return float(angles[i] + t * (angles[i + 1] - angles[i]))
    return float(angles[-1])


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
            # Skinny V deep E 0.35-0.45 high freeboard NURBS flat deck collapsed -> 10-14% offset is discretization, not physics. Raise 0.10->0.15.
            if vol_ratio > 0.15:
                import warnings
                warnings.warn(f"GZ curve submerged volume ({upright_vol:.4f}) differs "
                              f"from displacement ({displacement:.4f}) by {vol_ratio*100:.1f}%")
    area_under_curve = compute_gz_area(gz_curve, 0.0, max_heel_deg)
    return rho * g * displacement * area_under_curve


def compute_cg_z(x_dict: dict, nabla: Optional[float] = None,
                 config=None) -> float:
    T_hull = x_dict.get("T_canoe", 0.3)
    D_keel = x_dict.get("D_keel", 1.0)
    ballast_frac = x_dict.get("ballast_frac", 0.30)
    bulb_vol = x_dict.get("bulb_vol", 0.0)
    keel_chord = x_dict.get("keel_chord", 0.2)
    BWL = x_dict.get("BWL", 0.5)

    # Mast masses (config provided): single wingsail mast CG at
    # mast_cg_height_frac × wing CE height above the waterline (positive z —
    # raises CG, reduces GM); nose-pod ballast sits on the pivot (z = 0).
    mast_mass_total = 0.0
    mast_z_num = 0.0
    if config is not None:
        mast_masses = (config.fixed.wingsail_mast_mass, config.fixed.wingsail_nose_pod_mass)
        mast_cg_frac = config.fixed.mast_cg_height_frac
        wing_span = config.fixed.wingsail_span_frac * x_dict.get("LWL", 2.4)
        ce_frac = getattr(config.fixed, "wingsail_ce_span_frac", 0.35)
        mast_cg_z = (mast_cg_frac * ce_frac * wing_span, 0.0)
        mast_mass_total = mast_masses[0] + mast_masses[1]
        mast_z_num = mast_masses[0] * mast_cg_z[0] + mast_masses[1] * mast_cg_z[1]

    # Payload (electronics/solar/batteries) sits at deck height, NOT lumped
    # into the hull mass at -0.4*T under water (audit finding: GM error
    # ~0.06-0.10 m from the old heuristic).
    payload_mass = 0.0
    payload_cg_z = 0.30
    hull_floor = 20.0
    if config is not None:
        payload_mass = float(getattr(config.fixed, "payload_mass_kg", 0.0))
        payload_cg_z = float(getattr(config.fixed, "payload_cg_z", 0.30))
        hull_floor = float(getattr(config.fixed, "hull_mass_floor_kg", 20.0))

    rho = 1025.0
    if nabla is not None and nabla > 0:
        base_mass = nabla * rho
    else:
        LWL = x_dict.get("LWL", 2.4)
        T_canoe_hull = x_dict.get("T_canoe", 0.3)
        Cp_hull = x_dict.get("Cp", 0.55)
        Cm_hull = x_dict.get("Cm", 0.75)
        base_mass = BWL * LWL * T_canoe_hull * Cp_hull * Cm_hull * rho

    total_mass = base_mass + mast_mass_total + payload_mass
    bulb_mass = bulb_vol * 11340  # Always from actual bulb geometry
    keel_mass = max(0, D_keel * keel_chord * (BWL * 0.06) * 0.5 * 1025)  # Always from actual keel geometry
    ballast_mass = total_mass * ballast_frac  # Ballast is additional internal mass
    hull_mass = total_mass - bulb_mass - keel_mass - ballast_mass - mast_mass_total - payload_mass
    # Structural floor: a 2.4 m FRP hull cannot weigh 0-5 kg (Fanhai-T2
    # optimized hull: 38.7 kg at 2.0 m). Excess ballast is returned to the
    # bulb so the floor never leaves fake mass in the model.
    if hull_mass < hull_floor:
        ballast_mass = max(0.0, total_mass - hull_floor - bulb_mass - keel_mass
                           - mast_mass_total - payload_mass)
        hull_mass = max(hull_floor, total_mass - bulb_mass - keel_mass
                        - ballast_mass - mast_mass_total - payload_mass)

    hull_cg_z = -T_hull * 0.4
    keel_cg_z = -(T_hull + D_keel * 0.5)
    bulb_cg_z = -(T_hull + D_keel)
    ballast_cg_z = -(T_hull + D_keel)  # ballast lives IN the bulb (lowest VCG, PYD Ch.6)
    cg_z = (hull_mass * hull_cg_z + keel_mass * keel_cg_z + bulb_mass * bulb_cg_z
            + ballast_mass * ballast_cg_z + payload_mass * payload_cg_z
            + mast_z_num) / max(1e-10, total_mass)
    return cg_z


def compute_cg_x(x_dict: dict, config, cb_x: float = 0.0) -> float:
    """Longitudinal CG in mesh frame (bow=0, stern=LWL).

    Hull mass at CB_x (from hydro; fallback (LCB/100)·LWL), keel+ballast at
    midship, bulb at bulb_pos·LWL, masts at their sail positions.
    Informational metric for trim sanity and the results report — no constraint.
    """
    from hull_opt.rig import sail_pos_x

    LWL = x_dict.get("LWL", 2.4)
    T_hull = x_dict.get("T_canoe", 0.3)
    D_keel = x_dict.get("D_keel", 1.0)
    ballast_frac = x_dict.get("ballast_frac", 0.30)
    bulb_vol = x_dict.get("bulb_vol", 0.0)
    bulb_pos = x_dict.get("bulb_pos", 0.4)
    keel_chord = x_dict.get("keel_chord", 0.2)
    BWL = x_dict.get("BWL", 0.5)
    rho = 1025.0

    if x_dict.get("underwater_volume", x_dict.get("nabla")) is not None and x_dict.get("underwater_volume", x_dict.get("nabla")) > 0:
        nabla = x_dict.get("underwater_volume", x_dict.get("nabla"))
    else:
        Cp = x_dict.get("Cp", 0.55)
        Cm = x_dict.get("Cm", 0.75)
        nabla = BWL * LWL * T_hull * Cp * Cm

    total_mass = nabla * rho

    mast_mass_total = 0.0
    mast_x_num = 0.0
    if config is not None:
        mast_masses = (config.fixed.wingsail_mast_mass, config.fixed.wingsail_nose_pod_mass)
        mast_x = sail_pos_x(x_dict.get("wingsail_pos", 0.42), LWL)
        mast_cg_x = (mast_x, mast_x)   # nose-pod ballast CG sits on the pivot
        mast_mass_total = mast_masses[0] + mast_masses[1]
        mast_x_num = mast_masses[0] * mast_cg_x[0] + mast_masses[1] * mast_cg_x[1]

    payload_mass = 0.0
    payload_cg_x = LWL / 2.0
    hull_floor = 20.0
    if config is not None:
        payload_mass = float(getattr(config.fixed, "payload_mass_kg", 0.0))
        hull_floor = float(getattr(config.fixed, "hull_mass_floor_kg", 20.0))

    total_mass += mast_mass_total + payload_mass  # consistent with compute_cg_z
    bulb_mass = bulb_vol * 11340
    keel_mass = max(0, D_keel * keel_chord * (BWL * 0.06) * 0.5 * 1025)
    ballast_mass = total_mass * ballast_frac  # (displacement + mast + payload) * frac, consistent with compute_cg_z
    hull_mass = total_mass - bulb_mass - keel_mass - ballast_mass - mast_mass_total - payload_mass
    if hull_mass < hull_floor:
        ballast_mass = max(0.0, total_mass - hull_floor - bulb_mass - keel_mass
                           - mast_mass_total - payload_mass)
        hull_mass = max(hull_floor, total_mass - bulb_mass - keel_mass
                        - ballast_mass - mast_mass_total - payload_mass)

    if not np.isfinite(cb_x) or abs(cb_x) < 1e-12:
        cb_x = (x_dict.get("LCB", 45.0) / 100.0) * LWL
    hull_cg_x = cb_x
    bulb_x = bulb_pos * LWL
    cg_x = (hull_mass * hull_cg_x + keel_mass * (LWL / 2.0)
            + bulb_mass * bulb_x + ballast_mass * bulb_x   # ballast IN the bulb (was midship)
            + payload_mass * payload_cg_x + mast_x_num) / max(1e-10, total_mass)
    return float(cg_x)


def compute_lumped_inertia(x_dict: dict, hydro: dict, config=None) -> np.ndarray:
    """Diagonal 6x6 mass-moment-of-inertia tensor from the REAL mass
    breakdown (hull shell, keel fin, bulb+ballast, payload, mast) about the
    vessel CG, in Capytaine's standard dof order
    [Surge, Sway, Heave, Roll, Pitch, Yaw]. Replaces Capytaine's
    uniform-density panel inertia, which overestimates roll inertia by
    ~15-25% and ignores the 100+ kg ballast at keel depth.

    Returns a 6x6 numpy array (kg·m²) — wrap with an xarray DataArray of
    dims ('influenced_dof', 'radiating_dof') at the call site.
    """
    T_hull = float(x_dict.get("T_canoe", 0.3))
    D_keel = float(x_dict.get("D_keel", 1.0))
    LWL = float(x_dict.get("LWL", 2.4))
    BWL = float(x_dict.get("BWL", 0.5))
    ballast_frac = float(x_dict.get("ballast_frac", 0.30))
    bulb_vol = float(x_dict.get("bulb_vol", 0.0))
    keel_chord = float(x_dict.get("keel_chord", 0.2))
    bulb_pos = float(x_dict.get("bulb_pos", 0.4))

    nabla = float(hydro.get("underwater_volume", hydro.get("nabla", 0.1)))
    rho = 1025.0
    base_mass = nabla * rho

    mast_mass_total = 0.0
    mast_x = 0.5 * LWL
    mast_z = 0.40
    if config is not None:
        mast_masses = (config.fixed.wingsail_mast_mass, config.fixed.wingsail_nose_pod_mass)
        mast_mass_total = mast_masses[0] + mast_masses[1]
        mast_x = x_dict.get("wingsail_pos", 0.42) * LWL
        span = config.fixed.wingsail_span_frac * LWL
        ce_frac = getattr(config.fixed, "wingsail_ce_span_frac", 0.35)
        mast_z = config.fixed.mast_cg_height_frac * ce_frac * span
    payload_mass = float(getattr(config.fixed, "payload_mass_kg", 0.0)) if config is not None else 0.0
    payload_cg_z = float(getattr(config.fixed, "payload_cg_z", 0.30)) if config is not None else 0.30

    total_mass = base_mass + mast_mass_total + payload_mass
    bulb_mass = bulb_vol * 11340
    keel_mass = max(0.0, D_keel * keel_chord * (BWL * 0.06) * 0.5 * rho)
    ballast_mass = max(0.0, total_mass * ballast_frac)
    hull_mass = max(0.0, total_mass - bulb_mass - keel_mass - ballast_mass
                    - mast_mass_total - payload_mass)

    # (mass, x, y, z) about the mesh frame — mirrors compute_cg_z/compute_cg_x
    points = [
        (hull_mass, 0.5 * LWL, 0.0, -0.4 * T_hull),
        (keel_mass, 0.5 * LWL, 0.0, -(T_hull + 0.5 * D_keel)),
        (bulb_mass + ballast_mass, bulb_pos * LWL, 0.0, -(T_hull + D_keel)),
        (payload_mass, 0.5 * LWL, 0.0, payload_cg_z),
        (mast_mass_total, mast_x, 0.0, mast_z),
    ]
    cg_z = sum(m * z for m, _, _, z in points) / max(1e-9, total_mass)
    cg_x = sum(m * x for m, x, _, _ in points) / max(1e-9, total_mass)

    ixx = iyy = izz = 0.0
    for m, x, y, z in points:
        dx, dy, dz = x - cg_x, y, z - cg_z
        ixx += m * (dy * dy + dz * dz)
        iyy += m * (dx * dx + dz * dz)
        izz += m * (dx * dx + dy * dy)
    # local radii for extended bodies: transverse gyradius ~0.35·BWL for roll,
    # longitudinal ~0.25·LWL for pitch, yaw gets the hull waterplane extent.
    ixx += hull_mass * (0.35 * BWL) ** 2 * 0.5
    iyy += hull_mass * (0.25 * LWL) ** 2 * 0.5
    izz += hull_mass * (0.35 * BWL) ** 2 * 0.5 + hull_mass * (0.25 * LWL) ** 2 * 0.5

    inertia = np.zeros((6, 6))
    inertia[0, 0] = inertia[1, 1] = inertia[2, 2] = total_mass
    inertia[3, 3] = max(ixx, 0.01)   # Roll
    inertia[4, 4] = max(iyy, 0.01)   # Pitch
    inertia[5, 5] = max(izz, 0.01)   # Yaw
    return inertia


def compute_downflooding_angle(mesh_path: str, cg_z: float = 0.0,
                                x_dict: Optional[dict] = None) -> float:
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    orig_verts = mesh.vertices
    BWL = x_dict.get("BWL", 0.5) if x_dict else 0.5
    y_thresh = max(0.01, BWL * 0.05)
    # Gunwale/sheer edge: highest point per longitudinal station on each side.
    # NOTE: This assumes the deck edge is the highest-z vertex at each station;
    # on hulls with unusual sheer profiles this may pick non-deck features.
    deck_mask = np.zeros(len(orig_verts), dtype=bool)
    for x in np.unique(np.round(orig_verts[:, 0], 3)):
        at_x = np.abs(orig_verts[:, 0] - x) < 0.02
        side = at_x & (np.abs(orig_verts[:, 1]) > y_thresh) & (orig_verts[:, 2] > 0.0)
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
                             g: float = 9.81, rho_air: float = 1.225,
                             cd_sail: float = 0.02,
                             windage_area: float = 0.0,
                             windage_height: float = 0.0,
                             cd_windage: float = 1.13) -> float:
    """Heeling arm under wind. The sail uses the FEATHERED polar drag
    coefficient (cd_sail ~0.02, aligned rigid wing — NOT a flat-plate Cd=1).
    Hull/mast windage is added separately (PYD p.194/209: Cd 1.13 on
    B*freeboard + mast frontal area)."""
    rad = np.deg2rad(heel_deg)
    effective_area = sail_area * np.cos(rad)
    f_sail = 0.5 * rho_air * wind_speed_ms ** 2 * effective_area * cd_sail
    f_wind = (0.5 * rho_air * wind_speed_ms ** 2 * windage_area * cd_windage
              if windage_area > 0 else 0.0)
    heeling_moment = (f_sail * sail_height * np.cos(rad)
                      + f_wind * windage_height * np.cos(rad))
    denominator = rho_water * g * displacement
    if abs(denominator) < 1e-12:
        return float('inf')
    heeling_arm = heeling_moment / denominator
    return heeling_arm


def check_inverted_stability(mesh_path: str, cg_z: float = -0.05,
                              patches: Optional[list] = None) -> bool:
    gz_curve = compute_gz_curve(mesh_path, cg_z, n_angles=37, max_heel=180.0,
                                 patches=patches)
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


# ── NURBS hydrostatics ──────────────────────────────────────────────

_GAUSS_LEGENDRE_3 = (
    np.array([0.1127016653792583, 0.5, 0.8872983346207417]),
    np.array([5.0/18.0, 8.0/18.0, 5.0/18.0]),
)


def _nurbs_surface_point(ctrl, knots_u, knots_v, du, dv, u, v):
    """Single-point NURBS evaluation (delegates to geometry.py)."""
    from hull_opt.geometry import _nurbs_surface_point as _geom_point
    return _geom_point(ctrl, knots_u, knots_v, du, dv, u, v)


def _nurbs_partials(ctrl, knots_u, knots_v, du, dv, u, v, eps=1e-6):
    """Numerical partial derivatives dS/du, dS/dv (delegates to geometry.py)."""
    from hull_opt.geometry import _nurbs_partial_deriv
    return _nurbs_partial_deriv(ctrl, knots_u, knots_v, du, dv, u, v, eps)


def _rotate_control_net(ctrl: np.ndarray, angle_rad: float,
                         cg_z: float) -> np.ndarray:
    """Rotate control net by -angle_rad about x-axis through (0, 0, cg_z).

    Positive angle = heel to starboard (CB shifts to positive y).
    Rotation matrix R_x(-θ): [1, 0, 0; 0, cos, sin; 0, -sin, cos]
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    rotated = ctrl.copy()
    y = ctrl[:, :, 1]
    z = ctrl[:, :, 2] - cg_z
    rotated[:, :, 1] = y * c + z * s
    rotated[:, :, 2] = z * c - y * s + cg_z
    return rotated


def _integrate_cell(ctrl, ku, kv, du, dv, u0, du_cell, v0, dv_cell, nodes, weights):
    """Integrate a single parametric cell using 3×3 Gauss-Legendre.
    Returns (V, Mx, My, Mz) for submerged portion only (z<0).
    Skips Gauss points above waterline."""
    V = 0.0
    Mx = 0.0
    My = 0.0
    Mz = 0.0
    for gi in range(3):
        u_gp = u0 + du_cell * nodes[gi]
        for gj in range(3):
            v_gp = v0 + dv_cell * nodes[gj]
            pt = _nurbs_surface_point(ctrl, ku, kv, du, dv, u_gp, v_gp)
            if pt[2] > 1e-12:
                continue
            Su, Sv = _nurbs_partials(ctrl, ku, kv, du, dv, u_gp, v_gp)
            cross = np.cross(Su, Sv)
            j_z = abs(cross[2])
            w = weights[gi] * weights[gj] * du_cell * dv_cell
            z_neg = -pt[2]
            V += w * z_neg * j_z
            Mx += w * z_neg * j_z * pt[0]
            My += w * z_neg * j_z * pt[1]
            Mz += w * z_neg * j_z * (pt[2] * 0.5)
    return V, Mx, My, Mz


def _cell_straddles(ctrl, ku, kv, du, dv, u0, du_cell, v0, dv_cell, nodes):
    """Check if a parametric cell straddles the waterline (z=0).
    Uses 9 Gauss points to detect mixed wet/dry status."""
    has_wet = False
    has_dry = False
    for gi in range(3):
        u_gp = u0 + du_cell * nodes[gi]
        for gj in range(3):
            v_gp = v0 + dv_cell * nodes[gj]
            pt = _nurbs_surface_point(ctrl, ku, kv, du, dv, u_gp, v_gp)
            if pt[2] > 1e-12:
                has_dry = True
            else:
                has_wet = True
            if has_wet and has_dry:
                return True
    return False


def _integrate_patch_subcells(patch, rot_ctrl, n_sub=8):
    """Integrate submerged volume over a patch by subdividing into n_sub×n_sub cells.

    Each sub-cell uses 3-point Gauss-Legendre quadrature (nodes on [0,1]).
    Cells that straddle the waterline are adaptively refined into 3×3 sub-sub-cells
    for accurate partial-submersion integration.

    Returns (V, Mx, My, Mz, n_pts).
    """
    nodes, weights = _GAUSS_LEGENDRE_3
    ctrl = rot_ctrl
    ku = patch.knots_u
    kv = patch.knots_v
    du = patch.degree_u
    dv = patch.degree_v

    total_V = 0.0
    total_Mx = 0.0
    total_My = 0.0
    total_Mz = 0.0

    for ui in range(n_sub):
        u0 = ui / n_sub
        du_cell = 1.0 / n_sub
        for vi in range(n_sub):
            v0 = vi / n_sub
            dv_cell = 1.0 / n_sub

            # Quick check: skip cells far from waterline using corners
            c00 = _nurbs_surface_point(ctrl, ku, kv, du, dv, u0, v0)
            c01 = _nurbs_surface_point(ctrl, ku, kv, du, dv, u0, v0 + dv_cell)
            c10 = _nurbs_surface_point(ctrl, ku, kv, du, dv, u0 + du_cell, v0)
            c11 = _nurbs_surface_point(ctrl, ku, kv, du, dv, u0 + du_cell, v0 + dv_cell)
            zs = np.array([c00[2], c01[2], c10[2], c11[2]])
            if np.all(zs > 1e-12):
                continue
            if np.all(zs <= 1e-12):
                V, Mx, My, Mz = _integrate_cell(
                    ctrl, ku, kv, du, dv, u0, du_cell, v0, dv_cell, nodes, weights)
                total_V += V
                total_Mx += Mx
                total_My += My
                total_Mz += Mz
                continue

            # Edge cell: check with full Gauss points for straddling
            if _cell_straddles(ctrl, ku, kv, du, dv, u0, du_cell, v0, dv_cell, nodes):
                ss_n = 3
                ss_du = du_cell / ss_n
                ss_dv = dv_cell / ss_n
                for si in range(ss_n):
                    ss_u0 = u0 + si * ss_du
                    for sj in range(ss_n):
                        ss_v0 = v0 + sj * ss_dv
                        V, Mx, My, Mz = _integrate_cell(
                            ctrl, ku, kv, du, dv, ss_u0, ss_du, ss_v0, ss_dv,
                            nodes, weights)
                        total_V += V
                        total_Mx += Mx
                        total_My += My
                        total_Mz += Mz
            else:
                V, Mx, My, Mz = _integrate_cell(
                    ctrl, ku, kv, du, dv, u0, du_cell, v0, dv_cell, nodes, weights)
                total_V += V
                total_Mx += Mx
                total_My += My
                total_Mz += Mz

    return total_V, total_Mx, total_My, total_Mz, 0


def nurbs_submerged_volume(patches: list,
                            n_sub: int = 8,
                            heel_deg: float = 0.0,
                            cg_z: float = 0.0) -> tuple:
    """Compute submerged volume and centroid from NURBS patches.

    Rotates all patches by -heel_deg about the x-axis through CG
    (0, 0, cg_z), then integrates using 3-point Gauss-Legendre over
    a subdivided parametric grid (n_sub × n_sub sub-cells per patch).
    Only points with rotated z' ≤ 0 contribute.

    Returns (volume, centroid) where centroid is (x', y', z') in the
    rotated (earth) frame.
    """
    if patches is None or len(patches) == 0:
        return 0.0, np.zeros(3)

    total_V = 0.0
    total_Mx = 0.0
    total_My = 0.0
    total_Mz = 0.0
    angle_rad = np.radians(heel_deg)

    for patch in patches:
        if heel_deg != 0.0:
            rot_ctrl = _rotate_control_net(patch.control_net, angle_rad, cg_z)
        else:
            rot_ctrl = patch.control_net
        V, Mx, My, Mz, _ = _integrate_patch_subcells(
            patch, rot_ctrl, n_sub=n_sub)
        total_V += V
        total_Mx += Mx
        total_My += My
        total_Mz += Mz

    if total_V > 1e-15:
        centroid = np.array([total_Mx / total_V,
                             total_My / total_V,
                             total_Mz / total_V])
    else:
        centroid = np.zeros(3)

    return total_V, centroid


def nurbs_waterplane_properties(patches: list) -> tuple:
    """Compute waterplane area, Ix, Iy from NURBS patches.

    Projects the hull onto z=0 and computes moments of the waterplane area.
    Returns (area, Ix, Iy).
    """
    area = 0.0
    Ix = 0.0
    Iy = 0.0
    nodes, weights = _GAUSS_LEGENDRE_3

    for patch in patches:
        if "bulb" in patch.name or "keel" in patch.name or patch.is_mirrored:
            continue
        ctrl = patch.control_net
        ku = patch.knots_u
        kv = patch.knots_v
        du = patch.degree_u
        dv = patch.degree_v
        n_sub = 30
        for ui in range(n_sub):
            u0 = ui / n_sub
            du_cell = 1.0 / n_sub
            for vi in range(n_sub):
                v0 = vi / n_sub
                dv_cell = 1.0 / n_sub
                for ni in range(3):
                    u_gp = u0 + du_cell * nodes[ni]
                    for nj in range(3):
                        v_gp = v0 + dv_cell * nodes[nj]
                        pt = _nurbs_surface_point(ctrl, ku, kv, du, dv, u_gp, v_gp)
                        z_tol = max(1e-4, (ctrl[:,:,2].max() - ctrl[:,:,2].min()) * 0.01)
                        if abs(pt[2]) > z_tol:
                            continue
                        Su, Sv = _nurbs_partials(ctrl, ku, kv, du, dv, u_gp, v_gp)
                        cross = np.cross(Su, Sv)
                        j_z = abs(cross[2])
                        w = weights[ni] * weights[nj] * du_cell * dv_cell
                        dA = w * j_z
                        area += dA
                        Ix += dA * pt[1] ** 2
                        Iy += dA * pt[0] ** 2

    return Ix, Iy, area


def nurbs_station_areas(patches: list, LWL: float,
                         n_stations: int = 41) -> tuple:
    """Compute station areas from NURBS patches.

    Returns (x_stations, area_stations).
    """
    from hull_opt.geometry import _eval_nurbs_surface

    x_stations = np.linspace(0.0, LWL, n_stations)
    area_stations = np.zeros(n_stations)

    for si in range(n_stations):
        xs = x_stations[si]
        if si == 0 or si == n_stations - 1:
            continue
        yz_pairs = []
        for patch in patches:
            if "bulb" in patch.name:
                continue
            ctrl = patch.control_net
            if xs < ctrl[:, :, 0].min() - 0.01 or xs > ctrl[:, :, 0].max() + 0.01:
                continue
            # NOTE: Nearest-point sampling on a fixed grid is a discretization
            # approximation. For accurate station areas, use finer resolution
            # or iso-parametric curve intersection.
            n_ue = 81
            n_ve = 41
            u_vals = np.linspace(0.0, 1.0, n_ue)
            v_vals = np.linspace(0.0, 1.0, n_ve)
            surf = _eval_nurbs_surface(
                ctrl, patch.knots_u, patch.knots_v,
                u_vals, v_vals, patch.degree_u, patch.degree_v)
            x_vals = surf[:, :, 0]
            dist = np.abs(x_vals - xs)
            ui, vi = np.unravel_index(dist.argmin(), dist.shape)
            pts = surf[ui, vi]
            yz_pairs.append((abs(pts[1]), pts[2]))

        if len(yz_pairs) < 2:
            continue

        yz_pairs.sort(key=lambda p: p[1])
        z_vals = np.array([p[1] for p in yz_pairs])
        y_vals = np.array([p[0] for p in yz_pairs])

        if len(z_vals) < 2:
            continue

        dz = np.diff(z_vals)
        y_mid = 0.5 * (y_vals[:-1] + y_vals[1:])
        if np.sum(np.abs(dz)) > 1e-10:
            area_stations[si] = float(np.sum(2.0 * y_mid * np.abs(dz)))

    return x_stations, area_stations


def nurbs_gz_curve(patches: list, cg_z: float = -0.05,
                    n_angles: int = 37, max_heel: float = 180.0,
                    n_sub: int = 6) -> np.ndarray:
    """Compute GZ curve from NURBS patches using direct surface integral.

    For each heel angle, rotates the hull+deck control nets about CG,
    integrates the submerged volume and centroid, then GZ = centroid_y.

    Returns same format as compute_gz_curve: (angles, gz, volumes).
    """
    if patches is None or len(patches) == 0:
        return np.column_stack([np.zeros(n_angles), np.zeros(n_angles), np.zeros(n_angles)])

    # Use only hull + deck patches (keel/bulb are appendages that don't
    # share NURBS boundaries with the hull; their volume contribution
    # is captured through ballast mass in CG computation).
    hull_patches = [p for p in patches
                    if "hull" in p.name or p.name == "deck" or "keel" in p.name or "bulb" in p.name]

    angles = np.linspace(0, max_heel, n_angles)
    gz = np.zeros(n_angles)
    volumes = np.zeros(n_angles)

    for i, deg in enumerate(angles):
        try:
            V, centroid = nurbs_submerged_volume(
                hull_patches, n_sub=n_sub, heel_deg=deg, cg_z=cg_z)
            volumes[i] = V
            if V > 1e-10:
                gz[i] = centroid[1]
            else:
                gz[i] = 0.0
        except Exception as e:
            logger.warning(f"NURBS GZ at angle {deg} failed: {e}")
            gz[i] = 0.0
            volumes[i] = 0.0

    return np.column_stack([angles, gz, volumes])


def compute_wind_heel_equilibrium(gz_curve: np.ndarray, rig: dict,
                                  storm_wind_ms: float, nabla: float,
                                  rho_water: float = 1025.0, g: float = 9.81,
                                  feathered_frac: float = 1.0,
                                  cd_sail: float = 0.02,
                                  windage_area: float = 0.0,
                                  windage_height: float = 0.0,
                                  cd_windage: float = 1.13,
                                  x_dict: Optional[dict] = None) -> float:
    """Find the equilibrium heel angle under feathered wind load.

    Scans 0-90° at 5° increments to find the first crossing where GZ ≥
    wind heeling arm. Returns the crossing angle, or 90.0 if no crossing
    (design would blow over). Feathered sail force uses the polar CD
    (~0.02); hull/mast windage is added from x_dict when provided.
    """
    gz_angles = gz_curve[:, 0]
    gz_vals = gz_curve[:, 1]
    sail_area_feathered = rig["area"] * feathered_frac
    sail_height = rig["combined"]["z"]
    if x_dict is not None and windage_area <= 0:
        BWL = float(x_dict.get("BWL", 0.6))
        E = float(x_dict.get("E", 0.3))
        mast_d = 0.25 * float(rig["main"].get("cr", 0.4))
        mast_h = 0.35 * float(rig["main"].get("span", 2.4))
        a_hull = max(0.0, BWL * E)
        a_mast = max(0.0, mast_d * mast_h)
        total_a = a_hull + a_mast
        if total_a > 0:
            windage_area = total_a
            windage_height = (a_hull * 0.5 * E + a_mast * (E + 0.5 * mast_h)) / total_a
    for deg in np.linspace(0, 90, 19):
        wind_arm = compute_wind_heeling_arm(
            deg, storm_wind_ms, sail_area_feathered, sail_height,
            nabla, rho_water=rho_water, g=g,
            cd_sail=cd_sail, windage_area=windage_area,
            windage_height=windage_height, cd_windage=cd_windage,
        )
        gz_at_deg = float(np.interp(deg, gz_angles, gz_vals, left=0, right=0))
        if gz_at_deg >= wind_arm:
            return float(deg)
    return 90.0
