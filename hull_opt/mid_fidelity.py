"""
Mid-fidelity calibration using OpenFOAM RANS.
Generates geometry, writes OpenFOAM case, runs blockMesh/snappyHexMesh/interFoam,
extracts forces, and computes calibration delta between low-fi and CFD.
Key exports: run_mid_fidelity_calibration()
"""
import subprocess
import numpy as np
from pathlib import Path
import shutil
import logging
import trimesh

logger = logging.getLogger(__name__)

from hull_opt.geometry import generate_hull
from hull_opt.templates.openfoam import write_openfoam_case
from hull_opt.utils import (
    run_of_command, find_force_file, extract_openfoam_force,
    knots_to_ms, ensure_dir, run_parallel_inter_foam
)


def run_mid_fidelity_calibration(design_vector: np.ndarray,
                                  design_id: int, iteration: int,
                                  config) -> float | None:
    logger.info(f"Mid-fidelity calibration: design {design_id}, iteration {iteration}")

    try:
        output_root = ensure_dir(Path(config.paths.output_dir) / "calibration")
        case_dir = output_root / f"design_{design_id}_iter_{iteration}"
        if case_dir.exists():
            shutil.rmtree(case_dir)

        from hull_opt.param_layer import design_vector_to_physical
        x_dict = design_vector_to_physical(design_vector, config)
        hull_lwl = float(x_dict.get("LWL", config.fixed.LWL))
        B = x_dict["BWL"]
        T_hull = x_dict["T_canoe"]
        speed_knots = config.fixed.target_speed_knots
        speed_ms = knots_to_ms(speed_knots)
        max_cells = config.calibration.coarse_cells
        timeout = config.calibration.timeout
        of_env = config.paths.openfoam_env
        rho = config.fixed.rho_water
        nu = config.fixed.nu_water
        g = config.fixed.gravity

        # generate geometry
        stl_path, sac_path, hydro, hull_stl = generate_hull(
            design_vector,
            output_dir=str(case_dir),
            LWL=hull_lwl,
            target_displacement=config.fixed.target_displacement,
            config=config,
        )
        logger.info(f"Geometry generated: {stl_path}")

        # write OF case
        write_openfoam_case(
            case_dir=case_dir,
            stl_path=stl_path,
            speed_ms=speed_ms,
            LWL=hull_lwl, B=B, T=T_hull,
            rho=rho, nu=nu, gravity=g,
            mesh_levels=(2, 3),
            n_layers=3,
            solver="interFoam",
            end_time=5.0,
            delta_t=0.0005,
            write_interval=0.1,
            max_cells=max_cells,
        )
        logger.info(f"OpenFOAM case written to {case_dir}")

        # run blockMesh
        logger.info("Running blockMesh...")
        proc = run_of_command(
            ["blockMesh", "-case", str(case_dir)],
            case_dir, of_env, timeout=300
        )
        if proc.returncode != 0:
            logger.error(f"blockMesh failed: {proc.stderr[:500]}")
            return None
        logger.info("blockMesh OK")

        # run snappyHexMesh
        logger.info("Running snappyHexMesh...")
        proc = run_of_command(
            ["snappyHexMesh", "-case", str(case_dir), "-overwrite"],
            case_dir, of_env, timeout=timeout
        )
        if proc.returncode != 0:
            logger.error(f"snappyHexMesh failed: {proc.stderr[:500]}")
            return None
        logger.info("snappyHexMesh OK")

        # set water level via setFields
        _init_water_and_run_setfields(case_dir, T_hull, of_env)

        # run interFoam (parallel with decomposePar/reconstructPar when n_procs > 1)
        n_procs = getattr(config.calibration, 'n_procs', 1)
        logger.info(f"Running interFoam ({'parallel on ' + str(n_procs) + ' cores' if n_procs > 1 else 'serial'})...")
        proc = run_parallel_inter_foam(
            case_dir, of_env, n_procs=n_procs, timeout=timeout
        )
        if proc.returncode != 0:
            logger.warning(f"interFoam exited non-zero (rc={proc.returncode}):\n{proc.stdout[:500]}")

        # extract forces
        forces_dir = case_dir / "postProcessing" / "forces"
        force_file = find_force_file(forces_dir)
        if force_file is None:
            logger.error(f"Force file not found in {forces_dir} — check reconstructPar and OpenFOAM version compatibility")
            return None
        rt_cfd = extract_openfoam_force(force_file)

        if rt_cfd is None:
            logger.error(f"Could not parse force data from {force_file} — check OpenFOAM output format (v2512+ uses tabular layout)")
            return None

        logger.info(f"Rt_CFD = {rt_cfd:.4f} N")

        # Compute low-fidelity prediction for this design to get delta
        try:
            from hull_opt.geometry import compute_half_breadth_analytic
            from hull_opt.michell import compute_wave_resistance_michell
            from hull_opt.friction import compute_total_resistance

            sac_scale = hydro.get("sac_scale_factor", 1.0)
            hb_func = lambda xq, zq: compute_half_breadth_analytic(xq, zq, x_dict, hull_lwl, sac_scale=sac_scale)
            Rw = compute_wave_resistance_michell(hb_func, hull_lwl, B, T_hull, speed_ms, rho, g)
            hull_mesh_obj = trimesh.load(hull_stl)
            if isinstance(hull_mesh_obj, trimesh.Scene):
                hull_mesh_obj = hull_mesh_obj.dump(concatenate=True)
            # Slice at waterline (z=0) to get underwater wetted area
            try:
                underwater = trimesh.intersections.slice_mesh_plane(
                    hull_mesh_obj, [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], cap=False
                )
                if underwater is not None and hasattr(underwater, 'area') and underwater.area > 0:
                    area = underwater.area
                else:
                    area = hull_mesh_obj.area
            except Exception:
                area = hull_mesh_obj.area if hasattr(hull_mesh_obj, 'area') else 1.0
            Rt_lowfi, Rf, Rw_out = compute_total_resistance(speed_ms, area, hull_lwl, rho, nu, wave_resistance=Rw)
            logger.info(f"Low-fi components: Rw={Rw_out:.4f} N, Rf={Rf:.4f} N, Rt={Rt_lowfi:.4f} N, wetted_area={area:.6f} m²")
            delta = float(rt_cfd - Rt_lowfi)
            logger.info(f"Calibration delta: {delta:.4f} N (CFD={rt_cfd:.4f}, low-fi={Rt_lowfi:.4f})")
            return delta
        except Exception as e:
            logger.warning(f"Low-fi prediction for delta failed, returning None: {e}")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"OpenFOAM timed out after {timeout}s")
        return None
    except Exception as e:
        logger.error(f"Mid-fidelity calibration failed: {e}")
        return None


def _init_water_and_run_setfields(case_dir: Path, T_hull: float, of_env: str):
    """Set initial water level via setFields."""
    set_fields_dict = case_dir / "system" / "setFieldsDict"
    set_fields_dict.parent.mkdir(parents=True, exist_ok=True)
    set_fields_dict.write_text(f"""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2512                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      setFieldsDict;
}}

defaultFieldValues
(
    volScalarFieldValue alpha.water 0
);

regions
(
    boxToCell
    {{
        box ( -1000 -1000 -1000 ) ( 1000 1000 0 );
        fieldValues
        (
            volScalarFieldValue alpha.water 1
        );
    }}
);
""")
    from hull_opt.utils import run_of_command
    logger.info("Running setFields for water level initialization...")
    proc = run_of_command(
        ["setFields", "-case", str(case_dir)],
        case_dir, of_env, timeout=60
    )
    if proc.returncode != 0:
        logger.warning(f"setFields warning: {proc.stderr[:200]}")
