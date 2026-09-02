"""SPH pre-flight checks for calibration/validation cases.

Checks solver binary exists, STL exists and loads, and basic case directory structure.
Runs in seconds, before the multi-hour SPH run.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh

CORE_MODULES = (
    "fast_simplification",
    "trimesh",
    "capytaine",
    "torch",
    "meshio",
    "scipy",
    "numpy",
)


def check_python_env(config=None) -> List[str]:
    """Return error strings for core modules that fail to import under the
    actual running interpreter; empty list means all pass."""
    import importlib

    pip = Path(__file__).resolve().parent.parent / "venv" / "bin" / "pip"
    problems: List[str] = []

    def check(cond: bool, label: str) -> None:
        if not cond:
            problems.append(label)

    for mod in CORE_MODULES:
        try:
            importlib.import_module(mod)
            check(True, f"{mod} imported cleanly")
        except Exception as e:
            check(False, f"{mod}: {type(e).__name__}: {e} (fix: {pip} install {mod})")
    return problems


def preflight_case(case_dir, config) -> Tuple[bool, List[str]]:
    """Return (ok, message lines). Checks the rendered case before the SPH run."""
    case_dir = Path(case_dir)
    msgs: List[str] = []

    def check(cond: bool, label: str) -> None:
        msgs.append(("PASS " if cond else "FAIL ") + label)

    def warn(label: str) -> None:
        msgs.append("WARN " + label)

    # Check solver binary exists
    try:
        from hull_opt.sph_gates import find_solver
        find_solver(getattr(config.paths, "dualsphysics_dir", "/home/anon/apps/DualSPHysics_v5.4"))
        check(True, "DualSPHysics solver binary found")
    except FileNotFoundError as e:
        check(False, f"DualSPHysics solver binary: {e}")
    except Exception as e:
        check(False, f"Solver check failed: {e}")

    # Check GenCase tool exists
    from hull_opt.sph_gates import _find_tool
    ds_dir = getattr(config.paths, "dualsphysics_dir", "/home/anon/apps/DualSPHysics_v5.4")
    gencase = _find_tool("GenCase_linux64", ds_dir)
    check(bool(gencase) and Path(gencase).exists(), f"GenCase_linux64 at {gencase}")

    # Check for XML case file
    xml_files = sorted(case_dir.glob("case_*.xml"))
    check(len(xml_files) > 0, f"Case XML file found ({xml_files[0].name if xml_files else 'none'})")

    # Check STL exists and loads
    import re
    for xml_file in xml_files:
        text = xml_file.read_text(errors="replace")
        stl_paths = re.findall(r'file\s*=\s*"([^"]*\.stl[^"]*)"', text, re.IGNORECASE)
        for stl_path_str in stl_paths:
            stl_path = Path(stl_path_str)
            if not stl_path.is_absolute():
                stl_path = (xml_file.parent / stl_path).resolve()
            if stl_path.exists():
                try:
                    mesh = trimesh.load(str(stl_path))
                    if isinstance(mesh, trimesh.Scene):
                        mesh = mesh.dump(concatenate=True)
                    if not mesh.is_watertight:
                        # DualSPHysics does not require a watertight mesh
                        # (hull_full.stl has open keel/bulb patches by design),
                        # so this is a WARN.
                        warn(f"STL {stl_path.name} not watertight (expected for hull_full.stl)")
                    else:
                        check(True, f"STL {stl_path.name} watertight")
                    check(len(mesh.faces) > 0, f"STL {stl_path.name} has {len(mesh.faces)} faces")
                    ext = mesh.bounds[1] - mesh.bounds[0]
                    check(bool(np.all(np.isfinite(ext))) and float(np.max(ext)) > 0,
                          f"STL {stl_path.name} bounds finite and non-degenerate")
                except Exception as e:
                    check(False, f"STL {stl_path.name} load failed: {e}")
            else:
                check(False, f"STL path exists: {stl_path}")
        if not stl_paths:
            check(False, f"No STL file= reference in {xml_file.name}")

    ok = all(m.startswith(("PASS", "WARN")) for m in msgs)
    return ok, msgs
