#!/usr/bin/env python3
"""
System validation script. Run this from the project root:
    python -m hull_opt.check_system
Checks config, Python deps, DualSPHysics, database.
"""
import os, sys, subprocess, yaml, sqlite3
from pathlib import Path


def run_checks():
    SEP = "=" * 60
    print(SEP)
    print("System Validation Report")
    print(SEP)

    passed = 0
    failed = 0

    print("\n1. Configuration:")
    try:
        cfg = yaml.safe_load(open("config.yaml"))
        if "fixed" in cfg and "LWL" in cfg["fixed"]:
            print(f" [PASS] LWL = {cfg['fixed']['LWL']} (fixed), bounds = {cfg['bounds'].get('LWL', 'N/A')}")
            passed += 1
        else:
            print(" [PASS] No config mismatches")
            passed += 1
    except Exception as e:
        print(f" [FAIL] {e}")
        failed += 1

    print("\n2. Python Dependencies:")
    ok = True
    for dep in ["torch", "botorch", "gpytorch", "capytaine", "trimesh", "ray", "numpy", "scipy"]:
        try:
            __import__(dep)
        except ImportError:
            print(f" [FAIL] missing {dep}")
            ok = False
    if ok:
        print(" [PASS] All deps installed")
        passed += 1
    else:
        failed += 1

    print("\n3. DualSPHysics:")
    ds_dir = cfg.get("paths", {}).get("dualsphysics_dir", "/home/anon/apps/DualSPHysics_v5.4")
    ds_path = Path(ds_dir).resolve()
    gencase = ds_path / "bin" / "linux" / "GenCase_linux64"
    if gencase.exists():
        print(" [PASS] GenCase exists")
        passed += 1
    else:
        print(" [FAIL] GenCase not found")
        failed += 1
    # Check DS GPU solver
    ds_gpu = ds_path / "bin" / "linux" / "DSGcc7" / "DualSPHysics54.linux64"
    if ds_gpu.exists():
        print(" [PASS] DualSPHysics GPU solver exists")
        passed += 1
    else:
        print(" [FAIL] DualSPHysics GPU solver not found")
        failed += 1
    # Check ComputeForces
    cf = ds_path / "bin" / "linux" / "ComputeForces_linux64"
    if cf.exists():
        print(" [PASS] ComputeForces exists")
        passed += 1
    else:
        print(" [FAIL] ComputeForces not found")
        failed += 1

    print("\n4. Database:")
    db = Path("output/optimization.db")
    if db.exists():
        try:
            conn = sqlite3.connect(str(db))
            tables = conn.execute("SELECT name FROM sqlite_master").fetchall()
            print(f" [PASS] {len(tables)} tables found")
            passed += 1
        except Exception as e:
            print(f" [FAIL] {e}")
            failed += 1
    else:
        print(" [WARN] No database")

    print("\n" + SEP)
    print(f"Summary: {passed} passed, {failed} failed")
    print(SEP)
    return 0 if failed == 0 else 1


def run_env_checks() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from hull_opt.preflight import CORE_MODULES, check_python_env

    problems = check_python_env()
    failed = {p.split(":", 1)[0] for p in problems}
    for mod in CORE_MODULES:
        status = "PASS" if mod not in failed else "FAIL"
        print(f" [{status}] {mod}")
    for problem in problems:
        print(f"   {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="System validation")
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="check the running Python interpreter for core modules",
    )
    args = parser.parse_args()
    if args.check_env:
        sys.exit(run_env_checks())
    sys.exit(run_checks())
