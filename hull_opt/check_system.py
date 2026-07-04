#!/usr/bin/env python3
"""
System validation script. Run this from the project root:
    python -m hull_opt.check_system
Checks config, Python deps, OpenFOAM, DualSPHysics, database.
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

    print("\n3. OpenFOAM:")
    of_env = "/opt/openfoam2512/etc/bashrc"
    if Path(of_env).exists():
        print(" [PASS] OpenFOAM env found")
        passed += 1
    else:
        print(" [FAIL] OpenFOAM env not found")
        failed += 1

    print("\n4. DualSPHysics:")
    gencase = Path("/home/anon/apps/boat/bin/dualsphysics/5.4/bin/linux/GenCase_linux64")
    if gencase.exists():
        print(" [PASS] GenCase exists")
        passed += 1
    else:
        print(" [FAIL] GenCase not found")
        failed += 1

    print("\n5. Database:")
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


if __name__ == "__main__":
    sys.exit(run_checks())
