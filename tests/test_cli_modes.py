import os
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_dry_run_cli():
    """Full CLI dry-run: geometry, GZ, Michell, Capytaine RAOs + the storm
    BEM sweep (~6 min). Marked `slow`: skipped by default (`-m "not slow"`),
    run explicitly with `pytest -m slow` to validate the real CLI path."""
    result = subprocess.run(
        [sys.executable, "run_optimization.py", "--dry-run"],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "RAY_DEDUP_LOGS": "0"},
    )
    # rc 0 requires every external-tool check to pass (OpenFOAM, DualSPHysics,
    # ...); on boxes without those installed the dry-run reports errors and
    # exits 1. Either way it must complete and print the summary banner.
    assert result.returncode in (0, 1), f"dry-run crashed:\n{result.stderr[-2000:]}"
    assert "DRY RUN COMPLETE" in result.stdout, result.stdout[-2000:]


@pytest.mark.slow
def test_quick_test_cli():
    """Full quick-test run: 5 LHS + 2 BO iterations + validation gates
    (10+ min). Marked `slow`: skipped by default (`-m "not slow"`)."""
    result = subprocess.run(
        [sys.executable, "run_optimization.py", "--quick-test"],
        capture_output=True, text=True, timeout=1800,
        env={**os.environ, "RAY_DEDUP_LOGS": "0"},
    )
    assert result.returncode == 0, f"quick-test failed:\n{result.stderr[-2000:]}"
    assert "QUICK TEST COMPLETE" in result.stdout


def test_help():
    result = subprocess.run(
        [sys.executable, "run_optimization.py", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "dry-run" in result.stdout
    assert "quick-test" in result.stdout
    assert "validate-only" in result.stdout
    assert "resume" in result.stdout
