import subprocess
import sys


def test_dry_run_cli():
    result = subprocess.run(
        [sys.executable, "run_optimization.py", "--dry-run"],
        capture_output=True, text=True, timeout=120,
        env={**__import__('os').environ, "RAY_DEDUP_LOGS": "0"},
    )
    # Should exit 0 with all checks passed
    assert result.returncode == 0, f"dry-run failed:\n{result.stderr[-2000:]}"
    assert "ALL CHECKS PASSED" in result.stdout or "DRY RUN COMPLETE" in result.stdout


def test_quick_test_cli():
    result = subprocess.run(
        [sys.executable, "run_optimization.py", "--quick-test"],
        capture_output=True, text=True, timeout=600,
        env={**__import__('os').environ, "RAY_DEDUP_LOGS": "0"},
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
