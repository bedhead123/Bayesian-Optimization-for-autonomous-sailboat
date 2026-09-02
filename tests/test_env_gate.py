import subprocess
import sys
from pathlib import Path

from hull_opt.preflight import check_python_env

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_run_opt_sh_uses_venv_python():
    text = (REPO_ROOT / "run_opt.sh").read_text()
    assert "/venv/bin/python" in text
    assert "python3 run_optimization.py" not in text


def test_check_python_env_clean_under_venv():
    assert check_python_env() == []


def test_check_system_env_gate_exits_zero():
    result = subprocess.run(
        [sys.executable, "hull_opt/check_system.py", "--check-env"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS]" in result.stdout
