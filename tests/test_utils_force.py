"""Unit tests for hull_opt.of_legacy.extract_openfoam_force (window_s support)."""
import numpy as np
import pytest
from pathlib import Path

from hull_opt.of_legacy import extract_openfoam_force

pytestmark = pytest.mark.slow


def _write_trace(tmp_path: Path, rows, header=True):
    p = tmp_path / "force.dat"
    lines = ["# Force", "# Time \ttotal_x total_y total_z\tpressure_x pressure_y pressure_z\tviscous_x viscous_y viscous_z"]
    for t, fx in rows:
        lines.append(f"{t:.8f} {fx:.8e} 0 0 0 0 0 0 0 0")
    p.write_text("\n".join(lines) + "\n")
    return p


def test_tail_frac_mean(tmp_path):
    rows = [(i * 0.1, 100.0) for i in range(201)]
    p = _write_trace(tmp_path, rows)
    val = extract_openfoam_force(p)
    assert val is not None
    assert val == pytest.approx(100.0)


def test_window_s_mean(tmp_path):
    rows = [(i * 0.1, 50.0 + 5.0 * i) for i in range(201)]
    p = _write_trace(tmp_path, rows)
    val = extract_openfoam_force(p, window_s=3.0)
    expected = np.mean([50.0 + 5.0 * i for i in range(170, 201)])
    assert val == pytest.approx(expected, rel=1e-6)


def test_window_s_falls_back_for_short_trace(tmp_path):
    rows = [(i * 0.01, 100.0) for i in range(101)]
    p = _write_trace(tmp_path, rows)
    val = extract_openfoam_force(p, window_s=3.0)
    assert val == pytest.approx(100.0)


def test_archived_design_60_trace():
    p = Path("output/calibration/design_60_iter_60/postProcessing/forces/0/force.dat")
    if not p.exists():
        pytest.skip("archived calibration trace not present")
    win = extract_openfoam_force(p, window_s=3.0)
    tail = extract_openfoam_force(p, tail_frac=0.2)
    assert win is not None and tail is not None
    assert abs(win - tail) / tail < 0.01


def test_missing_file_returns_none(tmp_path):
    assert extract_openfoam_force(tmp_path / "nope.dat") is None


def test_empty_file_returns_none(tmp_path):
    p = tmp_path / "empty.dat"
    p.write_text("# only comments\n")
    assert extract_openfoam_force(p) is None
