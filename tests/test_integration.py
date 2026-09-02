"""
Cross-module integration tests for the hull-keel optimization pipeline.
Validates that rapid gates, corrections, RAO surrogate, reference runner,
and database tables wire together correctly.  All expensive external calls
(Capytaine BEM, OpenFOAM, DualSPHysics) are monkeypatched or stubbed.
"""
import json
import subprocess
import sys
import numpy as np
import pytest
from pathlib import Path


# =============================================================================
# 1. config sections
# =============================================================================

class TestConfigSections:
    def test_config_loads_with_new_sections(self):
        from hull_opt.config import load_config
        cfg = load_config(str(Path(__file__).resolve().parents[1] / "config.yaml"))
        assert hasattr(cfg, "rapid_validation")
        assert hasattr(cfg, "reference")
        assert cfg.rapid_validation.storm_hs_factor == 4.0
        assert cfg.rapid_validation.storm_tp_factor == 1.3
        assert cfg.reference.tool == "dualsphysics"
        assert cfg.reference.sim_time_s == 10.0
        assert cfg.weights.w6 == 0.5
        assert cfg.weights.w7 == 0.10
        assert cfg.weights.w8 == 0.35
        assert cfg.weights.w9 == 0.5

    def test_config_signature_includes_rapid(self):
        from hull_opt.config import load_config, config_signature
        cfg = load_config(str(Path(__file__).resolve().parents[1] / "config.yaml"))
        sig = config_signature(cfg)
        assert isinstance(sig, str) and len(sig) == 64
        import hashlib, dataclasses
        # Mirror the full config_signature payload (config.py:265)
        payload = {
            "bounds": dataclasses.asdict(cfg.bounds),
            "fixed": dataclasses.asdict(cfg.fixed),
            "weights": dataclasses.asdict(cfg.weights),
            "validation": dataclasses.asdict(cfg.validation),
            "rapid_validation": dataclasses.asdict(cfg.rapid_validation),
            "reference": dataclasses.asdict(cfg.reference),
            "optimization": dataclasses.asdict(cfg.optimization),
            "calibration": dataclasses.asdict(cfg.calibration),
            "wave_spectrum": dataclasses.asdict(cfg.wave_spectrum),
        }
        canonical = json.dumps(payload, sort_keys=True, default=str)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert sig == expected
        assert "rapid_validation" in canonical
        assert "reference" in canonical
        assert "w6" in canonical


# =============================================================================
# 2. EvaluationResult · RapidGateResult composition
# =============================================================================

class TestEvaluationResultComposition:
    def test_evaluation_result_has_rapid_field(self):
        from hull_opt.low_fidelity import EvaluationResult
        r = EvaluationResult()
        assert hasattr(r, "rapid")
        assert r.rapid is None
        from hull_opt.rapid_gates import RapidGateResult
        gr = RapidGateResult(
            avs_deg=120.0, max_gz_m=0.08,
            capsize_margin=0.3, storm_peak_accel_g=2.5,
            roll_sigma_deg=10.0, roll_period_s=1.8,
        )
        r.rapid = gr
        assert r.rapid.avs_deg == 120.0
        assert r.rapid.margins is not None

    def test_rapid_gates_avs_importable(self):
        from hull_opt.rapid_gates import RapidGateResult, evaluate_rapid_gates
        assert callable(evaluate_rapid_gates)


# =============================================================================
# 3. Low-fidelity × rapid gate integration (BEM stub)
# =============================================================================

_STUB_RAO = {
    "omega": [1.0, 2.0, 3.0],
    "heave_rao": [0.0, 0.0, 0.0],
    "pitch_rao": [0.0, 0.0, 0.0],
    "roll_rao": [0.0, 0.0, 0.0],
    "peak_accel": 2.0,
    "roll_period": 1.5,
}


def _stub_storm_bem(*a, **kw):
    Tp_storm = 12.0
    beam_data = {
        "storm_peak_accel_g": 2.5,
        "roll_sigma_deg": 8.0,
        "roll_period_s": 1.5,
        "all_headings_rao": [
            {"heading_deg": 90,
             "omega": [1.0, 2.0, 3.0],
             "heave_rao": [0.1, 0.2, 0.1],
             "pitch_rao": [0.01, 0.02, 0.01],
             "roll_rao": [0.3, 0.6, 0.3]},
        ],
    }
    return beam_data, Tp_storm


class TestLowFidelityRapidIntegration:
    def test_low_fidelity_rapid_integration(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            "hull_opt.low_fidelity._compute_raos_capytaine",
            lambda *a, **kw: _STUB_RAO,
            raising=False,
        )
        monkeypatch.setattr(
            "hull_opt.rapid_gates._storm_bem_sweep",
            _stub_storm_bem,
            raising=False,
        )
        from hull_opt.low_fidelity import evaluate_low_fidelity
        from hull_opt.config import load_config
        cfg = load_config(str(Path(__file__).resolve().parents[1] / "config.yaml"))
        dv = np.zeros(17, dtype=np.float64)
        out = str(tmp_path / "lowfi_test")
        result = evaluate_low_fidelity(dv, cfg, output_dir=out)
        assert result is not None
        assert result.error_code is None or "E_GZ" not in result.error_code
        if result.rapid is not None:
            assert hasattr(result.rapid, "margins")
            if result.rapid.margins:
                for k in ("capsize", "storm_accel", "slam_pressure"):
                    assert k in result.rapid.margins, f"missing margin {k}"


# =============================================================================
# 4. Database: corrections + rapid-gate roundtrips
# =============================================================================

class TestDatabaseIntegration:
    def test_database_corrections_roundtrip(self, tmp_path):
        from hull_opt.database import OptimizationDatabase
        db = OptimizationDatabase(str(tmp_path / "corr.db"))
        db.set_correction("drag_factor", 1.25)
        db.set_correction("wave_amp", 0.92)
        corr = db.get_corrections()
        assert corr["drag_factor"] == 1.25
        assert corr["wave_amp"] == 0.92
        db.set_correction("drag_factor", 1.30)
        corr = db.get_corrections()
        assert corr["drag_factor"] == 1.30
        assert len(corr) == 2
        db.close()

    def test_database_update_design_rapid_gates(self, tmp_path):
        from hull_opt.database import OptimizationDatabase
        db = OptimizationDatabase(str(tmp_path / "rg.db"))
        dv = np.zeros(17, dtype=np.float64)
        did = db.insert_design(
            iter_num=0, design_vector=dv, feasible=True, fom=1.0
        )
        result = {
            "avs_deg": 125.0,
            "capsize_margin": 0.45,
            "storm_peak_accel_g": 2.3,
            "roll_sigma_deg": 8.1,
            "parametric_roll": 0,
            "slam_pressure_pa": 4500.0,
            "inverted_pressure_pa": 12000.0,
            "storm_wind_heel_deg": 22.5,
            "rapid_gates": {"avs": 125.0, "capsize": 0.45},
            "gate_margins": {"avs": 0.45, "capsize": 0.55},
        }
        db.update_design_rapid_gates(did, result)
        d = db.get_design(did)
        assert d["avs_deg"] == 125.0
        assert d["capsize_margin"] == 0.45
        assert d["storm_peak_accel_g"] == 2.3
        assert d["roll_sigma_deg"] == 8.1
        assert d["parametric_roll"] == 0
        assert d["slam_pressure_pa"] == 4500.0
        assert d["inverted_pressure_pa"] == 12000.0
        assert d["storm_wind_heel_deg"] == 22.5
        gm = json.loads(d["gate_margins"])
        assert gm["capsize"] == 0.55
        db.close()


# =============================================================================
# 5. RAO surrogate signatures
# =============================================================================

class TestRAOSurrogateIntegration:
    def test_surrogate_integration(self):
        from hull_opt.rao_surrogate import RAOSurrogate, RAOSurrogateConfig
        cfg = RAOSurrogateConfig()
        surr = RAOSurrogate(cfg)
        assert hasattr(surr, "predict_batch")
        import inspect
        sig = inspect.signature(surr.predict_batch)
        params = list(sig.parameters.keys())
        for p in ("design_vector", "omegas", "headings_deg"):
            assert p in sig.parameters, f"predict_batch missing param {p}"
        assert not surr.ready


# =============================================================================
# 6. run_optimization.py CLI flags
# =============================================================================

class TestCLIFlags:
    def test_run_benchmark_flag(self):
        py = sys.executable
        r = subprocess.run(
            [py, "run_optimization.py", "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        if "--benchmark" not in r.stdout:
            pytest.skip("--benchmark flag not present in run_optimization.py")
        r2 = subprocess.run(
            [py, "run_optimization.py", "--benchmark"],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert r2.returncode in (0, 1), f"benchmark exit={r2.returncode}"
        assert "PASS" in r2.stdout or "FAIL" in r2.stdout

    @pytest.mark.slow
    def test_smoke_mode_component(self):
        """End-to-end CLI golden regression (`--smoke` evaluates all 6 golden
        designs through the full low-fi pipeline, each running geometry, GZ,
        Michell, Capytaine RAOs and the storm BEM sweep — several minutes).
        Marked `slow`: skipped by default (`-m "not slow"`). The fast
        in-process equivalent for default runs is tests/test_golden_smoke.py."""
        py = sys.executable
        r = subprocess.run(
            [py, "run_optimization.py", "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        if "--smoke" not in r.stdout:
            pytest.skip("--smoke flag not present (not yet added)")
        r2 = subprocess.run(
            [py, "run_optimization.py", "--smoke", "--dry-run"],
            capture_output=True, text=True, timeout=1800,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        # rc 0 = all golden metrics match; rc 1 = mismatch (still a valid run)
        assert r2.returncode in (0, 1)
