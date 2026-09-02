import numpy as np
import pytest
from hull_opt.corrections import (
    DEFAULT_CORRECTIONS,
    apply_corrections,
    update_correction,
    CorrectionMLP,
    TORCH_AVAILABLE,
)


class TestDefaultCorrections:
    def test_identity(self):
        margins = {"storm_accel": 0.5, "slam_pressure": 0.3, "capsize": 0.8}
        result = apply_corrections(margins)
        for k in margins:
            assert result[k] == margins[k]

    def test_unknown_key_preserved(self):
        margins = {"avs": 0.9, "storm_accel": 0.5}
        result = apply_corrections(margins)
        assert result["avs"] == 0.9
        assert result["storm_accel"] == 0.5


class TestCorrectionFactors:
    def test_scale_by_two(self):
        margins = {"storm_accel": 0.5}
        corrections = {"storm_accel": 2.0}
        result = apply_corrections(margins, corrections)
        assert result["storm_accel"] == 1.0

    def test_partial_corrections(self):
        margins = {"storm_accel": 0.5, "slam_pressure": 0.3}
        corrections = {"storm_accel": 0.5}
        result = apply_corrections(margins, corrections)
        assert result["storm_accel"] == 0.25
        assert result["slam_pressure"] == 0.3

    def test_zero_factor(self):
        margins = {"capsize": 0.8}
        corrections = {"capsize": 0.0}
        result = apply_corrections(margins, corrections)
        assert result["capsize"] == 0.0


class TestNanHandling:
    def test_none_value(self):
        result = apply_corrections({"storm_accel": None})
        assert result["storm_accel"] == -1.0

    def test_nan_value(self):
        result = apply_corrections({"storm_accel": float("nan")})
        assert result["storm_accel"] == -1.0

    def test_nan_with_factor(self):
        result = apply_corrections({"storm_accel": float("nan")}, {"storm_accel": 2.0})
        assert result["storm_accel"] == -1.0

    def test_inf_value(self):
        result = apply_corrections({"storm_accel": float("inf")})
        assert result["storm_accel"] == -1.0


class TestUpdateSmoothing:
    def test_smoothing(self):
        old = 1.0
        observed = 2.0
        new, changed = update_correction("storm_accel", old, observed, alpha=0.3)
        expected = 0.3 * 2.0 + 0.7 * 1.0
        assert abs(new - expected) < 1e-6
        assert changed

    def test_no_change(self):
        old = 1.0
        observed = 1.0
        new, changed = update_correction("storm_accel", old, observed)
        assert abs(new - 1.0) < 1e-6
        assert not changed


class TestUpdateClip:
    def test_clip_exceeded(self):
        old = 1.0
        observed = 10.0
        new, changed = update_correction("storm_accel", old, observed, alpha=0.3, clip=0.5)
        assert abs(new - 1.5) < 1e-6
        assert changed

    def test_clip_negative(self):
        old = 1.0
        observed = -5.0
        new, changed = update_correction("storm_accel", old, observed, alpha=0.3, clip=0.5)
        assert new == old
        assert not changed

    def test_clip_not_exceeded(self):
        old = 1.0
        observed = 1.3
        new, changed = update_correction("storm_accel", old, observed, alpha=0.3, clip=0.5)
        expected = 0.3 * 1.3 + 0.7 * 1.0
        assert abs(new - expected) < 1e-6
        assert changed


class TestUpdateEdgeCases:
    def test_zero_old_val(self):
        new, changed = update_correction("storm_accel", 0.0, 2.0)
        assert new == pytest.approx(0.3 * 2.0 + 0.7 * 1.0)
        assert changed

    def test_negative_old_val(self):
        new, changed = update_correction("storm_accel", -1.0, 2.0)
        assert new == pytest.approx(0.3 * 2.0 + 0.7 * 1.0)
        assert changed

    def test_nan_old_val(self):
        new, changed = update_correction("storm_accel", float("nan"), 2.0)
        assert new == pytest.approx(0.3 * 2.0 + 0.7 * 1.0)
        assert changed

    def test_nan_ratio(self):
        old = 1.0
        new, changed = update_correction("storm_accel", old, float("nan"))
        assert new == old
        assert not changed

    def test_zero_ratio(self):
        old = 1.0
        new, changed = update_correction("storm_accel", old, 0.0)
        assert new == old
        assert not changed

    def test_negative_ratio(self):
        old = 1.0
        new, changed = update_correction("storm_accel", old, -1.0)
        assert new == old
        assert not changed

    def test_inf_old_val(self):
        new, changed = update_correction("storm_accel", float("inf"), 2.0)
        assert np.isfinite(new)
        assert changed


class TestTorchAvailability:
    def test_is_bool(self):
        assert isinstance(TORCH_AVAILABLE, bool)


class TestMLPPredictBeforeTraining:
    def test_returns_defaults(self):
        mlp = CorrectionMLP()
        result = mlp.predict(np.zeros(17, dtype=np.float64))
        assert result == DEFAULT_CORRECTIONS

    def test_key_set_matches_defaults(self):
        mlp = CorrectionMLP()
        result = mlp.predict(np.zeros(17, dtype=np.float64))
        assert set(result.keys()) == set(DEFAULT_CORRECTIONS.keys())

    def test_values_are_before_training(self):
        mlp = CorrectionMLP()
        result = mlp.predict(np.zeros(17, dtype=np.float64))
        for k in DEFAULT_CORRECTIONS:
            assert result[k] == 1.0


class TestRapidGatesCorrectionsApplied:
    def test_corrections_applied_when_import_succeeds(self, monkeypatch):
        import hull_opt.rapid_gates as rg
        from hull_opt.corrections import apply_corrections as real_apply
        captured = []
        def mock_apply(margins, corrections=None):
            captured.append(1)
            return {k: v * 2.0 for k, v in margins.items()}
        monkeypatch.setattr("hull_opt.corrections.apply_corrections", mock_apply)
        from hull_opt.config import load_config
        config = load_config("config.yaml")
        result = rg.RapidGateResult()
        result.avs_deg = 120.0
        result.capsize_margin = 0.3
        result.storm_peak_accel_g = 10.0
        result.roll_sigma_deg = 20.0
        result.slam_pressure_pa = 50000.0
        result.inverted_pressure_pa = 30000.0
        result.storm_wind_heel_deg = 15.0
        result.parametric_roll_risk = False
        from hull_opt.rapid_gates import _build_margins
        _build_margins(result, config)
        assert len(captured) == 1
        assert result.margins["storm_accel"] > 0

    def test_works_when_import_fails(self, monkeypatch):
        import sys
        import hull_opt.rapid_gates as rg
        import builtins
        original_import = builtins.__import__
        def failing_import(name, *args, **kwargs):
            if name == "hull_opt.corrections":
                raise ImportError("corrections not available")
            return original_import(name, *args, **kwargs)
        if "hull_opt.corrections" in sys.modules:
            monkeypatch.delitem(sys.modules, "hull_opt.corrections")
        monkeypatch.setattr("builtins.__import__", failing_import)
        from hull_opt.config import load_config
        config = load_config("config.yaml")
        result = rg.RapidGateResult()
        result.avs_deg = 120.0
        result.capsize_margin = 0.3
        result.storm_peak_accel_g = 10.0
        result.roll_sigma_deg = 20.0
        result.slam_pressure_pa = 50000.0
        result.inverted_pressure_pa = 30000.0
        result.storm_wind_heel_deg = 15.0
        result.parametric_roll_risk = False
        from hull_opt.rapid_gates import _build_margins
        _build_margins(result, config)
        assert result.margins["storm_accel"] > 0


class TestBuildMarginsThresholdWiring:
    """Phase-0 regression: _build_margins must read every threshold from
    config.rapid_validation (not hardcoded values). Patching a threshold via
    dataclasses.replace must shift the corresponding margin."""

    @pytest.fixture
    def config(self):
        from hull_opt.config import load_config
        return load_config("config.yaml")

    @pytest.fixture
    def result(self):
        from hull_opt.rapid_gates import RapidGateResult
        r = RapidGateResult()
        r.avs_deg = 100.0
        r.roll_sigma_deg = 30.0
        r.storm_wind_heel_deg = 70.0
        return r

    @staticmethod
    def _patched(config, **rapid_fields):
        import dataclasses
        rv = dataclasses.replace(config.rapid_validation, **rapid_fields)
        return dataclasses.replace(config, rapid_validation=rv)

    @pytest.mark.parametrize("field,new_value,margin_key,expected", [
        ("min_avs_deg", 50.0, "avs", 100.0 / 50.0),
        ("min_avs_deg", 100.0, "avs", 1.0),
        ("roll_sigma_max_deg", 40.0, "roll_sigma", (40.0 - 30.0) / 40.0),
        ("roll_sigma_max_deg", 30.0, "roll_sigma", 0.0),
        ("wind_heel_max_frac", 0.75, "wind_heel",
         (0.75 * 100.0 - 70.0) / (0.75 * 100.0)),
        ("wind_heel_max_frac", 0.90, "wind_heel",
         (0.90 * 100.0 - 70.0) / (0.90 * 100.0)),
    ])
    def test_margin_tracks_patched_threshold(self, config, result, field,
                                             new_value, margin_key, expected):
        from hull_opt.rapid_gates import _build_margins
        cfg = self._patched(config, **{field: new_value})
        _build_margins(result, cfg)
        assert result.margins[margin_key] == pytest.approx(expected)
