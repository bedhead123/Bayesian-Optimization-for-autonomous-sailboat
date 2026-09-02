import logging
import numpy as np
from hull_opt.utils import (
    latin_hypercube_sample,
    scale_lhs_to_bounds,
    knots_to_ms,
    ms_to_knots,
    ensure_dir,
    logsumexp,
)
from pathlib import Path
import tempfile


def test_knots_to_ms():
    assert abs(knots_to_ms(1.0) - 0.514444) < 1e-6
    assert knots_to_ms(0.0) == 0.0


def test_ms_to_knots():
    assert abs(ms_to_knots(0.514444) - 1.0) < 1e-6
    assert ms_to_knots(0.0) == 0.0


def test_roundtrip_knots():
    for k in [0.0, 3.5, 10.0, 25.0]:
        assert abs(ms_to_knots(knots_to_ms(k)) - k) < 1e-6


def test_latin_hypercube_sample_shape():
    samples = latin_hypercube_sample(50, 8)
    assert samples.shape == (50, 8)


def test_latin_hypercube_sample_bounds():
    samples = latin_hypercube_sample(100, 4)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_latin_hypercube_sample_stratification():
    n = 50
    samples = latin_hypercube_sample(n, 2)
    for j in range(2):
        for i in range(n):
            assert 0 <= samples[i, j] <= 1


def test_scale_lhs_to_bounds():
    lhs = np.array([[0.0, 0.5, 1.0],
                    [0.25, 0.0, 0.75]])
    bounds = [(0.0, 10.0), (1.0, 5.0), (-1.0, 1.0)]
    scaled = scale_lhs_to_bounds(lhs, bounds)
    assert np.allclose(scaled[0], [0.0, 3.0, 1.0])
    assert np.allclose(scaled[1], [2.5, 1.0, 0.5])


def test_ensure_dir_creates(tmp_path):
    d = tmp_path / "new" / "nested" / "dir"
    assert not d.exists()
    result = ensure_dir(d)
    assert d.exists()
    assert result == d


def test_ensure_dir_exists(tmp_path):
    result = ensure_dir(tmp_path)
    assert result == tmp_path


def test_logsumexp():
    a = np.array([1.0, 2.0, 3.0])
    # log(sum(exp(a))) = log(exp(1) + exp(2) + exp(3))
    expected = np.log(np.sum(np.exp(a)))
    assert abs(logsumexp(a) - expected) < 1e-10


def test_logsumexp_2d():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = logsumexp(a, axis=1)
    expected = np.array([np.log(np.sum(np.exp([1.0, 2.0]))),
                         np.log(np.sum(np.exp([3.0, 4.0])))])
    assert np.allclose(result, expected)


# ── Phase-0 regression: pipeline logging + rapid summary helpers ──────────

def test_reassert_pipeline_logging_restores_root_after_capytaine():
    import capytaine  # noqa: F401 — real import; clobbers root logging once
    from rich.logging import RichHandler

    root = logging.getLogger()
    if not any(isinstance(h, RichHandler) for h in root.handlers):
        # capytaine was already imported (and its clobber reverted) by an
        # earlier test in this process; simulate the same clobber so the
        # test is order-independent.
        root.handlers = [RichHandler()]
        root.setLevel(logging.WARNING)

    from hull_opt.utils import reassert_pipeline_logging
    reassert_pipeline_logging()
    assert not any(isinstance(h, RichHandler) for h in root.handlers)
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    assert any(isinstance(h, logging.FileHandler) for h in root.handlers)
    assert root.level == logging.INFO


def test_format_rapid_summary_bare_result():
    from hull_opt.rapid_gates import RapidGateResult
    from hull_opt.utils import format_rapid_summary
    line = format_rapid_summary(RapidGateResult())
    assert line.startswith("rapid iter=")
    assert "min_margin=--" in line
    assert "worst=n/a" in line


def test_log_rapid_summary_emits_rapid_line(caplog):
    from hull_opt.rapid_gates import RapidGateResult
    from hull_opt.utils import log_rapid_summary, _rapid_logger_get
    caplog.set_level(logging.INFO)
    lg = _rapid_logger_get()
    lg.addHandler(caplog.handler)
    try:
        log_rapid_summary(7, RapidGateResult())
    finally:
        lg.removeHandler(caplog.handler)
    assert any(
        "[rapid]" in r.getMessage() and "design_id=7" in r.getMessage()
        for r in caplog.records
    )
