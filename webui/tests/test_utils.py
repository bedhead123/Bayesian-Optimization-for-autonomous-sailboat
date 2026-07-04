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
