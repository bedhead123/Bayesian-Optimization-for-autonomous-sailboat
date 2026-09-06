import numpy as np
import pytest
from hull_opt.config import design_vector_names
from hull_opt.rao_surrogate import RAOSurrogateConfig, RAOSurrogate


@pytest.fixture
def cfg():
    return RAOSurrogateConfig(min_samples=5, holdout_frac=0.2, hidden_dim=16, n_layers=1)


@pytest.fixture
def surrogate(cfg):
    return RAOSurrogate(cfg)


def test_mlp_architecture(surrogate):
    surrogate.train(force=True)
    if surrogate._model is not None:
        n_params = sum(p.numel() for p in surrogate._model.parameters())
        assert n_params > 0


def test_ready_flag_false_before_min_samples(surrogate):
    assert not surrogate.ready


def test_anchor_grid():
    grid = RAOSurrogate.anchor_grid(0.2, 6.0)
    assert len(grid) == 6
    assert np.isclose(grid[0], 0.2)
    assert np.isclose(grid[-1], 6.0)


def test_predict_before_ready_raises(surrogate):
    dv = np.zeros(len(design_vector_names()), dtype=np.float64)
    with pytest.raises(RuntimeError, match="not ready"):
        surrogate.predict(dv, 1.0, 90.0)


def test_add_samples_and_train_with_synthetic():
    cfg = RAOSurrogateConfig(min_samples=10, holdout_frac=0.2, hidden_dim=32, n_layers=2, lr=1e-3)
    surr = RAOSurrogate(cfg)

    n_samples = 50
    bem_data = []
    rng = np.random.default_rng(123)
    for _ in range(n_samples):
        dv = rng.uniform(-1, 1, len(design_vector_names())).tolist()
        omega = float(rng.uniform(0.5, 5.0))
        heading = float(rng.choice([90.0, 135.0, 180.0]))
        heave = 0.5 + 0.1 * dv[0] + 0.05 * omega
        pitch = 0.3 + 0.08 * dv[1] + 0.03 * omega
        roll = 1.0 + 0.2 * dv[2] + 0.1 * heading / 180.0
        bem_data.append({
            "design_vector": dv, "omega": omega, "heading_deg": heading,
            "heave_rao": heave + rng.normal(0, 0.01),
            "pitch_rao": pitch + rng.normal(0, 0.01),
            "roll_rao": roll + rng.normal(0, 0.01),
        })

    surr.add_samples(bem_data)
    assert surr._n_samples == n_samples
    assert not surr.ready

    surr.train(force=True)
    assert surr.ready

    for d in bem_data[:5]:
        dv = np.array(d["design_vector"], dtype=np.float64)
        h, p, r = surr.predict(dv, d["omega"], d["heading_deg"])
        assert np.isfinite(h)
        assert np.isfinite(p)
        assert np.isfinite(r)
        assert abs(h - d["heave_rao"]) < 0.2
        assert abs(p - d["pitch_rao"]) < 0.2
        assert abs(r - d["roll_rao"]) < 0.5


def test_predict_batch_shape(surrogate):
    cfg2 = RAOSurrogateConfig(min_samples=5, holdout_frac=0.2, hidden_dim=16, n_layers=1)
    surr2 = RAOSurrogate(cfg2)
    rng = np.random.default_rng(42)
    bem_data = []
    for _ in range(10):
        dv = rng.uniform(-1, 1, len(design_vector_names())).tolist()
        bem_data.append({
            "design_vector": dv, "omega": float(rng.uniform(0.5, 5.0)),
            "heading_deg": float(rng.choice([90.0, 135.0, 180.0])),
            "heave_rao": float(rng.uniform(0.1, 2.0)),
            "pitch_rao": float(rng.uniform(0.1, 1.0)),
            "roll_rao": float(rng.uniform(0.5, 3.0)),
        })
    surr2.add_samples(bem_data)
    surr2.train(force=True)

    dv = rng.uniform(-1, 1, len(design_vector_names()))
    omegas = np.linspace(0.5, 5.0, 8)
    headings = [90.0, 180.0]
    preds = surr2.predict_batch(dv, omegas, headings)
    assert len(preds["omega"]) == 16
    assert len(preds["heading_deg"]) == 16
    assert len(preds["heave_rao"]) == 16
    assert np.all(np.isfinite(preds["heave_rao"]))


def test_add_samples_rejects_bad_design_vector(surrogate):
    bem_data = [{"design_vector": [0.0] * 5, "omega": 1.0, "heading_deg": 90.0,
                 "heave_rao": 0.5, "pitch_rao": 0.3, "roll_rao": 1.0}]
    surrogate.add_samples(bem_data)
    assert surrogate._n_samples == 0
