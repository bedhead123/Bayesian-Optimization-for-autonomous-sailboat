"""
RAO surrogate: Torch MLP predicting (heave|pitch|roll) RAO magnitude from
17-dim design vector + omega + heading_deg. Used to reduce storm BEM cost
by evaluating at anchor frequencies only and interpolating rest via MLP.
Key exports: RAOSurrogateConfig, RAOSurrogate
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def _default_input_dim() -> int:
    """Design-vector dim + (omega, heading_deg). Derived so it tracks
    design_vector_names() and cannot drift when the vector changes."""
    from hull_opt.config import design_vector_names
    return len(design_vector_names()) + 2


@dataclass
class RAOSurrogateConfig:
    input_dim: int = field(default_factory=_default_input_dim)
    hidden_dim: int = 64
    n_layers: int = 2
    lr: float = 1e-3
    min_samples: int = 60
    holdout_frac: float = 0.1
    error_threshold: float = 0.10
    anchor_n_freq: int = 6
    device: str = "cpu"


class RAOSurrogate:
    def __init__(self, config: Optional[RAOSurrogateConfig] = None):
        self.config = config or RAOSurrogateConfig()
        self._model = None
        self._n_samples = 0
        self._trained = False
        self._data = []
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None
        self._device = self._resolve_device()

    def _resolve_device(self) -> str:
        if self.config.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def ready(self) -> bool:
        return self._trained and self._n_samples >= self.config.min_samples

    def add_samples(self, bem_data: list[dict]) -> None:
        from hull_opt.config import design_vector_names
        expected_dim = len(design_vector_names())
        for d in bem_data:
            dv = np.asarray(d["design_vector"], dtype=np.float64).ravel()
            if dv.shape[0] != expected_dim:
                continue
            self._data.append({
                "x": dv,
                "omega": float(d["omega"]),
                "heading": float(d["heading_deg"]),
                "y": np.array([
                    float(d["heave_rao"]),
                    float(d["pitch_rao"]),
                    float(d["roll_rao"]),
                ], dtype=np.float64),
            })
        self._n_samples = len(self._data)

    def train(self, force: bool = False) -> None:
        if not force and self._n_samples < self.config.min_samples:
            return
        n = len(self._data)
        if n == 0:
            return
        X_raw = np.zeros((n, self.config.input_dim), dtype=np.float64)
        Y_raw = np.zeros((n, 3), dtype=np.float64)
        for i, d in enumerate(self._data):
            X_raw[i] = np.concatenate([d["x"], [d["omega"], d["heading"]]])
            Y_raw[i] = d["y"]

        self._x_mean = np.mean(X_raw, axis=0)
        self._x_std = np.std(X_raw, axis=0)
        self._x_std = np.where(self._x_std < 1e-12, 1.0, self._x_std)
        self._y_mean = np.mean(Y_raw, axis=0)
        self._y_std = np.std(Y_raw, axis=0)
        self._y_std = np.where(self._y_std < 1e-12, 1.0, self._y_std)

        X_norm = (X_raw - self._x_mean) / self._x_std
        Y_norm = (Y_raw - self._y_mean) / self._y_std

        rng = np.random.default_rng(42)
        indices = rng.permutation(n)
        n_hold = max(1, int(n * self.config.holdout_frac))
        hold_idx = indices[:n_hold]
        train_idx = indices[n_hold:]

        X_tr = torch.tensor(X_norm[train_idx], dtype=torch.float32, device=self._device)
        Y_tr = torch.tensor(Y_norm[train_idx], dtype=torch.float32, device=self._device)
        X_ho = torch.tensor(X_norm[hold_idx], dtype=torch.float32, device=self._device)
        Y_ho = torch.tensor(Y_norm[hold_idx], dtype=torch.float32, device=self._device)

        layers = []
        in_dim = self.config.input_dim
        for _ in range(self.config.n_layers):
            layers.append(nn.Linear(in_dim, self.config.hidden_dim))
            layers.append(nn.SiLU())
            in_dim = self.config.hidden_dim
        layers.append(nn.Linear(self.config.hidden_dim, 3))
        model = nn.Sequential(*layers).to(self._device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        loss_fn = nn.MSELoss()

        best_ho_loss = float("inf")
        patience = 0
        max_patience = 50
        best_state = None

        for epoch in range(500):
            model.train()
            pred = model(X_tr)
            loss = loss_fn(pred, Y_tr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                ho_pred = model(X_ho)
                ho_loss = loss_fn(ho_pred, Y_ho).item()

            if ho_loss < best_ho_loss:
                best_ho_loss = ho_loss
                patience = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= max_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model
        self._trained = True

        model.eval()
        with torch.no_grad():
            ho_pred_np = model(X_ho).cpu().numpy() * self._y_std + self._y_mean
            Y_ho_np = Y_ho.cpu().numpy() * self._y_std + self._y_mean
            rmse = float(np.sqrt(np.mean((ho_pred_np - Y_ho_np) ** 2)))
            mean_y = float(np.mean(np.abs(Y_ho_np)))
            relative_error = rmse / max(1e-10, mean_y)

        if relative_error > self.config.error_threshold:
            logger.warning(
                f"RAO surrogate holdout RMSE/mean={relative_error:.4f} > "
                f"threshold={self.config.error_threshold}: low confidence"
            )

    def predict(self, design_vector: np.ndarray, omega: float, heading_deg: float) -> tuple[float, float, float]:
        if not self.ready:
            raise RuntimeError("RAO surrogate not ready: call train() first")
        x = np.concatenate([design_vector.ravel(), [omega, heading_deg]]).astype(np.float64)
        x_norm = ((x - self._x_mean) / self._x_std).astype(np.float32)
        xt = torch.tensor(x_norm, dtype=torch.float32, device=self._device).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            y_pred = self._model(xt).cpu().numpy().ravel()
        y_denorm = y_pred * self._y_std + self._y_mean
        return (float(y_denorm[0]), float(y_denorm[1]), float(y_denorm[2]))

    def predict_batch(self, design_vector: np.ndarray, omegas: np.ndarray, headings_deg: list[float]) -> dict:
        if not self.ready:
            raise RuntimeError("RAO surrogate not ready: call train() first")
        dv = design_vector.ravel()
        n_o = len(omegas)
        n_h = len(headings_deg)
        n_total = n_o * n_h

        X = np.zeros((n_total, self.config.input_dim), dtype=np.float64)
        omega_list = np.zeros(n_total, dtype=np.float64)
        heading_list = np.zeros(n_total, dtype=np.float64)
        idx = 0
        for h in headings_deg:
            for w in omegas:
                X[idx] = np.concatenate([dv, [w, h]])
                omega_list[idx] = w
                heading_list[idx] = h
                idx += 1

        X_norm = ((X - self._x_mean) / self._x_std).astype(np.float32)
        Xt = torch.tensor(X_norm, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            Y_pred = self._model(Xt).cpu().numpy()
        Y_denorm = Y_pred * self._y_std + self._y_mean

        return {
            "omega": omega_list,
            "heading_deg": heading_list,
            "heave_rao": Y_denorm[:, 0],
            "pitch_rao": Y_denorm[:, 1],
            "roll_rao": Y_denorm[:, 2],
        }

    @staticmethod
    def anchor_grid(omega_min: float, omega_max: float, n_freq: int = 6) -> np.ndarray:
        return np.linspace(omega_min, omega_max, n_freq)
