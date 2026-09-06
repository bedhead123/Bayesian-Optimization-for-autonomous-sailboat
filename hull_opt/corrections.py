import numpy as np
from typing import Optional

DEFAULT_CORRECTIONS = {
    "storm_accel": 1.0,
    "slam_pressure": 1.0,
    "roll_sigma": 1.0,
    "inverted_pressure": 1.0,
    "storm_wind_heel": 1.0,
    "capsize": 1.0,
}


def apply_corrections(margins: dict[str, float],
                      corrections: Optional[dict[str, float]] = None
                      ) -> dict[str, float]:
    if corrections is None:
        corrections = {}
    result = {}
    for key, val in margins.items():
        factor = corrections.get(key, 1.0)
        if val is None or not np.isfinite(val):
            result[key] = -1.0
        else:
            result[key] = val * factor
    return result


def update_correction(key: str, old_val: float, observed_ratio: float,
                       alpha: float = 0.3, clip: float = 0.5
                       ) -> tuple[float, bool]:
    if not np.isfinite(old_val) or old_val <= 0:
        old_val = 1.0
    if not np.isfinite(observed_ratio) or observed_ratio <= 0:
        return old_val, False
    candidate = alpha * observed_ratio + (1 - alpha) * old_val
    if abs(candidate - old_val) / max(old_val, 1e-10) > clip:
        clipped = old_val * (1.0 + np.sign(candidate - old_val) * clip)
        return float(clipped), True
    return float(candidate), candidate != old_val


try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CorrectionMLP:
    def __init__(self, input_dim: int = 21, hidden_dim: int = 64):
        if not TORCH_AVAILABLE:
            raise ImportError("torch not available")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6)
        )
        self._trained = False

    def predict(self, design_vector: np.ndarray) -> dict[str, float]:
        if not self._trained:
            return DEFAULT_CORRECTIONS.copy()
        with torch.no_grad():
            x = torch.from_numpy(design_vector.astype(np.float32)).unsqueeze(0)
            out = self.net(x).squeeze(0).numpy()
        keys = list(DEFAULT_CORRECTIONS.keys())
        return {keys[i]: float(out[i]) for i in range(len(keys))}

    def train(self, designs: list[np.ndarray], ratios: list[dict]) -> None:
        if len(designs) < 30:
            return
        self._trained = True
