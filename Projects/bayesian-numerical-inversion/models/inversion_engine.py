from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn


def forward_infiltration_numpy(
    time: np.ndarray,
    ks: np.ndarray,
    sorptivity: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Philip infiltration model: I(t) = S * sqrt(t) + A * Ks * t.
    Supports batched parameter ensembles via broadcasting without Python loops.

    Args:
        time: Shape (T,)
        ks: Saturated hydraulic conductivity, shape (B,) or scalar
        sorptivity: Soil sorptivity, shape (B,) or scalar

    Returns:
        Cumulative infiltration array of shape (B, T)
    """
    t = np.asarray(time, dtype=np.float64)[None, :]          # (1, T)
    ks_col = np.atleast_1d(np.asarray(ks, dtype=np.float64))[:, None]       # (B, 1)
    s_col = np.atleast_1d(np.asarray(sorptivity, dtype=np.float64))[:, None] # (B, 1)

    if np.any(t < 0):
        raise ValueError("Time values must be non-negative.")

    # Vectorized broadcast: (B, 1) * (1, T) -> (B, T)
    return s_col * np.sqrt(t) + (0.667 * ks_col) * t


def invert_parameters_logspace(
    time: np.ndarray,
    observed_infiltration: np.ndarray,
    noise_std: float = 0.005,
    prior_mean_log10: tuple[float, float] = (-4.0, -1.0),
    prior_std_log10: tuple[float, float] = (1.5, 1.0),
) -> dict[str, float]:
    """
    MAP (Maximum A Posteriori) Bayesian inversion using log10 parameter scaling
    to stabilize ill-conditioned optimization across orders of magnitude.
    """
    mu_p = np.asarray(prior_mean_log10, dtype=np.float64)
    sig_p = np.asarray(prior_std_log10, dtype=np.float64)
    obs = np.asarray(observed_infiltration, dtype=np.float64)[None, :]

    def neg_log_posterior(theta_log10: np.ndarray) -> float:
        ks = 10.0 ** theta_log10[0]
        sorp = 10.0 ** theta_log10[1]
        pred = forward_infiltration_numpy(time, ks, sorp)

        # Data likelihood + Gaussian log-prior
        misfit = 0.5 * np.sum(((obs - pred) / noise_std) ** 2)
        prior_penalty = 0.5 * np.sum(((theta_log10 - mu_p) / sig_p) ** 2)
        return float(misfit + prior_penalty)

    res = minimize(
        neg_log_posterior,
        x0=mu_p.copy(),
        method="L-BFGS-B",
        bounds=[(-8.0, 0.0), (-4.0, 1.0)],
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed to converge: {res.message}")

    return {
        "ks_est": float(10.0 ** res.x[0]),
        "sorptivity_est": float(10.0 ** res.x[1]),
        "objective": float(res.fun),
    }


class SurrogateInfiltrationNet(nn.Module):
    """Differentiable PyTorch surrogate predicting infiltration curves in log-parameter space."""

    def __init__(self, num_time_steps: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_time_steps),
        )

    def forward(self, log10_params: torch.Tensor) -> torch.Tensor:
        # Enforce non-negative cumulative infiltration smoothly via softplus
        return torch.nn.functional.softplus(self.net(log10_params))


def train_surrogate(
    time: np.ndarray,
    n_samples: int = 512,
    epochs: int = 250,
    lr: float = 1e-2,
    seed: int = 42,
) -> SurrogateInfiltrationNet:
    """Train PyTorch surrogate with deterministic seeding and gradient clipping."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    log_ks = rng.uniform(-6.0, -2.0, size=n_samples)
    log_s = rng.uniform(-2.0, 0.0, size=n_samples)
    targets_np = forward_infiltration_numpy(time, 10.0 ** log_ks, 10.0 ** log_s)

    x_tensor = torch.tensor(np.column_stack([log_ks, log_s]), dtype=torch.float32)
    y_tensor = torch.tensor(targets_np, dtype=torch.float32)

    model = SurrogateInfiltrationNet(num_time_steps=len(time))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        preds = model(x_tensor)
        loss = loss_fn(preds, y_tensor)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss encountered during training.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    return model