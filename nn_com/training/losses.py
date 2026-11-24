"""Normalized multi-state trajectory loss and physical parameter regularizers."""

from __future__ import annotations

import torch


# Physical tolerance scales: 1 K for temps, 0.2 bar for p_in, 2.0 bar for p_out
STATE_SCALES = [1.0, 1.0, 1.0, 1.0, 1.0, 0.2e5, 2.0e5, 1.0, 1.0]

# Balanced weights covering all 5 thermal states + cabin interior + BOTH pressures
STATE_WEIGHTS = [1.0, 1.0, 1.0, 0.0, 2.0, 1.2, 1.2, 0.5, 4.0]


def normalized_state_mse(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """Compute weighted normalized MSE over states [..., 9]."""
    scales = torch.tensor(STATE_SCALES, dtype=x_pred.dtype, device=x_pred.device)
    weights = torch.tensor(STATE_WEIGHTS, dtype=x_pred.dtype, device=x_pred.device)
    err = (x_pred - x_true) / scales
    return torch.mean(weights * (err ** 2))


def parameter_prior_loss(log_p: torch.Tensor, y_mean: torch.Tensor) -> torch.Tensor:
    """Mild quadratic prior keeping log-parameters centered near physical nominals."""
    return torch.mean((log_p - y_mean) ** 2)


def parameter_smoothness_loss(log_p_seq: torch.Tensor) -> torch.Tensor:
    """Temporal smoothness penalty on predicted log-parameters [T, B, n_params]."""
    if log_p_seq.shape[0] < 2:
        return torch.zeros((), dtype=log_p_seq.dtype, device=log_p_seq.device)
    return torch.mean((log_p_seq[1:] - log_p_seq[:-1]) ** 2)
