"""Bounded log-multiplier MLP parameter scheduler compatible with get_tuneS.m."""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


class NNScheduler(nn.Module):
    """
    Plain Linear+Tanh MLP predicting log-parameters centered on nominal physics constants.

    Exact forward math matches get_tuneS.m:
        z     = (feat - x_mean) / x_std
        log_p = mlp(z) * y_std + y_mean
        p     = clamp(exp(log_p), p_lo, p_hi)
    """

    def __init__(
        self,
        features: List[str],
        n_params: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        p_nom: np.ndarray,
        p_lo: np.ndarray,
        p_hi: np.ndarray,
        env_lo: np.ndarray,
        env_hi: np.ndarray,
        hidden_dim: int = 32,
        n_hidden_layers: int = 2,
        y_std_init: float = 0.5,
    ):
        super().__init__()
        self.features = list(features)
        self.n_params = int(n_params)

        layers: List[nn.Module] = []
        in_dim = len(features)
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        out_lin = nn.Linear(in_dim, n_params)
        nn.init.normal_(out_lin.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(out_lin.bias)
        layers.append(out_lin)
        self.net = nn.Sequential(*layers)

        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float64))
        self.register_buffer("x_std", torch.tensor(x_std, dtype=torch.float64))
        self.register_buffer("y_mean", torch.tensor(np.log(p_nom), dtype=torch.float64))
        self.register_buffer("y_std", torch.full((n_params,), y_std_init, dtype=torch.float64))
        self.register_buffer("p_lo", torch.tensor(p_lo, dtype=torch.float64))
        self.register_buffer("p_hi", torch.tensor(p_hi, dtype=torch.float64))
        self.register_buffer("env_lo", torch.tensor(env_lo, dtype=torch.float64))
        self.register_buffer("env_hi", torch.tensor(env_hi, dtype=torch.float64))

    def forward_log_params(self, f_raw: torch.Tensor) -> torch.Tensor:
        z = (f_raw - self.x_mean) / self.x_std
        return self.net(z) * self.y_std + self.y_mean

    def forward_params(self, f_raw: torch.Tensor) -> torch.Tensor:
        log_p = self.forward_log_params(f_raw)
        # Smooth soft bounding + exact clamp matching MATLAB export
        log_lo = torch.log(self.p_lo)
        log_hi = torch.log(self.p_hi)
        log_p_c = torch.clamp(log_p, min=log_lo, max=log_hi)
        return torch.exp(log_p_c)

    def forward(self, f_raw: torch.Tensor) -> torch.Tensor:
        return self.forward_params(f_raw)
