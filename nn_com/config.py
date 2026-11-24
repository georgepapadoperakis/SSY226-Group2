"""Configuration constants, parameter definitions, and train/val/test splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy.io as sio


STATE_NAMES: List[str] = [
    "motT", "invT", "dcdcT", "batSoc", "batT",
    "p_in", "p_out", "cabIntT", "cabAirT",
]

INPUT_NAMES: List[str] = [
    "compNrpm", "cabAirMfIn", "motPumpNrpm",
    "batPumpNrpm", "heaterPwr", "radFanrpm",
]

FEATURES: List[str] = [
    "Tamb_C", "motT", "invT", "dcdcT", "batT",
    "p_in", "p_out", "cabIntT", "cabAirT", "batI_avg", "speed",
]

# Paper-style benchmark states (5 thermal states + 2 pressures)
BENCHMARK_STATES: List[Tuple[str, int, float]] = [
    ("T_mot", 0, 1.0),
    ("T_inv", 1, 1.0),
    ("T_dcdc", 2, 1.0),
    ("T_b", 4, 1.0),
    ("T_cair", 8, 1.0),
    ("p_in", 5, 1e-5),
    ("p_out", 6, 1e-5),
]

# Paper Table 1 reference numbers (RMSE, MAE in degC for interp scheduler)
PAPER_TABLE1: Dict[str, List[Tuple[float, float]]] = {
    "T10": [(1.60, 1.35), (1.50, 1.26), (2.12, 1.61), (1.61, 1.36), (1.04, 0.58)],
    "T7":  [(1.70, 1.46), (1.53, 1.33), (2.00, 1.62), (1.93, 1.77), (1.45, 1.14)],
    "T5":  [(1.43, 1.18), (1.29, 1.05), (1.83, 1.30), (1.53, 1.33), (1.01, 0.81)],
}

# Physics-centered active-regime nominals for p1..p11 (avoids steady-state 0.10 bias)
DEFAULT_P1_11_NOM: List[float] = [
    1.50, 4.00, 0.85, 1.00, 1.00,
    2.40, 5.00, 0.95, 2.00, 1.00, 6.00,
]

# Tier 1 expanded parameters p12..p17 exact nominal values matching hardcoded MATLAB constants
P12_17_NOM: List[float] = [0.05, 1.0, 1.0, 0.9, 1.0, 1.0]

# Bounds for p1..p17 (log-multiplier NN stays strictly positive and well-conditioned)
P_LO_17: List[float] = [
    0.02, 0.02, 0.05, 0.02, 0.05,
    0.50, 0.05, 0.20, 0.05, 0.02, 0.20,
    0.005, 0.20, 0.30, 0.10, 0.20, 0.20,
]
P_HI_17: List[float] = [
    20.0, 20.0, 10.0, 20.0, 15.0,
    8.0,  15.0,  5.0, 25.0, 20.0, 30.0,
    1.0,   5.0,  3.0,  5.0,  3.0,  5.0,
]


def load_nominal_params(repo_root: Path, n_params: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_nom, p_lo, p_hi) of length n_params (11 or 17)."""
    if n_params not in (11, 17):
        raise ValueError(f"Unsupported n_params={n_params}; expected 11 or 17.")

    p1_11 = np.array(DEFAULT_P1_11_NOM, dtype=np.float64)

    if n_params == 11:
        p_nom = p1_11
    else:
        p_nom = np.concatenate([p1_11, np.array(P12_17_NOM, dtype=np.float64)])

    p_lo = np.array(P_LO_17[:n_params], dtype=np.float64)
    p_hi = np.array(P_HI_17[:n_params], dtype=np.float64)
    p_nom = np.clip(p_nom, p_lo * 1.01, p_hi * 0.99)
    return p_nom, p_lo, p_hi


@dataclass
class SplitConfig:
    """Train / validation / test split across drive cycles and ambient temperatures."""

    train_cycles: Set[str] = field(default_factory=lambda: {"WLTC_HWY", "NEDC", "US06"})
    train_temps: Set[int] = field(default_factory=lambda: {0, -5, -10, -12, -15, -25, -30})
    val_cycles: Set[str] = field(default_factory=lambda: {"WLTC_HWY", "NEDC", "US06"})
    val_temps: Set[int] = field(default_factory=lambda: {-7, -20})
    test_cycles: Set[str] = field(default_factory=lambda: {"UDDS"})
    paper_bench_cycle: str = "WLTC_HWY"
    paper_bench_temps: List[int] = field(default_factory=lambda: [-10, -7, -5])


@dataclass
class TrainConfig:
    """Hyperparameters for replay-stabilized curriculum differentiable-rollout training."""

    n_params: int = 17
    hidden_dim: int = 32
    n_hidden_layers: int = 2
    y_std_init: float = 1.5
    w_prior: float = 5e-4
    w_smooth: float = 1e-2
    grad_clip: float = 1.0
    seed: int = 42
    # Stage 1: short-horizon 60s Heun multi-shooting (cold-start enriched)
    s1_horizon: int = 60
    s1_steps: int = 60
    s1_lr: float = 2e-3
    # Stage 2: full 3600s continuous rollouts with 60s chunked TBPTT + cold-start replay
    s2_epochs: int = 5
    s2_chunk: int = 60
    s2_lr: float = 1e-3
    s2_val_every: int = 2
    lut_refresh_s: int = 5
