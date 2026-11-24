"""Dataset splitting and PyTorch tensor batching for multi-temp/multi-cycle rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from nn_com.config import FEATURES, SplitConfig
from nn_com.data.coolprop_engine import compute_supervisory_sequence
from nn_com.data.simlog_loader import ScenarioData


def build_feature_matrix_np(sc: ScenarioData) -> np.ndarray:
    """Build [N, 11] raw feature matrix matching FEATURES order."""
    n = len(sc.t)
    tamb_col = np.full(n, sc.tamb_c, dtype=np.float64)
    return np.column_stack([
        tamb_col,
        sc.X[:, 0],   # motT
        sc.X[:, 1],   # invT
        sc.X[:, 2],   # dcdcT
        sc.X[:, 4],   # batT
        sc.X[:, 5],   # p_in
        sc.X[:, 6],   # p_out
        sc.X[:, 7],   # cabIntT
        sc.X[:, 8],   # cabAirT
        sc.batI_avg,  # batI_avg
        sc.speed_kmh, # speed
    ])


def build_feature_tensor_step(
    x: torch.Tensor,
    tamb_c: torch.Tensor,
    bat_i_avg: torch.Tensor,
    speed_kmh: torch.Tensor,
) -> torch.Tensor:
    """Build [B, 11] feature tensor from live predicted state x [B, 9] in FEATURES order."""
    return torch.stack([
        tamb_c,
        x[:, 0],
        x[:, 1],
        x[:, 2],
        x[:, 4],
        x[:, 5],
        x[:, 6],
        x[:, 7],
        x[:, 8],
        bat_i_avg,
        speed_kmh,
    ], dim=-1)


@dataclass
class TensorScenario:
    """Pre-packed PyTorch scenario ready for vectorized rollout training or evaluation."""

    meta: ScenarioData
    X: torch.Tensor            # [T, 9]
    U: torch.Tensor            # [T, 6]
    F_ref: torch.Tensor        # [T, 11]
    tamb_c: torch.Tensor       # [T]
    env_t_k: torch.Tensor      # [T]
    speed_kmh: torch.Tensor    # [T]
    batI: torch.Tensor         # [T]
    batI_avg: torch.Tensor     # [T]
    motQ: torch.Tensor         # [T]
    invQ: torch.Tensor         # [T]
    dcdcQ: torch.Tensor        # [T]
    fan_rpm_prev: torch.Tensor # [T]
    ctrl: Dict[str, torch.Tensor]  # each [T]

    @property
    def length(self) -> int:
        return int(self.X.shape[0])


def to_tensor_scenario(sc: ScenarioData, device: torch.device) -> TensorScenario:
    """Convert ScenarioData to float64 tensors and precompute supervisory ctrl sequence."""
    ctrl_np = compute_supervisory_sequence(
        sc.tamb_c, sc.clntInvTout_K, sc.clntBatTin_K, sc.U[:, 0], sc.U[:, 2]
    )
    n = len(sc.t)
    fan_prev = np.empty(n, dtype=np.float64)
    fan_prev[0] = sc.U[0, 5]
    fan_prev[1:] = sc.U[:-1, 5]

    t64 = lambda a: torch.tensor(a, dtype=torch.float64, device=device)
    return TensorScenario(
        meta=sc,
        X=t64(sc.X),
        U=t64(sc.U),
        F_ref=t64(build_feature_matrix_np(sc)),
        tamb_c=t64(np.full(n, sc.tamb_c, dtype=np.float64)),
        env_t_k=t64(np.full(n, sc.tamb_c + 273.15, dtype=np.float64)),
        speed_kmh=t64(sc.speed_kmh),
        batI=t64(sc.batI),
        batI_avg=t64(sc.batI_avg),
        motQ=t64(sc.motQ),
        invQ=t64(sc.invQ),
        dcdcQ=t64(sc.dcdcQ),
        fan_rpm_prev=t64(fan_prev),
        ctrl={k: t64(v) for k, v in ctrl_np.items()},
    )


def split_scenarios(
    scenarios: List[ScenarioData],
    split_cfg: SplitConfig,
) -> Dict[str, List[ScenarioData]]:
    """Split scenarios into train, val, test, and paper_bench lists."""
    train, val, test, paper_bench = [], [], [], []
    for sc in scenarios:
        t_int = int(round(sc.tamb_c))
        if sc.cycle in split_cfg.test_cycles:
            test.append(sc)
        elif sc.cycle in split_cfg.val_cycles and t_int in split_cfg.val_temps:
            val.append(sc)
        elif sc.cycle in split_cfg.train_cycles and t_int in split_cfg.train_temps:
            train.append(sc)
        else:
            # Any extra dropped-in cycle/temp outside val/test defaults to test for unseen eval
            test.append(sc)

        if sc.cycle == split_cfg.paper_bench_cycle and t_int in split_cfg.paper_bench_temps:
            paper_bench.append(sc)

    return {
        "train": train,
        "val": val,
        "test": test,
        "paper_bench": sorted(paper_bench, key=lambda s: s.tamb_c),
    }


def compute_feature_stats(train_scenarios: List[ScenarioData]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (x_mean, x_std, env_lo, env_hi) across training trajectories."""
    feats = np.concatenate([build_feature_matrix_np(sc) for sc in train_scenarios], axis=0)
    x_mean = feats.mean(axis=0)
    x_std = feats.std(axis=0)
    x_std = np.where(x_std < 1e-9, 1.0, x_std)
    f_min = feats.min(axis=0)
    f_max = feats.max(axis=0)
    f_span = f_max - f_min
    env_lo = f_min - 0.35 * f_span
    env_hi = f_max + 0.35 * f_span
    return x_mean, x_std, env_lo, env_hi
