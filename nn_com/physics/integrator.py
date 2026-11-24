"""Open-loop RK4 integrators and classical prepParaVec interpolation baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import scipy.io as sio
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import torch

from nn_com.data.coolprop_engine import CoolPropLUTTorch, pressure_guard_np, pressure_guard_torch
from nn_com.data.dataset import TensorScenario, build_feature_tensor_step
from nn_com.physics.com_torch import COMPhysics


class InterpScheduler:
    """Exact Python port of Source Files/prepParaVec.m (11-param scattered interpolation)."""

    # 0-based state indices corresponding to MATLAB idxState = [8, 9, 5, 2, 1, 3, 5, 9, 6, 7, 6]
    IDX_STATE = [7, 8, 4, 1, 0, 2, 4, 8, 5, 6, 5]

    def __init__(self, tuned_vars_path: Path):
        m = sio.loadmat(tuned_vars_path, squeeze_me=True, struct_as_record=False)
        tv = m["tunedVariables"]
        amb = np.asarray(tv.ambient, dtype=np.float64).ravel()
        states = np.asarray(tv.states, dtype=np.float64)
        params = np.asarray(tv.parameters, dtype=np.float64)

        self.lin_interps = []
        self.near_interps = []
        for i in range(11):
            sj = self.IDX_STATE[i]
            pts = np.column_stack([amb, states[:, sj]])
            vals = params[:, i]
            good = np.all(np.isfinite(pts), axis=1) & np.isfinite(vals)
            self.lin_interps.append(LinearNDInterpolator(pts[good], vals[good]))
            self.near_interps.append(NearestNDInterpolator(pts[good], vals[good]))

    def predict_np(self, x_guarded: np.ndarray, tamb_c: np.ndarray) -> np.ndarray:
        """Predict [B, 11] parameters at (tamb_c [B], x_guarded [B, 9])."""
        b = x_guarded.shape[0]
        out = np.empty((b, 11), dtype=np.float64)
        for i in range(11):
            sj = self.IDX_STATE[i]
            q = np.column_stack([tamb_c, x_guarded[:, sj]])
            v = self.lin_interps[i](q)
            bad = ~np.isfinite(v)
            if np.any(bad):
                v[bad] = self.near_interps[i](q[bad])
            out[:, i] = v
        return out


@torch.no_grad()
def rollout_open_loop_batch(
    scenarios: List[TensorScenario],
    com: COMPhysics,
    lut: CoolPropLUTTorch,
    scheduler_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    dt: float = 1.0,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Run full-horizon open-loop RK4 simulation for a list of equal-length scenarios
    simultaneously (exact match to validate_com_fig2.m simulate_com).

    scheduler_fn(x_guarded [B,9], tamb_c [B], bat_i_avg [B], speed_kmh [B]) -> P [B, n_params]
    """
    lengths = {sc.length for sc in scenarios}
    if len(lengths) != 1:
        # Fallback to length-grouped calls if scenarios have differing durations
        results: List[np.ndarray] = [None] * len(scenarios)  # type: ignore
        by_len: dict[int, List[int]] = {}
        for idx, sc in enumerate(scenarios):
            by_len.setdefault(sc.length, []).append(idx)
        for idxs in by_len.values():
            sub_res = rollout_open_loop_batch([scenarios[i] for i in idxs], com, lut, scheduler_fn, dt=dt, verbose=verbose)
            for idx, arr in zip(idxs, sub_res):
                results[idx] = arr
        return results

    n_steps = scenarios[0].length
    b = len(scenarios)
    device = scenarios[0].X.device

    X_traj = torch.empty((b, n_steps, 9), dtype=torch.float64, device=device)
    x = torch.stack([sc.X[0] for sc in scenarios], dim=0)
    X_traj[:, 0] = x

    U_all = torch.stack([sc.U for sc in scenarios], dim=0)
    env_t_all = torch.stack([sc.env_t_k for sc in scenarios], dim=0)
    tamb_all = torch.stack([sc.tamb_c for sc in scenarios], dim=0)
    speed_all = torch.stack([sc.speed_kmh for sc in scenarios], dim=0)
    batI_all = torch.stack([sc.batI for sc in scenarios], dim=0)
    batI_avg_all = torch.stack([sc.batI_avg for sc in scenarios], dim=0)
    motQ_all = torch.stack([sc.motQ for sc in scenarios], dim=0)
    invQ_all = torch.stack([sc.invQ for sc in scenarios], dim=0)
    dcdcQ_all = torch.stack([sc.dcdcQ for sc in scenarios], dim=0)
    ctrl_all = {k: torch.stack([sc.ctrl[k] for sc in scenarios], dim=0) for k in scenarios[0].ctrl}

    for k in range(n_steps - 1):
        xp = pressure_guard_torch(x)
        P_k = scheduler_fn(xp, tamb_all[:, k], batI_avg_all[:, k], speed_all[:, k])
        est_k = lut.evaluate(
            xp,
            env_t_all[:, k],
            speed_all[:, k],
            batI_all[:, k],
            motQ_all[:, k],
            invQ_all[:, k],
            dcdcQ_all[:, k],
            U_all[:, k, 5],  # validate_com_fig2.m L209 passes u(6) as fanRpm_prev
        )
        ctrl_k = {ck: cv[:, k] for ck, cv in ctrl_all.items()}
        x = com.step_rk4(x, U_all[:, k], est_k, ctrl_k, P_k, dt=dt)
        X_traj[:, k + 1] = x
        if verbose and (k + 1) % 600 == 0:
            print(f"  rollout step {k + 1}/{n_steps - 1}...", flush=True)

    return [X_traj[i].cpu().numpy() for i in range(b)]


def make_interp_scheduler_fn(interp: InterpScheduler) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Wrap InterpScheduler into scheduler_fn compatible with rollout_open_loop_batch."""
    def fn(xp: torch.Tensor, tamb_c: torch.Tensor, bat_i_avg: torch.Tensor, speed_kmh: torch.Tensor) -> torch.Tensor:
        p_np = interp.predict_np(xp.cpu().numpy(), tamb_c.cpu().numpy())
        return torch.tensor(p_np, dtype=xp.dtype, device=xp.device)
    return fn


def make_nn_scheduler_fn(net) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Wrap NNScheduler into scheduler_fn compatible with rollout_open_loop_batch."""
    def fn(xp: torch.Tensor, tamb_c: torch.Tensor, bat_i_avg: torch.Tensor, speed_kmh: torch.Tensor) -> torch.Tensor:
        feat = build_feature_tensor_step(xp, tamb_c, bat_i_avg, speed_kmh)
        return net.forward_params(feat)
    return fn
