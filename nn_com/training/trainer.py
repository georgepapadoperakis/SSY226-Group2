"""Replay-stabilized 2-stage curriculum differentiable-rollout trainer for NNScheduler."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from nn_com.config import FEATURES, TrainConfig, load_nominal_params
from nn_com.data.coolprop_engine import CoolPropLUTTorch, pressure_guard_torch
from nn_com.data.dataset import (
    TensorScenario,
    build_feature_tensor_step,
    compute_feature_stats,
    to_tensor_scenario,
)
from nn_com.data.simlog_loader import ScenarioData
from nn_com.models.scheduler_net import NNScheduler
from nn_com.physics.com_torch import COMPhysics
from nn_com.physics.integrator import make_nn_scheduler_fn, rollout_open_loop_batch
from nn_com.training.losses import (
    normalized_state_mse,
    parameter_prior_loss,
    parameter_smoothness_loss,
)


def evaluate_validation_rmse(
    net: NNScheduler,
    val_tensors: List[TensorScenario],
    com: COMPhysics,
    lut: CoolPropLUTTorch,
) -> Dict[str, float]:
    """Run full 3600s RK4 open-loop rollouts on validation runs and return summary metrics."""
    net.eval()
    sched_fn = make_nn_scheduler_fn(net)
    preds = rollout_open_loop_batch(val_tensors, com, lut, sched_fn, dt=1.0)

    rmses_temp = []
    rmses_pin = []
    rmses_pout = []
    for sc, x_pred in zip(val_tensors, preds):
        x_true = sc.X.cpu().numpy()
        err = x_pred - x_true
        if not np.all(np.isfinite(err)):
            return {"mean_temp_rmse": 1e6, "pin_rmse_bar": 1e6, "pout_rmse_bar": 1e6, "score": 1e6}
        for idx in (0, 1, 2, 4, 8):
            rmses_temp.append(float(np.sqrt(np.mean(err[:, idx] ** 2))))
        rmses_pin.append(float(np.sqrt(np.mean((err[:, 5] / 1e5) ** 2))))
        rmses_pout.append(float(np.sqrt(np.mean((err[:, 6] / 1e5) ** 2))))

    mean_t = float(np.mean(rmses_temp))
    mean_pin = float(np.mean(rmses_pin))
    mean_pout = float(np.mean(rmses_pout))
    score = mean_t + (mean_pin / 0.2) + (mean_pout / 2.0)
    return {
        "mean_temp_rmse": mean_t,
        "pin_rmse_bar": mean_pin,
        "pout_rmse_bar": mean_pout,
        "score": score,
    }


def _pack_train_tensors(train_tensors: List[TensorScenario]) -> Dict[str, object]:
    """Stack equal-length training scenarios into contiguous [R, T, ...] tensors."""
    return {
        "X": torch.stack([s.X for s in train_tensors], dim=0),
        "U": torch.stack([s.U for s in train_tensors], dim=0),
        "env": torch.stack([s.env_t_k for s in train_tensors], dim=0),
        "tamb": torch.stack([s.tamb_c for s in train_tensors], dim=0),
        "spd": torch.stack([s.speed_kmh for s in train_tensors], dim=0),
        "bi": torch.stack([s.batI for s in train_tensors], dim=0),
        "bia": torch.stack([s.batI_avg for s in train_tensors], dim=0),
        "mq": torch.stack([s.motQ for s in train_tensors], dim=0),
        "iq": torch.stack([s.invQ for s in train_tensors], dim=0),
        "dq": torch.stack([s.dcdcQ for s in train_tensors], dim=0),
        "F_ref": torch.stack([s.F_ref for s in train_tensors], dim=0),
        "ctrl": {k: torch.stack([s.ctrl[k] for s in train_tensors], dim=0) for k in train_tensors[0].ctrl},
    }


def _sample_cold_biased_starts(
    rng: np.random.Generator,
    n_runs: int,
    t_max_start: int,
    cold_frac: float = 0.4,
    cold_end: int = 900,
) -> torch.Tensor:
    """Sample start indices with `cold_frac` biased to [0, cold_end] and rest across full drive."""
    n_cold = int(round(cold_frac * n_runs))
    s_cold = rng.integers(0, min(cold_end, t_max_start), size=n_cold)
    s_full = rng.integers(0, t_max_start, size=n_runs - n_cold)
    starts = np.concatenate([s_cold, s_full])
    rng.shuffle(starts)
    return torch.tensor(starts, dtype=torch.long)


def _run_heun_chunk(
    net: NNScheduler,
    com: COMPhysics,
    lut: CoolPropLUTTorch,
    tr: Dict[str, object],
    x_init: torch.Tensor,
    starts: torch.Tensor,
    horizon: int,
    lut_refresh_s: int,
    w_prior: float,
    w_smooth: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll out `horizon` seconds with Heun integration and return (total_loss, detached_x_end)."""
    x = x_init
    roll_loss = torch.zeros((), dtype=torch.float64, device=x.device)
    log_p_list = []
    run_idx = torch.arange(x.shape[0], device=x.device)
    log_lo = torch.log(net.p_lo)
    log_hi = torch.log(net.p_hi)

    est_k: Dict[str, torch.Tensor] = {}
    P_k = torch.empty((x.shape[0], net.n_params), dtype=torch.float64, device=x.device)

    for k in range(horizon):
        t_idx = starts + k
        xp = pressure_guard_torch(x)
        if k % lut_refresh_s == 0:
            est_k = lut.evaluate(
                xp,
                tr["env"][run_idx, t_idx],
                tr["spd"][run_idx, t_idx],
                tr["bi"][run_idx, t_idx],
                tr["mq"][run_idx, t_idx],
                tr["iq"][run_idx, t_idx],
                tr["dq"][run_idx, t_idx],
                tr["U"][run_idx, t_idx, 5],
            )
            feat_k = build_feature_tensor_step(
                xp, tr["tamb"][run_idx, t_idx], tr["bia"][run_idx, t_idx], tr["spd"][run_idx, t_idx]
            )
            log_p_k = torch.clamp(net.forward_log_params(feat_k), min=log_lo, max=log_hi)
            P_k = torch.exp(log_p_k)
            log_p_list.append(log_p_k)
        else:
            est_k = dict(est_k)
            est_k["batImeas"] = tr["bi"][run_idx, t_idx]
            est_k["motQmeas"] = tr["mq"][run_idx, t_idx]
            est_k["invQmeas"] = tr["iq"][run_idx, t_idx]
            est_k["dcdcQmeas"] = tr["dq"][run_idx, t_idx]

        ctrl_k = {ck: cv[run_idx, t_idx] for ck, cv in tr["ctrl"].items()}
        x = com.step_heun(x, tr["U"][run_idx, t_idx], est_k, ctrl_k, P_k, dt=1.0)
        roll_loss = roll_loss + normalized_state_mse(x, tr["X"][run_idx, t_idx + 1]) / horizon

    log_p_seq = torch.stack(log_p_list, dim=0)
    loss = roll_loss + w_prior * parameter_prior_loss(log_p_seq, net.y_mean) + w_smooth * parameter_smoothness_loss(log_p_seq)
    return loss, x.detach()


def train_nn_scheduler(
    repo_root: Path,
    train_scenarios: List[ScenarioData],
    val_scenarios: List[ScenarioData],
    cfg: TrainConfig,
    ckpt_dir: Path,
) -> Tuple[NNScheduler, Dict]:
    """Train NNScheduler with Stage 1 multi-shooting + Stage 2 replay-stabilized 3600s TBPTT."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cpu")

    t_start = time.time()
    lut = CoolPropLUTTorch(repo_root / "Data" / "coolprop_r134a_air_lut.npz").to(device)
    com = COMPhysics().to(device)

    x_mean, x_std, env_lo, env_hi = compute_feature_stats(train_scenarios)
    p_nom, p_lo, p_hi = load_nominal_params(repo_root, cfg.n_params)

    net = NNScheduler(
        features=FEATURES,
        n_params=cfg.n_params,
        x_mean=x_mean,
        x_std=x_std,
        p_nom=p_nom,
        p_lo=p_lo,
        p_hi=p_hi,
        env_lo=env_lo,
        env_hi=env_hi,
        hidden_dim=cfg.hidden_dim,
        n_hidden_layers=cfg.n_hidden_layers,
        y_std_init=cfg.y_std_init,
    ).to(device)

    train_tensors = [to_tensor_scenario(sc, device) for sc in train_scenarios]
    val_tensors = [to_tensor_scenario(sc, device) for sc in val_scenarios]
    tr = _pack_train_tensors(train_tensors)

    R, T_tot, _ = tr["X"].shape
    T_m1 = T_tot - 1
    run_arange = torch.arange(R, dtype=torch.long)

    best_metrics = evaluate_validation_rmse(net, val_tensors, com, lut)
    best_state = copy.deepcopy(net.state_dict())
    history: List[Dict] = [{"stage": 0, "step": 0, "elapsed_s": round(time.time() - t_start, 1), **best_metrics}]
    print(
        f"[Init] Val Temp RMSE={best_metrics['mean_temp_rmse']:.3f} C | "
        f"p_in={best_metrics['pin_rmse_bar']:.3f} bar | "
        f"p_out={best_metrics['pout_rmse_bar']:.3f} bar | score={best_metrics['score']:.3f}",
        flush=True,
    )

    def maybe_update_best(stage_id: int, step_id: int, train_loss: float) -> None:
        nonlocal best_metrics, best_state
        m = evaluate_validation_rmse(net, val_tensors, com, lut)
        history.append({"stage": stage_id, "step": step_id, "train_loss": train_loss, "elapsed_s": round(time.time() - t_start, 1), **m})
        marker = ""
        if m["score"] < best_metrics["score"]:
            best_metrics = m
            best_state = copy.deepcopy(net.state_dict())
            marker = " *BEST*"
        print(
            f"  [Val S{stage_id}] Temp={m['mean_temp_rmse']:.3f} C | "
            f"p_in={m['pin_rmse_bar']:.3f} | p_out={m['pout_rmse_bar']:.3f} | "
            f"score={m['score']:.3f}{marker} ({time.time()-t_start:.1f}s)",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Stage 1: Cold-start-enriched 60s Heun multi-shooting
    # ------------------------------------------------------------------
    print(f"\n--- Stage 1: {cfg.s1_horizon}s Heun multi-shooting ({cfg.s1_steps} steps, batch={R}, lr={cfg.s1_lr:.1e}) ---", flush=True)
    opt1 = torch.optim.Adam(net.parameters(), lr=cfg.s1_lr)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.s1_steps, eta_min=cfg.s1_lr * 0.1)
    last_l1 = 0.0
    for step in range(1, cfg.s1_steps + 1):
        net.train()
        starts = _sample_cold_biased_starts(rng, R, T_m1 - cfg.s1_horizon, cold_frac=0.4, cold_end=900)
        x_init = tr["X"][run_arange, starts].clone()
        opt1.zero_grad()
        loss, _ = _run_heun_chunk(
            net, com, lut, tr, x_init, starts, cfg.s1_horizon, cfg.lut_refresh_s, cfg.w_prior, cfg.w_smooth
        )
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt1.step()
            last_l1 = float(loss.item())
        sch1.step()
        if step % 10 == 0 or step == cfg.s1_steps:
            print(f"  [S1 {step:3d}/{cfg.s1_steps}] loss={last_l1:.5f} ({time.time()-t_start:.1f}s)", flush=True)
    maybe_update_best(1, cfg.s1_steps, last_l1)

    # ------------------------------------------------------------------
    # Stage 2: Full 3600s continuous rollouts + cold-start replay
    # ------------------------------------------------------------------
    n_chunks = T_m1 // cfg.s2_chunk
    replays_per_epoch = (n_chunks // 20) * 3
    total_s2_steps = cfg.s2_epochs * (n_chunks + replays_per_epoch)
    print(
        f"\n--- Stage 2: Full 3600s continuous rollouts ({cfg.s2_epochs} epochs x {n_chunks} chunks + cold replay, lr={cfg.s2_lr:.1e}) ---",
        flush=True,
    )
    opt2 = torch.optim.Adam(net.parameters(), lr=cfg.s2_lr)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=total_s2_steps, eta_min=cfg.s2_lr * 0.1)

    for epoch in range(1, cfg.s2_epochs + 1):
        net.train()
        x_cont = tr["X"][:, 0].clone()
        ep_losses: List[float] = []
        for c_idx in range(n_chunks):
            t0_c = c_idx * cfg.s2_chunk
            starts = torch.full((R,), t0_c, dtype=torch.long)
            opt2.zero_grad()
            loss, x_next = _run_heun_chunk(
                net, com, lut, tr, x_cont, starts, cfg.s2_chunk, cfg.lut_refresh_s, cfg.w_prior, cfg.w_smooth
            )
            if torch.isfinite(loss) and torch.all(torch.isfinite(x_next)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                opt2.step()
                ep_losses.append(float(loss.item()))
                x_cont = x_next
            else:
                print(f"  [Warn] Non-finite rollout at E{epoch} chunk {c_idx+1}, resetting state to reference.", flush=True)
                opt2.zero_grad()
                bad = ~torch.all(torch.isfinite(x_next), dim=-1)
                x_cont = torch.where(bad.unsqueeze(-1), tr["X"][:, t0_c + cfg.s2_chunk], x_next)
            sch2.step()

            # Cold-start replay after chunks 20, 40, 60
            if (c_idx + 1) % 20 == 0:
                for _ in range(3):
                    r_starts = _sample_cold_biased_starts(rng, R, min(1200, T_m1 - cfg.s2_chunk), cold_frac=0.85, cold_end=600)
                    r_init = tr["X"][run_arange, r_starts].clone()
                    opt2.zero_grad()
                    r_loss, _ = _run_heun_chunk(
                        net, com, lut, tr, r_init, r_starts, cfg.s2_chunk, cfg.lut_refresh_s, cfg.w_prior, cfg.w_smooth
                    )
                    if torch.isfinite(r_loss):
                        r_loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                        opt2.step()
                    sch2.step()

            if (c_idx + 1) % 10 == 0 or (c_idx + 1) == n_chunks:
                mean_ep = float(np.mean(ep_losses)) if ep_losses else float("nan")
                print(
                    f"  [S2 E{epoch} {c_idx+1:2d}/{n_chunks}] chunk_loss={ep_losses[-1] if ep_losses else float('nan'):.4f} | "
                    f"mean_ep={mean_ep:.4f} ({time.time()-t_start:.1f}s)",
                    flush=True,
                )

        if (epoch % cfg.s2_val_every == 0) or (epoch == cfg.s2_epochs):
            maybe_update_best(2, epoch * n_chunks, float(np.mean(ep_losses)) if ep_losses else 1e6)

    net.load_state_dict(best_state)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "features": net.features,
            "n_params": cfg.n_params,
            "hidden_dim": cfg.hidden_dim,
            "n_hidden_layers": cfg.n_hidden_layers,
            "y_std_init": cfg.y_std_init,
            "best_val_metrics": best_metrics,
        },
        ckpt_dir / "best.pt",
    )
    with open(ckpt_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(
        f"\n[OK] Training complete in {time.time()-t_start:.1f}s. Best Val Score={best_metrics['score']:.3f} saved to {ckpt_dir / 'best.pt'}",
        flush=True,
    )
    return net, best_metrics
