"""Automated Figure 2 & Table 1 benchmark across temperatures and drive cycles."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nn_com.config import BENCHMARK_STATES, PAPER_TABLE1
from nn_com.data.coolprop_engine import CoolPropLUTTorch
from nn_com.data.dataset import TensorScenario, to_tensor_scenario
from nn_com.data.simlog_loader import ScenarioData
from nn_com.models.scheduler_net import NNScheduler
from nn_com.physics.com_torch import COMPhysics
from nn_com.physics.integrator import (
    InterpScheduler,
    make_interp_scheduler_fn,
    make_nn_scheduler_fn,
    rollout_open_loop_batch,
)


def load_scheduler_checkpoint(ckpt_path: Path, device: torch.device) -> NNScheduler:
    """Reconstruct NNScheduler from checkpoint .pt."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    n_feats = len(ckpt["features"])
    dummy = np.zeros(n_feats, dtype=np.float64)
    p_dummy = np.ones(ckpt["n_params"], dtype=np.float64)
    net = NNScheduler(
        features=ckpt["features"],
        n_params=ckpt["n_params"],
        x_mean=dummy,
        x_std=np.ones(n_feats, dtype=np.float64),
        p_nom=p_dummy,
        p_lo=p_dummy * 0.01,
        p_hi=p_dummy * 50.0,
        env_lo=dummy,
        env_hi=dummy,
        hidden_dim=ckpt["hidden_dim"],
        n_hidden_layers=ckpt["n_hidden_layers"],
        y_std_init=ckpt["y_std_init"],
    ).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net


def compute_metrics_table(
    scenarios: List[ScenarioData],
    trajs_by_mode: Dict[str, List[np.ndarray]],
) -> List[Dict]:
    """Compute per-scenario, per-mode RMSE and MAE for all 7 benchmark states."""
    rows: List[Dict] = []
    for i, sc in enumerate(scenarios):
        t_int = int(round(sc.tamb_c))
        tg = f"T{abs(t_int)}"
        paper_vals = PAPER_TABLE1.get(tg, [])
        for mode, traj_list in trajs_by_mode.items():
            x_pred = traj_list[i]
            err = x_pred - sc.X
            for s_idx, (s_name, col_idx, scale) in enumerate(BENCHMARK_STATES):
                e = err[:, col_idx] * scale
                ok = np.isfinite(e)
                rmse = float(np.sqrt(np.mean(e[ok] ** 2))) if np.any(ok) else float("nan")
                mae = float(np.mean(np.abs(e[ok]))) if np.any(ok) else float("nan")
                p_rmse, p_mae = paper_vals[s_idx] if (sc.cycle == "WLTC_HWY" and s_idx < len(paper_vals)) else (float("nan"), float("nan"))
                rows.append({
                    "Cycle": sc.cycle,
                    "Tamb_C": t_int,
                    "Scheduler": mode,
                    "State": s_name,
                    "RMSE": rmse,
                    "MAE": mae,
                    "Paper_RMSE": p_rmse,
                    "Paper_MAE": p_mae,
                })
    return rows


def plot_fig2(
    sc: ScenarioData,
    preds_by_mode: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """Generate paper-styled 2x2 Figure 2 and 1x3 component plot for a single scenario."""
    colors = ["#0072BD", "#D95319", "#77AC30", "#7E2F8E", "#EDB120"]
    t = sc.t
    t_int = int(round(sc.tamb_c))
    tset_c = round(float(np.median(sc.X[max(0, len(t) - 1000):, 8]) - 273.15))

    specs = [
        (8, "(a) Cabin air temperature.", r"$T_{\mathrm{cair}}$ ($^\circ$C)", -273.15, 1.0, (-10.0, 30.0)),
        (4, "(b) Battery temperature.", r"$T_b$ ($^\circ$C)", -273.15, 1.0, (-12.0, 45.0)),
        (5, r"(c) Compressor inlet pressure $p_{\mathrm{in}}$.", "Pressure (bar)", 0.0, 1e-5, (0.0, 3.0)),
        (6, r"(d) Compressor outlet pressure $p_{\mathrm{out}}$.", "Pressure (bar)", 0.0, 1e-5, (0.0, 40.0)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), dpi=180)
    axes = axes.ravel()
    for s_idx, (col_idx, title_str, ylabel_str, off, scale, default_yl) in enumerate(specs):
        ax = axes[s_idx]
        all_vals = [(sc.X[:, col_idx] + off) * scale]
        for m_idx, (mode, xp) in enumerate(preds_by_mode.items()):
            yv = (xp[:, col_idx] + off) * scale
            all_vals.append(yv)
            ax.plot(t, yv, "-", color=colors[m_idx % len(colors)], lw=1.9, label=f"COM {mode} ({t_int} C)")
        ref_v = (sc.X[:, col_idx] + off) * scale
        ax.plot(t, ref_v, "--", color="#003F72", lw=1.5, label=f"Ref ({t_int} C)")
        if s_idx == 0:
            ax.plot([t[0], t[-1]], [tset_c, tset_c], "k:", lw=1.6, label=f"Setpoint ({tset_c} C)")

        flat = np.concatenate([a[np.isfinite(a)] for a in all_vals])
        yl_lo = min(default_yl[0], float(np.floor(flat.min()))) if flat.size else default_yl[0]
        yl_hi = max(default_yl[1], float(np.ceil(flat.max()))) if flat.size else default_yl[1]
        ax.set_xlim(0.0, float(t[-1]))
        ax.set_ylim(yl_lo, yl_hi)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel_str)
        ax.set_title(title_str, fontsize=10.5)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(f"Figure 2 Replication — {sc.cycle} at Tamb = {t_int} C", fontsize=12, y=0.99)
    fig.tight_layout()
    tag = f"{sc.cycle}_T{abs(t_int)}" if t_int < 0 else f"{sc.cycle}_T0"
    if t_int > 0:
        tag = f"{sc.cycle}_Tp{t_int}"
    fig.savefig(out_dir / f"fig2_{tag}.png", dpi=200)
    plt.close(fig)

    # Companion component temperatures plot
    fig2, axes2 = plt.subplots(1, 3, figsize=(13.0, 3.8), dpi=160)
    comp_specs = [(0, r"Motor $T_{\mathrm{mot}}$"), (1, r"Inverter $T_{\mathrm{inv}}$"), (2, r"DC-DC $T_{\mathrm{dcdc}}$")]
    for ax, (col_idx, ttl) in zip(axes2, comp_specs):
        ax.plot(t, sc.X[:, col_idx] - 273.15, "k--", lw=1.4, label="Ref")
        for m_idx, (mode, xp) in enumerate(preds_by_mode.items()):
            ax.plot(t, xp[:, col_idx] - 273.15, "-", color=colors[m_idx % len(colors)], lw=1.4, label=mode)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("degC")
        ax.set_title(ttl)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"components_{tag}.png", dpi=160)
    plt.close(fig2)


def run_benchmark(
    repo_root: Path,
    scenarios: List[ScenarioData],
    splits: Dict[str, List[ScenarioData]],
    schedulers: Dict[str, Callable],
    out_dir: Path,
    plot_scenarios: List[ScenarioData] | None = None,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Run full 3600s evaluation across scenarios, write CSV/TXT tables & Fig 2 plots."""
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    lut = CoolPropLUTTorch(repo_root / "Data" / "coolprop_r134a_air_lut.npz").to(device)
    com = COMPhysics().to(device)
    tensors = [to_tensor_scenario(sc, device) for sc in scenarios]

    trajs_by_mode: Dict[str, List[np.ndarray]] = {}
    for mode, sched_fn in schedulers.items():
        print(f"Running 3600s open-loop rollouts for '{mode}' across {len(tensors)} scenarios...")
        trajs_by_mode[mode] = rollout_open_loop_batch(tensors, com, lut, sched_fn, dt=1.0)

    rows = compute_metrics_table(scenarios, trajs_by_mode)

    # Save long CSV
    csv_path = out_dir / "table1_all_runs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Cycle", "Tamb_C", "Scheduler", "State", "RMSE", "MAE", "Paper_RMSE", "Paper_MAE"])
        writer.writeheader()
        writer.writerows(rows)

    # Generate Paper Benchmark Table 1 (WLTC_HWY at -10, -7, -5) and Split Summary Table
    modes = list(schedulers.keys())
    summary_txt_lines: List[str] = []

    def emit(line: str = "") -> None:
        print(line)
        summary_txt_lines.append(line)

    emit("=== Paper Benchmark Table 1: WLTC_HWY (-10, -7, -5 degC) ===")
    emit("(Temperatures in degC, pressures in bar; Paper reports temperatures only)")
    row_lookup = {(r["Cycle"], r["Tamb_C"], r["Scheduler"], r["State"]): (r["RMSE"], r["MAE"], r["Paper_RMSE"], r["Paper_MAE"]) for r in rows}

    for t_c in (-10, -7, -5):
        tag = " (unseen)" if t_c == -7 else ""
        emit(f"\n--- WLTC_HWY {t_c} degC{tag} ---")
        hdr = f"{'State':8s} {'Paper':>13s}" + "".join(f" {m:>15s}" for m in modes)
        emit(hdr)
        for s_name, _, _ in BENCHMARK_STATES:
            p_r, p_m = float("nan"), float("nan")
            vals_str = []
            for m in modes:
                tup = row_lookup.get(("WLTC_HWY", t_c, m, s_name))
                if tup is not None:
                    rmse, mae, pr, pm = tup
                    if np.isfinite(pr):
                        p_r, p_m = pr, pm
                    vals_str.append(f"{rmse:6.2f}/{mae:5.2f}".rjust(15))
                else:
                    vals_str.append("-".rjust(15))
            p_str = f"{p_r:6.2f}/{p_m:5.2f}" if np.isfinite(p_r) else "-"
            emit(f"{s_name:8s} {p_str:>13s}" + "".join(vals_str))

    emit("\n=== Multi-Temperature & Multi-Drive-Cycle Generalization Summary (Mean RMSE) ===")
    split_summary: Dict[str, Dict] = {}
    for split_name in ("train", "val", "test"):
        sub = splits.get(split_name, [])
        if not sub:
            continue
        sub_keys = {(sc.cycle, int(round(sc.tamb_c))) for sc in sub}
        emit(f"\n--- Split: {split_name.upper()} ({len(sub)} runs) ---")
        hdr = f"{'State':8s}" + "".join(f" {m:>14s}" for m in modes)
        emit(hdr)
        split_summary[split_name] = {}
        for s_name, _, _ in BENCHMARK_STATES:
            line = f"{s_name:8s}"
            split_summary[split_name][s_name] = {}
            for m in modes:
                vals = [
                    r["RMSE"] for r in rows
                    if (r["Cycle"], r["Tamb_C"]) in sub_keys and r["Scheduler"] == m and r["State"] == s_name and np.isfinite(r["RMSE"])
                ]
                mean_v = float(np.mean(vals)) if vals else float("nan")
                split_summary[split_name][s_name][m] = mean_v
                line += f" {mean_v:14.3f}"
            emit(line)

    with open(out_dir / "summary_tables.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_txt_lines) + "\n")

    # Generate Figure 2 plots for requested subset
    if plot_scenarios is None:
        plot_scenarios = [sc for sc in scenarios if (sc.cycle == "WLTC_HWY" and int(round(sc.tamb_c)) in (-10, -7, -5)) or sc.cycle == "UDDS" and int(round(sc.tamb_c)) in (-10, -20) or sc.cycle == "NEDC" and int(round(sc.tamb_c)) in (-7, -20) or sc.cycle == "US06" and int(round(sc.tamb_c)) in (-7, -20)]

    sc_to_idx = {(sc.cycle, int(round(sc.tamb_c))): idx for idx, sc in enumerate(scenarios)}
    for sc in plot_scenarios:
        idx = sc_to_idx[(sc.cycle, int(round(sc.tamb_c)))]
        preds = {m: trajs_by_mode[m][idx] for m in modes}
        plot_fig2(sc, preds, out_dir)

    print(f"\n[OK] Saved evaluation tables and {len(plot_scenarios)} Figure 2 plots to {out_dir}")
    return rows, split_summary
