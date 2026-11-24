"""Verify PyTorch COMPhysics open-loop parity against MATLAB CasADi COM results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import scipy.io as sio
import torch

from nn_com.config import BENCHMARK_STATES
from nn_com.data.coolprop_engine import CoolPropLUTTorch
from nn_com.data.dataset import to_tensor_scenario
from nn_com.data.simlog_loader import discover_scenarios
from nn_com.physics.com_torch import COMPhysics
from nn_com.physics.integrator import (
    InterpScheduler,
    make_interp_scheduler_fn,
    rollout_open_loop_batch,
)


def check_matlab_python_parity(repo_root: Path, matlab_mat_path: Path | None = None) -> Dict[str, Dict[str, float]]:
    """
    Compare PyTorch COMPhysics ('interp') against MATLAB validate_com_fig2 ('interp')
    on WLTC_HWY at -10, -7, -5 degC.
    """
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    if matlab_mat_path is None:
        # Note: Results/com_validation_2026_11_18_11_28/com_validation.mat is MAT-v7.3 (HDF5)
        cand = sorted((repo_root / "Results").glob("com_validation_*/com_validation.mat"))
        if not cand:
            raise FileNotFoundError("No Results/com_validation_*/com_validation.mat found.")
        matlab_mat_path = cand[-1]

    scenarios = discover_scenarios(repo_root)
    bench = [sc for sc in scenarios if sc.cycle == "WLTC_HWY" and int(round(sc.tamb_c)) in (-10, -7, -5)]
    bench = sorted(bench, key=lambda s: s.tamb_c)
    tensors = [to_tensor_scenario(sc, device) for sc in bench]

    lut = CoolPropLUTTorch(repo_root / "Data" / "coolprop_r134a_air_lut.npz").to(device)
    com = COMPhysics().to(device)
    interp = InterpScheduler(repo_root / "Data" / "tunedVariables_minus5_minus10.mat")
    sched_fn = make_interp_scheduler_fn(interp)

    py_trajs = rollout_open_loop_batch(tensors, com, lut, sched_fn, dt=1.0)

    # Compare against MATLAB Paper_RMSE / ours-interp CSV in the same folder (works even without h5py in dml env)
    csv_path = matlab_mat_path.parent / "table1_long.csv"
    matlab_metrics: Dict[tuple[int, str], tuple[float, float]] = {}
    if csv_path.is_file():
        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["Scheduler"] == "interp":
                    matlab_metrics[(int(float(row["Tamb_C"])), row["State"])] = (
                        float(row["RMSE"]),
                        float(row["MAE"]),
                    )

    summary: Dict[str, Dict[str, float]] = {}
    print(f"Parity check against {csv_path}")
    print(f"{'Temp':>5s} {'State':8s} {'Py_RMSE':>9s} {'MAT_RMSE':>9s} {'|Diff|':>9s} {'Py_MAE':>9s} {'MAT_MAE':>9s}")
    max_rmse_diff = 0.0
    for sc, py_x in zip(bench, py_trajs):
        t_int = int(round(sc.tamb_c))
        err = py_x - sc.X
        summary[sc.name] = {}
        for s_name, idx, scale in BENCHMARK_STATES:
            e = err[:, idx] * scale
            py_rmse = float(np.sqrt(np.mean(e ** 2)))
            py_mae = float(np.mean(np.abs(e)))
            mat_rmse, mat_mae = matlab_metrics.get((t_int, s_name), (float("nan"), float("nan")))
            diff = abs(py_rmse - mat_rmse) if np.isfinite(mat_rmse) else 0.0
            max_rmse_diff = max(max_rmse_diff, diff)
            summary[sc.name][s_name] = py_rmse
            print(f"{t_int:5d} {s_name:8s} {py_rmse:9.4f} {mat_rmse:9.4f} {diff:9.4f} {py_mae:9.4f} {mat_mae:9.4f}")

    print(f"\nMax RMSE difference (Python vs MATLAB 'interp'): {max_rmse_diff:.4f} (degC / bar)")
    return summary
