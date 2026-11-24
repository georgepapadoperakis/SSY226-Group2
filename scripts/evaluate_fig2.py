"""CLI to run full 3600s Figure 2 & Table 1 evaluation across all temps and drive cycles."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nn_com.config import SplitConfig
from nn_com.data.dataset import split_scenarios
from nn_com.data.simlog_loader import discover_scenarios
from nn_com.evaluation.fig2_benchmark import load_scheduler_checkpoint, run_benchmark
from nn_com.physics.integrator import (
    InterpScheduler,
    make_interp_scheduler_fn,
    make_nn_scheduler_fn,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate interp vs NN-11 vs NN-17 on Figure 2 & Table 1.")
    parser.add_argument("--ckpts", nargs="*", default=["anton11", "anton17"], help="Checkpoint folder names in Results/checkpoints/.")
    parser.add_argument("--no-interp", action="store_true", help="Skip interp baseline.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory.")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    device = torch.device("cpu")

    scenarios = discover_scenarios(REPO_ROOT)
    splits = split_scenarios(scenarios, SplitConfig())

    schedulers = {}
    if not args.no_interp:
        interp = InterpScheduler(REPO_ROOT / "Data" / "tunedVariables_minus5_minus10.mat")
        schedulers["interp"] = make_interp_scheduler_fn(interp)

    for name in args.ckpts:
        ckpt_path = REPO_ROOT / "Results" / "checkpoints" / name / "best.pt"
        if ckpt_path.is_file():
            net = load_scheduler_checkpoint(ckpt_path, device)
            schedulers[name] = make_nn_scheduler_fn(net)
        else:
            print(f"[Warn] Checkpoint not found: {ckpt_path}")

    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = Path(args.out_dir) if args.out_dir else (REPO_ROOT / "Results" / "evaluations" / f"eval_{stamp}")
    run_benchmark(REPO_ROOT, scenarios, splits, schedulers, out_dir)


if __name__ == "__main__":
    main()
