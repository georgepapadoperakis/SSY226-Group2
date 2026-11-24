"""CLI to train NN-11 or NN-17 parameter scheduler via 3-stage curriculum rollouts."""

import argparse
import os
from pathlib import Path
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nn_com.config import SplitConfig, TrainConfig
from nn_com.data.dataset import split_scenarios
from nn_com.data.simlog_loader import discover_scenarios
from nn_com.models.exporter import export_scheduler_to_mat
from nn_com.training.trainer import train_nn_scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NN COM parameter scheduler.")
    parser.add_argument("--param-set", type=int, choices=[11, 17], default=17, help="Number of scheduled COM parameters.")
    parser.add_argument("--name", type=str, default=None, help="Export name (writes nn_<name>.mat).")
    parser.add_argument("--hidden", type=int, default=32, help="Hidden layer width.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fast", action="store_true", help="Reduced step counts for quick smoke test.")
    args = parser.parse_args()

    torch.set_num_threads(min(4, os.cpu_count() or 2))
    exp_name = args.name or f"anton{args.param_set}"

    scenarios = discover_scenarios(REPO_ROOT)
    splits = split_scenarios(scenarios, SplitConfig())
    print(
        f"Discovered {len(scenarios)} runs -> "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}",
        flush=True,
    )

    cfg = TrainConfig(n_params=args.param_set, hidden_dim=args.hidden, seed=args.seed)
    if args.fast:
        cfg.s1_steps = 15
        cfg.s2_epochs = 1
        cfg.s2_val_every = 1

    ckpt_dir = REPO_ROOT / "Results" / "checkpoints" / exp_name
    net, _ = train_nn_scheduler(REPO_ROOT, splits["train"], splits["val"], cfg, ckpt_dir)

    out_mat = REPO_ROOT / f"nn_{exp_name}.mat"
    export_scheduler_to_mat(net, out_mat)


if __name__ == "__main__":
    main()
