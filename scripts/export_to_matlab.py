"""CLI to export any trained checkpoint in Results/checkpoints/<name>/best.pt to nn_<name>.mat."""

import argparse
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nn_com.evaluation.fig2_benchmark import load_scheduler_checkpoint
from nn_com.models.exporter import export_scheduler_to_mat


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained NN checkpoint to nn_<name>.mat.")
    parser.add_argument("name", type=str, help="Checkpoint name under Results/checkpoints/<name>/best.pt.")
    parser.add_argument("--out", type=str, default=None, help="Output .mat path (default: nn_<name>.mat in repo root).")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    ckpt_path = REPO_ROOT / "Results" / "checkpoints" / args.name / "best.pt"
    net = load_scheduler_checkpoint(ckpt_path, torch.device("cpu"))
    out_path = Path(args.out) if args.out else (REPO_ROOT / f"nn_{args.name}.mat")
    export_scheduler_to_mat(net, out_path)


if __name__ == "__main__":
    main()
