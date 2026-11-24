"""Export trained NNScheduler checkpoints to nn_<name>.mat and verify MATLAB forward parity."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import scipy.io as sio
import torch

from export_nn_for_matlab import export_nn_for_matlab, matlab_forward
from nn_com.models.scheduler_net import NNScheduler


def export_scheduler_to_mat(net: NNScheduler, out_path: Path, verify_samples: np.ndarray | None = None) -> float:
    """Export NNScheduler weights to nn_<name>.mat and return max verification error."""
    net.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_nn_for_matlab(
        net.net,
        features=net.features,
        x_mean=net.x_mean.cpu().numpy(),
        x_std=net.x_std.cpu().numpy(),
        path=str(out_path),
        y_mean=net.y_mean.cpu().numpy(),
        y_std=net.y_std.cpu().numpy(),
        out_transform="exp",
        p_lo=net.p_lo.cpu().numpy(),
        p_hi=net.p_hi.cpu().numpy(),
        env_lo=net.env_lo.cpu().numpy(),
        env_hi=net.env_hi.cpu().numpy(),
    )

    if verify_samples is None:
        rng = np.random.default_rng(0)
        verify_samples = net.x_mean.cpu().numpy() + rng.normal(0.0, 1.0, size=(32, len(net.features))) * net.x_std.cpu().numpy()

    d_raw: Dict = sio.loadmat(str(out_path), squeeze_me=False)
    d = {
        k: (v if not isinstance(v, np.ndarray) or v.dtype.kind != "U" else str(v.ravel()[0]))
        for k, v in d_raw.items()
        if not k.startswith("__")
    }
    d["act"] = str(np.asarray(d_raw["act"]).ravel()[0])
    d["out_transform"] = str(np.asarray(d_raw["out_transform"]).ravel()[0])
    d["n_layers"] = float(np.asarray(d["n_layers"]).ravel()[0])

    with torch.no_grad():
        f_t = torch.tensor(verify_samples, dtype=torch.float64, device=net.x_mean.device)
        py_preds = net.forward_params(f_t).cpu().numpy()
        mat_preds = np.array([matlab_forward(d, row) for row in verify_samples])
        max_err = float(np.max(np.abs(py_preds - mat_preds)))

    if max_err >= 1e-11:
        raise RuntimeError(f"MATLAB forward verification mismatch: max_err={max_err:.3e}")
    print(f"[OK] Exported {out_path.name} (n_params={net.n_params}, max_err={max_err:.2e})")
    return max_err
