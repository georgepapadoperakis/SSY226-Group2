"""
Export a PyTorch parameter-scheduler network to the common nn_<name>.mat format
read by get_tuneS.m (MATLAB, no toolboxes needed).

Every group member can use this for their own network, as long as it is a plain
stack of nn.Linear layers with one hidden activation type and it outputs the 11
COM parameters p1..p11 (in that order).

Example
-------
    from export_nn_for_matlab import export_nn_for_matlab
    export_nn_for_matlab(
        model,                                   # nn.Sequential(Linear, Tanh, Linear, ...)
        features=["Tamb_C", "batT", "cabAirT", "p_in", "p_out", "batI"],
        x_mean=x_mean, x_std=x_std,              # input normalisation used in training
        y_mean=y_mean, y_std=y_std,              # output de-normalisation (or None)
        out_transform="none",                    # or "exp" if the net predicts log(p)
        p_lo=p_lo, p_hi=p_hi,                    # plausible range of p1..p11 (or None)
        env_lo=f_min, env_hi=f_max,              # training envelope for OOD fallback (or None)
        path="nn_george.mat")

Allowed feature names (units as in the MPC; suffix converts units):
    Tamb_C, Tamb_K, motT, invT, dcdcT, SOC, batT, p_in, p_out, cabIntT, cabAirT, batI, speed
    temperatures in K unless suffixed _C (e.g. batT_C); pressures in Pa unless _bar / _MPa
"""
import numpy as np
import torch.nn as nn
from scipy.io import savemat

ALLOWED = ["Tamb_C", "Tamb_K", "motT", "invT", "dcdcT", "SOC", "batT", "p_in", "p_out",
           "cabIntT", "cabAirT", "batI", "speed"]
ACTS = {nn.Tanh: "tanh", nn.ReLU: "relu", nn.Sigmoid: "sigmoid", nn.GELU: "gelu", nn.Softplus: "softplus"}


def _check_feature(name):
    base = name
    for suf in ("_C", "_bar", "_MPa"):
        if name.endswith(suf) and name != "Tamb_C":
            base = name[: -len(suf)]
    if name not in ALLOWED and base not in ALLOWED:
        raise ValueError(f"Feature '{name}' is not available in the MPC. Allowed: {ALLOWED} (+ _C/_bar/_MPa)")


def export_nn_for_matlab(model, features, x_mean, x_std, path, y_mean=None, y_std=None,
                         out_transform="none", p_lo=None, p_hi=None, env_lo=None, env_hi=None):
    for f in features:
        _check_feature(f)
    linears, acts = [], set()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            linears.append(m)
        elif type(m) in ACTS:
            acts.add(ACTS[type(m)])
        elif isinstance(m, (nn.Sequential, nn.Dropout, nn.Identity)) or m is model:
            pass
        else:
            raise ValueError(f"Unsupported layer {type(m).__name__}; use Linear + one activation type.")
    if len(acts) > 1:
        raise ValueError(f"Use a single hidden activation type, found {acts}")
    if linears[0].in_features != len(features):
        raise ValueError(f"Network has {linears[0].in_features} inputs but {len(features)} feature names")
    if linears[-1].out_features != 11:
        raise ValueError(f"Network must output 11 values (p1..p11), has {linears[-1].out_features}")

    col = lambda a: np.asarray(a, dtype=float).reshape(-1, 1)
    d = {"features": np.array(features, dtype=object).reshape(-1, 1),
         "x_mean": col(x_mean), "x_std": col(x_std),
         "n_layers": float(len(linears)), "act": acts.pop() if acts else "tanh",
         "out_transform": out_transform}
    for i, lin in enumerate(linears, 1):
        d[f"W{i}"] = lin.weight.detach().cpu().double().numpy()        # (out x in)
        d[f"b{i}"] = lin.bias.detach().cpu().double().numpy().reshape(-1, 1)
    for k, v in [("y_mean", y_mean), ("y_std", y_std), ("p_lo", p_lo), ("p_hi", p_hi),
                 ("env_lo", env_lo), ("env_hi", env_hi)]:
        if v is not None:
            d[k] = col(v)
    savemat(path, d)
    print(f"Exported {path}: inputs {features}, {len(linears)} layers, act={d['act']}")


def matlab_forward(d, f):
    """Numpy copy of nn_forward in get_tuneS.m, used to verify an export."""
    z = (np.asarray(f, float) - d["x_mean"].ravel()) / d["x_std"].ravel()
    L = int(d["n_layers"])
    for i in range(1, L + 1):
        z = d[f"W{i}"] @ z + d[f"b{i}"].ravel()
        if i < L:
            a = d["act"]
            z = {"tanh": np.tanh, "relu": lambda v: np.maximum(v, 0), "sigmoid": lambda v: 1 / (1 + np.exp(-v)),
                 "softplus": lambda v: np.log1p(np.exp(v))}[a](z) if a != "gelu" else \
                0.5 * z * (1 + np.vectorize(__import__("math").erf)(z / np.sqrt(2)))
    if "y_mean" in d:
        z = z * d["y_std"].ravel() + d["y_mean"].ravel()
    if str(d.get("out_transform", "none")) == "exp":
        z = np.exp(z)
    if "p_lo" in d:
        z = np.maximum(z, d["p_lo"].ravel())
    if "p_hi" in d:
        z = np.minimum(z, d["p_hi"].ravel())
    return z
