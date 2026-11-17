"""
Train a neural-network scheduler for the group's NMPC parameters p1..p11
(supervised, on the window-identified parameters in tunedVariables_minus5_minus10.mat)
and export it as nn_<NAME>.mat for get_tuneS.m.

This is the proposal's "main path": labels from the offline identification
campaign -> regression from the operating point to the parameter vector.
Unlike prepParaVec (which interpolates each p_i over ambient temperature and
ONE state), the network uses ambient temperature and all states jointly.

Run:  python train_gamma_nn.py            (file must be in this folder or Data/)
Output: nn_george.mat (+ gamma_nn_report.txt, gamma_nn_fit.png)
"""
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from export_nn_for_matlab import export_nn_for_matlab, matlab_forward

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.set_default_dtype(torch.float64)

NAME = "george"                      # -> nn_george.mat, SCHED_MODE = 'george'
TV_FILE = "tunedVariables_minus5_minus10.mat"
STATE_NAMES = ["motT", "invT", "dcdcT", "SOC", "batT", "p_in", "p_out", "cabIntT", "cabAirT"]
FEATURES = ["Tamb_C"] + STATE_NAMES  # all available in the MPC (see get_tuneS.m)
IDX_STATE = [8, 9, 5, 2, 1, 3, 5, 9, 6, 7, 6]   # prepParaVec's driving state per parameter (1-based)
HIDDEN, WEIGHT_DECAY, EPOCHS, LR, VAL_FRACTION, SEED = 32, 1e-3, 4000, 3e-3, 0.2, 0
# The file holds one row per second, but parameters are identified per window
# (e.g. 120 s): consecutive rows share one label. Validation therefore holds out
# WHOLE windows, otherwise neighbouring seconds of the same window leak into training.


def load_tv():
    for p in [TV_FILE, os.path.join("Data", TV_FILE)]:
        if os.path.exists(p):
            m = loadmat(p, squeeze_me=True, struct_as_record=False)
            tv = m["tunedVariables"]
            S = np.atleast_2d(np.asarray(tv.states, float))
            P = np.atleast_2d(np.asarray(tv.parameters, float))
            A = np.asarray(tv.ambient, float).ravel()
            ok = np.isfinite(S).all(1) & np.isfinite(P).all(1) & np.isfinite(A)
            print(f"Loaded {p}: {ok.sum()} rows, ambient temps {sorted(set(np.round(A[ok], 2)))}")
            return S[ok], P[ok], A[ok]
    raise FileNotFoundError(f"{TV_FILE} not found here or in Data/")


def interp_baseline(S_tr, P_tr, A_tr, S_q, A_q):
    """Python copy of prepParaVec: 2-D linear interpolation per parameter, nearest outside."""
    out = np.zeros((len(S_q), 11))
    for i in range(11):
        pts = np.c_[A_tr, S_tr[:, IDX_STATE[i] - 1]]
        q = np.c_[A_q, S_q[:, IDX_STATE[i] - 1]]
        lin = LinearNDInterpolator(pts, P_tr[:, i])(q)
        near = NearestNDInterpolator(pts, P_tr[:, i])(q)
        out[:, i] = np.where(np.isfinite(lin), lin, near)
    return out


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    S, P, A = load_tv()
    X = np.c_[A, S]                                  # [Tamb_C, 9 states] in MPC units
    n = len(X)

    # window id: a new window starts whenever the label vector or the ambient changes
    change = np.r_[True, (np.abs(np.diff(P, axis=0)).max(1) > 1e-12) | (np.diff(A) != 0)]
    win = np.cumsum(change) - 1
    n_win = win.max() + 1
    val_win = rng.permutation(n_win)[: max(1, int(VAL_FRACTION * n_win))]
    is_val = np.isin(win, val_win)
    va, tr = np.where(is_val)[0], np.where(~is_val)[0]
    print(f"{n} rows in {n_win} identified windows -> holding out {len(val_win)} whole windows")

    lo_b, hi_b = P.min(0), P.max(0)
    n_bound = ((np.isclose(P, lo_b) | np.isclose(P, hi_b)) & (hi_b - lo_b > 1e-9)).mean(0)

    # targets in log space: parameters are positive scaling factors spanning ~0.1..10
    LP = np.log(np.maximum(P, 1e-6))
    xm, xs = X[tr].mean(0), X[tr].std(0); xs[xs < 1e-9] = 1.0
    ym, ys = LP[tr].mean(0), LP[tr].std(0); ys[ys < 1e-9] = 1.0
    Xt = torch.tensor((X - xm) / xs)
    Yt = torch.tensor((LP - ym) / ys)

    model = nn.Sequential(nn.Linear(X.shape[1], HIDDEN), nn.Tanh(),
                          nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 11))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, best_state = np.inf, None
    for ep in range(EPOCHS):
        loss = ((model(Xt[tr]) - Yt[tr]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            vl = ((model(Xt[va]) - Yt[va]) ** 2).mean().item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    with torch.no_grad():
        P_nn = np.clip(np.exp(model(Xt).numpy() * ys + ym), lo_b, hi_b)
    P_int = interp_baseline(S[tr], P[tr], A[tr], S[va], A[va])

    # linear (ridge) regression as a simple ML baseline
    Xa = np.c_[np.ones(len(tr)), (X[tr] - xm) / xs]
    W = np.linalg.solve(Xa.T @ Xa + 1e-2 * np.eye(Xa.shape[1]), Xa.T @ P[tr])
    P_lin = np.c_[np.ones(len(va)), (X[va] - xm) / xs] @ W
    P_lin = np.clip(P_lin, lo_b, hi_b)

    sd = P[tr].std(0)
    const = sd < 1e-6 * (np.abs(P[tr].mean(0)) + 1e-12)
    rel = lambda Pp: np.where(const, np.nan, np.sqrt(((Pp - P[va]) ** 2).mean(0)) / np.where(const, 1, sd))
    lines = [f"Supervised scheduler for p1..p11: {n} rows in {n_win} windows; "
             f"{len(val_win)} whole windows held out ({len(va)} rows)",
             "Normalised RMSE on held-out windows (RMSE / std of the parameter; lower is better)",
             f"{'param':6s}{'interp (prepParaVec)':>22s}{'linear':>10s}{'NN':>10s}{'labels at bound':>18s}"]
    ri, rl, rn = rel(P_int), rel(P_lin), rel(P_nn[va])
    for i in range(11):
        vals = np.unique(np.round(P[:, i], 9))
        if const[i]:
            lines.append(f"p{i+1:<5d}{'constant (' + format(P[0, i], '.4g') + ')':>42s}")
        elif len(vals) <= len(np.unique(A)):
            lines.append(f"p{i+1:<5d}{ri[i]:22.3f}{rl[i]:10.3f}{rn[i]:10.3f}{'one value per temperature':>27s}")
        else:
            lines.append(f"p{i+1:<5d}{ri[i]:22.3f}{rl[i]:10.3f}{rn[i]:10.3f}{100*n_bound[i]:16.0f} %")
    lines.append(f"{'mean':6s}{np.nanmean(ri):22.3f}{np.nanmean(rl):10.3f}{np.nanmean(rn):10.3f}")
    lines.append("'labels at bound' = share of identified values sitting at the min or max of their range;")
    lines.append("a high share means the window identification is poorly conditioned for that parameter.")
    rep = "\n".join(lines); print(rep)
    open("gamma_nn_report.txt", "w").write(rep + "\n")

    fig, ax = plt.subplots(3, 4, figsize=(15, 9))
    for i, a in enumerate(ax.flat[:11]):
        a.scatter(P[tr, i], P_nn[tr, i], s=10, label="train")
        a.scatter(P[va, i], P_nn[va, i], s=18, label="held out")
        lo, hi = P[:, i].min(), P[:, i].max()
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        a.set_title(f"p{i+1}: identified vs NN", fontsize=9)
    ax.flat[0].legend(fontsize=8); ax.flat[-1].axis("off")
    fig.tight_layout(); fig.savefig("gamma_nn_fit.png", dpi=110)

    fspan = X.max(0) - X.min(0)
    path = f"nn_{NAME}.mat"
    export_nn_for_matlab(model, FEATURES, xm, xs, path, y_mean=ym, y_std=ys, out_transform="exp",
                         p_lo=lo_b, p_hi=hi_b,          # never outside the identified range
                         env_lo=X.min(0) - 0.1 * fspan, env_hi=X.max(0) + 0.1 * fspan)
    d = loadmat(path, squeeze_me=False)
    d = {k: (v if not isinstance(v, np.ndarray) or v.dtype.kind != "U" else str(v.ravel()[0]))
         for k, v in d.items() if not k.startswith("__")}
    d["act"] = "tanh"; d["out_transform"] = "exp"; d["n_layers"] = float(np.asarray(d["n_layers"]).ravel()[0])
    err = max(np.abs(matlab_forward(d, X[k]) - P_nn[k]).max() for k in range(0, n, max(1, n // 50)))
    print(f"Check: MATLAB forward formula reproduces the network to {err:.1e}")


if __name__ == "__main__":
    main()