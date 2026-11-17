"""
train_gamma_nn_v2.py  -  improved neural-network parameter scheduler (p1..p11)

What changed compared with train_gamma_nn.py
--------------------------------------------
1. One training row per identification WINDOW (60 rows), not per second (7200 rows).
   Each window has one identified parameter vector gamma(j); its inputs are the
   window-MEAN operating point, as in the paper (Lokur & Murgovski, Sec. IV):
       (T_amb, mean state x_bar(j), mean battery current I_b_bar(j))
   The states and current come from the measured reference signals in
   Data/tuning_data_10.mat and Data/tuning_data_5.mat.
2. Battery current is an input (feature 'batI_avg' = 120 s moving average in MATLAB,
   matching the 120 s window mean used here). get_tuneS.m computes it.
3. Small network (1 hidden layer) + weight decay, trained as an ensemble of
   ENSEMBLE networks that is merged into ONE equivalent network for MATLAB.
   With only 60 labels a big network just memorises noise.
4. Labels that sit at the min/max of their identified range (poorly identified)
   get a low weight BOUND_WEIGHT in the loss.
5. Honest validation: blocked cross-validation over contiguous groups of windows
   (both temperatures), compared with a Python copy of prepParaVec (interpolation)
   and ridge regression. Lower = better.

Run (from the SSY226-Group2 folder):
    python train_gamma_nn_v2.py
Output:
    nn_george2.mat           -> use with SCHED_MODE = 'george2' / validate_com_fig2({'interp','george','george2'})
    gamma_nn_v2_report.txt   cross-validation results
    gamma_nn_v2_fit.png      identified vs predicted, per parameter
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

import export_nn_for_matlab as exporter
from export_nn_for_matlab import export_nn_for_matlab, matlab_forward

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.set_default_dtype(torch.float64)

# ----------------------------------------------------------------------------- settings
NAME = "george2"                       # -> nn_george2.mat, SCHED_MODE = 'george2'
TV_FILE = "tunedVariables_minus5_minus10.mat"
REF_FILES = {-10: "tuning_data_10.mat", -5: "tuning_data_5.mat"}
STATE_NAMES = ["motT", "invT", "dcdcT", "SOC", "batT", "p_in", "p_out", "cabIntT", "cabAirT"]
# Inputs (names understood by get_tuneS.m). SOC is left out on purpose: in these runs it
# mostly encodes "time since start of this drive cycle", which does not generalise.
FEATURES = ["Tamb_C", "motT", "invT", "dcdcT", "batT", "p_in", "p_out", "cabIntT", "cabAirT", "batI_avg"]
IDX_STATE = [8, 9, 5, 2, 1, 3, 5, 9, 6, 7, 6]   # prepParaVec's driving state per parameter (1-based)

HIDDEN = 8           # neurons per ensemble member (one hidden layer, tanh)
ENSEMBLE = 10        # members, merged into one network of HIDDEN*ENSEMBLE neurons
WEIGHT_DECAY = 3e-2  # L2 penalty on the weights (strong: only 60 labels)
EPOCHS, LR = 3000, 1e-2
BOUND_WEIGHT = 0.2   # loss weight of labels sitting at their min/max
CV_BLOCK = 3         # windows per cross-validation block (3 x 120 s = 6 min)
SEED = 0


# ----------------------------------------------------------------------------- data
def find(name):
    for p in [name, os.path.join("Data", name)]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found here or in Data/")


def load_windows():
    """One row per identified window: features (window means) and gamma(j)."""
    tv = loadmat(find(TV_FILE), squeeze_me=True, struct_as_record=False)["tunedVariables"]
    P_all = np.atleast_2d(np.asarray(tv.parameters, float))
    S_all = np.atleast_2d(np.asarray(tv.states, float))
    A_all = np.asarray(tv.ambient, float).ravel()
    wl = int(np.asarray(tv.meta.windowLength).ravel()[0])

    rows, labels, info = [], [], []
    for T, f in REF_FILES.items():
        r = loadmat(find(f), squeeze_me=True, struct_as_record=False)["resampled"]
        g = lambda n: np.asarray(getattr(r, n), float).ravel()
        # measured reference states in MPC units (K, Pa, -), same order as the COM
        X = np.c_[g("motTmeasC") + 273.15, g("invTmeasC") + 273.15, g("dcdcTmeasC") + 273.15,
                  g("batSocMeas"), g("batTmeasC") + 273.15, g("compPinMpa") * 1e6,
                  g("compPoutMpa") * 1e6, g("cabIntTInK"), g("cabAirTmeasC") + 273.15]
        I = g("batImeas")
        P = P_all[A_all == T]
        n_win = len(P) // wl
        for j in range(n_win):
            sl = slice(j * wl, (j + 1) * wl)
            lab = P[sl]
            if np.abs(lab - lab[0]).max() > 1e-9:
                raise ValueError(f"T={T}, window {j}: parameters not constant inside the window")
            xm = X[sl].mean(0)
            base = dict(zip(STATE_NAMES, xm))
            base.update(Tamb_C=float(T), batI_avg=I[sl].mean())
            rows.append([base[k] for k in FEATURES])
            labels.append(lab[0])
            info.append((T, j))
    print(f"Loaded {len(rows)} identification windows ({wl} s each) at T_amb = {sorted(REF_FILES)} degC")
    return np.array(rows), np.array(labels), np.array(info), (S_all, P_all, A_all, wl)


def interp_baseline(S_tr, P_tr, A_tr, S_q, A_q):
    """Python copy of prepParaVec: per parameter, 2-D linear interpolation over
    (T_amb, one driving state) on the per-second rows; nearest outside."""
    out = np.zeros((len(S_q), 11))
    for i in range(11):
        pts = np.c_[A_tr, S_tr[:, IDX_STATE[i] - 1]]
        q = np.c_[A_q, S_q[:, IDX_STATE[i] - 1]]
        lin = LinearNDInterpolator(pts, P_tr[:, i])(q)
        near = NearestNDInterpolator(pts, P_tr[:, i])(q)
        out[:, i] = np.where(np.isfinite(lin), lin, near)
    return out


# ----------------------------------------------------------------------------- models
class Scaler:
    def __init__(self, Z):
        self.m, self.s = Z.mean(0), Z.std(0)
        self.s[self.s < 1e-9] = 1.0


def fit_ensemble(X, LP, W, seed):
    """Train ENSEMBLE small networks on log-parameters; return list of models + scalers."""
    xs, ys = Scaler(X), Scaler(LP)
    Xt = torch.tensor((X - xs.m) / xs.s)
    Yt = torch.tensor((LP - ys.m) / ys.s)
    Wt = torch.tensor(W)
    models = []
    for e in range(ENSEMBLE):
        torch.manual_seed(seed * 1000 + e)
        m = nn.Sequential(nn.Linear(X.shape[1], HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, LP.shape[1]))
        opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for _ in range(EPOCHS):
            loss = (Wt * (m(Xt) - Yt) ** 2).sum() / Wt.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        models.append(m)
    return models, xs, ys


def merge_ensemble(models):
    """Average of K one-hidden-layer networks == one network with K*H hidden neurons."""
    K = len(models)
    H = models[0][0].out_features
    big = nn.Sequential(nn.Linear(models[0][0].in_features, K * H), nn.Tanh(),
                        nn.Linear(K * H, models[0][2].out_features))
    with torch.no_grad():
        big[0].weight.copy_(torch.cat([m[0].weight for m in models], 0))
        big[0].bias.copy_(torch.cat([m[0].bias for m in models], 0))
        big[2].weight.copy_(torch.cat([m[2].weight for m in models], 1) / K)
        big[2].bias.copy_(torch.stack([m[2].bias for m in models], 0).mean(0))
    return big


def predict(net, xs, ys, X, lo, hi):
    with torch.no_grad():
        z = net(torch.tensor((X - xs.m) / xs.s)).numpy()
    return np.clip(np.exp(z * ys.s + ys.m), lo, hi)


def fit_ridge(X, P, W, lam=1.0):
    xs = Scaler(X)
    Xa = np.c_[np.ones(len(X)), (X - xs.m) / xs.s]
    B = np.zeros((Xa.shape[1], P.shape[1]))
    for i in range(P.shape[1]):              # weighted ridge, one parameter at a time
        w = W[:, i]
        B[:, i] = np.linalg.solve(Xa.T @ (w[:, None] * Xa) + lam * np.eye(Xa.shape[1]), Xa.T @ (w * P[:, i]))
    return lambda Xq: np.c_[np.ones(len(Xq)), (Xq - xs.m) / xs.s] @ B


# ----------------------------------------------------------------------------- main
def main():
    np.random.seed(SEED)
    X, P, info, (S_all, P_all, A_all, wl) = load_windows()
    n = len(X)
    lo, hi = P.min(0), P.max(0)
    span = hi - lo
    # parameters with only one value per temperature (e.g. p6, p8) are not "at a bound":
    # every label is at the min or max simply because there are two temperatures
    n_unique = np.array([len(np.unique(np.round(P[:, i], 9))) for i in range(P.shape[1])])
    graded = (span > 1e-9) & (n_unique > len(REF_FILES))
    at_bound = (np.isclose(P, lo) | np.isclose(P, hi)) & graded
    W = np.where(at_bound, BOUND_WEIGHT, 1.0)
    LP = np.log(np.maximum(P, 1e-6))

    # ---- blocked cross-validation: hold out CV_BLOCK consecutive windows at BOTH temperatures
    block = info[:, 1] // CV_BLOCK
    folds = np.unique(block)
    pred = {k: np.zeros_like(P) for k in ["interp", "ridge", "nn"]}
    # window index (within its temperature) of every per-second identification row
    row_win = np.zeros(len(A_all), int)
    for T in REF_FILES:
        k = np.where(A_all == T)[0]
        row_win[k] = np.arange(len(k)) // wl
    print(f"Cross-validation: {len(folds)} folds, each holds out {CV_BLOCK} windows per temperature ...")
    for f in folds:
        te = block == f
        tr = ~te
        # interpolation sees the per-second rows of the training windows, like prepParaVec
        rmask = (row_win // CV_BLOCK) != f
        # and is queried at the window-mean of the identification states
        Sq = np.array([S_all[(A_all == T) & (row_win == j)].mean(0) for T, j in info[te]])
        pred["interp"][te] = interp_baseline(S_all[rmask], P_all[rmask], A_all[rmask], Sq, info[te, 0].astype(float))
        pred["ridge"][te] = np.clip(fit_ridge(X[tr], P[tr], W[tr])(X[te]), lo, hi)
        models, xs, ys = fit_ensemble(X[tr], LP[tr], W[tr], seed=int(f))
        pred["nn"][te] = predict(merge_ensemble(models), xs, ys, X[te], lo, hi)

    sd = P.std(0)
    varying = sd > 1e-9
    nrmse = {k: np.where(varying, np.sqrt(((v - P) ** 2).mean(0)) / np.where(varying, sd, 1), np.nan)
             for k, v in pred.items()}
    lines = [f"Parameter scheduler v2: {n} identification windows, inputs = {FEATURES}",
             f"Blocked cross-validation ({len(folds)} folds of {CV_BLOCK} consecutive windows per temperature).",
             "Normalised RMSE on held-out windows = RMSE / std of the parameter (lower is better; 1.0 = as bad as",
             "always predicting the mean).",
             f"{'param':6s}{'interp':>10s}{'ridge':>10s}{'NN v2':>10s}{'labels at bound':>18s}"]
    for i in range(11):
        if not varying[i]:
            lines.append(f"p{i+1:<5d}{'constant (' + format(P[0, i], '.4g') + ')':>30s}")
            continue
        tag = f"{100 * at_bound[:, i].mean():16.0f} %" if graded[i] else "   one value per temperature"
        lines.append(f"p{i+1:<5d}{nrmse['interp'][i]:10.3f}{nrmse['ridge'][i]:10.3f}{nrmse['nn'][i]:10.3f}{tag}")
    lines.append(f"{'mean':6s}{np.nanmean(nrmse['interp']):10.3f}{np.nanmean(nrmse['ridge']):10.3f}"
                 f"{np.nanmean(nrmse['nn']):10.3f}")
    lines.append("Labels at their min/max were down-weighted (weight %.1f) in training, but all labels count here." % BOUND_WEIGHT)
    lines.append("Note: this measures parameter error, not model-prediction error; the real test is")
    lines.append("validate_com_fig2 (Fig. 2 / Table 1), especially at -7 degC which no scheduler has seen.")
    rep = "\n".join(lines)
    print("\n" + rep)
    open("gamma_nn_v2_report.txt", "w").write(rep + "\n")

    # ---- final network on all 60 windows
    print("\nTraining the final network on all windows ...")
    models, xs, ys = fit_ensemble(X, LP, W, seed=SEED)
    net = merge_ensemble(models)
    P_fit = predict(net, xs, ys, X, lo, hi)

    fig, ax = plt.subplots(3, 4, figsize=(15, 9))
    for i, a in enumerate(ax.flat[:11]):
        for T, mk in [(-10, "o"), (-5, "s")]:
            k = info[:, 0] == T
            a.scatter(P[k, i], pred["nn"][k, i], s=18, marker=mk, label=f"{T} degC, held out (CV)")
        a.plot([lo[i], hi[i]], [lo[i], hi[i]], "k--", lw=0.8)
        a.set_title(f"p{i+1}: identified vs NN v2", fontsize=9)
    ax.flat[0].legend(fontsize=7); ax.flat[-1].axis("off")
    fig.tight_layout(); fig.savefig("gamma_nn_v2_fit.png", dpi=110)

    # ---- export for MATLAB (get_tuneS.m), with OOD envelope and output clamp
    if "batI_avg" not in exporter.ALLOWED:
        exporter.ALLOWED.append("batI_avg")
    fspan = X.max(0) - X.min(0)
    env_lo, env_hi = X.min(0) - 0.25 * fspan, X.max(0) + 0.25 * fspan
    path = f"nn_{NAME}.mat"
    export_nn_for_matlab(net, FEATURES, xs.m, xs.s, path, y_mean=ys.m, y_std=ys.s, out_transform="exp",
                         p_lo=lo, p_hi=hi, env_lo=env_lo, env_hi=env_hi)

    # check: the MATLAB forward formula reproduces the PyTorch network
    d = loadmat(path, squeeze_me=False)
    d = {k: (v if not isinstance(v, np.ndarray) or v.dtype.kind != "U" else str(v.ravel()[0]))
         for k, v in d.items() if not k.startswith("__")}
    d["act"] = "tanh"; d["out_transform"] = "exp"
    d["n_layers"] = float(np.asarray(d["n_layers"]).ravel()[0])
    err = max(np.abs(matlab_forward(d, X[k]) - P_fit[k]).max() for k in range(n))
    print(f"Check: MATLAB forward formula reproduces the network to {err:.1e}")

    # ---- sanity check at -7 degC (unseen): inside the training envelope? parameters between
    #      the -10 and -5 degC values?  (no labels exist at -7 degC, so this is not an accuracy test)
    try:
        r = loadmat(find("tuning_data_7.mat"), squeeze_me=True, struct_as_record=False)["resampled"]
        g = lambda nm: np.asarray(getattr(r, nm), float).ravel()
        X7 = np.c_[g("motTmeasC") + 273.15, g("invTmeasC") + 273.15, g("dcdcTmeasC") + 273.15,
                   g("batSocMeas"), g("batTmeasC") + 273.15, g("compPinMpa") * 1e6,
                   g("compPoutMpa") * 1e6, g("cabIntTInK"), g("cabAirTmeasC") + 273.15]
        I7 = g("batImeas")
        rows7 = []
        for j in range(len(I7) // wl):
            sl = slice(j * wl, (j + 1) * wl)
            base = dict(zip(STATE_NAMES, X7[sl].mean(0)))
            base.update(Tamb_C=-7.0, batI_avg=I7[sl].mean())
            rows7.append([base[k] for k in FEATURES])
        rows7 = np.array(rows7)
        ood = ((rows7 < env_lo) | (rows7 > env_hi)).any(1)
        P7 = predict(net, xs, ys, rows7, lo, hi)
        # same -7 degC operating points, but T_amb set to -10 / -5: a smooth scheduler should
        # give values in between
        Pa = predict(net, xs, ys, np.c_[np.full(len(rows7), -10.0), rows7[:, 1:]], lo, hi)
        Pb = predict(net, xs, ys, np.c_[np.full(len(rows7), -5.0), rows7[:, 1:]], lo, hi)
        inside = (P7 >= np.minimum(Pa, Pb) - 1e-6) & (P7 <= np.maximum(Pa, Pb) + 1e-6)
        msg = [f"\n-7 degC check on the unseen run ({len(rows7)} windows): "
               f"{ood.sum()} outside the training envelope (those would fall back to interpolation)",
               "Same operating points evaluated with T_amb = -10 / -7 / -5 degC (mean over windows);",
               "the -7 value should lie between the other two.",
               "  param   NN(T=-10)   NN(T=-7)   NN(T=-5)   windows in between"]
        for i in range(11):
            msg.append(f"  p{i+1:<5d}{Pa[:, i].mean():11.3f}{P7[:, i].mean():11.3f}{Pb[:, i].mean():11.3f}"
                       f"{100 * inside[:, i].mean():17.0f} %")
        print("\n".join(msg))
        open("gamma_nn_v2_report.txt", "a").write("\n".join(msg) + "\n")
    except FileNotFoundError:
        pass
    print(f"\nDone. Use it in MATLAB with:  validate_com_fig2({{'interp','george','{NAME}'}})")


if __name__ == "__main__":
    main()
