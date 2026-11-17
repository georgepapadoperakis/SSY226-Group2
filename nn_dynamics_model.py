"""
nn_dynamics_model.py  -  Option B: a neural network that REPLACES the control-oriented model

No gamma, no labels: the network learns the thermal dynamics directly from the Simscape
runs, so every tuning_data_*.mat file can be used.

    x(k+1) = x(k) + NN( x(k), u(k), d(k) )          one step = 1 s, as in the MPC
    SOC(k+1) = SOC(k) - I_bat(k) * dt / Q           exact physics, no network needed

    x  (8)  motor, inverter, DC-DC, battery, p_in, p_out, cabin interior, cabin air
    u  (6)  compressor speed, blower flow, motor pump, battery pump, heater power, radiator fan
    d  (6)  ambient temperature, vehicle speed, battery current, motor / inverter / DC-DC heat

Training is MULTI-STEP: the network rolls forward on its own predictions (up to 120 s) and is
penalised on the whole trajectory, so it stays accurate over the MPC horizon (30 s) instead of
drifting. Whole temperatures are held out for testing.

Run (from the project folder):   python nn_dynamics_model.py
Outputs in nn_dynamics_output/:
    results.txt                 free-run (Table-1 style) and 30 s horizon errors per run
    freerun_T<..>.png           Fig.-2 style plots, network vs Simscape, for each test run
    nn_dynamics.pt              trained model (PyTorch)
    nn_dynamics_weights.mat     weights + normalisation, for building the model in CasADi/MATLAB
and in the project folder:
    nn_dynamics.mat             same weights; validate_com_fig2({'interp','dynamics'}) uses it
Runtime: about 5-10 minutes on a laptop CPU.
"""
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat, savemat
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.set_default_dtype(torch.float64)

# ----------------------------------------------------------------------------- settings
RUN_PATTERN = "tuning_data_*.mat"         # searched in this folder and in Data/
TEST_TEMPS = [-12]                        # held out completely (never seen in training)
EXCLUDE_TEMPS = []                        # e.g. [-25, -30] to leave runs out entirely
OUT = "nn_dynamics_output"
HIDDEN = 64
CURRICULUM = [(1, 1500), (10, 1500), (30, 1500), (120, 1000)]   # (rollout length s, iterations)
BATCH = 64
LR = 1e-3
H_MPC = 30
SEED = 0

STATES = ["motT", "invT", "dcdcT", "batT", "p_in", "p_out", "cabIntT", "cabAirT"]
STATE_LABEL = ["Motor", "Inverter", "DC-DC", "Battery", "p_in", "p_out", "Cabin interior", "Cabin air"]
STATE_UNIT = ["degC"] * 4 + ["bar"] * 2 + ["degC"] * 2
INPUTS = ["compN", "blower", "motPump", "batPump", "heater", "fan"]
DISTS = ["Tamb", "speed", "batI", "motQ", "invQ", "dcdcQ"]


# ----------------------------------------------------------------------------- data
def load_run(f):
    """Returns dict with X (N x 8), SOC (N), U (N x 6), D (N x 6), or None if a signal is missing."""
    m = loadmat(f, squeeze_me=True, struct_as_record=False)
    T = int(np.asarray(m["envTemp"]).ravel()[0])
    r = m["resampled"]
    has = lambda n: hasattr(r, n)
    g = lambda n: np.asarray(getattr(r, n), float).ravel()
    if not has("radFanrpm"):
        return T, None, ("no radiator fan signal (radFanrpm). This is an original-format file: "
                         "re-run it with the same export as the new runs to include it")
    X = np.c_[g("motTmeasC"), g("invTmeasC"), g("dcdcTmeasC"), g("batTmeasC"),
              g("compPinMpa") * 10, g("compPoutMpa") * 10,          # MPa -> bar
              g("cabIntTInK") - 273.15, g("cabAirTmeasC")]
    heater = g("heaterQ") if has("heaterQ") else 4000.0 * g("heaterCmd")
    batpump = g("batPumpNrpm") if has("batPumpNrpm") else g("motPumpNrpm")
    U = np.c_[g("compNrpm"), g("cabAirMfIn"), g("motPumpNrpm"), batpump, heater, g("radFanrpm")]
    D = np.c_[np.full(len(X), float(T)), g("vehSpeedMs"), g("batImeas"),
              g("motQmeas"), g("invQmeas"), g("dcdcQmeas")]
    return T, dict(X=X, SOC=g("batSocMeas"), U=U, D=D, t=g("t_resampled"), file=f), None


def load_all():
    files = sorted(set(os.path.normpath(p) for p in glob.glob(RUN_PATTERN) + glob.glob(os.path.join("Data", RUN_PATTERN))))
    runs, skipped = {}, {}
    for f in files:
        T, run, why = load_run(f)
        if T in EXCLUDE_TEMPS:
            skipped[T] = "excluded (EXCLUDE_TEMPS)"
        elif run is None:
            skipped[T] = why
        else:
            runs[T] = run
    if not runs:
        raise FileNotFoundError(f"No usable {RUN_PATTERN} found here or in Data/")
    return runs, skipped


# ----------------------------------------------------------------------------- model
class Norm:
    def __init__(self, Z):
        self.m = torch.tensor(Z.mean(0))
        self.s = torch.tensor(np.maximum(Z.std(0), 1e-6))


class DynamicsNN(nn.Module):
    """x(k+1) = x(k) + dx_scale * MLP(normalised [x, u, d])"""

    def __init__(self, nx, nu, nd, xn, un, dn, dxs, hidden=HIDDEN):
        super().__init__()
        self.xn, self.un, self.dn = xn, un, dn
        self.register_buffer("dxs", dxs)
        self.net = nn.Sequential(nn.Linear(nx + nu + nd, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, nx))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)          # starts as "nothing changes"

    def step(self, x, u, d):
        z = torch.cat([(x - self.xn.m) / self.xn.s, (u - self.un.m) / self.un.s, (d - self.dn.m) / self.dn.s], -1)
        return x + self.dxs * self.net(z)

    def rollout(self, x0, U, D):
        """x0 (B,8), U (B,H,6), D (B,H,6) -> (B,H,8) predicted states at k+1..k+H"""
        x, out = x0, []
        for k in range(U.shape[1]):
            x = self.step(x, U[:, k], D[:, k])
            out.append(x)
        return torch.stack(out, 1)


def sample_windows(runs, temps, H, n, rng):
    X0, U, D, Y = [], [], [], []
    for _ in range(n):
        r = runs[temps[rng.integers(len(temps))]]
        k = rng.integers(0, len(r["X"]) - H - 1)
        X0.append(r["X"][k]); U.append(r["U"][k:k + H]); D.append(r["D"][k:k + H]); Y.append(r["X"][k + 1:k + H + 1])
    t = lambda a: torch.tensor(np.array(a))
    return t(X0), t(U), t(D), t(Y)


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    log = open(os.path.join(OUT, "results.txt"), "w")

    def say(s=""):
        print(s); log.write(s + "\n"); log.flush()

    t0 = time.time()
    runs, skipped = load_all()
    test = [T for T in TEST_TEMPS if T in runs]
    train = [T for T in sorted(runs) if T not in test]
    say(f"Training runs: {', '.join(f'{T:+d}' for T in train)} degC")
    say(f"Test runs (never seen in training): {', '.join(f'{T:+d}' for T in test) or '-'} degC")
    for T, why in sorted(skipped.items()):
        say(f"Skipped {T:+d} degC: {why}")
    if not train:
        raise ValueError("No training runs left")

    # ---- normalisation from training runs only
    cat = lambda key: np.vstack([runs[T][key] for T in train])
    dX = np.vstack([np.diff(runs[T]["X"], axis=0) for T in train])
    xn, un, dn = Norm(cat("X")), Norm(cat("U")), Norm(cat("D"))
    dxs = torch.tensor(np.maximum(dX.std(0), 1e-6))
    x_std = torch.tensor(cat("X").std(0))           # loss weighting: every state counts equally

    # ---- SOC by Coulomb counting: fit 1/capacity on training runs
    dsoc = np.concatenate([np.diff(runs[T]["SOC"]) for T in train])
    I = np.concatenate([runs[T]["D"][:-1, 2] for T in train])
    k_soc = -float(dsoc @ I / (I @ I))              # SOC(k+1) = SOC(k) - k_soc * I(k)
    say(f"SOC model: SOC(k+1) = SOC(k) - {k_soc:.3e} * I_bat   (capacity = {1 / k_soc / 3600:.1f} Ah)")

    model = DynamicsNN(len(STATES), len(INPUTS), len(DISTS), xn, un, dn, dxs)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- multi-step training with a growing rollout length
    say("\nTraining (rollout length grows so the model stays stable over long horizons) ...")
    for H, iters in CURRICULUM:
        for it in range(iters):
            X0, U, D, Y = sample_windows(runs, train, H, BATCH, rng)
            loss = (((model.rollout(X0, U, D) - Y) / x_std) ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        say(f"   rollout {H:4d} s: final loss {loss.item():.5f}   ({time.time() - t0:.0f} s)")
        for g_ in opt.param_groups:
            g_["lr"] *= 0.6

    # ---- evaluation
    def free_run(r):
        with torch.no_grad():
            X = model.rollout(torch.tensor(r["X"][:1]), torch.tensor(r["U"][None, :-1]),
                              torch.tensor(r["D"][None, :-1]))[0].numpy()
        return np.vstack([r["X"][:1], X])

    def horizon_err(r, H=H_MPC, stride=10):
        ks = np.arange(0, len(r["X"]) - H - 1, stride)
        X0 = torch.tensor(r["X"][ks])
        U = torch.tensor(np.stack([r["U"][k:k + H] for k in ks]))
        D = torch.tensor(np.stack([r["D"][k:k + H] for k in ks]))
        Y = np.stack([r["X"][k + 1:k + H + 1] for k in ks])
        with torch.no_grad():
            P = model.rollout(X0, U, D).numpy()
        hold = np.repeat(r["X"][ks][:, None], H, 1)      # baseline: "nothing changes"
        rm = lambda A: np.sqrt(((A - Y) ** 2).mean(axis=(0, 1)))
        return rm(P), rm(hold)

    say("\n[1] Free run over the full 3600 s drive, from the initial state only (same test as the")
    say("    paper's Table 1). RMSE / MAE; temperatures in degC, pressures in bar.")
    say("    " + f"{'run':>10s}" + "".join(f"{s[:9]:>15s}" for s in STATE_LABEL))
    FR = {}
    for T in train + test:
        r = runs[T]
        Xp = free_run(r)
        FR[T] = Xp
        e = Xp - r["X"]
        cells = "".join(f"{np.sqrt((e[:, i] ** 2).mean()):7.2f}/{np.abs(e[:, i]).mean():<6.2f}  " for i in range(len(STATES)))
        tag = "TEST " if T in test else "train"
        say(f"    {tag} {T:+4d}  " + cells)
    say("    For orientation only (different temperatures and controller settings): the paper's physics")
    say("    model reaches RMSE ~1.0-1.5 degC (cabin air) and ~1.5-1.9 degC (battery) at -5/-7/-10 degC.")

    say(f"\n[2] {H_MPC} s prediction horizon (what the MPC uses), restarted every 10 s from the true state.")
    say("    RMSE of the network vs. 'nothing changes' (a model that is not better than that is useless).")
    say("    " + f"{'run':>10s}" + "".join(f"{s[:9]:>15s}" for s in STATE_LABEL))
    for T in train + test:
        nn_e, hold_e = horizon_err(runs[T])
        tag = "TEST " if T in test else "train"
        say(f"    {tag} {T:+4d}  " + "".join(f"{a:7.3f}/{b:<6.3f}  " for a, b in zip(nn_e, hold_e)))
    say("    (each cell: network / nothing-changes)")

    # ---- plots for the test runs (Fig. 2 style)
    for T in test or train[:1]:
        r, Xp = runs[T], FR[T]
        fig, ax = plt.subplots(2, 3, figsize=(14, 7), facecolor="#fcfcfb")
        for a, i in zip(ax.flat, [7, 3, 0, 6, 4, 5]):
            a.set_facecolor("#fcfcfb")
            a.plot(r["t"], r["X"][:, i], color="#3d3d3a", ls="--", lw=1.5, label="Simscape reference")
            a.plot(r["t"], Xp[:, i], color="#2a78d6", lw=1.8, label="Neural network (free run)")
            a.set_title(f"{STATE_LABEL[i]} ({STATE_UNIT[i]})", loc="left", fontsize=10)
            a.grid(True, color="#e6e5e1", lw=0.6); a.spines[["top", "right"]].set_visible(False)
        for a in ax[1]:
            a.set_xlabel("Time (s)")
        ax[0, 0].legend(fontsize=8, frameon=False)
        kind = "held-out test run, never seen in training" if T in test else "training run"
        fig.suptitle(f"{T:+d} degC ({kind}): free run over the full drive from the initial state",
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"freerun_T{T}.png"), dpi=130); plt.close(fig)

    # ---- save
    torch.save({"state_dict": model.state_dict(), "k_soc": k_soc, "train": train, "test": test}, os.path.join(OUT, "nn_dynamics.pt"))
    lin = [m for m in model.net if isinstance(m, nn.Linear)]
    w = {f"W{i+1}": l.weight.detach().numpy() for i, l in enumerate(lin)}
    w.update({f"b{i+1}": l.bias.detach().numpy()[:, None] for i, l in enumerate(lin)})
    w.update(x_mean=xn.m.numpy()[:, None], x_std=xn.s.numpy()[:, None], u_mean=un.m.numpy()[:, None],
             u_std=un.s.numpy()[:, None], d_mean=dn.m.numpy()[:, None], d_std=dn.s.numpy()[:, None],
             dx_scale=dxs.numpy()[:, None], k_soc=k_soc, act="tanh",
             states=np.array(STATES, dtype=object), inputs=np.array(INPUTS, dtype=object),
             dists=np.array(DISTS, dtype=object),
             units="states: degC (temperatures), bar (pressures); u: rpm, kg/s, rpm, rpm, W, rpm; "
                   "d: degC, m/s, A, W, W, W",
             formula="x_next = x + dx_scale .* (W3*tanh(W2*tanh(W1*z+b1)+b2)+b3), "
                     "z = [(x-x_mean)./x_std; (u-u_mean)./u_std; (d-d_mean)./d_std]")
    savemat(os.path.join(OUT, "nn_dynamics_weights.mat"), w)
    savemat("nn_dynamics.mat", w)     # project folder: validate_com_fig2 finds it as mode 'dynamics'

    say(f"\nSaved model, weights and plots in {OUT}/  (total {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
