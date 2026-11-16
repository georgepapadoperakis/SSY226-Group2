"""
Learned parameter scheduling for a control-oriented thermal model (COM)
=======================================================================
SSY226 project: "Can a neural network replace offline system identification
in vehicle thermal-management MPC?"

What this script does (all from the rule-based Simscape runs at -5, -7, -10 degC):
  1. Builds a 9-state control-oriented model (COM) of the EV heat-pump system
     with 11 physical scaling parameters (theta, nominal = 1).
  2. Identifies the nominal COM on the training runs (-5 and -10 degC).
  3. Baseline (current practice): identifies ONE constant theta per ambient
     temperature by least-squares fitting of simulated trajectories, and gets
     -7 degC by linear interpolation between -5 and -10 degC.
  4. Proposed: trains a neural network that maps the measurable operating point
     (no time / cycle features) to theta, end-to-end on the COM prediction error.
     theta is held constant over the prediction horizon, as the MPC would use it.
  5. Tests everything on the held-out -7 degC run, for the MPC horizon (30 s)
     and a longer horizon (300 s); adds an out-of-distribution check with
     fallback to the interpolated parameters.

Run:  python com_parameter_scheduler.py      (needs the three *_export.mat files)
Outputs in com_output/: results.txt, plots, scheduler weights (.pt and .mat).
Runtime: about 10 minutes on a laptop CPU.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat, savemat
from scipy.optimize import nnls
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
torch.set_default_dtype(torch.float64)

RUN_FILES = {"T5": "simlog_rule_T5_export.mat", "T7": "simlog_rule_T7_export.mat", "T10": "simlog_rule_T10_export.mat"}
T_AMB = {"T5": -5.0, "T7": -7.0, "T10": -10.0}
TRAIN, TEST = ["T5", "T10"], "T7"
H_MPC, H_LONG, H_TRAIN = 30, 300, 120
OUT = "com_output"
SEED = 0

# =============================================================================
# 1. Data
# =============================================================================
K0 = 273.15
SIG = {
    "T_cab": "simlog.Cabin.Cabin_Air_Volume.T_I",
    "T_glass": "simlog.Cabin.Cabin_Heat_Transfer.Glass_Thermal_Mass.M.T",
    "T_doors": "simlog.Cabin.Cabin_Heat_Transfer.Doors_Thermal_Mass.M.T",
    "T_roof": "simlog.Cabin.Cabin_Heat_Transfer.Roof_Thermal_Mass.M.T",
    "T_sup": "simlog.Cabin.Pressure_Temperature_Sensor_MA.T",
    "T_bat1": "simlog.Battery.Pack_1.Thermal_Model.Battery_Thermal_Mass.T",
    "T_bat2": "simlog.Battery.Pack_2.Thermal_Model.Battery_Thermal_Mass.T",
    "T_bat3": "simlog.Battery.Pack_3.Thermal_Model.Battery_Thermal_Mass.T",
    "T_bat4": "simlog.Battery.Pack_4.Thermal_Model.Battery_Thermal_Mass.T",
    "T_mot": "simlog.Motor.Motor.M.T",
    "T_inv": "simlog.Inverter.Inverter.M.T",
    "T_dcdc": "simlog.DCDC.DC_DC_Converter.M.T",
    "T_htr": "simlog.Heater.Battery_Heater.M.T",
    "T_cool_C": "simlog.Heater.Heater_Coolant_Volume.T_I",
    "T_amb": "simlog.Cabin.Environment_Air.T",
    "w_comp": "simlog.Compressor.Compressor.omega",
    "V_blow": "simlog.Blower.Blower.V",
    "Q_htr": "simlog.Heater.Controlled_Heat_Flow_Rate_Source.S",
    "Q_mot": "simlog.Motor.Controlled_Heat_Flow_Rate_Source.S",
    "Q_inv": "simlog.Inverter.Controlled_Heat_Flow_Rate_Source.S",
    "Q_dcdc": "simlog.DCDC.Controlled_Heat_Flow_Rate_Source.S",
    "Q_bat1": "simlog.Battery.Pack_1.Thermal_Model.Controlled_Heat_Flow_Rate_Source.S",
    "Q_bat2": "simlog.Battery.Pack_2.Thermal_Model.Controlled_Heat_Flow_Rate_Source.S",
    "Q_bat3": "simlog.Battery.Pack_3.Thermal_Model.Controlled_Heat_Flow_Rate_Source.S",
    "Q_bat4": "simlog.Battery.Pack_4.Thermal_Model.Controlled_Heat_Flow_Rate_Source.S",
    "m_ram": "simlog.Condenser_Evaporator.Flow_Rate_Sensor_MA1.M",
    "p_suc": "simlog.Compressor.Compressor.A.p",
    "p_dis": "simlog.Compressor.Compressor.B.p",
    "Q_ic": "simlog.Inner_Condenser.Inner_Condenser.moist_air_2.Q",
    "Q_hex": "simlog.Heat_Exchanger.Heat_Exchanger.two_phase_fluid_2.Q",
}
STATES = ["T_cab", "T_sh", "T_sup", "T_bat", "T_mot", "T_inv", "T_dcdc", "T_htr", "T_cool"]
STATE_LABELS = ["Cabin air", "Cabin shell", "Supply air", "Battery", "Motor", "Inverter", "DC-DC", "Heater", "Coolant"]
INPUTS = ["T_amb", "w", "m_air", "Q_htr", "Q_mot", "Q_inv", "Q_dcdc", "Q_bat"]


def load_run(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found in {os.getcwd()} (run export_simlog_runs in MATLAB first)")
    m = loadmat(path, squeeze_me=True)
    names = [str(n) for n in m["names"]]
    g = {k: m["data"][:, names.index(v)].astype(np.float64) for k, v in SIG.items()}
    r = {"t": np.asarray(m["time"], float)}
    r["T_cab"], r["T_sup"] = g["T_cab"], g["T_sup"]
    r["T_sh"] = (g["T_glass"] + g["T_doors"] + g["T_roof"]) / 3
    r["T_bat"] = (g["T_bat1"] + g["T_bat2"] + g["T_bat3"] + g["T_bat4"]) / 4
    for k in ["T_mot", "T_inv", "T_dcdc", "T_htr", "T_amb", "Q_htr", "Q_mot", "Q_inv", "Q_dcdc", "m_ram", "p_suc", "p_dis"]:
        r[k] = g[k]
    r["T_cool"] = g["T_cool_C"] + K0
    r["w"] = g["w_comp"] / 1000.0                      # krpm
    r["m_air"] = 1.2 * g["V_blow"]                     # kg/s
    r["Q_bat"] = g["Q_bat1"] + g["Q_bat2"] + g["Q_bat3"] + g["Q_bat4"]
    r["Q_ic"], r["Q_hex"] = 1000 * g["Q_ic"], 1000 * g["Q_hex"]   # W
    r["X"] = np.stack([r[s] for s in STATES], 1)
    return r


# =============================================================================
# 2. Control-oriented model (9 states, 11 scaling parameters)
# =============================================================================
THETA_NAMES = ["UA cabin-shell", "UA shell-ambient", "Heat pump capacity", "Waste-heat recovery",
               "UA battery-coolant", "UA motor-coolant", "UA inverter/DCDC-coolant", "UA heater-coolant",
               "Heater efficiency", "Battery heat generation", "UA coolant-ambient"]
N_THETA = len(THETA_NAMES)
CP_AIR, Q_OCC = 1006.0, 70.0
NOMINAL_INIT = dict(C_cab=8.0e3, C_sh=5.8e4, tau_sup=10.0, C_bat=2.0e5, C_mot=4.35e4, C_inv=3.9e3,
                    C_dcdc=5.2e2, C_htr=9.2e2, C_cool=2.15e4, UA_cab_sh=42.0, UA_sh_amb=51.0,
                    UA_cab_amb=47.0, UA_bat=1.03e3, UA_mot=3.0e2, UA_inv=26.0, UA_dcdc=55.0,
                    UA_htr=2.0e2, UA_cool_amb=5.0)


def sat(w):
    return w / (w + 0.1)


def basis_ic(w, Tc, Tcab, Ta):
    s = sat(w)
    return [s, s * w, s * (Tc - Ta) / 10, s * (Tcab - Ta) / 10]


def basis_hex(w, Tc, Ta):
    s = sat(w)
    return [s * (Tc - Ta) / 10, s * w * (Tc - Ta) / 10]


class COM(nn.Module):
    """Energy-conserving lumped thermal model; Euler integration, 1 s sample time."""

    def __init__(self, c_ic, c_hex, dt=1.0, substeps=2):
        super().__init__()
        self.names = list(NOMINAL_INIT)
        self.logp = nn.Parameter(torch.log(torch.tensor([NOMINAL_INIT[k] for k in self.names])))
        self.register_buffer("c_ic", torch.tensor(c_ic))
        self.register_buffer("c_hex", torch.tensor(c_hex))
        self.dt, self.substeps = dt, substeps

    def p(self):
        v = torch.exp(self.logp)
        P = {k: v[i] for i, k in enumerate(self.names)}
        P["tau_sup"] = torch.clamp(P["tau_sup"], 2.0, 30.0)   # supply air reacts within seconds (physical bound)
        return P

    def deriv(self, x, u, th, P):
        Tcab, Tsh, Tsup, Tb, Tm, Ti, Td, Th, Tc = x.unbind(-1)
        Ta, w, mair, Qh, Qm, Qi, Qd, Qb = u.unbind(-1)
        t = th.unbind(-1)
        q_ic = t[2] * sum(c * b for c, b in zip(self.c_ic, basis_ic(w, Tc, Tcab, Ta)))
        q_hex = t[3] * sum(c * b for c, b in zip(self.c_hex, basis_hex(w, Tc, Ta)))
        q_cab_sh = t[0] * P["UA_cab_sh"] * (Tcab - Tsh)
        q_sh_amb = t[1] * P["UA_sh_amb"] * (Tsh - Ta)
        q_cab_amb = t[1] * P["UA_cab_amb"] * (Tcab - Ta)
        q_b = t[4] * P["UA_bat"] * (Tb - Tc)
        q_m = t[5] * P["UA_mot"] * (Tm - Tc)
        q_i = t[6] * P["UA_inv"] * (Ti - Tc)
        q_d = t[6] * P["UA_dcdc"] * (Td - Tc)
        q_h = t[7] * P["UA_htr"] * (Th - Tc)
        q_ca = t[10] * P["UA_cool_amb"] * (Tc - Ta)
        mcp = torch.clamp(mair * CP_AIR, min=20.0)
        return torch.stack([
            (mair * CP_AIR * (Tsup - Tcab) - q_cab_sh - q_cab_amb + Q_OCC) / P["C_cab"],
            (q_cab_sh - q_sh_amb) / P["C_sh"],
            (Tcab + q_ic / mcp - Tsup) / P["tau_sup"],
            (t[9] * Qb - q_b) / P["C_bat"],
            (Qm - q_m) / P["C_mot"],
            (Qi - q_i) / P["C_inv"],
            (Qd - q_d) / P["C_dcdc"],
            (t[8] * Qh - q_h) / P["C_htr"],
            (q_b + q_m + q_i + q_d + q_h - q_hex - q_ca) / P["C_cool"],
        ], -1)

    def simulate(self, x0, U, theta):
        """x0 (B,9), U (B,H,8), theta (B,11) constant over the horizon -> (B,H+1,9)"""
        P = self.p()
        h = self.dt / self.substeps
        x, out = x0, [x0]
        for k in range(U.shape[1]):
            for _ in range(self.substeps):
                x = x + h * self.deriv(x, U[:, k], theta, P)
            out.append(x)
        return torch.stack(out, 1)


# =============================================================================
# 3. Windows, losses
# =============================================================================
W_STATE = torch.tensor([1, 1, 0.2, 1, 1, 1, 1, 0.5, 1.0])


def windows(R, runs, H, stride=None, n=None, rng=None, t0=0):
    X0s, Us, Xs, ids = [], [], [], []
    for name in runs:
        X = torch.tensor(R[name]["X"])
        U = torch.tensor(np.stack([R[name][k] for k in INPUTS], 1))
        starts = np.arange(t0, len(X) - H, stride) if stride else rng.integers(t0, len(X) - H, n)
        for k in starts:
            X0s.append(X[k]); Us.append(U[k:k + H]); Xs.append(X[k:k + H + 1]); ids.append((name, int(k)))
    return torch.stack(X0s), torch.stack(Us), torch.stack(Xs), ids


def loss_fn(Xp, Xt):
    return (((Xp - Xt) ** 2) * W_STATE).mean()


def rmse_states(Xp, Xt):
    return torch.sqrt(((Xp[:, 1:] - Xt[:, 1:]) ** 2).mean(dim=(0, 1))).numpy()


# =============================================================================
# 4. Neural-network parameter scheduler
# =============================================================================
FEATURES = ["T_amb", "w", "m_air", "Q_htr", "Q_mot", "Q_bat", "m_ram", "p_suc", "p_dis",
            "T_cab", "T_sh", "T_bat", "T_cool", "Q_mot_ma60", "Q_bat_ma60"]


def causal_ma(x, n=60):
    c = np.cumsum(np.r_[0.0, x])
    k = np.arange(len(x))
    lo = np.maximum(0, k - n + 1)
    return (c[k + 1] - c[lo]) / (k + 1 - lo)


def feature_matrix(r):
    f = dict(r, Q_mot_ma60=causal_ma(r["Q_mot"]), Q_bat_ma60=causal_ma(r["Q_bat"]))
    return np.stack([f[k] for k in FEATURES], 1)


class Scheduler(nn.Module):
    def __init__(self, mean, std, hidden=32, max_log=1.2):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.net = nn.Sequential(nn.Linear(len(FEATURES), hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, N_THETA))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)       # starts at theta = 1 (nominal)
        self.max_log = max_log

    def forward(self, f):
        z = self.net((f - self.mean) / self.std)
        return torch.exp(self.max_log * torch.tanh(z / self.max_log))   # bounded to [0.30, 3.3]


class Envelope:
    """Out-of-distribution check: feature box of the training data with a margin."""

    def __init__(self, F, margin=0.10):
        lo, hi = F.min(0).values, F.max(0).values
        span = hi - lo + 1e-9
        self.lo, self.hi = lo - margin * span, hi + margin * span

    def inside(self, f):
        return ((f >= self.lo) & (f <= self.hi)).all(-1)


# =============================================================================
# 5. Main
# =============================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    log = open(os.path.join(OUT, "results.txt"), "w")

    def say(s=""):
        print(s)
        log.write(s + "\n")
        log.flush()

    t_start = time.time()
    R = {k: load_run(v) for k, v in RUN_FILES.items()}
    say(f"Loaded runs {list(R)}; training on {TRAIN}, testing on {TEST} (held out)")

    # --- nominal heat-pump correlations (from measured heat flows, training runs only)
    sl = slice(30, None)
    def stack(fn): return np.concatenate([fn(R[k])[sl] for k in TRAIN])
    Xic = stack(lambda r: np.stack(basis_ic(r["w"], r["T_cool"], r["T_cab"], r["T_amb"]), 1))
    Xhx = stack(lambda r: np.stack(basis_hex(r["w"], r["T_cool"], r["T_amb"]), 1))
    c_ic = np.linalg.lstsq(Xic, stack(lambda r: r["Q_ic"]), rcond=None)[0]
    c_hex = np.linalg.lstsq(Xhx, stack(lambda r: r["Q_hex"]), rcond=None)[0]

    # --- step 1: nominal COM identification (theta = 1) on training runs
    say("\n[1/4] Identifying nominal COM on training runs ...")
    model = COM(c_ic, c_hex)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    for it in range(301):
        X0, U, Xt, _ = windows(R, TRAIN, H_LONG, n=24, rng=rng)
        loss = loss_fn(model.simulate(X0, U, torch.ones(len(X0), N_THETA)), Xt)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            say(f"   iter {it:3d}  loss {loss.item():.4f}  ({time.time()-t_start:.0f} s)")
    for p in model.parameters():
        p.requires_grad_(False)

    # --- step 2: baseline = constant theta per temperature + linear interpolation
    say("\n[2/4] Offline identification of one constant theta per ambient temperature ...")
    def fit_const(runs, iters=150):
        lt = torch.zeros(N_THETA, requires_grad=True)
        o = torch.optim.Adam([lt], lr=0.05)
        for _ in range(iters):
            X0, U, Xt, _ = windows(R, runs, H_LONG, n=24, rng=rng)
            th = torch.exp(torch.clamp(lt, -1.2, 1.2)).expand(len(X0), -1)
            l = loss_fn(model.simulate(X0, U, th), Xt)
            o.zero_grad(); l.backward(); o.step()
        return torch.exp(torch.clamp(lt.detach(), -1.2, 1.2))
    th_const = {k: fit_const([k]) for k in TRAIN}
    a = (T_AMB[TEST] - T_AMB[TRAIN[0]]) / (T_AMB[TRAIN[1]] - T_AMB[TRAIN[0]])
    th_interp = th_const[TRAIN[0]] + a * (th_const[TRAIN[1]] - th_const[TRAIN[0]])
    say("   (oracle: also fitting a constant theta directly on the test run, as an upper bound)")
    th_oracle = fit_const([TEST])
    say(f"   done ({time.time()-t_start:.0f} s)")

    # --- step 3: NN scheduler trained end-to-end on prediction error
    say("\n[3/4] Training neural-network parameter scheduler ...")
    FEAT = {k: torch.tensor(feature_matrix(R[k])) for k in R}
    Ftr = torch.cat([FEAT[k] for k in TRAIN])
    sch = Scheduler(Ftr.mean(0), Ftr.std(0) + 1e-6)
    env = Envelope(Ftr)
    feats = lambda ids: torch.stack([FEAT[n][k] for n, k in ids])
    opt = torch.optim.Adam(sch.parameters(), lr=3e-3, weight_decay=1e-4)
    for it in range(601):
        X0, U, Xt, ids = windows(R, TRAIN, H_TRAIN, n=32, rng=rng)
        th = sch(feats(ids))
        loss = loss_fn(model.simulate(X0, U, th), Xt) + 1e-3 * (torch.log(th) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            say(f"   iter {it:3d}  loss {loss.item():.4f}  ({time.time()-t_start:.0f} s)")

    def nn_safe(ids):
        f = feats(ids)
        th = sch(f)
        ok = env.inside(f).unsqueeze(-1)
        return torch.where(ok, th, th_interp.expand_as(th)), ok

    # --- step 4: evaluation on the held-out run
    say(f"\n[4/4] Evaluation on held-out run {TEST} ({T_AMB[TEST]:.0f} degC), RMSE in K per state")
    methods = {
        "Nominal (theta=1)": lambda ids: torch.ones(len(ids), N_THETA),
        "Interpolation (baseline)": lambda ids: th_interp.expand(len(ids), -1),
        "NN scheduler (proposed)": lambda ids: nn_safe(ids)[0],
        "Oracle const. (fit on T7)": lambda ids: th_oracle.expand(len(ids), -1),
    }
    res = {}
    with torch.no_grad():
        for H in [H_MPC, H_LONG]:
            X0, U, Xt, ids = windows(R, [TEST], H, stride=5)
            say(f"\n  Horizon {H} s ({len(ids)} windows)")
            say("  " + f"{'method':28s}{'mean':>7s} " + " ".join(f"{s[:8]:>8s}" for s in STATE_LABELS))
            for nm, fn in methods.items():
                r = rmse_states(model.simulate(X0, U, fn(ids)), Xt)
                res[(nm, H)] = r
                say("  " + f"{nm:28s}{r.mean():7.4f} " + " ".join(f"{v:8.3f}" for v in r))
        gain = 100 * (1 - res[("NN scheduler (proposed)", H_MPC)].mean() / res[("Interpolation (baseline)", H_MPC)].mean())
        gainL = 100 * (1 - res[("NN scheduler (proposed)", H_LONG)].mean() / res[("Interpolation (baseline)", H_LONG)].mean())
        say(f"\n  NN vs interpolation: {gain:+.1f}% lower error at {H_MPC} s horizon, {gainL:+.1f}% at {H_LONG} s")

        # theta along the test run + OOD statistics
        ids_all = [(TEST, k) for k in range(len(FEAT[TEST]))]
        th_t, ok = nn_safe(ids_all)
        say(f"  Out-of-distribution fallback active on {100*(1-ok.double().mean()):.1f}% of test samples")
        say("\n  Learned parameters on test run (min / mean / max) vs interpolated constant:")
        for i, nm in enumerate(THETA_NAMES):
            say(f"   {nm:26s} {th_t[:, i].min():5.2f} / {th_t[:, i].mean():5.2f} / {th_t[:, i].max():5.2f}   interp {th_interp[i]:5.2f}")

        # plots
        t = R[TEST]["t"]
        fig, ax = plt.subplots(4, 3, figsize=(15, 11), sharex=True)
        for i, a_ in enumerate(ax.flat[:N_THETA]):
            a_.plot(t, th_t[:, i].numpy(), label="NN scheduler")
            a_.axhline(float(th_interp[i]), color="C1", ls="--", label="interpolated")
            a_.axhline(float(th_oracle[i]), color="C2", ls=":", label="oracle const.")
            a_.set_title(THETA_NAMES[i], fontsize=9)
        ax.flat[0].legend(fontsize=8)
        ax.flat[-1].axis("off")
        fig.suptitle(f"COM scaling parameters on held-out run {TEST} ({T_AMB[TEST]:.0f} degC)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "theta_test_run.png"), dpi=110); plt.close(fig)

        X0, U, Xt, ids = windows(R, [TEST], H_MPC, stride=5)
        tt = np.array([k for _, k in ids]) + H_MPC
        fig, ax = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
        for nm, c in [("Interpolation (baseline)", "C1"), ("NN scheduler (proposed)", "C0")]:
            Xp = model.simulate(X0, U, methods[nm](ids))
            e = (Xp[:, -1] - Xt[:, -1]).abs().numpy()
            for i, a_ in enumerate(ax.flat):
                a_.plot(tt, e[:, i], c, lw=0.8, label=nm)
                a_.set_title(f"{STATE_LABELS[i]}: |error| after {H_MPC} s [K]", fontsize=9)
        ax.flat[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "prediction_error_30s.png"), dpi=110); plt.close(fig)

        fig, a_ = plt.subplots(figsize=(9, 4))
        names = list(methods)
        x = np.arange(len(STATE_LABELS))
        for j, nm in enumerate(names):
            a_.bar(x + (j - 1.5) * 0.2, res[(nm, H_MPC)], 0.2, label=nm)
        a_.set_xticks(x); a_.set_xticklabels(STATE_LABELS, rotation=30)
        a_.set_ylabel(f"RMSE after up to {H_MPC} s [K]"); a_.legend(fontsize=8)
        a_.set_title(f"{H_MPC} s prediction error on held-out {TEST}")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "rmse_comparison.png"), dpi=110); plt.close(fig)

    # export
    torch.save({"com": model.state_dict(), "scheduler": sch.state_dict(), "c_ic": c_ic, "c_hex": c_hex,
                "theta_const": th_const, "theta_interp": th_interp}, os.path.join(OUT, "com_scheduler.pt"))
    sd = {k: v.numpy() for k, v in sch.state_dict().items()}
    savemat(os.path.join(OUT, "scheduler_params.mat"), {
        "W1": sd["net.0.weight"], "b1": sd["net.0.bias"][:, None], "W2": sd["net.2.weight"],
        "b2": sd["net.2.bias"][:, None], "W3": sd["net.4.weight"], "b3": sd["net.4.bias"][:, None],
        "f_mean": sd["mean"][:, None], "f_std": sd["std"][:, None], "max_log": 1.2,
        "env_lo": env.lo.numpy()[:, None], "env_hi": env.hi.numpy()[:, None],
        "theta_interp": th_interp.numpy()[:, None], "features": np.array(FEATURES, dtype=object),
        "theta_names": np.array(THETA_NAMES, dtype=object),
        "com_nominal": np.array([[float(v) for v in model.p().values()]]),
        "com_nominal_names": np.array(model.names, dtype=object), "c_ic": c_ic, "c_hex": c_hex})
    say(f"\nSaved results, plots and weights to '{OUT}/'  (total {time.time()-t_start:.0f} s)")


if __name__ == "__main__":
    main()