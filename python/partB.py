"""
Part B: learn the COM's γ (p1..p11) end-to-end so that the COM's 30-s predictions match the HFM.

Set up exactly like the NMPC uses the COM:
  - every window starts from the HFM state (in the NMPC: the measured state),
  - refrigerant/air properties (est) and supervisor commands (ctrl) are computed at the window start and held,
  - γ is computed once at the window start and held over the 30-s horizon (like prepParaVec in the NMPC),
  - inputs u and heat loads vary over the window (the NMPC knows them as decisions / preview),
  - RK4 with Ts = 1 s.

Main pieces
  load_hfm_profile(T)      old file Data/tuning_data_<T>.mat -> HFM states in COM units + inputs + properties
  make_windows(prof, H)    cut a profile into 30-s windows
  simulate(win, gamma)     COM rollout of all windows with given γ [W, 11]
  TableGamma               the interpolation table (prepParaVec) as a baseline
  GammaNet / features      the network that predicts γ at the window start
"""
import math, os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from com_torch import com_xdot, rk4_step, STATE_NAMES, INPUT_NAMES, CTRL_NAMES, EST_NAMES
from est_coolprop import estimate_params

DT = torch.float64
C2K = 273.15

# Loss/report scales per COM state: temperatures in K (1 K), SOC (0.001), pressures (1 bar)
STATE_SCALE = torch.tensor([1, 1, 1, 1e-3, 1, 1e5, 1e5, 1, 1], dtype=DT)
STATE_UNIT = ["K", "K", "K", "-", "K", "bar", "bar", "K", "K"]


# ════════════════════════════════════════════════════════════════════ data
def load_hfm_profile(T, data_dir, cache_dir=None):
    """Old file tuning_data_<T>.mat (T = 7 means -7 °C) -> dict of torch tensors over time (N = 3601).

    x    [N, 9]  HFM states in COM units (K, SOC, Pa)
    u    [N, 6]  inputs (compNrpm, cabAirMfIn, motPumpNrpm, batPumpNrpm, heaterPwr, radFanrpm)
    est  {name: [N]} properties from CoolProp at the HFM pressures (estimateParameters.m port)
    ctrl {name: [N]} supervisor commands
    Tamb ambient [°C]
    """
    cache = os.path.join(cache_dir, f"partB_profile_{T}.pt") if cache_dir else None
    if cache and os.path.exists(cache):
        return torch.load(cache)
    d = sio.loadmat(os.path.join(data_dir, f"tuning_data_{T}.mat"), squeeze_me=True, struct_as_record=False)
    r = d["resampled"]
    N = len(r.motTmeasC)
    Tamb = float(d["envTemp"])
    f = lambda a: np.asarray(a, float)

    x = np.stack([f(r.motTmeasC) + C2K, f(r.invTmeasC) + C2K, f(r.dcdcTmeasC) + C2K, f(r.batSocMeas),
                  f(r.batTmeasC) + C2K, f(r.compPinMpa) * 1e6, f(r.compPoutMpa) * 1e6,
                  f(r.cabIntTInK), f(r.cabAirTmeasC) + C2K], 1)
    # battery-pump speed is not in the old files; both pumps ran at the same speed in these runs
    u = np.stack([f(r.compNrpm), f(r.cabAirMfIn), f(r.motPumpNrpm), f(r.motPumpNrpm),
                  f(r.heaterCmd) * 4000.0, np.zeros(N)], 1)

    est = estimate_params(x[:, 5], x[:, 6], x[:, 8], f(d["superHeat"]), Tamb + C2K,
                          f(r.motQmeas), f(r.invQmeas), f(r.dcdcQmeas), f(r.batImeas))
    co = [c if hasattr(c, "_fieldnames") else None for c in d["controlOutCell"]]
    co = [c if c is not None else co[i - 1] for i, c in enumerate(co)]
    ctrl = {n: np.array([float(getattr(c, n)) for c in co]) for n in CTRL_NAMES}

    t = lambda a: torch.as_tensor(a, dtype=DT)
    prof = dict(T=T, Tamb=Tamb, x=t(x), u=t(u),
                est={k: t(v) for k, v in est.items() if k in EST_NAMES},
                ctrl={k: t(v) for k, v in ctrl.items()})
    if cache:
        torch.save(prof, cache)
    return prof


HELD_EST = [n for n in EST_NAMES if n not in ("motQmeas", "invQmeas", "dcdcQmeas", "batImeas", "speed")]


def make_windows(prof, H=30, stride=30):
    """Cut a profile into windows [k0, k0+H]. Properties/commands held at k0; inputs and heat loads per step."""
    N = prof["x"].shape[0]
    k0 = torch.arange(1, N - H, stride)                           # skip t = 0 (all states equal ambient)
    idx = k0[:, None] + torch.arange(H)[None]                     # [W, H]
    est_held = {n: prof["est"][n][k0][:, None].expand(-1, H) for n in HELD_EST}
    est_step = {n: prof["est"][n][idx] for n in ("motQmeas", "invQmeas", "dcdcQmeas", "batImeas", "speed")}
    return dict(T=prof["T"], Tamb=torch.full((len(k0),), prof["Tamb"], dtype=DT), k0=k0,
                x0=prof["x"][k0], x_true=prof["x"][idx + 1], u=prof["u"][idx],
                est={**est_held, **est_step},
                ctrl={n: v[k0][:, None].expand(-1, H) for n, v in prof["ctrl"].items()})


def cat_windows(ws):
    out = dict(T=[w["T"] for w in ws])
    for k in ("Tamb", "k0", "x0", "x_true", "u"):
        out[k] = torch.cat([w[k] for w in ws])
    for k in ("est", "ctrl"):
        out[k] = {n: torch.cat([w[k][n] for w in ws]) for n in ws[0][k]}
    return out


def simulate(win, gamma, clamp=True):
    """COM rollout of all windows. gamma [W, 11] held over the window. Returns states [W, H, 9]."""
    x, out = win["x0"], []
    H = win["u"].shape[1]
    for k in range(H):
        x = rk4_step(x, win["u"][:, k], {n: v[:, k] for n, v in win["est"].items()},
                     {n: v[:, k] for n, v in win["ctrl"].items()}, gamma)
        if clamp:   # keep a diverging window from producing NaN (it still gets a large error)
            x = torch.stack([x[:, i].clamp(150, 450) if i not in (3, 5, 6) else x[:, i] for i in range(9)], 1)
            x = torch.cat([x[:, :3], x[:, 3:4].clamp(0, 1.2), x[:, 4:5], x[:, 5:7].clamp(1e3, 1e7), x[:, 7:]], 1)
        out.append(x)
    return torch.stack(out, 1)


def horizon_rmse(pred, true):
    """RMSE per state over all windows and steps, in report units (K, -, bar)."""
    err = (pred - true) / torch.tensor([1, 1, 1, 1, 1, 1e5, 1e5, 1, 1], dtype=DT)
    return err.pow(2).mean((0, 1)).sqrt()


def scaled_loss(pred, true):
    """Mean squared error with every state in comparable units (1 K, 0.001 SOC, 1 bar)."""
    return ((pred - true) / STATE_SCALE).pow(2).mean()


# ════════════════════════════════════════════════════════════════════ interpolation table (baseline)
class TableGamma:
    """prepParaVec.m: p_i = scatteredInterpolant(ambient, state_j) from tunedVariables (linear, nearest outside)."""
    IDX_STATE = [8, 9, 5, 2, 1, 3, 5, 9, 6, 7, 6]       # driving state (1-based) for p1..p11

    def __init__(self, tv_path):
        tv = sio.loadmat(tv_path, squeeze_me=True, struct_as_record=False)["tunedVariables"]
        amb, St, Pm = np.asarray(tv.ambient, float), np.asarray(tv.states, float), np.asarray(tv.parameters, float)
        self.f = []
        for i in range(11):
            pts = np.stack([amb, St[:, self.IDX_STATE[i] - 1]], 1)
            z = Pm[:, i]
            good = np.isfinite(pts).all(1) & np.isfinite(z)
            # scatteredInterpolant uses the raw coordinates; with states in K/Pa next to ambient in °C the
            # triangulation is very skewed, exactly as in MATLAB. Kept the same on purpose.
            self.f.append((LinearNDInterpolator(pts[good], z[good]), NearestNDInterpolator(pts[good], z[good])))

    def __call__(self, Tamb, x0):
        Tamb, x0 = np.asarray(Tamb, float), np.asarray(x0, float)
        g = np.zeros((len(Tamb), 11))
        for i, (lin, near) in enumerate(self.f):
            q = np.stack([Tamb, x0[:, self.IDX_STATE[i] - 1]], 1)
            v = lin(q)
            bad = ~np.isfinite(v)
            if bad.any():
                v[bad] = near(q[bad])
            g[:, i] = v
        return torch.as_tensor(g, dtype=DT)


# ════════════════════════════════════════════════════════════════════ γ network
def features(win):
    """Network inputs at the window start: ambient, current state, inputs and heat loads, window-average inputs."""
    x0, u = win["x0"], win["u"]
    temps = (x0[:, [0, 1, 2, 4, 7, 8]] - C2K) / 50.0
    soc = x0[:, 3:4]
    press = x0[:, 5:7] / 1e6
    u_scale = torch.tensor([5000, 0.2, 1200, 1200, 4000, 1000], dtype=DT)
    u0 = u[:, 0] / u_scale
    u_mean = u.mean(1) / u_scale                                  # 30-s preview, as the NMPC has it
    q = torch.stack([win["est"]["motQmeas"][:, 0] / 4300, win["est"]["invQmeas"][:, 0] / 240,
                     win["est"]["batImeas"][:, 0] / 175], 1)
    return torch.cat([win["Tamb"][:, None] / 20.0, temps, soc, press, u0, u_mean, q], 1)


N_FEATURES = 1 + 6 + 1 + 2 + 6 + 6 + 3


class GammaNet(nn.Module):
    """features -> 11 multipliers in [lo, hi], log-symmetric; starts at γ = 1 (or at a given init)."""
    def __init__(self, n_in=N_FEATURES, n_gamma=11, hidden=(64, 64), lo=0.1, hi=10.0):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]; d = h
        layers += [nn.Linear(d, n_gamma)]
        self.net = nn.Sequential(*layers).to(DT)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.log_lo, self.log_hi = math.log(lo), math.log(hi)

    def forward(self, z):
        s = torch.sigmoid(self.net(z))
        return torch.exp(self.log_lo + s * (self.log_hi - self.log_lo))


# ════════════════════════════════════════════════════════════════════ training / evaluation helpers
def subset(win, idx):
    out = {}
    for k, v in win.items():
        if torch.is_tensor(v):
            out[k] = v[idx]
        elif isinstance(v, dict):
            out[k] = {n: a[idx] for n, a in v.items()}
        else:
            out[k] = v
    return out


class ConstantGamma(nn.Module):
    """One γ vector for all operating points (baseline: what a single re-tuned parameter set can do)."""
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(11, dtype=DT))
    def forward(self, z):
        g = torch.exp(self.theta.clamp(math.log(0.1), math.log(10)))
        return g[None].expand(z.shape[0], -1)


def train_gamma(model, train_win, norm, iters=200, batch=512, lr=3e-3, log_every=20, seed=0):
    """End-to-end training: γ = model(features) -> COM rollout -> loss against HFM states."""
    import time
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    W = train_win["x0"].shape[0]
    hist, t0 = [], time.time()
    for it in range(iters):
        sub = subset(train_win, torch.randperm(W)[:batch])
        g = model(norm(features(sub)))
        loss = scaled_loss(simulate(sub, g), sub["x_true"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(loss.item())
        hist.append(loss.item())
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"iter {it:4d}  loss {loss.item():8.4f}  lr {opt.param_groups[0]['lr']:.1e}  {time.time() - t0:5.0f}s",
                  flush=True)
    return hist


@torch.no_grad()
def evaluate(win, gammas):
    """{name: γ [W, 11]} -> {name: RMSE per state [9]} on the given windows."""
    return {name: horizon_rmse(simulate(win, g), win["x_true"]) for name, g in gammas.items()}
