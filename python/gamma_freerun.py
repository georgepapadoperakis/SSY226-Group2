"""
Free-run training and evaluation of the γ network (was startup_train.py), plus the 12- and 15-output networks.

  startup_rollout / finetune_startup   free run of the COM with γ from the network (11 γ, original COM)
  GammaNet12 / finetune_g12            12th output γ12 (suction density)
  GammaNet15                           3 more outputs γ13..γ15 for the fixed COM (heat-exchanger UAs)
  load_gamma11 / load_gamma15          the two final networks in models/, ready to use
  free_run_eval                        one-hour free run + error per state group (used by the test script)

Why free-run fine-tuning: in a free run the COM starts from the HFM state at t = 0 and then uses its OWN state for
the fluid properties and for the network's inputs.

Why: in a free run the COM starts from the HFM state at t = 0 and then uses its OWN state for the fluid properties
and for the network's inputs. During training the windows always started from the HFM state with HFM-based
properties, so the network never saw 'cold system, p_out still low, compressor running' and slowed p_out down.
Here the first START_S seconds of every training run are rolled out exactly like free_run (γ every second from the
predicted state, properties from the predicted pressures via fluid_props_table.PropTable), with gradients through all steps.
The usual 300 s window loss is kept in the same objective so the rest of the drive is not forgotten.
"""
import time, math, torch, numpy as np
from gamma_net import *
from fluid_props_table import PropTable
import com_torch

STEP_NAMES = ("motQmeas", "invQmeas", "dcdcQmeas", "batImeas", "speed")


def startup_rollout(net, norm, profs, tab, S, H=30, k_start=0):
    """Free run of S steps from the HFM state at k_start for a batch of profiles. Returns states [B, S, 9]."""
    X = torch.stack([p["x"] for p in profs]); U = torch.stack([p["u"] for p in profs]); N = X.shape[1]
    HEAT = {n: torch.stack([p["est"][n] for p in profs]) for n in STEP_NAMES}
    CTRL = {n: torch.stack([p["ctrl"][n] for p in profs]) for n in profs[0]["ctrl"]}
    TA = torch.tensor([p["Tamb"] for p in profs], dtype=DT)
    x = X[:, k_start].clone(); out = []
    for k in range(k_start, k_start + S):
        kk = torch.clamp(torch.arange(k, k + H), max=N - 1)
        g = net(norm(features(dict(Tamb=TA, x0=x, u=U[:, kk], est={n: v[:, kk] for n, v in HEAT.items()}))))
        est = tab(x, {n: v[:, k] for n, v in HEAT.items()}, TA + C2K)
        x = rk4_step(x, U[:, k], est, {n: v[:, k] for n, v in CTRL.items()}, g)
        x = torch.cat([x[:, :3], x[:, 3:4].clamp(0, 1.2), x[:, 4:5], x[:, 5:7].clamp(1e4, 6e6), x[:, 7:]], 1)
        out.append(x)
    return torch.stack(out, 1)


def finetune_startup(net, norm, PB, train_keys, tab, S=300, iters=60, lr=5e-4, batch_prof=6, batch_win=16,
                     w_startup=1.0, log=print, deadline=None, seed=3):
    torch.manual_seed(seed)
    w300s = cat_windows([make_windows(PB[k], H=300, stride=20, startup_stride=3) for k in train_keys])
    sc_win = loss_scale(w300s)
    with torch.no_grad():   # balanced scale for the start-up loss: error of the current network per state
        P0 = startup_rollout(net, norm, [PB[k] for k in train_keys], tab, S)
        T0 = torch.stack([PB[k]["x"][1:S + 1] for k in train_keys])
        sc_su = (P0 - T0).pow(2).mean((0, 1)).sqrt().clamp_min(1e-9)
        sc_su = torch.maximum(sc_su, sc_win)      # never weight a state more than in the window loss
    opt = torch.optim.Adam(net.parameters(), lr=lr); hist = []; t0 = time.time()
    for it in range(iters):
        if deadline and time.time() > deadline:
            log(f"  time budget reached after {it} iterations"); break
        pk = [train_keys[i] for i in torch.randperm(len(train_keys))[:batch_prof]]
        P = startup_rollout(net, norm, [PB[k] for k in pk], tab, S)
        T = torch.stack([PB[k]["x"][1:S + 1] for k in pk])
        l_su = scaled_loss(P, T, sc_su)
        sub = subset(w300s, torch.randperm(w300s["x0"].shape[0])[:batch_win])
        l_w = scaled_loss(simulate(sub, net(norm(features(sub)))), sub["x_true"], sc_win)
        loss = (w_startup * l_su + l_w) / (1 + w_startup)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        hist.append((loss.item(), l_su.item(), l_w.item()))
        if it % 5 == 0 or it == iters - 1:
            log(f"  iter {it:3d}  loss {loss.item():.4f}  start-up {l_su.item():.4f}  windows {l_w.item():.4f}  {time.time() - t0:5.0f}s")
    return hist, sc_su


# ═══════════════════════════════════════════════════════════════════ γ12 (suction density) extension
class GammaNet12(nn.Module):
    """GammaNet with a 12th output γ12 ∈ [0, G12_MAX] for the pressure-dependent suction volume (com_torch).
    Built from a trained 11-output GammaNet: same hidden layers, γ1..γ11 unchanged at the start,
    γ12 starts at g12_init (0.83 = fit of the HFM mass flow, see notes)."""
    G12_MAX = 1.5

    def __init__(self, base, g12_init=0.83):
        super().__init__()
        self.base = base
        last = base.net[-1]
        self.head12 = nn.Linear(last.in_features, 1).to(DT)
        nn.init.zeros_(self.head12.weight)
        nn.init.constant_(self.head12.bias, math.log(g12_init / (self.G12_MAX - g12_init)))

    def forward(self, z):
        h = self.base.net[:-1](z)
        g11 = torch.exp(self.base.log_lo + torch.sigmoid(self.base.net[-1](h)) * (self.base.log_hi - self.base.log_lo))
        g12 = self.G12_MAX * torch.sigmoid(self.head12(h))
        return torch.cat([g11, g12], 1)


def finetune_g12(net, norm, PB, train_keys, tab, iters=60, lr=5e-4, S_su=300, S_mid=600, b_su=4, b_mid=4, b_win=16,
                 log=print, deadline=None, seed=4):
    """Loss = start-up free run (0-300 s) + mid-run free run (600 s from a random point) + 300 s windows.
    The mid-run free runs put the slow p_in / temperature drift into the loss (own-state properties)."""
    torch.manual_seed(seed)
    w300s = cat_windows([make_windows(PB[k], H=300, stride=20, startup_stride=3) for k in train_keys])
    sc_win = loss_scale(w300s)
    profs = [PB[k] for k in train_keys]
    with torch.no_grad():
        P0 = startup_rollout(net, norm, profs, tab, S_su); T0 = torch.stack([p["x"][1:S_su + 1] for p in profs])
        sc_su = torch.maximum((P0 - T0).pow(2).mean((0, 1)).sqrt().clamp_min(1e-9), sc_win)
        P1 = startup_rollout(net, norm, profs, tab, S_mid, k_start=1500); T1 = torch.stack([p["x"][1501:1501 + S_mid] for p in profs])
        sc_mid = torch.maximum((P1 - T1).pow(2).mean((0, 1)).sqrt().clamp_min(1e-9), sc_win)
    opt = torch.optim.Adam(net.parameters(), lr=lr); hist = []; t0 = time.time()
    for it in range(iters):
        if deadline and time.time() > deadline:
            log(f"  time budget reached after {it} iterations"); break
        pk = [profs[i] for i in torch.randperm(len(profs))[:b_su]]
        l_su = scaled_loss(startup_rollout(net, norm, pk, tab, S_su), torch.stack([p["x"][1:S_su + 1] for p in pk]), sc_su)
        pk = [profs[i] for i in torch.randperm(len(profs))[:b_mid]]
        k0 = int(torch.randint(300, 3600 - S_mid - 1, (1,)))
        l_mid = scaled_loss(startup_rollout(net, norm, pk, tab, S_mid, k_start=k0),
                            torch.stack([p["x"][k0 + 1:k0 + 1 + S_mid] for p in pk]), sc_mid)
        sub = subset(w300s, torch.randperm(w300s["x0"].shape[0])[:b_win])
        l_w = scaled_loss(simulate(sub, net(norm(features(sub)))), sub["x_true"], sc_win)
        loss = (l_su + l_mid + l_w) / 3
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            g12 = net(norm(features(subset(w300s, torch.arange(0, w300s["x0"].shape[0], 50)))))[:, 11]
        hist.append((loss.item(), l_su.item(), l_mid.item(), l_w.item(), g12.mean().item()))
        if it % 5 == 0 or it == iters - 1:
            log(f"  iter {it:3d}  loss {loss.item():.4f}  start-up {l_su.item():.4f}  mid-run {l_mid.item():.4f}  "
                f"windows {l_w.item():.4f}  mean γ12 {g12.mean().item():.2f}  {time.time() - t0:5.0f}s")
    return hist


# ═══════════════════════════════════════════════════════════════════ γ13..γ15 (fixed COM) extension
K_CAL = dict(k_htx=2.6159, k_mot=1.5887, k_inv=5.1496 * 25)     # fixed-COM UA multipliers at γ = 1 (calibrated;
                                                                 # same values as MATLAB com_fix_constants)


class GammaNet15(nn.Module):
    """GammaNet12 + 3 outputs γ13..γ15 in [0.1, 10] (log-symmetric like γ1..γ11), starting at 1."""
    def __init__(self, net12):
        super().__init__()
        self.net12 = net12
        last = net12.base.net[-1]
        self.head = nn.Linear(last.in_features, 3).to(DT)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)       # sigmoid 0.5 -> γ = 1

    def forward(self, z):
        b = self.net12.base
        h = b.net[:-1](z)
        g = torch.exp(b.log_lo + torch.sigmoid(self.head(h)) * (b.log_hi - b.log_lo))
        return torch.cat([self.net12(z), g], 1)


def fill_speed(PB, ref):
    """Old WLTCHW exports (-5/-7/-10 °C) have no vehicle speed; the front HX needs it. Same cycle -> copy it."""
    for p in PB.values():
        if float(p["est"]["speed"].abs().max()) == 0.0:
            p["est"]["speed"] = ref["est"]["speed"][:len(p["x"])].clone()


def load_gamma11(path=MODELS / "gamma11_original_com.pt"):
    """11-γ network for the ORIGINAL COM. Returns (net, norm); call com_torch.set_fixes(False) before using it."""
    d = torch.load(path, weights_only=False)
    net = GammaNet(); net.load_state_dict(d["state"]); net.eval()
    return net, (lambda z, mu=d["mu"], sd=d["sd"]: (z - mu) / sd)


def load_gamma15(path=MODELS / "gamma15_fixed_com.pt"):
    """15-γ network for the FIXED COM. Returns (net, norm); call com_torch.set_fixes(True, **K_CAL) before using it."""
    d = torch.load(path, weights_only=False)
    net = GammaNet15(GammaNet12(GammaNet())); net.load_state_dict(d["state"]); net.eval()
    return net, (lambda z, mu=d["mu"], sd=d["sd"]: (z - mu) / sd)


@torch.no_grad()
def free_run_eval(net, norm, profs, tab, S=3600):
    """One-hour free run (γ and properties from the own state). Returns MAE per group and p_in drift."""
    P = startup_rollout(net, norm, profs, tab, S)
    T = torch.stack([p["x"][1:S + 1] for p in profs])
    e = (P - T).abs()
    t = torch.arange(1500, S, dtype=DT)
    slope = lambda y: ((t - t.mean()) * (y - y.mean())).sum() / ((t - t.mean()) ** 2).sum() * 3600 / 1e5
    drift = torch.stack([slope(P[i, 1500:S, 5]) - slope(T[i, 1500:S, 5]) for i in range(len(profs))])
    return dict(components=e[:, :, [0, 1, 2, 4]].mean((1, 2)), cab_air=e[:, :, 8].mean(1), cab_int=e[:, :, 7].mean(1),
                p_in=e[:, :, 5].mean(1) / 1e5, p_out=e[:, :, 6].mean(1) / 1e5, p_in_drift=drift, pred=P.float())
