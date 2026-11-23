"""
γ network library (was partB.py): learn the COM's tuning parameters γ end-to-end so that the COM's 30-s
predictions match the HFM (the Simscape reference model).

Set up exactly like the NMPC uses the COM:
  - every window starts from the HFM state (in the NMPC: the measured state),
  - refrigerant/air properties (est) and supervisor commands (ctrl) are computed at the window start and held,
  - γ is computed once at the window start and held over the 30-s horizon (like prepParaVec in the NMPC),
  - inputs u and heat loads vary over the window (the NMPC knows them as decisions / preview),
  - RK4 with Ts = 1 s.

Main pieces
  load_profiles(keys)      the committed rule-based runs from data_cache/ (no MATLAB data needed)
  load_all_profiles(...)   rebuild data_cache/ from Data/ and Results/ (MATLAB exports), or read it
  make_windows(prof, H)    cut a profile into H-s windows (start-up included, see START_RULE below)
  simulate(win, gamma)     COM rollout of all windows with given γ [W, 11]
  TableGamma               the interpolation table (prepParaVec) as a baseline
  GammaNet / features      the network that predicts γ at the window start
  train_recipe(...)        the 3-step training (train_gamma11_step1_base.py)
  evaluate / free_run      30-s horizon test (NMPC use) and full-hour free run (paper Fig. 2 / Table 1)
"""
import math, os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from com_torch import rk4_step, CTRL_NAMES, EST_NAMES
from fluid_props_coolprop import estimate_params

DT = torch.float64
C2K = 273.15

# Loss/report scales per COM state: temperatures in K (1 K), SOC (0.001), pressures (1 bar)
STATE_SCALE = torch.tensor([1, 1, 1, 1e-3, 1, 1e5, 1e5, 1, 1], dtype=DT)

# ═══════════════════════════════════════════════════════════════ folders, data split
from pathlib import Path
HERE = Path(__file__).absolute().parent          # the python/ folder
REPO = HERE.parent
DATA_CACHE = HERE / "data_cache"                # one file per rule-based run, e.g. UDDS_minus7C.pt
MODELS = HERE / "models"                        # trained networks (.pt for Python, .mat for MATLAB)
RESULTS_PY = HERE / "results"                   # outputs of the test scripts
PROP_TABLE = DATA_CACHE / "fluid_property_table.pt"

# Training: WLTCHW, NEDC, US06 at 0/-5/-10/-15/-20/-30 °C. Test: never used in training.
#   key = |Tamb| (int) for WLTCHW, "<CYCLE>_<|Tamb|>" for the other cycles, e.g. "UDDS_12" = UDDS at -12 °C.
TRAIN_T = (0, 5, 10, 15, 20, 30)
TRAIN_KEYS = list(TRAIN_T) + [f"{c}_{T}" for c in ("NEDC", "US06") for T in TRAIN_T]
TEST_KEYS = ["UDDS_0", "UDDS_7", "UDDS_12", "UDDS_20", "UDDS_25", 7, "NEDC_12", "US06_7"]
train_keys = TRAIN_KEYS                         # old name, still used by some scripts


def cache_name(key):
    """File name in data_cache/ for a profile key: 12 -> 'WLTCHW_minus12C.pt', 'UDDS_0' -> 'UDDS_0C.pt'."""
    cyc, T = (key.rsplit("_", 1)[0], int(key.rsplit("_", 1)[1])) if isinstance(key, str) else ("WLTCHW", int(key))
    return f"{cyc}_{'0C' if T == 0 else f'minus{T}C'}.pt"


def cache_file(key):
    return DATA_CACHE / cache_name(key)


def key_from_cache_name(name):
    """Inverse of cache_name, or None for other files."""
    import re
    m = re.fullmatch(r"([A-Z0-9]+)_(?:0C|minus(\d+)C)\.pt", name)
    if not m:
        return None
    T = int(m.group(2) or 0)
    return T if m.group(1) == "WLTCHW" else f"{m.group(1)}_{T}"


def load_profiles(keys):
    """{key: profile} from data_cache/ (the committed, preprocessed rule-based runs)."""
    PB = {}
    for k in keys:
        f = cache_file(k)
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: rebuild data_cache with load_all_profiles (needs Data/ and Results/)")
        PB[k] = torch.load(f, weights_only=False)
        PB[k]["cycle"] = k.rsplit("_", 1)[0] if isinstance(k, str) else "WLTCHW"
    return PB


# ════════════════════════════════════════════════════════════════════ data
SUPERHEAT_K = 2.0      # fixed superheat, as in the NMPC (MPCController_step_coreLQRNew.m)
P_CRIT = 4.0593e6      # R134a critical pressure [Pa]. Windows where the HFM p_out exceeds it are dropped (only the
                       # 0 °C start-up): above it CoolProp has no saturation properties.
P_OUT_SCALE = 3e5      # loss scale for p_out [Pa] = 3 bar. The "balanced" scale (γ = 1 error) is ~25 bar for p_out,
                       # which made the pressure almost weightless in training.


def supervisor_commands(envT_K, compN, pumpN, clntInvTout_K, clntBatTin_K):
    """Port of the command logic in thermalLogicAndEstimation.m, run over a whole profile (arrays over time).

    clntBatTin_K is the heater coolant outlet temperature (this reproduces the cached commands exactly).
    Chiller EXV is always 0 here: it only opens in cooling mode, which never happens in cold weather.
    """
    N = len(compN)
    out = {n: np.zeros(N) for n in CTRL_NAMES}
    setpoint = 1 if envT_K >= C2K + 21 else (2 if envT_K < C2K + 19 else 0)
    mode, heater, serial = 0, 0, 1
    for k in range(N):
        if mode != 0 and setpoint == 0:
            mode = 0
        elif mode == 0 and setpoint != 0 and compN[k] >= 10:
            mode = setpoint                                    # start once the compressor runs
        if mode > 1:
            serial = 1
        else:                                                  # hysteresis 35 / 30 °C
            Tc = max(envT_K, clntInvTout_K[k])
            serial = 0 if Tc >= C2K + 35 else (1 if Tc <= C2K + 30 else serial)
        if heater == 0:                                        # heater on: ambient < 5 °C, coolant < 15 °C, pump on
            heater = int(envT_K < C2K + 5 and clntBatTin_K[k] < C2K + 15 and pumpN[k] > 0)
        else:                                                  # off: ambient > 5 °C or coolant > 5 °C or pump off
            heater = int(not (envT_K > C2K + 5 or clntBatTin_K[k] > C2K + 5 or pumpN[k] <= 0))
        out["thermalMode"][k] = mode
        out["cmdBlendAir"][k] = out["cmdExvHeatpump"][k] = float(mode == 2)
        out["cmdExvEvap"][k] = float(mode == 1)
        out["cmdParallelSerial"][k] = serial
        out["cmdHeater"][k] = heater
    return out


def _profile(T, Tamb, x, u, heat, ctrl, superheat=None):
    """Common last step: CoolProp properties at the HFM pressures, everything as float64 tensors."""
    N = len(x)
    sh = np.full(N, SUPERHEAT_K) if superheat is None else superheat
    est = estimate_params(x[:, 5], x[:, 6], x[:, 8], sh, Tamb + C2K, *heat)
    t = lambda a: torch.as_tensor(a, dtype=DT)
    return dict(T=T, Tamb=Tamb, x=t(x), u=t(u),
                est={k: t(v) for k, v in est.items() if k in EST_NAMES},
                ctrl={k: t(v) for k, v in ctrl.items()})


def load_old_profile(path):
    """Old export Data/tuning_data_<T>.mat (-5, -7, -10 °C)."""
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    r, Tamb = d["resampled"], float(d["envTemp"])
    f = lambda a: np.asarray(a, float)
    N = len(r.motTmeasC)
    x = np.stack([f(r.motTmeasC) + C2K, f(r.invTmeasC) + C2K, f(r.dcdcTmeasC) + C2K, f(r.batSocMeas),
                  f(r.batTmeasC) + C2K, f(r.compPinMpa) * 1e6, f(r.compPoutMpa) * 1e6,
                  f(r.cabIntTInK), f(r.cabAirTmeasC) + C2K], 1)
    # battery-pump and fan speed are not in the old files; both pumps ran at the same speed in these runs
    u = np.stack([f(r.compNrpm), f(r.cabAirMfIn), f(r.motPumpNrpm), f(r.motPumpNrpm),
                  f(r.heaterCmd) * 4000.0, np.zeros(N)], 1)
    inv = f(r.clntInvTOut)
    inv = inv + C2K if inv.max() < 200 else inv
    ctrl = supervisor_commands(Tamb + C2K, f(r.compNrpm), f(r.motPumpNrpm), inv,
                               f(r.heaterClntToutMeasC))          # stored in K despite the name
    heat = (f(r.motQmeas), f(r.invQmeas), f(r.dcdcQmeas), f(r.batImeas), None)   # no speed in old files
    return _profile(int(round(abs(Tamb))), Tamb, x, u, heat, ctrl)


def load_new_profile(path):
    """New export Results/run_*/training_data_rule_T<T>.mat (from RunThermalComparison / export_training_data.m).

    Uses the m*cp-weighted cabin interior (cabIntTw_C) and the real superheat (compInT_C) when the file has
    them (exports made after 15 Nov); otherwise the plain wall average and the NMPC's fixed 2 K.
    """
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["trainingData"]
    has = lambda n: n in d._fieldnames and np.isfinite(np.asarray(getattr(d, n), float)).all()
    f = lambda a: np.asarray(a, float)
    Tamb = float(d.T_amb_C)
    cabInt = f(d.cabIntTw_C) if has("cabIntTw_C") else f(d.cabIntT_C)
    x = np.stack([f(d.motT_C) + C2K, f(d.invT_C) + C2K, f(d.dcdcT_C) + C2K, f(d.batSoc), f(d.batT_C) + C2K,
                  f(d.compPin_MPa) * 1e6, f(d.compPout_MPa) * 1e6, cabInt + C2K, f(d.cabAirT_C) + C2K], 1)
    u = np.stack([f(d.compNrpm), f(d.cabAirMfIn_kgs), f(d.motPumpNrpm), f(d.batPumpNrpm),
                  f(d.heaterQ_W), f(d.radFanrpm)], 1)
    ctrl = supervisor_commands(Tamb + C2K, f(d.compNrpm), f(d.motPumpNrpm),
                               f(d.clntInvTin_C) + C2K, f(d.heaterClntTout_C) + C2K)
    superheat = None
    if has("compInT_C"):
        import CoolProp.CoolProp as CP
        pin = np.clip(x[:, 5], 1e4, 4.0e6)
        tsat = np.array([CP.PropsSI("T", "P", p, "Q", 1, "R134a") for p in pin])
        superheat = np.clip(f(d.compInT_C) + C2K - tsat, 0.0, None)
    heat = (f(d.motQgen_W), f(d.invQgen_W), f(d.dcdcQgen_W), f(d.batI_A), f(d.vehSpeed_kmh))   # speed in km/h
    return _profile(int(round(abs(Tamb))), Tamb, x, u, heat, ctrl, superheat)


ORIGINAL_CYCLE = "WLTCHW"     # the paper's cycle (WLTC + 130 km/h highway); all data from before 18 Nov


def profile_key(T, cycle=ORIGINAL_CYCLE):
    """Key of a profile in the dict from load_all_profiles.

    Original cycle: |Tamb| as an int, as before (PB[12] is -12 °C), so existing code keeps working.
    Other cycles:   "<cycle>_<|Tamb|>", e.g. "UDDS_12" for UDDS at -12 °C.
    """
    T = int(abs(T))
    return T if cycle == ORIGINAL_CYCLE else f"{cycle}_{T}"


def profile_label(key, profiles=None):
    """Readable label, e.g. '-12 °C WLTCHW' or '-12 °C UDDS'."""
    cyc, T = key.rsplit("_", 1) if isinstance(key, str) else (ORIGINAL_CYCLE, key)
    return f"{-int(T):+d} °C {cyc}" if int(T) else f"0 °C {cyc}"


def parse_export_name(path):
    """(kind, ctrl, |T|, cycle) from a data file name, or None if it is not a profile file.

    Data/tuning_data_<T>.mat                         -> old export, original cycle
    training_data_<ctrl>_T<T>.mat                    -> new export, original cycle (before 18 Nov)
    training_data_<ctrl>_T<T>_<cycle>.mat            -> new export, cycle handle from RunThermalComparison
    """
    import re
    name = os.path.basename(path)
    m = re.fullmatch(r"tuning_data_(\d+)\.mat", name)
    if m:
        return "old", "rule", int(m.group(1)), ORIGINAL_CYCLE
    m = re.fullmatch(r"training_data_([a-z]+)_T(\d+)(?:_([A-Za-z0-9]+))?\.mat", name)
    if m:
        return "new", m.group(1), int(m.group(2)), m.group(3) or ORIGINAL_CYCLE
    return None


def load_all_profiles(data_dir, results_dir, cache_dir, cycles=(ORIGINAL_CYCLE,), verbose=True):
    """Rule-based profiles of the chosen drive cycles: old files in data_dir, new exports in results_dir.

    cycles: drive-cycle handles to load, e.g. ("NEDC", "UDDS", "US06"), or "all". Default: the original
            cycle only, so the notebook behaves exactly as before.
    Keys:   see profile_key: |Tamb| as int for the original cycle, "<cycle>_<|Tamb|>" for the others.
    If two files exist for the same cycle and temperature, a new export wins over an old one, then the
    newest file; the choice is printed.

    The CoolProp step takes ~15 s per profile, so each profile is cached in cache_dir. A cache file is rebuilt
    automatically when its source .mat file is newer (e.g. after re-running a simulation).
    """
    import glob, time
    os.makedirs(cache_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "tuning_data_*.mat")))
    files += sorted(glob.glob(os.path.join(results_dir, "**", "training_data_*.mat"), recursive=True))

    found = {}                                       # key -> list of (rank, path, kind, cycle)
    for path in files:
        info = parse_export_name(path)
        if info is None:
            continue
        kind, ctrl, T, cyc = info
        if ctrl != "rule" or (cycles != "all" and cyc not in cycles):
            continue
        rank = (kind == "new", os.path.getmtime(path))
        found.setdefault(profile_key(T, cyc), []).append((rank, path, kind, cyc))

    # Without the MATLAB exports (e.g. a fresh clone: Results/ is not in git) use the committed data_cache files.
    for name in sorted(os.listdir(cache_dir)):
        key = key_from_cache_name(name)
        if key is None or key in found:
            continue
        cyc = key.rsplit("_", 1)[0] if isinstance(key, str) else ORIGINAL_CYCLE
        if cycles == "all" or cyc in cycles:
            found[key] = [((False, 0), os.path.join(cache_dir, name), "cache", cyc)]

    profiles = {}
    for key, cands in found.items():
        cands.sort(reverse=True)
        _, path, kind, cyc = cands[0]
        T = abs(key) if isinstance(key, int) else int(key.rsplit("_", 1)[1])
        cache = os.path.join(cache_dir, cache_name(key))
        if kind == "cache" or (os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path)):
            prof = torch.load(cache, weights_only=False)
            how = "cache"
        else:
            t0 = time.time()
            prof = load_old_profile(path) if kind == "old" else load_new_profile(path)
            torch.save(prof, cache)
            how = f"built in {time.time() - t0:.0f} s"
        prof["cycle"] = cyc
        profiles[key] = prof
        if verbose:
            rel = os.path.relpath(path, os.path.dirname(os.path.abspath(results_dir)))
            extra = f"; also found {len(cands) - 1} older file(s), not used" if len(cands) > 1 else ""
            print(f"  {profile_label(key):>16s}  {kind}  {rel}  ({how}{extra})")
    order = lambda kv: (isinstance(kv[0], str), kv[1]["cycle"], abs(kv[1]["Tamb"]))
    return dict(sorted(profiles.items(), key=order))


HELD_EST = [n for n in EST_NAMES if n not in ("motQmeas", "invQmeas", "dcdcQmeas", "batImeas", "speed")]


def make_windows(prof, H=30, stride=30, startup_stride=None, props_every_step=False):
    """Cut a profile into H-s windows starting at k0 (from the HFM state at k0).

    Start-up rule: windows start from t = 1 s. With startup_stride (e.g. 3) the first minute gets extra windows,
    so the compressor start-up is well represented. Windows are dropped only if the HFM p_out exceeds R134a's
    critical pressure or the properties are not finite.
    Properties: held at k0 over the window (as the NMPC does), or taken from the data every second
    (props_every_step, used for long windows). Inputs and heat loads always vary per step.
    """
    N = prof["x"].shape[0]
    k0 = set(range(1, N - H, stride))
    if startup_stride:
        k0 |= set(range(1, 60, startup_stride))
    k0 = torch.tensor(sorted(k0))
    idx = k0[:, None] + torch.arange(H)[None]                     # [W, H]
    if props_every_step:
        est = {n: v[idx] for n, v in prof["est"].items()}
    else:
        est = {**{n: prof["est"][n][k0][:, None].expand(-1, H) for n in HELD_EST},
               **{n: prof["est"][n][idx] for n in ("motQmeas", "invQmeas", "dcdcQmeas", "batImeas", "speed")}}
    w = dict(T=prof["T"], Tamb=torch.full((len(k0),), prof["Tamb"], dtype=DT), k0=k0,
             x0=prof["x"][k0], x_true=prof["x"][idx + 1], u=prof["u"][idx], est=est,
             ctrl={n: v[k0][:, None].expand(-1, H) for n, v in prof["ctrl"].items()})
    ok = (w["x_true"][:, :, 6].max(1).values < P_CRIT) & (w["x0"][:, 6] < P_CRIT) & torch.isfinite(w["x_true"]).all((1, 2))
    ok &= torch.stack([torch.isfinite(v).all(1) for v in w["est"].values()], 1).all(1)
    return subset(w, torch.nonzero(ok).squeeze(1))


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


def scaled_loss(pred, true, scale=None):
    """Mean squared error per state, divided by `scale` (default: 1 K, 0.001 SOC, 1 bar)."""
    scale = STATE_SCALE if scale is None else scale
    return ((pred - true) / scale).pow(2).mean()


@torch.no_grad()
def nominal_scale(win):
    """"Balanced" loss scale: each state's 30-s RMSE with γ = 1, so every state starts at loss 1 and no state
    can be traded away for another (with plain units the battery got worse than nominal)."""
    err = simulate(win, torch.ones(win["x0"].shape[0], 11, dtype=DT)) - win["x_true"]
    return err.pow(2).mean((0, 1)).sqrt().clamp_min(1e-12)


def loss_scale(win, cap_p_out=True):
    """Balanced scale (computed on every 4th window), with p_out fixed at P_OUT_SCALE."""
    sc = nominal_scale(subset(win, torch.arange(0, win["x0"].shape[0], 4)))
    if cap_p_out:
        sc[6] = P_OUT_SCALE
    return sc


# ════════════════════════════════════════════════════════════════════ interpolation table (baseline)
class TableGamma:
    """prepParaVec.m: p_i = scatteredInterpolant(ambient, state_j) from tunedVariables (linear, nearest outside)."""
    IDX_STATE = [8, 9, 5, 2, 1, 3, 5, 9, 6, 7, 6]       # driving state (1-based) for p1..p11

    def __init__(self, tv_path):
        tv = sio.loadmat(str(tv_path), squeeze_me=True, struct_as_record=False)["tunedVariables"]
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


def train_gamma(model, wins, norm, iters, lr=3e-3, seed=0, plateau=True, log_every=10, deadline=None):
    """End-to-end training: γ = model(features) -> COM rollout -> loss against the HFM states.

    wins: list of (windows, batch, loss_scale). Each iteration draws `batch` windows from every set and averages
    the losses, so short and long windows can be trained together.
    deadline: optional wall-clock time (time.time()) at which training stops early, to cap the run time.
    """
    import time
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15) if plateau else None
    hist, t0 = [], time.time()
    for it in range(iters):
        if deadline is not None and time.time() > deadline:
            print(f"  time budget reached after {it} of {iters} iterations", flush=True)
            break
        loss = 0
        for w, batch, sc in wins:
            sub = subset(w, torch.randperm(w["x0"].shape[0])[:batch])
            loss = loss + scaled_loss(simulate(sub, model(norm(features(sub)))), sub["x_true"], sc) / len(wins)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched: sched.step(loss.item())
        hist.append(loss.item())
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"  iter {it:4d}  loss {loss.item():8.4f}  lr {opt.param_groups[0]['lr']:.1e}  {time.time() - t0:5.0f}s",
                  flush=True)
    return hist


def train_recipe(PB, train_T, log=print, budget_s=None):
    """The 3 steps of train_gamma11_step1_base.py (about 80 min on a laptop CPU, mostly step 2).

      1. 300 s windows from scratch, balanced loss                      (300 iterations)
      2. fine-tune on 300 s + 1200 s windows, p_out weighted at 3 bar   ( 50 iterations)
         -> the long windows fix the slow p_out drift in long free runs
      3. fine-tune on 300 s windows including the compressor start-up   (100 iterations)
    train_T: keys of PB to train on (|Tamb| ints for the original cycle, "<cycle>_<|T|>" for other cycles).
    budget_s: optional cap on the total training time [s]. Each step gets a share (35 / 40 / 25 %) and stops
              early when its share is used up, so the whole recipe never runs longer than budget_s.
    Returns the trained network, the input normalisation (mu, sd) and the loss history.
    """
    import time
    t_start = time.time()
    share = lambda f: None if budget_s is None else t_start + f * budget_s
    w300 = cat_windows([make_windows(PB[T], H=300, stride=20) for T in train_T])
    z = features(w300); mu, sd = z.mean(0), z.std(0) + 1e-6
    norm = lambda z: (z - mu) / sd
    net, hist = GammaNet(), []
    log("Step 1/3: 300 s windows")
    hist += train_gamma(net, [(w300, 48, loss_scale(w300, cap_p_out=False))], norm, iters=300, lr=3e-3,
                        deadline=share(0.35))
    log("Step 2/3: 300 s + 1200 s windows, p_out at 3 bar")
    w1200 = cat_windows([make_windows(PB[T], H=1200, stride=60, props_every_step=True) for T in train_T])
    hist += train_gamma(net, [(w300, 16, loss_scale(w300)), (w1200, 4, loss_scale(w1200))], norm,
                        iters=50, lr=1e-3, seed=1, plateau=False, deadline=share(0.75))
    log("Step 3/3: 300 s windows including start-up")
    w300s = cat_windows([make_windows(PB[T], H=300, stride=20, startup_stride=3) for T in train_T])
    hist += train_gamma(net, [(w300s, 32, loss_scale(w300s))], norm, iters=100, lr=1e-3, seed=2, plateau=False,
                        deadline=share(1.0))
    return net, mu, sd, hist


@torch.no_grad()
def evaluate(win, gammas):
    """{name: γ [W, 11]} -> {name: RMSE per state [9]} on the given windows."""
    return {name: horizon_rmse(simulate(win, g), win["x_true"]) for name, g in gammas.items()}


@torch.no_grad()
def free_run(profiles, gamma_fns, temps):
    """Full-hour free run of the COM from the HFM state at t = 0, no resets (paper Fig. 2 / Table 1).

    Properties are re-evaluated with CoolProp every second from the COM's OWN state (as in the paper's validation;
    with the HFM's pressures the high-side pressure never builds up). γ is recomputed every second.
    gamma_fns: {name: f(x [B, 9], k, ctx) -> γ [B, 11]}. Returns {name: states [len(temps), N, 9]}.
    All methods and temperatures run as one batch, so CoolProp dominates the run time (~3 min per 10 runs).
    """
    names = list(gamma_fns); cases = [(n, T) for n in names for T in temps]; B = len(cases)
    X = torch.stack([profiles[T]["x"] for _, T in cases]); U = torch.stack([profiles[T]["u"] for _, T in cases])
    N = X.shape[1]
    step_names = ("motQmeas", "invQmeas", "dcdcQmeas", "batImeas", "speed")
    HEAT = {n: torch.stack([profiles[T]["est"][n] for _, T in cases]) for n in step_names}
    CTRL = {n: torch.stack([profiles[T]["ctrl"][n] for _, T in cases]) for n in profiles[temps[0]]["ctrl"]}
    TA = torch.tensor([profiles[T]["Tamb"] for _, T in cases], dtype=DT)
    ctx = dict(Tamb=TA, U=U, HEAT=HEAT, N=N)
    rows = {n: [i for i, (m, _) in enumerate(cases) if m == n] for n in names}
    x = X[:, 0].clone(); traj = [x]
    for k in range(N - 1):
        xn = x.numpy()
        e = estimate_params(xn[:, 5], xn[:, 6], xn[:, 8], np.full(B, SUPERHEAT_K), TA.numpy() + C2K,
                            *(HEAT[n][:, k].numpy() for n in step_names))
        est = {n: torch.as_tensor(v, dtype=DT) for n, v in e.items() if n in EST_NAMES}
        g = torch.empty(B, 11, dtype=DT)
        for n, r in rows.items():
            sub = dict(Tamb=TA[r], U=U[r], HEAT={m: v[r] for m, v in HEAT.items()}, N=N)
            g[r] = gamma_fns[n](x[r], k, sub)
        x = rk4_step(x, U[:, k], est, {n: v[:, k] for n, v in CTRL.items()}, g)
        x = torch.cat([x[:, :5], x[:, 5:7].clamp(1e4, 6e6), x[:, 7:]], 1)
        traj.append(x)
    P = torch.stack(traj, 1)
    return {n: P[r] for n, r in rows.items()}


def network_gamma_fn(net, norm, H=30):
    """γ from the network for free_run: features from the current (predicted) state and the next H s of inputs."""
    def f(x, k, ctx):
        kk = torch.clamp(torch.arange(k, k + H), max=ctx["N"] - 1)
        return net(norm(features(dict(Tamb=ctx["Tamb"], x0=x, u=ctx["U"][:, kk],
                                      est={n: v[:, kk] for n, v in ctx["HEAT"].items()}))))
    return f
