"""
Open-loop test of the three ways to get the COM's tuning parameters γ, on the 8 held-out test runs
(UDDS at 0/-7/-12/-20/-25 °C, WLTCHW -7 °C, NEDC -12 °C, US06 -7 °C; none of them used in training):

    table  the paper's interpolation table (prepParaVec), ORIGINAL COM, 11 γ
    nn11   models/gamma11_original_com.pt,                ORIGINAL COM, 11 γ
    nn15   models/gamma15_fixed_com.pt,                   FIXED COM,    15 γ

Two tests, both against the HFM (Simscape) driven by the rule-based controller:
    1. 30-s horizon: restart the COM from the HFM state every 30 s and predict 30 s (how the NMPC uses the COM).
       Reported as RMSE over all 30-s predictions.
    2. 1-h free run: start from the HFM state at t = 0 only, then use the COM's own state for one hour.
       Reported as mean absolute error.

Run from the python folder:   python test_models_open_loop.py           (about 3 min)
                              python test_models_open_loop.py quick     (free run only 600 s, about 1 min)
Output: results/open_loop_test_results.pt   (read by make_figures_open_loop.py)
Expected (full run): 30-s cabin-air RMSE 0.61 / 0.31 / 0.31 K, free-run components 2.31 / 2.72 / 0.33 °C.
"""
import sys, time
import torch
from gamma_freerun import *

torch.set_grad_enabled(False)
QUICK = len(sys.argv) > 1 and sys.argv[1] == "quick"
S_FREE = 600 if QUICK else 3600
OUT = RESULTS_PY / ("open_loop_test_results_quick.pt" if QUICK else "open_loop_test_results.pt")
METHODS = ("table", "nn11", "nn15")
NAMES = {"table": "Table, original COM", "nn11": "NN 11 γ, original COM", "nn15": "NN 15 γ, fixed COM"}

t0 = time.time()
log = lambda *a: print(f"[{(time.time() - t0) / 60:4.1f} min]", *a, flush=True)

PB = load_profiles(TEST_KEYS + [0])
fill_speed(PB, PB[0])                       # old WLTCHW exports have no speed signal (needed by the fixed COM)
profs = [PB[k] for k in TEST_KEYS]
table = TableGamma(REPO / "Data" / "tunedVariables_minus5_minus10.mat")
nn11, norm11 = load_gamma11()
nn15, norm15 = load_gamma15()
tab = PropTable.build(SUPERHEAT_K)


def use_com(method):
    """Select the COM version that belongs to a method."""
    if method == "nn15":
        com_torch.set_fixes(True, **K_CAL)
    else:
        com_torch.set_fixes(False)


def window_gamma(method, W):
    if method == "table":
        return table(W["Tamb"].numpy(), W["x0"].numpy())
    if method == "nn11":
        return nn11(norm11(features(W)))
    return nn15(norm15(features(W)))


def table_free_run(S):
    """Free run with γ from the table every second (from the COM's own state)."""
    X = torch.stack([p["x"] for p in profs]); U = torch.stack([p["u"] for p in profs])
    HEAT = {n: torch.stack([p["est"][n] for p in profs]) for n in STEP_NAMES}
    CTRL = {n: torch.stack([p["ctrl"][n] for p in profs]) for n in profs[0]["ctrl"]}
    TA = torch.tensor([p["Tamb"] for p in profs], dtype=DT)
    x = X[:, 0].clone(); out = []
    for k in range(S):
        g = table(TA.numpy(), x.numpy())
        est = tab(x, {n: v[:, k] for n, v in HEAT.items()}, TA + C2K)
        x = rk4_step(x, U[:, k], est, {n: v[:, k] for n, v in CTRL.items()}, g)
        x = torch.cat([x[:, :3], x[:, 3:4].clamp(0, 1.2), x[:, 4:5], x[:, 5:7].clamp(1e4, 6e6), x[:, 7:]], 1)
        out.append(x)
    return torch.stack(out, 1)


def free_run_errors(P, S):
    T = torch.stack([p["x"][1:S + 1] for p in profs]); e = (P - T).abs()
    return dict(components=e[:, :, [0, 1, 2, 4]].mean((1, 2)), cab_air=e[:, :, 8].mean(1), cab_int=e[:, :, 7].mean(1),
                p_in=e[:, :, 5].mean(1) / 1e5, p_out=e[:, :, 6].mean(1) / 1e5, pred=P.float())


res = dict(test_keys=TEST_KEYS, labels=[profile_label(k) for k in TEST_KEYS], methods=METHODS, names=NAMES,
           free_run_s=S_FREE, horizon={}, free={},
           hfm={str(k): PB[k]["x"].float() for k in TEST_KEYS})

# ── 1. 30-s horizon ─────────────────────────────────────────────────────────────────────────────────────────
for m in METHODS:
    use_com(m); res["horizon"][m] = {}
    for k in TEST_KEYS:
        W = make_windows(PB[k], H=30, stride=30)
        res["horizon"][m][str(k)] = dict(P=simulate(W, window_gamma(m, W)).float(), T=W["x_true"].float(),
                                         x0=W["x0"].float())
    log(f"30-s horizon test done: {NAMES[m]}")

# ── 2. free run ─────────────────────────────────────────────────────────────────────────────────────────────
for m in METHODS:
    use_com(m)
    if m == "table":
        res["free"][m] = free_run_errors(table_free_run(S_FREE), S_FREE)
    else:
        net, norm = (nn11, norm11) if m == "nn11" else (nn15, norm15)
        res["free"][m] = free_run_errors(startup_rollout(net, norm, profs, tab, S_FREE), S_FREE)
    log(f"{S_FREE}-s free run done: {NAMES[m]}")
com_torch.set_fixes(False)
torch.save(res, OUT)

# ── summary ─────────────────────────────────────────────────────────────────────────────────────────────────
ST = [("T_mot", 0, 1), ("T_inv", 1, 1), ("T_dcdc", 2, 1), ("T_b", 4, 1), ("T_cair", 8, 1), ("p_in", 5, 1e5), ("p_out", 6, 1e5)]
print("\n30-s horizon, RMSE over the 8 test runs (K, bar)")
print(f"{'':24s}" + "".join(f"{n:>9s}" for n, _, _ in ST))
for m in METHODS:
    E = torch.cat([r["P"] - r["T"] for r in res["horizon"][m].values()]).double()
    rm = E.pow(2).mean((0, 1)).sqrt()
    print(f"{NAMES[m]:24s}" + "".join(f"{rm[i] / sc:9.2f}" for _, i, sc in ST))
G = [("components", "comp. °C"), ("cab_air", "T_cair °C"), ("cab_int", "T_cint °C"), ("p_in", "p_in bar"), ("p_out", "p_out bar")]
print(f"\n{S_FREE}-s free run, mean absolute error over the 8 test runs")
print(f"{'':24s}" + "".join(f"{n:>11s}" for _, n in G))
for m in METHODS:
    print(f"{NAMES[m]:24s}" + "".join(f"{float(res['free'][m][g].mean()):11.2f}" for g, _ in G))
log(f"saved {OUT}")
