"""
Training, 11-γ network (original COM), step 1 of 2: the base network + the multi-drive-cycle experiment.
Run from the python/ folder:

    python train_gamma11_step1_base.py         (full run, capped at 2 h)

Next step: train_gamma11_step2_startup.py.

1. Loads all rule-based runs: WLTCHW (the paper's cycle) + NEDC, UDDS, US06, at 9 ambient temperatures.
2. Trains a new γ network with the same 3-step recipe on WLTCHW + NEDC + US06 at the training temperatures.
   UDDS is held out completely (never seen in training), as are -7, -12 and -25 °C on every cycle.
3. Evaluates three γ sources on every run:
     table           the paper's interpolation table (prepParaVec)
     net (WLTCHW)    the 17 Nov network, trained on WLTCHW only   (models/history/gamma11_wltchw_only_17nov.pt)
     net (3 cycles)  the new network from step 2                 (models/history/gamma11_step1_base.pt)
   a) 30-s horizon test (how the NMPC uses the COM), all 36 runs;
   b) full-hour free run (paper Fig. 2 / Table 1) at -5/-7/-10 °C on all cycles + all 9 temperatures on UDDS.
4. Saves everything to results/cycles_experiment_results.pt; the notebook (section 6) only loads and plots it.

The total run time is capped: training gets what is left of TRAIN_BUDGET_MIN after loading, and the free run is
done cycle by cycle (saved after each), skipping what no longer fits in TOTAL_BUDGET_MIN.
Set QUICK=1 in the environment for a 2-minute smoke test on two profiles.
"""
import os, sys, time
T_START = time.time()
QUICK = os.environ.get("QUICK") == "1"

import numpy as np
import torch
from pathlib import Path
from gamma_net import *

TOTAL_BUDGET_MIN = 3 if QUICK else 115        # hard cap for the whole script (user limit: 2 h)
TRAIN_BUDGET_MIN = 0.5 if QUICK else 70

ALL_CYCLES = (ORIGINAL_CYCLE, "NEDC", "UDDS", "US06")
TRAIN_CYCLES = (ORIGINAL_CYCLE, "NEDC", "US06")
HELDOUT_CYCLE = "UDDS"
TRAIN_T = (0, 5, 10, 15, 20, 30)
TEST_T = (7, 12, 25)
FREE_T = (5, 7, 10)                            # the paper's three temperatures, on every cycle
FREE_T_HELDOUT = (0, 5, 7, 10, 12, 15, 20, 25, 30)

DATA, RESULTS = REPO / "Data", REPO / "Results"       # Results/ (MATLAB exports) is optional: data_cache/ is enough
CACHE = Path(os.environ.get("PARTB_CACHE", DATA_CACHE))
OLD_MODEL = MODELS / "history" / "gamma11_wltchw_only_17nov.pt"
NEW_MODEL = MODELS / "history" / "gamma11_step1_base.pt"
OUT = RESULTS_PY / "cycles_experiment_results.pt"
if QUICK:
    ALL_CYCLES, TRAIN_CYCLES, TRAIN_T, TEST_T = ("NEDC", "UDDS"), ("NEDC",), (10,), ()
    FREE_T, FREE_T_HELDOUT = (10,), (10,)
    NEW_MODEL, OUT = Path(os.environ.get("TMPDIR", "/tmp")) / "quick_model.pt", Path(os.environ.get("TMPDIR", "/tmp")) / "quick_results.pt"


def log(*a):
    print(f"[{(time.time() - T_START) / 60:5.1f} min]", *a, flush=True)


def left_s():
    return TOTAL_BUDGET_MIN * 60 - (time.time() - T_START)


log(f"torch {torch.__version__}, threads {torch.get_num_threads()}, budget {TOTAL_BUDGET_MIN} min")

# ── 1. data ───────────────────────────────────────────────────────────────────────────────────────────────
PB = load_all_profiles(DATA, RESULTS, CACHE, cycles=ALL_CYCLES)
keys_of = lambda cycles, temps: [profile_key(T, c) for c in cycles for T in temps if profile_key(T, c) in PB]
train_keys = keys_of(TRAIN_CYCLES, TRAIN_T)
missing = [profile_key(T, c) for c in ALL_CYCLES for T in TRAIN_T + TEST_T if profile_key(T, c) not in PB]
log(f"{len(PB)} profiles loaded; training on {len(train_keys)}; missing: {missing or 'none'}")

def group(key):
    cyc = PB[key]["cycle"]; T = abs(PB[key]["Tamb"])
    seen_c = cyc in TRAIN_CYCLES; seen_T = round(T) in TRAIN_T
    return {(True, True): "train", (True, False): "unseen temperature",
            (False, True): "unseen cycle", (False, False): "unseen cycle + temperature"}[(seen_c, seen_T)]

# ── 2. training ───────────────────────────────────────────────────────────────────────────────────────────
table = TableGamma(DATA / "tunedVariables_minus5_minus10.mat")
old = torch.load(OLD_MODEL, weights_only=False)
net_old = GammaNet(); net_old.load_state_dict(old["state"]); net_old.eval()
norm_old = lambda z, mu=old["mu"], sd=old["sd"]: (z - mu) / sd

budget = min(TRAIN_BUDGET_MIN * 60, left_s() - (1 if QUICK else 30) * 60)
log(f"training: {len(train_keys)} profiles, budget {budget / 60:.0f} min")
t0 = time.time()
net_new, mu, sd, hist = train_recipe(PB, train_keys, log=log, budget_s=budget)
net_new.eval()
norm_new = lambda z, mu=mu, sd=sd: (z - mu) / sd
torch.save(dict(state=net_new.state_dict(), mu=mu, sd=sd, train_T=TRAIN_T, train_cycles=TRAIN_CYCLES,
                train_keys=train_keys, hist=hist, train_min=(time.time() - t0) / 60), NEW_MODEL)
log(f"training done in {(time.time() - t0) / 60:.1f} min, saved {NEW_MODEL}")

METHODS = {"table": None, "net (WLTCHW)": (net_old, norm_old), "net (3 cycles)": (net_new, norm_new)}
res = dict(train_keys=train_keys, train_cycles=TRAIN_CYCLES, heldout_cycle=HELDOUT_CYCLE, train_T=TRAIN_T,
           test_T=TEST_T, groups={k: group(k) for k in PB}, hist=hist, methods=list(METHODS),
           train_min=(time.time() - t0) / 60, horizon={}, free={}, hfm={})

# ── 3a. 30-s horizon test on every run ────────────────────────────────────────────────────────────────────
with torch.no_grad():
    for key, prof in PB.items():
        w = make_windows(prof, H=30, stride=30)
        g = {"table": table(w["Tamb"].numpy(), w["x0"].numpy())}
        for name, (net, nrm) in list(METHODS.items())[1:]:
            g[name] = net(nrm(features(w)))
        res["horizon"][key] = {n: horizon_rmse(simulate(w, gg), w["x_true"]) for n, gg in g.items()}
torch.save(res, OUT)
log("30-s horizon test done")

# ── 3b. free run, one cycle at a time ─────────────────────────────────────────────────────────────────────
fns = {"table": lambda x, k, c: table(c["Tamb"].numpy(), x.numpy())}
for name, (net, nrm) in list(METHODS.items())[1:]:
    fns[name] = network_gamma_fn(net, nrm)
plan = [(c, FREE_T_HELDOUT if c == HELDOUT_CYCLE else FREE_T) for c in (HELDOUT_CYCLE,) + tuple(
    c for c in ALL_CYCLES if c != HELDOUT_CYCLE)]
sec_per_run = None
for cyc, temps in plan:
    keys = keys_of((cyc,), temps)
    if not keys:
        continue
    need = (sec_per_run or 20) * len(keys) * len(fns)
    if sec_per_run and need > left_s() - 60:
        keys = keys[:max(0, int((left_s() - 60) / (sec_per_run * len(fns))))]
        log(f"time budget: free run on {cyc} reduced to {keys}")
        if not keys:
            continue
    t1 = time.time()
    FR = free_run(PB, fns, keys)
    sec_per_run = (time.time() - t1) / (len(keys) * len(fns))
    for i, key in enumerate(keys):
        res["free"][key] = {n: P[i].float() for n, P in FR.items()}
        res["hfm"][key] = PB[key]["x"].float()
    torch.save(res, OUT)
    log(f"free run {cyc} {[-T for T in temps if profile_key(T, cyc) in keys]} °C done ({sec_per_run:.0f} s per run)")

res["total_min"] = (time.time() - T_START) / 60
torch.save(res, OUT)
log(f"all done, results in {OUT}")
