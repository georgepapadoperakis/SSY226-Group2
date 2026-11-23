"""
Training, 15-γ network (fixed COM), step 1 of 2: add γ12 (suction density).

COM change (com_torch.compressorMdl and dynamics_mexfreeNew.m): v_in = v_nom (2 bar / p_in)^γ12; γ12 = 0 is the
original COM. A fit of the HFM refrigerant mass flow gives γ12 ≈ 0.83: the COM mass-flow error drops from 134 % to 13 %
at −30 °C and from 36 % to 4 % on UDDS −20 °C.
Starts from models/gamma11_original_com.pt, adds a 12th output γ12 ∈ [0, 1.5] (starting at 0.83) and fine-tunes on
start-up free runs (0-300 s) + mid-run free runs (600 s from a random point) + 300-s windows.

Run from the python folder:   python train_gamma15_step1_gamma12.py [iterations=60] [budget_min=45]
Output: models/history/gamma12_step1_suction_density.pt   (≈ 40 min for 60 iterations on 2 cores)
"""
import sys, time, torch
from gamma_freerun import *

BASE = MODELS / "gamma11_original_com.pt"
OUT = MODELS / "history" / "gamma12_step1_suction_density.pt"

if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 45
    com_torch.set_fixes(False)
    PB = load_profiles(TRAIN_KEYS)
    d = torch.load(BASE, weights_only=False)
    base = GammaNet(); base.load_state_dict(d["state"]); net = GammaNet12(base)
    mu, sd = d["mu"], d["sd"]; norm = lambda z: (z - mu) / sd
    tab = PropTable.build(SUPERHEAT_K)
    t0 = time.time(); log = lambda *a: print(f"[{(time.time() - t0) / 60:5.1f} min]", *a, flush=True)
    log(f"γ12 fine-tuning: {len(TRAIN_KEYS)} runs, {iters} iterations, budget {budget} min")
    hist = finetune_g12(net, norm, PB, TRAIN_KEYS, tab, iters=iters, log=log, deadline=t0 + budget * 60)
    torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, n_gamma=12, base=BASE.name,
                    train_keys=TRAIN_KEYS, hist=hist, p_ref_suction=2e5), OUT)
    log(f"saved {OUT}")
