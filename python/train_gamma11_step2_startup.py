"""
Training, 11-γ network (original COM), step 2 of 2: start-up fine-tuning as a FREE RUN.

Loads models/history/gamma11_step1_base.pt (from train_gamma11_step1_base.py) and fine-tunes it on the first 300 s of
every training run rolled out as a free run (γ and fluid properties from the COM's own state), plus the usual
300-s window loss. Fluid properties come from fluid_props_table.PropTable (lookup tables built once from CoolProp).

Run from the python folder:   python train_gamma11_step2_startup.py [iterations=80] [budget_min=30]
Output: models/gamma11_original_com.pt   (the final 11-γ network; export it for MATLAB with export_networks_to_matlab.py)
About 25 min for 80 iterations on 2 CPU cores.
"""
import sys, time, torch
from gamma_freerun import *

BASE = MODELS / "history" / "gamma11_step1_base.pt"
OUT = MODELS / "gamma11_original_com.pt"

if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    budget_min = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    com_torch.set_fixes(False)                                   # original COM
    PB = load_profiles(TRAIN_KEYS)
    d = torch.load(BASE, weights_only=False)
    net = GammaNet(); net.load_state_dict(d["state"])
    mu, sd = d["mu"], d["sd"]; norm = lambda z: (z - mu) / sd
    tab = PropTable.build(SUPERHEAT_K)
    t0 = time.time()
    log = lambda *a: print(f"[{(time.time() - t0) / 60:5.1f} min]", *a, flush=True)
    log(f"fine-tuning on {len(TRAIN_KEYS)} runs, {iters} iterations, budget {budget_min} min")
    hist, sc = finetune_startup(net, norm, PB, TRAIN_KEYS, tab, iters=iters, log=log, deadline=t0 + budget_min * 60)
    torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, train_keys=TRAIN_KEYS, base=BASE.name,
                    startup_hist=hist, startup_scale=sc, startup_s=300), OUT)
    log(f"saved {OUT}")
