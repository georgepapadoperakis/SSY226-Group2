"""
Training, 15-γ network (fixed COM), step 2 of 2: COM fixes + γ13..γ15.

COM fixes (com_torch.set_fixes, all without new states; evidence in the notes "COM fault analysis"):
  front heat exchanger active (HFM-fitted conductance), both low-side HXs at T_sat(p_in), HTX UA × 2.62,
  inner condenser at T_sat(p_out) with HFM-fitted air-side UA, heater 100 % into coolant,
  motor / inverter -> coolant UA × 1.59 / 5.15 (calibrated at γ = 1 on the training runs, teacher-forced),
  cabin constants refitted (com_torch.CAB_K).
New γ: γ13 HTX UA, γ14 motor UA, γ15 inverter UA, each in [0.1, 10], starting at 1.
Starts from models/history/gamma12_step1_suction_density.pt and fine-tunes (300-s windows, then start-up and mid-run
free runs) on the fixed COM. Then free-runs the 8 test runs for one hour.

Run from the python folder:   python train_gamma15_step2_fixed_com.py [iterations=80] [budget_min=60] [warm] [reset_cabin | reset_scaled | reset_g11_g15]
("warm": start from the existing models/gamma15_fixed_com.pt instead of the γ12 network; the reset_* options were
used once each when constants in com_torch were refitted/rescaled)
Output: models/gamma15_fixed_com.pt, results/gamma15_training_eval.pt
"""
import sys, time, math, torch
from torch import nn
from gamma_freerun import *

BASE = MODELS / "history" / "gamma12_step1_suction_density.pt"
OUT = MODELS / "gamma15_fixed_com.pt"
EVAL_OUT = RESULTS_PY / "gamma15_training_eval.pt"
evaluate = free_run_eval


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    t0 = time.time(); log = lambda *a: print(f"[{(time.time() - t0) / 60:5.1f} min]", *a, flush=True)
    keys = list(dict.fromkeys(list(TRAIN_KEYS) + TEST_KEYS))
    PB = load_profiles(keys)
    fill_speed(PB, load_profiles([0])[0])
    d = torch.load(BASE, weights_only=False)
    mu, sd = d["mu"], d["sd"]; norm = lambda z: (z - mu) / sd
    base = GammaNet(); net12 = GammaNet12(base); net12.load_state_dict(d["state"])
    tab = PropTable.build(SUPERHEAT_K)
    tests = [PB[k] for k in TEST_KEYS]

    com_torch.set_fixes(False)
    log("baseline: γ12 network on the original COM, 1-h free runs of the 8 test runs")
    ev0 = evaluate(net12, norm, tests, tab)
    com_torch.set_fixes(True, **K_CAL)
    net = GammaNet15(net12)
    if len(sys.argv) > 3 and sys.argv[3] == "warm":          # warm start from the previous fixes network
        net.load_state_dict(torch.load(OUT, weights_only=False)["state"])
        log(f"warm start from {OUT}")
        if len(sys.argv) > 4 and sys.argv[4] == "reset_cabin":      # cabin refitted: γ1, γ2, γ8 back to 1
            with torch.no_grad():
                for i in (0, 1, 7):
                    net.net12.base.net[-1].weight[i].zero_(); net.net12.base.net[-1].bias[i].zero_()
            log("γ1, γ2, γ8 reset to 1 (cabin constants refitted)")
        if len(sys.argv) > 4 and sys.argv[4] == "reset_scaled":     # constants behind γ4, γ7, γ9, γ10, γ15 rescaled
            with torch.no_grad():
                for i in (3, 6, 8, 9):
                    net.net12.base.net[-1].weight[i].zero_(); net.net12.base.net[-1].bias[i].zero_()
                net.head.weight[2].zero_(); net.head.bias[2].zero_()
            log("γ4, γ7, γ9, γ10, γ15 reset to 1 (constants rescaled, com_torch.GAMMA_SCALE / K_INV)")
        if len(sys.argv) > 4 and sys.argv[4] == "reset_g11_g15":    # second rescaling round
            with torch.no_grad():
                net.net12.base.net[-1].weight[10].zero_(); net.net12.base.net[-1].bias[10].zero_()
                net.head.weight[2].zero_(); net.head.bias[2].zero_()
            log("γ11, γ15 reset to 1 (constants rescaled)")
    log("γ12 network + γ13..15 = 1 on the fixed COM (before fine-tuning)")
    ev1 = evaluate(net, norm, tests, tab)
    for n, ev in (("orig COM", ev0), ("fixed, untrained", ev1)):
        log(f"  {n:17s} comp {ev['components'].mean():.2f} °C  cab air {ev['cab_air'].mean():.2f} °C  "
            f"p_in {ev['p_in'].mean():.2f} bar  p_out {ev['p_out'].mean():.2f} bar  |p_in drift| {ev['p_in_drift'].abs().mean():.2f} bar/h")

    # γ1..γ11 were tuned to hide the old faults, so first re-fit all γ on windows (like train_recipe steps 1 and 3,
    # but warm-started), then the free-run fine-tuning of the γ12 recipe (its loss scales need a reasonable start).
    TR = {k: PB[k] for k in train_keys}
    w300 = cat_windows([make_windows(TR[k], H=300, stride=20) for k in train_keys])
    log("step 1/3: 300 s windows, all 15 γ (warm start)")
    hist = train_gamma(net, [(w300, 48, loss_scale(w300, cap_p_out=False))], norm, iters=150, lr=2e-3,
                       deadline=time.time() + 0.35 * budget * 60)
    w300s = cat_windows([make_windows(TR[k], H=300, stride=20, startup_stride=3) for k in train_keys])
    log("step 2/3: 300 s windows incl. start-up, p_out at 3 bar")
    hist += train_gamma(net, [(w300s, 32, loss_scale(w300s))], norm, iters=60, lr=1e-3, seed=2, plateau=False,
                        deadline=time.time() + 0.2 * budget * 60)
    log(f"step 3/3: free-run fine-tuning (γ12 recipe), {iters} iterations")
    hist = (hist, finetune_g12(net, norm, TR, train_keys, tab, iters=iters, log=log,
                               deadline=time.time() + 0.45 * budget * 60))
    torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, n_gamma=15, base=BASE.name,
                    com_fixes=K_CAL, front_g=com_torch.FRONT_G, train_keys=train_keys, hist=hist),
               OUT)
    log(f"saved {OUT}; evaluating")
    ev2 = evaluate(net, norm, tests, tab)
    log(f"  {'fixed, trained':17s} comp {ev2['components'].mean():.2f} °C  cab air {ev2['cab_air'].mean():.2f} °C  "
        f"p_in {ev2['p_in'].mean():.2f} bar  p_out {ev2['p_out'].mean():.2f} bar  |p_in drift| {ev2['p_in_drift'].abs().mean():.2f} bar/h")
    torch.save(dict(test_keys=TEST_KEYS, orig=ev0, fixed_untrained=ev1, fixed_trained=ev2), EVAL_OUT)
    log(f"saved {EVAL_OUT}")
