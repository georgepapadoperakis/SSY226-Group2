"""Replicates the model-validation test of Lokur & Murgovski (Table 1):
free-running COM over the full 3600 s cold-start drive, driven by the reference
model's inputs; RMSE / MAE of T_mot, T_inv, T_dcdc, T_b, T_cair at -10, -7, -5 degC.
Parameters: identified at -5 / -10 degC, interpolated at -7 degC (paper's method),
vs. the NN scheduler (updated every second from the current operating point,
using the model's OWN predicted temperatures as state features)."""
import numpy as np, torch
import Python as C

ck = torch.load("com_output/com_scheduler.pt", weights_only=False)
R = {k: C.load_run(v) for k, v in C.RUN_FILES.items()}
model = C.COM(ck["c_ic"], ck["c_hex"]); model.load_state_dict(ck["com"])
FEAT = {k: torch.tensor(C.feature_matrix(R[k])) for k in R}
Ftr = torch.cat([FEAT[k] for k in C.TRAIN])
sch = C.Scheduler(Ftr.mean(0), Ftr.std(0) + 1e-6); sch.load_state_dict(ck["scheduler"])
env = C.Envelope(Ftr)
th_const, th_interp = ck["theta_const"], ck["theta_interp"]
STATE_FEAT = {"T_cab": 0, "T_sh": 1, "T_bat": 3, "T_cool": 8}      # feature <- model state
fidx = {k: C.FEATURES.index(k) for k in STATE_FEAT}

def free_run(run, mode):
    r = R[run]; X = torch.tensor(r["X"]); U = torch.tensor(np.stack([r[k] for k in C.INPUTS], 1))
    P = model.p(); h = model.dt / model.substeps
    x = X[0:1]; out = [x]; n_ood = 0
    with torch.no_grad():
        for k in range(len(U) - 1):
            if mode == "nn":
                f = FEAT[run][k].clone()
                for name, si in STATE_FEAT.items(): f[fidx[name]] = x[0, si]
                f = f[None]
                if env.inside(f).all(): th = sch(f)
                else: th = th_interp[None]; n_ood += 1
            else:
                th = mode[None]
            for _ in range(model.substeps):
                x = x + h * model.deriv(x, U[k:k+1], th, P)
            out.append(x)
    return torch.cat(out).numpy(), r["X"], n_ood

cols = {"T_mot": 4, "T_inv": 5, "T_dcdc": 6, "T_b": 3, "T_cair": 0}
paper = {"T_mot": [(1.60,1.35),(1.70,1.46),(1.43,1.18)], "T_inv": [(1.50,1.26),(1.53,1.33),(1.29,1.05)],
         "T_dcdc": [(2.12,1.61),(2.00,1.62),(1.83,1.30)], "T_b": [(1.61,1.36),(1.93,1.77),(1.53,1.33)],
         "T_cair": [(1.04,0.58),(1.45,1.14),(1.01,0.81)]}
runs = ["T10", "T7", "T5"]
res = {}
for run in runs:
    base_th = th_interp if run == "T7" else th_const[run]   # paper: identified at -5/-10, interpolated at -7
    for nm, mode in [("Interp/identified", base_th), ("NN scheduler", "nn")]:
        Xp, Xt, nood = free_run(run, mode)
        e = Xp - Xt
        res[(nm, run)] = {s: (np.sqrt((e[:, i]**2).mean()), np.abs(e[:, i]).mean()) for s, i in cols.items()}
        res[(nm, run, "ood")] = nood
        np.save(f"com_output/freerun_{run}_{'nn' if mode=='nn' else 'base'}.npy", Xp)

lines = ["Free-running validation over the full 3600 s cold-start drive (RMSE / MAE in degC)",
         "Same test as Lokur & Murgovski, Table 1. -10/-5 degC: identification temperatures; -7 degC: unseen.", ""]
hdr = f"{'State':8s}" + "".join(f"{'  '+r_+' ('+str(int(C.T_AMB[r_]))+' C)':>36s}" for r_ in runs)
lines += [hdr, f"{'':8s}" + "".join(f"{'paper':>12s}{'ours-interp':>12s}{'ours-NN':>12s}" for _ in runs)]
for s in cols:
    row = f"{s:8s}"
    for j, run in enumerate(runs):
        pr = paper[s][j]; a = res[("Interp/identified", run)][s]; b = res[("NN scheduler", run)][s]
        row += f"{pr[0]:6.2f}/{pr[1]:4.2f} {a[0]:6.2f}/{a[1]:4.2f} {b[0]:6.2f}/{b[1]:4.2f}"
    lines.append(row)
lines.append("")
for run in runs:
    lines.append(f"NN out-of-distribution fallback steps on {run}: {res[('NN scheduler', run, 'ood')]} of 3600")
txt = "\n".join(lines); print(txt); open("com_output/paper_table1_comparison.txt", "w").write(txt)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = R["T7"]; t = r["t"]
fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for a_, (s, i) in zip(ax.flat, cols.items()):
    a_.plot(t, r["X"][:, i] - 273.15, "k", lw=1.2, label="Simscape reference")
    a_.plot(t, np.load("com_output/freerun_T7_base.npy")[:, i] - 273.15, "C1--", label="COM, interpolated params")
    a_.plot(t, np.load("com_output/freerun_T7_nn.npy")[:, i] - 273.15, "C0", lw=0.9, label="COM, NN scheduler")
    a_.set_title(f"{s} at -7 degC (unseen), free run", fontsize=10); a_.set_ylabel("degC")
ax.flat[0].legend(fontsize=8); ax.flat[-1].axis("off")
for a_ in ax[1]: a_.set_xlabel("time [s]")
fig.tight_layout(); fig.savefig("com_output/freerun_T7_like_paper_fig2.png", dpi=110)