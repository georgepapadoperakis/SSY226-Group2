"""
Closed-loop (NMPC) figures and table: rule-based controller vs NMPC with the three prediction models.

Reads the MATLAB results in ../Results/ (written by RunThermalComparison / RunAllComModels; not in git).
By default it takes, for the chosen cycle and temperature, the NEWEST folder of each kind:
    run_<time>_<CYCLE>_rule            rule-based controller
    run_<time>_<CYCLE>_nmpc-table      NMPC, original COM + interpolation table (paper setup)
    run_<time>_<CYCLE>_nmpc-nn11       NMPC, original COM + 11-γ network
    run_<time>_<CYCLE>_nmpc-nn15       NMPC, fixed COM + 15-γ network
Missing kinds are skipped. Folders can also be given explicitly.

Run from the python folder:
    python make_figures_nmpc.py                         WLTCHW, -10 °C, newest runs
    python make_figures_nmpc.py --temp -7 --cycle UDDS
    python make_figures_nmpc.py --rule <folder> --table <folder> --nn11 <folder> --nn15 <folder>
Output in figures/nmpc/ (PDF + PGF for LaTeX, PNG for a quick look):
    nmpc_<CYCLE>_<T>_states_inputs   cabin, battery, heater, compressor, blower, p_out over the hour
    nmpc_<CYCLE>_<T>_energy          energy per controller, split by consumer
    nmpc_<CYCLE>_<T>_table.tex       energy, comfort, battery and solver time (booktabs)
"""
import argparse, glob, os, re, shutil, subprocess
import numpy as np
import scipy.io as sio
import matplotlib
from pathlib import Path

HERE = Path(__file__).absolute().parent
USE_TEX = shutil.which("pdflatex") is not None
if USE_TEX:
    matplotlib.use("pgf")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "legend.fontsize": 6.5, "legend.frameon": True,
                     "legend.fancybox": False, "legend.edgecolor": "black", "legend.framealpha": 1.0})
if USE_TEX:
    plt.rcParams.update({"pgf.texsystem": "pdflatex", "text.usetex": True, "pgf.rcfonts": False,
                         "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{mathptmx}"})

KINDS = [("rule", "Rule-based", "rule", dict(color="black", ls=(0, (4, 2)), lw=1.0)),
         ("table", "NMPC, table, original COM", "nmpc-table", dict(color="#E69F00", ls="-.", lw=0.9)),
         ("nn11", r"NMPC, NN (11 $\gamma$), original COM", "nmpc-nn11", dict(color="#56B4E9", ls="-", lw=0.9)),
         ("nn15", r"NMPC, NN (15 $\gamma$), fixed COM", "nmpc-nn15", dict(color="#0072B2", ls="-", lw=1.1))]
PAPER = {-5: (2280, 1820), -7: (2570, 1900), -10: (3020, 2180)}    # paper Table 2: baseline, NMPC [Wh]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--temp", type=int, default=-10)
ap.add_argument("--cycle", default="WLTCHW")
ap.add_argument("--results", default=str(HERE.parent / "Results"))
for k, *_ in KINDS:
    ap.add_argument(f"--{k}", default=None, help=f"run folder for {k} (default: newest matching)")
args = ap.parse_args()
T, CYC = args.temp, args.cycle
TT = f"T{abs(T)}"
ctrl_of = lambda k: "rule" if k == "rule" else "nmpc"


def newest(tag):
    cands = sorted(glob.glob(os.path.join(args.results, f"run_*_{CYC}_*")))
    cands = [c for c in cands if re.search(rf"_{CYC}_(.*\+)?{re.escape(tag)}(\+.*)?$", os.path.basename(c))]
    cands = [c for c in cands if glob.glob(os.path.join(c, f"training_data_{'rule' if tag == 'rule' else 'nmpc'}_{TT}*.mat"))]
    return cands[-1] if cands else None


RUNS = {}
for k, lab, tag, sty in KINDS:
    folder = getattr(args, k) or newest(tag)
    if folder is None:
        print(f"  {k:5s}: no run found (Results/run_*_{CYC}_{tag}) - skipped")
        continue
    f = glob.glob(os.path.join(folder, f"training_data_{ctrl_of(k)}_{TT}*.mat"))[0]
    d = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["trainingData"]
    cr = sio.loadmat(os.path.join(folder, "comparison_results.mat"), squeeze_me=True, struct_as_record=False)["results"]
    E = getattr(getattr(cr, ctrl_of(k)), TT)
    timing = None
    tf = os.path.join(folder, f"mpc_timing_nmpc_{TT}.mat")
    if k != "rule" and os.path.exists(tf):
        s = sio.loadmat(tf, squeeze_me=True, struct_as_record=False)["mpc_timing"]
        st = np.asarray(s.solver_times_ms, float); timing = (np.nanmean(st), np.nanmax(st))
    RUNS[k] = dict(label=lab, style=sty, d=d, E=E, timing=timing, folder=folder)
    print(f"  {k:5s}: {os.path.basename(folder)}")
if not RUNS:
    raise SystemExit("no runs found")

OUT = HERE / "figures" / "nmpc"; OUT.mkdir(parents=True, exist_ok=True)
NAME = f"nmpc_{CYC}_{'0C' if T == 0 else f'minus{abs(T)}C'}"
ma = lambda v, w=60: np.convolve(np.asarray(v, float), np.ones(w) / w, mode="same")


def save(fig, name):
    base = str(OUT / name)
    fig.savefig(base + ".pdf")
    if USE_TEX:
        fig.savefig(base + ".pgf")
    plt.close(fig)
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", base + ".pdf", base], check=True)
    print("  ->", base)


# ── states and inputs over the hour ─────────────────────────────────────────────────────────────────────────
PAN = [(lambda d: np.asarray(d.cabAirT_C, float), r"Cabin air temperature $T_{\mathrm{cair}}$", r"$T_{\mathrm{cair}}$ ($^{\circ}$C)", False),
       (lambda d: np.asarray(d.batT_C, float), r"Battery temperature $T_{\mathrm{b}}$", r"$T_{\mathrm{b}}$ ($^{\circ}$C)", False),
       (lambda d: ma(d.heaterQ_W) / 1e3, "Heater power (60-s mean)", r"$P_{\mathrm{heater}}$ (kW)", True),
       (lambda d: np.asarray(d.compNrpm, float), "Compressor speed", r"$n_{\mathrm{comp}}$ (rpm)", False),
       (lambda d: ma(d.cabAirMfIn_kgs), "Blower air mass flow (60-s mean)", r"$\dot m_{\mathrm{air}}$ (kg/s)", True),
       (lambda d: ma(np.asarray(d.compPout_MPa, float) * 10), r"High-side pressure $p_{\mathrm{out}}$ (60-s mean)", r"$p_{\mathrm{out}}$ (bar)", True)]
order = [k for k in ("table", "nn11", "nn15", "rule") if k in RUNS]
fig, axs = plt.subplots(2, 3, figsize=(7.16, 4.5))
for n, (ax, (f, cap, yl, smooth)) in enumerate(zip(axs.flat, PAN)):
    for k in order:
        d = RUNS[k]["d"]; t = np.asarray(d.t_s, float); y = f(d)
        m = (t >= 30) & (t <= t[-1] - 30) if smooth else slice(None)
        ax.plot(t[m], y[m], label=RUNS[k]["label"], **RUNS[k]["style"])
    if n == 0:
        ax.axhline(21, color="grey", lw=0.5, ls=":")
    tmax = max(float(np.asarray(RUNS[k]["d"].t_s)[-1]) for k in order)
    ax.set_xlim(0, tmax); ax.set_xticks([x for x in (0, 1000, 2000, 3000) if x <= tmax])
    ax.set_xticklabels([f"{x:,}" for x in (0, 1000, 2000, 3000) if x <= tmax])
    ax.set_xlabel("Time (s)"); ax.set_ylabel(yl)
    ax.text(0.5, -0.36, f"({'abcdef'[n]}) {cap}.", transform=ax.transAxes, ha="center", va="top")
h, l = axs[0, 0].get_legend_handles_labels()
o = [order.index(k) for k in ("rule", "table", "nn11", "nn15") if k in order]
fig.legend([h[i] for i in o], [l[i] for i in o], loc="upper center", ncol=4, handlelength=2.4, borderpad=0.4,
           columnspacing=1.2)
fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.14, wspace=0.34, hspace=0.62)
save(fig, f"{NAME}_states_inputs")

# ── energy bars ──────────────────────────────────────────────────────────────────────────────────────────────
keys = [k for k in ("rule", "table", "nn11", "nn15") if k in RUNS]
names = {"rule": "Rule-\nbased", "table": "NMPC\ntable", "nn11": "NMPC\nNN 11$\\gamma$", "nn15": "NMPC\nNN 15$\\gamma$"}
comp = [RUNS[k]["E"].E_comp_Wh for k in keys]; heat = [RUNS[k]["E"].E_heater_Wh for k in keys]
oth = [RUNS[k]["E"].E_total_Wh - RUNS[k]["E"].E_comp_Wh - RUNS[k]["E"].E_heater_Wh for k in keys]
fig, ax = plt.subplots(figsize=(3.5, 2.6)); x = np.arange(len(keys))
ax.bar(x, comp, 0.6, color="#0072B2", edgecolor="black", lw=0.4, label="Compressor")
ax.bar(x, heat, 0.6, bottom=comp, color="#E69F00", edgecolor="black", lw=0.4, hatch="//", label="Heater")
ax.bar(x, oth, 0.6, bottom=np.add(comp, heat), color="#BBBBBB", edgecolor="black", lw=0.4, label="Pumps, fan, blower")
E_rule = RUNS["rule"]["E"].E_total_Wh if "rule" in RUNS else None
for i, k in enumerate(keys):
    tot = RUNS[k]["E"].E_total_Wh
    s = "" if (k == "rule" or E_rule is None) else f"\n(${100 * (tot / E_rule - 1):.1f}$\\,\\%)"
    ax.text(i, tot + 40, f"{tot:,.0f}\\,Wh{s}", ha="center", va="bottom", fontsize=6.5)
if CYC == "WLTCHW" and T in PAPER:
    ax.axhline(PAPER[T][1], color="black", lw=0.6, ls=":", label=f"Paper NMPC ({PAPER[T][1]} Wh)")
ax.set_xticks(x); ax.set_xticklabels([names[k] for k in keys]); ax.set_ylabel("Energy (Wh)")
ax.set_ylim(0, 1.33 * max(RUNS[k]["E"].E_total_Wh for k in keys))
ax.legend(loc="upper right", ncol=2, fontsize=6, handlelength=1.5, columnspacing=0.8)
fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.2)
save(fig, f"{NAME}_energy")


# ── LaTeX table ──────────────────────────────────────────────────────────────────────────────────────────────
def metrics(k):
    d = RUNS[k]["d"]; t = np.asarray(d.t_s, float); c = np.asarray(d.cabAirT_C, float)
    first = lambda thr: f"{t[np.argmax(c >= thr)]:.0f}" if (c >= thr).any() else "--"
    return dict(E=RUNS[k]["E"].E_total_Wh, Ec=RUNS[k]["E"].E_comp_Wh, Eh=RUNS[k]["E"].E_heater_Wh,
                cab=c[t >= 600].mean(), t20=first(20), t21=first(21), bat=np.asarray(d.batT_C, float)[-1])


M = {k: metrics(k) for k in keys}
rel = lambda k: "" if (k == "rule" or E_rule is None) else f" (${100 * (M[k]['E'] / E_rule - 1):.1f}$\\,\\%)"
paper = PAPER.get(T) if CYC == "WLTCHW" else None
pcol = paper is not None
rows = [("Total energy (Wh)", [f"{M[k]['E']:,.0f}".replace(",", "\\,") + rel(k) for k in keys],
         f"{paper[1]} (${100 * (paper[1] / paper[0] - 1):.1f}$\\,\\%)" if pcol else None),
        ("Compressor energy (Wh)", [f"{M[k]['Ec']:.0f}" for k in keys], "--"),
        ("Heater energy (Wh)", [f"{M[k]['Eh']:.0f}" for k in keys], "--"),
        (r"Mean $T_{\mathrm{cair}}$, 600\,s to end ($^{\circ}$C)", [f"{M[k]['cab']:.2f}" for k in keys], "--"),
        (r"First $T_{\mathrm{cair}}\geq 20\,^{\circ}$C (s)", [M[k]["t20"] for k in keys], "--"),
        (r"First $T_{\mathrm{cair}}\geq 21\,^{\circ}$C (s)", [M[k]["t21"] for k in keys], "--"),
        (r"$T_{\mathrm{b}}$ at the end ($^{\circ}$C)", [f"{M[k]['bat']:.1f}" for k in keys], "--"),
        ("IPOPT time, mean / max (ms)", ["--" if RUNS[k]["timing"] is None else
                                         f"{RUNS[k]['timing'][0]:.0f} / {RUNS[k]['timing'][1]:.0f}" for k in keys], "--")]
head = {"rule": "Rule-based", "table": "NMPC, table", "nn11": r"NMPC, NN 11\,$\gamma$", "nn15": r"NMPC, NN 15\,$\gamma$"}
sub = {"rule": "", "table": "original COM", "nn11": "original COM", "nn15": "fixed COM"}
Tlab = f"{'-' if T else ''}{abs(T)}"
tex = [rf"% Closed-loop NMPC results, {CYC} {T} C. Needs \usepackage{{booktabs}}.", r"\begin{table*}[t]", r"\centering",
       rf"\caption{{Closed-loop results on {CYC} at ${Tlab}\,^{{\circ}}$C (HFM as plant, $N=30$, $\Delta t=1$\,s). "
       r"Table = interpolation table (paper setup); NN = $\gamma$ network." + (" Paper values from Lokur and Murgovski (Table 2)." if pcol else "") + "}",
       rf"\label{{tab:{NAME}}}", r"\begin{tabular}{l " + "c" * (len(keys) + pcol) + "}", r"\toprule",
       " & " + " & ".join(head[k] for k in keys) + (" & Paper NMPC" if pcol else "") + r" \\",
       " & " + " & ".join(sub[k] for k in keys) + (" & " if pcol else "") + r" \\", r"\midrule"]
tex += [f"{a} & " + " & ".join(v) + (f" & {p}" if pcol else "") + r" \\" for a, v, p in rows]
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(OUT / f"{NAME}_table.tex").write_text("\n".join(tex) + "\n")
print("  ->", OUT / f"{NAME}_table.tex")
