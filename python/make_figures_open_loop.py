"""
LaTeX-style figures and tables of the open-loop test (paper Fig. 2 / Table 1 style), from
results/open_loop_test_results.pt (written by test_models_open_loop.py).

Run from the python folder:   python make_figures_open_loop.py
Output in figures/open_loop/ (PDF + PGF for LaTeX, PNG for a quick look):
    horizon30_rmse                     bar chart, 30-s horizon RMSE over the 8 test runs
    horizon30_UDDS_minus12C_first600s  30-s predictions restarted from the HFM every 30 s
    horizon30_table.tex                30-s RMSE per test run (booktabs)
    freerun_<CYCLE>_<T>                1-h free run per test run (6 panels)
    freerun_table.tex                  1-h free-run MAE per test run (booktabs)
Needs a LaTeX installation (pdflatex) for the text rendering; set USE_TEX = False to skip it.
"""
import os, shutil, subprocess
import numpy as np
import torch
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

R = torch.load(HERE / "results" / "open_loop_test_results.pt", weights_only=False)
OUT = HERE / "figures" / "open_loop"; OUT.mkdir(parents=True, exist_ok=True)
KEYS = [str(k) for k in R["test_keys"]]
S = R["free_run_s"]
METH = [("table", "Table, original COM", "#E69F00", "-.", 0.9),
        ("nn11", r"NN (11 $\gamma$), original COM", "#56B4E9", "-", 0.8),
        ("nn15", r"NN (15 $\gamma$), fixed COM", "#0072B2", "-", 1.2)]
REF_STYLE = dict(color="black", ls=(0, (4, 2)), lw=1.0)
C = lambda v: v - 273.15
BAR = lambda v: v / 1e5
PAN = [(8, r"Cabin air temperature $T_{\mathrm{cair}}$", r"$T_{\mathrm{cair}}$ ($^{\circ}$C)", C),
       (4, r"Battery temperature $T_{\mathrm{b}}$", r"$T_{\mathrm{b}}$ ($^{\circ}$C)", C),
       (0, r"Motor temperature $T_{\mathrm{mot}}$", r"$T_{\mathrm{mot}}$ ($^{\circ}$C)", C),
       (1, r"Inverter temperature $T_{\mathrm{inv}}$", r"$T_{\mathrm{inv}}$ ($^{\circ}$C)", C),
       (5, r"Low-side pressure $p_{\mathrm{in}}$", "Pressure (bar)", BAR),
       (6, r"High-side pressure $p_{\mathrm{out}}$", "Pressure (bar)", BAR)]


def file_tag(k):
    """'UDDS_12' -> 'UDDS_minus12C', '7' -> 'WLTCHW_minus7C'."""
    cyc, T = k.rsplit("_", 1) if "_" in k else ("WLTCHW", k)
    return f"{cyc}_{'0C' if T == '0' else f'minus{T}C'}"


def tex_label(k):
    cyc, T = k.rsplit("_", 1) if "_" in k else ("WLTCHW", k)
    return rf"{cyc}, ${'-' if T != '0' else ''}{T}\,^{{\circ}}$C"


def save(fig, name):
    base = str(OUT / name)
    fig.savefig(base + ".pdf")
    if USE_TEX:
        fig.savefig(base + ".pgf")
    plt.close(fig)
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", base + ".pdf", base], check=True)
    print("  ", base)


def six_panels():
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 4.3))
    return fig, axs


def finish_six(fig, axs, xlim, xticks):
    for n, (ax, (_, cap, yl, _)) in enumerate(zip(axs.flat, PAN)):
        ax.set_xlim(*xlim); ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:,}" for x in xticks]); ax.set_xlabel("Time (s)"); ax.set_ylabel(yl)
        ax.text(0.5, -0.36, f"({'abcdef'[n]}) {cap}.", transform=ax.transAxes, ha="center", va="top")
    h, l = axs[0, 0].get_legend_handles_labels(); o = [3, 0, 1, 2]
    axs[0, 0].legend([h[i] for i in o], [l[i] for i in o], loc="lower right", handlelength=2.6, borderpad=0.4,
                     labelspacing=0.25)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.15, wspace=0.32, hspace=0.62)


def best_bold(vals):
    """Format a group of three values, best (lowest after rounding) in bold."""
    b = int(np.argmin(np.round(vals, 2)))
    return [(r"\textbf{%.2f}" if i == b else "%.2f") % v for i, v in enumerate(vals)]


print("figures ->", OUT)

# ── 30-s horizon: bar chart ──────────────────────────────────────────────────────────────────────────────────
H = R["horizon"]
rm = {m: torch.cat([r["P"] - r["T"] for r in H[m].values()]).double().pow(2).mean((0, 1)).sqrt().numpy()
      for m, *_ in METH}
ST = [(8, r"$T_{\mathrm{cair}}$"), (4, r"$T_{\mathrm{b}}$"), (0, r"$T_{\mathrm{mot}}$"), (1, r"$T_{\mathrm{inv}}$"),
      (2, r"$T_{\mathrm{dcdc}}$"), (5, r"$p_{\mathrm{in}}$"), (6, r"$p_{\mathrm{out}}$")]
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.6), gridspec_kw=dict(width_ratios=[5, 2]))
for ax, sel, yl, sc in [(axs[0], ST[:5], "RMSE (K)", 1), (axs[1], ST[5:], "RMSE (bar)", 1e5)]:
    xs = np.arange(len(sel)); w = 0.26
    for j, (m, lab, c, ls, lw) in enumerate(METH):
        v = [rm[m][i] / sc for i, _ in sel]
        bars = ax.bar(xs + (j - 1) * w, v, w, color=c, edgecolor="black", linewidth=0.4, label=lab,
                      hatch=["//", "", ""][j])
        if sc > 1:
            for b, val in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.06, f"{val:.2f}", ha="center",
                        va="bottom", fontsize=6)
    ax.set_xticks(xs); ax.set_xticklabels([s for _, s in sel]); ax.set_ylabel(yl)
axs[0].text(0.5, -0.25, "(a) Temperatures.", transform=axs[0].transAxes, ha="center", va="top")
axs[1].text(0.5, -0.25, "(b) Refrigerant pressures.", transform=axs[1].transAxes, ha="center", va="top")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, handlelength=2.2, borderpad=0.4, columnspacing=1.5)
axs[1].set_ylim(0, 5.0)
fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.25, wspace=0.28)
save(fig, "horizon30_rmse")

# ── 30-s horizon: example (first 600 s of UDDS -12 °C) ──────────────────────────────────────────────────────
key, NS = "UDDS_12", 20
ref = H["nn15"][key]
Tfull = np.concatenate([np.concatenate([ref["x0"][s:s + 1].numpy(), ref["T"][s].numpy()])[:30] for s in range(NS)])
fig, axs = six_panels()
for ax, (j, cap, yl, f) in zip(axs.flat, PAN):
    for m, lab, c, ls, lw in METH:
        r = H[m][key]
        for s in range(NS):
            seg = np.concatenate([r["x0"][s:s + 1, j].numpy(), r["P"][s, :, j].numpy()])
            ax.plot(np.arange(30 * s, 30 * s + 31), f(seg), color=c, ls=ls, lw=lw, label=lab if s == 0 else None)
    ax.plot(np.arange(len(Tfull)), f(Tfull[:, j]), label="Reference (HFM)", **REF_STYLE)
    ax.plot(np.arange(0, 30 * NS, 30), f(ref["x0"][:NS, j].numpy()), "o", ms=1.6, color="black")
finish_six(fig, axs, (0, 600), [0, 150, 300, 450, 600])
save(fig, "horizon30_UDDS_minus12C_first600s")

# ── 30-s horizon: LaTeX table ────────────────────────────────────────────────────────────────────────────────
def rmse(m, keys, idx, sc=1.0):
    E = torch.cat([H[m][k]["P"] - H[m][k]["T"] for k in keys]).double()[:, :, idx]
    return float(E.pow(2).mean().sqrt()) / sc

GROUPS_H = [([0, 1, 2, 4], 1.0), ([8], 1.0), ([6], 1e5)]
rows = []
for k in KEYS + ["All"]:
    keys = KEYS if k == "All" else [k]
    cells = sum((best_bold([rmse(m, keys, idx, sc) for m, *_ in METH]) for idx, sc in GROUPS_H), [])
    rows.append(("All" if k == "All" else tex_label(k), cells))
tex = [r"% 30-s horizon RMSE, 8 test runs. Needs \usepackage{booktabs}.", r"\begin{table*}[t]", r"\centering",
       r"\caption{Prediction error over the NMPC horizon ($N = 30$, $\Delta t = 1$\,s): RMSE of 30-s predictions "
       r"started from the reference state every 30\,s. Components = mean over $T_{\mathrm{mot}}, T_{\mathrm{inv}}, "
       r"T_{\mathrm{dcdc}}, T_{\mathrm{b}}$. Tab.\ = interpolation table, original COM; NN = $\gamma$ network "
       r"(11 parameters), original COM; NN$^{\ast}$ = $\gamma$ network (15 parameters), fixed COM. Best per group in bold.}",
       r"\label{tab:horizon30}", r"\begin{tabular}{l ccc ccc ccc}", r"\toprule",
       r" & \multicolumn{3}{c}{Components (K)} & \multicolumn{3}{c}{$T_{\mathrm{cair}}$ (K)} & "
       r"\multicolumn{3}{c}{$p_{\mathrm{out}}$ (bar)} \\",
       r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
       r"Test run & Tab. & NN & NN$^{\ast}$ & Tab. & NN & NN$^{\ast}$ & Tab. & NN & NN$^{\ast}$ \\", r"\midrule"]
tex += [("\\midrule\n" if l == "All" else "") + f"{l} & {' & '.join(c)} \\\\" for l, c in rows]
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(OUT / "horizon30_table.tex").write_text("\n".join(tex) + "\n")
print("  ", OUT / "horizon30_table.tex")

# ── free run: one figure per test run ────────────────────────────────────────────────────────────────────────
F = R["free"]
idx = np.r_[0:min(300, S), 300:S:5] if S > 300 else np.arange(S)
t = idx + 1
for i, k in enumerate(KEYS):
    data = {"ref": R["hfm"][k][1:S + 1].double().numpy()}
    data.update({m: F[m]["pred"][i].double().numpy() for m, *_ in METH})
    fig, axs = six_panels()
    for ax, (j, cap, yl, f) in zip(axs.flat, PAN):
        for m, lab, c, ls, lw in METH:
            ax.plot(t, f(data[m][idx, j]), label=lab, color=c, ls=ls, lw=lw)
        ax.plot(t, f(data["ref"][idx, j]), label="Reference (HFM)", **REF_STYLE)
    ticks = [0, 1000, 2000, 3000] if S >= 3000 else list(range(0, S + 1, max(S // 4, 1)))
    finish_six(fig, axs, (0, S), ticks)
    save(fig, f"freerun_{file_tag(k)}")

# ── free run: LaTeX table ────────────────────────────────────────────────────────────────────────────────────
GROUPS_F = ["components", "cab_air", "cab_int", "p_in", "p_out"]
rows = []
for i, k in enumerate(KEYS):
    cells = sum((best_bold([float(F[m][g][i]) for m, *_ in METH]) for g in GROUPS_F), [])
    rows.append((tex_label(k), cells))
cells = sum((best_bold([float(F[m][g].mean()) for m, *_ in METH]) for g in GROUPS_F), [])
rows.append(("All", cells))
tex = [r"% 1-h free-run MAE, 8 test runs. Needs \usepackage{booktabs}.", r"\begin{table*}[t]", r"\centering",
       r"\caption{One-hour free run of the COM from the reference state at $t=0$ (no resets), driven by the inputs of "
       r"the rule-based reference simulation: mean absolute error against the reference model. Components = mean over "
       r"$T_{\mathrm{mot}}, T_{\mathrm{inv}}, T_{\mathrm{dcdc}}, T_{\mathrm{b}}$. Tab.\ = interpolation table, original "
       r"COM; NN = $\gamma$ network (11 parameters), original COM; NN$^{\ast}$ = $\gamma$ network (15 parameters), "
       r"fixed COM. Best per group in bold.}",
       r"\label{tab:freerun}", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{l ccc ccc ccc ccc ccc}", r"\toprule",
       r" & \multicolumn{3}{c}{Components ($^{\circ}$C)} & \multicolumn{3}{c}{$T_{\mathrm{cair}}$ ($^{\circ}$C)} & "
       r"\multicolumn{3}{c}{$T_{\mathrm{cint}}$ ($^{\circ}$C)} & \multicolumn{3}{c}{$p_{\mathrm{in}}$ (bar)} & "
       r"\multicolumn{3}{c}{$p_{\mathrm{out}}$ (bar)} \\",
       r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}\cmidrule(lr){14-16}",
       "Test run" + r" & Tab. & NN & NN$^{\ast}$" * 5 + r" \\", r"\midrule"]
tex += [("\\midrule\n" if l == "All" else "") + f"{l} & {' & '.join(c)} \\\\" for l, c in rows]
tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(OUT / "freerun_table.tex").write_text("\n".join(tex) + "\n")
print("  ", OUT / "freerun_table.tex")
