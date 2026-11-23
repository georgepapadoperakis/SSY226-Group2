"""
Tables and figures for the multi-drive-cycle experiment (notebook section 6).

Everything here only reads results/cycles_experiment_results.pt (written by train_gamma11_step1_base.py) and the cached profiles,
so the notebook can show the results in seconds without re-training.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

C2K = 273.15

# Colours (validated categorical slots 1–3, safe for colour-blind readers); the HFM is always black dashed.
COLORS = {"table": "#eb6834", "net (WLTCHW)": "#1baf7a", "net (3 cycles)": "#2a78d6"}
NICE = {"table": "Table (paper)", "net (WLTCHW)": "Network, WLTCHW only (17 Nov)",
        "net (3 cycles)": "Network, 3 cycles (new)"}
SHORT = {"table": "Table", "net (WLTCHW)": "Net (WLTCHW)", "net (3 cycles)": "Net (3 cycles)"}

# (label, state index, scale to report units) – temperatures in °C (= K differences), pressures in bar
STATES = [("$T_\\mathrm{mot}$", 0, 1.0), ("$T_\\mathrm{inv}$", 1, 1.0), ("$T_\\mathrm{dcdc}$", 2, 1.0),
          ("$T_\\mathrm{b}$", 4, 1.0), ("$T_\\mathrm{cair}$", 8, 1.0), ("$p_\\mathrm{in}$", 5, 1e-5),
          ("$p_\\mathrm{out}$", 6, 1e-5)]
PLAIN = {"$T_\\mathrm{mot}$": "T_mot (°C)", "$T_\\mathrm{inv}$": "T_inv (°C)", "$T_\\mathrm{dcdc}$": "T_dcdc (°C)",
         "$T_\\mathrm{b}$": "T_b (°C)", "$T_\\mathrm{cair}$": "T_cair (°C)", "$p_\\mathrm{in}$": "p_in (bar)",
         "$p_\\mathrm{out}$": "p_out (bar)"}
THERMAL = [0, 1, 2, 4, 8]                       # the five temperatures of the paper's Table 1
GROUPS = ["train", "unseen temperature", "unseen cycle", "unseen cycle + temperature"]


def key_of(T, cycle, original="WLTCHW"):
    return int(abs(T)) if cycle == original else f"{cycle}_{int(abs(T))}"


def cycle_of(key):
    return "WLTCHW" if isinstance(key, int) else key.rsplit("_", 1)[0]


def temp_of(key):
    return -(key if isinstance(key, int) else int(key.rsplit("_", 1)[1]))


# ════════════════════════════════════════════════════════════════════ error measures
def free_errors(res, key, method):
    """{state label: (RMSE, MAE)} of the full-hour free run (report units)."""
    P, H = res["free"][key][method].double().numpy(), res["hfm"][key].double().numpy()
    out = {}
    for lab, i, sc in STATES:
        e = (P[:, i] - H[:, i]) * sc
        out[lab] = (float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e))))
    return out


def thermal_mae(res, key, method):
    """Mean free-run MAE over the five temperatures (one number per run), °C."""
    P, H = res["free"][key][method].double().numpy(), res["hfm"][key].double().numpy()
    return float(np.mean(np.abs(P[:, THERMAL] - H[:, THERMAL])))


# ════════════════════════════════════════════════════════════════════ tables
def table1(res, cycle, temps=(-5, -7, -10), methods=None):
    """Paper Table 1 layout for one drive cycle: rows = state × method, columns = ambient × (RMSE, MAE).

    Free run over the full hour (no resets), as in the paper. Returns (DataFrame, Styler with the best method
    per state and column in bold).
    """
    methods = methods or res["methods"]
    keys = [key_of(T, cycle) for T in temps if key_of(T, cycle) in res["free"]]
    cols = pd.MultiIndex.from_tuples([(f"{temp_of(k):+d} °C".replace("-", "−"), m) for k in keys for m in ("RMSE", "MAE")])
    rows = pd.MultiIndex.from_tuples([(PLAIN[lab], SHORT[m]) for lab, _, _ in STATES for m in methods],
                                     names=["State", "γ from"])
    errs = {(k, m): free_errors(res, k, m) for k in keys for m in methods}
    data = [[errs[(k, m)][lab][j] for k in keys for j in (0, 1)] for lab, _, _ in STATES for m in methods]
    df = pd.DataFrame(data, index=rows, columns=cols)

    def bold_best(col):
        best = col.groupby(level=0).transform("min")
        return ["font-weight: bold" if np.isclose(v, b) else "" for v, b in zip(col, best)]

    sty = (df.style.format("{:.2f}").apply(bold_best, axis=0)
           .set_caption(f"{cycle}: free-run RMSE and MAE over the full hour against the HFM "
                        f"(°C for temperatures, bar for pressures). Bold = best per state.")
           .set_table_styles([{"selector": "caption", "props": "caption-side: top; font-weight: bold; text-align: left"},
                              {"selector": "th.row_heading.level0", "props": "border-top: 1px solid #999"}]))
    return df, sty


def table1_latex(df, path, caption, label):
    """Write a Table-1-style table as LaTeX (booktabs), best value per state in bold."""
    best = df.groupby(level=0).transform("min")
    body = df.copy().astype(object)
    for c in df.columns:
        body[c] = [f"\\textbf{{{v:.2f}}}" if np.isclose(v, b) else f"{v:.2f}" for v, b in zip(df[c], best[c])]
    temps = list(dict.fromkeys(c[0] for c in df.columns))
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
             "\\begin{tabular}{ll" + "rr" * len(temps) + "}", "\\toprule",
             " & & " + " & ".join("\\multicolumn{2}{c}{" + t.replace("°C", "$^\\circ$C").replace("−", "$-$") + "}"
                                  for t in temps) + " \\\\",
             "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(temps))),
             "State & $\\gamma$ from & " + " & ".join(["RMSE & MAE"] * len(temps)) + " \\\\", "\\midrule"]
    prev = None
    for (state, m), row in body.iterrows():
        if prev is not None and state != prev:
            lines.append("\\addlinespace")
        s = state.replace("°C", "$^\\circ$C").replace("_", "\\_") if state != prev else ""
        lines.append(f"{s} & {m} & " + " & ".join(row.values) + " \\\\")
        prev = state
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def summary_table(res):
    """Mean error per group of runs (columns) for each measure and method (rows). Bold = best method."""
    groups = [g for g in GROUPS if any(res["groups"][k] == g for k in res["horizon"])]
    fk = {g: [k for k in res["free"] if res["groups"][k] == g] for g in groups}
    hk = {g: [k for k in res["horizon"] if res["groups"][k] == g] for g in groups}
    measures = [("Free run MAE, temperatures (°C)", lambda k, m: thermal_mae(res, k, m), fk),
                ("Free run MAE, cabin air (°C)", lambda k, m: free_errors(res, k, m)["$T_\\mathrm{cair}$"][1], fk),
                ("Free run MAE, p_out (bar)", lambda k, m: free_errors(res, k, m)["$p_\\mathrm{out}$"][1], fk),
                ("30-s RMSE, temperatures (K)", lambda k, m: res["horizon"][k][m][THERMAL].mean().item(), hk),
                ("30-s RMSE, p_out (bar)", lambda k, m: res["horizon"][k][m][6].item(), hk)]
    data, idx = [], []
    for name, fn, ks in measures:
        for m in res["methods"]:
            idx.append((name, SHORT[m]))
            data.append([np.mean([fn(k, m) for k in ks[g]]) if ks[g] else np.nan for g in groups])
    cols = [f"{g}\n({len(fk[g])} free / {len(hk[g])} 30-s runs)" for g in groups]
    df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(idx, names=["Measure", "γ from"]), columns=cols)

    def bold_best(col):
        best = col.groupby(level=0).transform("min")
        return ["font-weight: bold" if np.isclose(v, b) else "" for v, b in zip(col, best)]
    sty = (df.style.format("{:.2f}", na_rep="–").apply(bold_best, axis=0)
           .set_caption("Mean error per group of runs (lower is better). Bold = best method.")
           .set_table_styles([{"selector": "caption", "props": "caption-side: top; font-weight: bold; text-align: left"},
                              {"selector": "th.col_heading", "props": "white-space: pre-wrap; text-align: center"},
                              {"selector": "th.row_heading.level0", "props": "border-top: 1px solid #999"}]))
    return df, sty


# ════════════════════════════════════════════════════════════════════ figures
def _legend(fig, ax, ncol=4):
    """One legend for the whole figure, below the plots (never on top of the data)."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, fontsize=10)


def _style(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_drive_cycles(profiles, res, minutes=30):
    """Vehicle speed of the four cycles (first `minutes`), marked as training or held-out."""
    cycles = ["WLTCHW", "NEDC", "UDDS", "US06"]
    fig, axs = plt.subplots(len(cycles), 1, figsize=(11, 6.2), sharex=True, sharey=True)
    for ax, c in zip(axs, cycles):
        key = key_of(-20, c)                            # a new export (the old -5/-7/-10 files have no speed)
        if key not in profiles:
            ax.set_visible(False); continue
        v = profiles[key]["est"]["speed"].numpy()
        n = min(len(v), minutes * 60)
        seen = c in res["train_cycles"]
        ax.fill_between(np.arange(n) / 60, v[:n], color="#2a78d6" if seen else "#888888", alpha=0.25, lw=0)
        ax.plot(np.arange(n) / 60, v[:n], color="#2a78d6" if seen else "#444444", lw=1.2)
        ax.text(0.005, 0.92, f"{c} – {'used for training' if seen else 'HELD OUT (never seen in training)'}",
                transform=ax.transAxes, va="top", fontsize=10, fontweight="bold")
        ax.text(0.995, 0.92, f"mean {np.mean(v):.0f} km/h, max {np.max(v):.0f} km/h", transform=ax.transAxes,
                va="top", ha="right", fontsize=9, color="#52514e")
        _style(ax); ax.set_ylabel("km/h")
    axs[-1].set_xlabel("Time (min)  –  each run repeats its cycle to fill one hour")
    fig.suptitle("The four drive cycles (first 30 min)", fontweight="bold")
    plt.tight_layout()
    return fig


def fig_split(res):
    """Which runs were used for training: a cycle × ambient-temperature grid."""
    cycles = ["WLTCHW", "NEDC", "UDDS", "US06"]
    temps = [0, -5, -7, -10, -12, -15, -20, -25, -30]
    col = {"train": "#2a78d6", "unseen temperature": "#a9c9ef", "unseen cycle": "#cfcfcf",
           "unseen cycle + temperature": "#ffffff"}
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for i, c in enumerate(cycles):
        for j, T in enumerate(temps):
            g = res["groups"].get(key_of(T, c))
            if g is None:
                continue
            ax.add_patch(plt.Rectangle((j, i), 0.94, 0.9, color=col[g], ec="#666", lw=0.6))
            if g == "train":
                ax.text(j + 0.47, i + 0.45, "train", ha="center", va="center", color="white", fontsize=8)
    ax.set_xlim(-0.05, len(temps)); ax.set_ylim(len(cycles), -0.1)
    ax.set_xticks(np.arange(len(temps)) + 0.47); ax.set_xticklabels([f"{T} °C".replace("-", "−") for T in temps])
    ax.set_yticks(np.arange(len(cycles)) + 0.45); ax.set_yticklabels(cycles)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(fc=col[g], ec="#666", label=g) for g in GROUPS], ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=9)
    ax.set_title("Training / test split of the 36 runs", fontweight="bold")
    plt.tight_layout()
    return fig


def fig_training(hist, train_min=None):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.semilogy(hist, color="#2a78d6", lw=1.2)
    _style(ax); ax.set_xlabel("iteration"); ax.set_ylabel("training loss (scaled)")
    ax.set_title("Training loss of the new network" + (f" ({train_min:.0f} min)" if train_min else ""),
                 fontweight="bold")
    plt.tight_layout()
    return fig


def fig_group_bars(res):
    """The headline figure: free-run error per group of runs, one bar per method."""
    groups = [g for g in GROUPS if any(res["groups"][k] == g for k in res["free"])]
    methods = res["methods"]
    panels = [("Temperatures: mean MAE (°C)", lambda k, m: thermal_mae(res, k, m)),
              ("Cabin air: MAE (°C)", lambda k, m: free_errors(res, k, m)["$T_\\mathrm{cair}$"][1]),
              ("High-side pressure $p_\\mathrm{out}$: MAE (bar)", lambda k, m: free_errors(res, k, m)["$p_\\mathrm{out}$"][1])]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    w = 0.8 / len(methods)
    for ax, (title, fn) in zip(axs, panels):
        for j, m in enumerate(methods):
            vals = [np.mean([fn(k, m) for k in res["free"] if res["groups"][k] == g]) for g in groups]
            x = np.arange(len(groups)) + (j - (len(methods) - 1) / 2) * w
            ax.bar(x, vals, w * 0.92, color=COLORS[m], label=NICE[m])
            for xi, v in zip(x, vals):
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#52514e")
        n = [sum(res["groups"][k] == g for k in res["free"]) for g in groups]
        ax.set_xticks(range(len(groups)))
        nl = "\n"
        ax.set_xticklabels([g.replace(" + ", " +" + nl).replace("unseen ", "unseen" + nl) + f"{nl}({k} runs)"
                            for g, k in zip(groups, n)], fontsize=9)
        ax.set_title(title, fontsize=11); _style(ax); ax.grid(axis="x", visible=False)
    fig.suptitle("Full-hour free run: error against the HFM (lower is better)", fontweight="bold")
    plt.tight_layout(rect=(0, 0.07, 1, 1)); _legend(fig, axs[0], 3)
    return fig


def fig_temperature_sweep(res, cycle):
    """Free-run MAE vs ambient temperature on one cycle (normally the held-out one)."""
    keys = sorted([k for k in res["free"] if cycle_of(k) == cycle], key=temp_of, reverse=True)
    T = [temp_of(k) for k in keys]
    panels = [("Temperatures: mean MAE (°C)", lambda k, m: thermal_mae(res, k, m)),
              ("Cabin air: MAE (°C)", lambda k, m: free_errors(res, k, m)["$T_\\mathrm{cair}$"][1]),
              ("$p_\\mathrm{out}$: MAE (bar)", lambda k, m: free_errors(res, k, m)["$p_\\mathrm{out}$"][1])]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (title, fn) in zip(axs, panels):
        for m in res["methods"]:
            ax.plot(T, [fn(k, m) for k in keys], "o-", color=COLORS[m], lw=2, ms=6, label=NICE[m])
        for t in T:
            if -t not in res["train_T"]:
                ax.axvspan(t - 1, t + 1, color="#000", alpha=0.06, lw=0)
        ax.set_title(title, fontsize=11); ax.set_xlabel("Ambient temperature (°C)"); _style(ax)
        ax.set_ylim(bottom=0)
    fig.suptitle(f"{cycle} (held-out cycle): free-run error at every ambient temperature  "
                 f"(shaded = temperature also unseen)", fontweight="bold")
    plt.tight_layout(rect=(0, 0.08, 1, 1)); _legend(fig, axs[0], 3)
    return fig


def fig_trajectories(res, profiles, key, methods=None):
    """Paper Fig. 2 style for one run: cabin air, battery, p_in, p_out over the hour + the vehicle speed."""
    methods = methods or res["methods"]
    H = res["hfm"][key].double().numpy(); t = np.arange(len(H))
    v = profiles[key]["est"]["speed"].numpy() if key in profiles else None
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.55, 2, 2])
    axv = fig.add_subplot(gs[0, :])
    if v is not None:
        axv.plot(t, v, color="#52514e", lw=0.6)
    axv.set_ylabel("km/h"); _style(axv); axv.set_xlim(0, len(t)); axv.set_title("Vehicle speed", fontsize=10, loc="left")
    spec = [(8, -C2K, 1, "$T_\\mathrm{cair}$ (°C)", "(a) Cabin air temperature"),
            (4, -C2K, 1, "$T_\\mathrm{b}$ (°C)", "(b) Battery temperature"),
            (5, 0, 1e-5, "Pressure (bar)", "(c) Low-side pressure $p_\\mathrm{in}$"),
            (6, 0, 1e-5, "Pressure (bar)", "(d) High-side pressure $p_\\mathrm{out}$")]
    for n, (i, off, sc, yl, tl) in enumerate(spec):
        ax = fig.add_subplot(gs[1 + n // 2, n % 2], sharex=axv)
        ax.plot(t, (H[:, i] + off) * sc, "k--", lw=1.8, label="HFM (reference)", zorder=5)
        for m in methods:
            ax.plot(t, (res["free"][key][m][:, i].double().numpy() + off) * sc, color=COLORS[m], lw=1.6, label=NICE[m])
        ax.set_ylabel(yl); ax.set_title(tl, fontsize=11, loc="left"); _style(ax)
        if n >= 2:
            ax.set_xlabel("Time (s)")
        if i == 5:
            ax.set_ylim(1, 3.5)
    g = res["groups"][key]
    fig.suptitle(f"Free run at {temp_of(key)} °C".replace("-", "−") + f" on {cycle_of(key)} ({g}): COM vs HFM over the full hour",
                 fontweight="bold")
    plt.tight_layout(rect=(0, 0.04, 1, 1)); _legend(fig, fig.axes[1], 4)
    return fig


def fig_horizon(res):
    """30-s test (how the NMPC uses the COM): mean RMSE per cycle, split into seen/unseen temperatures."""
    cycles = ["WLTCHW", "NEDC", "UDDS", "US06"]
    methods = res["methods"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.8))
    w = 0.8 / len(methods)
    for ax, (title, idx) in zip(axs, [("Temperatures: mean 30-s RMSE (K)", THERMAL), ("$p_\\mathrm{out}$: 30-s RMSE (bar)", [6])]):
        for j, m in enumerate(methods):
            vals = [np.mean([res["horizon"][k][m][idx].mean().item() for k in res["horizon"] if cycle_of(k) == c])
                    for c in cycles]
            x = np.arange(len(cycles)) + (j - (len(methods) - 1) / 2) * w
            ax.bar(x, vals, w * 0.92, color=COLORS[m], label=NICE[m])
        ax.set_xticks(range(len(cycles)))
        ax.set_xticklabels([f"{c}\n({'train' if c in res['train_cycles'] else 'HELD OUT'})" for c in cycles])
        ax.set_title(title, fontsize=11); _style(ax); ax.grid(axis="x", visible=False)
    fig.suptitle("30-s prediction test (as in the NMPC), mean over all 9 temperatures per cycle", fontweight="bold")
    plt.tight_layout(rect=(0, 0.08, 1, 1)); _legend(fig, axs[0], 3)
    return fig
