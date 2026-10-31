# -*- coding: utf-8 -*-
"""
Figures for the analyses added in the JNE version (manuscript numbering):
    Fig7  public-dataset replication: block vs interleaved designs (EEGMMIDB)
    Fig8  ground-truth simulation: scenarios and dose-response
    Fig9  data-calibrated simulation: variance components, reproduction, ablations
Reads the CSV outputs of public_replication.py, simulate_fingerprints.py and calibrated_simulation.py.

    python make_figs_jne_new.py
    python make_figs_jne_new.py --res d:\\path\\to\\analysis_output --tiff none
Output: <res>/figures_jne/Fig7, Fig8, Fig9 (.pdf, .png, .tif)
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
W2 = 180 * MM
COL = {"random": "#0072B2", "time": "#E69F00", "cross": "#009E73", "trial": "#CC79A7", "grey": "#999999"}
TIFF = "plain"


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                         "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def panel(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf")); fig.savefig(os.path.join(out, name + ".png"), dpi=300)
    if TIFF != "none":
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if TIFF == "lzw" else {}
        fig.savefig(os.path.join(out, name + ".tif"), dpi=600, **kw)
    plt.close(fig); print(f"  {name} saved")


def bars(ax, vals, los, his, labels, colors, chance):
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.65, color=colors)
    ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)], fmt="none",
                ecolor="black", elinewidth=0.7, capsize=2)
    ax.axhline(chance, ls="--", color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.02)


def fig7(res, out):
    p = pd.read_csv(os.path.join(res, "public_replication", "results.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 58 * MM), gridspec_kw={"width_ratios": [3, 4, 4]})
    for ax, design, schemes, title in (
            (axs[0], "A", ["random split", "time split", "cross-subject"], "Block design\n(eyes open vs eyes closed)"),
            (axs[1], "B", ["random split", "time split", "cross-subject", "trial-grouped (within participant)"],
             "Interleaved design\n(imagined left vs right fist)")):
        s = p[(p.design == design) & (~p.task.str.contains("pseudo|identification"))].set_index("scheme").loc[schemes]
        lab = {"random split": "Random\nsplit", "time split": "Time\nsplit", "cross-subject": "Cross-\nsubject",
               "trial-grouped (within participant)": "Trial-\ngrouped"}
        c = {"random split": COL["random"], "time split": COL["time"], "cross-subject": COL["cross"],
             "trial-grouped (within participant)": COL["trial"]}
        bars(ax, s.subj_mean, s.ci95_lo, s.ci95_hi, [lab[k] for k in schemes], [c[k] for k in schemes], 0.5)
        ax.set_title(title); ax.set_ylabel("Per-participant accuracy" if design == "A" else "")
    ax = axs[2]
    t = p[p.task.str.contains("pseudo")]
    groups = [("A", "eye_open", "Block:\nEO"), ("A", "eye_close", "Block:\nEC"),
              ("B", "left", "Interl.:\nleft"), ("B", "right", "Interl.:\nright")]
    x = np.arange(len(groups)); w = 0.38
    for j, (sch, col) in enumerate((("random split", COL["random"]), ("cross-subject", COL["cross"]))):
        s = [t[(t.design == d) & (t.task.str.contains(f"\\({c}\\)")) & (t.scheme == sch)].iloc[0] for d, c, _ in groups]
        v = np.array([r.subj_mean for r in s]); lo = np.array([r.ci95_lo for r in s]); hi = np.array([r.ci95_hi for r in s])
        ax.bar(x + (j - 0.5) * w, v, w, color=col, label="Random split" if j == 0 else "Cross-subject")
        ax.errorbar(x + (j - 0.5) * w, v, yerr=[v - lo, hi - v], fmt="none", ecolor="black", elinewidth=0.7, capsize=1.5)
    ax.axhline(1 / 3, ls="--", color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([g[2] for g in groups]); ax.set_ylim(0, 1.02)
    ax.set_title("Time-block pseudo-labels\n(no task information)"); ax.legend(frameon=False, loc="upper right")
    for k, ax in enumerate(axs):
        panel(ax, "ABC"[k])
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "Fig7")


def fig8(res, out):
    s = pd.read_csv(os.path.join(res, "simulation", "results.csv"))
    sc = s[~s.scenario.str.startswith("dose")].reset_index(drop=True)
    dose = s[s.scenario.str.startswith("dose")].reset_index(drop=True)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 62 * MM), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axs[0]
    names = ["No nuisance,\nno task effect", "Recording\noffsets", "Slow\ndrift", "Offsets +\ndrift",
             "True effect,\nno nuisance", "True effect\n+ nuisance"]
    x = np.arange(len(sc)); w = 0.27
    for j, (k, lab, col) in enumerate((("a", "Random split", COL["random"]), ("b", "Time split", COL["time"]),
                                       ("c", "Cross-subject", COL["cross"]))):
        v, lo, hi = sc[f"{k}_mean"], sc[f"{k}_lo"], sc[f"{k}_hi"]
        ax.bar(x + (j - 1) * w, v, w, color=col, label=lab)
        ax.errorbar(x + (j - 1) * w, v, yerr=[v - lo, hi - v], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.2)
    ax.axhline(1 / 3, ls="--", color="black", lw=0.8)
    ax.axvspan(-0.5, 3.5, color="#f2f2f2", zorder=-1)
    ax.text(1.5, 1.0, "no task information", ha="center", fontsize=6.5, style="italic")
    ax.text(4.5, 1.0, "true task effect", ha="center", fontsize=6.5, style="italic")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.06); ax.set_xlim(-0.5, len(sc) - 0.5)
    ax.set_ylabel("Per-participant accuracy")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_title("Synthetic data with known ground truth (three classes)")
    ax = axs[1]
    for k, lab, col in (("a", "Random split", COL["random"]), ("c", "Cross-subject", COL["cross"])):
        ax.plot(dose.sd_recording, dose[f"{k}_mean"], "o-", color=col, ms=3, lw=1, label=lab)
        ax.fill_between(dose.sd_recording, dose[f"{k}_lo"], dose[f"{k}_hi"], color=col, alpha=0.25, lw=0)
    ax.axhline(1 / 3, ls="--", color="black", lw=0.8)
    ax.set_xlabel("SD of recording-specific offset"); ax.set_ylim(0, 1.02)
    ax.set_title("Dose-response, no task information"); ax.legend(frameon=False, loc="center right")
    for k, ax in enumerate(axs):
        panel(ax, "AB"[k])
    fig.tight_layout(w_pad=2)
    save(fig, out, "Fig8")


def fig9(res, out):
    r = pd.read_csv(os.path.join(res, "calibrated_simulation", "results.csv"))
    v = pd.read_csv(os.path.join(res, "calibrated_simulation", "variance_components.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 72 * MM), gridspec_kw={"width_ratios": [1.15, 2, 2]})
    ax = axs[0]
    real = v[v.source == "real"].set_index("dataset").loc[["cohort", "eegmmidb"]]
    comps = [("participant", "Participant", "#56B4E9"), ("recording", "Recording offset", "#D55E00"),
             ("slow_drift", "Slow drift", "#E69F00"), ("fast_noise", "Fast noise", "#BBBBBB")]
    bottom = np.zeros(2)
    for k, lab, col in comps:
        ax.bar([0, 1], real[k], 0.6, bottom=bottom, color=col, label=lab); bottom += real[k].values
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Cohort\n(dry)", "EEGMMIDB\n(wet)*"]); ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share of feature variance"); ax.set_title("Variance\ncomponents")
    ax.legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax = axs[1]
    sch = [("(a) random split", "Random"), ("(b) within-recording time split", "Time"), ("(c) cross-subject", "Cross")]
    xs, ticks = [], []
    for d, ds in enumerate(["cohort", "eegmmidb"]):
        rr = r[r.dataset == ds]
        realr = rr[rr.model == "real data"].iloc[0]
        cal = rr[rr.model.str.startswith("calibrated model")]
        for i, (k, lab) in enumerate(sch):
            x0 = d * 4 + i
            ax.bar(x0 - 0.18, realr[k], 0.36, color=COL["grey"], label="Real data" if (d == 0 and i == 0) else None)
            m = cal[k].mean()
            ax.bar(x0 + 0.18, m, 0.36, color=COL["random"], label="Calibrated model" if (d == 0 and i == 0) else None)
            ax.errorbar(x0 + 0.18, m, yerr=[[m - cal[k].min()], [cal[k].max() - m]], fmt="none", ecolor="black",
                        elinewidth=0.7, capsize=1.5)
            xs.append(x0); ticks.append(lab[0])
    ax.set_xticks(xs); ax.set_xticklabels(ticks); ax.set_ylim(0, 1.18); ax.set_ylabel("Per-participant accuracy")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.text(1, -0.20, "Cohort", ha="center", transform=ax.get_xaxis_transform(), fontsize=6.5)
    ax.text(5, -0.20, "EEGMMIDB", ha="center", transform=ax.get_xaxis_transform(), fontsize=6.5)
    ax.set_title("Reproduction of\nobserved accuracy"); ax.legend(frameon=False, loc="upper center", ncol=2)
    ax = axs[2]
    models = [("real data", "Real\ndata"), ("calibrated model", "Calib-\nrated"), ("- recording offsets", "No\noffsets"),
              ("- slow drift", "No\ndrift"), ("- recording-specific (offsets + drift)", "No rec.-\nspecific"),
              ("- participant offsets", "No\npartic.")]
    x = np.arange(len(models)); w = 0.38
    for j, (ds, col) in enumerate((("cohort", COL["random"]), ("eegmmidb", COL["cross"]))):
        rr = r[r.dataset == ds]
        vals = [rr[rr.model.str.startswith(m)]["random - cross"].mean() for m, _ in models]
        ax.bar(x + (j - 0.5) * w, vals, w, color=col, label={"cohort": "Cohort", "eegmmidb": "EEGMMIDB"}[ds])
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in models], fontsize=5.6)
    ax.set_ylabel("Random minus cross-subject"); ax.set_title("Ablation of the\ncalibrated model")
    ax.legend(frameon=False, loc="upper right")
    for k, ax in enumerate(axs):
        panel(ax, "ABC"[k])
    fig.tight_layout(w_pad=1.8)
    save(fig, out, "Fig9")


def default_res():
    """Same results folder as the other programs (eeg_analysis.OUT_DIR_BASE), else EEG_OUTPUT_DIR."""
    try:
        import eeg_analysis as EA
        return EA.OUT_DIR_BASE
    except Exception:
        return os.environ.get("EEG_OUTPUT_DIR", "analysis_output")


NEEDED = {"fig7": ("public_replication", "results.csv"), "fig8": ("simulation", "results.csv"),
          "fig9": ("calibrated_simulation", "results.csv")}


def main():
    global TIFF
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=default_res(), help="analysis output folder (default: the one set in eeg_analysis.py)")
    ap.add_argument("--tiff", choices=["plain", "lzw", "none"], default="plain")
    a = ap.parse_args()
    TIFF = a.tiff
    style()
    out = os.path.join(a.res, "figures_jne")
    print(f"Results folder: {a.res}\nFigures -> {out}")
    for f in (fig7, fig8, fig9):
        need = os.path.join(a.res, *NEEDED[f.__name__])
        if not os.path.exists(need):
            print(f"  [SKIP] {f.__name__}: {need} not found (run the corresponding analysis first)"); continue
        try:
            f(a.res, out)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  [ERROR] {f.__name__}: {e}")


if __name__ == "__main__":
    main()
