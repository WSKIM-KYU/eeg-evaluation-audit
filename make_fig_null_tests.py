# -*- coding: utf-8 -*-
"""
Figure for the null-hypothesis experiments (JNE manuscript, Fig. 10):
  A  random split: observed accuracy, window-level null mean and dependence-preserving null (mean, 95th percentile)
  B  cross-subject: observed accuracy and dependence-preserving null mean
  C  simulation: rejection rate (Clopper-Pearson 95% CI) of each scheme x test in scenarios a-f
Reads the results.csv files of null_tests.py (copied to null_tests_eo_hu_999 and null_tests_3tasks_199) and
null_tests_simulation.py. Only summary statistics are plotted (no reconstructed distributions).

    python make_fig_null_tests.py
Output: <res>/figures_jne/Fig10_null_tests (.pdf, .png, .tif)
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
TASKS = [("cohort_eo_hu", "Cohort\nEO vs arm\n(neg. control)"), ("cohort_ec_eo", "Cohort\nEC vs EO"),
         ("mmidb_block", "EEGMMIDB\nblock"), ("mmidb_interleaved", "EEGMMIDB\ninterleaved")]
TESTS = [("random + binomial", "Random split + binomial", "#999999", "s"),
         ("random + window permutation", "Random split + window permutation", "#E69F00", "D"),
         ("random + dependence-preserving", "Random split + dependence-preserving", "#0072B2", "o"),
         ("cross-subject + dependence-preserving", "Cross-subject + dependence-preserving", "#009E73", "^")]


def default_res():
    try:
        import eeg_analysis as EA
        return EA.OUT_DIR_BASE
    except Exception:
        return os.environ.get("EEG_OUTPUT_DIR", "analysis_output")


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                         "axes.titlesize": 7.5, "xtick.labelsize": 6.3, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def panel(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def pfmt(p):
    return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=default_res())
    ap.add_argument("--tiff", choices=["plain", "lzw", "none"], default="plain")
    a = ap.parse_args()
    style()
    R = pd.concat([pd.read_csv(os.path.join(a.res, d, "results.csv"))
                   for d in ("null_tests_eo_hu_999", "null_tests_3tasks_199")]).set_index("task")
    # The 999-permutation negative-control run did not include the cross-subject test; take it from the earlier
    # 100-permutation run (save that results.csv as <res>/null_tests_eo_hu_100/results.csv) if available.
    if pd.isna(R.loc["cohort_eo_hu"].get("observed_cross_subject", np.nan)):
        extra = os.path.join(a.res, "null_tests_eo_hu_100", "results.csv")
        cols = ("observed_cross_subject", "cross_unit_null_mean", "cross_unit_perm_p")
        e = None
        if os.path.exists(extra):
            x = pd.read_csv(extra)
            if "task" in x and (x["task"] == "cohort_eo_hu").any() and all(c in x for c in cols):
                e = x.set_index("task").loc["cohort_eo_hu"]
            else:
                print(f"[NOTE] {extra} does not contain the negative-control cross-subject result "
                      "(task cohort_eo_hu with columns " + ", ".join(cols) + "); panel B omits it")
        if e is not None:
            for c in cols:
                R.loc["cohort_eo_hu", c] = e[c]
            print("negative-control cross-subject test taken from", extra)
        elif not os.path.exists(extra):
            print("[NOTE] negative-control cross-subject result not found; panel B omits it "
                  f"(expected {extra})")
    S = pd.read_csv(os.path.join(a.res, "null_tests_simulation", "results.csv"))
    fig = plt.figure(figsize=(180 * MM, 125 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.65, wspace=0.25)
    axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    x = np.arange(len(TASKS))
    for i, (t, _) in enumerate(TASKS):
        r = R.loc[t]
        axA.plot([i - 0.15], [r.window_null_mean], "D", color="#E69F00", ms=4, label="Window-level null (mean)" if i == 0 else None)
        axA.errorbar([i + 0.15], [r.unit_null_mean], yerr=[[0], [r.unit_null_95pct - r.unit_null_mean]], fmt="o",
                     color="#0072B2", ms=4, capsize=2, label="Dependence-preserving null\n(mean, 95th percentile)" if i == 0 else None)
        axA.plot([i], [r.observed_random_split], "_", color="black", ms=14, mew=2, label="Observed" if i == 0 else None)
        axA.text(i, 1.035, pfmt(r.unit_perm_p), ha="center", fontsize=6, color="#0072B2")
        if pd.isna(r.get("observed_cross_subject", np.nan)):
            continue
        axB.plot([i + 0.12], [r.cross_unit_null_mean], "o", color="#009E73", ms=4,
                 label="Dependence-preserving null (mean)" if i == 0 else None)
        axB.plot([i], [r.observed_cross_subject], "_", color="black", ms=14, mew=2, label="Observed" if i == 0 else None)
        axB.text(i, 1.035, pfmt(r.cross_unit_perm_p), ha="center", fontsize=6, color="#009E73")
    for ax, title in ((axA, "Random split"), (axB, "Cross-subject evaluation")):
        ax.set_xticks(x); ax.set_xticklabels([t[1] for t in TASKS]); ax.set_ylim(0.4, 1.08); ax.axhline(0.5, ls="--", color="grey", lw=0.7)
        ax.set_xlim(-0.6, len(TASKS) - 0.4)                    # keep the p-value labels clear of the y axis
        ax.set_ylabel("Per-participant accuracy"); ax.set_title(title, pad=14)
    axA.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.45), fontsize=5.8)
    axB.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.37), fontsize=5.8)   # empty band between 0.55 and 0.75
    scen = list(dict.fromkeys(S.scenario))
    short = {"a": "a: no effect,\nno nuisance", "b": "b: no effect,\nnuisance", "c": "c: effect,\nno nuisance",
             "d": "d: effect,\nnuisance", "e": "e: no effect, nuis.,\norder (fixed)",
             "f": "f: no effect, nuis.,\norder (counterbal.)"}
    lab = {s: short[s[0]] for s in scen}
    w = 0.19
    for j, (t, tl, col, mk) in enumerate(TESTS):
        sub = S[S.test == t].set_index("scenario").loc[scen]
        xs = np.arange(len(scen)) + (j - 1.5) * w
        axC.errorbar(xs, sub.rejection_rate, yerr=[sub.rejection_rate - sub.ci95_lo, sub.ci95_hi - sub.rejection_rate],
                     fmt=mk, color=col, ms=4, capsize=1.5, elinewidth=0.8, label=tl)
    axC.axhline(0.04, ls="--", color="black", lw=0.8)
    axC.text(-0.45, 0.06, "0.04", fontsize=6, ha="left")
    for k, s in enumerate(scen):
        if s[0] in "cd":
            axC.axvspan(k - 0.5, k + 0.5, color="#f2f2f2", zorder=-1)
    axC.set_xticks(np.arange(len(scen))); axC.set_xticklabels([lab[s] for s in scen], fontsize=6)
    axC.set_xlim(-0.5, len(scen) - 0.5); axC.set_ylim(-0.03, 1.05)
    axC.set_ylabel("Rejection rate (α = 0.05)")
    axC.set_title("Simulation: type I error (scenarios a, b, e, f) and power (shaded c, d); 40 datasets per scenario")
    axC.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    for ax, s in ((axA, "A"), (axB, "B"), (axC, "C")):
        panel(ax, s)
    out = os.path.join(a.res, "figures_jne"); os.makedirs(out, exist_ok=True)
    base = os.path.join(out, "Fig10_null_tests")
    fig.savefig(base + ".pdf"); fig.savefig(base + ".png", dpi=300)
    if a.tiff != "none":
        fig.savefig(base + ".tif", dpi=600, **({"pil_kwargs": {"compression": "tiff_lzw"}} if a.tiff == "lzw" else {}))
    print(f"Saved -> {base}.pdf/.png")


if __name__ == "__main__":
    main()
