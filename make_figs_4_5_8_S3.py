# -*- coding: utf-8 -*-
"""
Regenerate only the figures whose labels changed ("Three-class", "Two-class"), with manuscript numbering:

    Fig4  classification accuracy by task and evaluation scheme
    Fig5  per-participant accuracy under the three schemes
    Fig8  external validation on EEGMMIDB
    FigS3 EEGNet (raw signal) vs random forest (band power), with panel letters A/B

No model is retrained; everything is read from saved results. Takes well under a minute.
Needs make_figures.py (latest version) in the same folder.

    python make_figs_4_5_8_S3.py
    python make_figs_4_5_8_S3.py --res analysis_output\\policy_common --tiff none
Output: <res>\\figures\\Fig4 / Fig5 / Fig8 / FigS3 (.pdf, .png, .tif)
"""
import os, argparse
import numpy as np
import pandas as pd
import make_figures as MF

SCHEMES = ["(a) random split", "(b) within-recording time split", "(c) cross-subject"]
TICKS = ["Random\nsplit", "Time\nsplit", "Cross-\nsubject"]
TASKS = ["3-class", "eye_open vs hands_up"]
TITLES = {"3-class": "Three-class", "eye_open vs hands_up": "Eyes open vs arm elevation"}


def fig_s3(res, out):
    import matplotlib.pyplot as plt
    e = pd.read_csv(os.path.join(res, "eegnet", "eegnet_results.csv"))
    t = pd.read_excel(os.path.join(res, "results.xlsx"), sheet_name="Tasks_by_evaluation")
    rf = t[t.features == "QC-cleaned band power (RF)"]
    fig, axs = plt.subplots(1, 2, figsize=(MF.W2, 60 * MF.MM), sharey=True)
    w = 0.38
    for k, (ax, task) in enumerate(zip(axs, TASKS)):
        for j, (src, name, col) in enumerate(((rf, "Random forest (band power)", "#999999"),
                                               (e, "EEGNet (raw signal)", "#0072B2"))):
            s = src[src.task == task].set_index("evaluation").reindex(SCHEMES)
            if s.subj_mean.isna().all():
                raise SystemExit(f"[ERROR] no {name} results for {task}")
            x = np.arange(3) + (j - 0.5) * w
            ax.bar(x, s.subj_mean, w, color=col, label=name)
            ax.errorbar(x, s.subj_mean, yerr=[s.subj_mean - s.subj_ci95_lo, s.subj_ci95_hi - s.subj_mean],
                        fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
        ax.axhline(1 / 3 if task == "3-class" else 0.5, ls="--", color="black", lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(TICKS)
        ax.set_ylim(0, 1.02); ax.set_title(TITLES[task])
        if k == 0:
            ax.set_ylabel("Per-participant accuracy\n(mean, 95% CI)")
        MF.panel(ax, "AB"[k])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=2)
    MF.save(fig, out, "FigS3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=MF.DEFAULT_RES, help="analysis output folder (policy_common)")
    ap.add_argument("--tiff", choices=["plain", "lzw", "none"], default="plain")
    a = ap.parse_args()
    MF.TIFF_MODE = a.tiff
    MF.style()
    out = os.path.join(a.res, "figures")
    print(f"Results folder: {a.res}\nFigures -> {out}")
    xl = pd.read_excel(os.path.join(a.res, "results.xlsx"), sheet_name=None)
    steps = [("Fig4 (tasks by evaluation)", lambda: MF.fig3(xl, out)),      # saved as Fig4 (NAME_MAP)
             ("Fig5 (per participant)", lambda: MF.fig4(xl, out)),          # saved as Fig5
             ("Fig8 (external validation)", lambda: MF.fig7(a.res, out)),   # saved as Fig8
             ("FigS3 (EEGNet)", lambda: fig_s3(a.res, out))]
    failed = []
    for name, fn in steps:
        print(f"[{name}] ...", flush=True)
        try:
            fn()
        except Exception as ex:
            import traceback; traceback.print_exc()
            failed.append(name); print(f"  [ERROR] {name}: {ex}")
    print("Done." if not failed else f"Done with errors in: {', '.join(failed)}")


if __name__ == "__main__":
    main()
