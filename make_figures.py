# -*- coding: utf-8 -*-
"""
Submission figures for the BSPC manuscript (Fig. 1-7, Fig. S1) and Table S2.

Run after eeg_analysis.py (with the extra analyses) and external_validation.py, from the folder
that contains both scripts:

    python make_figures.py
    python make_figures.py --res analysis_output\\policy_common --trees 300

Inputs (in --res): results.xlsx, cache\\ + cache_manifest.json, external\\external_results.csv, cache_mmidb\\
Outputs (in --res\\figures): FigN.pdf (vector), FigN.tif (600 dpi, LZW), FigN.png (300 dpi preview),
                             TableS2_EEGMMIDB_participants.csv
Fig. 5 is recomputed from the cached features (cross-subject 3-class and EEGMMIDB -> cohort 3-class).
"""
import os, glob, json, zlib, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle

import eeg_analysis as EA
import external_validation as EV

warnings.filterwarnings("ignore")

DEFAULT_RES = os.path.join(EA.OUT_DIR_BASE, "policy_common")
MM = 1 / 25.4                                   # inches per mm
W1, W15, W2 = 90 * MM, 140 * MM, 190 * MM       # Elsevier column widths

COND_LABEL = {"eye_close": "Eyes closed", "eye_open": "Eyes open", "hands_up": "Arm elevation"}
COND_ORDER = ["eye_close", "eye_open", "hands_up"]
COND_COLOR = {"eye_close": "#0072B2", "eye_open": "#E69F00", "hands_up": "#CC79A7"}   # Okabe-Ito
SCHEME_KEYS = ["(a) random split", "(b) within-recording time split", "(c) cross-subject"]
SCHEME_LABEL = {"(a) random split": "Random split", "(b) within-recording time split": "Time split",
                "(c) cross-subject": "Cross-subject"}
SCHEME_COLOR = {"(a) random split": "#0072B2", "(b) within-recording time split": "#E69F00",
                "(c) cross-subject": "#009E73"}
TASK_ORDER = ["3-class", "eye_close vs eye_open", "eye_close vs hands_up", "eye_open vs hands_up"]
TASK_LABEL = {"3-class": "Three-class", "eye_close vs eye_open": "Eyes closed\nvs eyes open",
              "eye_close vs hands_up": "Eyes closed\nvs arm elevation",
              "eye_open vs hands_up": "Eyes open\nvs arm elevation"}
FEAT = "QC-cleaned band power (RF)"


def style():
    fams = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    font = "Arial" if "Arial" in fams else ("Helvetica" if "Helvetica" in fams else "DejaVu Sans")
    plt.rcParams.update({
        "font.family": font, "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02})


def panel(ax, letter, x=-0.14, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


TIFF_MODE = "plain"      # "plain" = uncompressed TIFF (safe), "lzw" = compressed (needs a working libtiff), "none"


# Fig. 1 of the manuscript is the electrode layout (supplied as an image), so every generated figure
# is saved one number higher: design schematic -> Fig2, ..., external validation -> Fig8.
NAME_MAP = {"Fig1": "Fig2", "Fig2": "Fig3", "Fig3": "Fig4", "Fig4": "Fig5", "Fig5": "Fig6", "Fig6": "Fig7",
            "Fig7": "Fig8"}


def save(fig, out, name):
    name = NAME_MAP.get(name, name)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=300)
    done = "pdf / png"
    if TIFF_MODE != "none":
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if TIFF_MODE == "lzw" else {}
        fig.savefig(os.path.join(out, name + ".tif"), dpi=600, **kw)
        done += f" / tif (600 dpi{', LZW' if TIFF_MODE == 'lzw' else ''})"
    plt.close(fig)
    print(f"  {name}: {done}", flush=True)


def task_label(t):
    for k, v in COND_LABEL.items():
        t = t.replace(k, v)
    return t


# ------------------------------------------------------------------ data
def load_recs(res):
    man = json.load(open(os.path.join(res, "cache_manifest.json")))
    files = [os.path.join(res, "cache", n) for n in man["files"]]
    recs = EA.load_all(files)
    recs = [EA.reorder_record(r, man["chans"]) for r in recs]
    return recs, man["chans"]


# ------------------------------------------------------------------ Fig. 1 (design schematic)
def fig1(out):
    rng = np.random.RandomState(3)
    fig, axs = plt.subplots(3, 1, figsize=(W2, 105 * MM))
    nwin, subjects = 20, ["Participant 1", "Participant 2"]
    titles = ["(a) Random split", "(b) Within-recording time split", "(c) Cross-subject evaluation"]
    TR, TE, GAP = "#0072B2", "#E69F00", "#d9d9d9"
    for ax, title, mode in zip(axs, titles, "abc"):
        y = 0
        for si, s in enumerate(subjects):
            for c in COND_ORDER:
                for w in range(nwin):
                    if mode == "a":
                        col = TE if rng.rand() < 0.2 else TR
                    elif mode == "b":
                        col = TR if w < 12 else (GAP if w < 14 else TE)
                    else:
                        col = TR if si == 0 else TE
                    ax.add_patch(Rectangle((w, y), 0.86, 0.72, color=col, lw=0))
                ax.text(-0.6, y + 0.36, f"{s} · {COND_LABEL[c]}", ha="right", va="center", fontsize=6.5)
                y += 1
            y += 0.5
        ax.set_xlim(-0.2, nwin + 0.2); ax.set_ylim(y - 0.3, -0.5)
        ax.axis("off")
        ax.set_title(title, loc="left", fontweight="bold", x=-0.3)
        if mode == "b":
            ax.annotate("first 60%", xy=(6, -0.25), ha="center", fontsize=6.5)
            ax.annotate("2-s gap", xy=(13, -0.25), ha="center", fontsize=6.5)
    h = [Rectangle((0, 0), 1, 1, color=TR), Rectangle((0, 0), 1, 1, color=TE), Rectangle((0, 0), 1, 1, color=GAP)]
    fig.legend(h, ["Training window", "Test window", "Unused (gap)"], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.6, -0.02))
    fig.text(0.6, 0.035, "2-s windows (1-s step) along each recording  →", ha="center", fontsize=6.5)
    fig.subplots_adjust(left=0.27, right=0.99, hspace=0.45, bottom=0.1)
    save(fig, out, "Fig1")


# ------------------------------------------------------------------ Fig. 2 (data quality)
def fig2(recs, chans, out):
    f = recs[0]["f"]; m = (f >= 1) & (f <= 45)
    ci = {c.upper(): i for i, c in enumerate(chans)}
    occ_i = [ci[c.upper()] for c in ["O1", "O2"] if c.upper() in ci]
    alpha_i = [ci[c.upper()] for c in EA.OCCIPITAL if c.upper() in ci]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 58 * MM), gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
    for c in COND_ORDER:
        ps = np.array([r["psd"][:, occ_i].mean(1) for r in recs if str(r["cond"]) == c])
        if len(ps):
            axs[0].semilogy(f[m], np.exp(np.log(ps[:, m] + 1e-12).mean(0)), color=COND_COLOR[c], lw=1.1,
                            label=f"{COND_LABEL[c]} (n = {len(ps)})")
    axs[0].axvspan(8, 13, color="grey", alpha=0.12, lw=0)
    axs[0].set(xlabel="Frequency (Hz)", ylabel="Power (µV²/Hz)", xlim=(1, 45))
    axs[0].legend(frameon=False); panel(axs[0], "A")
    rel = {}
    for r in recs:
        p = r["psd"]; tot = EA.band_power(f, p, 1, 45); al = EA.band_power(f, p, 8, 13)
        rel[(str(r["subject"]), str(r["cond"]))] = float((al / tot)[alpha_i].mean())
    subs = sorted({s for s, c in rel if (s, "eye_open") in rel and (s, "eye_close") in rel})
    eo = np.array([rel[(s, "eye_open")] for s in subs]); ec = np.array([rel[(s, "eye_close")] for s in subs])
    for a, b in zip(eo, ec):
        axs[1].plot([0, 1], [a, b], color="grey", lw=0.4, alpha=0.6)
    axs[1].scatter(np.zeros(len(eo)), eo, s=6, color=COND_COLOR["eye_open"], zorder=3)
    axs[1].scatter(np.ones(len(ec)), ec, s=6, color=COND_COLOR["eye_close"], zorder=3)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["Eyes open", "Eyes closed"]); axs[1].set_xlim(-0.4, 1.4)
    axs[1].set_ylabel("Relative occipital alpha"); panel(axs[1], "B")
    axs[1].text(0.5, 0.97, f"increase in {(ec > eo).sum()}/{len(subs)}", transform=axs[1].transAxes,
                ha="center", va="top", fontsize=6.5)
    data = [[float(r["acc_rms"]) for r in recs if str(r["cond"]) == c] for c in COND_ORDER]
    bp = axs[2].boxplot(data, widths=0.55, showfliers=False, patch_artist=True, medianprops=dict(color="k", lw=0.8))
    for patch, c in zip(bp["boxes"], COND_ORDER):
        patch.set_facecolor(COND_COLOR[c]); patch.set_alpha(0.45); patch.set_linewidth(0.6)
    rng = np.random.RandomState(0)
    for i, d in enumerate(data, 1):
        axs[2].scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=4, color="k", alpha=0.5, zorder=3)
    axs[2].set_xticks([1, 2, 3]); axs[2].set_xticklabels(["Eyes\nclosed", "Eyes\nopen", "Arm\nelevation"])
    axs[2].set_ylabel("Head-motion index (mg)"); panel(axs[2], "C")
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "Fig2")


# ------------------------------------------------------------------ Fig. 3 (tasks x schemes)
def fig3(xl, out):
    t = xl["Tasks_by_evaluation"]; t = t[t.features == FEAT]
    fig, ax = plt.subplots(figsize=(W15, 62 * MM))
    w = 0.26
    for j, sk in enumerate(SCHEME_KEYS):
        rows = [t[(t.task == k) & (t.evaluation == sk)] for k in TASK_ORDER]
        mean = [r.subj_mean.iloc[0] if len(r) else np.nan for r in rows]
        lo = [r.subj_mean.iloc[0] - r.subj_ci95_lo.iloc[0] if len(r) else 0 for r in rows]
        hi = [r.subj_ci95_hi.iloc[0] - r.subj_mean.iloc[0] if len(r) else 0 for r in rows]
        x = np.arange(len(TASK_ORDER)) + (j - 1) * w
        ax.bar(x, mean, w * 0.95, color=SCHEME_COLOR[sk], label=SCHEME_LABEL[sk])
        ax.errorbar(x, mean, yerr=[lo, hi], fmt="none", ecolor="k", elinewidth=0.6, capsize=1.5)
    for i, k in enumerate(TASK_ORDER):
        ax.hlines(1 / 3 if k == "3-class" else 0.5, i - 0.42, i + 0.42, colors="k", linestyles="--", lw=0.7)
    ax.set_xticks(range(len(TASK_ORDER))); ax.set_xticklabels([TASK_LABEL[k] for k in TASK_ORDER])
    ax.set(ylim=(0, 1.02), ylabel="Per-participant accuracy\n(mean, 95% CI)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    save(fig, out, "Fig3")


# ------------------------------------------------------------------ Fig. 4 (per-participant)
def fig4(xl, out):
    ps = xl["PerSubject_accuracy"]; ps = ps[ps.features == FEAT]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 62 * MM))
    for ax, task, chance, letter in ((axs[0], "3-class", 1 / 3, "A"), (axs[1], "eye_open vs hands_up", 0.5, "B")):
        d = ps[ps.task == task].pivot_table(index="subject", columns="evaluation", values="accuracy")
        keys = [k for k in SCHEME_KEYS if k in d]
        for _, row in d[keys].iterrows():
            ax.plot(range(len(keys)), row.values, color="grey", lw=0.35, alpha=0.5)
        rng = np.random.RandomState(1)
        for i, k in enumerate(keys):
            v = d[k].dropna().values
            ax.boxplot([v], positions=[i], widths=0.45, showfliers=False, medianprops=dict(color="k", lw=0.9),
                       boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
            ax.scatter(i + rng.uniform(-0.1, 0.1, len(v)), v, s=6, color=SCHEME_COLOR[k], zorder=3, alpha=0.85)
        ax.axhline(chance, ls="--", c="k", lw=0.7)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([SCHEME_LABEL[k] for k in keys])
        ax.set(ylim=(0, 1.02), ylabel="Per-participant accuracy", title=TASK_LABEL[task].replace("\n", " ")
               if task != "3-class" else "Three-class (eyes closed, eyes open, arm elevation)")
        ax.text(0.98, 0.03, f"n = {len(d)}", transform=ax.transAxes, ha="right", fontsize=6.5)
        panel(ax, letter)
    fig.tight_layout(w_pad=2)
    save(fig, out, "Fig4")


# ------------------------------------------------------------------ Fig. 5 (confusion matrices, recomputed)
def confusion(yt, yp, labels):
    cm = np.array([[np.sum((yt == a) & (yp == b)) for b in labels] for a in labels], float)
    return cm / np.maximum(cm.sum(1, keepdims=True), 1)


def draw_cm(ax, cm, row_labels, col_labels, title, letter):
    ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set(xlabel="Predicted", ylabel="True"); ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(True)
    panel(ax, letter, x=-0.42)


def fig5(recs, chans, res, out, trees):
    from sklearn.ensemble import RandomForestClassifier
    rf = lambda: RandomForestClassifier(trees, min_samples_leaf=2, n_jobs=-1, random_state=EA.RANDOM_STATE)
    print("  Fig5: recomputing cross-subject 3-class predictions ...")
    EA.channel_qc(recs)
    D = EA.build_clean(recs)
    yt, yp, g = EA.run_schemes(D["X"], D["y"], D["g"], D["ts"], D["dur"], rf, schemes=("c",))["c"]
    cmA = confusion(yt, yp, COND_ORDER)
    print(f"        cross-subject accuracy {np.mean(yt == yp):.3f} (should match Table 1 / results.xlsx)")
    print("  Fig5: recomputing EEGMMIDB -> cohort 3-class predictions ...")
    coh, ch2 = EV.load_cohort(os.path.join(res, "cache"))
    tag = f"{len(ch2)}ch_{zlib.crc32(','.join(ch2).encode()):08x}"
    files = sorted(glob.glob(os.path.join(res, "cache_mmidb", f"mmidb_S*_{tag}_hu*.npz")))
    if not files:
        raise SystemExit("No EEGMMIDB cache for this channel set; run external_validation.py with this --out first.")
    mm = EV.load_mmidb(files)
    mode = "relative + subject z-score"
    Xc, Xm = EV.transform(coh, mode, len(ch2)), EV.transform(mm, mode, len(ch2))
    model = RandomForestClassifier(trees, min_samples_leaf=2, class_weight="balanced", n_jobs=-1,
                                   random_state=EA.RANDOM_STATE).fit(Xm, mm["y"])
    pB = model.predict(Xc)
    cmB = confusion(coh["y"], pB, COND_ORDER)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 70 * MM))
    lab = ["Eyes\nclosed", "Eyes\nopen", "Arm\nelevation"]
    draw_cm(axs[0], cmA, lab, lab, "Cohort, cross-subject", "A")
    draw_cm(axs[1], cmB, lab, ["Eyes\nclosed", "Eyes\nopen", "Fist\nmovement"],
            "Trained on EEGMMIDB, tested on cohort", "B")
    fig.tight_layout(w_pad=3)
    save(fig, out, "Fig5")
    ids = sorted({str(s).replace("MMIDB_", "S") for s in mm["g"]})
    pd.DataFrame({"EEGMMIDB_participant": ids}).to_csv(os.path.join(out, "TableS2_EEGMMIDB_participants.csv"),
                                                      index=False)
    print(f"  TableS2: {len(ids)} EEGMMIDB participants -> TableS2_EEGMMIDB_participants.csv")
    print("          " + ", ".join(ids))


# ------------------------------------------------------------------ Fig. 6 (fingerprint controls)
def fig6(xl, out):
    f = xl["Fingerprint_controls"]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 58 * MM))
    for ax, prefix, keys, letter, ttl in (
            (axs[0], "time-block pseudo-labels", ["(a) random split", "(c) cross-subject"], "A",
             "Time-block pseudo-labels"),
            (axs[1], "subject identification", ["(a) random split", "(b) within-recording time split"], "B",
             "Subject identification")):
        w = 0.36
        for j, k in enumerate(keys):
            vals, chance = [], []
            for c in COND_ORDER:
                r = f[(f.task == f"{prefix} ({c})") & (f.evaluation == k)]
                vals.append(r.accuracy.iloc[0] if len(r) else np.nan)
                chance.append(r.chance.iloc[0] if len(r) else np.nan)
            x = np.arange(3) + (j - 0.5) * w
            ax.bar(x, vals, w * 0.95, color=SCHEME_COLOR[k], label=SCHEME_LABEL[k])
        for i in range(3):
            ax.hlines(chance[i], i - 0.4, i + 0.4, colors="k", linestyles="--", lw=0.7)
        ax.set_xticks(range(3)); ax.set_xticklabels([COND_LABEL[c] for c in COND_ORDER])
        ax.set(ylim=(0, 1.02), ylabel="Accuracy")
        ax.set_title(ttl, pad=20)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5)
        panel(ax, letter, y=1.13)
    fig.tight_layout(w_pad=2)
    save(fig, out, "Fig6")


# ------------------------------------------------------------------ Fig. 7 (external validation)
def fig7(res, out):
    e = pd.read_csv(os.path.join(res, "external", "external_results.csv"))
    e = e[(e.model == "RF") & (e.epochs == "all")]
    evals = ["Cohort CV (cross-subject)", "Cohort -> EEGMMIDB", "EEGMMIDB -> Cohort", "EEGMMIDB CV (cross-subject)"]
    elab = ["Cohort,\ncross-subject", "Cohort →\nEEGMMIDB", "EEGMMIDB →\ncohort", "EEGMMIDB,\ncross-subject"]
    tasks = [("2-class (eye_open vs eye_close)", "Two-class (eyes open vs eyes closed)", "#0072B2", 0.5),
             ("3-class (+ hands_up)", "Three-class", "#E69F00", 1 / 3)]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 60 * MM), sharey=True)
    for ax, norm, letter, ttl in ((axs[0], "relative", "A", "Log relative band power"),
                                  (axs[1], "relative + subject z-score", "B", "+ subject-wise standardisation")):
        w = 0.38
        for j, (tk, tl, col, ch) in enumerate(tasks):
            vals = [e[(e.normalisation == norm) & (e.task == tk) & (e.evaluation == ev)].balanced_accuracy.mean()
                    for ev in evals]
            x = np.arange(4) + (j - 0.5) * w
            ax.bar(x, vals, w * 0.95, color=col, label=tl)
            ax.hlines(ch, -0.5, 3.5, colors="k", linestyles="--" if ch == 0.5 else ":", lw=0.7)
        ax.set_xticks(range(4)); ax.set_xticklabels(elab)
        ax.set(ylim=(0, 1.02), title=ttl); panel(ax, letter, x=-0.1)
    axs[0].set_ylabel("Balanced accuracy")
    hs, ls = axs[0].get_legend_handles_labels()
    fig.legend(hs, ls, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1)
    save(fig, out, "Fig7")


# ------------------------------------------------------------------ Fig. S1 (channel QC)
def figS1(xl, chans, out):
    q = xl["ChannelQC"]
    frac = q.groupby("channel").bad.mean().reindex(chans) * 100
    nb = q.groupby(["subject", "cond"]).bad.sum()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 55 * MM), gridspec_kw=dict(width_ratios=[1.7, 1]))
    axs[0].bar(range(len(frac)), frac.values, color="#0072B2", width=0.7)
    axs[0].set_xticks(range(len(frac))); axs[0].set_xticklabels(frac.index, rotation=90)
    axs[0].set(ylabel="Recordings flagged bad (%)"); panel(axs[0], "A", x=-0.13)
    vc = nb.value_counts().sort_index()
    axs[1].bar(vc.index, vc.values, color="#009E73", width=0.7)
    axs[1].axvline(EA.MAX_BAD_CH + 0.5, ls="--", c="k", lw=0.7)
    axs[1].set(xlabel="Bad channels per recording", ylabel="Recordings"); panel(axs[1], "B")
    axs[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout(w_pad=2)
    save(fig, out, "FigS1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=DEFAULT_RES, help="analysis output folder of one channel policy")
    ap.add_argument("--trees", type=int, default=EA.N_TREES, help="random-forest size for the Fig. 5 recomputation")
    ap.add_argument("--skip-fig5", action="store_true", help="skip the recomputation for Fig. 5")
    ap.add_argument("--tiff", choices=["plain", "lzw", "none"], default="plain",
                    help="TIFF output: plain (default, uncompressed), lzw (compressed), none")
    a = ap.parse_args()
    global TIFF_MODE
    TIFF_MODE = a.tiff
    style()
    out = os.path.join(a.res, "figures")
    print(f"Results folder: {a.res}\nFigures -> {out}")
    xl = pd.read_excel(os.path.join(a.res, "results.xlsx"), sheet_name=None)
    recs, chans = load_recs(a.res)
    steps = [("Fig1", lambda: fig1(out)), ("Fig2", lambda: fig2(recs, chans, out)),
             ("Fig3", lambda: fig3(xl, out)), ("Fig4", lambda: fig4(xl, out)),
             ("Fig6", lambda: fig6(xl, out)), ("Fig7", lambda: fig7(a.res, out)),
             ("FigS1", lambda: figS1(xl, chans, out))]
    if not a.skip_fig5:          # slowest step (recomputation) last
        steps.append(("Fig5 + TableS2", lambda: fig5(recs, chans, a.res, out, a.trees)))
    failed = []
    for name, fn in steps:
        print(f"[{name}] ...", flush=True)
        try:
            fn()
        except Exception as e:
            import traceback
            failed.append(name)
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            plt.close("all")
    print("Done." if not failed else f"Done with errors in: {', '.join(failed)} (see messages above)")


if __name__ == "__main__":
    main()
