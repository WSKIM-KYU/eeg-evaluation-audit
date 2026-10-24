# -*- coding: utf-8 -*-
"""
Fold-level sensitivity analysis for the cross-subject comparison (reviewer point: participants in the
same cross-subject fold share one trained model, so they are not independent observations).

The ten cross-subject folds are reconstructed exactly (same data, same GroupKFold call as
eeg_analysis.py), and the per-participant accuracies already stored in results.xlsx are averaged
within each fold. Each fold then contributes ONE observation:
  - fold-level mean difference (random split - cross-subject) with a bootstrap 95% CI over folds
  - exact Wilcoxon signed-rank test over the 10 fold-level differences (smallest possible p = 0.002)
No model is retrained.

    python fold_level.py
Output: <policy folder>\\fold_level.md and fold_level.csv
"""
import os, json, argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold
import eeg_analysis as EA

RES = os.path.join(EA.OUT_DIR_BASE, "policy_common")
TASKS = {"3-class": ["eye_close", "eye_open", "hands_up"],
         "eye_close vs eye_open": ["eye_close", "eye_open"],
         "eye_close vs hands_up": ["eye_close", "hands_up"],
         "eye_open vs hands_up": ["eye_open", "hands_up"]}
FEAT = "QC-cleaned band power (RF)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.res, "cache_manifest.json")))
    recs = EA.load_all([os.path.join(a.res, "cache", f) for f in man["files"]])
    recs = [EA.reorder_record(r, man["chans"]) for r in recs]
    EA.channel_qc(recs)
    D = EA.build_clean(recs)
    ps = pd.read_excel(os.path.join(a.res, "results.xlsx"), sheet_name="PerSubject_accuracy")
    ps = ps[ps.features == FEAT].copy()
    ps["subject"] = ps["subject"].astype(str)
    rng = np.random.RandomState(0)
    rows, fold_rows = [], []
    for t, cls in TASKS.items():
        k = np.isin(D["y"], cls)
        X, y, g = D["X"][k], D["y"][k], D["g"][k]
        n_folds = min(10, len(np.unique(g)))
        piv = ps[ps.task == t].pivot_table(index="subject", columns="evaluation", values="accuracy")
        diffs = []
        for fi, (_, te) in enumerate(GroupKFold(n_folds).split(X, y, g), 1):
            subs = sorted(set(g[te]))
            ra = piv.loc[subs, "(a) random split"].mean()
            cs = piv.loc[subs, "(c) cross-subject"].mean()
            diffs.append(ra - cs)
            fold_rows.append(dict(task=t, fold=fi, n_participants=len(subs), random_split=ra, cross_subject=cs,
                                  difference=ra - cs, participants=" ".join(subs)))
        d = np.array(diffs)
        boots = d[rng.randint(0, len(d), (10000, len(d)))].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = wilcoxon(d).pvalue if np.any(d != 0) else np.nan
        rows.append(dict(task=t, n_folds=len(d), mean_difference=d.mean(), ci95_lo=lo, ci95_hi=hi,
                         folds_positive=int((d > 0).sum()), p_wilcoxon=p,
                         cross_subject_fold_mean=np.mean([r["cross_subject"] for r in fold_rows if r["task"] == t])))
        print(f"  {t}: fold-level difference {d.mean():.3f} [{lo:.3f}, {hi:.3f}], positive in "
              f"{(d > 0).sum()}/{len(d)} folds, p = {p:.4f}")
    res, folds = pd.DataFrame(rows), pd.DataFrame(fold_rows)
    res.to_csv(os.path.join(a.res, "fold_level.csv"), index=False)
    L = ["# Fold-level sensitivity analysis (cross-subject folds as units)", "",
         "Per-participant accuracies (QC-cleaned band power, RF; results.xlsx) averaged within each of the "
         "ten cross-subject folds; one observation per fold.", "",
         "## Summary", "", res.round(4).to_string(index=False), "",
         "## Folds", "", folds.round(4).to_string(index=False)]
    open(os.path.join(a.res, "fold_level.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"\nSaved -> {os.path.join(a.res, 'fold_level.md')}")


if __name__ == "__main__":
    main()
