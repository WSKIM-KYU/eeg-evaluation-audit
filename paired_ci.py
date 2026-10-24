# -*- coding: utf-8 -*-
"""
Paired bootstrap 95% CIs for within-participant accuracy differences between evaluation schemes.
Uses the per-participant accuracies that eeg_analysis.py (results.xlsx) and eegnet_validation.py
(eegnet_per_subject.csv) already saved, so nothing has to be retrained.

    python paired_ci.py                                    # primary analysis (policy_common)
    python paired_ci.py --res analysis_output\\policy_common_step2s

Output: <res>\\paired_differences.csv (+ printed table)
"""
import os, argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = os.path.join(os.environ.get("EEG_OUTPUT_DIR", "analysis_output"), "policy_common")
N_BOOT, SEED = 10000, 0
PAIRS = [("(a) random split", "(c) cross-subject"), ("(a) random split", "(b) within-recording time split"),
         ("(b) within-recording time split", "(c) cross-subject")]
SHORT = {"(a) random split": "random", "(b) within-recording time split": "time split", "(c) cross-subject": "cross-subject"}


def paired_rows(df, model):
    rows = []
    rng = np.random.RandomState(SEED)
    for (task, feat), d in df.groupby(["task", "features"]):
        piv = d.pivot_table(index="subject", columns="evaluation", values="accuracy")
        for e1, e2 in PAIRS:
            if e1 not in piv or e2 not in piv:
                continue
            diff = (piv[e1] - piv[e2]).dropna().values
            if len(diff) < 5:
                continue
            boots = diff[rng.randint(0, len(diff), (N_BOOT, len(diff)))].mean(1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan
            rows.append(dict(model=model, task=task, features=feat, comparison=f"{SHORT[e1]} - {SHORT[e2]}",
                             n_participants=len(diff), mean_difference=diff.mean(), ci95_lo=lo, ci95_hi=hi,
                             n_positive=int((diff > 0).sum()), p_wilcoxon=p))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    a = ap.parse_args()
    rows = []
    x = os.path.join(a.res, "results.xlsx")
    if os.path.exists(x):
        ps = pd.read_excel(x, sheet_name="PerSubject_accuracy")
        rows += paired_rows(ps, "RF / band power")
    else:
        print(f"[skip] {x} not found")
    e = os.path.join(a.res, "eegnet", "eegnet_per_subject.csv")
    if os.path.exists(e):
        en = pd.read_csv(e).assign(features="EEGNet (raw signal)")
        rows += paired_rows(en, "EEGNet")
    out = pd.DataFrame(rows)
    if out.empty:
        print("No per-participant results found."); return
    out.to_csv(os.path.join(a.res, "paired_differences.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print(f"\nSaved -> {os.path.join(a.res, 'paired_differences.csv')}")


if __name__ == "__main__":
    main()
