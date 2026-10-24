# -*- coding: utf-8 -*-
"""
Sample-size audit for the manuscript.

1. Lists every recording excluded by channel QC (participant, condition, bad channels).
2. Counts, for each analysis, how many participants / recordings / windows actually enter it, and how
   many participants have ALL classes of the task (complete cases); compares with results.xlsx.
3. Complete-case sensitivity: re-runs the QC-cleaned RF analyses using only participants who have
   every class of the task (same three evaluation schemes, same settings as eeg_analysis.py).

    python check_samples.py                 # primary analysis (policy_common), ~5 min
    python check_samples.py --no-rerun      # audit only, seconds
Outputs (in the policy folder): sample_audit.md, qc_excluded_recordings.csv, complete_case_results.csv
"""
import os, json, argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import eeg_analysis as EA

RES = os.path.join(EA.OUT_DIR_BASE, "policy_common")
TASKS = {"3-class": ["eye_close", "eye_open", "hands_up"],
         "eye_close vs eye_open": ["eye_close", "eye_open"],
         "eye_close vs hands_up": ["eye_close", "hands_up"],
         "eye_open vs hands_up": ["eye_open", "hands_up"]}


def boot_ci(v, n=10000, seed=0):
    rng = np.random.RandomState(seed); v = np.asarray(v)
    m = v[rng.randint(0, len(v), (n, len(v)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    ap.add_argument("--no-rerun", action="store_true")
    ap.add_argument("--trees", type=int, default=EA.N_TREES)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.res, "cache_manifest.json")))
    recs = EA.load_all([os.path.join(a.res, "cache", f) for f in man["files"]])
    recs = [EA.reorder_record(r, man["chans"]) for r in recs]
    EA.channel_qc(recs)

    # ---------- 1. recording-level table ----------
    rows = []
    for r in recs:
        bad = [str(c) for c, b in zip(r["chans"], r["bad"]) if b]
        rows.append(dict(subject=str(r["subject"]), cond=str(r["cond"]), n_windows=len(r["feats"]),
                         n_bad=len(bad), bad_channels=";".join(bad), excluded=len(bad) > EA.MAX_BAD_CH))
    rec = pd.DataFrame(rows)
    exc = rec[rec.excluded].sort_values(["subject", "cond"])
    exc.to_csv(os.path.join(a.res, "qc_excluded_recordings.csv"), index=False)
    kept = rec[~rec.excluded]

    L = ["# Sample-size audit", "", f"- Results folder: {a.res}",
         f"- Recordings: {len(rec)} ({rec.subject.nunique()} participants); channels: {len(man['chans'])}",
         f"- Excluded by channel QC (> {EA.MAX_BAD_CH} bad channels): {len(exc)} recordings from "
         f"{exc.subject.nunique()} participants; by condition: "
         + ", ".join(f"{c} {int((exc.cond == c).sum())}" for c in sorted(rec.cond.unique())), "",
         "## Excluded recordings", "", exc.to_string(index=False) if len(exc) else "(none)", "",
         "## Participants and windows entering each QC-cleaned analysis", ""]

    # ---------- 2. per-analysis counts vs results.xlsx ----------
    xl = os.path.join(a.res, "results.xlsx")
    tasks_x = pd.read_excel(xl, sheet_name="Tasks_by_evaluation") if os.path.exists(xl) else None
    fing_x = pd.read_excel(xl, sheet_name="Fingerprint_controls") if os.path.exists(xl) else None
    count_rows = []
    for t, cls in TASKS.items():
        k = kept[kept.cond.isin(cls)]
        per = k.groupby("subject").cond.nunique()
        rep = np.nan
        if tasks_x is not None:
            m = tasks_x[(tasks_x.task == t) & (tasks_x.features == "QC-cleaned band power (RF)")]
            rep = int(m.n_subjects.iloc[0]) if len(m) else np.nan
        count_rows.append(dict(analysis=t, participants_any_class=int(len(per)),
                               participants_all_classes=int((per == len(cls)).sum()),
                               recordings=int(len(k)), windows=int(k.n_windows.sum()),
                               reported_n_in_results=rep,
                               match=(rep == len(per)) if rep == rep else "n/a"))
    for c in sorted(rec.cond.unique()):
        k = kept[kept.cond == c]
        rep = np.nan
        if fing_x is not None:
            m = fing_x[fing_x.task == f"subject identification ({c})"]
            rep = int(m.n_subjects.iloc[0]) if len(m) else np.nan
        count_rows.append(dict(analysis=f"single condition: {c}", participants_any_class=int(k.subject.nunique()),
                               participants_all_classes=int(k.subject.nunique()), recordings=int(len(k)),
                               windows=int(k.n_windows.sum()), reported_n_in_results=rep,
                               match=(rep == k.subject.nunique()) if rep == rep else "n/a"))
    cnt = pd.DataFrame(count_rows)
    L += [cnt.to_string(index=False), "",
          "participants_any_class = participants with at least one usable recording of the task's classes "
          "(this is the n used for per-participant accuracies in results.xlsx); participants_all_classes = "
          "complete cases.", ""]
    print("\n".join(L))

    # ---------- 3. complete-case sensitivity ----------
    if not a.no_rerun:
        from sklearn.ensemble import RandomForestClassifier
        rf = lambda: RandomForestClassifier(a.trees, min_samples_leaf=2, n_jobs=-1, random_state=EA.RANDOM_STATE)
        D = EA.build_clean(recs)
        out = []
        for t, cls in TASKS.items():
            k = np.isin(D["y"], cls)
            g = D["g"][k]; y = D["y"][k]
            full = [s for s in np.unique(g) if set(np.unique(y[g == s])) == set(cls)]
            m = np.isin(g, full)
            X, yy, gg, ts, du = D["X"][k][m], y[m], g[m], D["ts"][k][m], D["dur"][k][m]
            preds = EA.run_schemes(X, yy, gg, ts, du, rf)
            accs = {}
            for sch, (yt, yp, gs) in preds.items():
                ps = EA.per_subject_acc(yt, yp, gs); accs[sch] = ps
                lo, hi = EA.bootstrap_ci(ps.values)
                p = wilcoxon(ps.values - 1 / len(cls)).pvalue if np.any(ps.values != 1 / len(cls)) else np.nan
                out.append(dict(task=t, analysis="complete cases", evaluation=EA.SCHEME_NAME[sch],
                                n_participants=len(ps), pooled_accuracy=float((yt == yp).mean()),
                                subj_mean=ps.mean(), ci95_lo=lo, ci95_hi=hi, p_vs_chance=p))
            d = (accs["a"] - accs["c"]).dropna()
            lo, hi = boot_ci(d.values)
            out.append(dict(task=t, analysis="complete cases", evaluation="random - cross-subject (paired)",
                            n_participants=len(d), pooled_accuracy=np.nan, subj_mean=d.mean(), ci95_lo=lo,
                            ci95_hi=hi, p_vs_chance=wilcoxon(d.values).pvalue))
            print(f"  {t}: complete cases n = {len(full)} | " + "  ".join(
                f"{r['evaluation'][:3]} {r['subj_mean']:.3f}" for r in out[-4:]))
        res = pd.DataFrame(out)
        res.to_csv(os.path.join(a.res, "complete_case_results.csv"), index=False)
        L += ["## Complete-case sensitivity (QC-cleaned band power, RF; only participants with every class)", "",
              res.round(4).to_string(index=False)]
    open(os.path.join(a.res, "sample_audit.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"\nSaved -> {os.path.join(a.res, 'sample_audit.md')}")


if __name__ == "__main__":
    main()
