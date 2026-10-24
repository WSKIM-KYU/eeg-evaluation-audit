# -*- coding: utf-8 -*-
"""
Seed-repetition analysis (random forest) for the random-split vs cross-subject comparison.

Everything except the random seed is taken unchanged from eeg_analysis.py: cached band-power features,
channel QC (interpolation, exclusion of recordings with > 4 bad channels), RF hyperparameters
(300 trees, min_samples_leaf = 2) and the participant-level summaries.

Step 1 - reproduction check: the ORIGINAL protocol (seed 0; StratifiedKFold(5, shuffle, 0) for the random
         split, unshuffled GroupKFold(10) for cross-subject) must reproduce results.xlsx exactly.
Step 2 - repetition: for seeds 0-4, the random-split folds are reshuffled, participants are re-assigned
         at random to 10 cross-subject folds, and the RF seed is set to the same value.
Tasks: three-class and eyes open vs arm elevation.

    python seed_repeat.py --jobs -1            (about 30-60 min on a desktop PC)
    python seed_repeat.py --trees 50           (quick test)
Output: <policy folder>\\seed_repeat.csv, seed_repeat_summary.md
"""
import os, json, time, argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import eeg_analysis as EA

RES = os.path.join(EA.OUT_DIR_BASE, "policy_common")
TASKS = {"3-class": ["eye_close", "eye_open", "hands_up"], "eye_open vs hands_up": ["eye_open", "hands_up"]}
SEEDS = [0, 1, 2, 3, 4]
N_FOLDS_CS = 10


def subject_folds(g, seed, n=N_FOLDS_CS):
    """Random assignment of participants to n folds (sizes differ by at most one participant)."""
    subs = np.unique(g)
    perm = np.random.RandomState(seed).permutation(subs)
    return [np.isin(g, part) for part in np.array_split(perm, n)]


def evaluate(X, y, g, seed, trees, jobs, original=False):
    mk = lambda: RandomForestClassifier(trees, min_samples_leaf=2, n_jobs=jobs, random_state=seed)
    if original:                                  # exactly the manuscript protocol
        EA.RANDOM_STATE = seed
        p = EA.run_schemes(X, y, g, None, None, mk, schemes=("a", "c"))
        return p["a"][1], p["c"][1]
    ya = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        ya[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    yc = np.empty_like(y)
    for te in subject_folds(g, seed):
        yc[te] = mk().fit(X[~te], y[~te]).predict(X[te])
    return ya, yc


def metrics(task, seed, protocol, y, g, ya, yc):
    pa, pc = EA.per_subject_acc(y, ya, g), EA.per_subject_acc(y, yc, g)
    d = (pa - pc).dropna()
    return dict(task=task, protocol=protocol, seed=seed, n_participants=len(pa),
                random_pooled=float((ya == y).mean()), cross_pooled=float((yc == y).mean()),
                random_subj_mean=pa.mean(), cross_subj_mean=pc.mean(),
                difference_subj_mean=d.mean(), participants_positive=int((d > 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    ap.add_argument("--trees", type=int, default=EA.N_TREES)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.res, "cache_manifest.json")))
    recs = EA.load_all([os.path.join(a.res, "cache", f) for f in man["files"]])
    recs = [EA.reorder_record(r, man["chans"]) for r in recs]
    EA.channel_qc(recs)
    D = EA.build_clean(recs)
    print(f"QC-cleaned windows: {len(D['y'])}, participants: {len(np.unique(D['g']))}, trees: {a.trees}")
    rows = []
    for task, cls in TASKS.items():
        k = np.isin(D["y"], cls)
        X, y, g = D["X"][k], D["y"][k], D["g"][k]
        jobs = [("original", 0)] + [("repetition", s) for s in SEEDS]
        for protocol, seed in jobs:
            t0 = time.time()
            ya, yc = evaluate(X, y, g, seed, a.trees, a.jobs, original=(protocol == "original"))
            r = metrics(task, seed, protocol, y, g, ya, yc); rows.append(r)
            print(f"  {task:22s} {protocol:10s} seed {seed}: random {r['random_subj_mean']:.3f}  cross "
                  f"{r['cross_subj_mean']:.3f}  difference {r['difference_subj_mean']:.3f}  "
                  f"positive {r['participants_positive']}/{r['n_participants']}  [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.res, "seed_repeat.csv"), index=False)

    # reproduction check against results.xlsx
    L = ["# Seed-repetition analysis (random forest, QC-cleaned band power)", "",
         f"- Seeds {SEEDS}; {a.trees} trees, min_samples_leaf = 2; random split = StratifiedKFold(5, shuffle); "
         f"cross-subject = participants randomly assigned to {N_FOLDS_CS} folds per seed.", "",
         "## Reproduction of the manuscript values (original protocol, seed 0)", ""]
    try:
        t = pd.read_excel(os.path.join(a.res, "results.xlsx"), sheet_name="Tasks_by_evaluation")
        t = t[t.features == "QC-cleaned band power (RF)"]
        for task in TASKS:
            o = df[(df.task == task) & (df.protocol == "original")].iloc[0]
            for ev, col in (("(a) random split", "random_subj_mean"), ("(c) cross-subject", "cross_subj_mean")):
                ref = float(t[(t.task == task) & (t.evaluation == ev)].subj_mean.iloc[0])
                ok = abs(ref - o[col]) < 1e-9
                L.append(f"- {task}, {ev}: manuscript {ref:.4f}, rerun {o[col]:.4f} -> "
                         f"{'REPRODUCED' if ok else 'DIFFERENT (check trees/version)'}")
    except Exception as e:
        L.append(f"- results.xlsx not readable ({e})")
    L += ["", "## Summary over seeds 0-4 (repetition protocol)", ""]
    rep = df[df.protocol == "repetition"]
    srows = []
    for task in TASKS:
        s = rep[rep.task == task]
        for col, name in (("random_subj_mean", "random split (per-participant mean)"),
                          ("cross_subj_mean", "cross-subject (per-participant mean)"),
                          ("difference_subj_mean", "random - cross-subject difference"),
                          ("participants_positive", "participants with positive difference")):
            v = s[col].astype(float)
            srows.append(dict(task=task, metric=name, mean=v.mean(), sd=v.std(ddof=1), min=v.min(), max=v.max()))
    summ = pd.DataFrame(srows)
    L += [summ.round(4).to_string(index=False), "", "## Per seed", "", df.round(4).to_string(index=False)]
    open(os.path.join(a.res, "seed_repeat_summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:12]))
    print(f"\nSaved -> {os.path.join(a.res, 'seed_repeat_summary.md')}")


if __name__ == "__main__":
    main()
