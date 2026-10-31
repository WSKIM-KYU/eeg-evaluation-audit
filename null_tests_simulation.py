# -*- coding: utf-8 -*-
"""
Type I error and power of significance tests for EEG classification, on synthetic data with known ground truth.

Data come from the generator of simulate_fingerprints.py (participant offsets, recording-specific offsets, slow drift,
overlap-correlated window noise, optional task effect shared by all participants), with two classes and one recording
per participant and class. Scenarios:
    a  no task effect, no nuisance                     (type I error; exchangeable null)
    b  no task effect, recording offsets+drift           (type I error under nuisance; exchangeable null)
    c  task effect, no nuisance                        (power)
    d  task effect, recording offsets+drift              (power under nuisance)
    e  no task effect, nuisance + ORDER effect, fixed order      (null in which exchangeability is broken: a shift
                                                        common to all participants affects the second recording,
                                                        which always carries the same label, as in the cohort)
    f  no task effect, nuisance + order effect, counterbalanced order (half of the participants recorded in reverse
                                                        order; the order effect no longer coincides with the label)
    g  implementation check: fully i.i.d. windows (no participant offsets, no window correlation, no nuisance)
    h  implementation check: participant offsets only, i.i.d. windows (no window correlation, no nuisance)
Combinations of evaluation scheme and test (statistic: per-participant mean accuracy):
    random split + binomial test (windows treated as independent)
    random split + window-level permutation test
    random split + dependence-preserving permutation (labels exchanged between a participant's whole recordings)
    cross-subject + dependence-preserving permutation
The rejection rate at alpha = 0.05 is reported with a Clopper-Pearson 95% interval.

    python null_tests_simulation.py                 # 40 datasets x 6 scenarios, 49 permutations (several hours)
    python null_tests_simulation.py --quick         # small setup check
    python null_tests_simulation.py --scenarios g,h --tag iid      # implementation checks (i.i.d. nulls)
    python null_tests_simulation.py --scenarios b --datasets 160 --seed-offset 1000 --tag ext_b
                                                    # precision extension: independent datasets, separate output folder
Output: <EEG_OUTPUT_DIR>/null_tests_simulation/summary.md, results.csv, per_dataset.csv
"""
import os, time, argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest, beta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
import eeg_analysis as EA
import simulate_fingerprints as SF

OUT = os.path.join(EA.OUT_DIR_BASE, "null_tests_simulation")
SCENARIOS = [  # name, sd_recording, sd_drift, task effect (None = --delta), order design
    ("a: no effect, no nuisance", 0.0, 0.0, 0.0, None), ("b: no effect, nuisance", 0.3, 0.3, 0.0, None),
    ("c: effect, no nuisance", 0.0, 0.0, None, None), ("d: effect, nuisance", 0.3, 0.3, None, None),
    ("e: no effect, nuisance, order effect (fixed order)", 0.3, 0.3, 0.0, "fixed"),
    ("f: no effect, nuisance, order effect (counterbalanced)", 0.3, 0.3, 0.0, "counterbalanced"),
    ("g: i.i.d. windows (implementation check)", 0.0, 0.0, 0.0, "iid"),
    ("h: participant offsets only, i.i.d. windows (implementation check)", 0.0, 0.0, 0.0, "iid_offsets")]
DEFAULT_SCENARIOS = "a,b,c,d,e,f"     # the main analysis; g and h are run separately as implementation checks
TESTS = ["random + binomial", "random + window permutation", "random + dependence-preserving",
         "cross-subject + dependence-preserving"]


def acc(y, p, g):
    return float(EA.per_subject_acc(y, p, g).mean())


def cv_random(X, y, mk):
    p = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=EA.RANDOM_STATE).split(X, y):
        p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    return p


def cv_cross(X, y, g, mk, folds):
    p = np.empty_like(y)
    for tr, te in GroupKFold(folds).split(X, y, g):
        p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    return p


def unit_perm(y, g, rng):
    """Two classes, one recording each: swap a participant's two labels with probability 0.5."""
    yp = y.copy()
    for s in np.unique(g):
        if rng.rand() < 0.5:
            m = g == s
            a, b = np.unique(y[m])
            yp[m] = np.where(y[m] == a, b, a)
    return yp


def simulate_iid(n_sub, dur, n_feat, sd_sub, seed):
    """Two classes, one recording each; window features are independent N(0, 1) draws (no window correlation, no drift,
    no recording offset, no task effect), plus an optional participant offset shared by both recordings."""
    rng = np.random.RandomState(seed)
    starts = np.arange(0, dur - EA.EPOCH_SEC + 1, EA.STEP_SEC); nw = len(starts)
    X, y, g = [], [], []
    for s in range(n_sub):
        u = rng.randn(n_feat) * sd_sub
        for c in range(2):
            X.append(u + rng.randn(nw, n_feat)); y += [f"c{c}"] * nw; g += [f"P{s:02d}"] * nw
    return np.vstack(X).astype(np.float32), np.array(y), np.array(g)


def add_order_effect(X, y, g, size, design, seed):
    """Add a shift, common to all participants, to the recording made SECOND. Fixed order: class c1 is always second
    (order coincides with the label). Counterbalanced: a random half of the participants have c0 second."""
    rng = np.random.RandomState(seed + 777)
    patt = np.zeros(X.shape[1]); k = min(10, X.shape[1])
    idx = np.random.RandomState(54321).choice(X.shape[1], k, replace=False)   # order pattern fixed across datasets
    patt[idx] = np.random.RandomState(54322).choice([-1.0, 1.0], k)
    X = X.copy()
    subs = np.unique(g)
    flip = set(rng.choice(subs, len(subs) // 2, replace=False)) if design == "counterbalanced" else set()
    for s in subs:
        second = "c0" if s in flip else "c1"
        X[(g == s) & (y == second)] += size * patt
    return X


def one_dataset(X, y, g, B, mk, folds, rng):
    pr = cv_random(X, y, mk); obs_r = acc(y, pr, g)
    pc = cv_cross(X, y, g, mk, folds); obs_c = acc(y, pc, g)
    out = {"random + binomial": binomtest(int((pr == y).sum()), len(y), 0.5, alternative="greater").pvalue}
    nw, nu, nc = [], [], []
    for _ in range(B):
        yw = rng.permutation(y); nw.append(acc(yw, cv_random(X, yw, mk), g))
        yu = unit_perm(y, g, rng); nu.append(acc(yu, cv_random(X, yu, mk), g))
        yc = unit_perm(y, g, rng); nc.append(acc(yc, cv_cross(X, yc, g, mk, folds), g))
    pval = lambda v, o: (1 + np.sum(np.array(v) >= o)) / (len(v) + 1)
    out["random + window permutation"] = pval(nw, obs_r)
    out["random + dependence-preserving"] = pval(nu, obs_r)
    out["cross-subject + dependence-preserving"] = pval(nc, obs_c)
    return out, obs_r, obs_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=40)
    ap.add_argument("--perms", type=int, default=49)
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--dur", type=float, default=90.0)
    ap.add_argument("--features", type=int, default=30)
    ap.add_argument("--delta", type=float, default=0.3, help="task effect for scenarios c and d")
    ap.add_argument("--order", type=float, default=0.45,
                    help="order-effect size for scenarios e and f (0.45 ~ the class difference produced by --delta 0.3)")
    ap.add_argument("--trees", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="subset of scenarios by letter, e.g. b or g,h")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="added to every dataset seed; use a value >= the main run's --datasets for independent datasets")
    ap.add_argument("--tag", default="", help="output folder suffix, so that an extension does not overwrite the main run")
    a = ap.parse_args()
    if a.quick:
        a.datasets, a.perms, a.subjects, a.dur, a.trees = 4, 39, 10, 40.0, 10
    if 1.0 / (a.perms + 1) >= a.alpha:
        raise SystemExit(f"--perms {a.perms} cannot give p < {a.alpha} (minimum p = 1/(perms+1)); use at least "
                         f"{int(np.ceil(1 / a.alpha))} permutations")
    global OUT
    if a.tag:
        OUT = OUT + "_" + a.tag
    os.makedirs(OUT, exist_ok=True)
    keep = set(x.strip() for x in a.scenarios.split(","))
    if a.seed_offset and a.seed_offset < 40:
        raise SystemExit("--seed-offset must not overlap the main run's dataset indices (use e.g. 1000)")
    mk = lambda: RandomForestClassifier(a.trees, min_samples_leaf=2, n_jobs=-1, random_state=EA.RANDOM_STATE)
    folds = min(5, a.subjects)
    rows = []
    print(f"{a.datasets} datasets x {sum(n[0] in keep for n, *_ in SCENARIOS)} scenario(s); {a.subjects} participants, {a.dur:.0f}-s recordings, "
          f"{a.features} features; {a.perms} permutations; {a.trees} trees; delta {a.delta}")
    for si, (name, sr, sd, de, od) in enumerate(SCENARIOS):
        if name[0] not in keep:
            continue
        de = a.delta if de is None else de
        t0 = time.time()
        for d in range(a.datasets):
            seed = 10000 * (si + 1) + a.seed_offset + d
            if od in ("iid", "iid_offsets"):
                X, y, g = simulate_iid(a.subjects, a.dur, a.features, 1.0 if od == "iid_offsets" else 0.0, seed)
            else:
                X, y, g, ts, dur = SF.simulate(a.subjects, 2, a.dur, a.features, sr, sd, de, seed)
                if od is not None:
                    X = add_order_effect(X, y, g, a.order, od, seed)
            pv, obs_r, obs_c = one_dataset(X, y, g, a.perms, mk, folds, np.random.RandomState(seed))
            rows.append(dict(scenario=name, dataset=d, random_acc=obs_r, cross_acc=obs_c, **pv))
        sub = pd.DataFrame([r for r in rows if r["scenario"] == name])
        rates = "  ".join(f"{t.split(' + ')[0][:6]}+{t.split(' + ')[1][:6]} {100*np.mean(sub[t] < a.alpha):.0f}%" for t in TESTS)
        print(f"  {name[:40]:40s} {rates}  [{time.time()-t0:.0f}s]", flush=True)
    per = pd.DataFrame(rows); per.to_csv(os.path.join(OUT, "per_dataset.csv"), index=False)
    summ = []
    for name, g_ in per.groupby("scenario", sort=False):
        for t in TESTS:
            k, n = int((g_[t] < a.alpha).sum()), len(g_)
            lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
            hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
            summ.append(dict(scenario=name, test=t, rejection_rate=k / n, ci95_lo=lo, ci95_hi=hi, n_datasets=n,
                             mean_random_acc=g_.random_acc.mean(), mean_cross_acc=g_.cross_acc.mean()))
    S = pd.DataFrame(summ); S.to_csv(os.path.join(OUT, "results.csv"), index=False)
    L = ["# Type I error and power of evaluation-scheme x test combinations (ground-truth simulation)", "",
         f"- {a.datasets} datasets per scenario; {a.subjects} participants x 2 classes, one {a.dur:.0f}-s recording per class; "
         f"{a.features} features; random forest {a.trees} trees; {a.perms} permutations; alpha {a.alpha}; task effect {a.delta}; "
         f"order effect {a.order}; scenarios {a.scenarios}; seed offset {a.seed_offset}",
         "- rejection rate = type I error in scenarios a, b (exchangeable nulls) and e, f (order effect, no task effect), "
         "power in scenarios c, d (Clopper-Pearson 95% CI over datasets)",
         "- e: exchangeability broken by a fixed order (as in the cohort); f: same order effect, counterbalanced order", "",
         S.round(3).to_string(index=False)]
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"Saved -> {os.path.join(OUT, 'summary.md')}")


if __name__ == "__main__":
    main()
