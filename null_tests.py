# -*- coding: utf-8 -*-
"""
Null-hypothesis experiments: do standard statistical safeguards detect spurious random-split accuracy?

For a classifier evaluated with the RANDOM WINDOW SPLIT (5 stratified folds), three significance tests are compared:
  1. binomial test          test windows treated as independent trials (common practice)
  2. window-level permutation   labels shuffled across windows, the whole random-split CV refitted (common practice)
  3. dependence-preserving permutation  labels permuted within each participant between whole dependence units,
                                recordings (block designs) or trials (interleaved design); windows keep the label of their
                                unit, so recording/trial fingerprints are preserved under the null. With one recording per
                                condition and a fixed order, this is a CONDITIONAL null (labels carry no information beyond
                                what distinguishes a participant's units); it cannot separate a task effect from an order effect.
Procedure for every permutation (pre-specified): the same fold-generation rule is applied to the permuted labels
(StratifiedKFold(5, shuffle, seed 0)), and the classifier is refitted in every fold. Band-power features, channel QC and
interpolation are computed without labels, so they are identical under every permutation and need not be recomputed.
plus the effective sample size of the test: within-recording autocorrelation of out-of-fold correctness gives
n_eff = sum over recordings of n_r / (1 + 2 * sum_k rho_k), compared with the number of windows, recordings, participants.
Optionally (--with-cross) the cross-subject scheme is tested with the unit-level permutation as a reference.

Tasks: cohort eyes open vs arm elevation (negative control), cohort eyes closed vs eyes open, EEGMMIDB block design
(eyes open vs eyes closed) and EEGMMIDB interleaved design (imagined left vs right fist).

    python null_tests.py                       # all tasks, 100 permutations each (several hours)
    python null_tests.py --tasks cohort_eo_hu --perms 999      # main-text resolution (p >= 0.001)
    python null_tests.py --tasks cohort_eo_hu --perms 200
    python null_tests.py --quick               # 20 permutations, 50 trees: setup check
Output: <EEG_OUTPUT_DIR>/null_tests/summary.md, results.csv, null_distributions.csv
"""
import os, glob, time, argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest, beta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
import eeg_analysis as EA
import calibrated_simulation as CS

OUT = os.path.join(EA.OUT_DIR_BASE, "null_tests")
TASKS = {
    "cohort_eo_hu": "cohort: eyes open vs arm elevation (negative control)",
    "cohort_ec_eo": "cohort: eyes closed vs eyes open",
    "mmidb_block": "EEGMMIDB block design: eyes open vs eyes closed",
    "mmidb_interleaved": "EEGMMIDB interleaved design: imagined left vs right fist",
}


# ------------------------------ data ------------------------------
def cohort_task(res, classes):
    D = CS.load_cohort(res)
    k = np.isin(D["y"], classes)
    T = {key: D[key][k] for key in ("X", "y", "g", "ts", "dur", "rec")}
    T["unit"] = T["rec"]                                   # dependence unit = recording
    return T


def mmidb_task(cache_dir, design):
    files = sorted(glob.glob(os.path.join(cache_dir, "pub_S*.npz")))
    if not files:
        raise FileNotFoundError(f"no public_replication caches in {cache_dir}")
    P = {k: [] for k in ("X", "y", "g", "ts", "dur", "unit", "rec")}
    for f in files:
        z = np.load(f, allow_pickle=False)
        m = z["design"] == design
        s = str(z["subject"]); n = int(m.sum())
        P["X"].append(z["feats"][m]); P["y"] += list(z["label"][m]); P["g"] += [s] * n
        P["ts"].append(z["onset"][m]); P["dur"].append(z["dur"][m])
        rec = [f"{s}|{r}" for r in z["rec"][m]]
        P["rec"] += rec
        P["unit"] += rec if design == "A" else [f"{s}|{r}|{t}" for r, t in zip(z["rec"][m], z["trial"][m])]
    T = {k: (np.vstack(v) if k == "X" else np.concatenate(v) if k in ("ts", "dur") else np.array(v)) for k, v in P.items()}
    return T


def load_task(name, a):
    if name == "cohort_eo_hu":
        return cohort_task(a.res, ["eye_open", "hands_up"])
    if name == "cohort_ec_eo":
        return cohort_task(a.res, ["eye_close", "eye_open"])
    if name == "mmidb_block":
        return mmidb_task(a.pub_cache, "A")
    return mmidb_task(a.pub_cache, "B")


# ------------------------------ evaluation ------------------------------
def random_split_pred(X, y, mk):
    p = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=EA.RANDOM_STATE).split(X, y):
        p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    return p


def cross_subject_pred(X, y, g, mk):
    p = np.empty_like(y)
    for tr, te in GroupKFold(10).split(X, y, g):
        p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    return p


def score(y, p, g):
    return float(EA.per_subject_acc(y, p, g).mean())


def unit_permutation(y, g, unit, rng):
    """Permute labels between whole units within each participant (windows inherit their unit's label)."""
    yp = y.copy()
    for s in np.unique(g):
        m = g == s
        units = np.unique(unit[m])
        labs = np.array([y[m][unit[m] == u][0] for u in units])
        perm = rng.permutation(labs)
        for u, l in zip(units, perm):
            yp[m & (unit == u)] = l
    return yp


def effective_n(correct, rec, max_lag=60):
    """Effective number of independent test outcomes from the within-recording ACF of correctness."""
    c = correct.astype(float)
    total, n_eff = 0, 0.0
    lags = np.arange(1, max_lag + 1)
    num, den = np.zeros(len(lags)), np.zeros(len(lags))
    var = []
    for r in np.unique(rec):
        x = c[rec == r]
        if len(x) < 3 or x.std() == 0:
            continue
        x = x - x.mean(); var.append(np.mean(x ** 2))
        for i, k in enumerate(lags):
            if len(x) > k:
                num[i] += np.sum(x[:-k] * x[k:]); den[i] += len(x) - k
    v = np.mean(var) if var else 1.0
    rho = np.where(den > 0, num / np.maximum(den, 1) / v, 0.0)
    cut = np.argmax(rho <= 0) if np.any(rho <= 0) else len(rho)   # sum the ACF up to its first non-positive lag
    infl = 1 + 2 * np.sum(rho[:cut])
    for r in np.unique(rec):
        n_eff += np.sum(rec == r) / infl
    return float(n_eff), float(infl)


def run_task(name, a, mk):
    T = load_task(name, a)
    X, y, g, unit, rec = T["X"], T["y"], T["g"], T["unit"], T["rec"]
    order = np.lexsort((T["ts"], rec)); X, y, g, unit, rec = X[order], y[order], g[order], unit[order], rec[order]
    classes = np.unique(y); chance = 1 / len(classes)
    print(f"\n=== {TASKS[name]}: {len(y)} windows, {len(np.unique(unit))} units, {len(np.unique(g))} participants", flush=True)
    t0 = time.time()
    p_obs = random_split_pred(X, y, mk)
    obs = score(y, p_obs, g)
    correct = p_obs == y
    k, n = int(correct.sum()), len(y)
    p_binom = binomtest(k, n, chance, alternative="greater").pvalue
    n_eff, infl = effective_n(correct, rec)
    print(f"  observed random-split accuracy {obs:.3f}  (binomial p = {p_binom:.2e}; n = {n}, n_eff = {n_eff:.0f}) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    rng = np.random.RandomState(a.seed)
    nulls = {"window": [], "unit": []}
    for b in range(a.perms):
        t0 = time.time()
        yw = rng.permutation(y)
        nulls["window"].append(score(yw, random_split_pred(X, yw, mk), g))
        yu = unit_permutation(y, g, unit, rng)
        nulls["unit"].append(score(yu, random_split_pred(X, yu, mk), g))
        if (b + 1) % max(1, a.perms // 10) == 0 or b == 0:
            print(f"  permutation {b+1}/{a.perms}: window null {nulls['window'][-1]:.3f}  unit null {nulls['unit'][-1]:.3f}  "
                  f"[{time.time()-t0:.0f}s each]", flush=True)
    row = dict(task=name, description=TASKS[name], n_windows=n, n_units=len(np.unique(unit)),
               n_recordings=len(np.unique(rec)), n_participants=len(np.unique(g)), chance=chance,
               observed_random_split=obs, binomial_p=p_binom, n_eff=n_eff, variance_inflation=infl, perms=a.perms)
    for kind in ("window", "unit"):
        v = np.array(nulls[kind]); kx = int(np.sum(v >= obs)); B = len(v)
        row[f"{kind}_null_mean"] = v.mean(); row[f"{kind}_null_sd"] = v.std(ddof=1); row[f"{kind}_null_95pct"] = np.percentile(v, 95)
        row[f"{kind}_perm_p"] = (1 + kx) / (B + 1)
        row[f"{kind}_p_mc_lo"] = beta.ppf(0.025, kx, B - kx + 1) if kx > 0 else 0.0   # Clopper-Pearson, exceedance rate
        row[f"{kind}_p_mc_hi"] = beta.ppf(0.975, kx + 1, B - kx) if kx < B else 1.0
    ps = EA.per_subject_acc(y, p_obs, g)
    lo_b, hi_b = EA.bootstrap_ci(ps.values)
    row.update(window_level_ci_halfwidth=1.96 * np.sqrt(obs * (1 - obs) / n),
               participant_bootstrap_lo=lo_b, participant_bootstrap_hi=hi_b)
    if a.with_cross:
        pc = cross_subject_pred(X, y, g, mk); obs_c = score(y, pc, g)
        vc = []
        for b in range(a.perms):
            yu = unit_permutation(y, g, unit, rng); vc.append(score(yu, cross_subject_pred(X, yu, g, mk), g))
        vc = np.array(vc)
        row.update(observed_cross_subject=obs_c, cross_unit_null_mean=vc.mean(),
                   cross_unit_perm_p=(1 + np.sum(vc >= obs_c)) / (len(vc) + 1))
        nulls["cross_unit"] = list(vc)
    dist = pd.DataFrame([dict(task=name, kind=k_, value=v_) for k_, vs in nulls.items() for v_ in vs])
    print(f"  window-level permutation: null mean {row['window_null_mean']:.3f}, p = {row['window_perm_p']:.4f} | "
          f"dependence-preserving: null mean {row['unit_null_mean']:.3f}, p = {row['unit_perm_p']:.4f} "
          f"(MC 95% CI {row['unit_p_mc_lo']:.3f}-{row['unit_p_mc_hi']:.3f})", flush=True)
    return row, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-cross", action="store_true", help="also test cross-subject accuracy (unit-level null)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--res", default=os.path.join(EA.OUT_DIR_BASE, "policy_common"))
    ap.add_argument("--pub-cache", default=os.path.join(EA.OUT_DIR_BASE, "public_replication", "cache"))
    a = ap.parse_args()
    if a.quick:
        a.perms, a.trees = 20, 50
    os.makedirs(OUT, exist_ok=True)
    mk = lambda: RandomForestClassifier(a.trees, min_samples_leaf=2, n_jobs=-1, random_state=EA.RANDOM_STATE)
    rows, dists = [], []
    for name in [t.strip() for t in a.tasks.split(",") if t.strip()]:
        try:
            r, d = run_task(name, a, mk); rows.append(r); dists.append(d)
        except FileNotFoundError as e:
            print(f"  [SKIP] {name}: {e}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "results.csv"), index=False)
    if dists:
        pd.concat(dists).to_csv(os.path.join(OUT, "null_distributions.csv"), index=False)
    cols = ["task", "n_windows", "n_units", "n_participants", "observed_random_split", "binomial_p",
            "window_null_mean", "window_perm_p", "unit_null_mean", "unit_null_sd", "unit_null_95pct", "unit_perm_p",
            "unit_p_mc_lo", "unit_p_mc_hi", "window_level_ci_halfwidth", "participant_bootstrap_lo",
            "participant_bootstrap_hi", "n_eff"]
    cols += [c for c in ("observed_cross_subject", "cross_unit_null_mean", "cross_unit_perm_p") if c in res]
    L = ["# Null-hypothesis experiments for random-split accuracy", "",
         f"- Random forest {a.trees} trees; {a.perms} permutations per null; seed {a.seed}; "
         "accuracy = per-participant mean; permutation p = (1 + #null >= observed) / (perms + 1)",
         "- window = labels shuffled across windows; unit = dependence-preserving permutation (labels exchanged within "
         "participants between whole recordings in block designs or trials in the interleaved design; a conditional null)",
         "- unit_p_mc_lo/hi = Clopper-Pearson 95% CI of the exceedance probability (Monte Carlo uncertainty of p)",
         "- window_level_ci_halfwidth = binomial 95% half-width treating windows as independent; compare with the "
         "participant-level bootstrap interval",
         "- n_eff (auxiliary only) = effective number of independent outcomes from the within-recording ACF of "
         "correctness; not a bound on the number of independent samples", "",
         res[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}")]
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"\nSaved -> {os.path.join(OUT, 'summary.md')}")


if __name__ == "__main__":
    main()
