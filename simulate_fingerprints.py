# -*- coding: utf-8 -*-
"""
Ground-truth simulation: can recording-specific structure alone inflate random-split accuracy?

Synthetic band-power features follow the cohort's layout (50 participants, 3 conditions, one 300-s
recording per condition, 2-s windows with a 1-s step, 18 channels x 5 bands = 90 features):

    x[s,c,t,:] = sd_subject * u_s            participant offset (same in all of the participant's recordings)
               + sd_recording * v_sc         recording-specific offset (one per participant x condition)
               + d_sc(t)                     slow within-recording drift, AR(1) with rho = 0.98, SD sd_drift
               + e_sc(t)                     window noise, correlated 0.5 between adjacent (overlapping) windows
               + delta * m_c                 TRUE task effect: a fixed class pattern shared by all participants

Because the ground truth is known, accuracy above chance is "spurious" whenever delta = 0.
The cohort's evaluation code (eeg_analysis.run_schemes) and random-forest settings are used unchanged.

Scenarios: (1) no nuisance, no task effect; (2) recording offsets only; (3) drift only;
(4) offsets + drift; (5) true effect without nuisance; (6) true effect + nuisance; plus a dose-response
series over sd_recording with no task effect. Label-free diagnostics (time-block pseudo-labels,
participant identification) are computed for scenarios 1-6.

    python simulate_fingerprints.py                  # full run (about 1-3 h on a desktop PC)
    python simulate_fingerprints.py --quick          # smaller run to check the setup (minutes)
Output: <EEG_OUTPUT_DIR>/simulation/summary.md, results.csv
"""
import os, time, argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
import eeg_analysis as EA

OUT = os.path.join(EA.OUT_DIR_BASE, "simulation")
SCENARIOS = [  # name, sd_recording, sd_drift, delta
    ("1 no nuisance, no task effect", 0.0, 0.0, 0.0),
    ("2 recording offsets only", 0.3, 0.0, 0.0),
    ("3 slow drift only", 0.0, 0.3, 0.0),
    ("4 offsets + drift, no task effect", 0.3, 0.3, 0.0),
    ("5 true effect, no nuisance", 0.0, 0.0, 0.3),
    ("6 true effect + offsets + drift", 0.3, 0.3, 0.3),
]
DOSE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def simulate(n_sub, n_cond, dur, n_feat, sd_rec, sd_drift, delta, seed, sd_sub=1.0, rho=0.98, k_task=10):
    rng = np.random.RandomState(seed)
    starts = np.arange(0, dur - EA.EPOCH_SEC + 1, EA.STEP_SEC)          # window onsets (s)
    n_win = len(starts)
    pat_rng = np.random.RandomState(12345)                               # task pattern fixed across seeds
    patt = np.zeros((n_cond, n_feat))
    for c in range(n_cond):
        idx = pat_rng.choice(n_feat, k_task, replace=False)
        patt[c, idx] = pat_rng.choice([-1.0, 1.0], k_task)
    X, y, g, ts = [], [], [], []
    for s in range(n_sub):
        u = rng.randn(n_feat)
        for c in range(n_cond):
            v = rng.randn(n_feat)
            d = np.zeros((n_win, n_feat))
            if sd_drift > 0:
                d[0] = rng.randn(n_feat) * sd_drift
                for t in range(1, n_win):
                    d[t] = rho * d[t - 1] + np.sqrt(1 - rho ** 2) * sd_drift * rng.randn(n_feat)
            half = rng.randn(n_win + 1, n_feat)
            e = (half[:-1] + half[1:]) / np.sqrt(2)                        # adjacent windows share half their data
            X.append(sd_sub * u + sd_rec * v + d + e + delta * patt[c])
            y += [f"c{c}"] * n_win; g += [f"P{s:02d}"] * n_win; ts.append(starts)
    return np.vstack(X).astype(np.float32), np.array(y), np.array(g), np.concatenate(ts), np.full(len(y), float(dur))


def summary_row(name, out, chance, extra=None):
    row = dict(scenario=name, chance=chance)
    acc = {}
    for k, (yt, yp, gs) in out.items():
        ps = EA.per_subject_acc(yt, yp, gs); acc[k] = ps
        lo, hi = EA.bootstrap_ci(ps.values)
        row.update({f"{k}_mean": ps.mean(), f"{k}_lo": lo, f"{k}_hi": hi})
    if "a" in acc and "c" in acc:
        d = (acc["a"] - acc["c"]).dropna(); row["gap_random_minus_cross"] = d.mean()
    row.update(extra or {})
    return row


def diagnostics(X, y, g, ts, dur, mk):
    """Pseudo-labels and participant identification within one condition (c0)."""
    m = y == "c0"
    third = dur[m] / 3
    k1, k2 = np.floor(ts[m] / third), np.floor((ts[m] + EA.EPOCH_SEC - 1e-9) / third)
    keep = k1 == k2
    Xb, yb, gb = X[m][keep], k1[keep].astype(int).astype(str), g[m][keep]
    p = np.empty_like(yb)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=EA.RANDOM_STATE).split(Xb, yb):
        p[te] = mk().fit(Xb[tr], yb[tr]).predict(Xb[te])
    pr = EA.per_subject_acc(yb, p, gb).mean()
    p = np.empty_like(yb)
    for tr, te in GroupKFold(10).split(Xb, yb, gb):
        p[te] = mk().fit(Xb[tr], yb[tr]).predict(Xb[te])
    pc = EA.per_subject_acc(yb, p, gb).mean()
    gm = g[m]; p = np.empty_like(gm)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=EA.RANDOM_STATE).split(X[m], gm):
        p[te] = mk().fit(X[m][tr], gm[tr]).predict(X[m][te])
    pid = float((p == gm).mean())
    return {"pseudo_random": pr, "pseudo_cross": pc, "participant_id_random": pid,
            "participant_id_chance": 1 / len(np.unique(gm))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=50)
    ap.add_argument("--dur", type=float, default=300.0)
    ap.add_argument("--features", type=int, default=90)
    ap.add_argument("--trees", type=int, default=EA.N_TREES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dose", action="store_true")
    ap.add_argument("--quick", action="store_true", help="20 participants, 120-s recordings, 100 trees")
    a = ap.parse_args()
    if a.quick:
        a.subjects, a.dur, a.trees = 20, 120.0, 100
    os.makedirs(OUT, exist_ok=True)
    mk = lambda: RandomForestClassifier(a.trees, min_samples_leaf=2, n_jobs=-1, random_state=EA.RANDOM_STATE)
    rows = []
    print(f"Simulation: {a.subjects} participants x 3 conditions, {a.dur:.0f}-s recordings, {a.features} features, "
          f"{a.trees} trees, seed {a.seed}")
    for name, sr, sd, de in SCENARIOS:
        t0 = time.time()
        X, y, g, ts, dur = simulate(a.subjects, 3, a.dur, a.features, sr, sd, de, a.seed)
        out = EA.run_schemes(X, y, g, ts, dur, mk)
        r = summary_row(name, out, 1 / 3, dict(sd_recording=sr, sd_drift=sd, delta=de, **diagnostics(X, y, g, ts, dur, mk)))
        rows.append(r)
        print(f"  {name:36s} random {r['a_mean']:.3f}  time {r['b_mean']:.3f}  cross {r['c_mean']:.3f}  "
              f"pseudo(random) {r['pseudo_random']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    if not a.no_dose:
        for sr in DOSE:
            t0 = time.time()
            X, y, g, ts, dur = simulate(a.subjects, 3, a.dur, a.features, sr, 0.0, 0.0, a.seed)
            out = EA.run_schemes(X, y, g, ts, dur, mk, schemes=("a", "c"))
            r = summary_row(f"dose sd_recording={sr}", out, 1 / 3, dict(sd_recording=sr, sd_drift=0.0, delta=0.0))
            rows.append(r)
            print(f"  dose sd_recording={sr:<4}  random {r['a_mean']:.3f}  cross {r['c_mean']:.3f}  [{time.time()-t0:.0f}s]",
                  flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "results.csv"), index=False)
    L = ["# Ground-truth simulation of recording fingerprints", "",
         f"- {a.subjects} participants x 3 conditions; {a.dur:.0f}-s recordings; 2-s windows, 1-s step; "
         f"{a.features} features; random forest {a.trees} trees; seed {a.seed}",
         "- a = random split, b = within-recording time split, c = cross-subject (per-participant means, bootstrap 95% CI)",
         "- delta = 0 means no task information: any accuracy above chance is spurious", "",
         res.round(4).to_string(index=False)]
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"Saved -> {os.path.join(OUT, 'summary.md')}")


if __name__ == "__main__":
    main()
