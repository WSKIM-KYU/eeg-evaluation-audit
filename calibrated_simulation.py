# -*- coding: utf-8 -*-
"""
Data-calibrated simulation of recording fingerprints.

1. VARIANCE DECOMPOSITION of the real window-level band-power features (90 = 18 channels x 5 bands)
   into components with full cross-feature covariance:
       condition effect     delta_c   (true task information: condition means)
       participant offset   Sigma_s   (shared by all recordings of a participant)
       recording offset     Sigma_r   (specific to one participant x condition recording)
       slow drift           share A of the within-recording residual, AR(1) with rho (from the ACF)
       fast noise           remaining within-recording residual (adjacent windows correlated 0.5)
   Method of moments for a nested design; the mean-noise of each recording mean is corrected using the
   fitted autocorrelation, and covariance matrices are projected to positive semi-definite.

2. SIMULATION with the estimated components, mirroring the real design exactly (same participants,
   recordings, window counts and window times), evaluated with the cohort's code (eeg_analysis.run_schemes,
   same random forest). The question: does the fingerprint model reproduce the observed accuracies?

3. ABLATIONS of the calibrated model: no recording offsets; no slow drift (residual variance kept);
   no task effect (spurious accuracy only); no participant offsets.

4. CALIBRATION CHECK: components re-estimated on the simulated data vs the real data.

Datasets: the cohort (QC-cleaned, three-class) and EEGMMIDB block design (eyes open vs eyes closed, from
public_replication.py caches).

    python calibrated_simulation.py                    # both datasets, full run
    python calibrated_simulation.py --quick            # 100 trees, smaller check
    python calibrated_simulation.py --datasets cohort  # one dataset
Output: <EEG_OUTPUT_DIR>/calibrated_simulation/summary.md, results.csv, variance_components.csv
"""
import os, json, glob, time, argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import eeg_analysis as EA

OUT = os.path.join(EA.OUT_DIR_BASE, "calibrated_simulation")
MAX_LAG = 30            # windows used to fit the slow-drift autocorrelation
OVERLAP_R1 = 0.5        # lag-1 correlation of the fast component (2-s windows, 1-s step)


# ------------------------------ data ------------------------------
def load_cohort(res):
    man = json.load(open(os.path.join(res, "cache_manifest.json")))
    recs = EA.load_all([os.path.join(res, "cache", f) for f in man["files"]])
    recs = [EA.reorder_record(r, man["chans"]) for r in recs]
    EA.channel_qc(recs)
    D = EA.build_clean(recs)
    D["rec"] = np.array([f"{g}|{y}" for g, y in zip(D["g"], D["y"])])
    return D


def load_eegmmidb(cache_dir):
    files = sorted(glob.glob(os.path.join(cache_dir, "pub_S*.npz")))
    if not files:
        raise FileNotFoundError(f"no public_replication caches in {cache_dir} (run public_replication.py first)")
    X, y, g, ts, dur = [], [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        m = z["design"] == "A"
        X.append(z["feats"][m]); y += list(z["label"][m]); g += [str(z["subject"])] * int(m.sum())
        ts.append(z["onset"][m]); dur.append(z["dur"][m])
    D = dict(X=np.vstack(X).astype(np.float64), y=np.array(y), g=np.array(g), ts=np.concatenate(ts), dur=np.concatenate(dur))
    D["rec"] = np.array([f"{a}|{b}" for a, b in zip(D["g"], D["y"])])
    return D


def order(D):
    o = np.lexsort((D["ts"], D["rec"]))
    return {k: (v[o] if isinstance(v, np.ndarray) and len(v) == len(o) else v) for k, v in D.items()}


# ------------------------------ estimation ------------------------------
def psd(M):
    w, V = np.linalg.eigh((M + M.T) / 2)
    return (V * np.clip(w, 0, None)) @ V.T


def msqrt(M):
    w, V = np.linalg.eigh(psd(M))
    return V * np.sqrt(np.clip(w, 0, None))


def demean_fraction(rho, n):
    """Fraction of an AR(1) variance that lies in the mean of n consecutive samples (removed by demeaning)."""
    if rho <= 0:
        return 1.0 / n
    k = np.arange(1, n)
    return float((1 + 2 * np.sum((1 - k / n) * rho ** k)) / n)


def acf_fit(w, rec, nbar):
    """Slow-drift and fast-noise variances (as fractions of the demeaned residual variance) and rho.

    The ACF of a DEMEANED AR(1) series is approximately s_d * (rho^k - f_d) for lags k >= 2, where f_d is the
    fraction of the drift variance removed by subtracting each recording mean; fitting this form avoids the
    downward bias of fitting rho^k to the demeaned data."""
    var0 = np.mean(w ** 2, axis=0)
    lags = np.arange(2, max(4, min(MAX_LAG, int(nbar // 3))) + 1)   # lags with enough pairs in every recording
    r, used = [], []
    for k in lags:
        num, den = 0.0, 0
        for rr in np.unique(rec):
            idx = np.where(rec == rr)[0]
            if len(idx) > k:
                num += np.sum(w[idx[:-k]] * w[idx[k:]], axis=0); den += len(idx) - k
        if den > 0:
            r.append(np.mean(num / den / var0)); used.append(k)
    r, lags = np.array(r), np.array(used)
    best = (np.inf, 0.0, 0.0, 1.0 / nbar)
    for rho in np.concatenate([np.linspace(0.30, 0.99, 70), np.linspace(0.991, 0.999, 9)]):
        f_d = demean_fraction(rho, int(round(nbar)))
        b = rho ** lags - f_d
        s_d = max(float(np.dot(b, r) / np.dot(b, b)), 0.0)
        sse = float(np.sum((r - s_d * b) ** 2))
        if sse < best[0]:
            best = (sse, rho, s_d, f_d)
    _, rho, s_d, f_d = best
    f_e = (1 + 2 * OVERLAP_R1) / nbar
    s_e = max((1 - s_d * (1 - f_d)) / (1 - f_e), 1e-6)       # observed variance = s_d(1-f_d) + s_e(1-f_e) = 1
    return dict(rho=rho, s_d=s_d, s_e=s_e, f_d=f_d, f_e=f_e, acf=r)


def estimate(D):
    X, y, g, rec = D["X"], D["y"], D["g"], D["rec"]
    conds = sorted(np.unique(y)); C = len(conds)
    delta = {c: X[y == c].mean(0) for c in conds}
    z = X - np.stack([delta[c] for c in y])
    recs = np.unique(rec)
    m = {r: z[rec == r].mean(0) for r in recs}
    n = {r: int((rec == r).sum()) for r in recs}
    w = z - np.stack([m[r] for r in rec])
    Sw_obs = np.cov(w.T)                                     # demeaned within-recording residual
    nbar = np.mean(list(n.values()))
    fit = acf_fit(w, rec, nbar)
    rho = fit["rho"]
    Sw = Sw_obs * (fit["s_d"] + fit["s_e"])                  # true within-recording covariance (demeaning undone)
    A = fit["s_d"] / (fit["s_d"] + fit["s_e"])               # slow-drift share of it
    Vmean = Sw_obs * (fit["s_d"] * fit["f_d"] + fit["s_e"] * fit["f_e"])   # noise contained in each recording mean
    acf = fit["acf"]
    subs = np.unique(g)
    u = {s: np.mean([m[r] for r in recs if r.split("|")[0] == s], axis=0) for s in subs}
    dev = np.stack([m[r] - u[r.split("|")[0]] for r in recs])
    k_rec = np.mean([sum(1 for r in recs if r.split("|")[0] == s) for s in subs])  # recordings per participant
    Sr = psd(np.cov(dev.T) * k_rec / max(k_rec - 1, 1) - Vmean)
    Ss = psd(np.cov(np.stack(list(u.values())).T) - (Sr + Vmean) / k_rec)
    tot = np.trace(Ss) + np.trace(Sr) + np.trace(Sw)
    shares = dict(participant=np.trace(Ss) / tot, recording=np.trace(Sr) / tot,
                  slow_drift=A * np.trace(Sw) / tot, fast_noise=(1 - A) * np.trace(Sw) / tot)
    shares["recording_specific"] = shares["recording"] + shares["slow_drift"]   # identifiable even in short recordings
    task = np.mean([np.sum((delta[c] - np.mean(list(delta.values()), 0)) ** 2) for c in conds]) / tot
    return dict(conds=conds, delta=delta, Ss=Ss, Sr=Sr, Sw=Sw, A=A, rho=rho, shares=shares,
                task_to_total=task, acf=acf, windows_per_recording=float(nbar))


# ------------------------------ simulation ------------------------------
def simulate(D, P, seed, recording=True, drift=True, task=True, participant=True,
             rec_scale=1.0, drift_scale=1.0, rho=None, overlap=0.5):
    rng = np.random.RandomState(seed)
    p = D["X"].shape[1]
    Ls, Lr, Lw = msqrt(P["Ss"]), msqrt(P["Sr"] * rec_scale), msqrt(P["Sw"])
    A = min(P["A"] * drift_scale, 1.0) if drift else 0.0
    rho = P["rho"] if rho is None else rho
    mu0 = np.mean(list(P["delta"].values()), axis=0)
    Xs = np.empty_like(D["X"])
    us = {s: Ls @ rng.randn(p) for s in np.unique(D["g"])}
    for r in np.unique(D["rec"]):
        idx = np.where(D["rec"] == r)[0]
        s, c = r.split("|")[0], D["y"][idx[0]]
        nw = len(idx)
        base = (P["delta"][c] if task else mu0) + (us[s] if participant else 0) + (Lr @ rng.randn(p) if recording else 0)
        d = np.zeros((nw, p))
        if A > 0:
            e0 = rng.randn(nw, p)
            d[0] = e0[0]
            for t in range(1, nw):
                d[t] = rho * d[t - 1] + np.sqrt(1 - rho ** 2) * e0[t]
        half = rng.randn(nw + 1, p)                       # MA(1) noise, lag-1 correlation `overlap` (<= 0.5)
        th = (1 - np.sqrt(max(1 - 4 * overlap ** 2, 0))) / (2 * overlap) if overlap > 0 else 0.0
        e = (half[1:] + th * half[:-1]) / np.sqrt(1 + th ** 2)
        Xs[idx] = base + (np.sqrt(A) * d + np.sqrt(1 - A) * e) @ Lw.T
    return Xs


# ------------------------------ evaluation ------------------------------
def evaluate(D, X, mk, label):
    out = EA.run_schemes(X, D["y"], D["g"], D["ts"], D["dur"], mk)
    row = dict(model=label)
    acc = {}
    for k, (yt, yp, gs) in out.items():
        ps = EA.per_subject_acc(yt, yp, gs); acc[k] = ps
        lo, hi = EA.bootstrap_ci(ps.values)
        row.update({f"{EA.SCHEME_NAME[k]}": ps.mean(), f"{EA.SCHEME_NAME[k]} lo": lo, f"{EA.SCHEME_NAME[k]} hi": hi})
    row["random - cross"] = (acc["a"] - acc["c"]).dropna().mean()
    return row


def run_dataset(name, D, a, mk):
    D = order(D)
    if a.sens_only:
        P = estimate(D)
        print(f"\n=== {name}: sensitivity only (rho {P['rho']:.3f}, A {P['A']:.3f})")
        return [], [], sensitivity(name, D, P, a)
    print(f"\n=== {name}: {len(D['y'])} windows, {len(np.unique(D['g']))} participants, "
          f"{len(np.unique(D['rec']))} recordings, classes {sorted(np.unique(D['y']))}")
    P = estimate(D)
    sh = P["shares"]
    print(f"  variance shares: participant {sh['participant']:.3f}  recording {sh['recording']:.3f}  "
          f"slow drift {sh['slow_drift']:.3f}  fast noise {sh['fast_noise']:.3f}  (rho {P['rho']:.3f}, A {P['A']:.3f})")
    print(f"  recording-specific (offset + drift) {sh['recording_specific']:.3f}; "
          f"{P['windows_per_recording']:.0f} windows per recording")
    if P["windows_per_recording"] < 150:
        print("  [note] recordings are short: offset vs slow drift is not separately identifiable; "
              "interpret their sum (recording_specific)")
    comp = [dict(dataset=name, source="real", **sh, rho=P["rho"], A=P["A"], task_to_total=P["task_to_total"],
                 windows_per_recording=P["windows_per_recording"])]
    rows = []
    t0 = time.time(); r = evaluate(D, D["X"], mk, "real data"); rows.append(r)
    print(f"  real data         random {r['(a) random split']:.3f}  time {r['(b) within-recording time split']:.3f}  "
          f"cross {r['(c) cross-subject']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    variants = [("calibrated model", {}),
                ("- recording offsets", dict(recording=False)),
                ("- slow drift", dict(drift=False)),
                ("- recording-specific (offsets + drift)", dict(recording=False, drift=False)),
                ("- participant offsets", dict(participant=False)),
                ("- task effect (spurious only)", dict(task=False))]
    for label, kw in variants:
        reps = a.reps if label == "calibrated model" else 1
        for rep in range(reps):
            t0 = time.time()
            Xs = simulate(D, P, seed=a.seed + rep, **kw)
            r = evaluate(D, Xs, mk, label + (f" (rep {rep})" if reps > 1 else "")); rows.append(r)
            print(f"  {r['model']:30s} random {r['(a) random split']:.3f}  time {r['(b) within-recording time split']:.3f}  "
                  f"cross {r['(c) cross-subject']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
            if label == "calibrated model" and rep == 0:
                Ds = dict(D); Ds["X"] = Xs
                Q = estimate(Ds)
                comp.append(dict(dataset=name, source="simulated (calibration check)", **Q["shares"], rho=Q["rho"],
                                 A=Q["A"], task_to_total=Q["task_to_total"],
                                 windows_per_recording=Q["windows_per_recording"]))
    for r in rows:
        r["dataset"] = name
    sens = sensitivity(name, D, P, a) if a.sensitivity else []
    return rows, comp, sens


def bootstrap_params(D, seed):
    """Re-estimate the components on a participant bootstrap sample (participants renamed to stay unique)."""
    rng = np.random.RandomState(seed)
    subs = np.unique(D["g"]); pick = rng.choice(subs, len(subs), replace=True)
    parts = []
    for k, s_ in enumerate(pick):
        m = D["g"] == s_
        part = {key: D[key][m] for key in ("X", "y", "ts", "dur")}
        part["g"] = np.array([f"b{k}"] * int(m.sum())); part["rec"] = np.array([f"b{k}|{c}" for c in part["y"]])
        parts.append(part)
    B = {key: np.concatenate([q[key] for q in parts]) for key in parts[0]}
    return estimate(order(B))


def gap_ac(D, X, mk):
    """Random-split minus cross-subject gap (per-participant means); only the two schemes that define the gap."""
    out = EA.run_schemes(X, D["y"], D["g"], D["ts"], D["dur"], mk, schemes=("a", "c"))
    pa = EA.per_subject_acc(*out["a"]); pc = EA.per_subject_acc(*out["c"])
    return float(pa.mean()), float(pc.mean()), float((pa - pc).dropna().mean())


def sensitivity(name, D, P, a):
    """Uncertainty of the reproduction rate (model gap / real-data gap), all with the same forest size:
       (1) participant bootstrap: participants resampled, components re-estimated, data simulated with the real design;
       (2) one-at-a-time perturbations of the model assumptions;
       (3) optional 2D grid over the recording covariance and the drift AR coefficient."""
    mk = lambda: RandomForestClassifier(a.sens_trees, min_samples_leaf=2, n_jobs=-1, random_state=EA.RANDOM_STATE)
    ra, rc, real_gap = gap_ac(D, D["X"], mk)
    print(f"  [sens] real data ({a.sens_trees} trees): random {ra:.3f}  cross {rc:.3f}  gap {real_gap:.3f}", flush=True)
    rows = [dict(dataset=name, analysis="reference", setting="real data", random=ra, cross=rc, gap=real_gap, reproduction=1.0)]
    def run(analysis, setting, Q, **kw):
        t0 = time.time()
        r_, c_, g_ = gap_ac(D, simulate(D, Q, seed=a.seed, **kw), mk)
        rows.append(dict(dataset=name, analysis=analysis, setting=setting, random=r_, cross=c_, gap=g_, reproduction=g_ / real_gap))
        print(f"  [sens] {analysis:12s} {setting:28s} gap {g_:.3f}  reproduction {100*g_/real_gap:.0f}%  [{time.time()-t0:.0f}s]",
              flush=True)
    run("model", "calibrated (point estimate)", P)
    for b in range(a.boot):
        run("bootstrap", f"replicate {b + 1}", bootstrap_params(D, 1000 + b))
    for setting, kw in (("recording covariance x0.5", dict(rec_scale=0.5)), ("recording covariance x1.5", dict(rec_scale=1.5)),
                        ("drift share x0.5", dict(drift_scale=0.5)), ("drift share x1.5", dict(drift_scale=1.5)),
                        ("drift rho = 0.90", dict(rho=0.90)), ("drift rho = 0.99", dict(rho=0.99)),
                        ("overlap correlation 0.25", dict(overlap=0.25)), ("overlap correlation 0.40", dict(overlap=0.40))):
        run("perturbation", setting, P, **kw)
    if a.grid:
        for rs in (0.5, 1.0, 1.5):
            for rh in (0.90, P["rho"], 0.99):
                run("grid", f"rec x{rs}, rho {rh:.2f}", P, rec_scale=rs, rho=rh)
    return rows


def write_sensitivity(S, a):
    S.to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
    L = ["# Sensitivity of the reproduction rate (calibrated simulation)", "",
         f"- Random forest {a.sens_trees} trees throughout (reference gap recomputed with the same forest); "
         f"{a.boot} participant-bootstrap replicates; seed {a.seed}",
         "- reproduction = model gap / real-data gap, gap = random-split minus cross-subject per-participant accuracy",
         "- The bootstrap interval (parameter uncertainty) and the perturbation range (assumption sensitivity) answer "
         "different questions and are reported separately.", ""]
    for ds, g in S.groupby("dataset", sort=False):
        ref = g[g.analysis == "reference"].iloc[0]; pt = g[g.analysis == "model"].iloc[0]
        bt = g[g.analysis == "bootstrap"]["reproduction"]; pe = g[g.analysis == "perturbation"]
        L += [f"## {ds}", "", f"- Real-data gap: {ref.gap:.3f}; calibrated point estimate: gap {pt.gap:.3f} "
              f"(reproduction {100*pt.reproduction:.0f}%)"]
        if len(bt):
            L.append(f"- Participant bootstrap (n = {len(bt)}): median {100*bt.median():.0f}%, "
                     f"95% percentile interval {100*bt.quantile(0.025):.0f}-{100*bt.quantile(0.975):.0f}%")
        if len(pe):
            L.append(f"- Assumption perturbations: range {100*pe.reproduction.min():.0f}-{100*pe.reproduction.max():.0f}%")
            L += [f"    - {r.setting}: {100*r.reproduction:.0f}% (gap {r.gap:.3f})" for r in pe.itertuples()]
        gr = g[g.analysis == "grid"]
        if len(gr):
            L.append("- Grid (recording covariance x drift rho): " + "; ".join(f"{r.setting} {100*r.reproduction:.0f}%" for r in gr.itertuples()))
        L.append("")
    open(os.path.join(OUT, "sensitivity_summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"Saved -> {os.path.join(OUT, 'sensitivity_summary.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="cohort,eegmmidb")
    ap.add_argument("--res", default=os.path.join(EA.OUT_DIR_BASE, "policy_common"))
    ap.add_argument("--pub-cache", default=os.path.join(EA.OUT_DIR_BASE, "public_replication", "cache"))
    ap.add_argument("--trees", type=int, default=EA.N_TREES)
    ap.add_argument("--reps", type=int, default=3, help="simulated datasets for the full calibrated model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sensitivity", action="store_true", help="participant bootstrap + assumption perturbations")
    ap.add_argument("--sens-only", action="store_true", help="run only the sensitivity analysis (implies --sensitivity)")
    ap.add_argument("--boot", type=int, default=200, help="participant-bootstrap replicates (default 200)")
    ap.add_argument("--sens-trees", type=int, default=100, help="random-forest size used throughout the sensitivity analysis")
    ap.add_argument("--grid", action="store_true", help="add a 3 x 3 grid: recording covariance x drift AR coefficient")
    a = ap.parse_args()
    if a.quick:
        a.trees, a.reps = 100, 1
    os.makedirs(OUT, exist_ok=True)
    mk = lambda: RandomForestClassifier(a.trees, min_samples_leaf=2, n_jobs=-1, random_state=EA.RANDOM_STATE)
    if a.sens_only:
        a.sensitivity = True
    rows, comps, sens = [], [], []
    for name in [s.strip() for s in a.datasets.split(",") if s.strip()]:
        D = load_cohort(a.res) if name == "cohort" else load_eegmmidb(a.pub_cache)
        r, c, se = run_dataset(name, D, a, mk); rows += r; comps += c; sens += se
    if sens:
        write_sensitivity(pd.DataFrame(sens), a)
    if a.sens_only:
        return
    res, comp = pd.DataFrame(rows), pd.DataFrame(comps)
    res.to_csv(os.path.join(OUT, "results.csv"), index=False); comp.to_csv(os.path.join(OUT, "variance_components.csv"), index=False)
    cols = ["dataset", "model", "(a) random split", "(b) within-recording time split", "(c) cross-subject", "random - cross"]
    L = ["# Data-calibrated simulation of recording fingerprints", "",
         f"- Random forest {a.trees} trees; {a.reps} simulated dataset(s) for the calibrated model; seed {a.seed}",
         "- Variance shares are fractions of the total feature variance (trace of covariance); task_to_total = "
         "mean squared deviation of condition means / total variance",
         "- recording_specific = recording offset + slow drift; with short recordings (< 150 windows) only this sum is "
         "identifiable (verified by parameter-recovery tests)", "",
         "## Variance components (real vs simulated)", "", comp.round(4).to_string(index=False), "",
         "## Accuracy (per-participant mean)", "", res[cols].round(4).to_string(index=False), "",
         "## With bootstrap 95% CIs", "", res.round(4).to_string(index=False)]
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"\nSaved -> {os.path.join(OUT, 'summary.md')}")


if __name__ == "__main__":
    main()
