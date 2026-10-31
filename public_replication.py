# -*- coding: utf-8 -*-
"""
Public-dataset replication on the PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB).

Two designs with opposite class-recording structure are analysed with the cohort pipeline
(same channels, preprocessing, band-power features, random forest and evaluation schemes):

  A  BLOCK design        R01 eyes open vs R02 eyes closed - one recording per class
                         (class and recording coincide, as in the cohort)
  B  INTERLEAVED design  R04/R08/R12 imagined left (T1) vs right (T2) fist - both classes
                         alternate within every run (class and recording are dissociated)

For each design:
  (a) random window split, (b) within-recording time split (train before 60 % of each
  recording, test from 60 % + 2 s), (c) cross-subject (10 participant-disjoint folds);
  design B additionally (d) trial-grouped split within participants (all windows of a trial
  held out together). Label-free diagnostics: time-block pseudo-labels (thirds of each
  recording; random vs cross-subject) and participant identification (random vs time split).

    python public_replication.py --jobs 4 --download          # downloads missing EDF files (PhysioNet)
    python public_replication.py --jobs 4 --download --aws    # same, from the faster AWS S3 copy
    python public_replication.py --jobs 4 --subjects 1-109    # files already present
Output: <EEG_OUTPUT_DIR>/public_replication/summary.md, results.csv, per_subject.csv
"""
import os, json, time, argparse, warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
import eeg_analysis as EA
import external_validation as EV

warnings.filterwarnings("ignore")
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "C4", "T4",
            "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]          # the cohort's 18 primary-analysis channels
BLOCK_RUNS = {"eye_open": 1, "eye_close": 2}
MI_RUNS = [4, 8, 12]                                       # imagined left/right fist
MI_EVENTS = {"T1": "left", "T2": "right"}
OUT = os.path.join(EA.OUT_DIR_BASE, "public_replication")


# ------------------------------ features ------------------------------
def load_run(s, r, download=False):
    p = EV.edf_path(s, r)
    if not (os.path.exists(p) and os.path.getsize(p) > 1000):
        raise FileNotFoundError(p)
    sigs, fss, ann = EV.read_edf(p)
    lut = {EV.norm_name(k): k for k in sigs}
    names = [lut.get(EV.norm_name(EV.ALIASES.get(c, c))) for c in CHANNELS]
    if None in names:
        raise KeyError(f"S{s:03d}R{r:02d}: missing {[c for c, n in zip(CHANNELS, names) if n is None]}")
    fs = fss[names[0]]
    x = EV.preprocess_fs(np.stack([sigs[n] for n in names], axis=1), fs)
    return x, fs, ann


def windows(x, fs, a, b):
    """Band-power feature windows inside [a, b) seconds; returns features and window onsets."""
    W, S = int(round(EA.EPOCH_SEC * fs)), int(round(EA.STEP_SEC * fs))
    F, T = [], []
    for st in range(max(int(a * fs), 0), min(int(b * fs), len(x)) - W + 1, S):
        f_, p_ = EV.welch(x[st:st + W], fs, nperseg=int(round(fs)), axis=0)
        F.append(np.log([EA.band_power(f_, p_, lo, hi) + 1e-12 for lo, hi in EA.BANDS.values()]).ravel())
        T.append(st / fs)
    return F, T


def process_subject(args):
    s, cache_dir, mmidb_dir = args
    if mmidb_dir and EV.EEGMMIDB_DIR != mmidb_dir:      # worker processes start with module defaults
        EV.EEGMMIDB_DIR, EV._INDEX = mmidb_dir, None
    cache = os.path.join(cache_dir, f"pub_S{s:03d}.npz")
    if os.path.exists(cache):
        return cache
    rows = []                    # design, label, recording, trial, onset, duration, features
    for lab, r in BLOCK_RUNS.items():
        x, fs, _ = load_run(s, r)
        if abs(fs - 160) > 1e-6:
            raise ValueError(f"S{s:03d}R{r:02d}: fs = {fs}")
        T = len(x) / fs
        F, on = windows(x, fs, EA.TRIM_SEC, T - EA.TRIM_SEC)
        rows += [("A", lab, f"R{r:02d}", -1, t, T, f) for f, t in zip(F, on)]
    for r in MI_RUNS:
        x, fs, ann = load_run(s, r)
        T = len(x) / fs
        k = 0
        for onset, du, txt in ann:
            if txt in MI_EVENTS and du > 0:
                F, on = windows(x, fs, onset, onset + du)
                rows += [("B", MI_EVENTS[txt], f"R{r:02d}", k, t, T, f) for f, t in zip(F, on)]
                k += 1
    d, lab, rec, tr, on, du, F = zip(*rows)
    np.savez_compressed(cache, design=np.array(d), label=np.array(lab), rec=np.array(rec), trial=np.array(tr),
                        onset=np.array(on, float), dur=np.array(du, float), feats=np.array(F, np.float32),
                        subject=f"S{s:03d}")
    return cache


def load(caches):
    parts = [dict(np.load(c, allow_pickle=False)) for c in caches]
    D = {k: np.concatenate([p[k] for p in parts]) for k in ("design", "label", "rec", "trial", "onset", "dur", "feats")}
    D["g"] = np.concatenate([[str(p["subject"])] * len(p["label"]) for p in parts])
    return D


# ------------------------------ evaluation ------------------------------
def rf(trees):
    return lambda: RandomForestClassifier(trees, min_samples_leaf=2, n_jobs=-1, random_state=EA.RANDOM_STATE)


def evaluate(X, y, g, onset, dur, mk, trial_groups=None):
    """Out-of-fold predictions for each scheme (same rules as eeg_analysis.py)."""
    out = {}
    p = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=EA.RANDOM_STATE).split(X, y):
        p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    out["random split"] = p
    tr = (onset + EA.EPOCH_SEC) <= 0.6 * dur
    te = onset >= 0.6 * dur + EA.EPOCH_SEC
    p = np.full(len(y), None, dtype=object)
    if len(np.unique(y[tr])) > 1:
        p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    out["time split"] = p
    p = np.empty_like(y)
    for tr_, te_ in GroupKFold(10).split(X, y, g):
        p[te_] = mk().fit(X[tr_], y[tr_]).predict(X[te_])
    out["cross-subject"] = p
    if trial_groups is not None:                   # trial-grouped split within each participant
        p = np.empty_like(y)
        for s in np.unique(g):
            m = np.where(g == s)[0]
            gt = trial_groups[m]
            for tr_, te_ in GroupKFold(min(5, len(np.unique(gt)))).split(X[m], y[m], gt):
                p[m[te_]] = mk().fit(X[m[tr_]], y[m[tr_]]).predict(X[m[te_]])
        out["trial-grouped (within participant)"] = p
    return out


def summarise(design, task, preds, y, g, chance, rows, subj_rows):
    accs = {}
    for scheme, p in preds.items():
        m = np.array([v is not None for v in p])
        ps = EA.per_subject_acc(y[m], p[m].astype(y.dtype), g[m]); accs[scheme] = ps
        lo, hi = EA.bootstrap_ci(ps.values)
        rows.append(dict(design=design, task=task, scheme=scheme, n_participants=len(ps),
                         pooled_accuracy=float((p[m] == y[m]).mean()), subj_mean=ps.mean(), ci95_lo=lo, ci95_hi=hi,
                         chance=chance))
        subj_rows += [dict(design=design, task=task, scheme=scheme, subject=k, accuracy=v) for k, v in ps.items()]
    if "random split" in accs and "cross-subject" in accs:
        d = (accs["random split"] - accs["cross-subject"]).dropna()
        rng = np.random.RandomState(0)
        b = d.values[rng.randint(0, len(d), (10000, len(d)))].mean(1)
        rows.append(dict(design=design, task=task, scheme="random - cross-subject (paired)", n_participants=len(d),
                         pooled_accuracy=np.nan, subj_mean=d.mean(), ci95_lo=np.percentile(b, 2.5),
                         ci95_hi=np.percentile(b, 97.5), chance=np.nan, n_positive=int((d > 0).sum())))


def pseudo_blocks(onset, dur):
    """Time-block index (thirds of each recording); -1 for windows that straddle a boundary."""
    third = dur / 3
    k = np.floor(onset / third)
    k2 = np.floor((onset + EA.EPOCH_SEC - 1e-9) / third)
    return np.where(k == k2, k, -1).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-109")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--trees", type=int, default=EA.N_TREES)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--mmidb-dir", default=None)
    ap.add_argument("--aws", action="store_true", help="download from the PhysioNet open-data copy on AWS S3 (faster)")
    ap.add_argument("--threads", type=int, default=8, help="parallel downloads (default 8)")
    a = ap.parse_args()
    if a.aws:
        EV.BASE_URL = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0"
    EV.DOWNLOAD_THREADS = a.threads
    if a.mmidb_dir:
        EV.EEGMMIDB_DIR, EV._INDEX = a.mmidb_dir, None
    os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
    subs = EV.parse_subjects(a.subjects)
    runs = list(BLOCK_RUNS.values()) + MI_RUNS
    print(f"EEGMMIDB folder: {EV.EEGMMIDB_DIR}\nDownload source: {EV.BASE_URL} ({EV.DOWNLOAD_THREADS} parallel)")
    present = lambda s, r: os.path.exists(EV.edf_path(s, r)) and os.path.getsize(EV.edf_path(s, r)) > 1000
    need = [(s, r) for s in subs for r in runs if not present(s, r)]
    print(f"[0/3] EDF files: {len(subs) * len(runs) - len(need)} present, {len(need)} missing")
    if need and a.download:
        from concurrent.futures import ThreadPoolExecutor
        t0, mb, fails = time.time(), 0.0, []
        with ThreadPoolExecutor(EV.DOWNLOAD_THREADS) as ex:
            for i, (p_, n, e) in enumerate(ex.map(lambda sr: EV._download_one(*sr), need), 1):
                mb += n / 1e6; fails += [e] if e else []
                el = time.time() - t0
                print(f"\r  downloading {i}/{len(need)}  {mb:.0f} MB  {el:.0f}s  ({mb/max(el,1):.2f} MB/s)  failed {len(fails)}",
                      end="", flush=True)
                if i == 10 and len(fails) == 10:
                    raise SystemExit(f"\n[STOP] the first 10 downloads failed ({fails[0]}).\n"
                                     "Try the other source (with or without --aws) or download with MNE.")
        print()
        EV._INDEX = None
        if fails:
            print(f"  [WARN] {len(fails)} file(s) failed, e.g. {fails[0]}")
    elif need:
        print("  missing files will be skipped (add --download to fetch them)")
    print(f"[1/3] Features for {len(subs)} participants (cache: {os.path.join(OUT, 'cache')})")
    caches, skipped = [], []
    jobs = [(s, os.path.join(OUT, "cache"), EV.EEGMMIDB_DIR) for s in subs]
    t0 = time.time()
    def report(i):
        print(f"\r  {i}/{len(subs)} participants  used {len(caches)}  skipped {len(skipped)}  {time.time()-t0:.0f}s",
              end="", flush=True)
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(a.jobs) as ex:
            futs = {ex.submit(process_subject, j): j[0] for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                try: caches.append(fut.result())
                except Exception as e: skipped.append((futs[fut], str(e)[:120]))
                report(i)
    else:
        for i, j in enumerate(jobs, 1):
            try: caches.append(process_subject(j))
            except Exception as e: skipped.append((j[0], str(e)[:120]))
            report(i)
    print(f"\n  used {len(caches)}, skipped {len(skipped)}")
    caches.sort()
    D = load(caches)
    mk = rf(a.trees)
    rows, subj_rows = [], []
    print("[2/3] Evaluation")
    for design, task in (("A", "eyes open vs eyes closed (block)"), ("B", "imagined left vs right fist (interleaved)")):
        k = D["design"] == design
        X, y, g, on, du = D["feats"][k], D["label"][k], D["g"][k], D["onset"][k], D["dur"][k]
        rec = D["rec"][k]; trial = D["trial"][k]
        tg = np.char.add(np.char.add(rec.astype(str), "_"), trial.astype(str)) if design == "B" else None
        t0 = time.time()
        preds = evaluate(X, y, g, on, du, mk, trial_groups=tg)
        summarise(design, task, preds, y, g, 0.5, rows, subj_rows)
        print(f"  {task}: done [{time.time()-t0:.0f}s]")
        # label-free diagnostics within a single class / recording type
        for lab in np.unique(y):
            m = y == lab
            blk = pseudo_blocks(on[m], du[m]); keep = blk >= 0
            Xb, yb, gb = X[m][keep], blk[keep].astype(str), g[m][keep]
            pb = {}
            p = np.empty_like(yb)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=EA.RANDOM_STATE).split(Xb, yb):
                p[te] = mk().fit(Xb[tr], yb[tr]).predict(Xb[te])
            pb["random split"] = p
            p = np.empty_like(yb)
            for tr, te in GroupKFold(10).split(Xb, yb, gb):
                p[te] = mk().fit(Xb[tr], yb[tr]).predict(Xb[te])
            pb["cross-subject"] = p
            summarise(design, f"time-block pseudo-labels ({lab})", pb, yb, gb, 1 / 3, rows, subj_rows)
            gm, onm, dum = g[m], on[m], du[m]
            pid = {}
            p = np.empty_like(gm)
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=EA.RANDOM_STATE).split(X[m], gm):
                p[te] = mk().fit(X[m][tr], gm[tr]).predict(X[m][te])
            pid["random split"] = p
            tr = (onm + EA.EPOCH_SEC) <= 0.6 * dum; te = onm >= 0.6 * dum + EA.EPOCH_SEC
            p = np.full(len(gm), None, dtype=object); p[te] = mk().fit(X[m][tr], gm[tr]).predict(X[m][te])
            pid["time split"] = p
            summarise(design, f"participant identification ({lab})", pid, gm, gm, 1 / len(np.unique(gm)),
                      rows, subj_rows)
    print("[3/3] Writing outputs")
    res = pd.DataFrame(rows); sub = pd.DataFrame(subj_rows)
    res.to_csv(os.path.join(OUT, "results.csv"), index=False); sub.to_csv(os.path.join(OUT, "per_subject.csv"), index=False)
    L = ["# Public-dataset replication (EEGMMIDB)", "",
         f"- Participants used: {len(np.unique(D['g']))}; skipped: {len(skipped)}",
         f"- Channels: {len(CHANNELS)} (cohort set); {a.trees} trees; seed {EA.RANDOM_STATE}", "",
         res.round(4).to_string(index=False), "", "## Skipped participants", ""]
    L += [f"- S{s:03d}: {e}" for s, e in skipped] or ["(none)"]
    open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"Saved -> {os.path.join(OUT, 'summary.md')}")


if __name__ == "__main__":
    main()
