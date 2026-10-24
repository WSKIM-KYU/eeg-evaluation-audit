# -*- coding: utf-8 -*-
"""
External validation with the PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)

Place this file in the same folder as eeg_analysis.py and run the main analysis first
(it creates the per-recording cache that this module reuses).

    python eeg_analysis.py --jobs 4              # 1) own 50-subject cohort
    python external_validation.py --jobs 4       # 2) download/process EEGMMIDB + external validation
    (or in one go: python eeg_analysis.py --jobs 4 --external)

Condition mapping
    eye_open  <- run R01 (baseline, eyes open, 1 min)
    eye_close <- run R02 (baseline, eyes closed, 1 min)
    hands_up  <- runs R05/R09/R13, event T1 (real movement, both fists)
                 closest available bilateral upper-limb task; note that fist clenching
                 produces much less head/body motion than raising both arms.

Evaluations (same preprocessing and 2-s band-power features as eeg_analysis.py, EEG only)
    Cohort -> EEGMMIDB      train on all cohort subjects, test on EEGMMIDB
    EEGMMIDB -> Cohort      reverse direction
    Cohort CV / EEGMMIDB CV cross-subject reference within each dataset
for 2-class (eye_open vs eye_close) and 3-class tasks, two feature normalisations
(relative log band power; + subject-wise standardisation) and LR / RF models.

No extra packages are needed: EDF+ files are read by a built-in reader.
"""
import os, re, glob, time, argparse, urllib.request, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch
from scipy.stats import wilcoxon

import eeg_analysis as EA   # shares OUT_DIR, BANDS, EPOCH_SEC, STEP_SEC, TRIM_SEC, ARTIFACT_PTP_UV ...

warnings.filterwarnings("ignore")

# =============================== CONFIG ===============================
EEGMMIDB_DIR = os.environ.get("EEGMMIDB_DIR", os.path.join("data", "eegmmidb"))     # EDF files: <dir>\S001\S001R01.edf ...
BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"
SUBJECTS = list(range(1, 110))
EXCLUDE = [88, 92, 100, 104]     # commonly excluded: irregular sampling rate / annotations
RUNS = {"eye_open": [1], "eye_close": [2], "hands_up": [5, 9, 13]}
HANDS_EVENT = "T1"               # runs 5/9/13: T1 = both fists, T2 = both feet
LINE_FREQ_MMIDB = 60             # recorded in the USA
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}   # 10-20 old names -> 10-10 names
# ======================================================================


# ----------------------------- EDF+ reader -----------------------------
def read_edf(path):
    """Minimal EDF/EDF+ reader. Returns {label: signal}, {label: fs}, [(onset, dur, text)]."""
    with open(path, "rb") as f:
        h = f.read(256)
        nrec, dur, ns = int(h[236:244]), float(h[244:252]), int(h[252:256])
        sh = f.read(256 * ns)
        pos = 0

        def field(width, conv=str):
            nonlocal pos
            out = [conv(sh[pos + i * width: pos + (i + 1) * width].decode("latin-1").strip()) for i in range(ns)]
            pos += width * ns
            return out
        labels = field(16); field(80); units = field(8)
        pmin = field(8, float); pmax = field(8, float); dmin = field(8, float); dmax = field(8, float)
        field(80); nsamp = field(8, int); field(32)
        raw = np.fromfile(f, dtype="<i2")
    spr = sum(nsamp)
    if nrec <= 0:
        nrec = len(raw) // spr
    rec = raw[:nrec * spr].reshape(nrec, spr)
    sigs, fss, ann, p = {}, {}, [], 0
    for i, lab in enumerate(labels):
        blk = rec[:, p:p + nsamp[i]]; p += nsamp[i]
        if lab.startswith("EDF Annotations"):
            txt = blk.astype("<i2").tobytes().decode("latin-1")
            for tal in txt.split("\x00"):
                if not tal.strip("\x14"):
                    continue
                parts = tal.split("\x14")
                on, _, du = parts[0].partition("\x15")
                for t in parts[1:]:
                    if t.strip():
                        ann.append((float(on), float(du) if du else 0.0, t.strip()))
            continue
        gain = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i])
        x = (blk.ravel().astype(np.float64) - dmin[i]) * gain + pmin[i]
        u = units[i].lower()
        x *= 1e6 if u == "v" else 1e3 if u == "mv" else 1.0     # -> microvolts
        sigs[lab] = x; fss[lab] = nsamp[i] / dur
    return sigs, fss, ann


def norm_name(s):
    return s.replace(".", "").replace("EEG", "").strip().upper()


# ------------------------------ download ------------------------------
DOWNLOAD_TIMEOUT = 60     # seconds without data before a download attempt is abandoned
DOWNLOAD_THREADS = 4
_INDEX = None


def _file_index():
    """Find EDF files anywhere under EEGMMIDB_DIR (also works with the extracted PhysioNet ZIP layout)."""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        for root, _, files in os.walk(EEGMMIDB_DIR):
            for f in files:
                if f.lower().endswith(".edf"):
                    _INDEX[f.upper()] = os.path.join(root, f)
    return _INDEX


def edf_path(s, r):
    name = f"S{s:03d}R{r:02d}.edf"
    return _file_index().get(name.upper(), os.path.join(EEGMMIDB_DIR, f"S{s:03d}", name))


def _download_one(s, r):
    import socket
    p = os.path.join(EEGMMIDB_DIR, f"S{s:03d}", f"S{s:03d}R{r:02d}.edf")
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return p, 0, None
    os.makedirs(os.path.dirname(p), exist_ok=True)
    url = f"{BASE_URL}/S{s:03d}/S{s:03d}R{r:02d}.edf"
    err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as resp, open(p + ".part", "wb") as out:
                n = 0
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    out.write(chunk); n += len(chunk)
            os.replace(p + ".part", p)
            return p, n, None
        except Exception as e:
            err = e; time.sleep(3)
    return p, 0, f"{url}: {err}"


def download_all(subs):
    """Download missing EDF files with progress, timeouts and retries (before feature extraction)."""
    from concurrent.futures import ThreadPoolExecutor
    need = [(s, r) for s in subs for runs in RUNS.values() for r in runs
            if not (os.path.exists(edf_path(s, r)) and os.path.getsize(edf_path(s, r)) > 1000)]
    if not need:
        print("  all EEGMMIDB files already present"); return []
    print(f"  downloading {len(need)} files from PhysioNet ({DOWNLOAD_THREADS} parallel, timeout {DOWNLOAD_TIMEOUT}s)")
    t0, mb, fails, stopped = time.time(), 0.0, [], False
    with ThreadPoolExecutor(DOWNLOAD_THREADS) as ex:
        futs = [ex.submit(_download_one, s, r) for s, r in need]
        for i, fu in enumerate(futs, 1):
            p, n, e = fu.result()
            mb += n / 1e6
            if e:
                fails.append(e)
            el = time.time() - t0
            print(f"\r  {i}/{len(need)} files  {mb:.0f} MB  {el:.0f}s  ({mb/max(el,1):.2f} MB/s)  failed: {len(fails)}",
                  end="")
            if i == 10 and len(fails) == 10:
                print("\n  [STOP] the first 10 downloads all failed -> network/firewall problem")
                ex.shutdown(wait=False, cancel_futures=True)
                stopped = True
                break
    print()
    global _INDEX
    _INDEX = None
    if fails:
        print(f"  [WARN] {len(fails)} file(s) could not be downloaded, e.g. {fails[0]}")
        print("  Manual alternative: download the ZIP from https://physionet.org/content/eegmmidb/1.0.0/ ,")
        print(f"  extract it anywhere under {EEGMMIDB_DIR} and run: python external_validation.py --no-download")
    if stopped:
        raise SystemExit("External validation stopped: EEGMMIDB could not be downloaded.")
    return fails


def ensure_file(s, r, download=False):
    p = edf_path(s, r)
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return p
    raise FileNotFoundError(p)


# --------------------------- feature extraction ---------------------------
def preprocess_fs(x, fs):
    x = x - x.mean(0)
    for f0 in (LINE_FREQ_MMIDB, 2 * LINE_FREQ_MMIDB):
        if f0 < fs / 2 - 1:
            b, a = iirnotch(f0 / (fs / 2), 30); x = filtfilt(b, a, x, axis=0)
    sos = butter(4, [EA.BAND[0] / (fs / 2), min(EA.BAND[1], fs / 2 - 2) / (fs / 2)], "bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def epoch_features(x, fs, segments):
    W, S = int(round(EA.EPOCH_SEC * fs)), int(round(EA.STEP_SEC * fs))
    feats, ptp = [], []
    for a, b in segments:
        a, b = int(a * fs), int(b * fs)
        for s in range(max(a, 0), min(b, len(x)) - W + 1, S):
            w = x[s:s + W]
            fe, pe = welch(w, fs, nperseg=int(round(fs)), axis=0)
            feats.append(np.log([EA.band_power(fe, pe, lo, hi) + 1e-12 for lo, hi in EA.BANDS.values()]).ravel())
            ptp.append(np.ptp(w, axis=0).max())
    return feats, ptp


def process_subject(args):
    s, chans, cache_dir, download, mmidb_dir = args
    global EEGMMIDB_DIR, _INDEX
    if mmidb_dir and EEGMMIDB_DIR != mmidb_dir:   # worker processes (Windows "spawn") start with defaults
        EEGMMIDB_DIR, _INDEX = mmidb_dir, None
    import zlib
    def present(r):
        p = edf_path(s, r)
        return os.path.exists(p) and os.path.getsize(p) > 1000
    # eye_open / eye_close runs are required; for hands_up any available run among 5/9/13 is used
    for cond in ("eye_open", "eye_close"):
        for r in RUNS[cond]:
            if not present(r):
                raise FileNotFoundError(edf_path(s, r))
    hands_runs = [r for r in RUNS["hands_up"] if present(r)]
    if not hands_runs:
        raise FileNotFoundError(f"S{s:03d}: none of the hands_up runs {RUNS['hands_up']} found")
    tag = (f"{len(chans)}ch_{zlib.crc32(','.join(chans).encode()):08x}"      # cache tied to channel set
           f"_hu{''.join(str(r) for r in hands_runs)}")                        # and to the runs used
    cache = os.path.join(cache_dir, f"mmidb_S{s:03d}_{tag}.npz")
    if os.path.exists(cache):
        return cache
    F, P, C, info = [], [], [], {}
    use = {"eye_open": RUNS["eye_open"], "eye_close": RUNS["eye_close"], "hands_up": hands_runs}
    for cond, runs in use.items():
        for r in runs:
            sigs, fss, ann = read_edf(ensure_file(s, r, download))
            lut = {norm_name(k): k for k in sigs}
            names = [lut.get(norm_name(ALIASES.get(c, c))) for c in chans]
            if None in names:
                miss = [c for c, n in zip(chans, names) if n is None]
                raise KeyError(f"S{s:03d}R{r:02d}: channels not found {miss}")
            fs = fss[names[0]]; info[f"fs_R{r:02d}"] = fs
            x = preprocess_fs(np.stack([sigs[n] for n in names], axis=1), fs)
            T = len(x) / fs
            if cond == "hands_up":
                segs = [(on, on + du) for on, du, t in ann if t == HANDS_EVENT and du > 0]
            else:
                segs = [(EA.TRIM_SEC, T - EA.TRIM_SEC)]
            f_, p_ = epoch_features(x, fs, segs)
            F += f_; P += p_; C += [cond] * len(f_)
    np.savez_compressed(cache, feats=np.array(F, np.float32), ptp=np.array(P, np.float32),
                        cond=np.array(C), subject=f"MMIDB_{s:03d}", fs=min(info.values()))
    return cache


# ------------------------------ datasets ------------------------------
def load_cohort(cache_dir):
    man = os.path.join(os.path.dirname(cache_dir), "cache_manifest.json")
    target = None
    if os.path.exists(man):      # caches and channel set chosen by the latest eeg_analysis.py run
        import json
        m = json.load(open(man))
        names = m["files"] if isinstance(m, dict) else m
        target = m.get("chans") if isinstance(m, dict) else None
        files = [os.path.join(cache_dir, n) for n in names if os.path.exists(os.path.join(cache_dir, n))]
    else:
        files = [f for f in glob.glob(os.path.join(cache_dir, "*.npz")) if not os.path.basename(f).startswith("mmidb_")]
    if not files:
        raise SystemExit(f"No cohort cache in {cache_dir}. Run eeg_analysis.py first.")
    X, y, g, ptp = [], [], [], []
    for f in files:
        z = np.load(f)
        r = {k: z[k] for k in ("feats", "psd", "samples", "chans")}
        if target is None:
            target = [str(c) for c in r["chans"]]
        r = EA.reorder_record(r, target)
        X.append(r["feats"]); n = len(r["feats"])
        y += [str(z["cond"])] * n; g += [f"COH_{z['subject']}"] * n; ptp.append(z["ptp"])
    return dict(X=np.vstack(X), y=np.array(y), g=np.array(g), ptp=np.concatenate(ptp)), list(target)


def load_mmidb(caches):
    X, y, g, ptp = [], [], [], []
    for c in caches:
        z = np.load(c)
        X.append(z["feats"]); y += list(z["cond"]); g += [str(z["subject"])] * len(z["feats"]); ptp.append(z["ptp"])
    return dict(X=np.vstack(X), y=np.array(y), g=np.array(g), ptp=np.concatenate(ptp))


def transform(D, mode, nch):
    F = D["X"].reshape(len(D["X"]), len(EA.BANDS), nch).astype(np.float64)
    m = F.max(1, keepdims=True)
    rel = F - (m + np.log(np.exp(F - m).sum(1, keepdims=True)))     # log relative band power
    X = rel.reshape(len(F), -1)
    if mode == "relative + subject z-score":
        X = X.copy()
        for s in np.unique(D["g"]):
            k = D["g"] == s
            X[k] = (X[k] - X[k].mean(0)) / (X[k].std(0) + 1e-6)
    return X


def subset(D, X, classes, clean):
    k = np.isin(D["y"], classes)
    if clean:
        k &= D["ptp"] <= EA.ARTIFACT_PTP_UV
    return X[k], D["y"][k], D["g"][k]


# ----------------------------- evaluation -----------------------------
def models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {"LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")),
            "RF": lambda: RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1,
                                                 random_state=EA.RANDOM_STATE)}


def metrics(yt, yp, g, classes):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
    ps = [balanced_accuracy_score(yt[g == s], yp[g == s]) for s in np.unique(g) if len(np.unique(yt[g == s])) > 1]
    return dict(accuracy=accuracy_score(yt, yp), balanced_accuracy=balanced_accuracy_score(yt, yp),
                macro_f1=f1_score(yt, yp, average="macro"), n_test_subjects=len(np.unique(g)),
                per_subject_bacc_median=float(np.median(ps)) if ps else np.nan,
                per_subject_bacc_q1=float(np.percentile(ps, 25)) if ps else np.nan,
                per_subject_bacc_q3=float(np.percentile(ps, 75)) if ps else np.nan), \
        confusion_matrix(yt, yp, labels=classes)


def cross_subject(X, y, g, mk, k=10):
    from sklearn.model_selection import GroupKFold
    yp = np.empty_like(y)
    for tr, te in GroupKFold(min(k, len(np.unique(g)))).split(X, y, g):
        yp[te] = mk().fit(X[tr], y[tr]).predict(X[te])
    return yp


def alpha_check(D, chans):
    occ = [chans.index(c) for c in EA.OCCIPITAL if c in chans]
    X = transform(D, "relative", len(chans)).reshape(len(D["X"]), len(EA.BANDS), len(chans))
    ai = list(EA.BANDS).index("alpha")
    df = pd.DataFrame(dict(g=D["g"], y=D["y"], a=np.exp(X[:, ai, occ]).mean(1)))
    m = df[df.y.isin(["eye_open", "eye_close"])].groupby(["g", "y"]).a.mean().unstack().dropna()
    if len(m) < 5:
        return "alpha check: not enough subjects"
    p = wilcoxon(m.eye_close, m.eye_open).pvalue
    return (f"Occipital relative alpha, eyes closed vs open: increase in {(m.eye_close > m.eye_open).sum()}/{len(m)} "
            f"subjects, median ratio {np.median(m.eye_close / m.eye_open):.2f}, Wilcoxon p = {p:.2g}")


def run(jobs=1, download=True, subjects=None, out_dir=None, mmidb_dir=None):
    global EEGMMIDB_DIR
    if mmidb_dir:
        EEGMMIDB_DIR = mmidb_dir
    out_dir = out_dir or EA.OUT_DIR
    cache_dir = os.path.join(out_dir, "cache")
    ext_dir = os.path.join(out_dir, "external"); os.makedirs(ext_dir, exist_ok=True)
    mm_cache = os.path.join(out_dir, "cache_mmidb"); os.makedirs(mm_cache, exist_ok=True)

    coh, chans = load_cohort(cache_dir)
    print(f"[external] cohort: {len(np.unique(coh['g']))} subjects, {len(coh['y'])} epochs, channels: {', '.join(chans)}")
    subs = [s for s in (subjects or SUBJECTS) if s not in EXCLUDE]
    print(f"[external] EEGMMIDB: {len(subs)} subjects -> {EEGMMIDB_DIR}  (download={download}, "
          f"exists: {os.path.isdir(EEGMMIDB_DIR)})")
    if download:
        download_all(subs)
    else:
        found = [x for x in subs if os.path.exists(edf_path(x, RUNS["eye_open"][0]))]
        print(f"  --no-download: EDF files found for {len(found)} of {len(subs)} subjects")
        subs = found
    jobs_args = [(s, chans, mm_cache, False, EEGMMIDB_DIR) for s in subs]
    caches, t0 = [], time.time()
    if jobs > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as ex:
            futs = {s: ex.submit(process_subject, a) for s, a in zip(subs, jobs_args)}
            for i, (s, fu) in enumerate(futs.items(), 1):
                try:
                    caches.append(fu.result())
                except Exception as e:
                    print(f"\n  [skip] S{s:03d}: {e}")
                print(f"\r  {i}/{len(subs)}  {time.time()-t0:.0f}s", end="")
    else:
        for i, a in enumerate(jobs_args, 1):
            try:
                caches.append(process_subject(a))
            except Exception as e:
                print(f"\n  [skip] S{a[0]:03d}: {e}")
            print(f"\r  {i}/{len(subs)}  {time.time()-t0:.0f}s", end="")
    print()
    if len(caches) < 5:
        raise SystemExit("Too few EEGMMIDB subjects processed.")
    mm = load_mmidb(caches)
    print(f"[external] EEGMMIDB processed: {len(caches)} subjects, epochs per class:",
          pd.Series(mm["y"]).value_counts().to_dict())
    checks = ["Cohort  - " + alpha_check(coh, chans), "EEGMMIDB - " + alpha_check(mm, chans)]
    print("\n".join("  " + c for c in checks))

    tasks = {"2-class (eye_open vs eye_close)": ["eye_close", "eye_open"],
             "3-class (+ hands_up)": ["eye_close", "eye_open", "hands_up"]}
    rows, cms = [], {}
    for norm in ["relative", "relative + subject z-score"]:
        Xc, Xm = transform(coh, norm, len(chans)), transform(mm, norm, len(chans))
        for clean in [False, True]:
            for tname, cls in tasks.items():
                ac, yc, gc = subset(coh, Xc, cls, clean)
                am, ym, gm = subset(mm, Xm, cls, clean)
                if len(np.unique(yc)) < len(cls) or len(np.unique(ym)) < len(cls):
                    continue
                for mname, mk in models().items():
                    t1 = time.time()
                    evals = {"Cohort -> EEGMMIDB": (ym, mk().fit(ac, yc).predict(am), gm),
                             "EEGMMIDB -> Cohort": (yc, mk().fit(am, ym).predict(ac), gc),
                             "Cohort CV (cross-subject)": (yc, cross_subject(ac, yc, gc, mk), gc),
                             "EEGMMIDB CV (cross-subject)": (ym, cross_subject(am, ym, gm, mk), gm)}
                    for ename, (yt, yp, gg) in evals.items():
                        met, cm = metrics(yt, yp, gg, cls)
                        rows.append(dict(task=tname, normalisation=norm, epochs="artifact-free" if clean else "all",
                                         model=mname, evaluation=ename, n_train_epochs=None, n_test_epochs=len(yt), **met))
                        cms[(tname, norm, clean, mname, ename)] = (cm, cls)
                    short = {"Cohort -> EEGMMIDB": "C->M", "EEGMMIDB -> Cohort": "M->C",
                             "Cohort CV (cross-subject)": "C-CV", "EEGMMIDB CV (cross-subject)": "M-CV"}
                    print(f"  {tname[:7]} | {norm:27s} | {'clean' if clean else 'all  '} | {mname}: " +
                          "  ".join(f"{short[r['evaluation']]}={r['balanced_accuracy']:.3f}" for r in rows[-4:]) +
                          f"  [{time.time()-t1:.0f}s]")
    res = pd.DataFrame(rows).drop(columns="n_train_epochs")
    res["chance"] = res.task.map(lambda t: 0.5 if t.startswith("2") else 1 / 3)
    res.to_csv(os.path.join(ext_dir, "external_results.csv"), index=False)
    with pd.ExcelWriter(os.path.join(ext_dir, "external_results.xlsx")) as xw:
        res.to_excel(xw, sheet_name="External", index=False)
    figures_external(res, cms, ext_dir)
    write_summary(ext_dir, coh, mm, checks, res)
    print(f"[external] done -> {ext_dir}")
    return res


def figures_external(res, cms, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    sel = res[(res.model == "RF") & (res.epochs == "all")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for ax, norm in zip(axs, ["relative", "relative + subject z-score"]):
        piv = sel[sel.normalisation == norm].pivot(index="evaluation", columns="task", values="balanced_accuracy")
        piv.plot.barh(ax=ax, width=.75, legend=ax is axs[1])
        ax.axvline(.5, ls="--", c="k", lw=.7); ax.axvline(1 / 3, ls=":", c="k", lw=.7)
        ax.set(xlim=(0, 1), xlabel="Balanced accuracy", ylabel="", title=f"RF, all epochs — {norm}")
    axs[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig(os.path.join(out, "figE1_external_validation.png"), dpi=200); plt.close(fig)

    keys = [k for k in cms if k[1] == "relative + subject z-score" and not k[2] and k[3] == "RF"
            and k[4] in ("Cohort -> EEGMMIDB", "EEGMMIDB -> Cohort")]
    if keys:
        fig, axs = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4))
        for ax, k in zip(np.atleast_1d(axs), keys):
            cm, cls = cms[k]; cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
            for i in range(len(cls)):
                for j in range(len(cls)):
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if cmn[i, j] > .6 else "black")
            ax.set(xticks=range(len(cls)), yticks=range(len(cls)), xticklabels=cls, yticklabels=cls,
                   xlabel="Predicted", ylabel="True", title=f"{k[4]}\n{k[0]}")
        fig.suptitle("RF, relative + subject z-score, all epochs"); fig.tight_layout()
        fig.savefig(os.path.join(out, "figE2_external_confusion.png"), dpi=200); plt.close(fig)


def write_summary(out, coh, mm, checks, res):
    cols = ["task", "normalisation", "epochs", "model", "evaluation", "balanced_accuracy", "macro_f1",
            "per_subject_bacc_median", "n_test_subjects"]
    L = ["# External validation (EEGMMIDB)", "",
         f"- Cohort: {len(np.unique(coh['g']))} subjects, {len(coh['y'])} epochs",
         f"- EEGMMIDB: {len(np.unique(mm['g']))} subjects, {len(mm['y'])} epochs "
         f"({pd.Series(mm['y']).value_counts().to_dict()})",
         "- hands_up in EEGMMIDB = event T1 (real both-fist movement) from the available runs among 5/9/13", "",
         "## Sanity check", "", *[f"- {c}" for c in checks], "",
         "## Results (balanced accuracy; chance 0.50 for 2-class, 0.33 for 3-class)", "",
         res[cols].round(3).to_string(index=False), "",
         "Interpretation guide: compare 'Cohort -> EEGMMIDB' with 'Cohort CV'. A small gap means the features "
         "generalise across devices, sites and populations. A larger drop for the 3-class task than for the "
         "2-class task suggests that hands_up separation in the cohort relies on movement-related signals "
         "(arm raising) that are absent in the low-motion fist task of EEGMMIDB."]
    open(os.path.join(out, "summary_external.md"), "w", encoding="utf-8").write("\n".join(L))


def parse_subjects(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-download", action="store_true", help="use only EDF files already in EEGMMIDB_DIR")
    ap.add_argument("--subjects", type=str, default=None, help="e.g. 1-20 or 1,2,5-9")
    ap.add_argument("--out", type=str, default=None, help="override OUT_DIR of eeg_analysis.py")
    ap.add_argument("--mmidb-dir", type=str, default=None)
    a = ap.parse_args()
    run(jobs=a.jobs, download=not a.no_download, subjects=parse_subjects(a.subjects) if a.subjects else None,
        out_dir=a.out, mmidb_dir=a.mmidb_dir)
