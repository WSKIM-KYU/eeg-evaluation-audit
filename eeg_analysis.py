# -*- coding: utf-8 -*-
"""
EEG temporal-continuity follow-up analysis (19-ch EEG + 3-axis accelerometer)

Usage (Windows, Anaconda Prompt):
    pip install numpy scipy pandas scikit-learn matplotlib openpyxl
    python eeg_analysis.py                  # full run
    python eeg_analysis.py --peek           # show detected file format (delimiter, header, channels)
    python eeg_analysis.py --check          # only list files / subject matching
    python eeg_analysis.py --scan           # read all files once: duration, amplitude, corruption check
    python eeg_analysis.py --scan-dir cohort22   # same check for the 22-subject raw folder
    python eeg_analysis.py --jobs 4         # parallel feature extraction
    python eeg_analysis.py --jobs 4 --external   # + EEGMMIDB external validation (external_validation.py)

Steps
  1. File inventory and subject matching across condition folders
  2. Preprocessing (DC removal, 60 Hz notch, 1-45 Hz band-pass, edge trimming)
  3. Per-recording quality control (artifact epochs, flat channels, line noise, head motion)
  4. Spectral analysis (PSD, absolute/relative band power) + paired statistics
  5. Epoch features (2-s windows) cached per file -> rerun is fast
  6. Classification under three evaluation schemes
        (a) random epoch split  (b) within-recording time split  (c) cross-subject
     for EEG band power, accelerometer only, and paper-style single samples
  7. Figures, CSV/Excel tables, summary.md
"""
import os, re, glob, json, argparse, warnings, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# =============================== CONFIG ===============================
# Paths can be set with environment variables (EEG_DATA_DIR, EEG_OUTPUT_DIR) or edited here.
DATA_DIR = os.environ.get("EEG_DATA_DIR", "data")
CONDITIONS = {                      # label : folder (one file per participant, e.g. S01_EC.bdf)
    "eye_close": os.path.join(DATA_DIR, "eye_close"),
    "eye_open":  os.path.join(DATA_DIR, "eye_open"),
    "hands_up":  os.path.join(DATA_DIR, "hands_up"),
}
OUT_DIR = os.environ.get("EEG_OUTPUT_DIR", "analysis_output")
# Optional second dataset recorded with the same device; only used by --scan-dir cohort22 checks.
COHORT22_DIR = os.environ.get("EEG_COHORT22_DIR", os.path.join("data", "cohort22"))
# Recordings with fewer than the 19 reference channels:
#   "exclude" -> drop those recordings and keep the full 19-channel montage (default)
#   "common"  -> keep every recording and use only channels present in all of them
CHANNEL_POLICY = "exclude"
# Channel-level quality control (per recording). A channel is "bad" if it is flat, if more than
# BAD_CH_EPOCH_FRAC of its 2-s epochs exceed ARTIFACT_PTP_UV, or if its amplitude is an outlier
# among the channels of the same recording (robust z of log SD > BAD_CH_ROBUST_Z).
BAD_CH_FLAT_UV = 1.0
BAD_CH_EPOCH_FRAC = 0.5
BAD_CH_ROBUST_Z = 3.5
MAX_BAD_CH = 4            # recordings with more bad channels are left out of the QC-cleaned analyses
N_TREES = 300             # random-forest size (--trees)
# results of each channel policy go to their own subfolder, e.g. ...\analysis_output\policy_common
OUT_DIR_BASE = OUT_DIR
OUT_DIR = os.path.join(OUT_DIR_BASE, f"policy_{CHANNEL_POLICY}")
FILE_GLOB = ["*.bdf", "*.txt", "*.csv"]   # file patterns; if a .bdf and a .txt/.csv share a name, the .bdf (raw) is used
# Subject ID is taken from the file name with this regex (first group).
# Example: "S012_eye_open.csv" -> "012", "kim_012.csv" -> "012".
# Run with --check first and adjust if subjects do not match across folders.
SUBJECT_REGEX = r"(\d+)"
FS = 500                 # sampling rate (Hz)
TRIM_SEC = 1.5           # removed at start and end of each recording
LINE_FREQ = 60           # Korea: 60 Hz
BAND = (1.0, 45.0)
EPOCH_SEC, STEP_SEC = 2.0, 1.0
ARTIFACT_PTP_UV = 150.0  # epoch flagged if any channel exceeds this peak-to-peak
SAMPLES_PER_REC = 300    # for paper-style single-sample classification
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
OCCIPITAL = ["O1", "O2", "Pz", "P3", "P4"]
# used only if a file has no header row (assumed column order after an optional time column)
DEFAULT_CHANNELS = ["F7", "Fp1", "Fp2", "F8", "F3", "Fz", "F4", "C3", "Cz", "P8", "P7", "Pz", "P4", "T3", "P3",
                    "O1", "O2", "C4", "T4"]
MOTOR = ["C3", "Cz", "C4"]
RANDOM_STATE = 0
# ======================================================================


# --------------------------- file handling ---------------------------
def subject_of(path):
    m = re.search(SUBJECT_REGEX, os.path.splitext(os.path.basename(path))[0])
    if not m:
        return os.path.splitext(os.path.basename(path))[0]
    return m.group(1).lstrip("0") or "0"


def inventory():
    rows = []
    for cond, d in CONDITIONS.items():
        pats = FILE_GLOB if isinstance(FILE_GLOB, (list, tuple)) else [FILE_GLOB]
        files = sorted({f for pat in pats for f in glob.glob(os.path.join(d, pat))})
        raw_stems = {os.path.splitext(f)[0].lower() for f in files if f.lower().endswith(".bdf")}
        files = [f for f in files if f.lower().endswith(".bdf") or os.path.splitext(f)[0].lower() not in raw_stems]
        if not files:
            print(f"[WARN] no files in {d}")
        for f in files:
            rows.append(dict(cond=cond, path=f, subject=subject_of(f), size_mb=os.path.getsize(f) / 1e6))
    inv = pd.DataFrame(rows)
    if inv.empty:
        raise SystemExit("No files found. Check CONDITIONS paths.")
    return inv


def report_inventory(inv):
    tab = inv.pivot_table(index="subject", columns="cond", values="path", aggfunc="count", fill_value=0)
    print(f"\nFiles: {len(inv)}  |  subjects: {tab.shape[0]}  |  total {inv.size_mb.sum():.0f} MB")
    print(inv.groupby("cond").size().to_string())
    complete = tab[(tab > 0).all(axis=1)]
    print(f"Subjects with all {len(CONDITIONS)} conditions: {len(complete)}")
    multi = tab[(tab > 1).any(axis=1)]
    if len(multi):
        print(f"[WARN] {len(multi)} subjects have >1 file in a condition (check SUBJECT_REGEX):")
        print(multi.head(10).to_string())
    missing = tab[(tab == 0).any(axis=1)]
    if len(missing):
        print(f"[INFO] {len(missing)} subjects missing at least one condition:")
        print(missing.head(10).to_string())
    return tab


KNOWN_CH = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6",
            "O1", "O2", "T7", "T8", "P7", "P8"]
_LAYOUT = {}


def _clean(name):
    return re.sub(r"\(.*?\)|\[.*?\]", "", str(name)).strip().strip('"').strip()


def _is_num(tok):
    try:
        float(tok); return True
    except ValueError:
        return False


def detect_layout(path, n_lines=60):
    """Detect encoding, delimiter, header line and EEG/ACC column indices of a text/CSV file."""
    for enc in ("utf-8-sig", "cp949", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                lines = []
                for _ in range(n_lines):
                    l = f.readline()
                    if not l:
                        break
                    lines.append(l.rstrip("\r\n"))
            break
        except UnicodeDecodeError:
            continue
    best = None
    for sep in [",", "\t", ";", r"\s+"]:
        split = (lambda l: l.split()) if sep == r"\s+" else (lambda l, s=sep: l.split(s))
        rows = [[t.strip() for t in split(l)] for l in lines if l.strip()]
        num_rows = [i for i, r in enumerate(rows) if len(r) >= 19 and sum(_is_num(t) for t in r if t) >= 19]
        if len(num_rows) >= 5 and (best is None or len(rows[num_rows[0]]) > best[1]):
            best = (sep, len(rows[num_rows[0]]), rows, num_rows[0])
    if best is None:
        raise ValueError("could not find >=19 numeric columns; run --peek and check the file format")
    sep, ncol, rows, first_num = best
    raw_lines = [l for l in lines if l.strip()]
    data_start = lines.index(raw_lines[first_num])
    header = None
    for i in range(first_num - 1, -1, -1):
        names = [_clean(t).upper() for t in rows[i]]
        if sum(n in KNOWN_CH for n in names) >= 10:
            header = [_clean(t) for t in rows[i]]; break
    if header:
        up = [h.upper() for h in header]
        eeg_idx = [i for i, h in enumerate(up) if h in KNOWN_CH]
        acc_idx = [i for i, h in enumerate(up) if h.startswith("ACC") or h in ("X", "Y", "Z", "AX", "AY", "AZ")
                   or "ACCEL" in h][:3]
        chans = [header[i] for i in eeg_idx]
    else:  # no header: [time] + 19 EEG (+ 3 ACC)
        off = 1 if ncol in (20, 23, 24) else 0
        eeg_idx = list(range(off, off + 19))
        acc_idx = list(range(off + 19, off + 22)) if ncol >= off + 22 else []
        chans = DEFAULT_CHANNELS
    return dict(encoding=enc, sep=sep, data_start=data_start, ncol=ncol, header=header,
                eeg_idx=eeg_idx, acc_idx=acc_idx, chans=chans, preview=lines[:max(data_start + 3, 8)])


BIOTECK_MAGIC = b"Bioteck Data File"


def is_bioteck(path):
    with open(path, "rb") as f:
        return BIOTECK_MAGIC in f.read(64)


def read_bioteck(path):
    """Raw recording of the CGX/Bioteck acquisition software (.bdf, 'Bioteck Data File ver.3').
    Layout (verified against exported CSV): text header lines, a tab-separated channel-name line,
    then little-endian float32 rows [time, ch1 ... chN], followed by a software metadata trailer."""
    with open(path, "rb") as f:
        buf = f.read()
    end_hdr, lines, pos = None, [], buf.find(b"\n") + 1
    while pos < len(buf) and len(lines) < 50:
        nl = buf.find(b"\n", pos)
        line = buf[pos:nl].decode("latin-1")
        lines.append(line); pos = nl + 1
        if "\t" in line and sum(_clean(t).upper() in KNOWN_CH for t in line.split("\t")) >= 10:
            end_hdr = pos; break
    if end_hdr is None:
        raise ValueError("Bioteck header: channel line not found")
    meta = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip() for l in lines if ":" in l}
    fs = float(meta.get("Sampling Rate", FS))
    dtypes = [t for t in meta.get("Data Type", "").split(";") if t != ""]
    if dtypes and set(dtypes) - {"0", "2"}:
        raise ValueError(f"unsupported Bioteck data type codes {sorted(set(dtypes))}; please send one file for checking")
    names = [t for t in lines[-1].split("\t") if t.strip()]
    ncol = len(names) + 1
    nrow = (len(buf) - end_hdr) // (4 * ncol)
    a = np.frombuffer(buf[end_hdr:end_hdr + nrow * 4 * ncol], dtype="<f4").reshape(nrow, ncol).astype(np.float64)
    t = a[:, 0]
    ok = np.isfinite(t) & (np.abs(t - np.arange(nrow) / fs) < 0.5 / fs)
    n = int(np.argmin(ok)) if not ok.all() else nrow          # stop at the metadata trailer
    a = a[:n]
    if abs(fs - FS) > 1e-6:
        raise ValueError(f"sampling rate in file is {fs} Hz but FS = {FS}")
    up = [_clean(x).upper() for x in names]
    eeg_i = [i for i, u in enumerate(up) if u in KNOWN_CH]
    acc_i = [i for i, u in enumerate(up) if u.startswith("ACC")][:3]
    eeg = a[:, [i + 1 for i in eeg_i]]
    acc = a[:, [i + 1 for i in acc_i]] if acc_i else np.zeros((n, 3))
    return eeg, acc, [_clean(names[i]) for i in eeg_i], dict(meta=meta, n_rows_total=nrow, n_valid=n,
                                                             names=[_clean(x) for x in names])


def read_recording(path):
    if is_bioteck(path):
        eeg, acc, chans, _ = read_bioteck(path)
        return eeg, acc, chans
    lay = detect_layout(path)
    if len(lay["eeg_idx"]) != 19:
        print(f"\n  [WARN] {os.path.basename(path)}: {len(lay['eeg_idx'])} EEG channels detected")
    cols = lay["eeg_idx"] + lay["acc_idx"]
    kw = dict(sep=lay["sep"])
    base = dict(skiprows=lay["data_start"], header=None, usecols=cols, encoding=lay["encoding"],
                engine="c", dtype=np.float32, **kw)
    try:                                   # pandas >= 1.3
        df = pd.read_csv(path, on_bad_lines="skip", **base)
    except TypeError:                      # older pandas
        df = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=False, **base)
    df = df.dropna()
    eeg = df[lay["eeg_idx"]].to_numpy(np.float64)
    acc = df[lay["acc_idx"]].to_numpy(np.float64) if lay["acc_idx"] else np.zeros((len(df), 3))
    return eeg, acc, [str(c) for c in lay["chans"]]


def scan(inv, out_csv):
    """Read every file once; report raw/text type, duration, amplitude and plausibility."""
    rows, t0 = [], time.time()
    for i, r in enumerate(inv.itertuples(), 1):
        row = dict(cond=r.cond, subject=r.subject, file=os.path.basename(r.path))
        try:
            raw = is_bioteck(r.path)
            row["type"] = "bioteck_raw" if raw else "text"
            if raw:
                eeg, acc, ch, info = read_bioteck(r.path); allnames = info["names"]
            else:
                eeg, acc, ch = read_recording(r.path); lay = detect_layout(r.path); allnames = lay["header"] or []
            d = float(np.median(np.abs(np.diff(eeg, axis=0))))
            have = {norm_ch(c) for c in ch}
            missing = [c for c in DEFAULT_CHANNELS if norm_ch(c) not in have]
            other = [n for n in allnames if n and norm_ch(n) not in KNOWN_CH and not n.upper().startswith("ACC")
                     and n.upper() != "TIME" and not n.startswith("Unnamed")]
            row.update(n_channels=len(ch), missing_channels=";".join(missing), other_columns=";".join(other),
                       duration_s=len(eeg) / FS, eeg_std_median_uV=float(np.median(eeg.std(0))),
                       median_abs_step_uV=d, acc_std_mean=float(acc.std(0).mean()),
                       status="OK" if (d < 50 and not missing) else "CHECK")
        except Exception as e:
            row.update(status="ERROR", error=str(e)[:200])
        rows.append(row)
        print(f"\r  {i}/{len(inv)}  {time.time()-t0:.0f}s", end="")
    if not rows:
        print("  No files found. Check the folder path and file extensions (FILE_GLOB)."); return
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
    print(f"\n\nStatus: {df.status.value_counts().to_dict()}")
    if "duration_s" in df:
        print(df.groupby("cond").duration_s.describe()[["count", "min", "50%", "max"]].round(1).to_string())
    bad = df[df.status != "OK"]
    if len(bad):
        cols = [c for c in ["cond", "subject", "file", "n_channels", "missing_channels", "other_columns", "duration_s",
                            "median_abs_step_uV", "status", "error"] if c in bad]
        print("\nFiles needing attention:"); print(bad[cols].head(30).to_string(index=False))
    print(f"\nSaved -> {out_csv}")


_ALIAS = {"T7": "T3", "T8": "T4", "T5": "P7", "T6": "P8"}


def norm_ch(c):
    u = _clean(c).upper()
    return _ALIAS.get(u, u)


def reorder_record(r, target):
    """Select/reorder a cached recording's channels (features, PSD, samples) to the target channel list."""
    cur = [norm_ch(c) for c in r["chans"]]
    idx = [cur.index(norm_ch(t)) for t in target]
    nb, n = len(BANDS), len(r["feats"])
    r["feats"] = r["feats"].reshape(n, nb, len(cur))[:, :, idx].reshape(n, -1)
    r["psd"] = r["psd"][:, idx]
    r["samples"] = r["samples"][:, idx]
    if "ptp_ch" in r:
        r["ptp_ch"] = r["ptp_ch"][:, idx]
    if "ch_std" in r:
        r["ch_std"] = r["ch_std"][idx]
    r["chans"] = np.array(target)
    return r


def harmonize_channels(recs, caches):
    sets = [{norm_ch(c) for c in r["chans"]} for r in recs]
    ref = [c for c in DEFAULT_CHANNELS]
    full = [all(norm_ch(c) in s for c in ref) for s in sets]
    if all(full):
        target, keep_idx = ref, list(range(len(recs)))
    elif CHANNEL_POLICY == "common":
        common = set.intersection(*sets)
        target = [c for c in ref if norm_ch(c) in common]
        keep_idx = list(range(len(recs)))
        print(f"[channels] policy=common: using {len(target)} channels present in all recordings "
              f"(dropped: {[c for c in ref if c not in target]})")
    else:
        target = ref
        keep_idx = [i for i, f in enumerate(full) if f]
        dropped = [(str(recs[i]["subject"]), str(recs[i]["cond"]),
                    [c for c in ref if norm_ch(c) not in sets[i]]) for i in range(len(recs)) if not full[i]]
        print(f"[channels] policy=exclude: {len(dropped)} recording(s) without all 19 channels excluded:")
        for s_, c_, m_ in dropped:
            print(f"    subject {s_} {c_}: missing {m_}")
    recs = [reorder_record(recs[i], target) for i in keep_idx]
    return recs, [caches[i] for i in keep_idx], target


def peek(inv):
    for cond in inv.cond.unique():
        f = inv[inv.cond == cond].path.iloc[0]
        print(f"\n===== {cond}: {f}")
        try:
            if is_bioteck(f):
                eeg, acc, ch, info = read_bioteck(f)
                for k, v in info["meta"].items():
                    print(f"  | {k}: {v[:100]}")
                print(f"  -> Bioteck raw file: {len(ch)} EEG channels {ch}")
                print(f"  -> {info['n_valid']} samples = {info['n_valid']/FS:.1f} s at {FS} Hz "
                      f"(trailer rows ignored: {info['n_rows_total'] - info['n_valid']})")
                d = np.median(np.abs(np.diff(eeg, axis=0)))
                print(f"  -> EEG std (uV) median {np.median(eeg.std(0)):.1f}; median |sample-to-sample change| {d:.2f} uV; "
                      f"ACC std {np.round(acc.std(0), 2)}")
                continue
            lay = detect_layout(f)
            for l in lay["preview"]:
                print("  | " + l[:160])
            sep_name = {",": "comma", "\t": "tab", ";": "semicolon"}.get(lay["sep"], "whitespace")
            print(f"  -> encoding={lay['encoding']}, delimiter={sep_name}, data starts at line {lay['data_start']+1}, "
                  f"{lay['ncol']} columns, header={'yes' if lay['header'] else 'NO (assumed layout)'}")
            print(f"  -> EEG columns ({len(lay['eeg_idx'])}): {lay['chans']}")
            print(f"  -> ACC columns: {[lay['header'][i] for i in lay['acc_idx']] if lay['header'] else lay['acc_idx']}")
            eeg, acc, ch = read_recording(f)
            d = np.median(np.abs(np.diff(eeg, axis=0)))
            print(f"  -> read OK: {len(eeg)} samples = {len(eeg)/FS:.1f} s at {FS} Hz; "
                  f"EEG std (uV) median {np.median(eeg.std(0)):.1f}; median |sample-to-sample change| {d:.2f} uV; "
                  f"ACC std {np.round(acc.std(0), 2)}")
            if d > 50:
                print("  [WARN] sample-to-sample changes are implausibly large for EEG in uV -> values look corrupted")
        except Exception as e:
            print(f"  [ERROR] {e}")


# --------------------------- preprocessing ---------------------------
# normalised frequencies (compatible with old SciPy versions without the fs= argument)
_SOS = butter(4, [BAND[0] / (FS / 2), BAND[1] / (FS / 2)], "bandpass", output="sos")
_NOTCH = [iirnotch(f0 / (FS / 2), 30) for f0 in (LINE_FREQ, 2 * LINE_FREQ) if f0 < FS / 2]


def preprocess(eeg):
    x = eeg - np.nanmean(eeg, axis=0)
    x = np.nan_to_num(x)
    for b, a in _NOTCH:
        x = filtfilt(b, a, x, axis=0)
    x = sosfiltfilt(_SOS, x, axis=0)
    t = int(TRIM_SEC * FS)
    return x[t:len(x) - t] if len(x) > 2 * t + FS * EPOCH_SEC else x


def band_power(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return p[m].sum(axis=0) * (f[1] - f[0])


# ------------------------ per-file processing ------------------------
def process_file(args):
    path, cond, subject, cache_dir = args[:4]
    step = args[4] if len(args) > 4 else STEP_SEC        # passed explicitly: worker processes start with defaults
    key = re.sub(r"[^\w]", "_", f"{cond}_{os.path.basename(path)}") + "_v2" + \
        ("" if step == 1.0 else f"_step{step:g}s")
    cache = os.path.join(cache_dir, key + ".npz")
    if os.path.exists(cache):
        return cache
    eeg, acc, chans = read_recording(path)
    t = int(TRIM_SEC * FS)
    raw = eeg - eeg.mean(0)
    fr, praw = welch(raw[t:-t], FS, nperseg=2 * FS, axis=0)
    line_ratio = float(np.median(praw[np.argmin(abs(fr - LINE_FREQ))] /
                                 praw[(fr > 40) & (fr < 55)].mean(0)))
    x = preprocess(eeg)
    a = acc - acc.mean(0)
    a = a[t:len(a) - t][:len(x)]
    f, psd = welch(x, FS, nperseg=2 * FS, axis=0)          # (freq, ch)

    W, S = int(EPOCH_SEC * FS), int(step * FS)
    starts = np.arange(0, len(x) - W + 1, S)
    feats, accf, ptp, ptp_ch = [], [], [], []
    for s in starts:
        w = x[s:s + W]
        fe, pe = welch(w, FS, nperseg=FS, axis=0)
        feats.append(np.log([band_power(fe, pe, lo, hi) + 1e-12 for lo, hi in BANDS.values()]).ravel())
        aw = a[s:s + W]
        fa, pa = welch(aw, FS, nperseg=FS, axis=0)
        accf.append(np.r_[aw.std(0), np.log([band_power(fa, pa, lo, hi) + 1e-9
                                             for lo, hi in [(0.5, 2), (2, 5), (5, 10), (10, 30)]]).ravel()])
        pc = np.ptp(w, axis=0); ptp_ch.append(pc); ptp.append(pc.max())
    rng = np.random.RandomState(abs(hash(key)) % 2**32)
    idx = np.sort(rng.choice(len(x), min(SAMPLES_PER_REC, len(x)), replace=False))
    np.savez_compressed(
        cache, chans=np.array(chans), f=f, psd=psd, feats=np.array(feats, np.float32),
        accf=np.array(accf, np.float32), ptp=np.array(ptp, np.float32), ptp_ch=np.array(ptp_ch, np.float32),
        ch_std=x.std(0).astype(np.float32),
        start_sec=starts / FS, dur_sec=len(x) / FS,
        samples=x[idx].astype(np.float32), sample_sec=idx / FS,
        acc_rms=float(np.linalg.norm(a, axis=1).std()),
        flat=int((x.std(0) < 0.5).sum()), line_ratio=line_ratio,
        raw_dur=len(eeg) / FS, cond=cond, subject=subject)
    return cache


# ------------------------------ analysis -----------------------------
def load_all(caches):
    recs = []
    for c in caches:
        z = np.load(c, allow_pickle=False)
        recs.append({k: z[k] for k in z.files})
    return recs


def qc_table(recs):
    rows = []
    for r in recs:
        rows.append(dict(subject=str(r["subject"]), cond=str(r["cond"]), duration_s=float(r["raw_dur"]),
                         n_epochs=len(r["ptp"]), artifact_epoch_frac=float((r["ptp"] > ARTIFACT_PTP_UV).mean()),
                         median_max_ptp_uV=float(np.median(r["ptp"])), flat_channels=int(r["flat"]),
                         line_noise_ratio=float(r["line_ratio"]), head_acc_rms_mg=float(r["acc_rms"])))
    return pd.DataFrame(rows)


def bandpower_table(recs):
    rows = []
    for r in recs:
        f, p, chans = r["f"], r["psd"], [str(c) for c in r["chans"]]
        tot = band_power(f, p, *BAND)
        for b, (lo, hi) in BANDS.items():
            bp = band_power(f, p, lo, hi)
            for ch, v, rv in zip(chans, bp, bp / tot):
                rows.append(dict(subject=str(r["subject"]), cond=str(r["cond"]), band=b, channel=ch,
                                 abs_power=v, rel_power=rv))
    return pd.DataFrame(rows)


def paired_test(bp, c1, c2, band, chans, value="rel_power", log_ratio=False):
    d = bp[(bp.band == band) & (bp.channel.isin(chans)) & (bp.cond.isin([c1, c2]))]
    m = d.groupby(["subject", "cond"])[value].mean().unstack().dropna()
    if c1 not in m or c2 not in m or len(m) < 5:
        return None
    a, b = m[c1].values, m[c2].values
    stat = wilcoxon(np.log(b / a)) if log_ratio else wilcoxon(b, a)
    return dict(test=f"{band} {'/'.join(chans)}: {c2} vs {c1}", n=len(m), mean_c1=a.mean(), mean_c2=b.mean(),
                median_ratio=float(np.median(b / a)), n_increase=int((b > a).sum()), p_wilcoxon=float(stat.pvalue))


def statistics(bp):
    tests = []
    pairs = [("eye_open", "eye_close", "alpha", OCCIPITAL), ("eye_open", "eye_close", "theta", ["Fz", "F3", "F4"]),
             ("eye_open", "hands_up", "alpha", MOTOR), ("eye_open", "hands_up", "beta", MOTOR),
             ("eye_open", "hands_up", "gamma", ["T3", "T4", "F7", "F8"]), ("eye_open", "hands_up", "delta", ["Fp1", "Fp2"])]
    for c1, c2, band, ch in pairs:
        r = paired_test(bp, c1, c2, band, ch, value="abs_power" if c2 == "hands_up" else "rel_power")
        if r:
            tests.append(r)
    return pd.DataFrame(tests)


def build_epochs(recs, clean_only=False):
    X, A, y, g, ts, te, dur = [], [], [], [], [], [], []
    labels = sorted({str(r["cond"]) for r in recs})
    for r in recs:
        keep = r["ptp"] <= ARTIFACT_PTP_UV if clean_only else np.ones(len(r["ptp"]), bool)
        n = keep.sum()
        X.append(r["feats"][keep]); A.append(r["accf"][keep])
        y += [labels.index(str(r["cond"]))] * n; g += [str(r["subject"])] * n
        ts.append(r["start_sec"][keep]); dur += [float(r["dur_sec"])] * n
    return (np.vstack(X), np.vstack(A), np.array(y), np.array(g), np.concatenate(ts), np.array(dur), labels)


def build_samples(recs):
    labels = sorted({str(r["cond"]) for r in recs})
    X = np.vstack([r["samples"] for r in recs])
    y = np.concatenate([[labels.index(str(r["cond"]))] * len(r["samples"]) for r in recs])
    g = np.concatenate([[str(r["subject"])] * len(r["samples"]) for r in recs])
    t = np.concatenate([r["sample_sec"] for r in recs])
    d = np.concatenate([[float(r["dur_sec"])] * len(r["samples"]) for r in recs])
    return X, y, g, t, d, labels


def evaluate(X, y, g, tstart, dur, make_model, win_sec):
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    res, cms = {}, {}

    def score(name, yt, yp):
        res[name] = dict(accuracy=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro"))
        cms[name] = confusion_matrix(yt, yp, labels=np.unique(y))

    yt_all, yp_all = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X, y):
        m = make_model().fit(X[tr], y[tr]); yt_all.append(y[te]); yp_all.append(m.predict(X[te]))
    score("(a) random split", np.concatenate(yt_all), np.concatenate(yp_all))

    split = 0.6 * dur                           # first 60 % train, then a gap of one window
    tr = (tstart + win_sec) <= split
    te = tstart >= split + win_sec
    m = make_model().fit(X[tr], y[tr]); score("(b) within-recording time split", y[te], m.predict(X[te]))

    n_sub = len(np.unique(g)); k = min(10, n_sub)
    yt_all, yp_all, per_sub = [], [], {}
    for tr, te in GroupKFold(k).split(X, y, g):
        m = make_model().fit(X[tr], y[tr]); p = m.predict(X[te])
        yt_all.append(y[te]); yp_all.append(p)
        for s in np.unique(g[te]):
            mk = g[te] == s; per_sub[s] = accuracy_score(y[te][mk], p[mk])
    score(f"(c) cross-subject ({k}-fold by subject)", np.concatenate(yt_all), np.concatenate(yp_all))
    return res, cms, per_sub


def classification(recs, out):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rf = lambda: RandomForestClassifier(N_TREES, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    rows, all_cms, persub = [], {}, {}
    X, A, y, g, ts, dur, labels = build_epochs(recs)
    sets = {"EEG band power (LR)": (X, lr), "EEG band power (RF)": (X, rf), "Accelerometer only (RF)": (A, rf),
            "EEG + accelerometer (RF)": (np.hstack([X, A]), rf)}
    Xc, Ac, yc, gc, tsc, durc, _ = build_epochs(recs, clean_only=True)
    if len(np.unique(yc)) == len(labels) and len(yc) > 200:
        sets["EEG band power, artifact-free epochs (RF)"] = (Xc, rf)
    for name, (F, mk) in sets.items():
        yy, gg, tt, dd = (yc, gc, tsc, durc) if "artifact-free" in name else (y, g, ts, dur)
        t0 = time.time(); r, cms, ps = evaluate(F, yy, gg, tt, dd, mk, EPOCH_SEC)
        print(f"  {name}: " + ", ".join(f"{k.split(')')[0]})={v['accuracy']:.3f}" for k, v in r.items())
              + f"  [{time.time()-t0:.0f}s]")
        for k, v in r.items():
            rows.append(dict(features=name, evaluation=k, n_epochs=len(yy), **v))
        all_cms[name] = cms; persub[name] = ps
    try:
        S, sy, sg, st, sd, _ = build_samples(recs)
        light = lambda: RandomForestClassifier(100, min_samples_leaf=5, n_jobs=-1,
                                               random_state=RANDOM_STATE)
        r, cms, ps = evaluate(S, sy, sg, st, sd, light, 0.0)
        print("  Single samples (paper-style, RF): " + ", ".join(f"{k.split(')')[0]})={v['accuracy']:.3f}" for k, v in r.items()))
        for k, v in r.items():
            rows.append(dict(features="Single 2-ms samples, paper-style (RF)", evaluation=k, n_epochs=len(sy), **v))
        all_cms["Single samples (RF)"] = cms
    except Exception as e:
        print(f"  [WARN] single-sample analysis skipped: {e}")
    res = pd.DataFrame(rows)
    res["chance"] = 1 / len(labels)
    return res, all_cms, pd.DataFrame(persub), labels



# ======================= extra analyses (QC, binary, fingerprint, per-subject) =======================
NEIGHBORS = {
    "FP1": ["F7", "F3", "FZ", "FP2"], "FP2": ["F8", "F4", "FZ", "FP1"], "F7": ["FP1", "F3", "T3"],
    "F3": ["FP1", "F7", "FZ", "C3"], "FZ": ["FP1", "FP2", "F3", "F4", "CZ"], "F4": ["FP2", "F8", "FZ", "C4"],
    "F8": ["FP2", "F4", "T4"], "T3": ["F7", "C3", "P7"], "C3": ["F3", "T3", "CZ", "P3"],
    "CZ": ["FZ", "C3", "C4", "PZ"], "C4": ["F4", "CZ", "T4", "P4"], "T4": ["F8", "C4", "P8"],
    "P7": ["T3", "P3", "O1"], "P3": ["C3", "P7", "PZ", "O1"], "PZ": ["CZ", "P3", "P4", "O1", "O2"],
    "P4": ["C4", "PZ", "P8", "O2"], "P8": ["T4", "P4", "O2"], "O1": ["P7", "P3", "PZ", "O2"],
    "O2": ["P8", "P4", "PZ", "O1"]}


def channel_qc(recs):
    """Flag bad channels per recording; stores r['bad'] and returns a long table."""
    rows = []
    for r in recs:
        ch = [str(c) for c in r["chans"]]
        sd = np.asarray(r["ch_std"], float)
        frac = (np.asarray(r["ptp_ch"]) > ARTIFACT_PTP_UV).mean(0)
        lsd = np.log(sd + 1e-9); med = np.median(lsd)
        mad = np.median(np.abs(lsd - med)) * 1.4826 + 1e-9
        z = (lsd - med) / mad
        flat, noisy, outl = sd < BAD_CH_FLAT_UV, frac > BAD_CH_EPOCH_FRAC, np.abs(z) > BAD_CH_ROBUST_Z
        bad = flat | noisy | outl
        r["bad"] = bad
        for i, c in enumerate(ch):
            why = ",".join(w for w, f in (("flat", flat[i]), ("artifact", noisy[i]), ("outlier", outl[i])) if f)
            rows.append(dict(subject=str(r["subject"]), cond=str(r["cond"]), channel=c, std_uV=sd[i],
                             artifact_epoch_frac=frac[i], robust_z=z[i], bad=bool(bad[i]), reason=why))
    return pd.DataFrame(rows)


def cleaned_record(r):
    """Band-power features with bad channels replaced by the mean of good neighbours (log domain),
    and epoch peak-to-peak computed over good channels only."""
    ch = [norm_ch(c) for c in r["chans"]]
    bad = r["bad"]; nb, n = len(BANDS), len(r["feats"])
    F = r["feats"].reshape(n, nb, len(ch)).astype(np.float64).copy()
    good = [i for i in range(len(ch)) if not bad[i]]
    for i in np.where(bad)[0]:
        nbr = [ch.index(x) for x in NEIGHBORS.get(ch[i], []) if x in ch]
        nbr = [j for j in nbr if not bad[j]] or good
        F[:, :, i] = F[:, :, nbr].mean(2)
    ptp_good = np.asarray(r["ptp_ch"])[:, good].max(1) if good else np.asarray(r["ptp"])
    return F.reshape(n, -1), ptp_good


def build_clean(recs, conds=None, artifact_free=False):
    X, y, g, ts, dur, rid = [], [], [], [], [], []
    conds = conds or sorted({str(r["cond"]) for r in recs})
    for k, r in enumerate(recs):
        c = str(r["cond"])
        if c not in conds or r["bad"].sum() > MAX_BAD_CH:
            continue
        F, pg = cleaned_record(r)
        keep = pg <= ARTIFACT_PTP_UV if artifact_free else np.ones(len(F), bool)
        n = int(keep.sum())
        X.append(F[keep]); y += [c] * n; g += [str(r["subject"])] * n
        ts.append(np.asarray(r["start_sec"])[keep]); dur += [float(r["dur_sec"])] * n; rid += [k] * n
    if not X:
        return None
    return dict(X=np.vstack(X), y=np.array(y), g=np.array(g), ts=np.concatenate(ts), dur=np.array(dur),
                rid=np.array(rid), labels=conds)


def run_schemes(X, y, g, ts, dur, mk, schemes=("a", "b", "c"), win=None):
    """Out-of-fold predictions for (a) random split, (b) within-recording time split, (c) cross-subject."""
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    win = EPOCH_SEC if win is None else win
    out = {}
    if "a" in schemes:
        yp = np.empty_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(X, y):
            yp[te] = mk().fit(X[tr], y[tr]).predict(X[te])
        out["a"] = (y, yp, g)
    if "b" in schemes:
        split = 0.6 * dur
        tr, te = (ts + win) <= split, ts >= split + win
        out["b"] = (y[te], mk().fit(X[tr], y[tr]).predict(X[te]), g[te])
    if "c" in schemes:
        yp = np.empty_like(y)
        for tr, te in GroupKFold(min(10, len(np.unique(g)))).split(X, y, g):
            yp[te] = mk().fit(X[tr], y[tr]).predict(X[te])
        out["c"] = (y, yp, g)
    return out


SCHEME_NAME = {"a": "(a) random split", "b": "(b) within-recording time split", "c": "(c) cross-subject"}


def per_subject_acc(yt, yp, g):
    return pd.Series({s: float((yt[g == s] == yp[g == s]).mean()) for s in np.unique(g)})


def bootstrap_ci(v, n=5000, seed=0):
    rng = np.random.RandomState(seed); v = np.asarray(v)
    m = rng.choice(v, (n, len(v)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(task, feat, preds, chance):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    rows, subj = [], {}
    for k, (yt, yp, g) in preds.items():
        ps = per_subject_acc(yt, yp, g); subj[k] = ps
        lo, hi = bootstrap_ci(ps.values)
        p = wilcoxon(ps.values - chance).pvalue if len(ps) >= 5 and np.any(ps.values != chance) else np.nan
        rows.append(dict(task=task, features=feat, evaluation=SCHEME_NAME[k], accuracy=float((yt == yp).mean()),
                         balanced_accuracy=balanced_accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro"),
                         n_subjects=len(ps), subj_mean=ps.mean(), subj_ci95_lo=lo, subj_ci95_hi=hi,
                         subj_min=ps.min(), subj_max=ps.max(), p_vs_chance=p, chance=chance))
    return rows, subj


def extra_analyses(recs, out):
    from sklearn.ensemble import RandomForestClassifier
    rf = lambda: RandomForestClassifier(N_TREES, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)
    R = {}
    # ---------- 1. channel QC ----------
    chq = channel_qc(recs)
    nbad = chq.groupby(["subject", "cond"]).bad.sum().rename("n_bad").reset_index()
    by_ch = chq.groupby("channel").bad.mean().reindex([str(c) for c in recs[0]["chans"]]).rename("bad_fraction")
    excluded = nbad[nbad.n_bad > MAX_BAD_CH]
    print(f"  [QC] bad channels per recording: median {nbad.n_bad.median():.0f}, max {nbad.n_bad.max()}, "
          f"recordings with >{MAX_BAD_CH} bad channels (excluded from cleaned analyses): {len(excluded)}")
    print("  [QC] most frequently bad channels: " +
          ", ".join(f"{c} {v:.0%}" for c, v in by_ch.sort_values(ascending=False).head(5).items()))
    D = build_clean(recs); Dc = build_clean(recs, artifact_free=True)
    raw_af = float(np.mean(np.concatenate([np.asarray(r["ptp"]) <= ARTIFACT_PTP_UV for r in recs])))
    cln_af = float(np.mean(np.concatenate([cleaned_record(r)[1] <= ARTIFACT_PTP_UV
                                           for r in recs if r["bad"].sum() <= MAX_BAD_CH])))
    print(f"  [QC] artifact-free epochs: {raw_af:.1%} before -> {cln_af:.1%} after excluding bad channels")
    R.update(chq=chq, nbad=nbad, by_ch=by_ch, excluded=excluded, af=(raw_af, cln_af))

    # ---------- 2. three-class and binary tasks (QC-cleaned features) ----------
    rows, subj = [], {}
    tasks = {"3-class": D["labels"]}
    labs = D["labels"]
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            tasks[f"{labs[i]} vs {labs[j]}"] = [labs[i], labs[j]]
    for tname, cls in tasks.items():
        for fname, src in (("QC-cleaned band power (RF)", D), ("QC-cleaned, artifact-free epochs (RF)", Dc)):
            if src is None:
                continue
            k = np.isin(src["y"], cls)
            if len(np.unique(src["y"][k])) < len(cls) or k.sum() < 200:
                continue
            t0 = time.time()
            preds = run_schemes(src["X"][k], src["y"][k], src["g"][k], src["ts"][k], src["dur"][k], rf)
            rr, ss = summarize(tname, fname, preds, 1 / len(cls))
            rows += rr; subj[(tname, fname)] = ss
            print(f"  {tname:24s} | {fname:38s}: " + "  ".join(f"{x['evaluation'][:3]}={x['accuracy']:.3f}" for x in rr)
                  + f"  [{time.time()-t0:.0f}s]")
    tasks_df = pd.DataFrame(rows)
    # paired comparison of evaluation schemes within subjects
    pair_rows = []
    for (tname, fname), ss in subj.items():
        for k1, k2 in (("a", "c"), ("a", "b"), ("b", "c")):
            if k1 in ss and k2 in ss:
                common = ss[k1].index.intersection(ss[k2].index)
                d = ss[k1][common] - ss[k2][common]
                p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
                pair_rows.append(dict(task=tname, features=fname, comparison=f"{SCHEME_NAME[k1]} - {SCHEME_NAME[k2]}",
                                      n_subjects=len(common), mean_difference=d.mean(), median_difference=d.median(),
                                      n_positive=int((d > 0).sum()), p_wilcoxon=p))
    R.update(tasks=tasks_df, pairs=pd.DataFrame(pair_rows), subj=subj)

    # ---------- 3. fingerprint controls ----------
    frows = []
    # 3a. pseudo-labels: three consecutive time blocks within the same recording (no task difference)
    for c in D["labels"]:
        k = D["y"] == c
        X, g, ts, dur = D["X"][k], D["g"][k], D["ts"][k], D["dur"][k]
        bl = dur / 3
        b0, b1 = np.floor(ts / bl), np.floor((ts + EPOCH_SEC - 1e-9) / bl)
        m = (b0 == b1) & (b0 < 3)
        yb = np.array([f"block{int(v) + 1}" for v in b0[m]])
        preds = run_schemes(X[m], yb, g[m], ts[m], dur[m], rf, schemes=("a", "c"))
        rr, _ = summarize(f"time-block pseudo-labels ({c})", "QC-cleaned band power (RF)", preds, 1 / 3)
        frows += rr
        print(f"  pseudo-labels ({c}): " + "  ".join(f"{x['evaluation'][:3]}={x['accuracy']:.3f}" for x in rr))
    # 3b. subject identification from a single condition (chance = 1 / n_subjects)
    for c in D["labels"]:
        k = D["y"] == c
        ys = D["g"][k]
        if len(np.unique(ys)) < 5:
            continue
        preds = run_schemes(D["X"][k], ys, ys, D["ts"][k], D["dur"][k], rf, schemes=("a", "b"))
        from sklearn.metrics import balanced_accuracy_score
        for kk, (yt, yp, gg) in preds.items():
            frows.append(dict(task=f"subject identification ({c})", features="QC-cleaned band power (RF)",
                              evaluation=SCHEME_NAME[kk], accuracy=float((yt == yp).mean()),
                              balanced_accuracy=balanced_accuracy_score(yt, yp), n_subjects=len(np.unique(ys)),
                              chance=1 / len(np.unique(ys))))
        print(f"  subject identification ({c}): " + "  ".join(
            f"{SCHEME_NAME[kk][:3]}={(yt == yp).mean():.3f}" for kk, (yt, yp, gg) in preds.items())
              + f"  (chance {1/len(np.unique(ys)):.3f})")
    R["finger"] = pd.DataFrame(frows)
    return R


def extra_figures(R, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    # QC
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    R["by_ch"].mul(100).plot.bar(ax=ax[0], color="#4c72b0")
    ax[0].set(ylabel="% of recordings flagged bad", title="(A) Bad-channel frequency by electrode", xlabel="")
    vc = R["nbad"].n_bad.value_counts().sort_index()
    ax[1].bar(vc.index, vc.values, color="#55a868"); ax[1].axvline(MAX_BAD_CH + .5, ls="--", c="k", lw=.8)
    ax[1].set(xlabel="Bad channels per recording", ylabel="Recordings", title="(B) Bad channels per recording")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_channel_qc.png"), dpi=200); plt.close(fig)
    # tasks
    t = R["tasks"]; t = t[t.features == "QC-cleaned band power (RF)"]
    if len(t):
        piv = t.pivot(index="task", columns="evaluation", values="accuracy")
        order = [x for x in ["3-class"] + sorted(i for i in piv.index if i != "3-class")]
        piv = piv.reindex(order)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        piv.plot.bar(ax=ax, width=.8, rot=0)
        for i, task in enumerate(piv.index):
            ch = 1 / 3 if task == "3-class" else .5
            ax.hlines(ch, i - .4, i + .4, colors="k", linestyles="--", lw=.8)
        ax.set(ylim=(0, 1), ylabel="Accuracy", xlabel="", title="Accuracy by task and evaluation scheme (dashed = chance)")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_tasks_by_evaluation.png"), dpi=200); plt.close(fig)
    # fingerprint
    f = R["finger"]
    if len(f):
        fig, ax = plt.subplots(figsize=(10, 4.2))
        piv = f.pivot(index="task", columns="evaluation", values="accuracy")
        piv.plot.barh(ax=ax, width=.75)
        for i, task in enumerate(piv.index):
            ax.vlines(f[f.task == task].chance.iloc[0], i - .4, i + .4, colors="k", linestyles="--", lw=.8)
        ax.set(xlim=(0, 1), xlabel="Accuracy", ylabel="", title="Fingerprint controls (dashed = chance)")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig6_fingerprint_controls.png"), dpi=200); plt.close(fig)
    # per-subject accuracy, 3-class
    key = ("3-class", "QC-cleaned band power (RF)")
    if key in R["subj"]:
        ss = R["subj"][key]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ks = [k for k in ("a", "b", "c") if k in ss]
        ax.boxplot([ss[k].values for k in ks], widths=.5, showfliers=False)
        rng = np.random.RandomState(0)
        for i, k in enumerate(ks, 1):
            ax.scatter(i + rng.uniform(-.12, .12, len(ss[k])), ss[k].values, s=10, alpha=.6)
        common = ss["a"].index.intersection(ss["c"].index) if "a" in ss and "c" in ss else []
        for s_ in common:
            ax.plot([ks.index("a") + 1, ks.index("c") + 1], [ss["a"][s_], ss["c"][s_]], c="grey", lw=.3, alpha=.4)
        ax.axhline(1 / 3, ls="--", c="k", lw=.8)
        ax.set_xticks(range(1, len(ks) + 1)); ax.set_xticklabels([SCHEME_NAME[k] for k in ks], fontsize=8)
        ax.set(ylim=(0, 1), ylabel="Per-subject accuracy", title="3-class, QC-cleaned features (RF)")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig7_per_subject_accuracy.png"), dpi=200); plt.close(fig)


def extra_summary(R):
    t = R["tasks"]
    cols = ["task", "features", "evaluation", "accuracy", "balanced_accuracy", "subj_mean", "subj_ci95_lo",
            "subj_ci95_hi", "p_vs_chance", "chance"]
    raw_af, cln_af = R["af"]
    return ["", "## Channel quality control", "",
            f"- Bad-channel rule: flat (SD < {BAD_CH_FLAT_UV} µV), > {BAD_CH_EPOCH_FRAC:.0%} of epochs above "
            f"{ARTIFACT_PTP_UV:.0f} µV, or robust z(log SD) > {BAD_CH_ROBUST_Z}",
            f"- Bad channels per recording: median {R['nbad'].n_bad.median():.0f}, max {R['nbad'].n_bad.max()}; "
            f"recordings excluded from cleaned analyses (> {MAX_BAD_CH} bad): {len(R['excluded'])}",
            f"- Artifact-free epochs: {raw_af:.1%} (all channels) -> {cln_af:.1%} (good channels only)", "",
            "Bad-channel fraction by electrode:", "", R["by_ch"].round(3).to_string(), "",
            "## Tasks by evaluation scheme (QC-cleaned; per-subject mean with bootstrap 95% CI; Wilcoxon vs chance)", "",
            t[cols].round(4).to_string(index=False) if len(t) else "(none)", "",
            "## Paired comparison of evaluation schemes within subjects (Wilcoxon signed-rank)", "",
            R["pairs"].round(4).to_string(index=False) if len(R["pairs"]) else "(none)", "",
            "## Fingerprint controls", "",
            R["finger"][[c for c in ["task", "evaluation", "accuracy", "balanced_accuracy", "n_subjects", "chance"]
                         if c in R["finger"]]].round(4).to_string(index=False) if len(R["finger"]) else "(none)", "",
            "Interpretation: time-block pseudo-labels carry no task information. High random-split accuracy for them "
            "shows that random splitting rewards temporal proximity within a recording. High subject-identification "
            "accuracy shows strong individual signatures, which inflate any evaluation that mixes a subject's data "
            "across training and test sets."]


# ------------------------------ figures ------------------------------
def figures(recs, qc, bp, res, cms, labels, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    conds = sorted({str(r["cond"]) for r in recs})
    chans = [str(c) for c in recs[0]["chans"]]
    idx = [chans.index(c) for c in ("O1", "O2") if c in chans]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for c in conds:
        ps = np.array([r["psd"][:, idx].mean(1) for r in recs if str(r["cond"]) == c])
        f = recs[0]["f"]; m = (f >= 1) & (f <= 45)
        ax[0].semilogy(f[m], np.exp(np.log(ps[:, m] + 1e-12).mean(0)), label=f"{c} (n={len(ps)})")
    ax[0].axvspan(8, 13, color="grey", alpha=.12)
    ax[0].set(title="(A) Occipital PSD (O1/O2, geometric mean)", xlabel="Frequency (Hz)", ylabel="µV²/Hz"); ax[0].legend(fontsize=7)
    if {"eye_open", "eye_close"} <= set(conds):
        d = bp[(bp.band == "alpha") & bp.channel.isin(OCCIPITAL)].groupby(["subject", "cond"]).rel_power.mean().unstack()
        d = d[["eye_open", "eye_close"]].dropna()
        for _, row in d.iterrows():
            ax[1].plot([0, 1], row.values, color="grey", lw=.7, marker="o", ms=2.5)
        ax[1].set(xticks=[0, 1], xticklabels=["Eyes open", "Eyes closed"], xlim=(-.3, 1.3),
                  ylabel="Relative alpha (8–13 Hz)", title=f"(B) Alpha reactivity: {(d.eye_close > d.eye_open).sum()}/{len(d)} ↑")
    ax[2].boxplot([qc[qc.cond == c].head_acc_rms_mg for c in conds]); ax[2].set_xticks(range(1, len(conds) + 1)); ax[2].set_xticklabels(conds)
    ax[2].set(ylabel="Head accelerometer RMS (mg)", title="(C) Head motion by condition")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_spectra_alpha_motion.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    piv = res.pivot(index="features", columns="evaluation", values="accuracy")
    piv.plot.barh(ax=ax, width=.8)
    ax.axvline(1 / len(labels), ls="--", color="k", lw=.8, label="chance")
    ax.set(xlabel="Accuracy", xlim=(0, 1), ylabel="", title="Classification accuracy by evaluation scheme")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_classification_by_evaluation.png"), dpi=200); plt.close(fig)

    key = "EEG band power (RF)"
    if key in cms:
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        for a, (name, cm) in zip(axs, cms[key].items()):
            cmn = cm / cm.sum(1, keepdims=True)
            a.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
            for i in range(len(cm)):
                for j in range(len(cm)):
                    a.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                           color="white" if cmn[i, j] > .6 else "black")
            a.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
                  title=name, xlabel="Predicted", ylabel="True")
        fig.suptitle(key); fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_confusion_matrices.png"), dpi=200); plt.close(fig)


def write_summary(out, inv, qc, stats, res, labels, extra=None):
    L = ["# EEG analysis summary", "",
         f"- Recordings: {len(qc)} | subjects: {qc.subject.nunique()} | conditions: {', '.join(labels)}",
         f"- Median duration: {qc.duration_s.median():.1f} s | sampling rate: {FS} Hz", "",
         "## Quality control (mean per condition)", "",
         qc.groupby("cond")[["artifact_epoch_frac", "median_max_ptp_uV", "line_noise_ratio", "head_acc_rms_mg"]]
         .mean().round(3).to_string(), "",
         f"artifact_epoch_frac = fraction of 2-s epochs with peak-to-peak > {ARTIFACT_PTP_UV:.0f} µV in any channel.", "",
         "## Paired statistics (Wilcoxon signed-rank)", "",
         stats.round(4).to_string(index=False) if len(stats) else "(not enough paired subjects)", "",
         "## Classification", "", res.round(3).to_string(index=False) if len(res) else "(skipped)", "",
         "Interpretation guide: a large drop from (a) random split to (b)/(c) indicates that random-split accuracy "
         "is inflated by temporally adjacent, highly correlated windows appearing in both training and test sets. "
         "(c) cross-subject is the relevant estimate for a BCI used by new users. "
         "High accelerometer-only accuracy indicates that motion, not neural activity, can separate the classes."]
    if extra:
        L += extra
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L))


# -------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only inventory files")
    ap.add_argument("--peek", action="store_true", help="show file format detection for one file per condition")
    ap.add_argument("--scan", action="store_true", help="read every file and check duration/plausibility (no analysis)")
    ap.add_argument("--scan-dir", type=str, default=None, help="scan all files in this folder instead (e.g. COHORT22_DIR)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes for feature extraction")
    ap.add_argument("--no-clf", action="store_true", help="skip classification")
    ap.add_argument("--external", action="store_true", help="also run EEGMMIDB external validation")
    ap.add_argument("--no-extra", action="store_true", help="skip QC / binary / fingerprint / per-subject analyses")
    ap.add_argument("--trees", type=int, default=None, help="random-forest size (default N_TREES)")
    ap.add_argument("--policy", choices=["exclude", "common"], default=None,
                    help="channel policy for recordings missing channels (overrides CHANNEL_POLICY)")
    ap.add_argument("--step", type=float, default=None,
                    help="window step in s (default 1 = 50%% overlap; 2 = non-overlapping windows)")
    a = ap.parse_args()
    global N_TREES, CHANNEL_POLICY, OUT_DIR, STEP_SEC
    if a.trees:
        N_TREES = a.trees
    if a.policy:
        CHANNEL_POLICY = a.policy
    if a.step:
        STEP_SEC = a.step
    OUT_DIR = os.path.join(OUT_DIR_BASE, f"policy_{CHANNEL_POLICY}" + ("" if STEP_SEC == 1.0 else f"_step{STEP_SEC:g}s"))
    print(f"Channel policy: {CHANNEL_POLICY} | window step {STEP_SEC:g} s "
          f"({100 * max(0.0, 1 - STEP_SEC / EPOCH_SEC):.0f}% overlap)  ->  results in {OUT_DIR}")
    import scipy, sklearn, matplotlib, sys
    print(f"Python {sys.version.split()[0]} | numpy {np.__version__} | pandas {pd.__version__} | scipy {scipy.__version__}"
          f" | scikit-learn {sklearn.__version__} | matplotlib {matplotlib.__version__}")

    os.makedirs(OUT_DIR, exist_ok=True); cache_dir = os.path.join(OUT_DIR, "cache"); os.makedirs(cache_dir, exist_ok=True)
    if a.scan_dir:
        d = COHORT22_DIR if a.scan_dir.lower() == "cohort22" else a.scan_dir
        print(f"Scanning folder: {d}  (exists: {os.path.isdir(d)})")
        files = sorted({f for pat in FILE_GLOB for f in glob.glob(os.path.join(d, pat))})
        scan(pd.DataFrame(dict(cond="-", path=files, subject=[subject_of(f) for f in files])),
             os.path.join(OUT_DIR, "scan_" + re.sub(r"[^\w]", "_", os.path.basename(d.rstrip("\\/"))) + ".csv"))
        return
    inv = inventory(); tab = report_inventory(inv)
    tab.to_csv(os.path.join(OUT_DIR, "file_inventory_by_subject.csv"))
    if a.peek:
        peek(inv); return
    if a.scan:
        scan(inv, os.path.join(OUT_DIR, "scan_files.csv")); return
    if a.check:
        return

    jobs = [(r.path, r.cond, r.subject, cache_dir, STEP_SEC) for r in inv.itertuples()]
    print(f"\n[1/4] Feature extraction ({len(jobs)} files, {a.jobs} process(es); cached files are skipped)")
    caches, t0 = [], time.time()
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            for i, c in enumerate(ex.map(process_file, jobs), 1):
                caches.append(c); print(f"\r  {i}/{len(jobs)}  {time.time()-t0:.0f}s", end="")
    else:
        for i, j in enumerate(jobs, 1):
            try:
                caches.append(process_file(j))
            except Exception as e:
                print(f"\n  [ERROR] {j[0]}: {e}")
            print(f"\r  {i}/{len(jobs)}  {time.time()-t0:.0f}s", end="")
    print()
    recs = load_all(caches)
    recs, keep, target = harmonize_channels(recs, caches)
    with open(os.path.join(OUT_DIR, "cache_manifest.json"), "w") as f:   # caches + channel set of this run
        json.dump(dict(files=[os.path.basename(c) for c in keep], chans=target, policy=CHANNEL_POLICY), f, indent=1)

    print("[2/4] QC, band power, statistics")
    qc = qc_table(recs); bp = bandpower_table(recs); stats = statistics(bp)
    print(qc.groupby("cond")[["artifact_epoch_frac", "head_acc_rms_mg"]].mean().round(3).to_string())
    if len(stats):
        print(stats[["test", "n", "median_ratio", "n_increase", "p_wilcoxon"]].round(4).to_string(index=False))

    res, cms, persub, labels = pd.DataFrame(), {}, pd.DataFrame(), sorted(CONDITIONS)
    if not a.no_clf:
        print("[3/4] Classification")
        res, cms, persub, labels = classification(recs, OUT_DIR)
    R = None
    if not a.no_clf and not a.no_extra:
        print("[3b] Channel QC, binary tasks, fingerprint controls, per-subject statistics")
        R = extra_analyses(recs, OUT_DIR)

    print("[4/4] Writing outputs")
    with pd.ExcelWriter(os.path.join(OUT_DIR, "results.xlsx")) as xw:
        qc.to_excel(xw, sheet_name="QC", index=False)
        bp.groupby(["cond", "band", "channel"])[["abs_power", "rel_power"]].mean().reset_index().to_excel(xw, sheet_name="BandPower_mean", index=False)
        stats.to_excel(xw, sheet_name="Statistics", index=False)
        if len(res):
            res.to_excel(xw, sheet_name="Classification", index=False); persub.to_excel(xw, sheet_name="CrossSubject_perSubject")
        if R is not None:
            R["chq"].to_excel(xw, sheet_name="ChannelQC", index=False)
            R["nbad"].to_excel(xw, sheet_name="BadChannelsPerRecording", index=False)
            R["tasks"].to_excel(xw, sheet_name="Tasks_by_evaluation", index=False)
            R["pairs"].to_excel(xw, sheet_name="Scheme_paired_tests", index=False)
            R["finger"].to_excel(xw, sheet_name="Fingerprint_controls", index=False)
            ps = []
            for (tn, fn), ss in R["subj"].items():
                for k, v in ss.items():
                    ps.append(v.rename("accuracy").to_frame().assign(task=tn, features=fn, evaluation=SCHEME_NAME[k]))
            if ps:
                pd.concat(ps).rename_axis("subject").reset_index().to_excel(xw, sheet_name="PerSubject_accuracy", index=False)
    bp.to_csv(os.path.join(OUT_DIR, "bandpower_long.csv"), index=False)
    if len(res):
        figures(recs, qc, bp, res, cms, labels, OUT_DIR)
    if R is not None:
        extra_figures(R, OUT_DIR)
    write_summary(OUT_DIR, inv, qc, stats, res, labels, extra=extra_summary(R) if R is not None else None)
    print(f"Done -> {OUT_DIR}")
    if a.external:
        import external_validation
        external_validation.run(jobs=a.jobs, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
