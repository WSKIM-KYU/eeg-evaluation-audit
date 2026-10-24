# -*- coding: utf-8 -*-
"""
EEGNet robustness check: does a raw-signal CNN show the same evaluation-scheme effect as the
band-power random forest?  (Supplementary analysis for the BSPC manuscript)

Same data, channels, channel QC and evaluation schemes as eeg_analysis.py:
    (a) random split (5 stratified folds over all windows)
    (b) within-recording time split (first 60 % train, test from 60 % + 2 s)
    (c) cross-subject (10 folds by participant)
Early-stopping validation sets follow the same rule as each scheme (for (c): held-out
training PARTICIPANTS), so no extra leakage is introduced by model selection.

Install PyTorch once (RTX 30xx, CUDA 12.x driver):
    pip install torch --index-url https://download.pytorch.org/whl/cu126

Run (after eeg_analysis.py, from the same folder):
    python eegnet_validation.py --quick          # 2-epoch smoke test (a few minutes)
    python eegnet_validation.py                  # full run (about 1-2 h on an RTX 3080)
Options: --policy common|exclude  --tasks "3-class,eye_open vs hands_up"  --epochs 100
         --patience 10  --no-qc  --max-folds N  --dry-run (pipeline test without PyTorch)

Outputs: <policy folder>\\eegnet\\  eegnet_results.csv, eegnet_per_subject.csv,
         eegnet_paired_tests.csv, summary_eegnet.md, FigS2_eegnet_vs_rf.(pdf|png)
"""
import os, sys, json, time, copy, argparse, warnings
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eeg_analysis as EA                     # shared paths, reader, preprocessing, QC, statistics

# =============================== CONFIG ===============================
FS_NET = 128                 # EEGNet's standard input rate (500 Hz -> 128 Hz: up 32, down 125)
WIN_SEC, STEP_SEC = 2.0, 1.0  # identical to the band-power analysis
SCALE_UV, CLIP = 50.0, 20.0  # input scaling: x / 50 uV, clipped to +-20 (= +-1000 uV)
TASKS_DEFAULT = "3-class,eye_open vs hands_up"
SEED = 0
# ======================================================================


# ------------------------------ data ---------------------------------
def qc_table(out):
    """Bad channels and excluded recordings from the band-power analysis of the same policy."""
    man = json.load(open(os.path.join(out, "cache_manifest.json")))
    recs = EA.load_all([os.path.join(out, "cache", n) for n in man["files"]])
    recs = [EA.reorder_record(r, man["chans"]) for r in recs]
    EA.channel_qc(recs)
    info = {}
    for r in recs:
        bad = [str(c) for c, b in zip(r["chans"], r["bad"]) if b]
        info[(str(r["subject"]), str(r["cond"]))] = dict(bad=bad, excluded=int(r["bad"].sum()) > EA.MAX_BAD_CH)
    return man["chans"], info


def interpolate_bad(x, chans, bad):
    """Replace bad channels by the mean of their good neighbours (same rule as the feature-level QC)."""
    if not bad:
        return x
    norm = [EA.norm_ch(c) for c in chans]
    badn = {EA.norm_ch(b) for b in bad}
    good = [i for i, c in enumerate(norm) if c not in badn]
    x = x.copy()
    for i, c in enumerate(norm):
        if c in badn:
            nb = [norm.index(n) for n in EA.NEIGHBORS.get(c, []) if n in norm and n not in badn] or good
            x[:, i] = x[:, nb].mean(1)
    return x


def prepare(out, use_qc, cache_path):
    if os.path.exists(cache_path):
        try:
            z = np.load(cache_path, allow_pickle=False)
            D = {k: z[k] for k in z.files}
            z.close()
            print(f"  loaded cached windows: {cache_path}")
            return D
        except Exception as e:                       # empty / truncated file from an interrupted run
            print(f"  [WARN] cached windows unreadable ({type(e).__name__}: {e}) -> rebuilding")
            os.remove(cache_path)
    target, info = qc_table(out)
    inv = EA.inventory()
    X, y, g, ts, dur = [], [], [], [], []
    W, S = int(WIN_SEC * FS_NET), int(STEP_SEC * FS_NET)
    t0, n_skip = time.time(), 0
    for i, r in enumerate(inv.itertuples(), 1):
        key = (str(r.subject), str(r.cond))
        if key not in info or (use_qc and info[key]["excluded"]):
            n_skip += 1
            continue
        eeg, _, chans = EA.read_recording(r.path)
        norm = [EA.norm_ch(c) for c in chans]
        idx = [norm.index(EA.norm_ch(c)) for c in target]
        x = EA.preprocess(eeg[:, idx])                      # 500 Hz: notch, 1-45 Hz band-pass, edge trim
        if use_qc:
            x = interpolate_bad(x, target, info[key]["bad"])
        x = resample_poly(x, 32, 125, axis=0)               # 500 -> 128 Hz
        x = np.clip(x / SCALE_UV, -CLIP, CLIP).astype(np.float32)
        starts = np.arange(0, len(x) - W + 1, S)
        X.append(np.stack([x[s:s + W].T for s in starts]))
        y += [r.cond] * len(starts); g += [str(r.subject)] * len(starts)
        ts.append(starts / FS_NET); dur += [len(x) / FS_NET] * len(starts)
        print(f"\r  extracting windows {i}/{len(inv)}  {time.time()-t0:.0f}s", end="", flush=True)
    print(f"\n  recordings used: {len(inv) - n_skip}, skipped: {n_skip}")
    D = dict(X=np.concatenate(X), y=np.array(y), g=np.array(g), ts=np.concatenate(ts), dur=np.array(dur),
             chans=np.array(target))
    tmp = cache_path[:-4] + "_partial.npz"           # write to a temporary file, then rename (atomic)
    np.savez(tmp, **D)
    os.replace(tmp, cache_path)
    print(f"  windows: {D['X'].shape} (windows, channels, samples) -> {cache_path}")
    return D


# ------------------------------ splits -------------------------------
def split_indices(scheme, y, g, ts, dur, max_folds=None):
    """Yield (train, validation, test) index arrays for one evaluation scheme."""
    from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split, GroupShuffleSplit
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(y))
    folds = []
    if scheme == "a":
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(idx, y):
            tr2, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=SEED)
            folds.append((tr2, va, te))
    elif scheme == "b":
        split = 0.6 * dur
        tr = idx[(ts + WIN_SEC) <= split]
        te = idx[ts >= split + WIN_SEC]
        tr2, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=SEED)
        folds.append((tr2, va, te))
    else:
        for tr, te in GroupKFold(min(10, len(np.unique(g)))).split(idx, y, g):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
            a, b = next(gss.split(tr, y[tr], g[tr]))       # validation = held-out training participants
            folds.append((tr[a], tr[b], te))
    return folds[:max_folds] if max_folds else folds


# ------------------------------ model --------------------------------
def build_eegnet(n_ch, n_times, n_classes, F1=8, D=2, F2=16, k1=64, drop=0.5):
    """EEGNet-8,2 (Lawhern et al., 2018, J. Neural Eng. 15:056013)."""
    import torch
    import torch.nn as nn

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
                nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),      # depthwise spatial filter
                nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
            self.block2 = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # separable
                nn.Conv2d(F1 * D, F2, 1, bias=False),
                nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
            with torch.no_grad():
                n = self.block2(self.block1(torch.zeros(1, 1, n_ch, n_times))).numel()
            self.fc = nn.Linear(n, n_classes)

        def forward(self, x):                      # x: (batch, channels, samples)
            return self.fc(self.block2(self.block1(x.unsqueeze(1))).flatten(1))

    return EEGNet()


def to_tensor(a, dtype, device):
    """numpy -> torch without the numpy bridge (works even if torch and numpy versions are mismatched)."""
    import torch
    a = np.ascontiguousarray(a)
    if a.size == 0:
        return torch.empty(a.shape, dtype=dtype, device=device)
    return torch.frombuffer(bytearray(a.tobytes()), dtype=dtype).reshape(a.shape).to(device)


def idx_tensor(ix, device):
    import torch
    return to_tensor(np.asarray(ix, dtype=np.int64), torch.int64, device)


def fit_predict_eegnet(Xd, Yd, tr, va, te, n_classes, args, device):
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = build_eegnet(Xd.shape[1], Xd.shape[2], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    counts = np.array(torch.bincount(Yd[idx_tensor(tr, device)], minlength=n_classes).cpu().tolist(), dtype=float)
    w = torch.tensor((len(tr) / (n_classes * np.maximum(counts, 1))).tolist(), dtype=torch.float32, device=device)
    lossf = nn.CrossEntropyLoss(weight=w)
    bs, rng = args.batch, np.random.RandomState(SEED)
    va_t = idx_tensor(va, device)

    def batched_eval(ix):
        model.eval(); outs = []
        with torch.no_grad():
            for b in range(0, len(ix), 1024):
                j = idx_tensor(ix[b:b + 1024], device)
                outs.append(model(Xd[j]))
        return torch.cat(outs)

    best, best_state, wait, ep = np.inf, None, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(tr)
        for b in range(0, len(perm), bs):
            j = idx_tensor(perm[b:b + bs], device)
            opt.zero_grad()
            loss = lossf(model(Xd[j]), Yd[j])
            loss.backward()
            opt.step()
        vloss = lossf(batched_eval(va), Yd[va_t]).item()
        if vloss < best - 1e-4:
            best, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    return np.array(batched_eval(te).argmax(1).cpu().tolist()), ep


def fit_predict_dry(Xn, Yn, tr, va, te, n_classes, args, device):
    """Pipeline test without PyTorch: logistic regression on log-variance per channel."""
    from sklearn.linear_model import LogisticRegression
    F = np.log(Xn.var(axis=2) + 1e-6)
    m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(F[tr], Yn[tr])
    return m.predict(F[te]), 0


# ------------------------------ main ---------------------------------
def run(args):
    policy = args.policy or EA.CHANNEL_POLICY
    out = os.path.join(EA.OUT_DIR_BASE, f"policy_{policy}")
    res_dir = os.path.join(out, "eegnet"); os.makedirs(res_dir, exist_ok=True)
    print(f"Policy: {policy}  |  results: {res_dir}")
    device = None
    if not args.dry_run:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch {torch.__version__} | device: {device}"
              + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "  [WARN] no GPU: slow"))
        torch.backends.cudnn.benchmark = True

    print("[1/3] Preparing raw-signal windows")
    D = prepare(out, not args.no_qc, os.path.join(out, f"eegnet_windows_{'qc' if not args.no_qc else 'raw'}.npz"))
    labels_all = sorted(np.unique(D["y"]))
    tasks = {}
    for t in [s.strip() for s in args.tasks.split(",") if s.strip()]:
        tasks[t] = labels_all if t == "3-class" else [c.strip() for c in t.split(" vs ")]

    print("[2/3] Training and evaluation")
    rows, subj_rows, pair_rows = [], [], []
    for tname, cls in tasks.items():
        k = np.isin(D["y"], cls)
        Xn, yn, gn, tsn, dn = D["X"][k], D["y"][k], D["g"][k], D["ts"][k], D["dur"][k]
        Yn = np.array([cls.index(v) for v in yn])
        if not args.dry_run:
            import torch
            Xd = to_tensor(Xn.astype(np.float32), torch.float32, device)
            Yd = to_tensor(Yn.astype(np.int64), torch.int64, device)
        preds = {}
        for sch in ("a", "b", "c"):
            folds = split_indices(sch, yn, gn, tsn, dn, args.max_folds)
            yp = np.full(len(yn), -1)
            for fi, (tr, va, te) in enumerate(folds, 1):
                t0 = time.time()
                if args.dry_run:
                    p, ep = fit_predict_dry(Xn, Yn, tr, va, te, len(cls), args, device)
                else:
                    p, ep = fit_predict_eegnet(Xd, Yd, tr, va, te, len(cls), args, device)
                yp[te] = p
                print(f"  {tname:22s} {EA.SCHEME_NAME[sch]:34s} fold {fi}/{len(folds)}: "
                      f"acc {np.mean(p == Yn[te]):.3f}  epochs {ep}  [{time.time()-t0:.0f}s]", flush=True)
            m = yp >= 0
            preds[sch] = (yn[m], np.array(cls)[yp[m]], gn[m])
        rr, ss = EA.summarize(tname, "EEGNet (raw signal)" if not args.dry_run else "dry-run LR", preds,
                              1 / len(cls))
        rows += rr
        for sch, s in ss.items():
            for sub, acc in s.items():
                subj_rows.append(dict(task=tname, evaluation=EA.SCHEME_NAME[sch], subject=sub, accuracy=acc))
        for k1, k2 in (("a", "c"), ("a", "b"), ("b", "c")):
            common = ss[k1].index.intersection(ss[k2].index)
            d = ss[k1][common] - ss[k2][common]
            p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
            pair_rows.append(dict(task=tname, comparison=f"{EA.SCHEME_NAME[k1]} - {EA.SCHEME_NAME[k2]}",
                                  n_subjects=len(common), mean_difference=d.mean(), n_positive=int((d > 0).sum()),
                                  p_wilcoxon=p))
        for r in rr:
            print(f"  => {tname}: {r['evaluation']}: accuracy {r['accuracy']:.3f}, per-subject mean "
                  f"{r['subj_mean']:.3f} [{r['subj_ci95_lo']:.3f}, {r['subj_ci95_hi']:.3f}]")

    print("[3/3] Writing outputs")
    res, subj, pairs = pd.DataFrame(rows), pd.DataFrame(subj_rows), pd.DataFrame(pair_rows)
    res.to_csv(os.path.join(res_dir, "eegnet_results.csv"), index=False)
    subj.to_csv(os.path.join(res_dir, "eegnet_per_subject.csv"), index=False)
    pairs.to_csv(os.path.join(res_dir, "eegnet_paired_tests.csv"), index=False)
    rf = None
    try:
        t = pd.read_excel(os.path.join(out, "results.xlsx"), sheet_name="Tasks_by_evaluation")
        rf = t[t.features == ("QC-cleaned band power (RF)" if not args.no_qc else "QC-cleaned band power (RF)")]
    except Exception as e:
        print(f"  [info] RF comparison not available ({e})")
    figure(res, rf, res_dir, args)
    cols = ["task", "evaluation", "accuracy", "balanced_accuracy", "subj_mean", "subj_ci95_lo", "subj_ci95_hi",
            "p_vs_chance", "chance"]
    L = ["# EEGNet robustness check", "",
         f"- Policy: {policy}; channels: {len(D['chans'])}; windows: {len(D['y'])}; participants: {len(np.unique(D['g']))}",
         f"- Input: {FS_NET} Hz, {WIN_SEC:.0f}-s windows ({STEP_SEC:.0f}-s step), "
         f"{'QC-cleaned (bad channels interpolated, recordings with > %d bad channels excluded)' % EA.MAX_BAD_CH if not args.no_qc else 'no channel QC'}",
         f"- EEGNet-8,2; Adam (lr 1e-3), batch {args.batch}, max {args.epochs} epochs, early stopping "
         f"(patience {args.patience}) on a scheme-consistent validation set; class-balanced loss",
         "" if not args.max_folds else f"- [partial run: first {args.max_folds} fold(s) per scheme only]", "",
         "## Results", "", res[cols].round(4).to_string(index=False), "",
         "## Paired comparison of evaluation schemes (Wilcoxon signed-rank)", "",
         pairs.round(4).to_string(index=False)]
    if rf is not None and len(rf):
        L += ["", "## Random forest (band power) for comparison", "",
              rf[[c for c in cols if c in rf]].round(4).to_string(index=False)]
    open(os.path.join(res_dir, "summary_eegnet.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"Done -> {res_dir}")


def figure(res, rf, res_dir, args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    tasks = list(dict.fromkeys(res.task))
    names = {"3-class": "Three-class", "eye_open vs hands_up": "Eyes open vs arm elevation",
             "eye_close vs eye_open": "Eyes closed vs eyes open", "eye_close vs hands_up": "Eyes closed vs arm elevation"}
    sch = ["(a) random split", "(b) within-recording time split", "(c) cross-subject"]
    lab = ["Random\nsplit", "Time\nsplit", "Cross-\nsubject"]
    fig, axs = plt.subplots(1, len(tasks), figsize=(3.4 * len(tasks), 2.4), squeeze=False)
    for ax, t in zip(axs[0], tasks):
        w = 0.38
        for j, (src, name, col) in enumerate(((rf, "Random forest (band power)", "#999999"),
                                               (res, str(res.features.iloc[0]), "#0072B2"))):
            if src is None or not len(src[src.task == t]):
                continue
            s = src[src.task == t].set_index("evaluation").reindex(sch)
            x = np.arange(3) + (j - 0.5) * w
            ax.bar(x, s.subj_mean, w, color=col, label=name)
            ax.errorbar(x, s.subj_mean, yerr=[s.subj_mean - s.subj_ci95_lo, s.subj_ci95_hi - s.subj_mean],
                        fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5)
        ch = 1 / 3 if t == "3-class" else 0.5
        ax.axhline(ch, ls="--", color="black", lw=0.7)
        ax.set_xticks(range(3)); ax.set_xticklabels(lab)
        ax.set(ylim=(0, 1.02), title=names.get(t, t), ylabel="Accuracy (per-participant mean)")
    axs[0][0].legend(frameon=False, loc="lower left", fontsize=6)
    fig.tight_layout()
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 300})):
        fig.savefig(os.path.join(res_dir, "FigS2_eegnet_vs_rf" + ext), bbox_inches="tight", **kw)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["common", "exclude"], default="common")
    ap.add_argument("--tasks", default=TASKS_DEFAULT)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-folds", type=int, default=None, help="run only the first N folds per scheme")
    ap.add_argument("--no-qc", action="store_true", help="no channel interpolation / exclusion")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 epochs, 1 fold per scheme")
    ap.add_argument("--dry-run", action="store_true", help="test the pipeline with logistic regression (no PyTorch)")
    ap.add_argument("--figure-only", action="store_true", help="redraw Fig. S3 from saved results (no training)")
    a = ap.parse_args()
    if a.figure_only:
        out = os.path.join(EA.OUT_DIR_BASE, f"policy_{a.policy}")
        rd = os.path.join(out, "eegnet")
        res = pd.read_csv(os.path.join(rd, "eegnet_results.csv"))
        t = pd.read_excel(os.path.join(out, "results.xlsx"), sheet_name="Tasks_by_evaluation")
        figure(res, t[t.features == "QC-cleaned band power (RF)"], rd, a)
        print(f"Figure redrawn -> {os.path.join(rd, 'FigS2_eegnet_vs_rf.png')}")
        raise SystemExit
    if a.quick:
        a.epochs, a.patience, a.max_folds = 2, 2, 1
    run(a)
