# -*- coding: utf-8 -*-
"""
Fig. S2 (external-validation confusion matrices) drawn with Arial, matching the other figures.
Values are those of figE2_external_confusion (RF, log relative band power + subject-wise
standardisation, all windows; row-normalised).

    python make_figS2_arial.py
    python make_figS2_arial.py --out analysis_output\\policy_common\\figures
Writes FigS2_external_confusion.pdf / .png / .tif and reports every file actually written.
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

DEFAULT_OUT = os.path.join(os.environ.get("EEG_OUTPUT_DIR", "analysis_output"), "policy_common", "figures")
PANELS = [
    ("Cohort → EEGMMIDB\neyes open vs eyes closed", [[0.80, 0.20], [0.12, 0.88]],
     ["Eyes closed", "Eyes open"], ["Eyes closed", "Eyes open"], "True (test-set classes)"),
    ("EEGMMIDB → cohort\neyes open vs eyes closed", [[0.90, 0.10], [0.12, 0.88]],
     ["Eyes closed", "Eyes open"], ["Eyes closed", "Eyes open"], ""),
    ("Cohort → EEGMMIDB\nthree-class", [[0.79, 0.14, 0.07], [0.11, 0.44, 0.45], [0.07, 0.36, 0.57]],
     ["Eyes closed", "Eyes open", "Fist\nmovement"], ["Eyes\nclosed", "Eyes\nopen", "Arm\nelevation"],
     "True (EEGMMIDB classes)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    found = font_manager.findfont("Arial", fallback_to_default=False) if any(
        f.name == "Arial" for f in font_manager.fontManager.ttflist) else None
    print("Font:", found if found else "[WARN] Arial not found -> default font used")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                         "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    fig, axs = plt.subplots(1, 3, figsize=(7.48, 2.55), gridspec_kw={"width_ratios": [2, 2, 3]})
    for k, (ax, (title, m, rows, cols, ylab)) in enumerate(zip(axs, PANELS)):
        m = np.array(m)
        ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
        ax.set_xlabel("Predicted (training-set classes)"); ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=7.5)
        ax.text(-0.08, 1.02, "ABC"[k], transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
    fig.tight_layout(w_pad=2.2)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, "FigS2_external_confusion")
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600}), (".tif", {"dpi": 600})):
        path = base + ext
        try:
            fig.savefig(path, **kw)
            print(f"  written: {path}  ({os.path.getsize(path) / 1024:.0f} KB)")
        except Exception as e:
            print(f"  [ERROR] {ext}: {type(e).__name__}: {e}")
    plt.close(fig)


if __name__ == "__main__":
    main()
