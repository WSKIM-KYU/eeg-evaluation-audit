# EEG evaluation audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23608168.svg)](https://doi.org/10.5281/zenodo.23608168)

Analysis code for *"Random data splitting and window-level significance tests overstate evidence from EEG classification: dependence-preserving permutation tests, design comparisons and simulations"* (Cho & Kim).

The pipeline evaluates EEG classification under random-window, within-recording time and cross-subject splits; compares binomial, window-level permutation and dependence-preserving permutation tests (labels exchanged between whole recordings or trials); runs label-free diagnostics (time-block pseudo-labels, participant identification); and contains ground-truth and data-calibrated simulations.

## Scripts and the results they reproduce (JNE manuscript numbering)

| Script | Reproduces |
|---|---|
| `eeg_analysis.py` | Reader, preprocessing, band-power features, channel QC; Table 1; data for Figs 2–6 and Figs S1–S2 (drawn by `make_figures.py`); Table S3 (channel policy, `--step 2` non-overlapping windows) |
| `paired_ci.py`, `check_samples.py`, `fold_level.py`, `seed_repeat.py` | Paired CIs, QC exclusions (Table S2), complete cases, fold-level and five-seed analyses |
| `eegnet_validation.py` | EEGNet analyses (Table S3, Fig. S4 in the supplement) |
| `public_replication.py` | EEGMMIDB block and interleaved designs (Table 2, Fig. 7) |
| `simulate_fingerprints.py` | Ground-truth simulation (Fig. 8) |
| `calibrated_simulation.py` | Variance decomposition, calibrated simulation, ablations (Table 3, Fig. 9); `--sens-only --grid` for the sensitivity analysis |
| `null_tests.py` | Null-hypothesis experiments on real data: binomial, window-level and dependence-preserving permutation tests (Table 4, Fig. 10A–B) |
| `null_tests_simulation.py` | Type I error and power of evaluation-scheme × test combinations, scenarios a–f; `--scenarios g,h` implementation checks; `--scenarios b --datasets 160 --seed-offset 1000` precision extension (Table 5, Fig. 10C) |
| `external_validation.py` | External transfer (Supplementary Results S2, Table S4, Figs S3 and S5) |
| `make_figures.py`, `make_figs_4_5_8_S3.py`, `make_figS2_arial.py`, `make_figs_jne_new.py`, `make_fig_null_tests.py` | Figures |

Figure numbers follow the submitted JNE manuscript; see the figure captions for the exact mapping.

## Installation

Python 3.11. `pip install -r requirements.txt` (PyTorch only for `eegnet_validation.py`).

## Data and paths

Set `EEG_DATA_DIR`, `EEG_OUTPUT_DIR` and `EEGMMIDB_DIR` (or edit the CONFIG block of `eeg_analysis.py`). The cohort recordings cannot be shared publicly under the IRB-approved protocol; EEGMMIDB is public at https://physionet.org/content/eegmmidb/1.0.0/.

## Reproducing the article

```
python eeg_analysis.py --jobs 4 --policy common
python eeg_analysis.py --jobs 4 --policy exclude
python eeg_analysis.py --jobs 4 --policy common --step 2
python external_validation.py --jobs 4 --out analysis_output/policy_common
python eegnet_validation.py
python paired_ci.py
python check_samples.py
python fold_level.py
python seed_repeat.py
python public_replication.py --jobs 4 --download --aws
python simulate_fingerprints.py
python calibrated_simulation.py
python calibrated_simulation.py --sens-only --grid
python null_tests.py --tasks cohort_eo_hu --perms 999
python null_tests.py --tasks cohort_ec_eo,mmidb_block,mmidb_interleaved --perms 199 --with-cross
python null_tests_simulation.py
python null_tests_simulation.py --scenarios b --datasets 160 --seed-offset 1000 --tag ext_b
python null_tests_simulation.py --scenarios g,h --tag iid
python make_figures.py
python make_figs_jne_new.py
python make_fig_null_tests.py
```

All random seeds are fixed. Permutation p values use p = (1 + #null >= observed)/(B + 1).

## Citation and licence

Archived at Zenodo: https://doi.org/10.5281/zenodo.23608168 (concept DOI, all versions). See `CITATION.cff`. MIT licence.
