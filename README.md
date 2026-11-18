# Random versus chronological evaluation of burst detection models

Code for the paper *A random versus chronological evaluation of five machine learning models
for pipe burst detection on utility SCADA data* (Koşucu, Koşucu, Kesgin, Kızılöz and Şişman, 2026).

Five classifiers (random forest, XGBoost, LightGBM, an RBF support vector machine and a
one-dimensional CNN) are trained to detect pipe bursts in 15-minute SCADA data from 43 district
metered areas (DMAs) in Kocaeli, Türkiye. The same features, labels and model settings are
evaluated under three protocols:

| Protocol | Test data | Script |
|---|---|---|
| A | stratified random 80/20 split of all 15-minute steps (as in the original analysis script) | `b_eval.py random` |
| B | last 15% of each DMA record; the 15% before it is left out as a gap; rolling features computed inside each part, scaling fitted on the training part | `b_eval.py chrono` |
| C | five contiguous time blocks, each tested once, with a 2-day gap on each side | `b_eval.py blocked` |

Under protocol A the models look strong (XGBoost PR-AUC 0.46, ROC-AUC 0.996). Under protocols B and C
all five are close to chance. A split-unit diagnostic (`c_diag.py`) shows that assigning whole
DMA-days instead of single steps to the test set is enough to remove the advantage of the random split.

## Data

The SCADA records and the incident log belong to the Kocaeli Water and Sewerage Administration
(İSU). They are confidential and are **not included**. Researchers who wish to use them must apply
to İSU directly. `data/README.md` describes the expected files, so the code can also be run on
other utility data in the same format.

The JSON files in `results/` hold the summary results reported in the paper. They contain metrics
only and do not replace access to the data.

## Repository layout

```
burst_detection_5models.py  original analysis script (feature engineering, labeling, the five models)
a_build.py                  reads the data with the original functions, saves per-DMA features
b_eval.py                   protocols A, B and C
c_diag.py                   split-unit diagnostic (Table 4 of the paper)
d_eventci.py                event-level AUC of protocol B with 95% intervals
e_bootstrap_ci.py           step-level PR-AUC and ROC-AUC 95% intervals, protocols A and B
results/                    result summaries (JSON), as reported in the paper
data/                       place the data files here (ignored by git)
work/                       intermediate files, created when the scripts run (ignored by git)
```

The evaluation scripts import the functions of `burst_detection_5models.py` and do not change
them. Only the test data selection differs between the protocols.

## Installation

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The results were produced with Python 3.13 and the package versions in `requirements.txt`.
TensorFlow is needed only for the CNN.

## Usage

Run the scripts in this order from the repository folder:

```
python a_build.py              # features per DMA -> work/feat/, incident log -> work/incidents.pkl
python b_eval.py random        # protocol A -> results/res_random.json
python b_eval.py chrono        # protocol B -> results/res_chrono.json
python b_eval.py blocked       # protocol C -> results/res_blocked.json
python c_diag.py               # split-unit diagnostic -> results/res_diag.json
python d_eventci.py            # event-level AUC intervals, protocol B -> results/res_event_ci_chrono.json
python e_bootstrap_ci.py       # PR-AUC / ROC-AUC intervals, protocols A and B -> results/res_ci.json
```

`d_eventci.py` and `e_bootstrap_ci.py` use the test scores saved by `b_eval.py`, so they must run
after it. The full sequence took about one hour on a two-core machine; SVM and CNN training take
most of the time.

## Results files

| File | Content | In the paper |
|---|---|---|
| `res_random.json` | protocol A: PR-AUC, ROC-AUC and prevalence per model | Table 3, Figure 3 |
| `res_chrono.json` | protocol B: PR-AUC, ROC-AUC, event-level AUC per model | Table 3, Figure 3 |
| `res_blocked.json` | protocol C: per-fold and mean values per model | Table 3, Figure 3 |
| `res_ci.json` | 95% bootstrap intervals of PR-AUC and ROC-AUC (1,000 resamples of DMA-days) | Table 3 |
| `res_event_ci_chrono.json` | event-level AUC of protocol B with 95% intervals (2,000 resamples of incidents) | Table 3 |
| `res_diag.json` | XGBoost with 15-minute steps, DMA-hours, DMA-days, DMA-weeks or calendar days assigned at random to the test set; steps and DMA-days also without the month and day-of-week features (3 seeds) | Table 4 |

## Reproducibility

- Random forest, XGBoost, LightGBM, the SVM and the undersampling use a fixed seed (42), as in the
  original script.
- The original script sets no seed for TensorFlow. `b_eval.py` therefore fixes the TensorFlow,
  NumPy and Python seeds and enables deterministic TensorFlow operations before training
  (`seed_all`). Repeated runs on the same machine give identical results for all five models.
  On other hardware or TensorFlow versions the CNN values may differ slightly.
- The confidence intervals use fixed random generators, so they are also repeatable.

## Notes on the original script

`burst_detection_5models.py` is the script used for the original analysis. For this repository,
only the data and output paths in its configuration block were changed to relative paths. Two
details matter when comparing with the paper:

- Excel files are read with `usecols=range(5)`. The file of DMA1-06 has four columns and could not
  be read this way, which is why 43 of the 44 DMAs were used.
- The entry `dma40011-07.xlsx` appears twice in `DMA_FILE_MAP`. This has no effect.

## Citation

If you use this code, please cite:

> Koşucu, O., Koşucu, M. M., Kesgin, E., Kızılöz, B. and Şişman, E. (2026). A random versus
> chronological evaluation of five machine learning models for pipe burst detection on utility
> SCADA data. Manuscript submitted for publication.

## Acknowledgments

This work was supported by the Istanbul Technical University Scientific Research Projects
Department (project MAB-2024-45834). Claude (Anthropic) was used during the preparation of this work.

## License

The code is released under the MIT License (see `LICENSE`). The license does not cover the
utility data, which are not part of this repository.
