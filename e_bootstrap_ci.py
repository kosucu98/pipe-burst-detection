"""E. 95% bootstrap intervals of step-level PR-AUC and ROC-AUC for protocols A (random) and B (chrono).
Resampling unit: DMA-day (all steps of one day in one DMA stay together), 1,000 resamples.
Run after: python b_eval.py random  and  python b_eval.py chrono  (they write scores_random.npz, scores_chrono.npz).

A resample is a multiset of DMA-days, i.e. a weight (0, 1, 2, ...) for every test step. PR-AUC (average precision)
and ROC-AUC are computed with these weights from one pre-sorted score list, which gives the same values as
sklearn's average_precision_score / roc_auc_score on the concatenated resample, much faster."""
import os, json
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
WORK, RESULTS = HERE + 'work/', HERE + 'results/'
MOD = ['xgb', 'lgbm', 'rf', 'svm', 'cnn']
B = 1000


def prep(y, s):
    o = np.argsort(-s, kind='mergesort')
    ys, ss = y[o].astype(float), s[o]
    last = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]      # last index of each distinct score value
    return o, ys, last


def weighted_ap_roc(o, ys, last, w):
    ws = w[o]
    tp = np.cumsum(ws * ys)[last]; fp = np.cumsum(ws * (1 - ys))[last]
    P, N = tp[-1], fp[-1]
    prec = tp / np.maximum(tp + fp, 1e-300)
    rec = tp / P
    ap = np.sum(np.diff(np.r_[0.0, rec]) * prec)
    tpr = np.r_[0.0, tp / P]; fpr = np.r_[0.0, fp / N]
    roc = np.trapz(tpr, fpr) if hasattr(np, 'trapz') else np.trapezoid(tpr, fpr)
    return ap, roc


ci = {}
rng = np.random.default_rng(0)
for p in ['random', 'chrono']:
    Z = np.load(WORK + f'scores_{p}.npz', allow_pickle=True)
    y = Z['y'].astype(int)
    key = pd.Series(Z['dma']).astype(str) + '|' + pd.to_datetime(Z['t']).strftime('%Y-%m-%d')
    codes, uniq = pd.factorize(key)
    ncl = len(uniq)
    W = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, ncl, ncl), minlength=ncl)
        W.append(cnt[codes].astype(float))
    for m in MOD:
        s = Z[m].astype(float)
        o, ys, last = prep(y, s)
        pr, roc = [], []
        for w in W:
            if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0:
                continue
            a, r = weighted_ap_roc(o, ys, last, w)
            pr.append(a); roc.append(r)
        ci[f'{p}_{m}'] = dict(pr=average_precision_score(y, s), roc=roc_auc_score(y, s),
                              pr_ci=np.percentile(pr, [2.5, 97.5]).tolist(),
                              roc_ci=np.percentile(roc, [2.5, 97.5]).tolist())
        print(p, m, round(ci[f'{p}_{m}']['pr'], 4), np.round(ci[f'{p}_{m}']['pr_ci'], 4),
              round(ci[f'{p}_{m}']['roc'], 4), np.round(ci[f'{p}_{m}']['roc_ci'], 4), flush=True)
json.dump(ci, open(RESULTS + 'res_ci.json', 'w'), indent=1)
