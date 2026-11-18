"""
C. Why does the random split score high? Original features and the script's XGBoost (same parameters), only the unit
that is randomly assigned to train or test changes. 80/20, stratified where possible, 3 seeds.
"""
import sys, os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from xgboost import XGBClassifier
sys.argv = [sys.argv[0], 'none']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'b_eval.py')).read().split("t0 = time.time()")[0])

parts = [scale_label(f, dma, np.ones(len(f), bool)) for dma, f in frames.items()]
P, X, y, cols = pool(parts)
day = P.index.floor('D').astype(str)
units = {'15-min step': np.arange(len(y)),
         'DMA-hour': (P['dma_id'] + '|' + P.index.floor('h').astype(str)).values,
         'DMA-day': (P['dma_id'] + '|' + day).values,
         'DMA-week': (P['dma_id'] + '|' + P.index.to_period('W').astype(str)).values,
         'calendar day (all DMAs)': day.values}
nocal = [i for i, c in enumerate(cols) if c not in ('month', 'dow')]


def xgb(spw):
    return XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=5, scale_pos_weight=spw, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
                         n_jobs=1, eval_metric='logloss')


out = {}
for name, g in units.items():
    for tag, cs in [('all features', slice(None)), ('without month, day of week', nocal)]:
        if tag != 'all features' and name not in ('15-min step', 'DMA-day'):
            continue
        r = []
        for seed in [42, 1, 2]:
            if name == '15-min step':
                tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
            else:
                tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, y, g))
            b = tr[undersample(y[tr], seed)]
            spw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
            s = xgb(spw).fit(X[b][:, cs], y[b]).predict_proba(X[te][:, cs])[:, 1]
            r.append((average_precision_score(y[te], s), roc_auc_score(y[te], s), y[te].mean()))
        a = np.array(r)
        out[f'{name} | {tag}'] = dict(pr_auc=round(a[:, 0].mean(), 4), roc_auc=round(a[:, 1].mean(), 4), prevalence=round(a[:, 2].mean(), 5))
        print(name, '|', tag, out[f'{name} | {tag}'], flush=True)
json.dump(out, open(RESULTS + 'res_diag.json', 'w'), indent=1)
