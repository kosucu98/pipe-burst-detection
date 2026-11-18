"""D. Event-level AUC of protocol B with 95% intervals from resampling the incidents (run after: python b_eval.py chrono)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
WORK, RESULTS = HERE + 'work/', HERE + 'results/'
import numpy as np, pandas as pd, json
d = np.load(WORK + 'scores_chrono.npz', allow_pickle=True)
inc = pd.read_pickle(WORK + 'incidents.pkl')
P = pd.DataFrame({'dma_id': d['dma'], 'label': d['y']}, index=pd.DatetimeIndex(d['t']))
PRE, POST = 6, 2; L = (PRE + POST) * 4 + 1
out = {}
for mname in ['xgb', 'lgbm', 'rf', 'svm', 'cnn']:
    rng = np.random.default_rng(0); P['s'] = d[mname]; vals = []
    for dma, g in P.groupby('dma_id', sort=False):
        lab = g['label'].values; s = g['s'].values; t = g.index
        for _, r in inc[inc['dma_name'] == dma].iterrows():
            w = (t >= r['incident_datetime'] - pd.Timedelta(hours=PRE)) & (t <= r['incident_datetime'] + pd.Timedelta(hours=POST))
            if w.sum() < 8: continue
            ok = [i for i in rng.integers(0, max(1, len(s) - L), 150) if lab[i:i + L].sum() == 0][:50]
            if not ok: continue
            R = np.array([s[i:i + L].max() for i in ok]); a = s[w].max()
            vals.append(np.mean(a > R) + 0.5 * np.mean(a == R))
    v = np.array(vals); b = np.random.default_rng(1)
    bs = [v[b.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    out[mname] = dict(n=len(v), auc=round(v.mean(), 3), lo=round(np.percentile(bs, 2.5), 2), hi=round(np.percentile(bs, 97.5), 2))
print(out); json.dump(out, open(RESULTS + 'res_event_ci_chrono.json', 'w'))
