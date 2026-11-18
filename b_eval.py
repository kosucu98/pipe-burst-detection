"""
B. Train and test the ORIGINAL script's five models (same features, same labels, same hyper-parameters, same
undersampling, same train_all_models function) under three ways of choosing the test data:

  random  : the script's own prepare_data(): stratified random 80/20 split of 15-min steps   (protocol A)
  chrono  : manuscript Section 4.6 (protocol B): per DMA, first 70 % of steps train, next 15 % left out as a gap, last 15 % test,
            per-DMA scaling fitted on the training part only, rolling features recomputed inside each
            part and its first 31 steps dropped                                              (protocol B)
  blocked : five contiguous time blocks; each block is the test set once; 2-day gap on each side removed from training
            (same features, scaling fitted on the training blocks)                           (protocol C)

usage: python b_eval.py <protocol> [models comma list: rf,xgb,svm,lgbm,cnn]
"""
import sys, os, json, glob, time, pickle
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a_build import load_orig, HERE, WORK, RESULTS

PROTO = sys.argv[1]
WANT = ['rf', 'xgb', 'svm', 'lgbm', 'cnn']
NAMES = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'svm': 'SVM (RBF)', 'lgbm': 'LightGBM', 'cnn': 'CNN'}
m = load_orig()
inc = pickle.load(open(WORK + 'incidents.pkl', 'rb'))
TEMPORAL = ['hour', 'dow', 'month', 'is_wknd', 'is_night', 'period_8h']

frames = {}
for fn, cfg in m.DMA_FILE_MAP.items():
    p = WORK + 'feat/' + cfg['dma_name'] + '.pkl'
    if os.path.exists(p) and cfg['dma_name'] not in frames:
        frames[cfg['dma_name']] = pd.read_pickle(p)
RAWCOLS = lambda f: [c for c in f.columns if c.rsplit('_', 1)[-1] in ('mean', 'min', 'max', 'std')]


def scale_label(f, dma, fit_mask):
    f = f.copy()
    sensor = [c for c in f.columns if c not in TEMPORAL]
    sc = StandardScaler().fit(f.loc[fit_mask, sensor])
    f[sensor] = sc.transform(f[sensor])
    f = m.label_incidents(f, inc[inc['dma_name'] == dma])
    f['dma_id'] = dma
    return f


def pool(parts):
    P = pd.concat(parts, axis=0)
    cols = [c for c in P.columns if c not in ('label', 'dma_id', 'part')]
    X = np.nan_to_num(P[cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return P, X, P['label'].values, cols


def undersample(y_tr, seed=42):
    pos = np.where(y_tr == 1)[0]; neg = np.where(y_tr == 0)[0]
    np.random.seed(seed)
    ns = np.random.choice(neg, size=min(len(neg), len(pos) * m.UNDERSAMPLE_RATIO), replace=False)
    idx = np.concatenate([pos, ns]); np.random.shuffle(idx)
    return idx


LAST = {}


def seed_all(seed=42):
    """Make the CNN reproducible: the original script sets random_state=42 for RF, XGBoost, LightGBM and SVM,
    but not for TensorFlow. Fix the TensorFlow, NumPy and Python seeds and use deterministic TensorFlow ops."""
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        np.random.seed(seed)


def run_models(X_bal, y_bal, X_te, y_te, y_tr):
    """Train the five models with the script's own train_all_models and return the test scores."""
    seed_all(42)
    res = m.train_all_models(X_bal, y_bal, X_te, y_te, y_tr, X_bal.shape[1])
    LAST['res'] = res
    return {k: res[NAMES[k]][1] for k in WANT}


def event_auc(P_te, score, n_rand=50, seed=0):
    """threshold-free event-level check: for every logged incident whose window lies in the test data, compare the
    highest score inside its window with the highest score in random incident-free windows of the same length in the
    same DMA and the same test period. 0.5 = chance."""
    rng = np.random.default_rng(seed)
    P = P_te[['dma_id', 'label']].copy(); P['s'] = score
    wins, rnds = [], []
    L = (m.PRE_INCIDENT_HOURS + m.POST_INCIDENT_HOURS) * 4 + 1
    for dma, g in P.groupby('dma_id', sort=False):
        lab = g['label'].values; s = g['s'].values; t = g.index
        ev = inc[(inc['dma_name'] == dma)]
        for _, r in ev.iterrows():
            w = (t >= r['incident_datetime'] - pd.Timedelta(hours=m.PRE_INCIDENT_HOURS)) & (t <= r['incident_datetime'] + pd.Timedelta(hours=m.POST_INCIDENT_HOURS))
            if w.sum() < 8:
                continue
            ok = [i for i in rng.integers(0, max(1, len(s) - L), n_rand * 3) if lab[i:i + L].sum() == 0][:n_rand]
            if not ok:
                continue
            wins.append(s[w].max()); rnds.append([s[i:i + L].max() for i in ok])
    if not wins:
        return dict(n_events=0)
    a = np.array(wins); R = rnds
    auc = np.mean([np.mean(a[i] > np.array(R[i])) + 0.5 * np.mean(a[i] == np.array(R[i])) for i in range(len(a))])
    return dict(n_events=len(a), event_auc=round(float(auc), 3))


def metrics(y, s, P_te, events=True):
    d = dict(n_test=int(len(y)), n_pos=int(y.sum()), prevalence=round(float(y.mean()), 5),
             pr_auc=round(float(average_precision_score(y, s)), 4), roc_auc=round(float(roc_auc_score(y, s)), 4),
             f1_at_0_5=round(float(f1_score(y, (s >= 0.5).astype(int), zero_division=0)), 4))
    if events:
        d.update(event_auc(P_te, s))
    return d


t0 = time.time()
results = {}
if PROTO == 'random':
    parts = []
    for dma, f in frames.items():
        parts.append(scale_label(f, dma, np.ones(len(f), bool)))       # script: scaler fitted on the whole series
    P, X, y, cols = pool(parts)
    idx = np.arange(len(y))
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(idx, test_size=m.TEST_SIZE, random_state=m.RANDOM_STATE, stratify=y)   # = prepare_data()
    b = undersample(y[tr])
    S = run_models(X[tr][b], y[tr][b], X[te], y[te], y[tr])
    for k, s in S.items():
        results[k] = metrics(y[te], s, None, events=False)
    np.savez_compressed(WORK + 'scores_random.npz', y=y[te], dma=P['dma_id'].values[te], t=P.index.values[te], cols=np.array(cols),
                        **{k: v for k, v in S.items()},
                        **{'imp_' + k: LAST['res'][NAMES[k]][2] for k in ('rf', 'xgb', 'lgbm')})
elif PROTO == 'chrono':
    tr_parts, te_parts = [], []
    for dma, f in frames.items():
        raw = f[RAWCOLS(f)]
        n = len(raw); a, b = int(0.70 * n), int(0.85 * n)
        segs = {}
        for name, (lo, hi) in {'train': (0, a), 'val': (a, b), 'test': (b, n)}.items():
            g = m.engineer_features(raw.iloc[lo:hi].copy()).iloc[31:]       # rolling features inside the part
            g['part'] = name
            segs[name] = g
        g = pd.concat(segs.values())
        g = scale_label(g.drop(columns='part'), dma, (g['part'] == 'train').values)
        g['part'] = pd.concat([s['part'] for s in segs.values()]).values
        tr_parts.append(g[g.part == 'train']); te_parts.append(g[g.part == 'test'])
    Ptr, Xtr, ytr, cols = pool(tr_parts)
    Pte = pd.concat(te_parts)[Ptr.columns]
    Xte = np.nan_to_num(Pte[cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0); yte = Pte['label'].values
    print('test window per DMA:', Pte.groupby('dma_id').apply(lambda d: f'{d.index.min():%Y-%m-%d} .. {d.index.max():%Y-%m-%d}').value_counts().head(3).to_dict())
    print('train pos', int(ytr.sum()), 'test pos', int(yte.sum()), 'test rows', len(yte))
    b = undersample(ytr)
    S = run_models(Xtr[b], ytr[b], Xte, yte, ytr)
    for k, s in S.items():
        results[k] = metrics(yte, s, Pte)
    np.savez_compressed(WORK + 'scores_chrono.npz', y=yte, dma=Pte['dma_id'].values, t=Pte.index.values, cols=np.array(cols),
                        **{k: v for k, v in S.items()},
                        **{'imp_' + k: LAST['res'][NAMES[k]][2] for k in ('rf', 'xgb', 'lgbm')})
elif PROTO == 'blocked':
    allidx = pd.concat([f.index.to_series() for f in frames.values()])
    t_lo, t_hi = allidx.min(), allidx.max()
    edges = pd.date_range(t_lo, t_hi, periods=6)
    agg = {}
    for k_fold in range(5):
        a, b = edges[k_fold], edges[k_fold + 1]
        gap = pd.Timedelta(days=2)
        tr_parts, te_parts = [], []
        for dma, f in frames.items():
            te_m = (f.index >= a) & (f.index < b) if k_fold < 4 else (f.index >= a)
            tr_m = ~((f.index >= a - gap) & (f.index < b + gap))
            if tr_m.sum() < 100:          # DMA with no data outside this block (FYA1-04: 16-30 Sep only): cannot be scaled
                print('fold', k_fold, 'skip', dma, 'no training rows'); continue
            g = scale_label(f, dma, tr_m)
            tr_parts.append(g[tr_m]); te_parts.append(g[te_m])
        Ptr, Xtr, ytr, cols = pool(tr_parts)
        Pte = pd.concat(te_parts)[Ptr.columns]
        Xte = np.nan_to_num(Pte[cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0); yte = Pte['label'].values
        bidx = undersample(ytr)
        S = run_models(Xtr[bidx], ytr[bidx], Xte, yte, ytr)
        for k, s in S.items():
            agg.setdefault(k, []).append(metrics(yte, s, Pte))
        print('fold', k_fold, {k: (v[-1]['pr_auc'], v[-1]['roc_auc'], v[-1].get('event_auc')) for k, v in agg.items()}, flush=True)
    for k, v in agg.items():
        results[k] = dict(folds=v, pr_auc_mean=round(float(np.mean([x['pr_auc'] for x in v])), 4),
                          prevalence_mean=round(float(np.mean([x['prevalence'] for x in v])), 5),
                          roc_auc_mean=round(float(np.mean([x['roc_auc'] for x in v])), 4),
                          n_events=int(sum(x['n_events'] for x in v)),
                          event_auc_mean=round(float(np.nanmean([x.get('event_auc', np.nan) for x in v])), 3))
results['_seconds'] = round(time.time() - t0)
json.dump(results, open(RESULTS + f'res_{PROTO}.json', 'w'), indent=1, default=str)
for k, v in results.items():
    if k != '_seconds':
        print(k, {kk: vv for kk, vv in v.items() if kk != 'folds'})
