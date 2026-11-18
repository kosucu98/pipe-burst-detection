"""
A. Build the dataset with the ORIGINAL script's own functions (load_and_resample, engineer_features, label_incidents),
unchanged. Only differences: data folder path (./data next to this file), outputs go to ./work, and pd.read_excel uses the calamine engine
when it is installed (faster reader, same values).
Saves per-DMA unscaled feature frames, so that the split can be changed later without touching the features.
"""
import sys, os, types, importlib.util, time, pickle
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
WORK = HERE + 'work/'        # intermediate files (features, scores); not under version control
RESULTS = HERE + 'results/'  # result summaries (JSON)
os.makedirs(WORK, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)


def load_orig():
    try:
        import tensorflow  # noqa
    except Exception:       # TF is only used by the CNN; stub it so the rest of the script imports
        tf = types.ModuleType('tensorflow'); tf.get_logger = lambda: types.SimpleNamespace(setLevel=lambda *a: None)
        ks = types.ModuleType('tensorflow.keras'); tf.keras = ks
        for n in ['layers', 'callbacks']:
            m = types.ModuleType('tensorflow.keras.' + n); setattr(ks, n, m); sys.modules['tensorflow.keras.' + n] = m
        sys.modules['tensorflow'] = tf; sys.modules['tensorflow.keras'] = ks
    try:                    # calamine is only a faster Excel reader (same values); fall back to openpyxl if absent
        import python_calamine  # noqa
        _re = pd.read_excel
        pd.read_excel = lambda *a, **k: _re(*a, **{**k, 'engine': 'calamine'})
    except ImportError:
        pass
    spec = importlib.util.spec_from_file_location('orig', HERE + 'burst_detection_5models.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.DATA_DIR = HERE + 'data'
    m.INCIDENT_FILE = HERE + 'data/ARIZALAR.xlsx'
    m.OUTPUT_DIR = WORK + 'original_output'
    os.makedirs(m.OUTPUT_DIR, exist_ok=True)
    return m


if __name__ == '__main__':
    m = load_orig()
    inc = m.load_incidents(m.INCIDENT_FILE)
    os.makedirs(WORK + 'feat', exist_ok=True)
    for fn, cfg in m.DMA_FILE_MAP.items():
        out = WORK + 'feat/' + cfg['dma_name'] + '.pkl'
        if os.path.exists(out):
            continue
        p = os.path.join(m.DATA_DIR, fn)
        if not os.path.exists(p):
            print('SKIP', fn); continue
        t0 = time.time()
        try:
            r = m.load_and_resample(p, header_row=cfg['header_row'])
        except Exception as e:
            print('ERROR', fn, e); continue
        f = m.engineer_features(r)
        f.to_pickle(out)
        print(cfg['dma_name'], len(f), f.shape[1], f.index.min(), f.index.max(), f'{time.time()-t0:.0f}s', flush=True)
    pickle.dump(inc, open(WORK + 'incidents.pkl', 'wb'))
    print('DONE')
