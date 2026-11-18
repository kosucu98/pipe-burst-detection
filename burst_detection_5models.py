#!/usr/bin/env python3
"""
=============================================================================
Pipe Burst / Incident Detection in Water Distribution Networks
5-Method Comparison: RF, XGBoost, SVM, CNN, LightGBM
on Real SCADA Data
=============================================================================

This script:
1. Loads SCADA sensor data from multiple DMAs (Excel files)
2. Loads the incident (ARIZALAR) log
3. Resamples all data to uniform 15-minute intervals
4. Engineers temporal and statistical features (rolling stats, z-scores, etc.)
5. Labels time windows around known incidents (6h before, 2h after)
6. Pools data across DMAs with per-DMA normalization
7. Trains and evaluates 5 classifiers:
   - Random Forest
   - XGBoost
   - Support Vector Machine (RBF kernel)
   - 1D Convolutional Neural Network
   - LightGBM
8. Runs 5-fold stratified cross-validation
9. Produces classification reports, confusion matrices, ROC/PR curves,
   feature importance plots, and model comparison charts

Author : Mehmet
Date   : 2026
Data   : Real operational SCADA data from Kocaeli, Türkiye
Sensors: Inlet Pressure, Flow (Debi), Outlet Pressure, Liquid Detection

Requirements:
    pip install pandas numpy scikit-learn xgboost lightgbm tensorflow
              imbalanced-learn matplotlib openpyxl
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
from datetime import timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve,
    ConfusionMatrixDisplay, f1_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# TensorFlow / Keras for CNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF info messages
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- File Paths ---
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, 'data')                                  # Directory with sensor Excel files
INCIDENT_FILE = os.path.join(_HERE, 'data', 'ARIZALAR.xlsx')
OUTPUT_DIR = os.path.join(_HERE, 'work', 'original_output')            # Output directory for plots
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- DMA Sensor File Mapping ---
# Keys: Excel filenames
# Values: DMA name (must match ARIZALAR column B), header row index
# *** ADD MORE FILES HERE AS YOU UPLOAD THEM ***
DMA_FILE_MAP = {
    'd2-01.xlsx':       {'dma_name': 'D2-01',       'header_row': 0},
    'd2-02.xlsx':       {'dma_name': 'D2-02',       'header_row': 1},  # header in row 1
    'd3-01.xlsx':       {'dma_name': 'D3-01',       'header_row': 0},
    'd3-02.xlsx':       {'dma_name': 'D3-02',       'header_row': 0},
    'dma40011-06.xlsx': {'dma_name': 'DMA40011-06', 'header_row': 0},
    'dma40011-07.xlsx': {'dma_name': 'DMA40011-07', 'header_row': 0},
    'd3-03.xlsx':       {'dma_name': 'D3-03',       'header_row': 0},
    'd3-04.xlsx':       {'dma_name': 'D3-04',       'header_row': 0},
    'd3-05.xlsx':       {'dma_name': 'D3-05',       'header_row': 0},
    'd3-06.xlsx':       {'dma_name': 'D3-06',       'header_row': 0},
    'd3-07.xlsx':       {'dma_name': 'D3-07',       'header_row': 0},
    'dma21-01.xlsx':    {'dma_name': 'DMA21-01',    'header_row': 0},
    'dma1-01.xlsx':     {'dma_name': 'DMA1-01',     'header_row': 0},
    'dma1-06.xlsx':     {'dma_name': 'DMA1-06',     'header_row': 0},
    'dma1-07.xlsx':     {'dma_name': 'DMA1-07',     'header_row': 0},
    'dma1-09.xlsx':     {'dma_name': 'DMA1-09',     'header_row': 0},
    'dma2-01.xlsx':     {'dma_name': 'DMA2-01',     'header_row': 0},
    'dma2-04.xlsx':     {'dma_name': 'DMA2-04',     'header_row': 0},
    'dma2-05.xlsx':     {'dma_name': 'DMA2-05',     'header_row': 0},
    'dma3-01.xlsx':     {'dma_name': 'DMA3-01',     'header_row': 0},
    'dma4-01.xlsx':     {'dma_name': 'DMA4-01',     'header_row': 0},
    'dma5-01.xlsx':     {'dma_name': 'DMA5-01',     'header_row': 0},
    'dma5-02.xlsx':     {'dma_name': 'DMA5-02',     'header_row': 0},
    'dma5-03.xlsx': {'dma_name': 'DMA5-03', 'header_row': 0},
    'dma5-04.xlsx': {'dma_name': 'DMA5-04', 'header_row': 0},
    'dma6-02.xlsx': {'dma_name': 'DMA6-02', 'header_row': 0},
    'dma6-03.xlsx': {'dma_name': 'DMA6-03', 'header_row': 0},
    'dma6-05.xlsx': {'dma_name': 'DMA6-05', 'header_row': 0},
    'dma41-01.xlsx': {'dma_name': 'DMA41-01', 'header_row': 0},
    'dma40011-07.xlsx': {'dma_name': 'DMA40011-07', 'header_row': 0},
    'dma40011-08.xlsx': {'dma_name': 'DMA40011-08', 'header_row': 0},
    'dmsa1-06.xlsx': {'dma_name': 'DMSA1-06', 'header_row': 0},
    'dya1-01.xlsx': {'dma_name': 'DYA1-01', 'header_row': 0},
    'dya1-02.xlsx': {'dma_name': 'DYA1-02', 'header_row': 0},
    'dya1-03.xlsx': {'dma_name': 'DYA1-03', 'header_row': 0},
    'dya1-04.xlsx': {'dma_name': 'DYA1-04', 'header_row': 0},
    'dya1-05.xlsx': {'dma_name': 'DYA1-05', 'header_row': 0},
    'dya1-06.xlsx': {'dma_name': 'DYA1-06', 'header_row': 0},
    'dya1-07.xlsx': {'dma_name': 'DYA1-07', 'header_row': 0},
    'dyg1-01.xlsx': {'dma_name': 'DYG1-01', 'header_row': 0},
    'dyg4-01.xlsx': {'dma_name': 'DYG4-01', 'header_row': 0},
    'dyg5-01.xlsx': {'dma_name': 'DYG5-01', 'header_row': 0},
    'dyg5-02.xlsx': {'dma_name': 'DYG5-02', 'header_row': 0},
    'fya1-01.xlsx': {'dma_name': 'FYA1-01', 'header_row': 0},
    'fya1-04.xlsx': {'dma_name': 'FYA1-04', 'header_row': 0},

    # Add more DMAs:
    # 'dma1-01.xlsx':   {'dma_name': 'DMA1-01',     'header_row': 0},
}

# --- Turkish → English Column Name Mapping ---
COL_MAP = {
    'Tarih/Saat':      'datetime',
    'Giriş Basıncı':   'inlet_pressure',
    'Debi':            'flow',
    'Çıkış Basıncı':   'outlet_pressure',
    'Sıvı Tespiti':    'liquid_detection',
}

# --- Model Parameters ---
RESAMPLE_INTERVAL = '15min'       # Unify 1-min and 15-min data
PRE_INCIDENT_HOURS = 6            # Hours before incident to label as positive
POST_INCIDENT_HOURS = 2           # Hours after incident to label as positive
ROLLING_WINDOWS = [4, 8, 16, 32]  # Rolling windows in 15-min steps (1h, 2h, 4h, 8h)
UNDERSAMPLE_RATIO = 10            # Neg:Pos ratio for training (combat class imbalance)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Plot colors for each model
MODEL_COLORS = {
    'Random Forest': '#2196F3',
    'XGBoost':       '#4CAF50',
    'SVM (RBF)':     '#FF9800',
    'LightGBM':      '#9C27B0',
    'CNN':           '#F44336',
}


# =============================================================================
# STEP 1: LOAD INCIDENT DATA
# =============================================================================

def load_incidents(filepath):
    """Load and parse the ARIZALAR (incidents) file."""
    df = pd.read_excel(filepath)
    df.columns = ['city', 'dma_name', 'incident_datetime', 'description', 'pipe_length_km']
    df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
    df['dma_name'] = df['dma_name'].str.strip()
    print(f"[INCIDENTS] Loaded {len(df)} incidents across {df['dma_name'].nunique()} DMAs")
    return df


# =============================================================================
# STEP 2: LOAD & RESAMPLE SENSOR DATA
# =============================================================================

def load_and_resample(filepath, header_row=0):
    """
    Load a DMA sensor Excel file, standardize column names,
    and resample to uniform 15-minute intervals with aggregation stats.
    """
    df = pd.read_excel(filepath, header=header_row, usecols=range(5))

    # Map Turkish column names to English
    col_remap = {}
    for orig_col in df.columns:
        orig_clean = str(orig_col).strip().replace('.', '').replace('...', '')
        for turkish, english in COL_MAP.items():
            if turkish.lower() in orig_clean.lower() or orig_clean.lower() in turkish.lower():
                col_remap[orig_col] = english
                break
    df = df.rename(columns=col_remap)

    # Keep only sensor columns
    keep = [c for c in ['datetime', 'inlet_pressure', 'flow', 'outlet_pressure',
                         'liquid_detection'] if c in df.columns]
    df = df[keep]

    # Parse datetime and set as index
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()

    # Ensure numeric types
    for col in ['inlet_pressure', 'flow', 'outlet_pressure', 'liquid_detection']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Resample to 15-minute intervals with multiple aggregation statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg_dict = {col: ['mean', 'min', 'max', 'std'] for col in numeric_cols}
    resampled = df[numeric_cols].resample(RESAMPLE_INTERVAL).agg(agg_dict)
    resampled.columns = ['_'.join(col) for col in resampled.columns]
    resampled = resampled.dropna(how='all')

    return resampled


# =============================================================================
# STEP 3: FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """
    Create temporal and statistical features from resampled sensor data.

    Feature categories:
    - Temporal: hour, day of week, month, weekend flag, night flag, 8h period
    - Rolling statistics: mean, std, range over 1h/2h/4h/8h windows
    - Rate of change: 15-min and 1-hour differentials
    - Z-score anomaly: deviation from 8-hour rolling baseline
    - Pressure-flow interaction: ratio and pressure drop
    """
    feat = df.copy()

    # --- Temporal features ---
    feat['hour'] = feat.index.hour
    feat['dow'] = feat.index.dayofweek
    feat['month'] = feat.index.month
    feat['is_wknd'] = (feat.index.dayofweek >= 5).astype(int)
    feat['is_night'] = ((feat.index.hour >= 22) | (feat.index.hour <= 5)).astype(int)

    # 8-hour period segmentation (night/day/evening)
    feat['period_8h'] = pd.cut(
        feat.index.hour, bins=[0, 8, 16, 24],
        labels=[0, 1, 2], right=False, include_lowest=True
    ).astype(int)

    # --- Rolling statistics for mean columns ---
    base_cols = [c for c in feat.columns if c.endswith('_mean') and
                 any(s in c for s in ['flow', 'inlet_pressure', 'outlet_pressure'])]

    for col in base_cols:
        for w in ROLLING_WINDOWS:
            feat[f'{col}_rm{w}'] = feat[col].rolling(w, min_periods=1).mean()
            feat[f'{col}_rs{w}'] = feat[col].rolling(w, min_periods=1).std()
            feat[f'{col}_rr{w}'] = (
                feat[col].rolling(w, min_periods=1).max() -
                feat[col].rolling(w, min_periods=1).min()
            )

        # Rate of change
        feat[f'{col}_d1'] = feat[col].diff(1)   # 15-min change
        feat[f'{col}_d4'] = feat[col].diff(4)   # 1-hour change

        # Z-score relative to 8-hour rolling baseline
        rm32 = feat[col].rolling(32, min_periods=4).mean()
        rs32 = feat[col].rolling(32, min_periods=4).std()
        feat[f'{col}_z'] = (feat[col] - rm32) / rs32.replace(0, np.nan)

    # --- Pressure-flow ratio ---
    p_cols = [c for c in base_cols if 'pressure' in c]
    f_cols = [c for c in base_cols if 'flow' in c]
    if p_cols and f_cols:
        feat['pf_ratio'] = feat[p_cols[0]] / feat[f_cols[0]].replace(0, np.nan)

    # --- Pressure drop (inlet - outlet) ---
    i_cols = [c for c in base_cols if 'inlet' in c]
    o_cols = [c for c in base_cols if 'outlet' in c]
    if i_cols and o_cols:
        feat['pdrop'] = feat[i_cols[0]] - feat[o_cols[0]]
        feat['pdrop_rm8'] = feat['pdrop'].rolling(8, min_periods=1).mean()

    # Fill NaN from rolling/diff operations
    feat = feat.fillna(0)

    return feat


# =============================================================================
# STEP 4: LABEL INCIDENTS
# =============================================================================

def label_incidents(df, incidents_for_dma):
    """
    Label each 15-min time step:
      1 = within incident window (6h before to 2h after reported time)
      0 = normal operation
    """
    df['label'] = 0
    for _, row in incidents_for_dma.iterrows():
        t = row['incident_datetime']
        window_start = t - timedelta(hours=PRE_INCIDENT_HOURS)
        window_end = t + timedelta(hours=POST_INCIDENT_HOURS)
        mask = (df.index >= window_start) & (df.index <= window_end)
        df.loc[mask, 'label'] = 1
    return df


# =============================================================================
# STEP 5: BUILD POOLED DATASET
# =============================================================================

def build_pooled_dataset(data_dir, incidents_df, dma_file_map):
    """
    Load all DMA files, engineer features, normalize per-DMA,
    label incidents, and pool into a single dataset.
    """
    all_data = []

    for filename, config in dma_file_map.items():
        filepath = os.path.join(data_dir, filename)
        dma_name = config['dma_name']

        if not os.path.exists(filepath):
            print(f"  [SKIP] {filename} not found")
            continue

        t0 = time.time()
        print(f"\n  Processing {dma_name} ({filename})...", end=' ')

        # Load and resample
        try:
            resampled = load_and_resample(filepath, header_row=config['header_row'])
        except Exception as e:
            print(f"[ERROR] {e}")
            continue

        # Engineer features
        featured = engineer_features(resampled)

        # Per-DMA normalization (z-score sensor features, keep temporal as-is)
        temporal_cols = ['hour', 'dow', 'month', 'is_wknd', 'is_night', 'period_8h']
        sensor_cols = [c for c in featured.columns if c not in temporal_cols]
        scaler = StandardScaler()
        featured[sensor_cols] = scaler.fit_transform(featured[sensor_cols])

        # Label with incidents
        dma_incidents = incidents_df[incidents_df['dma_name'] == dma_name]
        featured = label_incidents(featured, dma_incidents)

        n_pos = featured['label'].sum()
        elapsed = time.time() - t0
        print(f"{len(resampled)} rows → {featured.shape[1]-1} features | "
              f"{len(dma_incidents)} incidents, {n_pos} positive samples | {elapsed:.0f}s")

        featured['dma_id'] = dma_name
        all_data.append(featured)
        del resampled

    if not all_data:
        raise ValueError("No DMA data loaded. Check file paths and DMA_FILE_MAP.")

    pooled = pd.concat(all_data, axis=0)
    return pooled


# =============================================================================
# STEP 6: PREPARE TRAINING DATA
# =============================================================================

def prepare_data(pooled):
    """Prepare feature matrix and labels, split into train/test."""
    exclude_cols = ['label', 'dma_id']
    feature_cols = [c for c in pooled.columns if c not in exclude_cols]
    X = pooled[feature_cols].values.astype(np.float32)
    y = pooled['label'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Undersample majority class for tractable training
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    np.random.seed(RANDOM_STATE)
    neg_sample = np.random.choice(
        neg_idx, size=min(len(neg_idx), len(pos_idx) * UNDERSAMPLE_RATIO), replace=False
    )
    train_idx = np.concatenate([pos_idx, neg_sample])
    np.random.shuffle(train_idx)
    X_bal, y_bal = X_train[train_idx], y_train[train_idx]

    return X, y, X_train, X_test, y_train, y_test, X_bal, y_bal, feature_cols


# =============================================================================
# STEP 7: TRAIN ALL 5 MODELS
# =============================================================================

def train_all_models(X_bal, y_bal, X_test, y_test, y_train, n_features):
    """
    Train all 5 classifiers and return results dict.
    Returns: {model_name: (predictions, probabilities, feature_importances_or_None)}
    """
    results = {}

    # ================================================================
    # 1. RANDOM FOREST
    # ================================================================
    print("\n  " + "="*55)
    print("  RANDOM FOREST")
    print("  " + "="*55)
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=5,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_bal, y_bal)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    print(f"  Trained in {time.time()-t0:.1f}s")
    print(classification_report(y_test, rf_pred,
                                target_names=['Normal', 'Incident'], zero_division=0))
    results['Random Forest'] = (rf_pred, rf_prob, rf.feature_importances_)

    # ================================================================
    # 2. XGBoost
    # ================================================================
    print("  " + "="*55)
    print("  XGBoost")
    print("  " + "="*55)
    spw = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=spw, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1,
        eval_metric='logloss', use_label_encoder=False
    )
    xgb.fit(X_bal, y_bal)
    xgb_pred = xgb.predict(X_test)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    print(f"  Trained in {time.time()-t0:.1f}s")
    print(classification_report(y_test, xgb_pred,
                                target_names=['Normal', 'Incident'], zero_division=0))
    results['XGBoost'] = (xgb_pred, xgb_prob, xgb.feature_importances_)

    # ================================================================
    # 3. SUPPORT VECTOR MACHINE (RBF kernel)
    # ================================================================
    print("  " + "="*55)
    print("  SVM (RBF Kernel)")
    print("  " + "="*55)
    t0 = time.time()
    # SVM benefits from feature scaling
    svm_scaler = StandardScaler()
    X_bal_sc = svm_scaler.fit_transform(X_bal)
    X_test_sc = svm_scaler.transform(X_test)

    svm = SVC(
        kernel='rbf', C=10.0, gamma='scale',
        class_weight='balanced', probability=True,
        random_state=RANDOM_STATE, cache_size=500
    )
    svm.fit(X_bal_sc, y_bal)
    svm_pred = svm.predict(X_test_sc)
    svm_prob = svm.predict_proba(X_test_sc)[:, 1]
    print(f"  Trained in {time.time()-t0:.1f}s")
    print(classification_report(y_test, svm_pred,
                                target_names=['Normal', 'Incident'], zero_division=0))
    results['SVM (RBF)'] = (svm_pred, svm_prob, None)  # SVM has no feature_importances_

    # ================================================================
    # 4. LightGBM
    # ================================================================
    print("  " + "="*55)
    print("  LightGBM")
    print("  " + "="*55)
    t0 = time.time()
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, scale_pos_weight=spw,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    lgbm.fit(X_bal, y_bal)
    lgbm_pred = lgbm.predict(X_test)
    lgbm_prob = lgbm.predict_proba(X_test)[:, 1]
    print(f"  Trained in {time.time()-t0:.1f}s")
    print(classification_report(y_test, lgbm_pred,
                                target_names=['Normal', 'Incident'], zero_division=0))
    results['LightGBM'] = (lgbm_pred, lgbm_prob, lgbm.feature_importances_)

    # ================================================================
    # 5. 1D CONVOLUTIONAL NEURAL NETWORK
    # ================================================================
    print("  " + "="*55)
    print("  1D CNN")
    print("  " + "="*55)
    t0 = time.time()

    # Scale features for CNN
    cnn_scaler = StandardScaler()
    X_bal_cnn = cnn_scaler.fit_transform(X_bal).astype(np.float32)
    X_test_cnn = cnn_scaler.transform(X_test).astype(np.float32)

    # Reshape for 1D CNN: (samples, features, 1)
    X_bal_cnn = X_bal_cnn.reshape(-1, n_features, 1)
    X_test_cnn = X_test_cnn.reshape(-1, n_features, 1)

    # Build model
    model = keras.Sequential([
        layers.Input(shape=(n_features, 1)),
        # Conv Block 1
        layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        # Conv Block 2
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        # Conv Block 3
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        # Dense classification head
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Class weights for imbalance
    n_pos = y_bal.sum()
    n_neg = len(y_bal) - n_pos
    class_weight = {0: 1.0, 1: n_neg / max(1, n_pos)}

    # Train with early stopping
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    model.fit(
        X_bal_cnn, y_bal,
        epochs=50, batch_size=64,
        validation_split=0.15,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=1
    )
    cnn_prob = model.predict(X_test_cnn, verbose=0).flatten()
    cnn_pred = (cnn_prob >= 0.5).astype(int)
    print(f"  Trained in {time.time()-t0:.1f}s")
    print(classification_report(y_test, cnn_pred,
                                target_names=['Normal', 'Incident'], zero_division=0))
    results['CNN'] = (cnn_pred, cnn_prob, None)  # CNN has no feature_importances_

    return results


# =============================================================================
# STEP 8: CROSS-VALIDATION
# =============================================================================

def run_cross_validation(X, y):
    """Run 5-fold stratified CV on a balanced subsample."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    np.random.seed(RANDOM_STATE)
    neg_sub = np.random.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 15), replace=False)
    idx = np.concatenate([pos_idx, neg_sub])
    np.random.shuffle(idx)
    X_cv, y_cv = X[idx], y[idx]

    print(f"  CV dataset: {len(X_cv)} samples ({y_cv.sum()} positive)")

    spw = (y_cv == 0).sum() / max(1, (y_cv == 1).sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_results = {}
    for name, model_obj in [
        ('Random Forest', RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1)),
        ('XGBoost', XGBClassifier(
            n_estimators=100, max_depth=8, learning_rate=0.05,
            scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric='logloss', use_label_encoder=False)),
        ('SVM (RBF)', SVC(
            kernel='rbf', C=10.0, gamma='scale', class_weight='balanced',
            probability=True, random_state=RANDOM_STATE)),
        ('LightGBM', LGBMClassifier(
            n_estimators=100, max_depth=8, learning_rate=0.05,
            scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
    ]:
        f1_scores = cross_val_score(model_obj, X_cv, y_cv, cv=cv, scoring='f1', n_jobs=1)
        auc_scores = cross_val_score(model_obj, X_cv, y_cv, cv=cv, scoring='roc_auc', n_jobs=1)
        cv_results[name] = {
            'f1_mean': f1_scores.mean(), 'f1_std': f1_scores.std(),
            'auc_mean': auc_scores.mean(), 'auc_std': auc_scores.std()
        }
        print(f"  {name}: F1={f1_scores.mean():.4f}±{f1_scores.std():.4f}, "
              f"AUC={auc_scores.mean():.4f}±{auc_scores.std():.4f}")

    # Note: CNN is excluded from CV due to training time.
    # For CNN CV, consider reducing epochs or using a simpler architecture.
    return cv_results


# =============================================================================
# STEP 9: VISUALIZATION
# =============================================================================

def generate_plots(results, y_test, feature_names, cv_results, output_dir):
    """Generate all evaluation plots and save to output directory."""

    colors = [MODEL_COLORS[nm] for nm in results.keys()]

    # ---- 1. ROC Curves ----
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (name, (pred, prob, imp)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2, color=colors[i])
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — 5-Method Comparison\nPipe Burst Detection (Real SCADA Data)',
                 fontweight='bold', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_5models.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---- 2. Precision-Recall Curves ----
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (name, (pred, prob, imp)) in enumerate(results.items()):
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        ax.plot(rec, prec, label=f'{name} (AP = {ap:.4f})', linewidth=2, color=colors[i])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — 5-Method Comparison', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves_5models.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---- 3. Confusion Matrices ----
    n_models = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()
    for i, (name, (pred, prob, imp)) in enumerate(results.items()):
        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Incident']).plot(
            ax=axes_flat[i], cmap='Blues', colorbar=False)
        axes_flat[i].set_title(f'{name}', fontweight='bold', fontsize=12)
    # Hide unused subplot
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].axis('off')
    plt.suptitle('Confusion Matrices — 5-Method Comparison',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_5models.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---- 4. Feature Importance (tree-based models only) ----
    fi_models = {nm: (imp, MODEL_COLORS[nm])
                 for nm, (pred, prob, imp) in results.items() if imp is not None}
    if fi_models:
        fig, axes = plt.subplots(1, len(fi_models), figsize=(8 * len(fi_models), 9))
        if len(fi_models) == 1:
            axes = [axes]
        for i, (nm, (imp, col)) in enumerate(fi_models.items()):
            n = 20
            idx = np.argsort(imp)[-n:]
            axes[i].barh(range(n), imp[idx], color=col, alpha=0.8,
                          edgecolor='black', linewidth=0.3)
            axes[i].set_yticks(range(n))
            axes[i].set_yticklabels([feature_names[j] for j in idx], fontsize=9)
            axes[i].set_xlabel('Feature Importance', fontsize=11)
            axes[i].set_title(f'{nm} — Top {n} Features', fontweight='bold', fontsize=13)
            axes[i].grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_5models.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ---- 5. Model Comparison Bar Chart ----
    model_names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_data = {
        'F1-Score': [f1_score(y_test, results[m][0], zero_division=0) for m in model_names],
        'ROC-AUC':  [roc_auc_score(y_test, results[m][1]) for m in model_names],
        'Recall':   [results[m][0][y_test == 1].sum() / max(1, y_test.sum()) for m in model_names],
    }
    for ax_i, (metric_name, values) in enumerate(metrics_data.items()):
        bars = axes[ax_i].bar(range(len(model_names)), values, color=colors, alpha=0.85,
                               edgecolor='black', linewidth=0.5)
        axes[ax_i].set_xticks(range(len(model_names)))
        axes[ax_i].set_xticklabels(model_names, rotation=25, ha='right', fontsize=10)
        axes[ax_i].set_ylabel(metric_name, fontsize=12)
        axes[ax_i].set_title(metric_name, fontweight='bold', fontsize=13)
        axes[ax_i].set_ylim(0, 1.1)
        axes[ax_i].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            axes[ax_i].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                             f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)
    plt.suptitle('Model Performance Comparison — Pipe Burst Detection',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_5models.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  All plots saved to {output_dir}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 65)
    print("  PIPE BURST DETECTION — 5-Method Comparison")
    print("  RF | XGBoost | SVM | CNN | LightGBM")
    print("  Real SCADA Data from Kocaeli Water Distribution Network")
    print("=" * 65)

    # 1. Load incidents
    print("\n[1] Loading incident data...")
    incidents = load_incidents(INCIDENT_FILE)

    # 2. Build pooled dataset
    print("\n[2] Loading and processing DMA sensor data...")
    pooled = build_pooled_dataset(DATA_DIR, incidents, DMA_FILE_MAP)
    print(f"\n  Pooled dataset: {len(pooled)} samples")
    print(f"  Positive (incident): {pooled['label'].sum()}")
    print(f"  Negative (normal):   {(pooled['label']==0).sum()}")
    print(f"  Class ratio: 1:{(pooled['label']==0).sum() // max(1, pooled['label'].sum())}")

    # 3. Prepare X, y and split
    print("\n[3] Preparing feature matrix and splitting data...")
    X, y, X_train, X_test, y_train, y_test, X_bal, y_bal, feature_cols = prepare_data(pooled)
    n_features = len(feature_cols)
    print(f"  Features: {n_features}")
    print(f"  Train (undersampled): {len(X_bal)} ({y_bal.sum()} pos)")
    print(f"  Test: {len(X_test)} ({y_test.sum()} pos)")

    # 4. Train all 5 models
    print("\n[4] Training all 5 models...")
    results = train_all_models(X_bal, y_bal, X_test, y_test, y_train, n_features)

    # 5. Cross-validation (tree-based + SVM)
    print("\n[5] Running 5-fold cross-validation...")
    cv_results = run_cross_validation(X, y)

    # 6. Generate plots
    print("\n[6] Generating evaluation plots...")
    generate_plots(results, y_test, feature_cols, cv_results, OUTPUT_DIR)

    # 7. Summary
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY — 5-METHOD COMPARISON")
    print("=" * 65)
    print(f"\n  Dataset: {len(pooled)} total samples across "
          f"{pooled['dma_id'].nunique()} DMAs")
    print(f"  Incidents: {pooled['label'].sum()} positive windows from "
          f"{len(incidents)} reported events\n")

    for name, (pred, prob, imp) in results.items():
        f1 = f1_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, prob)
        acc = (pred == y_test).mean()
        recall = pred[y_test == 1].sum() / max(1, y_test.sum())
        print(f"  {name:18s}: Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  Recall={recall:.2f}")

    if cv_results:
        print(f"\n  Cross-Validation (5-fold, tree-based + SVM):")
        for name, scores in cv_results.items():
            print(f"    {name:18s}: F1={scores['f1_mean']:.4f}±{scores['f1_std']:.4f}, "
                  f"AUC={scores['auc_mean']:.4f}±{scores['auc_std']:.4f}")

    print("\n" + "=" * 65)
    print(f"  Output files: {OUTPUT_DIR}")
    print("=" * 65)

    return results, cv_results


if __name__ == '__main__':
    results, cv_results = main()
