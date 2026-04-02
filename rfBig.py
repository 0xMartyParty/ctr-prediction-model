#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 20:33:30 2025

@author: maxmarte
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest CTR Model (NO class-imbalance adjustments)
- Train on a stratified subset of train_engineered.csv (for feasibility)
- Evaluate log-loss on FULL val_engineered.csv
- Clean, minimal console output

Notes:
- No class_weight, no reweighting, no resampling tricks (per your professor's warning)
"""

import os

# Optional: reduce thread/process shutdown noise on some macOS/Python setups
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


# --------------------
# Paths
# --------------------
BASE = "/Users/maxmarte/Desktop/AI final Project"
TRAIN_FILE = os.path.join(BASE, "train_engineered.csv")
VAL_FILE   = os.path.join(BASE, "val_engineered.csv")


# --------------------
# Controls (EDITABLE)
# --------------------
TRAIN_SUBSET_ROWS = 2_000_000   # try 500_000, 1_000_000, 2_000_000 depending on RAM/time
RANDOM_STATE = 42

# Random Forest hyperparameters (strong, probability-stable defaults)
N_ESTIMATORS = 400
MAX_DEPTH = 18
MIN_SAMPLES_LEAF = 200
MAX_FEATURES = "sqrt"


def main():
    print("Loading engineered datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df   = pd.read_csv(VAL_FILE)

    print(f"Train (full): {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
    print(f"Val:          {val_df.shape[0]:,} rows × {val_df.shape[1]} columns")

    # --------------------
    # Separate X / y
    # --------------------
    y_train_full = train_df["click"].astype(int).to_numpy()
    X_train_full = train_df.drop(columns=["click"]).to_numpy(dtype=np.float32)

    y_val = val_df["click"].astype(int).to_numpy()
    X_val = val_df.drop(columns=["click"]).to_numpy(dtype=np.float32)

    print(f"Train click rate (full): {y_train_full.mean():.4f}")
    print(f"Val click rate:          {y_val.mean():.4f}")

    # --------------------
    # Baseline log-loss
    # --------------------
    p_base = float(y_train_full.mean())
    ll_base = log_loss(
        y_val,
        np.full_like(y_val, p_base, dtype=float),
        labels=[0, 1]
    )
    print(f"Baseline log loss (predict p={p_base:.4f}): {ll_base:.6f}")

    # --------------------
    # Stratified subsample for feasibility
    # --------------------
    print("Subsampling training data for Random Forest...")

    if TRAIN_SUBSET_ROWS is None or TRAIN_SUBSET_ROWS >= X_train_full.shape[0]:
        X_train = X_train_full
        y_train = y_train_full
    else:
        X_train, _, y_train, _ = train_test_split(
            X_train_full,
            y_train_full,
            train_size=TRAIN_SUBSET_ROWS,
            stratify=y_train_full,
            random_state=RANDOM_STATE
        )

    print(f"Train subset: {X_train.shape[0]:,} rows")
    print(f"Subset click rate: {y_train.mean():.4f}")

    # --------------------
    # Train Random Forest (NO class imbalance handling)
    # --------------------
    print("Training Random Forest on training subset...")

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=True,
        n_jobs=1,                 # keep simple/robust on macOS; change to -1 if you want
        random_state=RANDOM_STATE
    )

    rf.fit(X_train, y_train)

    # --------------------
    # Evaluate on FULL validation
    # --------------------
    p_val = rf.predict_proba(X_val)[:, 1]
    ll = log_loss(y_val, p_val, labels=[0, 1])

    print("\n==============================")
    print(f"Validation log loss: {ll}")
    print("==============================")


if __name__ == "__main__":
    main()
