#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Logistic Regression (clean output, avoids multiprocessing resource_tracker warnings)
- Train on FULL train_engineered.csv
- Evaluate log-loss on val_engineered.csv
"""

import os

# Limit native thread pools to avoid multiprocessing/resource_tracker shutdown noise on some setups
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


# --------------------
# Paths
# --------------------
BASE = "/Users/maxmarte/Desktop/AI final Project"
TRAIN_FILE = os.path.join(BASE, "train_engineered.csv")
VAL_FILE   = os.path.join(BASE, "val_engineered.csv")


def main():
    print("Loading engineered datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df   = pd.read_csv(VAL_FILE)

    print(f"Train: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
    print(f"Val:   {val_df.shape[0]:,} rows × {val_df.shape[1]} columns")

    # --------------------
    # Separate X / y
    # --------------------
    y_train = train_df["click"].astype(int).to_numpy()
    X_train = train_df.drop(columns=["click"]).to_numpy(dtype=np.float32)

    y_val = val_df["click"].astype(int).to_numpy()
    X_val = val_df.drop(columns=["click"]).to_numpy(dtype=np.float32)

    print(f"Train click rate: {y_train.mean():.4f}")
    print(f"Val click rate:   {y_val.mean():.4f}")

    # --------------------
    # Baseline log-loss
    # --------------------
    p_base = float(y_train.mean())
    ll_base = log_loss(
        y_val,
        np.full_like(y_val, p_base, dtype=float),
        labels=[0, 1]
    )
    print(f"Baseline log loss (predict p={p_base:.4f}): {ll_base:.6f}")

    # --------------------
    # Scale features (fit on train only)
    # --------------------
    print("Scaling features...")
    scaler = StandardScaler(copy=False)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # --------------------
    # Train Logistic Regression
    # --------------------
    print("Training logistic regression on FULL training data...")

    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=1.0,
        max_iter=1000,
        n_jobs=1,          # IMPORTANT: avoids multiprocessing/threadpool shutdown warnings on some setups
        random_state=0
    )
    model.fit(X_train_s, y_train)

    # --------------------
    # Evaluate on validation
    # --------------------
    p_val = model.predict_proba(X_val_s)[:, 1]
    ll = log_loss(y_val, p_val, labels=[0, 1])

    print("\n==============================")
    print(f"Validation log loss: {ll}")
    print("==============================")


if __name__ == "__main__":
    main()
