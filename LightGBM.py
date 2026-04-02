#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LightGBM CTR Model
- Train on FULL train_engineered.csv
- Evaluate log-loss on val_engineered.csv
- Clean, minimal output
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
import lightgbm as lgb


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
    # LightGBM datasets
    # --------------------
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    # --------------------
    # LightGBM parameters (CTR-safe defaults)
    # --------------------
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",

        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,

        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,

        "verbosity": -1
    }

    # --------------------
    # Train
    # --------------------
    print("Training LightGBM on FULL training data...")

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False)
        ]
    )

    # --------------------
    # Evaluate
    # --------------------
    p_val = model.predict(X_val, num_iteration=model.best_iteration)
    ll = log_loss(y_val, p_val, labels=[0, 1])

    print("\n==============================")
    print(f"Validation log loss: {ll}")
    print("==============================")


if __name__ == "__main__":
    main()
