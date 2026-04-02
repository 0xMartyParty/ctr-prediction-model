#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 17:20:58 2025

@author: maxmarte
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XGBoost (xgb.train) on engineered features
- Train: train_engineered.csv
- Validate: val_engineered.csv
- Prints baseline log loss + XGBoost validation log loss
- Uses early stopping correctly (no sklearn wrapper API issues)

If this still underperforms, the most common fix is simply increasing NROWS.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import xgboost as xgb

BASE = "/Users/maxmarte/Desktop/AI final Project"
TRAIN_FILE = os.path.join(BASE, "train_engineered.csv")
VAL_FILE   = os.path.join(BASE, "val_engineered.csv")

# Start here; if your laptop can handle it, try 1_000_000 next
NROWS = 500_000

NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 100

print("Loading engineered data...")
train_df = pd.read_csv(TRAIN_FILE, nrows=NROWS)
val_df   = pd.read_csv(VAL_FILE,   nrows=NROWS)

print(f"Train: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
print(f"Val:   {val_df.shape[0]:,} rows × {val_df.shape[1]} columns")

y_train = train_df["click"].astype(int).to_numpy()
X_train = train_df.drop(columns=["click"]).to_numpy(dtype=float)

y_val = val_df["click"].astype(int).to_numpy()
X_val = val_df.drop(columns=["click"]).to_numpy(dtype=float)

print(f"Train click rate: {y_train.mean():.4f}")
print(f"Val click rate:   {y_val.mean():.4f}")

# ---- Baseline log loss (predict constant p) ----
p_base = float(y_train.mean())
ll_base = log_loss(y_val, np.full_like(y_val, p_base, dtype=float), labels=[0, 1])
print(f"Baseline log loss (predict p={p_base:.4f}): {ll_base:.6f}")

print("Building DMatrix...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

# A strong, fairly safe starting config for log-loss
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",

    # Model complexity (these matter most)
    "max_depth": 5,
    "min_child_weight": 20,
    "gamma": 0.0,

    # Regularization
    "lambda": 5.0,
    "alpha": 0.0,

    # Sampling
    "subsample": 0.8,
    "colsample_bytree": 0.8,

    # Optimization
    "eta": 0.05,               # learning_rate
    "tree_method": "hist",
    "max_bin": 255,

    # Class imbalance handling (helps logloss sometimes)
    "scale_pos_weight": float((y_train == 0).sum() / max(1, (y_train == 1).sum())),

    # Reproducibility (you can remove if you want)
    "seed": 42,
}

evals = [(dtrain, "train"), (dval, "val")]

print("Training XGBoost with early stopping...")
booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=NUM_BOOST_ROUND,
    evals=evals,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=100
)

# Predict using the best iteration (works across xgboost versions)
best_iteration = getattr(booster, "best_iteration", None)
if best_iteration is not None:
    # Use only trees up to best_iteration+1
    p_val = booster.predict(dval, iteration_range=(0, best_iteration + 1))
else:
    p_val = booster.predict(dval)

ll = log_loss(y_val, p_val, labels=[0, 1])

best_score = getattr(booster, "best_score", None)

print("\n==============================")
print("Validation log loss:", ll)
print("Best iteration:", best_iteration)
print("Best val logloss (from training):", best_score)
print("==============================")
