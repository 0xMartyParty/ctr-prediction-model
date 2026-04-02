#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XGBoost train/validation with Early Stopping (works across older/newer xgboost APIs)
- Trains on train_engineered.csv
- Evaluates on val_engineered.csv
- Uses xgboost.train + DMatrix (avoids sklearn wrapper API mismatch)
- Prints baseline log loss and validation log loss
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import xgboost as xgb

BASE = "/Users/maxmarte/Desktop/AI final Project"
TRAIN_FILE = os.path.join(BASE, "train_engineered.csv")
VAL_FILE   = os.path.join(BASE, "val_engineered.csv")

NROWS = 2_000_000  # increase later if you want

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
p_base = y_train.mean()
ll_base = log_loss(y_val, np.full_like(y_val, p_base, dtype=float), labels=[0, 1])
print(f"Baseline log loss (predict p={p_base:.4f}): {ll_base:.6f}")

print("Building DMatrix...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "alpha": 0.0,
    "min_child_weight": 1,
    "tree_method": "hist",
    "seed": 42,
}

num_boost_round = 2000
early_stopping_rounds = 50

print("Training XGBoost with early stopping...")
evals = [(dtrain, "train"), (dval, "val")]

booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=evals,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=50
)

# Predict probabilities on validation (use best_ntree_limit when available)
best_ntree_limit = getattr(booster, "best_ntree_limit", None)
if best_ntree_limit is not None and best_ntree_limit > 0:
    p_val = booster.predict(dval, ntree_limit=best_ntree_limit)
else:
    p_val = booster.predict(dval)

ll = log_loss(y_val, p_val, labels=[0, 1])

best_iteration = getattr(booster, "best_iteration", None)

print("\n==============================")
print("Validation log loss:", ll)
print("Best iteration:", best_iteration)
print("==============================")
