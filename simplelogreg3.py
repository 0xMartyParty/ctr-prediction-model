#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

BASE = "/Users/maxmarte/Desktop/AI final Project"
TRAIN_FILE = os.path.join(BASE, "train_engineered.csv")

NROWS = 1_000_000
TEST_SIZE = 0.2

print("Loading data...")
df = pd.read_csv(TRAIN_FILE, nrows=NROWS)
print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# --- drop target-encoded columns to avoid leakage from your engineered file ---
X_df = df.drop(columns=["click"])
encoded_cols = X_df.columns[X_df.columns.str.endswith("_encoded")]
print(f"Dropping {len(encoded_cols)} target-encoded columns:", list(encoded_cols)[:10], "..." if len(encoded_cols) > 10 else "")
X_df = X_df.loc[:, ~X_df.columns.str.endswith("_encoded")]

y = df["click"].astype(int).to_numpy()
X = X_df.to_numpy(dtype=float)

# random split every run, stratified so click rates match
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y
)

print(f"Train size: {X_train.shape[0]:,}")
print(f"Val size:   {X_val.shape[0]:,}")
print(f"Train click rate: {y_train.mean():.4f}")
print(f"Val click rate:   {y_val.mean():.4f}")

# baseline logloss (constant p)
p_base = y_train.mean()
ll_base = log_loss(y_val, np.full_like(y_val, p_base, dtype=float), labels=[0, 1])
print(f"Baseline log loss (predict p={p_base:.4f} for all): {ll_base:.6f}")

# scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

print("Training logistic regression...")
model = LogisticRegression(max_iter=1000, solver="lbfgs")
model.fit(X_train_s, y_train)

p_val = model.predict_proba(X_val_s)[:, 1]
ll = log_loss(y_val, p_val, labels=[0, 1])

print("\n==============================")
print("Hold-out log loss:", ll)
print("==============================")

