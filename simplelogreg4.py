#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 16:30:55 2025

@author: maxmarte
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

BASE = "/Users/maxmarte/Desktop/AI final Project"

TRAIN_FILE = os.path.join(BASE, "train_engineered.csv")
VAL_FILE   = os.path.join(BASE, "val_engineered.csv")

NROWS = 500_000  # use a subset to keep it fast

print("Loading engineered data...")

train_df = pd.read_csv(TRAIN_FILE, nrows=NROWS)
val_df   = pd.read_csv(VAL_FILE,   nrows=NROWS)

print(f"Train: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
print(f"Val:   {val_df.shape[0]:,} rows × {val_df.shape[1]} columns")

# Separate target and features
y_train = train_df["click"].astype(int).to_numpy()
X_train = train_df.drop(columns=["click"]).to_numpy(dtype=float)

y_val = val_df["click"].astype(int).to_numpy()
X_val = val_df.drop(columns=["click"]).to_numpy(dtype=float)

print(f"Train click rate: {y_train.mean():.4f}")
print(f"Val click rate:   {y_val.mean():.4f}")

# ---- baseline log-loss ----
p_base = y_train.mean()
ll_base = log_loss(
    y_val,
    np.full_like(y_val, p_base, dtype=float),
    labels=[0, 1]
)
print(f"Baseline log loss (predict p={p_base:.4f}): {ll_base:.6f}")

# ---- scale features (fit on TRAIN ONLY) ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

print("Training logistic regression...")

model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

model.fit(X_train_s, y_train)

# ---- evaluate on validation ----
p_val = model.predict_proba(X_val_s)[:, 1]
ll = log_loss(y_val, p_val, labels=[0, 1])

print("\n==============================")
print("Validation log loss:", ll)
print("==============================")
