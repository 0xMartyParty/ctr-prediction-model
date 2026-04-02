#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 17:28:44 2025

@author: maxmarte
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final Logistic Regression
- Train on ALL rows of train_engineered.csv
- Evaluate log-loss on val_engineered.csv
- No randomness, no tuning, no test usage
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

BASE = "/Users/maxmarte/Desktop/AI final Project"
TRAIN_FILE = os.path.join(BASE, "train_engineered.csv")
VAL_FILE   = os.path.join(BASE, "val_engineered.csv")

print("Loading engineered datasets...")
train_df = pd.read_csv(TRAIN_FILE)
val_df   = pd.read_csv(VAL_FILE)

print(f"Train: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
print(f"Val:   {val_df.shape[0]:,} rows × {val_df.shape[1]} columns")

# --------------------
# Separate X / y
# --------------------
y_train = train_df["click"].astype(int).to_numpy()
X_train = train_df.drop(columns=["click"]).to_numpy(dtype=float)

y_val = val_df["click"].astype(int).to_numpy()
X_val = val_df.drop(columns=["click"]).to_numpy(dtype=float)

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
# Scale features (FIT ON TRAIN ONLY)
# --------------------
print("Scaling features...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

# --------------------
# Train Logistic Regression
# --------------------
print("Training logistic regression on FULL training data...")

model = LogisticRegression(
    penalty="l2",
    C=0.5,              # slightly stronger regularization
    solver="lbfgs",
    max_iter=2000,
    n_jobs=None
)

model.fit(X_train_s, y_train)

# --------------------
# Evaluate on validation
# --------------------
p_val = model.predict_proba(X_val_s)[:, 1]
ll = log_loss(y_val, p_val, labels=[0, 1])

print("\n==============================")
print("Validation log loss:", ll)
print("==============================")
