#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 12:49:39 2025

@author: maxmarte
"""



import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# ---- Paths (edit TEAM_NUM) ----
BASE = "/Users/maxmarte/Desktop/AI final Project"
TEAM_NUM = "X"  # <-- change to your team number later

TRAIN_PATH = os.path.join(BASE, "ProjectTrainingData.csv")
TEST_PATH  = os.path.join(BASE, "ProjectTestData.csv")
SUB_PATH   = os.path.join(BASE, f"ProjectSubmission-Team{TEAM_NUM}.csv")

OUT_SUB_PATH = os.path.join(BASE, f"ProjectSubmission-Team{TEAM_NUM}.csv")  # overwritten with predictions

# ---- Settings ----
CHUNK_SIZE = 250_000           # adjust up/down based on your RAM
N_FEATURES = 2**20             # 1,048,576 hashed features (common sweet spot)
RANDOM_SEED = 42

# Columns: test has all except "click"
# We'll read header from train file automatically
train_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
y_col = "click"
x_cols = [c for c in train_cols if c != y_col]

hasher = FeatureHasher(n_features=N_FEATURES, input_type="string")

# SGD logistic regression for massive sparse data
clf = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-6,
    learning_rate="optimal",
    random_state=RANDOM_SEED
)

def row_to_tokens(df: pd.DataFrame) -> list[list[str]]:
    """
    Convert each row to tokens like "col=value" so hashing works well.
    All treated as categorical.
    """
    # ensure strings
    df = df.astype(str)
    tokens = []
    for _, r in df.iterrows():
        tokens.append([f"{c}={r[c]}" for c in df.columns])
    return tokens

# ---- 1) Train incrementally on chunks ----
print("Training in chunks...")
classes = np.array([0, 1])

first_fit = True
val_losses = []

for chunk in pd.read_csv(TRAIN_PATH, chunksize=CHUNK_SIZE):
    y = chunk[y_col].astype(int).values
    X_df = chunk[x_cols]

    # quick holdout split INSIDE chunk (keeps memory small)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_df, y, test_size=0.1, random_state=RANDOM_SEED, stratify=y
    )

    X_tr_h = hasher.transform(row_to_tokens(X_tr))
    X_va_h = hasher.transform(row_to_tokens(X_va))

    if first_fit:
        clf.partial_fit(X_tr_h, y_tr, classes=classes)
        first_fit = False
    else:
        clf.partial_fit(X_tr_h, y_tr)

    # evaluate on validation slice for feedback
    p_va = clf.predict_proba(X_va_h)[:, 1]
    loss = log_loss(y_va, p_va, labels=[0, 1])
    val_losses.append(loss)
    print(f"Chunk val logloss: {loss:.5f} | running avg: {np.mean(val_losses):.5f}")

print("Done training. Avg val logloss:", float(np.mean(val_losses)))

# ---- 2) Predict on test in chunks (keep row order) ----
print("Predicting on test...")
preds = []

for chunk in pd.read_csv(TEST_PATH, chunksize=CHUNK_SIZE):
    X_h = hasher.transform(row_to_tokens(chunk[x_cols]))
    p = clf.predict_proba(X_h)[:, 1]
    preds.append(p)

preds = np.concatenate(preds)
print("Total predictions:", preds.shape[0])

# ---- 3) Fill submission template ----
sub = pd.read_csv(SUB_PATH)
if "P(click)" not in sub.columns:
    raise ValueError("Submission file must have column named 'P(click)'")

if len(sub) != len(preds):
    raise ValueError(f"Row count mismatch: submission={len(sub)} test_preds={len(preds)}")

sub["P(click)"] = preds
sub.to_csv(OUT_SUB_PATH, index=False)
print("Wrote:", OUT_SUB_PATH)

