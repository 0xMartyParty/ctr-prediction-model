import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# ---- paths ----
BASE = "/Users/maxmarte/Desktop/AI final Project"
TRAIN_PATH = os.path.join(BASE, "ProjectTrainingData.csv")

# ---- settings ----
NROWS = 20_000_000      # small enough to run, big enough to learn
TEST_SIZE = 0.2        # 80% train, 20% hold-out
N_FEATURES = 2**18
RANDOM_SEED = 42

# 1) Load part of the training data
df = pd.read_csv(TRAIN_PATH, nrows=NROWS)

# 2) Split into train / hold-out
train_df, val_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=df["click"]
)

# 3) Separate X and y
y_train = train_df["click"].astype(int).values
X_train_df = train_df.drop(columns=["click"]).astype(str)

y_val = val_df["click"].astype(int).values
X_val_df = val_df.drop(columns=["click"]).astype(str)

# 4) Convert rows to tokens for hashing
def to_tokens(X_df):
    return X_df.apply(
        lambda r: [f"{c}={r[c]}" for c in X_df.columns],
        axis=1
    )

train_tokens = to_tokens(X_train_df)
val_tokens = to_tokens(X_val_df)

# 5) Hash features
hasher = FeatureHasher(n_features=N_FEATURES, input_type="string")
X_train = hasher.transform(train_tokens)
X_val = hasher.transform(val_tokens)

# 6) Train model
model = SGDClassifier(
    loss="log_loss",
    alpha=1e-6,
    random_state=RANDOM_SEED
)
model.fit(X_train, y_train)

# 7) Evaluate ONCE on hold-out sample
p_val = model.predict_proba(X_val)[:, 1]
ll = log_loss(y_val, p_val, labels=[0, 1])

print("Hold-out log loss:", ll)
p_base = y_train.mean()
ll_base = log_loss(y_val, np.full_like(y_val, p_base, dtype=float), labels=[0,1])
print("Baseline (constant p) logloss:", ll_base, "p=", p_base)

