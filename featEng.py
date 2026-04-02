import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE ENGINEERING FOR AD CLICK PREDICTION")
print("=" * 80)
import os

BASE_DIR = "/Users/maxmarte/Desktop/AI final Project"
OUTPUT_TRAIN    = os.path.join(BASE_DIR, "train_engineered.csv")
OUTPUT_TEST     = os.path.join(BASE_DIR, "test_engineered.csv")
OUTPUT_ENCODERS = os.path.join(BASE_DIR, "feature_encoders.pkl")

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
TRAIN_FILE = "/Users/maxmarte/Desktop/AI final Project/ProjectTrainingData.csv"
TEST_FILE = "/Users/maxmarte/Desktop/AI final Project/ProjectTestData.csv"

# Output paths
OUTPUT_TRAIN = "train_engineered.csv"
OUTPUT_TEST = "test_engineered.csv"
OUTPUT_ENCODERS = "feature_encoders.pkl"

# Feature engineering parameters
RARE_THRESHOLD = 0.001  # Categories appearing <0.1% will be grouped as 'rare'
TARGET_ENCODE_FEATURES = ['site_id', 'site_domain', 'app_id', 'app_domain',
                          'device_model', 'device_id', 'device_ip']
FREQUENCY_ENCODE_FEATURES = ['site_category', 'app_category']
ONEHOT_ENCODE_FEATURES = ['C1', 'banner_pos', 'device_type', 'device_conn_type']

print("\n[1] LOADING DATA...")
print("This may take a few minutes with 32M rows...")

# Load full training data
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"✓ Training: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
print(f"✓ Test: {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")

# Store target variable
y_train = train_df['click'].copy()
train_df = train_df.drop('click', axis=1)

print(f"\nTarget distribution:")
print(f"  No Click: {(y_train == 0).sum():,} ({(y_train == 0).mean() * 100:.2f}%)")
print(f"  Click:    {(y_train == 1).sum():,} ({(y_train == 1).mean() * 100:.2f}%)")

# ============================================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("[2] TEMPORAL FEATURE ENGINEERING")
print("=" * 80)


def extract_temporal_features(df):
    """Extract temporal features from hour column (format: YYMMDDHH)"""
    df = df.copy()

    # Convert to string with padding
    df['hour_str'] = df['hour'].astype(str).str.zfill(8)

    # Extract components
    df['year'] = df['hour_str'].str[0:2].astype(int)
    df['month'] = df['hour_str'].str[2:4].astype(int)
    df['day'] = df['hour_str'].str[4:6].astype(int)
    df['hour_of_day'] = df['hour_str'].str[6:8].astype(int)

    # Create derived features
    df['is_weekend'] = df['day'].isin([6, 7]).astype(int)  # Assuming 6, 7 are weekend

    # Cyclical encoding for hour (important for temporal continuity)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

    # Cyclical encoding for day
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    # Time slot (based on distribution analysis showing night has higher CTR)
    df['time_slot'] = pd.cut(df['hour_of_day'],
                             bins=[-1, 6, 12, 18, 24],
                             labels=[0, 1, 2, 3])  # Night, Morning, Afternoon, Evening
    df['time_slot'] = df['time_slot'].astype(int)

    # Drop the string column
    df = df.drop('hour_str', axis=1)

    return df


print("Extracting temporal features...")
train_df = extract_temporal_features(train_df)
test_df = extract_temporal_features(test_df)

print("✓ Created features:")
print("  - year, month, day, hour_of_day")
print("  - is_weekend, time_slot")
print("  - hour_sin, hour_cos (cyclical encoding)")
print("  - day_sin, day_cos (cyclical encoding)")

# ============================================================================
# CATEGORICAL FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("[3] CATEGORICAL FEATURE ENGINEERING")
print("=" * 80)

# Dictionary to store encoders
encoders = {}

# 3.1 TARGET ENCODING (for high cardinality features)
print("\n[3.1] Target Encoding (high cardinality features)...")


def target_encode_with_smoothing(train_df, test_df, y_train, feature, alpha=10):
    """
    Target encoding with smoothing to prevent overfitting
    Uses global mean and category mean weighted by sample size
    """
    # Calculate global mean
    global_mean = y_train.mean()

    # Calculate category statistics
    category_stats = pd.DataFrame({
        'category': train_df[feature],
        'target': y_train
    }).groupby('category')['target'].agg(['mean', 'count'])

    # Smooth the encoding (add-k smoothing)
    category_stats['encoded'] = (
            (category_stats['mean'] * category_stats['count'] + global_mean * alpha) /
            (category_stats['count'] + alpha)
    )

    # Create encoding dictionary
    encoding_dict = category_stats['encoded'].to_dict()

    # Apply encoding
    train_encoded = train_df[feature].map(encoding_dict).fillna(global_mean)
    test_encoded = test_df[feature].map(encoding_dict).fillna(global_mean)

    return train_encoded, test_encoded, encoding_dict


for feature in TARGET_ENCODE_FEATURES:
    if feature in train_df.columns:
        print(f"  Encoding {feature}...")
        train_encoded, test_encoded, encoding_dict = target_encode_with_smoothing(
            train_df, test_df, y_train, feature
        )

        # Replace original feature
        train_df[f'{feature}_encoded'] = train_encoded
        test_df[f'{feature}_encoded'] = test_encoded

        # Store encoder
        encoders[f'{feature}_target'] = encoding_dict

        # Drop original (keep encoded version)
        train_df = train_df.drop(feature, axis=1)
        test_df = test_df.drop(feature, axis=1)

        print(f"    ✓ {feature}_encoded created (unique values: {len(encoding_dict):,})")

# 3.2 FREQUENCY ENCODING (for medium cardinality features)
print("\n[3.2] Frequency Encoding (medium cardinality features)...")


def frequency_encode(train_df, test_df, feature):
    """Count-based encoding"""
    freq_dict = train_df[feature].value_counts().to_dict()

    train_encoded = train_df[feature].map(freq_dict).fillna(0)
    test_encoded = test_df[feature].map(freq_dict).fillna(0)

    return train_encoded, test_encoded, freq_dict


for feature in FREQUENCY_ENCODE_FEATURES:
    if feature in train_df.columns:
        print(f"  Encoding {feature}...")
        train_encoded, test_encoded, freq_dict = frequency_encode(train_df, test_df, feature)

        train_df[f'{feature}_freq'] = train_encoded
        test_df[f'{feature}_freq'] = test_encoded

        encoders[f'{feature}_freq'] = freq_dict

        # Keep both original and encoded for now
        print(f"    ✓ {feature}_freq created")

# 3.3 LABEL ENCODING (for low cardinality that we'll one-hot later)
print("\n[3.3] Label Encoding (low/medium cardinality features)...")

for feature in ONEHOT_ENCODE_FEATURES:
    if feature in train_df.columns:
        print(f"  Encoding {feature}...")

        # Combine train and test to ensure same categories
        combined = pd.concat([
            train_df[feature].astype(str),
            test_df[feature].astype(str)
        ])

        le = LabelEncoder()
        le.fit(combined)

        train_df[f'{feature}_label'] = le.transform(train_df[feature].astype(str))
        test_df[f'{feature}_label'] = le.transform(test_df[feature].astype(str))

        encoders[f'{feature}_label'] = le

        print(f"    ✓ {feature}_label created ({len(le.classes_)} categories)")

# 3.4 ONE-HOT ENCODING (create dummy variables)
print("\n[3.4] One-Hot Encoding...")

for feature in ONEHOT_ENCODE_FEATURES:
    label_feature = f'{feature}_label'
    if label_feature in train_df.columns:
        print(f"  Creating dummies for {feature}...")

        # Get unique values
        n_categories = train_df[label_feature].nunique()

        # One-hot encode (drop_first to avoid multicollinearity)
        train_dummies = pd.get_dummies(train_df[label_feature],
                                       prefix=feature,
                                       drop_first=True)
        test_dummies = pd.get_dummies(test_df[label_feature],
                                      prefix=feature,
                                      drop_first=True)

        # Align columns (in case test has fewer categories)
        test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

        # Add to dataframes
        train_df = pd.concat([train_df, train_dummies], axis=1)
        test_df = pd.concat([test_df, test_dummies], axis=1)

        # Drop label encoded version
        train_df = train_df.drop(label_feature, axis=1)
        test_df = test_df.drop(label_feature, axis=1)

        print(f"    ✓ Created {len(train_dummies.columns)} dummy variables")

# ============================================================================
# INTERACTION FEATURES
# ============================================================================

print("\n" + "=" * 80)
print("[4] INTERACTION FEATURES")
print("=" * 80)

print("Creating interaction features based on insights...")

# Based on distribution analysis: hour × device_type interaction
if 'hour_of_day' in train_df.columns and 'device_type_label' in train_df.columns:
    print("  - hour_device interaction")
    # This will be captured by the model, but we can create explicit interaction
    # For simplicity, we'll let tree-based models learn this automatically

# App category × site category (content matching)
if 'app_category_freq' in train_df.columns and 'site_category_freq' in train_df.columns:
    print("  - app_site_interaction")
    train_df['app_site_interaction'] = train_df['app_category_freq'] * train_df['site_category_freq']
    test_df['app_site_interaction'] = test_df['app_category_freq'] * test_df['site_category_freq']

print("✓ Interaction features created")

# ============================================================================
# HANDLE MISSING VALUES & CLEANUP
# ============================================================================

print("\n" + "=" * 80)
print("[5] FINAL CLEANUP")
print("=" * 80)

# Drop original categorical columns that we've encoded
original_cats_to_drop = FREQUENCY_ENCODE_FEATURES + ONEHOT_ENCODE_FEATURES
for col in original_cats_to_drop:
    if col in train_df.columns:
        train_df = train_df.drop(col, axis=1)
        test_df = test_df.drop(col, axis=1)

# Drop the original 'hour' column (we have extracted features)
if 'hour' in train_df.columns:
    train_df = train_df.drop('hour', axis=1)
    test_df = test_df.drop('hour', axis=1)

# Check for any remaining missing values
print("\nChecking for missing values...")
train_missing = train_df.isnull().sum().sum()
test_missing = test_df.isnull().sum().sum()

if train_missing > 0:
    print(f"⚠ Training has {train_missing} missing values - filling with 0")
    train_df = train_df.fillna(0)

if test_missing > 0:
    print(f"⚠ Test has {test_missing} missing values - filling with 0")
    test_df = test_df.fillna(0)

# Ensure all features are numeric
print("\nEnsuring all features are numeric...")
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        print(f"  Converting {col} to numeric")
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)

# ============================================================================
# SAVE ENGINEERED DATA
# ============================================================================

print("\n" + "=" * 80)
print("[6] SAVING ENGINEERED FEATURES")
print("=" * 80)

print("\nFinal dataset shapes:")
print(f"  Train: {train_df.shape[0]:,} rows × {train_df.shape[1]} features")
print(f"  Test:  {test_df.shape[0]:,} rows × {test_df.shape[1]} features")
print(f"  Target: {len(y_train):,} labels")

# Add target back to training data for saving
train_df['click'] = y_train

print(f"\nSaving to files...")
print(f"  - {OUTPUT_TRAIN}")
print(f"  - {OUTPUT_TEST}")
print(f"  - {OUTPUT_ENCODERS}")

# Save as CSV (may take a while with 32M rows)
print("\nWriting training data (this may take several minutes)...")
train_df.to_csv(OUTPUT_TRAIN, index=False)
print("✓ Training data saved")

print("\nWriting test data...")
test_df.to_csv(OUTPUT_TEST, index=False)
print("✓ Test data saved")

# Save encoders for future use
with open(OUTPUT_ENCODERS, 'wb') as f:
    pickle.dump(encoders, f)
print("✓ Encoders saved")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("[7] FEATURE ENGINEERING SUMMARY")
print("=" * 80)

print(f"\nOriginal features: {TRAIN_FILE}")
print(f"  - Training: 27 columns")
print(f"  - Test: 23 columns")

print(f"\nEngineered features: {OUTPUT_TRAIN}")
print(f"  - Training: {train_df.shape[1]} columns (including target)")
print(f"  - Test: {test_df.shape[1]} columns")

print("\nFeature types created:")
print(f"  ✓ Temporal features: 10 (hour, day, cyclical encodings, time slots)")
print(f"  ✓ Target encoded: {len(TARGET_ENCODE_FEATURES)} high-cardinality features")
print(f"  ✓ Frequency encoded: {len(FREQUENCY_ENCODE_FEATURES)} medium-cardinality features")
print(f"  ✓ One-hot encoded: {len(ONEHOT_ENCODE_FEATURES)} low-cardinality features")
print(f"  ✓ Interaction features: app×site interaction")

print("\nFeature list:")
feature_list = [col for col in train_df.columns if col != 'click']
for i, col in enumerate(feature_list, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE!")
print("=" * 80)
print("\nNext steps:")
print("  1. Load train_engineered.csv and test_engineered.csv")
print("  2. Split training data (80/20 train/validation)")
print("  3. Train models (LightGBM, XGBoost, Logistic Regression)")
print("  4. Evaluate on validation set")
print("  5. Make predictions on test set")