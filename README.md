CTR Prediction Model

Overview
This project focuses on predicting the probability that a user will click on an online advertisement (Click-Through Rate, CTR). The model is trained on a large-scale dataset containing over 30 million observations of ad impressions and user/device attributes.

The goal is to generate accurate probability predictions that minimize log loss, a standard evaluation metric for classification problems involving probabilities.

Problem Statement
Given historical ad impression data, predict the likelihood that a displayed advertisement will be clicked.

- Target variable: `click` (1 = clicked, 0 = not clicked)
- Evaluation metric: **Log Loss** (lower is better)

As defined in the project:  

---

Dataset
The dataset consists of:

- **Training Data** (~30M rows)
- **Test Data** (~13M rows)

### Key Features:
- `hour`: timestamp of ad impression
- `site_id`, `site_domain`, `site_category`
- `app_id`, `app_domain`, `app_category`
- `device_id`, `device_ip`, `device_model`
- `device_type`, `device_conn_type`
- `C1`, `C14–C21`: anonymized categorical variables

---

Feature Engineering
Significant preprocessing and feature engineering were performed:

- Time-based features:
  - Hour of day
  - Day of week
  - Weekend indicator
  - Cyclical encoding (sin/cos)

- Encoding strategies:
  - **Target encoding** for high-cardinality features (e.g., device_id, site_id)
  - **Frequency encoding** for medium-cardinality features
  - **One-hot encoding** for low-cardinality features

- Interaction features:
  - App category × Site category
  - Device and time interactions

---

Models Used
Multiple models were explored:

- Logistic Regression (baseline)
- SGD Classifier with feature hashing
- **LightGBM (primary model)**
- XGBoost (tuned)

Final Model:
LightGBM achieved the best performance on validation data.

---

Results
- Baseline Log Loss: ~0.456
- Final Model Log Loss: ~0.40

This represents a significant improvement over a naive baseline model.

---

Tech Stack
- Python
- Pandas / NumPy
- Scikit-learn
- LightGBM
- XGBoost
