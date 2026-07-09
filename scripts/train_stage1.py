"""
train_stage1.py
---------------
Stage 1: Pre-Screening Model (Reject Inference)

Builds a pre-screening model using both accepted and rejected loan data.
Rejected applicants have no real default labels, so their labels are inferred
via a proxy Logistic Regression model trained on accepted data (fuzzy augmentation).
The final Stage 1 model is an XGBoost classifier.

Usage:
    python scripts/train_stage1.py

Outputs:
    models/stage1_prescreen_model.pkl — XGBoost pre-screening pipeline
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths  (anchored to project root — works from any working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
MODEL_PKL    = os.path.join(MODELS_DIR, "stage1_prescreen_model.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

STAGE1_FEATURES = ["loan_amnt", "fico", "dti", "emp_length"]

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
print("Loading data...")
df_model     = pd.read_csv(os.path.join(DATA_DIR, "final_model_data.csv"))
rejected_df  = pd.read_csv(os.path.join(DATA_DIR, "cleaned_rejected_loans.csv"))

# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), STAGE1_FEATURES),
])

# ---------------------------------------------------------------------------
# 3. Proxy Model — trained on accepted data to infer rejected labels
# ---------------------------------------------------------------------------
print("Training proxy model on accepted data...")
proxy_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=42
    )),
])
proxy_model.fit(df_model[STAGE1_FEATURES], df_model["is_default"])

# ---------------------------------------------------------------------------
# 4. Reject Inference — assign synthetic labels to rejected applicants
# ---------------------------------------------------------------------------
print("Inferring labels for rejected applicants...")
X_rejected      = rejected_df[STAGE1_FEATURES].fillna(rejected_df[STAGE1_FEATURES].median())
rejected_probs  = proxy_model.predict_proba(X_rejected)[:, 1]
rejected_df["is_default"] = (rejected_probs >= 0.50).astype(int)

print(f"  Rejected → predicted default rate: {rejected_df['is_default'].mean():.2%}")

# ---------------------------------------------------------------------------
# 5. Combine Accepted + Rejected for Stage 1 Training
# ---------------------------------------------------------------------------
accepted_subset = df_model[STAGE1_FEATURES].copy()
accepted_subset["is_default"] = df_model["is_default"].values

rejected_subset = rejected_df[STAGE1_FEATURES].copy()
rejected_subset["is_default"] = rejected_df["is_default"].values

combined = pd.concat([accepted_subset, rejected_subset], ignore_index=True)
X_stage1 = combined[STAGE1_FEATURES]
y_stage1 = combined["is_default"]

print(f"  Combined dataset size: {len(combined):,} rows")

# ---------------------------------------------------------------------------
# 6. Train Final Stage 1 Pre-Screening Model
# ---------------------------------------------------------------------------
print("Training Stage 1 XGBoost model...")
stage1_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=4,
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )),
])
stage1_model.fit(X_stage1, y_stage1)

# ---------------------------------------------------------------------------
# 7. Save Model
# ---------------------------------------------------------------------------
joblib.dump(stage1_model, MODEL_PKL)
print(f"\nStage 1 pre-screening model saved → {MODEL_PKL}")
print("Done.")
