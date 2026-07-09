"""
train_stage2.py
---------------
Stage 2: Main Credit Risk Model Training

Trains an XGBoost classifier on accepted loan data to predict the probability
of default. Includes feature engineering, preprocessing, model selection,
probability calibration, evaluation, and artifact export.

Usage:
    python scripts/train_stage2.py

Outputs:
    models/credit_risk_model.pkl     — Calibrated XGBoost pipeline
    data/portfolio_predictions.csv   — Test-set predictions with risk bands
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths  (anchored to project root — works from any working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
INPUT_CSV    = os.path.join(DATA_DIR,   "cleaned_loans.csv")
OUTPUT_CSV   = os.path.join(DATA_DIR,   "portfolio_predictions.csv")
MODEL_PKL    = os.path.join(MODELS_DIR, "credit_risk_model.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(INPUT_CSV)

# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------
print("Engineering features...")

df["loan_income_ratio"] = (df["loan_amnt"] / df["annual_inc"]).round(4)

df["dti_bucket"] = pd.cut(
    df["dti"],
    bins=[-1, 10, 20, 30, 40, 100],
    labels=["Very Low", "Low", "Moderate", "High", "Very High"],
)

df["income_bucket"] = pd.cut(
    df["annual_inc"],
    bins=[0, 40_000, 80_000, 120_000, 200_000, np.inf],
    labels=["Low", "Lower Middle", "Middle", "Upper Middle", "High"],
)

df["fico_bucket"] = pd.cut(
    df["fico"],
    bins=[300, 580, 670, 740, 800, 850],
    labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
)

df["revol_util_bucket"] = pd.cut(
    df["revol_util"],
    bins=[0, 30, 50, 75, 100],
    labels=["Low", "Moderate", "High", "Critical"],
)

df["loan_amount_bucket"] = pd.cut(
    df["loan_amnt"],
    bins=[0, 5_000, 10_000, 20_000, 40_000, np.inf],
    labels=["Small", "Medium", "Large", "Very Large", "Premium"],
)

df["employment_category"] = np.select(
    [
        df["emp_length"] == -1,
        df["emp_length"] <= 1,
        df["emp_length"].between(2, 5),
        df["emp_length"] > 5,
    ],
    ["Unknown", "New", "Mid Career", "Experienced"],
    default="Experienced",
)

df["term_category"] = df["term"].map({36: "Short Term", 60: "Long Term"})

df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
df["issue_d"]          = pd.to_datetime(df["issue_d"],          errors="coerce")
df["credit_history_years"] = (
    (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365
).round(1)

df["risk_segment"] = np.select(
    [
        (df["fico"] >= 740) & (df["dti"] < 20),
        (df["fico"] >= 670) & (df["dti"] < 30),
        (df["fico"] >= 580) & (df["dti"] < 40),
    ],
    ["Low Risk", "Medium Risk", "High Risk"],
    default="Very High Risk",
)

# Fill categorical NaNs
for col in ["dti_bucket", "income_bucket", "revol_util_bucket"]:
    df[col] = df[col].cat.add_categories("Unknown").fillna("Unknown")

# Replace infinities
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ---------------------------------------------------------------------------
# 3. Target Variable
# ---------------------------------------------------------------------------
df["is_default"] = np.where(df["loan_status"] == "Fully Paid", 0, 1)

print(f"Class distribution:\n{df['is_default'].value_counts(normalize=True).round(3)}\n")

# ---------------------------------------------------------------------------
# 4. Stratified Sample (300k rows for tractable training)
# ---------------------------------------------------------------------------
df_model, _ = train_test_split(
    df,
    train_size=300_000,
    stratify=df["is_default"],
    random_state=42,
)

# ---------------------------------------------------------------------------
# 5. Features & Train/Test Split
# ---------------------------------------------------------------------------
FEATURES = [
    "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
    "purpose", "dti", "delinq_2yrs", "revol_util", "revol_bal",
    "open_acc", "total_acc", "pub_rec", "fico",
    "loan_income_ratio", "dti_bucket", "income_bucket", "fico_bucket",
    "revol_util_bucket", "loan_amount_bucket", "employment_category",
    "term_category", "credit_history_years",
]

X = df_model[FEATURES]
y = df_model["is_default"]

numeric_features     = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}\n")

# ---------------------------------------------------------------------------
# 6. Preprocessing Pipeline
# ---------------------------------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline,     numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

# ---------------------------------------------------------------------------
# 7. Model Comparison
# ---------------------------------------------------------------------------
MODELS = {
    "Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        learning_rate=0.05, max_depth=6, n_estimators=200,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    ),
}

results        = []
trained_models = {}

for name, model in MODELS.items():
    print(f"Training {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   model),
    ])
    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    auc    = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.50

    results.append({
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1 Score":  f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC":   auc,
    })

comparison = pd.DataFrame(results).sort_values("ROC AUC", ascending=False)
print(f"\nModel Comparison:\n{comparison.to_string(index=False)}\n")

# ---------------------------------------------------------------------------
# 8. Champion Model Selection & Probability Calibration
# ---------------------------------------------------------------------------
best_model_name = comparison.iloc[0]["Model"]
best_model      = trained_models[best_model_name]
print(f"Champion Model: {best_model_name}")

print(classification_report(y_test, best_model.predict(X_test)))

print("Calibrating probabilities...")
X_test_processed = best_model.named_steps["preprocessor"].transform(X_test)

calibrated_classifier = CalibratedClassifierCV(
    estimator=FrozenEstimator(best_model.named_steps["classifier"]),
    method="isotonic",
    cv=None,
)
calibrated_classifier.fit(X_test_processed, y_test)
best_model.steps[-1] = ("classifier", calibrated_classifier)
print("Calibration complete.\n")

# ---------------------------------------------------------------------------
# 9. Save Model
# ---------------------------------------------------------------------------
joblib.dump(best_model, MODEL_PKL)
print(f"Champion model saved → {MODEL_PKL}")

# ---------------------------------------------------------------------------
# 10. Threshold Tuning & Portfolio Export
# ---------------------------------------------------------------------------
y_prob = best_model.predict_proba(X_test)[:, 1]

threshold_results = []
for t in [0.30, 0.40, 0.50, 0.60]:
    pred = (y_prob >= t).astype(int)
    threshold_results.append({
        "Threshold": t,
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall":    recall_score(y_test, pred, zero_division=0),
        "F1":        f1_score(y_test, pred, zero_division=0),
    })

threshold_df   = pd.DataFrame(threshold_results)
best_threshold = threshold_df.loc[threshold_df["F1"].idxmax(), "Threshold"]
print(f"\nThreshold Tuning:\n{threshold_df.to_string(index=False)}")
print(f"\nBest Threshold: {best_threshold}")

portfolio = X_test.copy()
portfolio["Actual_Default"] = y_test.values
portfolio["Probability"]    = y_prob
portfolio["Prediction"]     = (y_prob >= best_threshold).astype(int)
portfolio["Risk_Band"]      = pd.cut(
    portfolio["Probability"],
    bins=[0, 0.20, 0.40, 0.60, 0.80, 1.0],
    labels=["Very Low", "Low", "Moderate", "High", "Very High"],
)

portfolio.to_csv(OUTPUT_CSV, index=False)
print(f"\nPortfolio predictions exported → {OUTPUT_CSV}")
print("\nDone.")
