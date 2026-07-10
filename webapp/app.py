"""
Credit Risk Analytics - Flask Backend
End-to-End Credit Risk Scoring & Portfolio Risk Analytics System
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
MODEL_PATH      = os.path.join(PARENT, "models", "credit_risk_model.pkl")
PORTFOLIO_PATH  = os.path.join(PARENT, "data",   "portfolio_predictions.csv")
STAGE1_MODEL_PATH = os.path.join(PARENT, "models", "stage1_prescreen_model.pkl")

# ─── Load model & pre-compute explainer once ──────────────────────────────────
from sklearn.pipeline import Pipeline as SKPipeline
model = joblib.load(MODEL_PATH)
print(f"[INFO] Model loaded: {type(model).__name__}")

stage1_model = None
if os.path.exists(STAGE1_MODEL_PATH):
    stage1_model = joblib.load(STAGE1_MODEL_PATH)
    print(f"[INFO] Stage 1 Model loaded: {type(stage1_model).__name__}")
else:
    print("[WARN] Stage 1 model not found. Pre-screening disabled.")

# Pre-build preprocessor sub-pipeline and SHAP explainer at startup (fast inference later)
_steps      = list(model.steps)
_classifier = model.named_steps["classifier"]
_preprocessor = SKPipeline(_steps[:-1])   # everything except the final classifier

# Get the 68 encoded feature names once
try:
    _encoded_feat_names = list(_preprocessor.get_feature_names_out())
except Exception:
    _encoded_feat_names = None

# Recursively unwrap CalibratedClassifierCV / FrozenEstimator to reach raw XGBoost
def _unwrap_estimator(est):
    if hasattr(est, "calibrated_classifiers_"):
        return _unwrap_estimator(est.calibrated_classifiers_[0].estimator)
    if hasattr(est, "estimator"):
        return _unwrap_estimator(est.estimator)
    return est

_xgb_classifier = _unwrap_estimator(_classifier)
print(f"[INFO] Unwrapped classifier: {type(_xgb_classifier).__name__}")

# Pre-build SHAP TreeExplainer (uses raw XGBoost, not the calibrated wrapper)
_shap_explainer = shap.TreeExplainer(_xgb_classifier)
print(f"[INFO] SHAP explainer ready | encoded features: {len(_encoded_feat_names) if _encoded_feat_names else 'unknown'}")

# Load portfolio data (cached at startup)
port_df = pd.read_csv(PORTFOLIO_PATH)
print(f"[INFO] Portfolio loaded: {len(port_df):,} rows")

# ─── Feature columns expected by the model ─────────────────────────────────────
# Derived from training notebook (04_Credit_Risk_Modeling)
FEATURE_COLS = [
    "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
    "purpose", "dti", "delinq_2yrs", "revol_util",
    "revol_bal", "open_acc", "total_acc", "pub_rec", "fico",
    "loan_income_ratio", "dti_bucket", "income_bucket", "fico_bucket",
    "revol_util_bucket", "loan_amount_bucket", "employment_category",
    "term_category", "credit_history_years"
]

# Human-readable display names for SHAP chart
FEATURE_DISPLAY = {
    "loan_amnt":           "Loan Amount (₹)",
    "term":                "Loan Term (months)",
    "emp_length":          "Employment Length (yrs)",
    "home_ownership":      "Home Ownership",
    "annual_inc":          "Annual Income (₹)",
    "verification_status": "Income Verification",
    "purpose":             "Loan Purpose",
    "dti":                 "Debt-to-Income Ratio",
    "delinq_2yrs":         "Delinquencies (2 yrs)",
    "revol_util":          "Revolving Utilisation %",
    "revol_bal":           "Revolving Balance (₹)",
    "open_acc":            "Open Accounts",
    "total_acc":           "Total Accounts",
    "pub_rec":             "Public Records",
    "fico":                "FICO / Credit Score",
    "loan_income_ratio":   "Loan-to-Income Ratio",
    "dti_bucket":          "DTI Category",
    "income_bucket":       "Income Category",
    "fico_bucket":         "FICO Category",
    "revol_util_bucket":   "Utilisation Category",
    "loan_amount_bucket":  "Loan Size Category",
    "employment_category": "Employment Category",
    "term_category":       "Term Category",
    "credit_history_years":"Credit History (yrs)",
    # Indian context
    "foir":                "FOIR (Fixed Obligation Ratio)",
    "city_tier_risk":      "City Tier Risk Weight",
    "is_priority_sector":  "Priority Sector Loan",
}

# ─── Indian context helpers ────────────────────────────────────────────────────
LOAN_PURPOSE_MAP = {
    "debt_consolidation": "Personal Loan – Debt Consolidation",
    "credit_card":        "Credit Card Refinance",
    "home_improvement":   "Home Renovation Loan",
    "small_business":     "MSME Working Capital Loan",
    "major_purchase":     "Consumer Durable Loan",
    "medical":            "Medical Emergency Loan",
    "educational":        "Education Loan",
    "moving":             "Relocation Loan",
    "vacation":           "Personal Loan – Lifestyle",
    "wedding":            "Personal Loan – Wedding",
    "other":              "Other Personal Loan",
    "house":              "Housing Finance Loan",
    "renewable_energy":   "Green Finance Loan",
    "car":                "Vehicle Finance Loan",
}

PRIORITY_SECTOR = {"small_business", "educational", "renewable_energy", "house"}

CITY_TIER_RISK = {"Tier 1": 0.8, "Tier 2": 1.0, "Tier 3": 1.3}

def compute_indian_features(form):
    """Return extra Indian-context features for display."""
    monthly_emi = form.get("monthly_emi", 0)
    monthly_inc = form.get("annual_inc_inr", 1) / 12
    foir = round(monthly_emi / monthly_inc, 3) if monthly_inc > 0 else 0
    city_tier = form.get("city_tier", "Tier 2")
    is_priority = 1 if form.get("purpose", "") in PRIORITY_SECTOR else 0
    return {
        "foir":               foir,
        "city_tier":          city_tier,
        "city_tier_risk":     CITY_TIER_RISK.get(city_tier, 1.0),
        "is_priority_sector": is_priority,
        "purpose_india":      LOAN_PURPOSE_MAP.get(form.get("purpose", "other"), "Other Personal Loan"),
    }

# ─── Feature engineering (mirrors notebook) ────────────────────────────────────
def engineer_features(raw: dict) -> pd.DataFrame:
    d = dict(raw)
    d["loan_income_ratio"] = d["loan_amnt"] / max(d["annual_inc"], 1)

    # DTI bucket
    dti = d["dti"]
    if   dti < 10:  d["dti_bucket"] = "Very Low"
    elif dti < 20:  d["dti_bucket"] = "Low"
    elif dti < 30:  d["dti_bucket"] = "Moderate"
    elif dti < 40:  d["dti_bucket"] = "High"
    else:           d["dti_bucket"] = "Very High"

    # Income bucket (USD equivalent – PPP adjusted)
    inc = d["annual_inc"]
    if   inc < 30000:  d["income_bucket"] = "Low"
    elif inc < 60000:  d["income_bucket"] = "Lower Middle"
    elif inc < 90000:  d["income_bucket"] = "Middle"
    elif inc < 150000: d["income_bucket"] = "Upper Middle"
    else:              d["income_bucket"] = "High"

    # FICO bucket
    fico = d["fico"]
    if   fico < 580:  d["fico_bucket"] = "Poor"
    elif fico < 670:  d["fico_bucket"] = "Fair"
    elif fico < 740:  d["fico_bucket"] = "Good"
    elif fico < 800:  d["fico_bucket"] = "Very Good"
    else:             d["fico_bucket"] = "Excellent"

    # Revol util bucket
    ru = d["revol_util"]
    if   ru < 20:  d["revol_util_bucket"] = "Low"
    elif ru < 50:  d["revol_util_bucket"] = "Moderate"
    elif ru < 80:  d["revol_util_bucket"] = "High"
    else:          d["revol_util_bucket"] = "Very High"

    # Loan amount bucket
    la = d["loan_amnt"]
    if   la < 5000:  d["loan_amount_bucket"] = "Small"
    elif la < 15000: d["loan_amount_bucket"] = "Medium"
    elif la < 25000: d["loan_amount_bucket"] = "Large"
    else:            d["loan_amount_bucket"] = "Very Large"

    # Employment category
    emp = d["emp_length"]
    if   emp < 0:  d["employment_category"] = "Unknown"
    elif emp < 2:  d["employment_category"] = "Entry Level"
    elif emp < 5:  d["employment_category"] = "Mid Career"
    elif emp < 10: d["employment_category"] = "Experienced"
    else:          d["employment_category"] = "Senior"

    # Term category
    d["term_category"] = "Long Term" if d["term"] == 60 else "Short Term"

    # Credit history (proxy)
    d["credit_history_years"] = max(d.get("credit_history_years", 10), 1)

    return pd.DataFrame([{c: d.get(c, 0) for c in FEATURE_COLS}])

# ─── SHAP helper ──────────────────────────────────────────────────────────────
def _clean_encoded_name(enc_name: str) -> str:
    """
    Convert sklearn ColumnTransformer encoded names → human-readable labels.
    e.g.  'num__loan_amnt'           → 'Loan Amount'
          'cat__home_ownership_RENT' → 'Home Ownership'
          'cat__purpose_credit_card' → 'Loan Purpose'
    """
    # Strip prefix
    if '__' in enc_name:
        prefix, rest = enc_name.split('__', 1)
    else:
        prefix, rest = '', enc_name

    if prefix == 'num':
        # Numeric — map directly
        return FEATURE_DISPLAY.get(rest, rest.replace('_', ' ').title())
    else:
        # Categorical one-hot — collapse to the parent feature
        # e.g. home_ownership_RENT → home_ownership
        # Take the part before the last underscore-separated token IF it's a known feature
        parts = rest.rsplit('_', 1)
        parent = parts[0] if len(parts) > 1 else rest
        if parent in FEATURE_DISPLAY:
            return FEATURE_DISPLAY[parent]
        # Try full name
        return FEATURE_DISPLAY.get(rest, rest.replace('_', ' ').title())


def get_shap_values(X_df: pd.DataFrame):
    """
    Transform X through the pipeline preprocessor, compute SHAP values on
    the raw XGBoost classifier, then map back to readable feature names.
    Returns top-10 contributors as a list of dicts.
    """
    try:
        # Step 1: Transform through the preprocessor (one-hot encoding etc.)
        X_transformed = _preprocessor.transform(X_df)          # shape (1, 68)

        # Step 2: SHAP on the transformed data
        shap_vals = _shap_explainer.shap_values(X_transformed)  # shape (1, 68)
        sv = shap_vals[0] if not isinstance(shap_vals, list) else shap_vals[1][0]

        # Step 3: Map encoded feature names → readable labels
        if _encoded_feat_names and len(_encoded_feat_names) == len(sv):
            raw_pairs = list(zip(_encoded_feat_names, sv))
        else:
            raw_pairs = [(f'feat_{i}', v) for i, v in enumerate(sv)]

        # Step 4: Collapse one-hot features — sum SHAP values per parent feature
        from collections import defaultdict
        collapsed = defaultdict(float)
        for enc_name, shap_val in raw_pairs:
            readable = _clean_encoded_name(enc_name)
            collapsed[readable] += float(shap_val)

        # Step 5: Sort by |SHAP| and return top 10
        top10 = sorted(collapsed.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        return [
            {"feature": name, "value": round(val, 4)}
            for name, val in top10
        ]

    except Exception as e:
        import traceback
        print(f"[WARN] SHAP failed: {e}")
        traceback.print_exc()
        return []

# ─── Risk label helper ──────────────────────────────────────────────────────────────
def risk_label(prob: float):
    if   prob < 0.20: return "Very Low",  "#10b981"
    elif prob < 0.40: return "Low",       "#34d399"
    elif prob < 0.60: return "Moderate",  "#f59e0b"
    elif prob < 0.80: return "High",      "#f97316"
    else:             return "Very High", "#ef4444"

# ─── SHAP Narrative Generator ───────────────────────────────────────────────────────
# Recommendation playbook: feature → (high-risk advice, low-risk note)
REC_PLAYBOOK = {
    "FICO / Credit Score": (
        "The FICO / CIBIL score is below the preferred threshold. Borrower should target a score above 720 by reducing outstanding balances and avoiding new credit inquiries for 6 months.",
        "Strong CIBIL score is positively contributing to this application."
    ),
    "Loan-to-Income Ratio": (
        "Loan-to-income ratio is elevated. Consider reducing the requested loan amount or demonstrating a higher verifiable income source to bring this ratio below 35%.",
        "Loan amount is well within the borrower's income capacity."
    ),
    "Debt-to-Income Ratio": (
        "DTI is high, indicating heavy existing debt obligations. Paying down one or two existing loans before this application would materially improve the risk profile.",
        "Debt-to-income ratio is within acceptable limits."
    ),
    "Revolving Utilisation %": (
        "High revolving credit utilisation is a red flag. Reducing credit card balances to below 50% of the credit limit is the fastest way to improve this score.",
        "Revolving utilisation is at a healthy level."
    ),
    "Revolving Balance (₹)": (
        "A large outstanding revolving balance increases perceived financial stress. Consider clearing or consolidating this balance before reapplying.",
        "Revolving balance is at a manageable level."
    ),
    "Delinquencies (2 yrs)": (
        "Recent delinquencies are a strong negative signal. Lenders typically require a minimum of 12–24 months of clean repayment history before reconsidering.",
        "Clean delinquency record strengthens this application."
    ),
    "Loan Amount (₹)": (
        "The loan amount is large relative to the borrower profile. A smaller initial loan with a strong repayment track record would build lender confidence.",
        "Loan amount is proportionate to the borrower's profile."
    ),
    "FOIR (Fixed Obligation Ratio)": (
        "FOIR exceeds the 50% RBI guideline. Monthly fixed obligations should be reduced by prepaying or closing an existing loan before this application.",
        "FOIR is within RBI's recommended range."
    ),
    "Home Ownership": (
        "Home ownership status adds some risk in the current profile. Asset ownership generally strengthens repayment capacity.",
        "Home ownership status is a positive factor in this application."
    ),
    "Income Verification": (
        "Income is not fully verified. Providing Form 16, ITR or bank statements for the last 2 years would significantly increase lender confidence.",
        "Verified income is a positive signal for this application."
    ),
    "Loan Term (months)": (
        "Longer loan tenure increases overall exposure and total interest cost. A shorter term with slightly higher EMIs demonstrates repayment confidence.",
        "Loan tenure is appropriate for the borrower's profile."
    ),
}


def generate_shap_narrative(shap_data: list, prob: float, approved: bool, raw: dict, indian: dict) -> dict:
    """
    Converts SHAP values into a plain-English explanation paragraph and
    a prioritised list of actionable recommendations.
    """
    risk_up   = [(d["feature"], d["value"]) for d in shap_data if d["value"] > 0.01]
    risk_down = [(d["feature"], d["value"]) for d in shap_data if d["value"] < -0.01]
    risk_up.sort(key=lambda x: -x[1])
    risk_down.sort(key=lambda x: x[1])   # most negative first

    # ─ Decision lead sentence
    if prob < 35:
        lead = f"This borrower presents a <strong>low default risk of {prob:.1f}%</strong>, well within acceptable underwriting limits."
    elif prob < 60:
        lead = f"This borrower carries a <strong>moderate default probability of {prob:.1f}%</strong>, placing the application in the manual review zone."
    else:
        lead = f"This borrower shows a <strong>high default probability of {prob:.1f}%</strong>, exceeding the standard underwriting threshold for automatic approval."

    # ─ Top risk driver sentence
    driver_sentences = []
    if risk_up:
        top = risk_up[0][0]
        driver_sentences.append(
            f"The primary risk driver is <strong>{top}</strong>, which is the single largest factor pushing up default probability."
        )
    if len(risk_up) > 1:
        second = risk_up[1][0]
        driver_sentences.append(
            f"<strong>{second}</strong> also contributes meaningfully to the risk elevation."
        )

    # ─ Protective factor sentence
    protect_sentences = []
    if risk_down:
        top_p = risk_down[0][0]
        protect_sentences.append(
            f"On the positive side, <strong>{top_p}</strong> is the strongest protective factor, partially offsetting the risk profile."
        )

    # ─ India context note
    india_note = ""
    foir = indian.get("foir", 0)
    if foir > 0.60:
        india_note = f"Additionally, the borrower's FOIR of <strong>{foir*100:.1f}%</strong> exceeds the RBI-recommended 60% ceiling, which is a significant concern for Indian lenders."
    elif foir > 0.50:
        india_note = f"The FOIR of <strong>{foir*100:.1f}%</strong> is borderline — Indian NBFCs typically enforce a hard limit of 50%."

    narrative = " ".join(filter(None, [lead] + driver_sentences + protect_sentences + [india_note]))

    # ─ Recommendations — pick from playbook based on top risk-increasing features
    recommendations = []
    seen_features = set()
    for feat, val in risk_up:
        if feat in REC_PLAYBOOK and feat not in seen_features:
            recommendations.append({"feature": feat, "text": REC_PLAYBOOK[feat][0], "impact": "high"})
            seen_features.add(feat)
        if len(recommendations) >= 4:
            break

    # Fill with generic advice if fewer than 3
    if len(recommendations) < 2:
        recommendations.append({
            "feature": "Repayment History",
            "text": "Maintain a clean repayment record for 12 consecutive months. Payment history is the single strongest positive signal for credit risk models.",
            "impact": "high"
        })
    if len(recommendations) < 3 and raw.get("delinq_2yrs", 0) == 0:
        recommendations.append({
            "feature": "Documentation",
            "text": "Submitting fully verified income documentation (ITR, Form 16, 6-month bank statement) significantly improves lender confidence and model score.",
            "impact": "medium"
        })

    return {
        "narrative": narrative,
        "recommendations": recommendations
    }


# ─── Portfolio analytics (pre-computed) ────────────────────────────────────────
def build_portfolio_stats():
    df = port_df.copy()
    total = len(df)
    defaults = int(df["Actual_Default"].sum())
    npa_rate = round(defaults / total * 100, 2)
    total_exposure = round(df["loan_amnt"].sum() / 1e7, 2)   # ₹ Crore (PPP)
    
    # Calculate Expected Credit Loss (ECL = PD * EAD * LGD) assuming 50% LGD
    ecl_total = (df["Probability"] * df["loan_amnt"] * 0.50).sum()
    ecl_crore = round(ecl_total / 1e7, 2)

    # Risk band counts
    rb = df["Risk_Band"].value_counts().to_dict()

    # Default rate by purpose (top 8)
    purpose_default = (
        df.groupby("purpose")["Actual_Default"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "default_rate", "count": "volume"})
        .sort_values("volume", ascending=False)
        .head(8)
    )
    purpose_default["default_rate"] = (purpose_default["default_rate"] * 100).round(2)
    purpose_default["purpose"] = purpose_default["purpose"].map(LOAN_PURPOSE_MAP).fillna(purpose_default["purpose"])

    # Default rate by FICO bucket
    fico_order = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
    fico_default = (
        df.groupby("fico_bucket")["Actual_Default"]
        .mean()
        .reindex(fico_order)
        .dropna()
        .mul(100).round(2)
        .reset_index()
    )

    # Default rate by risk band
    rb_order = ["Very Low", "Low", "Moderate", "High", "Very High"]
    rb_default = (
        df.groupby("Risk_Band")["Actual_Default"]
        .mean()
        .reindex(rb_order)
        .dropna()
        .mul(100).round(2)
        .reset_index()
    )

    # DTI vs default (binned)
    df["dti_bin"] = pd.cut(df["dti"].clip(0, 50), bins=[0,10,20,30,40,50],
                           labels=["0–10","10–20","20–30","30–40","40–50"])
    dti_default = (
        df.groupby("dti_bin", observed=True)["Actual_Default"]
        .mean().mul(100).round(2)
        .reset_index()
    )

    # Income distribution by default
    df["inc_bin"] = pd.cut(df["annual_inc"].clip(0, 200000),
                           bins=[0,30000,60000,90000,150000,200000],
                           labels=["<30K","30–60K","60–90K","90–150K",">150K"])
    inc_default = (
        df.groupby("inc_bin", observed=True)["Actual_Default"]
        .mean().mul(100).round(2)
        .reset_index()
    )

    # Home ownership default
    home_default = (
        df.groupby("home_ownership")["Actual_Default"]
        .agg(["mean","count"])
        .reset_index()
        .rename(columns={"mean":"default_rate","count":"volume"})
    )
    home_default["default_rate"] = (home_default["default_rate"]*100).round(2)
    home_default = home_default[home_default["volume"] > 50]

    # Portfolio exposure by risk band
    rb_exposure = (
        df.groupby("Risk_Band")["loan_amnt"]
        .sum()
        .reindex(rb_order)
        .dropna()
        .div(1e7).round(2)
        .reset_index()
        .rename(columns={"loan_amnt": "exposure_cr"})
    )

    return {
        "kpis": {
            "total_loans": f"{total:,}",
            "npa_rate": f"{npa_rate}%",
            "total_exposure": f"₹{total_exposure} Cr",
            "ecl_crore": f"₹{ecl_crore} Cr"
        },
        "risk_bands":    rb,
        "purpose_default": purpose_default.to_dict(orient="records"),
        "fico_default":    fico_default.to_dict(orient="records"),
        "rb_default":      rb_default.to_dict(orient="records"),
        "dti_default":     dti_default.to_dict(orient="records"),
        "inc_default":     inc_default.to_dict(orient="records"),
        "home_default":    home_default.to_dict(orient="records"),
        "rb_exposure":     rb_exposure.to_dict(orient="records"),
    }

PORTFOLIO_STATS = build_portfolio_stats()
print("[INFO] Portfolio stats pre-computed")

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Build feature dataframe
        raw = {
            "loan_amnt":           float(data.get("loan_amnt", 10000)),
            "term":                float(data.get("term", 36)),
            "emp_length":          float(data.get("emp_length", 5)),
            "home_ownership":      data.get("home_ownership", "RENT"),
            "annual_inc":          float(data.get("annual_inc", 60000)),
            "purpose":             data.get("purpose", "debt_consolidation"),
            "dti":                 float(data.get("dti", 15)),
            "delinq_2yrs":         float(data.get("delinq_2yrs", 0)),
            "revol_util":          float(data.get("revol_util", 40)),
            "revol_bal":           float(data.get("revol_bal", 5000)),
            "open_acc":            float(data.get("open_acc", 8)),
            "total_acc":           float(data.get("total_acc", 15)),
            "pub_rec":             float(data.get("pub_rec", 0)),
            "fico":                float(data.get("fico", 700)),
            "credit_history_years":float(data.get("credit_history_years", 10)),
        }

        X_df = engineer_features(raw)
        
        # Stage 1 Pre-Screening
        stage1_reject = False
        if stage1_model is not None:
            stage1_features = pd.DataFrame([{
                'loan_amnt': raw['loan_amnt'],
                'fico': raw['fico'],
                'dti': raw['dti'],
                'emp_length': raw['emp_length']
            }])
            stage1_prob = float(stage1_model.predict_proba(stage1_features)[0][1])
            if stage1_prob > 0.50:
                stage1_reject = True
                
        prob = float(model.predict_proba(X_df)[0][1])
        label, color = risk_label(prob)
        approved = prob < 0.40

        shap_contributors = get_shap_values(X_df)
        
        # Use the true INR income sent from the frontend for the FOIR calculation
        indian_payload = {**data, "annual_inc_inr": float(data.get("annual_inc_inr", raw["annual_inc"] * 23.9))}
        indian = compute_indian_features(indian_payload)
        
        if stage1_reject:
            label = "Very High (Pre-Screen Decline)"
            color = "#ef4444"
            approved = False
            prob = max(prob, stage1_prob) # Override probability to reflect stage 1 severity
            narrative = f"This application was instantly flagged by our Stage 1 Pre-Screening model (Probability of Default: {stage1_prob*100:.1f}%). The requested loan amount combined with the borrower's FICO score ({raw['fico']}) and DTI ({raw['dti']}%) aligns closely with historically rejected applications."
            recommendations = [{"feature": "Pre-Screening Criteria", "text": "Application declined based on early-stage risk indicators. Reduce loan amount or improve FICO score before reapplying.", "impact": "high"}]
        else:
            narrative_data = generate_shap_narrative(shap_contributors, round(prob * 100, 1), approved, raw, indian)
            narrative = narrative_data["narrative"]
            recommendations = narrative_data["recommendations"]

        return jsonify({
            "probability":  round(prob * 100, 1),
            "approved":     approved,
            "risk_band":    label,
            "risk_color":   color,
            "shap":         shap_contributors,
            "indian":       indian,
            "narrative":    narrative,
            "recommendations": recommendations,
            "stage1_reject": stage1_reject,
            "features": {
                "fico":     raw["fico"],
                "dti":      raw["dti"],
                "revol_util": raw["revol_util"],
                "loan_income_ratio": round(raw["loan_amnt"] / max(raw["annual_inc"], 1), 3),
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/portfolio")
def portfolio():
    return jsonify(PORTFOLIO_STATS)


if __name__ == "__main__":
    app.run(debug=True, port=5050)
