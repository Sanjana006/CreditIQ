import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
from sklearn.pipeline import Pipeline as SKPipeline

# ─── Configuration ────────────────────────────────────────────────────────────
st.set_page_config(page_title="CreditIQ Risk Assessment", page_icon="🏦", layout="wide")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "credit_risk_model.pkl")
STAGE1_MODEL_PATH = os.path.join(BASE, "models", "stage1_prescreen_model.pkl")

# ─── Load model & explainer (Cached for Speed) ──────────────────────────────
@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    
    stage1_model = None
    if os.path.exists(STAGE1_MODEL_PATH):
        stage1_model = joblib.load(STAGE1_MODEL_PATH)
        
    _steps = list(model.steps)
    _classifier = model.named_steps["classifier"]
    preprocessor = SKPipeline(_steps[:-1])
    
    def _unwrap_estimator(est):
        if hasattr(est, "calibrated_classifiers_"):
            return _unwrap_estimator(est.calibrated_classifiers_[0].estimator)
        if hasattr(est, "estimator"):
            return _unwrap_estimator(est.estimator)
        return est

    xgb_classifier = _unwrap_estimator(_classifier)
    shap_explainer = shap.TreeExplainer(xgb_classifier)
    
    try:
        encoded_feat_names = list(preprocessor.get_feature_names_out())
    except Exception:
        encoded_feat_names = None
        
    return model, stage1_model, preprocessor, shap_explainer, encoded_feat_names

model, stage1_model, preprocessor, shap_explainer, encoded_feat_names = load_models()

# ─── Feature Engineering ──────────────────────────────────────────────────────
FEATURE_COLS = [
    "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
    "purpose", "dti", "delinq_2yrs", "revol_util", "revol_bal",
    "open_acc", "total_acc", "pub_rec", "fico",
    "loan_income_ratio", "dti_bucket", "income_bucket", "fico_bucket",
    "revol_util_bucket", "loan_amount_bucket", "employment_category",
    "term_category", "credit_history_years"
]

FEATURE_DISPLAY = {
    "loan_amnt":           "Loan Amount (₹)",
    "term":                "Loan Term (months)",
    "emp_length":          "Employment Length (yrs)",
    "home_ownership":      "Home Ownership",
    "annual_inc":          "Annual Income (₹)",
    "purpose":             "Loan Purpose",
    "dti":                 "Debt-to-Income Ratio",
    "delinq_2yrs":         "Delinquencies (2 yrs)",
    "revol_util":          "Revolving Utilisation %",
    "revol_bal":           "Revolving Balance (₹)",
    "open_acc":            "Open Accounts",
    "total_acc":           "Total Accounts",
    "pub_rec":             "Public Records",
    "fico":                "FICO / CIBIL Score",
    "loan_income_ratio":   "Loan-to-Income Ratio",
    "dti_bucket":          "DTI Category",
    "income_bucket":       "Income Category",
    "fico_bucket":         "FICO Category",
    "revol_util_bucket":   "Utilisation Category",
    "loan_amount_bucket":  "Loan Size Category",
    "employment_category": "Employment Category",
    "term_category":       "Term Category",
    "credit_history_years":"Credit History (yrs)",
}

def engineer_features(raw: dict) -> pd.DataFrame:
    d = dict(raw)
    d["loan_income_ratio"] = d["loan_amnt"] / max(d["annual_inc"], 1)

    dti = d["dti"]
    if   dti < 10:  d["dti_bucket"] = "Very Low"
    elif dti < 20:  d["dti_bucket"] = "Low"
    elif dti < 30:  d["dti_bucket"] = "Moderate"
    elif dti < 40:  d["dti_bucket"] = "High"
    else:           d["dti_bucket"] = "Very High"

    inc = d["annual_inc"]
    if   inc < 30000:  d["income_bucket"] = "Low"
    elif inc < 60000:  d["income_bucket"] = "Lower Middle"
    elif inc < 90000:  d["income_bucket"] = "Middle"
    elif inc < 150000: d["income_bucket"] = "Upper Middle"
    else:              d["income_bucket"] = "High"

    fico = d["fico"]
    if   fico < 580:  d["fico_bucket"] = "Poor"
    elif fico < 670:  d["fico_bucket"] = "Fair"
    elif fico < 740:  d["fico_bucket"] = "Good"
    elif fico < 800:  d["fico_bucket"] = "Very Good"
    else:             d["fico_bucket"] = "Excellent"

    ru = d["revol_util"]
    if   ru < 20:  d["revol_util_bucket"] = "Low"
    elif ru < 50:  d["revol_util_bucket"] = "Moderate"
    elif ru < 80:  d["revol_util_bucket"] = "High"
    else:          d["revol_util_bucket"] = "Very High"

    la = d["loan_amnt"]
    if   la < 5000:  d["loan_amount_bucket"] = "Small"
    elif la < 15000: d["loan_amount_bucket"] = "Medium"
    elif la < 25000: d["loan_amount_bucket"] = "Large"
    else:            d["loan_amount_bucket"] = "Very Large"

    emp = d["emp_length"]
    if   emp < 0:  d["employment_category"] = "Unknown"
    elif emp < 2:  d["employment_category"] = "Entry Level"
    elif emp < 5:  d["employment_category"] = "Mid Career"
    elif emp < 10: d["employment_category"] = "Experienced"
    else:          d["employment_category"] = "Senior"

    d["term_category"] = "Long Term" if d["term"] == 60 else "Short Term"
    d["credit_history_years"] = max(d.get("credit_history_years", 10), 1)

    return pd.DataFrame([{c: d.get(c, 0) for c in FEATURE_COLS}])

def _clean_encoded_name(enc_name: str) -> str:
    if '__' in enc_name:
        prefix, rest = enc_name.split('__', 1)
    else:
        prefix, rest = '', enc_name

    if prefix == 'num':
        return FEATURE_DISPLAY.get(rest, rest.replace('_', ' ').title())
    else:
        parts = rest.rsplit('_', 1)
        parent = parts[0] if len(parts) > 1 else rest
        if parent in FEATURE_DISPLAY:
            return FEATURE_DISPLAY[parent]
        return FEATURE_DISPLAY.get(rest, rest.replace('_', ' ').title())

def get_shap_values(X_df: pd.DataFrame):
    X_transformed = preprocessor.transform(X_df)
    shap_vals = shap_explainer.shap_values(X_transformed)
    sv = shap_vals[0] if not isinstance(shap_vals, list) else shap_vals[1][0]

    if encoded_feat_names and len(encoded_feat_names) == len(sv):
        raw_pairs = list(zip(encoded_feat_names, sv))
    else:
        raw_pairs = [(f'feat_{i}', v) for i, v in enumerate(sv)]

    from collections import defaultdict
    collapsed = defaultdict(float)
    for enc_name, shap_val in raw_pairs:
        readable = _clean_encoded_name(enc_name)
        collapsed[readable] += float(shap_val)

    top10 = sorted(collapsed.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    return [{"feature": name, "value": val} for name, val in top10]

# ─── UI Rendering ─────────────────────────────────────────────────────────────
st.title("🏦 CreditIQ Underwriting System")
st.markdown("Advanced Machine Learning architecture for predicting loan defaults in the Indian Banking Context.")

with st.sidebar:
    st.header("Applicant Details")
    
    st.subheader("Loan Details")
    loan_amnt_inr = st.number_input("Loan Amount (₹ Lakhs)", 0.5, 9.0, 3.5, 0.25, help="Total loan amount requested.")
    term = st.selectbox("Loan Term", [36, 60], format_func=lambda x: f"{x} Months", help="Repayment duration.")
    purpose = st.selectbox("Loan Purpose", ["debt_consolidation", "credit_card", "home_improvement", "small_business", "major_purchase", "medical", "educational", "car", "house", "other"])
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"], index=1)
    
    st.subheader("Borrower Profile")
    annual_inc_inr = st.number_input("Annual Income (₹ Lakhs)", 4.0, 36.0, 8.0, 0.5, help="Gross annual income before taxes.")
    monthly_emi = st.number_input("Monthly EMI Obligations (₹)", 0, 100000, 5000, 500, help="Existing monthly obligations.")
    emp_length = st.number_input("Employment Length (years)", 0, 40, 5)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"], index=2)
    
    st.subheader("Credit Profile")
    fico = st.number_input("CIBIL / FICO Score", 300, 850, 700)
    dti = st.number_input("Debt-to-Income Ratio (%)", 0.0, 100.0, 18.0)
    revol_util = st.number_input("Revolving Utilisation (%)", 0, 100, 40)
    revol_bal_k = st.number_input("Revolving Balance (₹ Thousands)", 0, 2000, 100)
    delinq_2yrs = st.number_input("Delinquencies (last 2 yrs)", 0, 20, 0)
    open_acc = st.number_input("Open Accounts", 0, 100, 8)
    total_acc = st.number_input("Total Accounts", 0, 200, 15)
    pub_rec = st.number_input("Public Records", 0, 20, 0)
    credit_history_years = st.number_input("Credit History (years)", 0, 50, 10)
    
    analyze_btn = st.button("Assess Credit Risk", type="primary", use_container_width=True)

# ─── Inference Logic ──────────────────────────────────────────────────────────
if analyze_btn:
    # Convert INR to USD mapping
    loan_amnt_usd = round((loan_amnt_inr * 100000) / 23.9)
    annual_inc_usd = round((annual_inc_inr * 100000) / 23.9)
    revol_bal_usd = round((revol_bal_k * 1000) / 23.9)

    raw_data = {
        "loan_amnt": loan_amnt_usd,
        "term": float(term),
        "emp_length": float(emp_length),
        "home_ownership": home_ownership,
        "annual_inc": annual_inc_usd,
        "purpose": purpose,
        "dti": float(dti),
        "delinq_2yrs": float(delinq_2yrs),
        "revol_util": float(revol_util),
        "revol_bal": revol_bal_usd,
        "open_acc": float(open_acc),
        "total_acc": float(total_acc),
        "pub_rec": float(pub_rec),
        "fico": float(fico),
        "credit_history_years": float(credit_history_years)
    }

    X_df = engineer_features(raw_data)
    
    # Stage 1
    stage1_reject = False
    stage1_prob = 0.0
    if stage1_model is not None:
        stage1_features = pd.DataFrame([{'loan_amnt': loan_amnt_usd, 'fico': float(fico), 'dti': float(dti), 'emp_length': float(emp_length)}])
        stage1_prob = float(stage1_model.predict_proba(stage1_features)[0][1])
        if stage1_prob > 0.50:
            stage1_reject = True
            
    # Stage 2
    prob = float(model.predict_proba(X_df)[0][1])
    
    if stage1_reject:
        prob = max(prob, stage1_prob)
        st.error(f"### 🚨 Stage 1 Decline: Very High Risk ({prob*100:.1f}%)")
        st.write("This application was instantly flagged by the Stage 1 Pre-Screening model due to extreme heuristic violations (e.g., poor FICO/CIBIL or excessive DTI).")
    else:
        approved = prob < 0.40
        if approved:
            st.success(f"### ✅ Approved: Low Risk ({prob*100:.1f}%)")
        else:
            st.warning(f"### ⚠️ Manual Review Required: Elevated Risk ({prob*100:.1f}%)")
            
        st.markdown("---")
        st.subheader("Model Explainability (SHAP Top Drivers)")
        
        shap_vals = get_shap_values(X_df)
        shap_df = pd.DataFrame(shap_vals)
        shap_df["Color"] = shap_df["value"].apply(lambda x: "Risk Increasing" if x > 0 else "Risk Decreasing")
        shap_df["Absolute Value"] = shap_df["value"].abs()
        shap_df = shap_df.sort_values(by="Absolute Value", ascending=True)
        
        st.bar_chart(shap_df, x="feature", y="value", color="Color")
        
        st.info("**What does this chart mean?** Bars extending to the right increase the probability of default. Bars extending to the left decrease it.")
