<div align="center">
  <h1>🏦 CreditIQ: Advanced Credit Risk Assessment System</h1>
  <p><i>A production-grade machine learning architecture for predicting loan defaults and Non-Performing Assets (NPAs).</i></p>
  
  <p>
    <b>Created by:</b> Sanjana Nathani
  </p>

  <p>
    <a href="https://www.kaggle.com/datasets/wordsforthewise/lending-club">View Dataset (Lending Club)</a>
    ·
    <a href="#-architecture--methodology">View Architecture</a>
    ·
    <a href="#-frequently-asked-questions-faqs--defenses">Read the FAQ</a>
  </p>
</div>

<hr />

## 🌟 Overview & The Problem
In the modern banking sector, particularly in growing economies like India, managing **Non-Performing Assets (NPAs)** is critical to financial stability. A slight increase in default rates can wipe out millions in profitability. 

Traditional underwriting often relies on rigid rule engines and manual review, which can be slow and opaque. **CreditIQ** is built to solve this by providing a highly accurate, heavily calibrated machine learning pipeline that not only predicts the **Probability of Default (PD)** but also *explains* its reasoning using SHAP values, ensuring compliance and interpretability for underwriters.

## 🚀 Key Features
- **Two-Stage Risk Assessment**: Combines a deterministic pre-screener (to catch extreme risks immediately) with an advanced XGBoost model.
- **Isotonic Calibration**: Machine learning models often output raw scores that aren't true probabilities. This model uses `CalibratedClassifierCV` to ensure a 60% score mathematically means a 60% chance of default.
- **SHAP Explainability**: Integrates TreeExplainer to break down exactly *why* an applicant was flagged as risky, generating human-readable narratives for the underwriting team.
- **Regulatory Context**: Maps inputs to Indian banking standards, calculating metrics like **FOIR** (Fixed Obligation to Income Ratio) and assessing RBI Priority Sector lending rules.
- **Premium Underwriter UI**: A highly polished, dynamic dashboard that presents the risk profile, key metrics, and portfolio-level analytics in an intuitive interface.

---

## 🌍 The "Indian Context" Transformation
This model was trained on the publicly available **[US Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)**, which contains over a decade of real-world loan default data. 

To demonstrate this architecture in an **Indian Banking Context**, the backend dynamically applies **Purchasing Power Parity (PPP)** transformations (approx. 1 USD = ₹23.9 PPP). 
- When an underwriter inputs an applicant's salary in ₹ Lakhs, the system scales it to the mathematical equivalent in the US training distribution.
- **Why do this?** Because banking data is heavily protected, real Indian loan default datasets are not publicly available. This PPP transformation allows us to demonstrate a fully functional, mathematically sound underwriting app.
- **Production Readiness:** For a real Indian lending company, **the architecture remains exactly the same.** You simply swap the Lending Club dataset with the bank's internal historical data, change the word "FICO" to "CIBIL", retrain the pipeline using the provided `train_stage2.py` script, and the entire app works out of the box.

---

## 📈 Exploratory Data Analysis & SQL Portfolio Analytics

Before modeling the credit risk, extensive **Exploratory Data Analysis (EDA)** and **SQL-based Portfolio Analytics** were performed to understand macro trends in the data. You can explore the detailed queries and statistical distributions in `notebooks/03_SQL_Portfolio_Analytics.ipynb`.

### Analytics Snapshots
*(Click to expand the images)*

<p align="center">
  <img src="images/Screenshot%202026-07-09%20at%2023.55.56.png" width="800" alt="Analysis Snapshot 1" style="border-radius:8px; border: 1px solid #ddd; margin-bottom: 10px;"/>
  <br/>
  <img src="images/Screenshot%202026-07-09%20at%2023.56.11.png" width="800" alt="Analysis Snapshot 2" style="border-radius:8px; border: 1px solid #ddd; margin-bottom: 10px;"/>
  <br/>
  <img src="images/Screenshot%202026-07-09%20at%2023.56.27.png" width="800" alt="Analysis Snapshot 3" style="border-radius:8px; border: 1px solid #ddd;"/>
</p>

---

## 🏗 Architecture & Methodology

### 1. Stage 1: The Pre-Screener (Reject Inference Mitigation)
Because the Lending Club dataset only contains data on loans the company *chose to approve*, training an ML model purely on this data causes **Reject Inference Bias**. The model never learns what a truly terrible applicant looks like.
To solve this, Stage 1 acts as a strict, rule-based gatekeeper. It automatically declines applications with severe red flags (e.g., FICO < 600, DTI > 45%, or FOIR > 65%), simulating the bank's initial rejection phase.

### 2. Stage 2: The Calibrated XGBoost Model
Applicants who pass Stage 1 are passed to the ML pipeline:
- **Imputation & Scaling**: Handles missing values and scales features.
- **XGBoost Classifier**: A gradient boosting tree model captures complex, non-linear relationships between borrower features.
- **Calibration Layer**: The XGBoost output is passed through Isotonic Regression to map raw ML scores into true empirical probabilities.

### 3. The Flask & JS Frontend
The UI takes the inputs, hits the Flask `/api/predict` endpoint, un-wraps the calibrated model to generate SHAP values, builds a narrative, and visually renders the decision using a responsive, CSS-grid-based premium dashboard.

---

## 📊 Model Performance & Technical Highlights

- **Champion Model**: XGBoost Classifier tuned with `scale_pos_weight` to aggressively penalize the minority class, ensuring the model prioritizes identifying true loan defaults despite the heavy natural class imbalance.
- **Evaluation Metric**: Optimized for **ROC-AUC (~0.71)** and Recall over raw accuracy. For credit risk, missing a default (False Negative) is vastly more expensive to the bank than falsely declining a good loan (False Positive).
- **Probability Calibration**: Tree-based algorithms often produce raw confidence scores rather than true probabilities. This architecture routes the raw XGBoost output through `CalibratedClassifierCV` using **Isotonic Regression**. This strictly ensures that when the model outputs a 60% PD, 60% of applicants in that bracket historically default.
- **Feature Pipeline**: Missing data is handled via median/mode imputation, and categorical features are transformed using `OneHotEncoder`. The entire flow is securely encapsulated inside an `sklearn` Pipeline to completely prevent data leakage during testing.

---

## ❓ FAQs (Model & Methodology)

**Q: Is the model predicting whether a loan will be *received* (approved), or whether it will *default*?**
> **A:** The model predicts the **Probability of Default (PD)** after a loan is disbursed. It is designed to be used *during* the underwriting phase to help the bank decide whether to approve or reject the loan based on the predicted risk.

**Q: Since the model was trained on loans that the bank *already accepted*, isn't analyzing the risk after disbursement questionable? Doesn't the model lack data on bad applicants?**
> **A:** Excellent observation. This is a classic industry challenge called **Reject Inference**. Because the bank already filtered out the worst applicants, our training data is biased toward "better" profiles. To combat this, we implemented a **Two-Stage Architecture**. We added a deterministic Pre-Screener (Stage 1) that automatically rejects extreme high-risk profiles based on standard banking heuristics. The ML model (Stage 2) is then correctly utilized to score the nuanced risk of the remaining "acceptable" pool. In a real-world bank, we would further close this gap by incorporating credit bureau data on rejected loans.

**Q: You trained this on US bank data but built the app for the Indian market. Won't this be questioned? Is it valid?**
> **A:** This project is a demonstration of a highly scalable, production-grade **machine learning architecture**, not a final localized economic model. To make the US-trained model work logically with Indian Rupees (₹), we apply **Purchasing Power Parity (PPP)** transformations. While the exact economic behaviors differ between the US and India, the fundamental engineering architecture—the XGBoost pipelines, isotonic calibration, SHAP explainability, and the Flask backend—is completely agnostic. An Indian bank simply needs to swap out the dataset with their own, and the entire system operates perfectly.

---

## 🚧 Limitations & Future Work

While this architecture is robust, there are areas for future enhancement:
1. **True Reject Inference Validation:** In a production setting, acquiring bureau performance data on historically rejected applicants to train a combined model.
2. **Macro-Economic Variables:** The current model evaluates micro-features (the borrower's specific financials). Future iterations could inject macro-features (inflation rates, GDP growth) to adjust risk thresholds dynamically.
3. **Alternative Data:** Incorporating non-traditional data (e.g., UPI transaction history, utility bill payments) for borrowers with thin credit files.

---

## 🛠 Tech Stack
- **Machine Learning**: Python, Scikit-Learn, XGBoost, SHAP, Pandas
- **Backend**: Flask (Python)
- **Frontend**: HTML5, Vanilla JavaScript (ES6), Custom Premium CSS, Chart.js
- **Model Storage**: Pickle pipelines

---
<div align="center">
  <p><i>"Transforming data into definitive underwriting decisions."</i></p>
</div>
