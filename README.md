<div align="center">
  <h1>🏦 CreditIQ: Advanced Credit Risk Assessment System</h1>
  <p><i>A production-grade machine learning architecture for predicting loan defaults and Non-Performing Assets (NPAs).</i></p>
  
  <p>
    <b>Created by:</b> Sanjana Nathani
  </p>

  <p>
    <a href="https://creditiq-xn3q.onrender.com/"><b>🚀 View Live Application</b></a>
    ·
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

## 🇮🇳 India Market Context & Ind AS 109 ECL Framework

This architecture strictly aligns with the **Ind AS 109 Expected Credit Loss (ECL)** framework, which mandates that financial institutions estimate forward-looking Probability of Default (PD) rather than just looking at historical losses.

Because banking data is highly protected and real Indian default datasets are rarely public, the core ML engine was trained on the robust **US Lending Club Dataset**. To make this mathematically viable for the Indian market, the system applies **Purchasing Power Parity (PPP) transformations** (approx. 1 USD = ₹23.9 PPP) to seamlessly map an applicant's Indian Rupee (₹) socioeconomic status to the US risk distribution.

Furthermore, the frontend dynamically calculates critical Indian banking metrics:
- **FOIR (Fixed Obligation to Income Ratio):** Ensures the applicant's existing monthly obligations do not exceed the RBI's recommended limits (typically 50%).
- **City Tier Risk Assessment:** Dynamically applies risk multipliers (e.g., Tier 1, Tier 2) based on the macroeconomic stability and cost of living in the borrower's location.
- **Priority Sector Lending (RBI):** Flags whether the loan purpose (e.g., MSME, Agriculture, Education) qualifies for Indian Priority Sector lending targets.

**Production Readiness:** For a real Indian NBFC or Bank, the architecture remains exactly the same. You simply swap the dataset with internal historical data, retrain using the provided script, and the entire app—including the Ind AS 109 ECL dashboard—works out of the box.

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
- **Evaluation Metric**: Optimized for **ROC-AUC (~0.70)** and Recall over raw accuracy. For credit risk, missing a default (False Negative) is vastly more expensive to the bank than falsely declining a good loan (False Positive). The final decision boundary incorporates strict threshold tuning (e.g., 0.20-0.30 on calibrated probabilities) to prioritize a high recall rate.
- **Probability Calibration**: Tree-based algorithms often produce raw confidence scores rather than true probabilities. This architecture routes the raw XGBoost output through `CalibratedClassifierCV` using **Isotonic Regression**. This strictly ensures that when the model outputs a 60% PD, 60% of applicants in that bracket historically default.
- **Feature Pipeline**: Missing data is handled via median/mode imputation, and categorical features are transformed using `OneHotEncoder`. The entire flow is securely encapsulated inside an `sklearn` Pipeline to completely prevent data leakage during testing.
- **Training Scale**: The model is trained on the full cleaned dataset of **~1.34 Million** historical loans (80/20 split), leveraging parallel processing to capture the most nuanced patterns in minority default events.

---

## ❓ FAQs (Model & Methodology)

**Q: Is the model predicting whether a loan will be *received* (approved), or whether it will *default*?**
> **A:** The model predicts the **Probability of Default (PD)** after a loan is disbursed. It is designed to be used *during* the underwriting phase to help the bank decide whether to approve or reject the loan based on the predicted risk.

**Q: Since the model was trained on loans that the bank *already accepted*, isn't analyzing the risk after disbursement questionable? Doesn't the model lack data on bad applicants?**
> **A:** Excellent observation. This is a classic industry challenge called **Reject Inference**. Because the bank already filtered out the worst applicants, our training data is biased toward "better" profiles. To combat this, we implemented a **Two-Stage Architecture**. We added a deterministic Pre-Screener (Stage 1) that automatically rejects extreme high-risk profiles based on standard banking heuristics. The ML model (Stage 2) is then correctly utilized to score the nuanced risk of the remaining "acceptable" pool. In a real-world bank, we would further close this gap by incorporating credit bureau data on rejected loans.

**Q: You trained this on US bank data but built the app for the Indian market. Won't this be questioned? Is it valid?**
> **A:** This project is a demonstration of a highly scalable, production-grade **machine learning architecture**, not a final localized economic model. To make the US-trained model work logically with Indian Rupees (₹), we apply **Purchasing Power Parity (PPP)** transformations. While the exact economic behaviors differ between the US and India, the fundamental engineering architecture—the XGBoost pipelines, isotonic calibration, SHAP explainability, and the Flask backend—is completely agnostic. An Indian bank simply needs to swap out the dataset with their own, and the entire system operates perfectly.

**Q: The original dataset has over 23 Lakh (2.3 Million) accepted loans. Why is the final modeling dataset ~13.4 Lakh rows?**
> **A:** The raw dataset contains 2.3 million rows, but rigorous data hygiene is required for credit risk. We strictly removed "Current" loans (where the final default status is unknown), dropped columns with massive null values or data leakage (information not available at the time of underwriting), and handled severe outliers. This filtering yields a highly robust, fully resolved dataset of **~1.34 Million loans**. The XGBoost pipeline trains on this entire cleaned dataset to ensure it learns the absolute most from rare default patterns without hitting artificial sampling limits.

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
