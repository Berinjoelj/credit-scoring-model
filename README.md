# 💳 Credit Scoring Model Using Machine Learning
### An End-to-End Classification Engine for Predicting Creditworthiness & Default Probability

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning project designed to predict consumer creditworthiness, calculate the **Probability of Default (PD)**, generate simulated FICO-style credit scores, and deliver automated underwriting recommendations via an interactive **Streamlit** dashboard.

---

## 📌 Table of Contents
1. [Project Overview & Problem Statement](#-project-overview--problem-statement)
2. [Dataset Description & Feature Dictionary](#-dataset-description--feature-dictionary)
3. [Project Architecture & Workflow](#-project-architecture--workflow)
4. [Data Preprocessing & Preventing Leakage](#-data-preprocessing--preventing-leakage)
5. [Exploratory Data Analysis (EDA) Insights](#-exploratory-data-analysis-eda-insights)
6. [Machine Learning Algorithms Benchmarked](#-machine-learning-algorithms-benchmarked)
7. [Evaluation Metrics & Banking Trade-offs](#-evaluation-metrics--banking-trade-offs)
8. [Interactive Web Application (Streamlit)](#-interactive-web-application-streamlit)
9. [Responsible AI & Fair Lending Compliance](#-responsible-ai--fair-lending-compliance)
10. [Project Directory Structure](#-project-directory-structure)
11. [Installation & Quick Start Guide](#-installation--quick-start-guide)
12. [Interview & Viva Questions with Answers](#-interview--viva-questions-with-answers)

---

## 🎯 Project Overview & Problem Statement

Retail banks and FinTech lenders face an asymmetric risk dilemma:
* **The Revenue Model:** Granting loans generates revenue through interest and origination fees.
* **The Default Risk:** When an approved applicant defaults, the lender loses up to **100% of the loan principal**, plus recovery expenses.
* **Error Asymmetry:**
  * **False Negative (FN) - Approving a Defaulter:** Severe financial loss (catastrophic).
  * **False Positive (FP) - Rejecting a Creditworthy Applicant:** Missed interest profit (marginal opportunity cost).

**Objective:** Develop a robust classification pipeline that accurately identifies high-risk loan applicants, maximizes **Recall on Defaulters**, and maintains a strong **ROC-AUC** benchmark while ensuring compliance with fair lending principles.

---

## 📊 Dataset Description & Feature Dictionary

The model is built on the academic benchmark **German Credit Dataset** (Statlog Credit Data / OpenML ID `31`).

* **Total Samples:** 1,000 applicant loan records
* **Class Distribution:** 700 Good Credit (70%), 300 Bad Credit / Defaulters (30%)
* **Features:** 20 financial, transactional, and demographic attributes

| Feature Name | Type | Description & Domain Values |
| :--- | :--- | :--- |
| `checking_status` | Categorical | Balance in checking account (`<0 DM`, `0<=X<200 DM`, `>=200 DM`, `no checking`) |
| `duration` | Numerical | Credit repayment duration in months (4 to 72 months) |
| `credit_history` | Categorical | Past repayment behavior (`critical/other existing credit`, `existing paid`, `delayed previously`, etc.) |
| `purpose` | Categorical | Loan purpose (`radio/tv`, `new car`, `furniture`, `education`, `business`, etc.) |
| `credit_amount` | Numerical | Principal credit amount requested in Deutsche Mark (250 to 18,424 DM) |
| `savings_status` | Categorical | Liquid savings & bond reserves (`<100 DM`, `100<=X<500`, `500<=X<1000`, `>=1000`, `no known savings`) |
| `employment` | Categorical | Length of current employment (`unemployed`, `<1 yr`, `1<=X<4 yrs`, `4<=X<7 yrs`, `>=7 yrs`) |
| `installment_commitment` | Numerical | Installment rate in % of disposable income (1% to 4%) |
| `personal_status` | Categorical | Demographic marital status & sex (`male single`, `female div/dep/mar`, etc.) |
| `other_parties` | Categorical | Guarantors or co-debtors (`none`, `guarantor`, `co applicant`) |
| `residence_since` | Numerical | Tenure at present address in years (1 to 4) |
| `property_magnitude` | Categorical | Collateral / property owned (`real estate`, `building society/life insurance`, `car`, `no known property`) |
| `age` | Numerical | Applicant age in years (19 to 75) |
| `other_payment_plans` | Categorical | Installment plans at other banks or department stores (`bank`, `stores`, `none`) |
| `housing` | Categorical | Housing arrangement (`own`, `rent`, `for free`) |
| `existing_credits` | Numerical | Number of existing active loans at this bank (1 to 4) |
| `job` | Categorical | Employment qualification level (`unskilled`, `skilled`, `high qualif/mgmt`) |
| `num_dependents` | Numerical | Number of maintenance dependents (1 or 2) |
| `own_telephone` | Categorical | Registered phone number (`yes`, `none`) |
| `foreign_worker` | Categorical | Foreign worker status (`yes`, `no`) |
| **`class` (Target)** | Binary | **`good`** (Creditworthy / 0) vs **`bad`** (Default Risk / 1) |

---

## 🏗️ Project Architecture & Workflow

```
   Raw Financial Dataset (German Credit Data)
                    │
                    ▼
     Stratified Train-Test Split (80/20)
       ├── Training Set (800 samples)
       └── Test Set (200 samples)
                    │
                    ▼
       Scikit-Learn ColumnTransformer
       ├── StandardScaler (7 Numerical Features)
       └── OneHotEncoder (13 Categorical Features, handle_unknown='ignore')
                    │
                    ▼
   5-Fold Stratified Cross-Validation Benchmark
   ├── 1. Logistic Regression (L2, class_weight='balanced')
   ├── 2. Decision Tree Classifier (max_depth=5, balanced)
   ├── 3. Random Forest Classifier (150 trees, balanced)
   └── 4. Gradient Boosting Classifier (120 estimators)
                    │
                    ▼
     Model Evaluation & Business Cost Matrix
       ├── Recall on Defaulters (Primary Banking Metric)
       ├── ROC-AUC Discrimination Score
       └── Confusion Matrix Analysis
                    │
                    ▼
      Production Pipeline Serialization
       ├── models/credit_scoring_pipeline.joblib
       └── models/model_metadata.json
                    │
                    ▼
    Interactive Streamlit Web Dashboard
       ├── Real-time Default Probability (PD) Calculation
       ├── Simulated Credit Score (300 - 850 FICO Scale)
       ├── Underwriting Tier & Actionable Recommendations
       └── Responsible AI & Ethical Lending Tab
```

---

## 🔒 Data Preprocessing & Preventing Leakage

Data leakage occurs when information outside the training dataset is inadvertently used to create the model. In credit risk, this results in dangerously optimistic test accuracy that collapses in production.

### Leakage Prevention Strategy:
1. **Strict Partitioning:** We split data into training (80%) and testing (20%) sets using `StratifiedKFold` *before* computing any summary statistics.
2. **Encapsulated Preprocessor:** We use Scikit-learn's `ColumnTransformer` inside a unified `Pipeline`:
   - **Numerical Features:** Scaled using `StandardScaler` (fitting $\mu, \sigma$ strictly on training data).
   - **Categorical Features:** Encoded with `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to gracefully handle novel or rare categories during live inference without throwing runtime errors.
3. **Serialized Pipeline:** The serialized `.joblib` file contains both the preprocessor and the classifier. When an applicant submits data in Streamlit, raw user inputs pass directly through the pipeline, eliminating manual feature alignment errors.

---

## 📈 Exploratory Data Analysis (EDA) Insights

1. **Negative Checking Accounts are High-Risk Indicators:**
   Applicants with `<0 DM` checking accounts default at an alarming **49.3% rate**, whereas applicants with `>=200 DM` or no checking account default at under **15%**.
2. **Savings Cushion:**
   Applicants with liquid savings under `100 DM` experience a default rate of **36%**, highlighting the importance of emergency liquidity buffers.
3. **Repayment Term Length:**
   Long-term loans (>36 months) exhibit over **40% default incidence**, while short-term loans (<12 months) default at under **18%**.
4. **Collateral & Real Estate:**
   Borrowers owning real estate default at half the rate of borrowers without collateral.

---

## 🤖 Machine Learning Algorithms Benchmarked

| Model | Hyperparameters | Strategic Rationale |
| :--- | :--- | :--- |
| **Logistic Regression** | `C=0.8`, `class_weight='balanced'`, `max_iter=1000` | Baseline linear model; provides calibrated log-odds and maximum defaulter sensitivity. |
| **Decision Tree** | `max_depth=5`, `min_samples_leaf=10`, `class_weight='balanced'` | Highly interpretable rule-based benchmark; prone to high variance. |
| **Random Forest** | `n_estimators=150`, `max_depth=8`, `min_samples_leaf=4`, `balanced` | Non-linear ensemble bagging; reduces variance and models feature interactions. |
| **Gradient Boosting** | `n_estimators=120`, `learning_rate=0.08`, `max_depth=3` | Sequential boosting; optimizes loss function; strong discrimination capability. |

---

## 🏆 Evaluation Metrics & Benchmark Results

### Test Set Performance ($N=200$ unseen applicants, 60 actual defaulters):

| Model | Test ROC-AUC | Test Recall (Defaulters) | Test Precision | Test Accuracy | Test F1-Score | Defaulters Caught (TP/60) | Missed Defaulters (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** 🏆 | **0.806** | **80.0%** | **55.8%** | **75.0%** | **0.658** | **48 / 60** | **12** |
| **Gradient Boosting** | **0.793** | 53.3% | 65.3% | 77.5% | 0.587 | 32 / 60 | 28 |
| **Random Forest** | **0.787** | 73.3% | 50.0% | 70.0% | 0.595 | 44 / 60 | 16 |
| **Decision Tree** | 0.634 | 71.7% | 39.1% | 58.0% | 0.506 | 43 / 60 | 17 |

### Why Accuracy is Deceptive in Credit Scoring:
* **The Trap:** If a model simply predicts that *every* applicant is creditworthy, it achieves **70% accuracy** because 70% of loans are good.
* **The Disaster:** That naive model would catch **0% of defaulters** (0% Recall), approving 60 bad loans and bankrupting the lender.
* **The Optimal Selection:** **Logistic Regression with Balanced Class Weights** achieves the highest ROC-AUC (**0.806**) and successfully flags **80% of all defaulters** (48 out of 60), minimizing overall credit losses.

---

## 💻 Interactive Web Application (Streamlit)

The web dashboard (`app.py`) provides an interactive interface for credit officers and underwriters:

1. **Borrower Archetype Presets:**
   - 🌟 *Prime Borrower (Low Risk)*: High reserves, stable employment, low debt.
   - ⚖️ *Borderline Borrower*: Moderate savings, medium loan duration.
   - 🚨 *High-Risk Defaulter*: Negative checking balance, extended tenure, low savings.
2. **Dynamic Risk Threshold Slider:** Adjust default cut-off probability $p_{\text{default}}$ from $0.20$ to $0.80$.
3. **Probability of Default (PD) & Gauge:** Calibrated probability estimate.
4. **Simulated FICO Credit Score (300 - 850):**
   $$\text{Credit Score} = 300 + (1 - \text{PD}) \times 550$$
5. **Underwriting Action Tiers:**
   - **Tier 1 (Prime, $\ge 740$):** Auto-Approve with preferred interest rates.
   - **Tier 2 (Near-Prime, $670 - 739$):** Approve with standard terms.
   - **Tier 3 (Sub-Prime, $580 - 669$):** Manual Underwriter Review & collateral requirement.
   - **Tier 4 (High Risk, $< 580$):** Decline due to excessive default risk.

---

## ⚖️ Responsible AI & Fair Lending Compliance

Credit scoring models are legally bound by fair lending regulations:
* **Equal Credit Opportunity Act (ECOA - Regulation B):** Prohibits credit discrimination based on race, sex, marital status, religion, national origin, or age.
* **Fair Credit Reporting Act (FCRA):** Mandates that lenders issue **Adverse Action Notices** specifying key factual factors when credit is denied.
* **EU Artificial Intelligence Act:** Classifies credit scoring algorithms as **High-Risk AI**, enforcing auditability, bias mitigation, and human oversight.

### Ethical Safeguards Implemented:
* **Proxy Awareness:** Identifying attributes such as `personal_status` and `foreign_worker` as demographic proxies that must be audited or excluded in commercial deployment.
* **Explainability:** Feature importance extraction explains the primary driving factors behind individual risk scores.
* **Mandatory Disclaimer:** Prominently states that predictions are mathematical estimates intended solely for research and portfolio demonstration.

---

## 📂 Project Directory Structure

```
credit-scoring-model/
├── data/
│   └── german_credit_data.csv          # 1,000 applicant loan records
├── notebooks/
│   ├── credit_scoring_eda_and_modeling.ipynb # Complete EDA & modeling notebook
│   └── build_notebook.py               # Notebook generation automation script
├── models/
│   ├── credit_scoring_pipeline.joblib  # Serialized best model pipeline
│   ├── random_forest_pipeline.joblib   # Serialized ensemble pipeline
│   ├── model_metadata.json             # Model configuration & categories
│   ├── model_metrics.csv               # Benchmark evaluation metrics
│   ├── model_comparison.png            # Performance comparison bar chart
│   ├── roc_curves.png                  # Test set ROC curves
│   ├── confusion_matrices.png          # Test set confusion matrices
│   └── feature_importance.png          # Random Forest feature importances
├── app.py                              # Interactive Streamlit web dashboard
├── train_model.py                      # Production model training script
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Version control ignore list
└── README.md                           # Documentation & portfolio guide
```

---

## 🚀 Installation & Quick Start Guide

### 1. Clone or Open Workspace
```bash
cd "credit-scoring-model"
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models & Generate Benchmarks
```bash
python train_model.py
```
*This downloads the dataset, trains the 4 classifiers, computes 5-fold cross-validation, saves evaluation charts to `models/`, and serializes the best pipeline.*

### 5. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎓 Interview & Viva Questions with Answers

### Q1: Why is Accuracy insufficient when evaluating a credit scoring model?
> **Answer:** Credit risk datasets are inherently imbalanced (e.g. 70% good, 30% bad). A naive classifier predicting every applicant as "good" achieves 70% accuracy but misses 100% of defaulters. In banking, the cost of a False Negative (losing loan principal) is exponentially greater than a False Positive (lost interest margin). Therefore, **Recall on Defaulters** and **ROC-AUC** are the critical metrics.

### Q2: How did you prevent data leakage in your preprocessing pipeline?
> **Answer:** We performed a stratified train-test split *before* computing any scaling parameters or categorical encodings. All transformations (`StandardScaler` and `OneHotEncoder`) were wrapped within a Scikit-learn `Pipeline` and fitted strictly on the training partition. The test partition was transformed solely using the fitted training pipeline.

### Q3: How do you handle class imbalance in credit scoring?
> **Answer:** Several techniques can be applied:
> 1. **Cost-Sensitive Learning (Used here):** Setting `class_weight='balanced'`, which automatically penalizes misclassifications of the minority (defaulter) class inversely proportional to class frequencies.
> 2. **Resampling:** Using SMOTE (Synthetic Minority Over-sampling Technique) or Random Undersampling.
> 3. **Threshold Moving:** Lowering the default decision cut-off threshold from 0.50 to 0.35 to catch more marginal defaulters.

### Q4: What is the difference between ROC-AUC and Precision-Recall AUC?
> **Answer:** ROC-AUC plots True Positive Rate vs False Positive Rate across all thresholds. It is threshold-independent and measures the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance. Precision-Recall AUC focuses strictly on the positive class and is especially informative in severe class imbalance situations.

### Q5: What regulatory constraints govern machine learning models in banking?
> **Answer:** Regulated financial institutions must comply with ECOA (Regulation B) preventing disparate impact across protected classes, FCRA requiring adverse action explanations for denied applicants, and the EU AI Act classifying credit assessment as High-Risk AI requiring bias auditing, robustness validation, and human review.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
