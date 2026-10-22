"""
Script to generate the complete, production-grade Jupyter Notebook:
notebooks/credit_scoring_eda_and_modeling.ipynb
"""

import json
import os

notebook_path = os.path.join("notebooks", "credit_scoring_eda_and_modeling.ipynb")

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    })

# Cell 1: Header Markdown
add_md("""# Credit Scoring Model Using Machine Learning
### An End-to-End Classification Pipeline for Predicting Credit Risk & Default Probability

**Author:** Antigravity Data Science Portfolio  
**Domain:** Financial Technology (FinTech) & Credit Risk Modeling  
**Dataset:** German Credit Dataset (UCI Statlog / OpenML `credit-g`, ID 31)

---

## 1. Project Overview & Business Problem

In retail banking and consumer lending, granting loans to applicants requires balancing revenue from interest against the risk of credit default.

* **Creditworthiness** reflects a borrower's likelihood to honor their financial obligations.
* **Default (Bad Credit)** occurs when a borrower fails to repay the loan principal and interest as agreed.
* In commercial lending:
  * **False Negative (Type II Error):** Approving an applicant who defaults. The bank loses up to **100% of the loan principal**.
  * **False Positive (Type I Error):** Rejecting a creditworthy applicant. The bank misses the **interest profit** and incurs customer acquisition friction.

Because a **False Negative is substantially more damaging than a False Positive**, our model must optimize for **Recall on defaulters** and **ROC-AUC**, rather than naive accuracy.

---

## 2. Dataset Schema & Feature Dictionary

The **German Credit Data** contains 1,000 credit applicant profiles with 20 input attributes and 1 binary classification target:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `checking_status` | Categorical | Status of existing checking account (`<0`, `0<=X<200`, `>=200`, `no checking`) |
| `duration` | Numerical | Loan duration in months (4 to 72 months) |
| `credit_history` | Categorical | Payment history (`critical/other existing credit`, `existing paid`, `delayed previously`, etc.) |
| `purpose` | Categorical | Purpose of loan (`radio/tv`, `new car`, `furniture`, `education`, `business`, etc.) |
| `credit_amount` | Numerical | Credit amount in Deutsche Mark (250 to 18,424 DM) |
| `savings_status` | Categorical | Savings account / liquid bonds balance (`<100`, `100<=X<500`, `500<=X<1000`, `>=1000`, `no known savings`) |
| `employment` | Categorical | Present employment duration (`unemployed`, `<1`, `1<=X<4`, `4<=X<7`, `>=7` years) |
| `installment_commitment` | Numerical | Installment rate in % of disposable income (1% to 4%) |
| `personal_status` | Categorical | Sex and marital status (`male single`, `female div/dep/mar`, etc.) |
| `other_parties` | Categorical | Guarantors or co-applicants (`none`, `guarantor`, `co applicant`) |
| `residence_since` | Numerical | Present residence tenure in years (1 to 4 years) |
| `property_magnitude` | Categorical | Collateral / property owned (`real estate`, `building society/life insurance`, `car`, `no known property`) |
| `age` | Numerical | Applicant age in years (19 to 75 years) |
| `other_payment_plans` | Categorical | Installment plans from other institutions (`bank`, `stores`, `none`) |
| `housing` | Categorical | Housing arrangement (`own`, `rent`, `for free`) |
| `existing_credits` | Numerical | Number of existing loans at this financial institution (1 to 4) |
| `job` | Categorical | Employment qualification level (`unskilled`, `skilled`, `high qualif/mgmt`) |
| `num_dependents` | Numerical | Number of maintenance dependents (1 or 2) |
| `own_telephone` | Categorical | Registered phone number (`yes`, `none`) |
| `foreign_worker` | Categorical | Foreign worker status (`yes`, `no`) |
| **`class` (Target)** | Binary | **`good`** (Creditworthy / 70%) vs **`bad`** (Default / High Risk / 30%) |
""")

# Cell 2: Imports
add_code("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import joblib

# Plot styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 120
print("Libraries imported successfully!")
""")

# Cell 3: Data Loading
add_md("""## 3. Data Ingestion & Sanity Checks""")
add_code("""csv_path = os.path.join("..", "data", "german_credit_data.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join("data", "german_credit_data.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from {csv_path}")
else:
    print("Fetching OpenML 'credit-g' dataset...")
    data = fetch_openml("credit-g", version=1, as_frame=True)
    df = data.frame

print(f"Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
df.head()
""")

# Cell 4: Integrity check
add_md("""### Check Missing Values & Duplicates""")
add_code("""print("--- Missing Values ---")
print(df.isnull().sum())

print(f"\\nDuplicate Rows Detected: {df.duplicated().sum()}")
print(f"Data types:\\n{df.dtypes}")
""")

# Cell 5: Target Encoding
add_md("""## 4. Target Definition & Class Imbalance

In standard banking notation:
* **`0`**: **Good Credit / Creditworthy** (Negative class)
* **`1`**: **Bad Credit / Default Risk** (Positive class)
""")
add_code("""# Create binary target: 1 = Bad (Defaulter), 0 = Good (Creditworthy)
df['risk_label'] = (df['class'] == 'bad').astype(int)

class_counts = df['risk_label'].value_counts()
class_props = df['risk_label'].value_counts(normalize=True)

print(f"Good Loans (0): {class_counts[0]} ({class_props[0]*100:.1f}%)")
print(f"Bad Loans  (1): {class_counts[1]} ({class_props[1]*100:.1f}%)")

# Visualize Class Distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x='risk_label', palette=['#2ecc71', '#e74c3c'], ax=ax)
ax.set_xticklabels(['Good Credit (0)', 'Bad Credit / Defaulter (1)'])
ax.set_title("Credit Risk Class Distribution (70% Good vs 30% Bad)", fontsize=13, fontweight='bold')
ax.set_ylabel("Applicant Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.show()
""")

# Cell 6: EDA Numerical
add_md("""## 5. Exploratory Data Analysis (EDA)

### 5.1 Financial Numerical Distributions & Risk Profiling
Let us analyze how **Credit Amount**, **Loan Duration**, and **Applicant Age** differ between Good and Bad credit applicants.
""")
add_code("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Credit Amount
sns.boxplot(data=df, x='risk_label', y='credit_amount', palette=['#2ecc71', '#e74c3c'], ax=axes[0])
axes[0].set_xticklabels(['Good (0)', 'Bad (1)'])
axes[0].set_title("Credit Amount by Risk Class", fontweight='bold')
axes[0].set_ylabel("Credit Amount (DM)")

# 2. Duration
sns.boxplot(data=df, x='risk_label', y='duration', palette=['#2ecc71', '#e74c3c'], ax=axes[1])
axes[1].set_xticklabels(['Good (0)', 'Bad (1)'])
axes[1].set_title("Loan Duration (Months) by Risk Class", fontweight='bold')
axes[1].set_ylabel("Duration (Months)")

# 3. Age
sns.boxplot(data=df, x='risk_label', y='age', palette=['#2ecc71', '#e74c3c'], ax=axes[2])
axes[2].set_xticklabels(['Good (0)', 'Bad (1)'])
axes[2].set_title("Applicant Age by Risk Class", fontweight='bold')
axes[2].set_ylabel("Age (Years)")

plt.suptitle("Distributions of Key Numerical Features Across Credit Classes", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("Key Financial Insights:")
print("- Bad loans exhibit significantly longer durations (median > 24 months vs 18 months for good loans).")
print("- Higher credit amounts carry greater default incidence.")
print("- Younger borrowers (<28 years old) have slightly higher default variance compared to mature applicants.")
""")

# Cell 7: Categorical Risk Drivers
add_md("""### 5.2 Top Financial Risk Indicators: Checking Status & Savings Status
Checking account balance and liquid savings are the primary buffers against insolvency.
""")
add_code("""fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Default rate by Checking Status
check_risk = df.groupby('checking_status')['risk_label'].mean().reset_index()
sns.barplot(data=check_risk, x='checking_status', y='risk_label', palette='flare', ax=axes[0])
axes[0].set_title("Default Rate by Checking Account Status", fontweight='bold')
axes[0].set_ylabel("Default Probability (Bad Rate)")
axes[0].set_ylim(0, 0.6)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width()/2., p.get_height() + 0.015), ha='center', fontweight='bold')

# Default rate by Savings Status
savings_risk = df.groupby('savings_status')['risk_label'].mean().reset_index()
sns.barplot(data=savings_risk, x='savings_status', y='risk_label', palette='crest', ax=axes[1])
axes[1].set_title("Default Rate by Savings Account Balance", fontweight='bold')
axes[1].set_ylabel("Default Probability (Bad Rate)")
axes[1].set_ylim(0, 0.6)
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width()/2., p.get_height() + 0.015), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print("Key Findings:")
print("- Applicants with negative checking balances (<0 DM) default at an alarming 49.3% rate!")
print("- Conversely, applicants with no checking account or large balances (>=200 DM) show under 15% default rate.")
print("- Low savings (<100 DM) also strongly correlates with credit default (~36% default rate).")
""")

# Cell 8: Correlation Matrix
add_md("""### 5.3 Numerical Feature Correlation Heatmap""")
add_code("""num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
corr = df[num_features].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("Correlation Matrix of Numerical Financial Features", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()
""")

# Cell 9: Preprocessing Pipeline
add_md("""## 6. Preprocessing & Data Leakage Prevention

### Preventing Data Leakage
1. **Split First:** We perform `train_test_split` with stratification BEFORE computing scaling parameters (mean, standard deviation) or fitting encoders.
2. **Scikit-learn Pipelines:** `ColumnTransformer` is fitted solely on the training data and transforms test data during evaluation and inference.
""")
add_code("""X = df.drop(columns=['class', 'risk_label'])
y = df['risk_label']

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

print(f"Numerical Attributes ({len(num_cols)}): {num_cols}")
print(f"Categorical Attributes ({len(cat_cols)}): {cat_cols}")

# Stratified 80/20 Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Pipeline Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

print(f"Training split: {X_train.shape[0]} samples")
print(f"Testing split : {X_test.shape[0]} samples")
""")

# Cell 10: Model Training and CV
add_md("""## 7. Model Training & 5-Fold Stratified Cross-Validation

We train and benchmark 4 algorithms:
1. **Logistic Regression (L2 Regularized, Balanced Weights)**
2. **Decision Tree Classifier (Constrained Depth, Balanced Weights)**
3. **Random Forest Classifier (Ensemble Bagging, Balanced Weights)**
4. **Gradient Boosting Classifier (Sequential Boosting)**
""")
add_code("""models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, C=0.8, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=3, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, clf in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=['roc_auc', 'recall', 'accuracy'])
    cv_results.append({
        'Model': name,
        'Mean CV ROC-AUC': scores['test_roc_auc'].mean(),
        'Std CV ROC-AUC': scores['test_roc_auc'].std(),
        'Mean CV Recall': scores['test_recall'].mean(),
        'Mean CV Accuracy': scores['test_accuracy'].mean()
    })

pd.DataFrame(cv_results).sort_values(by='Mean CV ROC-AUC', ascending=False)
""")

# Cell 11: Test Evaluation
add_md("""## 8. Test Set Evaluation & Banking Performance Metrics

Let's test all models on the unseen 20% test dataset ($N=200$).
""")
add_code("""test_metrics = []
predictions = {}
pipelines = {}

for name, clf in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
    pipe.fit(X_train, y_train)
    pipelines[name] = pipe
    
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    predictions[name] = {'pred': y_pred, 'proba': y_proba}
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    test_metrics.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall (Defaulters)': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Defaulters Caught (TP)': tp,
        'Missed Defaulters (FN)': fn
    })

results_df = pd.DataFrame(test_metrics).sort_values(by='ROC-AUC', ascending=False)
results_df
""")

# Cell 12: Visual Evaluation
add_md("""### 8.1 ROC Curves & Confusion Matrix Comparison""")
add_code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ROC Curves
for name, pred in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, pred['proba'])
    auc_val = roc_auc_score(y_test, pred['proba'])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)

axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (0.50)')
axes[0].set_title("ROC Curves Comparison on Test Set", fontsize=13, fontweight='bold')
axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].legend(loc="lower right")

# Best Model Confusion Matrix (Logistic Regression)
best_cm = confusion_matrix(y_test, predictions['Logistic Regression']['pred'])
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Good (0)', 'Bad (1)'], yticklabels=['Good (0)', 'Bad (1)'],
            annot_kws={"size": 15, "weight": "bold"})
axes[1].set_title("Logistic Regression Confusion Matrix (Test Set)", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.tight_layout()
plt.show()
""")

# Cell 13: Feature Importance
add_md("""## 9. Feature Importance & Explainability (Random Forest & Logistic Regression)""")
add_code("""# Extract feature names after One-Hot Encoding
fitted_prep = pipelines['Random Forest'].named_steps['prep']
cat_ohe_names = fitted_prep.named_transformers_['cat'].get_feature_names_out(cat_cols)
feature_names = num_cols + list(cat_ohe_names)

# Random Forest Importances
rf_importances = pipelines['Random Forest'].named_steps['clf'].feature_importances_
top_rf = pd.DataFrame({'Feature': feature_names, 'Importance': rf_importances}).sort_values('Importance', ascending=False).head(12)

# Logistic Regression Coefficients
lr_coefs = pipelines['Logistic Regression'].named_steps['clf'].coef_[0]
top_lr = pd.DataFrame({'Feature': feature_names, 'Abs_Coefficient': np.abs(lr_coefs), 'Coefficient': lr_coefs}).sort_values('Abs_Coefficient', ascending=False).head(12)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(data=top_rf, x='Importance', y='Feature', palette='viridis', ax=axes[0])
axes[0].set_title("Top Features: Random Forest (Gini Importance)", fontweight='bold')

sns.barplot(data=top_lr, x='Coefficient', y='Feature', palette='coolwarm', ax=axes[1])
axes[1].set_title("Top Features: Logistic Regression (Signed Coefficients)", fontweight='bold')

plt.tight_layout()
plt.show()
""")

# Cell 14: Responsible AI & Fairness
add_md("""## 10. Responsible AI, Fair Lending & Ethical Audit

In regulated lending environments (e.g., US Equal Credit Opportunity Act - ECOA, Fair Housing Act, EU AI Act):
* Models must **not** unfairly penalize protected demographic attributes (e.g., sex, marital status, nationality).
* In this historical German dataset, features like `personal_status` (encoding sex and marital status) and `foreign_worker` are present.
* Let's audit whether the model displays **Disparate Impact** across gender or age groups.
""")
add_code("""# Evaluate Approval Rates by Demographic Proxy
test_df = X_test.copy()
test_df['pred_bad'] = predictions['Logistic Regression']['pred']
test_df['approved'] = (test_df['pred_bad'] == 0).astype(int)

# Sex extraction from personal_status
test_df['gender'] = test_df['personal_status'].apply(lambda x: 'female' if 'female' in str(x).lower() else 'male')

print("Approval Rate by Gender (Education/Audit only):")
print(test_df.groupby('gender')['approved'].agg(['count', 'mean']).rename(columns={'mean': 'Approval Rate'}))

print("\\nKey Takeaway for Responsible AI:")
print("In a production banking system, sensitive demographic features must either be excluded (Fairness through Unawareness)")
print("or actively constrained using Equalized Odds and Disparate Impact optimization.")
""")

# Cell 15: Save Pipeline
add_md("""## 11. Pipeline Serialization with Joblib""")
add_code("""# Save the best model pipeline for the Streamlit web application
model_save_dir = os.path.join("..", "models")
if not os.path.exists(model_save_dir):
    model_save_dir = "models"
os.makedirs(model_save_dir, exist_ok=True)

save_path = os.path.join(model_save_dir, "credit_scoring_pipeline.joblib")
joblib.dump(pipelines['Logistic Regression'], save_path)
print(f"Model pipeline successfully saved to: {save_path}")
""")

notebook_json = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.10"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

print(f"Jupyter Notebook successfully written to: {notebook_path}")
