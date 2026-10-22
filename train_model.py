"""
=============================================================================
Credit Scoring Model - Training & Evaluation Pipeline
=============================================================================
Project: Credit Scoring Model Using Machine Learning
Dataset: German Credit Data (Statlog / OpenML credit-g)
Objective: Predict credit risk (Good / Creditworthy vs Bad / High Risk)
=============================================================================
"""

import os
import json
import joblib
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# -----------------------------------------------------------------------------
# 1. Configuration & Directories
# -----------------------------------------------------------------------------
DATA_DIR = "data"
MODELS_DIR = "models"
DATA_FILE = os.path.join(DATA_DIR, "german_credit_data.csv")
PIPELINE_PATH = os.path.join(MODELS_DIR, "credit_scoring_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.csv")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Data Acquisition & Loading
# -----------------------------------------------------------------------------
def load_data(filepath: str = DATA_FILE) -> pd.DataFrame:
    """
    Loads German Credit data from local CSV if present,
    otherwise downloads it directly from OpenML (Statlog credit-g).
    """
    if os.path.exists(filepath):
        print(f"[INFO] Loading dataset from local file: {filepath}")
        df = pd.read_csv(filepath)
    else:
        print("[INFO] Local file not found. Fetching OpenML 'credit-g' (dataset 31)...")
        dataset = fetch_openml("credit-g", version=1, as_frame=True)
        df = dataset.frame
        df.to_csv(filepath, index=False)
        print(f"[INFO] Dataset successfully saved to: {filepath}")

    print(f"[INFO] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# -----------------------------------------------------------------------------
# 3. Preprocessing Setup
# -----------------------------------------------------------------------------
def prepare_features_and_target(df: pd.DataFrame):
    """
    Prepares X (features) and y (target).
    Target Encoding:
      - 'bad'  -> 1 (High Risk / Potential Defaulter)
      - 'good' -> 0 (Low Risk / Creditworthy)
    """
    df = df.copy()
    
    # Target definition
    if 'class' in df.columns:
        y = (df['class'] == 'bad').astype(int)
        X = df.drop(columns=['class'])
    else:
        raise KeyError("Column 'class' not found in dataset.")

    # Identify column types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    print(f"[INFO] Numerical features ({len(num_cols)}): {num_cols}")
    print(f"[INFO] Categorical features ({len(cat_cols)}): {cat_cols}")
    print(f"[INFO] Target class distribution:\n{y.value_counts(normalize=True).rename({0: 'Good (0)', 1: 'Bad (1)'})}")

    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """
    Builds a Scikit-learn ColumnTransformer:
      - Numerical columns: StandardScaler (zero mean, unit variance)
      - Categorical columns: OneHotEncoder (handle_unknown='ignore' to prevent runtime errors)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    return preprocessor


# -----------------------------------------------------------------------------
# 4. Model Training & Comparison
# -----------------------------------------------------------------------------
def get_model_candidates():
    """Returns a dictionary of candidate classification models."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=0.8,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=3,
            subsample=0.9,
            random_state=RANDOM_STATE
        )
    }


def evaluate_models(models: dict, preprocessor: ColumnTransformer, X_train, X_test, y_train, y_test):
    """
    Trains each candidate model, runs 5-fold cross-validation,
    and computes full test evaluation metrics.
    """
    results = []
    fitted_pipelines = {}
    test_predictions = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n" + "="*80)
    print("MODEL TRAINING & CROSS-VALIDATION BENCHMARK")
    print("="*80)

    for name, clf in models.items():
        print(f"\n--> Training & Evaluating: {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # 5-Fold Stratified Cross-Validation on Training data
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
        )

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predictions on unseen test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        test_predictions[name] = {
            'y_pred': y_pred,
            'y_proba': y_proba
        }

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # In credit lending, approving a defaulter (FN) is typically 5x more costly
        # than rejecting a creditworthy borrower (FP)
        cost_fp = 1    # Missed profit / opportunity cost
        cost_fn = 5    # Direct principal default loss
        estimated_loss = (fp * cost_fp) + (fn * cost_fn)

        results.append({
            'Model': name,
            'CV_AUC_Mean': float(np.mean(cv_scores['test_roc_auc'])),
            'CV_Recall_Mean': float(np.mean(cv_scores['test_recall'])),
            'Test_Accuracy': float(acc),
            'Test_Precision': float(prec),
            'Test_Recall': float(rec),
            'Test_F1': float(f1),
            'Test_ROC_AUC': float(auc),
            'True_Negatives': int(tn),
            'False_Positives': int(fp),
            'False_Negatives': int(fn),
            'True_Positives': int(tp),
            'Business_Risk_Score': int(estimated_loss)
        })

        print(f"    Test Accuracy : {acc:.4f}")
        print(f"    Test Precision: {prec:.4f}")
        print(f"    Test Recall   : {rec:.4f} (Defaulters identified: {tp}/{tp+fn})")
        print(f"    Test F1-Score : {f1:.4f}")
        print(f"    Test ROC-AUC  : {auc:.4f}")
        print(f"    Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    results_df = pd.DataFrame(results).sort_values(by="Test_ROC_AUC", ascending=False)
    return results_df, fitted_pipelines, test_predictions


# -----------------------------------------------------------------------------
# 5. Visualization Generation
# -----------------------------------------------------------------------------
def generate_and_save_charts(results_df, test_predictions, y_test, fitted_pipelines, num_cols, cat_cols):
    """
    Generates and saves publication-quality charts:
      1. Model Comparison Bar Chart
      2. ROC-AUC Curves Comparison
      3. Confusion Matrices Heatmaps
      4. Feature Importance Plot (for Best Tree and Linear Models)
    """
    sns.set_theme(style="whitegrid")

    # 1. Model Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    metrics_melted = results_df.melt(
        id_vars=['Model'],
        value_vars=['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_ROC_AUC'],
        var_name='Metric',
        value_name='Score'
    )
    palette = sns.color_palette("mako", 5)
    ax = sns.barplot(data=metrics_melted, x='Model', y='Score', hue='Metric', palette=palette)
    plt.title("Credit Risk Model Performance Comparison (Test Set)", fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.xlabel("Model Candidate", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    chart1_path = os.path.join(MODELS_DIR, "model_comparison.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved comparison chart to: {chart1_path}")

    # 2. ROC Curves Comparison
    plt.figure(figsize=(9, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (name, preds) in enumerate(test_predictions.items()):
        fpr, tpr, _ = roc_curve(y_test, preds['y_proba'])
        auc_score = roc_auc_score(y_test, preds['y_proba'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", color=colors[idx % len(colors)], lw=2.2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.tight_layout()
    chart2_path = os.path.join(MODELS_DIR, "roc_curves.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved ROC curves to: {chart2_path}")

    # 3. Confusion Matrix Grid
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()
    for idx, (name, preds) in enumerate(test_predictions.items()):
        cm = confusion_matrix(y_test, preds['y_pred'])
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=axes[idx],
            xticklabels=['Good (0)', 'Bad (1)'],
            yticklabels=['Good (0)', 'Bad (1)'],
            annot_kws={"size": 14, "weight": "bold"}
        )
        axes[idx].set_title(f"{name}", fontsize=13, fontweight='bold')
        axes[idx].set_xlabel("Predicted Credit Class", fontsize=11)
        axes[idx].set_ylabel("Actual Credit Class", fontsize=11)

    plt.suptitle("Confusion Matrices Across Models (Test Set: N=200)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    chart3_path = os.path.join(MODELS_DIR, "confusion_matrices.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Confusion Matrix grid to: {chart3_path}")

    # 4. Feature Importance Extraction (from Random Forest & Logistic Regression)
    try:
        rf_pipe = fitted_pipelines["Random Forest"]
        preprocessor = rf_pipe.named_steps['preprocessor']
        rf_clf = rf_pipe.named_steps['classifier']
        
        ohe = preprocessor.named_transformers_['cat']
        encoded_cat_names = ohe.get_feature_names_out(cat_cols)
        all_feature_names = num_cols + list(encoded_cat_names)
        importances = rf_clf.feature_importances_

        feat_imp_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp_df, x='Importance', y='Feature', hue='Feature', palette='crest', legend=False)
        plt.title("Top 15 Most Influential Features (Random Forest Classifier)", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Gini Importance Score", fontsize=11)
        plt.ylabel("Feature / Encoded Attribute", fontsize=11)
        plt.tight_layout()
        chart4_path = os.path.join(MODELS_DIR, "feature_importance.png")
        plt.savefig(chart4_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved Feature Importance chart to: {chart4_path}")
    except Exception as e:
        print(f"[WARNING] Could not plot feature importance: {e}")


# -----------------------------------------------------------------------------
# 6. Metadata and Model Persistence
# -----------------------------------------------------------------------------
def save_best_model_and_metadata(results_df, fitted_pipelines, X, num_cols, cat_cols):
    """
    Selects the best overall model (balancing ROC-AUC, Recall for defaulters, and F1),
    saves the fitted pipeline using joblib, and stores rich metadata in JSON.
    """
    # Select best model: Logistic Regression has top ROC-AUC (0.806) and Defaulter Recall (80%)
    # Random Forest is top non-linear ensemble (ROC-AUC 0.787, Recall 73.3%)
    # Let's save Random Forest as primary production pipeline or Logistic Regression
    # We can choose Logistic Regression as Primary or Random Forest. Both are serialized!
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n[INFO] Best performing model based on test evaluation: '{best_model_name}'")
    best_pipeline = fitted_pipelines[best_model_name]

    # Save Primary Pipeline
    joblib.dump(best_pipeline, PIPELINE_PATH)
    print(f"[INFO] Serialized best pipeline to: {PIPELINE_PATH}")

    # Also save Random Forest as alternative if best wasn't RF
    if "Random Forest" in fitted_pipelines and best_model_name != "Random Forest":
        rf_path = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
        joblib.dump(fitted_pipelines["Random Forest"], rf_path)
        print(f"[INFO] Also serialized Random Forest pipeline to: {rf_path}")

    # Build unique categories dictionary for Streamlit UI dropdowns
    category_options = {}
    for col in cat_cols:
        category_options[col] = sorted(X[col].dropna().unique().tolist())

    numerical_ranges = {}
    for col in num_cols:
        numerical_ranges[col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'median': float(X[col].median()),
            'mean': float(X[col].mean())
        }

    # Extract top features for explainability
    top_features = []
    try:
        clf = best_pipeline.named_steps['classifier']
        prep = best_pipeline.named_steps['preprocessor']
        cat_names = prep.named_transformers_['cat'].get_feature_names_out(cat_cols)
        all_features = num_cols + list(cat_names)

        if hasattr(clf, 'feature_importances_'):
            imps = clf.feature_importances_
            sorted_idx = np.argsort(imps)[::-1][:10]
            top_features = [{"feature": all_features[i], "score": float(imps[i])} for i in sorted_idx]
        elif hasattr(clf, 'coef_'):
            coefs = clf.coef_[0]
            sorted_idx = np.argsort(np.abs(coefs))[::-1][:10]
            top_features = [{"feature": all_features[i], "score": float(coefs[i])} for i in sorted_idx]
    except Exception as e:
        print(f"[WARNING] Could not parse feature importance for metadata: {e}")

    metadata = {
        "best_model_name": best_model_name,
        "dataset_name": "German Credit Data (Statlog credit-g, OpenML ID 31)",
        "sample_count": int(len(X)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "target_mapping": {
            "0": "Good (Creditworthy / Low Risk)",
            "1": "Bad (High Risk / Potential Defaulter)"
        },
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "category_options": category_options,
        "numerical_ranges": numerical_ranges,
        "top_features": top_features,
        "models_benchmarked": results_df.to_dict(orient="records")
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[INFO] Metadata successfully written to: {METADATA_PATH}")

    # Save CSV benchmark
    results_df.to_csv(METRICS_PATH, index=False)
    print(f"[INFO] Model metrics table written to: {METRICS_PATH}")


# -----------------------------------------------------------------------------
# 7. Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    print("="*80)
    print("CREDIT SCORING MODEL TRAINING PIPELINE")
    print("="*80)

    # 1. Load data
    df = load_data()

    # 2. Check duplicates & null values
    dup_count = df.duplicated().sum()
    null_count = df.isnull().sum().sum()
    print(f"[INFO] Duplicate rows detected: {dup_count}")
    print(f"[INFO] Total missing values: {null_count}")

    # 3. Features and Target split
    X, y, num_cols, cat_cols = prepare_features_and_target(df)

    # 4. Train-Test Split (Stratified to maintain 70/30 good/bad ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set    : {X_test.shape[0]} samples")

    # 5. Preprocessing
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # 6. Candidate Models
    models = get_model_candidates()

    # 7. Model Evaluation
    results_df, fitted_pipelines, test_predictions = evaluate_models(
        models, preprocessor, X_train, X_test, y_train, y_test
    )

    # 8. Charts
    generate_and_save_charts(results_df, test_predictions, y_test, fitted_pipelines, num_cols, cat_cols)

    # 9. Model Persistence
    save_best_model_and_metadata(results_df, fitted_pipelines, X, num_cols, cat_cols)

    print("\n" + "="*80)
    print("FINAL BENCHMARK SUMMARY (ORDERED BY TEST ROC-AUC):")
    print("="*80)
    print(results_df[['Model', 'Test_ROC_AUC', 'Test_Recall', 'Test_Precision', 'Test_Accuracy', 'Test_F1', 'Business_Risk_Score']].to_string(index=False))
    print("="*80)
    print("[SUCCESS] Model training and evaluation completed successfully!\n")


if __name__ == "__main__":
    main()
