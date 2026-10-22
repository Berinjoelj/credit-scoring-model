"""
=============================================================================
Credit Scoring Model - Interactive Streamlit Web Application
=============================================================================
Project: Credit Scoring Model Using Machine Learning
Dataset: German Credit Data (Statlog / OpenML credit-g)
Objective: Predict credit risk & default probability with responsible AI
=============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditScore AI | Credit Scoring & Risk Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Result Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 15px;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .badge-approved {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
        margin-bottom: 20px;
    }
    
    .badge-declined {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.35);
        margin-bottom: 20px;
    }
    
    .badge-review {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        font-size: 1.35rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.35);
        margin-bottom: 20px;
    }
    
    .disclaimer-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #475569;
        margin-top: 25px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Resource Loading & Caching
# -----------------------------------------------------------------------------
MODELS_DIR = "models"
PRIMARY_PIPELINE_PATH = os.path.join(MODELS_DIR, "credit_scoring_pipeline.joblib")
RF_PIPELINE_PATH = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.csv")
DATA_PATH = os.path.join("data", "german_credit_data.csv")

@st.cache_resource
def load_model_assets():
    """Loads trained pipelines and metadata once and caches them."""
    if not os.path.exists(PRIMARY_PIPELINE_PATH):
        st.error(f"Trained model not found at {PRIMARY_PIPELINE_PATH}. Please run `python train_model.py` first.")
        st.stop()
        
    primary_pipeline = joblib.load(PRIMARY_PIPELINE_PATH)
    
    rf_pipeline = None
    if os.path.exists(RF_PIPELINE_PATH):
        rf_pipeline = joblib.load(RF_PIPELINE_PATH)
        
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
        
    metrics_df = None
    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)
        
    raw_df = None
    if os.path.exists(DATA_PATH):
        raw_df = pd.read_csv(DATA_PATH)
        
    return primary_pipeline, rf_pipeline, metadata, metrics_df, raw_df

primary_pipeline, rf_pipeline, metadata, metrics_df, raw_df = load_model_assets()


# -----------------------------------------------------------------------------
# 3. Sidebar Controls & Presets
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-cards.png", width=64)
    st.markdown("### **CreditScore AI**")
    st.caption("Machine Learning Creditworthiness Engine")
    st.markdown("---")
    
    # Model Selection
    st.markdown("#### ⚙️ **Model Configuration**")
    available_models = ["Logistic Regression (High Recall Balanced)", "Random Forest Classifier (Ensemble)"]
    selected_model_label = st.selectbox(
        "Active Classifier",
        available_models,
        index=0,
        help="Logistic Regression optimizes Defaulter Recall (80%), while Random Forest balances non-linear relationships."
    )
    
    active_pipeline = primary_pipeline if "Logistic" in selected_model_label else (rf_pipeline or primary_pipeline)
    
    # Decision Threshold Slider
    st.markdown("#### 🎯 **Risk Threshold ($p_{default}$)**")
    decision_threshold = st.slider(
        "Default Cut-off Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="If Probability of Default exceeds this threshold, the loan is classified as High Risk (Bad). Conservative banks lower this to 0.35 to catch more defaulters."
    )
    
    st.markdown("---")
    st.markdown("#### 🚀 **Test Borrower Presets**")
    preset_choice = st.radio(
        "Quick Fill Profile:",
        ["Custom Inputs", "Prime Borrower (Low Risk)", "Borderline Borrower", "High-Risk Defaulter"],
        index=0
    )
    
    st.markdown("---")
    st.markdown(
        """
        <div style='font-size: 0.8rem; color: #64748b;'>
        <b>Built with:</b> Scikit-learn, Streamlit, Pandas<br>
        <b>Benchmark:</b> German Credit Data (Statlog)<br>
        <b>Metric:</b> Recall & ROC-AUC Optimized
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# 4. Preset Profile Generator
# -----------------------------------------------------------------------------
def get_preset_values(preset_name: str):
    """Returns default input values according to standard credit risk archetypes."""
    if preset_name == "Prime Borrower (Low Risk)":
        return {
            'checking_status': '>=200',
            'duration': 12,
            'credit_history': 'critical/other existing credit',
            'purpose': 'radio/tv',
            'credit_amount': 2500,
            'savings_status': '>=1000',
            'employment': '>=7',
            'installment_commitment': 2,
            'personal_status': 'male single',
            'other_parties': 'none',
            'residence_since': 4,
            'property_magnitude': 'real estate',
            'age': 45,
            'other_payment_plans': 'none',
            'housing': 'own',
            'existing_credits': 2,
            'job': 'high qualif/self emp/mgmt',
            'num_dependents': 1,
            'own_telephone': 'yes',
            'foreign_worker': 'no'
        }
    elif preset_name == "Borderline Borrower":
        return {
            'checking_status': '0<=X<200',
            'duration': 24,
            'credit_history': 'existing paid',
            'purpose': 'furniture/equipment',
            'credit_amount': 4500,
            'savings_status': '100<=X<500',
            'employment': '1<=X<4',
            'installment_commitment': 3,
            'personal_status': 'female div/dep/mar',
            'other_parties': 'none',
            'residence_since': 2,
            'property_magnitude': 'car',
            'age': 32,
            'other_payment_plans': 'bank',
            'housing': 'rent',
            'existing_credits': 1,
            'job': 'skilled',
            'num_dependents': 1,
            'own_telephone': 'none',
            'foreign_worker': 'yes'
        }
    elif preset_name == "High-Risk Defaulter":
        return {
            'checking_status': '<0',
            'duration': 48,
            'credit_history': 'delayed previously',
            'purpose': 'business',
            'credit_amount': 9500,
            'savings_status': '<100',
            'employment': '<1',
            'installment_commitment': 4,
            'personal_status': 'male div/sep',
            'other_parties': 'none',
            'residence_since': 1,
            'property_magnitude': 'no known property',
            'age': 23,
            'other_payment_plans': 'stores',
            'housing': 'for free',
            'existing_credits': 3,
            'job': 'unskilled resident',
            'num_dependents': 2,
            'own_telephone': 'none',
            'foreign_worker': 'yes'
        }
    else:  # Custom Default
        return {
            'checking_status': 'no checking',
            'duration': 18,
            'credit_history': 'existing paid',
            'purpose': 'radio/tv',
            'credit_amount': 3000,
            'savings_status': '<100',
            'employment': '1<=X<4',
            'installment_commitment': 2,
            'personal_status': 'male single',
            'other_parties': 'none',
            'residence_since': 2,
            'property_magnitude': 'car',
            'age': 35,
            'other_payment_plans': 'none',
            'housing': 'own',
            'existing_credits': 1,
            'job': 'skilled',
            'num_dependents': 1,
            'own_telephone': 'yes',
            'foreign_worker': 'yes'
        }

preset_values = get_preset_values(preset_choice)


# -----------------------------------------------------------------------------
# 5. Header Section
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Credit Scoring & Risk Evaluation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent machine learning engine predicting individual creditworthiness, default risk, and automated lending recommendations.</div>', unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💳 Real-Time Credit Evaluation",
    "📊 Model Benchmark & Metrics",
    "🔍 Exploratory Data Analysis (EDA)",
    "⚖️ Responsible AI & Fair Lending",
    "ℹ️ Technical Architecture"
])


# =============================================================================
# TAB 1: Real-Time Credit Evaluation
# =============================================================================
with tab1:
    st.markdown("### **Applicant Financial & Credit Profile**")
    st.caption("Enter the loan application attributes below or select a quick-fill archetype from the left sidebar.")
    
    with st.form("credit_evaluation_form"):
        # Section A: Liquidity & Banking
        st.markdown("#### 🏦 **1. Liquidity & Existing Banking Relationship**")
        col1, col2, col3 = st.columns(3)
        with col1:
            checking_status = st.selectbox(
                "Checking Account Status",
                metadata["category_options"]["checking_status"],
                index=metadata["category_options"]["checking_status"].index(preset_values['checking_status']),
                help="Balance in applicant's primary transactional account."
            )
        with col2:
            savings_status = st.selectbox(
                "Savings & Liquid Assets",
                metadata["category_options"]["savings_status"],
                index=metadata["category_options"]["savings_status"].index(preset_values['savings_status']),
                help="Total liquid savings, deposits, or bonds."
            )
        with col3:
            other_payment_plans = st.selectbox(
                "Other Installment Plans",
                metadata["category_options"]["other_payment_plans"],
                index=metadata["category_options"]["other_payment_plans"].index(preset_values['other_payment_plans']),
                help="Existing payment arrangements with other banks or retail stores."
            )
            
        st.markdown("---")
        # Section B: Loan Application Details
        st.markdown("#### 💰 **2. Requested Loan Details**")
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            credit_amount = st.number_input(
                "Requested Credit Amount (DM)",
                min_value=250,
                max_value=20000,
                value=int(preset_values['credit_amount']),
                step=250,
                help="Total credit principal requested."
            )
        with col5:
            duration = st.number_input(
                "Loan Duration (Months)",
                min_value=4,
                max_value=72,
                value=int(preset_values['duration']),
                step=1,
                help="Duration of the repayment term in months."
            )
        with col6:
            installment_commitment = st.slider(
                "Installment Rate (% of Income)",
                min_value=1,
                max_value=4,
                value=int(preset_values['installment_commitment']),
                help="Installment payment expressed as percentage of monthly disposable income."
            )
        with col7:
            purpose = st.selectbox(
                "Loan Purpose",
                metadata["category_options"]["purpose"],
                index=metadata["category_options"]["purpose"].index(preset_values['purpose']),
                help="Intended usage of credit proceeds."
            )
            
        st.markdown("---")
        # Section C: Credit History & Past Performance
        st.markdown("#### 📜 **3. Credit History & Liabilities**")
        col8, col9, col10 = st.columns(3)
        with col8:
            credit_history = st.selectbox(
                "Credit Payment History",
                metadata["category_options"]["credit_history"],
                index=metadata["category_options"]["credit_history"].index(preset_values['credit_history']),
                help="Historical repayment track record across past credit lines."
            )
        with col9:
            existing_credits = st.slider(
                "Existing Credits at this Bank",
                min_value=1,
                max_value=4,
                value=int(preset_values['existing_credits']),
                help="Number of active or past loan contracts at this lending institution."
            )
        with col10:
            other_parties = st.selectbox(
                "Other Debtors / Guarantors",
                metadata["category_options"]["other_parties"],
                index=metadata["category_options"]["other_parties"].index(preset_values['other_parties']),
                help="Presence of a co-applicant or guarantor on the loan."
            )
            
        st.markdown("---")
        # Section D: Employment, Housing & Personal Stability
        st.markdown("#### 👤 **4. Employment, Housing & Demographics**")
        col11, col12, col13, col14 = st.columns(4)
        with col11:
            employment = st.selectbox(
                "Present Employment Tenure",
                metadata["category_options"]["employment"],
                index=metadata["category_options"]["employment"].index(preset_values['employment']),
                help="Years with current employer or business."
            )
            job = st.selectbox(
                "Job Qualification Level",
                metadata["category_options"]["job"],
                index=metadata["category_options"]["job"].index(preset_values['job'])
            )
        with col12:
            housing = st.selectbox(
                "Housing Arrangement",
                metadata["category_options"]["housing"],
                index=metadata["category_options"]["housing"].index(preset_values['housing'])
            )
            property_magnitude = st.selectbox(
                "Collateral / Property Owned",
                metadata["category_options"]["property_magnitude"],
                index=metadata["category_options"]["property_magnitude"].index(preset_values['property_magnitude'])
            )
        with col13:
            age = st.number_input(
                "Applicant Age (Years)",
                min_value=18,
                max_value=85,
                value=int(preset_values['age']),
                step=1
            )
            residence_since = st.slider(
                "Residence Tenure (Years)",
                min_value=1,
                max_value=4,
                value=int(preset_values['residence_since'])
            )
        with col14:
            num_dependents = st.selectbox(
                "Number of Dependents",
                [1, 2],
                index=[1, 2].index(preset_values['num_dependents'])
            )
            personal_status = st.selectbox(
                "Personal Status & Marital Status",
                metadata["category_options"]["personal_status"],
                index=metadata["category_options"]["personal_status"].index(preset_values['personal_status']),
                help="Used historically in the German credit benchmark."
            )
            own_telephone = st.selectbox(
                "Registered Telephone",
                metadata["category_options"]["own_telephone"],
                index=metadata["category_options"]["own_telephone"].index(preset_values['own_telephone'])
            )
            foreign_worker = st.selectbox(
                "Foreign Worker Status",
                metadata["category_options"]["foreign_worker"],
                index=metadata["category_options"]["foreign_worker"].index(preset_values['foreign_worker'])
            )
            
        submitted = st.form_submit_button("⚡ Evaluate Creditworthiness & Predict Default Risk", use_container_width=True)

    # -------------------------------------------------------------------------
    # Prediction Results Processing
    # -------------------------------------------------------------------------
    if submitted or preset_choice != "Custom Inputs":
        # Build DataFrame with identical schema
        input_dict = {
            'checking_status': [checking_status],
            'duration': [int(duration)],
            'credit_history': [credit_history],
            'purpose': [purpose],
            'credit_amount': [float(credit_amount)],
            'savings_status': [savings_status],
            'employment': [employment],
            'installment_commitment': [int(installment_commitment)],
            'personal_status': [personal_status],
            'other_parties': [other_parties],
            'residence_since': [int(residence_since)],
            'property_magnitude': [property_magnitude],
            'age': [int(age)],
            'other_payment_plans': [other_payment_plans],
            'housing': [housing],
            'existing_credits': [int(existing_credits)],
            'job': [job],
            'num_dependents': [int(num_dependents)],
            'own_telephone': [own_telephone],
            'foreign_worker': [foreign_worker]
        }
        input_df = pd.DataFrame(input_dict)
        
        # Inference through trained pipeline
        prob_dist = active_pipeline.predict_proba(input_df)[0]
        prob_good = prob_dist[0]
        prob_bad = prob_dist[1]  # Probability of Default (PD)
        
        # Apply user threshold
        is_high_risk = prob_bad >= decision_threshold
        
        # Simulated Credit Score Calculation (300 to 850 FICO-style scale)
        # Score = 300 + (1 - PD) * 550
        simulated_score = int(round(300 + (1.0 - prob_bad) * 550))
        simulated_score = max(300, min(850, simulated_score))
        
        # Credit Score Tier
        if simulated_score >= 740:
            tier = "Tier 1: Prime"
            tier_color = "#10b981"
            recommendation = "Auto-Approve (Standard/Preferred Interest Rate)"
        elif simulated_score >= 670:
            tier = "Tier 2: Near-Prime"
            tier_color = "#06b6d4"
            recommendation = "Approve with Standard Terms"
        elif simulated_score >= 580:
            tier = "Tier 3: Sub-Prime"
            tier_color = "#f59e0b"
            recommendation = "Conditional Approval (Manual Review / Enhanced Collateral Required)"
        else:
            tier = "Tier 4: Deep Sub-Prime / High Risk"
            tier_color = "#ef4444"
            recommendation = "Decline (Excessive Probability of Default)"

        st.markdown("---")
        st.markdown("### 📋 **Credit Underwriting Decision & Assessment**")
        
        # Status Banner
        if not is_high_risk:
            st.markdown(
                f'<div class="badge-approved">✅ CREDITWORTHY: LOW DEFAULT RISK<br>'
                f'<span style="font-size: 1.05rem; font-weight: normal;">Applicant satisfies credit risk criteria under the {decision_threshold:.0%} risk threshold.</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="badge-declined">⚠️ HIGH RISK: POTENTIAL CREDIT DEFAULT<br>'
                f'<span style="font-size: 1.05rem; font-weight: normal;">Applicant exhibits elevated risk indicators exceeding the {decision_threshold:.0%} risk threshold.</span></div>',
                unsafe_allow_html=True
            )

        # 4 Core Metrics
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric(
                label="Estimated Probability of Default (PD)",
                value=f"{prob_bad:.1%}",
                delta=f"Threshold: {decision_threshold:.0%}",
                delta_color="inverse"
            )
        with res_col2:
            st.metric(
                label="Simulated Credit Score",
                value=f"{simulated_score} / 850",
                delta=f"{tier}"
            )
        with res_col3:
            st.metric(
                label="Creditworthiness (P_Good)",
                value=f"{prob_good:.1%}",
                delta="Solvency Ratio"
            )
        with res_col4:
            st.metric(
                label="Underwriting Action",
                value="Approved" if not is_high_risk else "Declined/Review",
                delta="Policy Compliant"
            )

        # Risk Gauge Bar
        st.markdown("#### **Probability of Default Gauge**")
        st.progress(min(1.0, float(prob_bad)))
        st.caption(f"Default Risk: {prob_bad*100:.1f}% | Confidence Level: {max(prob_good, prob_bad)*100:.1f}%")
        
        # Policy Recommendation Box
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 5px solid {tier_color};">
                <h4 style="margin: 0; color: {tier_color};">Underwriter Policy Recommendation: {tier}</h4>
                <p style="margin: 5px 0 0 0; font-size: 1.02rem; font-weight: 500;"><b>Action:</b> {recommendation}</p>
                <p style="margin: 3px 0 0 0; color: #64748b; font-size: 0.9rem;">
                Monthly Installment Burden: <b>{installment_commitment}% of income</b> | Requested Loan Duration: <b>{duration} months</b> | Amount: <b>{credit_amount:,.0f} DM</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Key Explanatory Drivers
        st.markdown("#### 🔍 **Key Decision Drivers for This Profile**")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("##### 🟢 **Positive Credit Drivers**")
            pos_drivers = []
            if checking_status in ['>=200', 'no checking']:
                pos_drivers.append("Healthy checking status without overdraft incidence.")
            if savings_status in ['500<=X<1000', '>=1000']:
                pos_drivers.append("Strong liquid savings reserve buffer (>500 DM).")
            if employment in ['4<=X<7', '>=7']:
                pos_drivers.append("Long-term stable employment tenure (>= 4 years).")
            if housing == 'own':
                pos_drivers.append("Homeownership adds substantial asset stability.")
            if duration <= 18:
                pos_drivers.append("Short loan duration reduces macroeconomic default exposure.")
            if not pos_drivers:
                pos_drivers.append("No exceptional positive factors detected.")
            for d in pos_drivers:
                st.markdown(f"- {d}")

        with exp_col2:
            st.markdown("##### 🔴 **Risk-Elevating Factors**")
            neg_drivers = []
            if checking_status == '<0':
                neg_drivers.append("Overdrawn/negative checking account status (< 0 DM).")
            if savings_status == '<100':
                neg_drivers.append("Low liquid savings (< 100 DM) limiting cushion for financial shocks.")
            if duration > 36:
                neg_drivers.append("Extended loan tenure (> 36 months) significantly elevates default risk.")
            if installment_commitment >= 4:
                neg_drivers.append("High debt-to-income installment burden (>= 4% max tier).")
            if credit_history in ['delayed previously', 'no credits/all paid']:
                neg_drivers.append("History of delayed payments or unestablished credit line.")
            if not neg_drivers:
                neg_drivers.append("No critical risk flags triggered.")
            for d in neg_drivers:
                st.markdown(f"- {d}")


# =============================================================================
# TAB 2: Model Benchmark & Metrics
# =============================================================================
with tab2:
    st.markdown("### 📊 **Machine Learning Model Comparison & Evaluation**")
    st.markdown("To ensure fair, rigorous evaluation, four distinct classifiers were trained and tested on an identical stratified split (80% train, 20% test, $N=200$).")
    
    if metrics_df is not None:
        st.markdown("#### **Test Set Performance Benchmark Table**")
        st.dataframe(
            metrics_df.style.highlight_max(
                subset=['Test_ROC_AUC', 'Test_Recall', 'Test_Precision', 'Test_Accuracy', 'Test_F1'],
                color='#dcfce7'
            ).highlight_min(
                subset=['Business_Risk_Score'],
                color='#dbeafe'
            ),
            use_container_width=True
        )
        
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### **Receiver Operating Characteristic (ROC) Curves**")
        roc_path = os.path.join(MODELS_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True, caption="ROC Curves comparing True Positive Rate vs False Positive Rate.")
            
    with col_chart2:
        st.markdown("#### **Confusion Matrices on Test Set ($N=200$)**")
        cm_path = os.path.join(MODELS_DIR, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True, caption="Confusion Matrices: True Negatives, False Positives, False Negatives, True Positives.")

    st.markdown("---")
    st.markdown("### 💡 **Financial Context: Why Recall & ROC-AUC Outweigh Accuracy in Credit Risk**")
    
    st.markdown("""
    In typical commercial consumer lending, the cost of errors is profoundly **asymmetric**:
    
    * **False Negative (FN) - Approving a Defaulter:**
      The bank lends $10,000. The borrower defaults after 2 months. The bank loses the remaining principal ($9,000+), legal fees, and recovery costs.
      
    * **False Positive (FP) - Rejecting a Good Borrower:**
      The bank rejects a creditworthy borrower. The loss is strictly the missed profit margin (e.g. 5-8% net interest margin = $500).
      
    **Conclusion:** A single False Negative is roughly **5x to 10x more costly** than a False Positive!
    * A naive model that predicts everyone is "Good" gets **70% accuracy**, but catches **0% of defaulters** (0% Recall), causing bank insolvency.
    * **Logistic Regression with Balanced Weights** catches **80% of all defaulters** (Recall = 0.80) with a top ROC-AUC of **0.806**, minimizing the bank's aggregate credit losses.
    """)


# =============================================================================
# TAB 3: Exploratory Data Analysis (EDA)
# =============================================================================
with tab3:
    st.markdown("### 🔍 **Exploratory Data Analysis: Key Financial Drivers of Default**")
    st.markdown("Analysis based on 1,000 historical loan records from the German Credit Dataset.")
    
    if raw_df is not None:
        eda_col1, eda_col2, eda_col3 = st.columns(3)
        with eda_col1:
            st.metric("Total Loan Applicants", f"{len(raw_df):,}")
        with eda_col2:
            good_pct = (raw_df['class'] == 'good').mean() * 100
            st.metric("Creditworthy (Good)", f"{good_pct:.1f}%")
        with eda_col3:
            bad_pct = (raw_df['class'] == 'bad').mean() * 100
            st.metric("Defaulters (Bad)", f"{bad_pct:.1f}%")
            
        st.markdown("---")
        
        # Interactive EDA Charts
        eda_tab1, eda_tab2, eda_tab3 = st.tabs([
            "Checking & Savings Accounts",
            "Loan Amount vs Duration",
            "Feature Importances"
        ])
        
        with eda_tab1:
            st.markdown("#### **Default Rate by Financial Reserve Categories**")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Checking Status
            check_order = ['<0', '0<=X<200', '>=200', 'no checking']
            chk_bad_rate = raw_df.groupby('checking_status')['class'].apply(lambda s: (s == 'bad').mean()).reindex(check_order)
            sns.barplot(x=chk_bad_rate.index, y=chk_bad_rate.values * 100, hue=chk_bad_rate.index, palette='magma', legend=False, ax=ax1)
            ax1.set_title("Default Rate by Checking Account Status", fontsize=12, fontweight='bold')
            ax1.set_ylabel("Default Rate (%)")
            ax1.set_xlabel("Checking Status")
            for p in ax1.patches:
                ax1.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2., p.get_height() + 1), ha='center', fontweight='bold')
                
            # Savings Status
            sav_bad_rate = raw_df.groupby('savings_status')['class'].apply(lambda s: (s == 'bad').mean())
            sns.barplot(x=sav_bad_rate.index, y=sav_bad_rate.values * 100, hue=sav_bad_rate.index, palette='viridis', legend=False, ax=ax2)
            ax2.set_title("Default Rate by Savings Status", fontsize=12, fontweight='bold')
            ax2.set_ylabel("Default Rate (%)")
            ax2.set_xlabel("Savings Status")
            ax2.tick_params(axis='x', rotation=30)
            for p in ax2.patches:
                ax2.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2., p.get_height() + 1), ha='center', fontweight='bold')
                
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        with eda_tab2:
            st.markdown("#### **Credit Amount vs Repayment Duration by Credit Risk**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.scatterplot(
                data=raw_df,
                x='duration',
                y='credit_amount',
                hue='class',
                palette={'good': '#10b981', 'bad': '#ef4444'},
                alpha=0.75,
                s=60,
                ax=ax
            )
            ax.set_title("Credit Amount vs Loan Duration (Months)", fontsize=13, fontweight='bold')
            ax.set_xlabel("Duration in Months")
            ax.set_ylabel("Credit Amount (DM)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        with eda_tab3:
            feat_imp_path = os.path.join(MODELS_DIR, "feature_importance.png")
            if os.path.exists(feat_imp_path):
                st.image(feat_imp_path, use_container_width=True, caption="Top Feature Importances from Random Forest Classifier.")


# =============================================================================
# TAB 4: Responsible AI & Fair Lending
# =============================================================================
with tab4:
    st.markdown("### ⚖️ **Responsible AI, Model Fairness & Regulatory Compliance**")
    
    st.markdown("""
    Credit scoring models directly dictate whether consumers obtain mortgages, auto loans, or small business credit. Consequently, automated financial decisioning is strictly regulated under global consumer protection frameworks:
    """)
    
    col_reg1, col_reg2, col_reg3 = st.columns(3)
    with col_reg1:
        st.markdown(
            """
            <div class="metric-card">
                <h4>📜 ECOA (Regulation B)</h4>
                <p style="font-size: 0.9rem; color: #475569;">
                <b>Equal Credit Opportunity Act:</b> Prohibits credit discrimination based on race, color, religion, national origin, sex, marital status, or age.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
    with col_reg2:
        st.markdown(
            """
            <div class="metric-card">
                <h4>🔍 FCRA & Adverse Action</h4>
                <p style="font-size: 0.9rem; color: #475569;">
                <b>Fair Credit Reporting Act:</b> Mandates that lenders provide specific, actionable reasons (Adverse Action Notices) when credit is denied.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
    with col_reg3:
        st.markdown(
            """
            <div class="metric-card">
                <h4>🇪🇺 EU AI Act (High-Risk AI)</h4>
                <p style="font-size: 0.9rem; color: #475569;">
                Classifies AI credit scoring as <b>High-Risk AI</b>, requiring rigorous bias testing, continuous post-market monitoring, and human oversight.
                </p>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("#### **Key Ethical Dilemmas in Historical Financial Datasets**")
    st.markdown("""
    1. **Proxy Discrimination:** Even if protected attributes like race or gender are omitted, algorithms can reconstruct them using proxy variables (e.g. zip code, retail spending habits, or historical personal status).
    2. **Historical Bias Amplification:** Historical lending data reflects past societal disparities in wage gaps and credit access. An unconstrained ML model risks perpetuating these historical inequities.
    3. **Fairness Metrics:**
       * **Demographic Parity:** Approval rates are equal across protected demographic groups ($P(\\hat{Y}=1 | A=0) = P(\\hat{Y}=1 | A=1)$).
       * **Equalized Odds:** True Positive and False Positive rates are equivalent across groups.
    """)


# =============================================================================
# TAB 5: Technical Architecture
# =============================================================================
with tab5:
    st.markdown("### ℹ️ **End-to-End System Architecture & Methodology**")
    
    st.markdown("""
    ```mermaid
    flowchart LR
        A[Raw German Credit Data] --> B[Data Preprocessing Pipeline]
        B -->|StandardScaler| C1[Numerical Features]
        B -->|OneHotEncoder| C2[Categorical Features]
        C1 --> D[Stratified 80/20 Train-Test Split]
        C2 --> D
        D --> E[Model Benchmark & Tuning]
        E --> F1[Logistic Regression - Balanced]
        E --> F2[Decision Tree]
        E --> F3[Random Forest]
        E --> F4[Gradient Boosting]
        F1 & F2 & F3 & F4 --> G[Evaluation: ROC-AUC, Recall, Cost Loss]
        G --> H[Joblib Serialization]
        H --> I[Streamlit Web Dashboard]
    ```
    """)
    
    st.markdown("#### **Evaluation Metrics Reference Sheet**")
    st.markdown("""
    * **Recall (Sensitivity):** $\\frac{\\text{TP}}{\\text{TP} + \\text{FN}}$ — Percentage of actual defaulters caught by the model.
    * **Precision:** $\\frac{\\text{TP}}{\\text{TP} + \\text{FP}}$ — Percentage of flagged high-risk loans that actually defaulted.
    * **F1-Score:** $2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$ — Harmonic mean of precision and recall.
    * **ROC-AUC:** Area under the Receiver Operating Characteristic curve; measures discriminative capacity across all potential classification thresholds.
    """)


# -----------------------------------------------------------------------------
# 6. Global Regulatory Disclaimer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-box">
        <b>⚖️ Regulatory & Educational Disclaimer:</b><br>
        This machine learning application is built strictly for educational, portfolio demonstration, and academic evaluation purposes using historical benchmark data (German Credit Data). 
        Predictions and simulated credit scores generated by this system are mathematical estimates and <b>must not</b> be utilized as the sole or primary basis for real-world underwriting, credit issuance, or adverse financial decisions without independent validation, demographic fairness audits, and regulatory certification under applicable credit lending laws.
    </div>
    """,
    unsafe_allow_html=True
)
