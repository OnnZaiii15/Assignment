"""
Heart Disease Prediction Web App (Robust)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# ---------------------------
# Streamlit app configuration
# ---------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Prediction Web App")
st.markdown("Built with **Streamlit** | Uses **K‑Nearest Neighbors (KNN)** and **Support Vector Machine (SVM)**")

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Dataset Overview", "Train Models", "Predict Heart Disease", "Data Visualization"])

# ---------------------------
# Load data with robust preprocessing
# ---------------------------
@st.cache_data
def load_data():
    
    # Try to read the CSV – first assume it has a header
    try:
        df = pd.read_csv("heart.csv")
        st.write(f"Read CSV with header, shape: {df.shape}")
    except:
        # If that fails, try reading without header (UCI format)
        column_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                        'thalach','exang','oldpeak','slope','ca','thal','target_raw']
        df = pd.read_csv("heart.csv", names=column_names, header=None)
        st.write(f"Read CSV without header, shape: {df.shape}")

    # Check the target column name – might be 'target', 'Heart Disease', or 'target_raw'
    possible_target_cols = ['target', 'Heart Disease', 'target_raw', 'num', 'condition']
    target_col = None
    for col in possible_target_cols:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        # If none of the above, assume the last column is the target
        target_col = df.columns[-1]
        st.warning(f"No known target column name found; assuming last column '{target_col}' is the target.")
    
    # Convert target column to binary (0 = no disease, 1 = disease)
    # First, handle missing values (represented as '?')
    df = df.replace('?', np.nan)
    
    # Try to convert the target column to numeric; if it fails, it contains strings
    try:
        numeric_target = pd.to_numeric(df[target_col])
        # Numeric: original 0 = no disease, 1,2,3,4 = disease -> map >0 to 1
        df['target'] = numeric_target.apply(lambda x: 1 if x > 0 else 0)
    except (ValueError, TypeError):
        # String target: typical values are 'Presence'/'Absence' or 'Yes'/'No'
        target_str = df[target_col].astype(str).str.lower()
        if 'presence' in target_str.values or 'present' in target_str.values:
            df['target'] = target_str.apply(lambda x: 1 if x in ['presence', 'present'] else 0)
        elif 'yes' in target_str.values:
            df['target'] = target_str.apply(lambda x: 1 if x == 'yes' else 0)
        else:
            # Fallback: assume any non-zero string means disease
            df['target'] = target_str.apply(lambda x: 0 if x == '0' or x == 'absence' or x == 'no' else 1)
    
    # Drop the original target column
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    
    # Replace '?' in feature columns with NaN and drop rows with any NaN
    df = df.apply(pd.to_numeric, errors='coerce')  # force all feature columns to numeric
    before = len(df)
    df = df.dropna()
    st.write(f"Dropped {before - len(df)} rows with missing values.")
    
    # Ensure all columns are numeric and target is integer
    df = df.astype(float)
    df['target'] = df['target'].astype(int)
    
    st.write(f"✅ Final dataset: {len(df)} patients, {df.shape[1]-1} features")
    st.write("Target distribution:")
    st.write(df['target'].value_counts())
    return df


df = load_data()
    X = df.drop('target', axis=1)
    y = df['target']
def train_models(X_train, y_train):
    """Train KNN and SVM models and return them."""
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(probability=True, random_state=42)
    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(knn, "models/knn.joblib")
    joblib.dump(svm, "models/svm.joblib")
    return {"KNN": knn, "SVM": svm}
    

# Session state
if "trained" not in st.session_state:
    st.session_state.trained = False
if "performance_train" not in st.session_state:
    st.session_state.performance_train = []
if "performance_test" not in st.session_state:
    st.session_state.performance_test = []
if "scaler" not in st.session_state:
    st.session_state.scaler = None

# ---------------------------
# Menu: Dataset Overview
# ---------------------------
if menu == "Dataset Overview":
    st.subheader("📊 Dataset Overview")
    st.markdown("""
    **Cleveland Heart Disease Dataset** – 303 patient records, 13 clinical features.
    - **target**: 0 = no heart disease, 1 = heart disease
    """)
    st.write("First 10 rows after preprocessing:")
    st.dataframe(df.head(10))
    st.write("Last 10 rows:")
    st.dataframe(df.tail(10))
    st.write(f"**Shape:** {df.shape}")
    st.write("**Missing values:**", df.isnull().sum().sum())
    st.write("**Target distribution:**")
    st.bar_chart(df['target'].value_counts())
    col1, col2 = st.columns(2)
    col1.metric("No Heart Disease", (df['target']==0).sum())
    col2.metric("Heart Disease", (df['target']==1).sum())

if st.button("🚀 Train Models"):
    with st.spinner("Training in progress..."):
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        # SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)
        # Save scaler
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.joblib")
        
        # Train models (inline, no function call)
        knn = KNeighborsClassifier(n_neighbors=5)
        svm = SVC(probability=True, random_state=42)
        knn.fit(X_train_scaled, y_train_res)
        svm.fit(X_train_scaled, y_train_res)
        joblib.dump(knn, "models/knn.joblib")
        joblib.dump(svm, "models/svm.joblib")
        models = {"KNN": knn, "SVM": svm}
        
# ---------------------------
# Menu: Predict Heart Disease
# ---------------------------
elif menu == "Predict Heart Disease":
    st.subheader("🩺 Enter Patient Data")
    if not st.session_state.trained and not (os.path.exists("models/knn.joblib") and os.path.exists("models/scaler.joblib")):
        st.warning("Models not found. Please go to **Train Models** first.")
    else:
        # Input fields (all 13 features)
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", 1, 120, 50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            cp = st.selectbox("Chest Pain Type", [0,1,2,3], format_func=lambda x: {0:"Typical angina",1:"Atypical angina",2:"Non-anginal pain",3:"Asymptomatic"}[x])
            trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0,1], format_func=lambda x: "False" if x==0 else "True")
            restecg = st.selectbox("Resting ECG", [0,1,2], format_func=lambda x: {0:"Normal",1:"ST-T abnormality",2:"LV hypertrophy"}[x])
        with col2:
            thalach = st.number_input("Max Heart Rate (bpm)", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
            oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
            slope = st.selectbox("Slope of ST Segment", [0,1,2], format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
            ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
            thal = st.selectbox("Thalassemia", [1,2,3], format_func=lambda x: {1:"Normal",2:"Fixed defect",3:"Reversible defect"}[x])

        model_choice = st.radio("Select Model", ["KNN", "SVM"], horizontal=True)

        if st.button("Predict"):
            # Load scaler and model
            scaler = joblib.load("models/scaler.joblib")
            model = load_model(model_choice)
            if model is None:
                st.error(f"{model_choice} model not found. Train models first.")
            else:
                input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                        thalach, exang, oldpeak, slope, ca, thal]])
                input_scaled = scaler.transform(input_data)
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0][pred] * 100
                result = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
                st.success(f"**Prediction:** {result} (confidence {prob:.1f}%)")

# ---------------------------
# Menu: Data Visualization
# ---------------------------
elif menu == "Data Visualization":
    st.subheader("📊 Data Visualizations")
    if not st.session_state.trained and not os.path.exists("models/knn.joblib"):
        st.warning("Models not trained. Please train them first in 'Train Models'.")
    else:
        # Correlation heatmap
        st.write("### Correlation Matrix of Features")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Model performance comparison
        if st.session_state.performance_test:
            st.write("### Model Performance Comparison (Test Set)")
            perf_df = pd.DataFrame(st.session_state.performance_test)
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            melted = perf_df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
            fig2, ax2 = plt.subplots(figsize=(8,5))
            sns.barplot(data=melted, x='Metric', y='Score', hue='Model', ax=ax2)
            ax2.set_ylim(0,1)
            ax2.set_title("KNN vs SVM Performance")
            st.pyplot(fig2)

            # Confusion matrices (load models and compute)
            st.write("### Confusion Matrices")
            scaler = joblib.load("models/scaler.joblib")
            X = df.drop('target', axis=1)
            y = df['target']
            X_scaled = scaler.fit_transform(X)  # refit scaler on full data for display (or use saved)
            # For simplicity, we compute on whole dataset (not ideal but for demo)
            knn = load_model("KNN")
            svm = load_model("SVM")
            if knn and svm:
                fig3, axes = plt.subplots(1,2, figsize=(12,5))
                for ax, (name, model) in zip(axes, [("KNN", knn), ("SVM", svm)]):
                    y_pred = model.predict(X_scaled)
                    cm = confusion_matrix(y, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['No Disease', 'Disease'],
                                yticklabels=['No Disease', 'Disease'])
                    ax.set_title(f"{name} - Confusion Matrix")
                st.pyplot(fig3)

                # ROC curves
                st.write("### ROC Curves")
                fig4, ax4 = plt.subplots(figsize=(8,6))
                for name, model in [("KNN", knn), ("SVM", svm)]:
                    if hasattr(model, "predict_proba"):
                        y_score = model.predict_proba(X_scaled)[:,1]
                    else:
                        y_score = model.decision_function(X_scaled)
                    fpr, tpr, _ = roc_curve(y, y_score)
                    roc_auc = auc(fpr, tpr)
                    ax4.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
                ax4.plot([0,1], [0,1], 'k--')
                ax4.set_xlabel("False Positive Rate")
                ax4.set_ylabel("True Positive Rate")
                ax4.set_title("ROC Curves")
                ax4.legend()
                st.pyplot(fig4)
        else:
            st.info("No performance data yet. Train models first.")
