"""
Heart Disease Prediction Web App
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
# Load data (robust)
# ---------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("heart.csv")
        st.write(f"Read CSV with header, shape: {df.shape}")
    except:
        col_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                     'thalach','exang','oldpeak','slope','ca','thal','target_raw']
        df = pd.read_csv("heart.csv", names=col_names, header=None)
        st.write(f"Read CSV without header, shape: {df.shape}")

    possible_targets = ['target', 'Heart Disease', 'target_raw', 'num', 'condition']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]
        st.warning(f"Assuming last column '{target_col}' is the target.")
    
    df = df.replace('?', np.nan)
    
    try:
        numeric_target = pd.to_numeric(df[target_col])
        df['target'] = numeric_target.apply(lambda x: 1 if x > 0 else 0)
    except:
        target_str = df[target_col].astype(str).str.lower()
        if 'presence' in target_str.values or 'present' in target_str.values:
            df['target'] = target_str.apply(lambda x: 1 if x in ['presence','present'] else 0)
        elif 'yes' in target_str.values:
            df['target'] = target_str.apply(lambda x: 1 if x == 'yes' else 0)
        else:
            df['target'] = target_str.apply(lambda x: 0 if x in ['0','absence','no'] else 1)

    df = df.drop(columns=[target_col])
    df = df.apply(pd.to_numeric, errors='coerce')
    before = len(df)
    df = df.dropna()
    df = df.astype(float)
    df['target'] = df['target'].astype(int)
    return df

def show_target_distribution(df):
    target_counts = df['target'].value_counts().reset_index()
    target_counts.columns = ['Heart Disease', 'Count']
    target_counts['Percentage'] = (target_counts['Count'] / df.shape[0] * 100).round(2)
    target_counts['Heart Disease'] = target_counts['Heart Disease'].map({0: 'No Disease', 1: 'Disease'})
    st.table(target_counts)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Prediction Web App")

df = load_data()
if df is None or df.empty:
    st.error("No data loaded. Please check your heart.csv file.")
    st.stop()

X = df.drop('target', axis=1)
y = df['target']

menu = st.sidebar.selectbox("Menu", ["Dataset Overview", "Train Models", "Predict Heart Disease", "Data Visualization"])

# Session state
if "trained" not in st.session_state:
    st.session_state.trained = False
if "performance_test" not in st.session_state:
    st.session_state.performance_test = []
if "models" not in st.session_state:
    st.session_state.models = {}          # will store {'KNN': knn_model, 'SVM': svm_model}
if "X_test_scaled" not in st.session_state:
    st.session_state.X_test_scaled = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None

# ---------------------------
# Menu: Dataset Overview
# ---------------------------
if menu == "Dataset Overview":
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(10))
    st.dataframe(df.tail(10))
    st.write(f"**Total samples:** {df.shape[0]}")
    st.write("**Target distribution:**")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df['target'].value_counts())
    with col2:
        show_target_distribution(df)

# ---------------------------
# Menu: Train Models
# ---------------------------
elif menu == "Train Models":
    st.subheader("⚙️ Train KNN & SVM with SMOTE")
    show_target_distribution(df)
    st.write("### Original Class Distribution (before SMOTE)")
    st.write(y.value_counts())
    
    test_size = st.slider("Test size (%)", 10, 40, 20) / 100
    
    if st.button("🚀 Train Models"):
        with st.spinner("Training in progress..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            st.write("After SMOTE (training set):")
            st.write(pd.Series(y_train_res).value_counts())
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_res)
            X_test_scaled = scaler.transform(X_test)
            
            os.makedirs("models", exist_ok=True)
            joblib.dump(scaler, "models/scaler.joblib")
            
            knn = KNeighborsClassifier(n_neighbors=5)
            svm = SVC(probability=True, random_state=42)
            knn.fit(X_train_scaled, y_train_res)
            svm.fit(X_train_scaled, y_train_res)
            joblib.dump(knn, "models/knn.joblib")
            joblib.dump(svm, "models/svm.joblib")
            
            # Store models and test data for later visualisation
            st.session_state.models = {"KNN": knn, "SVM": svm}
            st.session_state.X_test_scaled = X_test_scaled
            st.session_state.y_test = y_test
            
            # Evaluate
            results = []
            for name, model in [("KNN", knn), ("SVM", svm)]:
                y_pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                                "Recall": rec, "F1 Score": f1, "MSE": mse, "RMSE": rmse})
            st.session_state.performance_test = results
            st.session_state.trained = True
            
            st.subheader("📊 Model Performance Comparison (Test Set)")
            comparison_df = pd.DataFrame(results)
            for col in comparison_df.columns:
                if col != "Model":
                    comparison_df[col] = comparison_df[col].map(lambda x: f"{x:.4f}")
            st.table(comparison_df)
        st.success("✅ All models trained and saved")

# ---------------------------
# Menu: Predict Heart Disease
# ---------------------------
elif menu == "Predict Heart Disease":
    st.subheader("🩺 Enter Patient Data")
    show_target_distribution(df)
    if not os.path.exists("models/knn.joblib"):
        st.warning("Models not trained yet. Please go to 'Train Models' first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 1, 120, 50)
            sex = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
            cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
            trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
            restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
        with col2:
            thalach = st.number_input("Max Heart Rate (bpm)", 60, 220, 150)
            exang = st.selectbox("Exercise Angina", [0,1])
            oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
            slope = st.selectbox("Slope (0-2)", [0,1,2])
            ca = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
            thal = st.selectbox("Thalassemia (1-3)", [1,2,3])
        
        model_choice = st.radio("Select Model", ["KNN", "SVM"], horizontal=True)
        
        if st.button("Predict"):
            scaler = joblib.load("models/scaler.joblib")
            model = joblib.load(f"models/{model_choice.lower()}.joblib")
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
    st.subheader("📊 Visualizations")
    
    # 1. Target distribution table
    st.write("### Target Distribution")
    show_target_distribution(df)
    
    # 2. Dataset correlation matrix (once)
    st.write("### Correlation Matrix of Features (Dataset)")
    fig_corr, ax_corr = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)
    
    if not st.session_state.trained:
        st.warning("Please train models first to see confusion matrices and ROC curves.")
    else:
        # 3. Confusion matrices for KNN and SVM
        st.write("### Confusion Matrices: KNN vs SVM")
        models = st.session_state.models
        X_test_scaled = st.session_state.X_test_scaled
        y_test = st.session_state.y_test
        
        fig_cm, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (name, model) in zip(axes, models.items()):
            y_pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Disease', 'Disease'],
                        yticklabels=['No Disease', 'Disease'])
            ax.set_title(f'{name} - Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        st.pyplot(fig_cm)
        
        # 4. ROC Curves with AUC for both models
        st.write("### ROC Curves (with AUC)")
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_score = model.decision_function(X_test_scaled)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curves – KNN vs SVM')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(alpha=0.3)
        st.pyplot(fig_roc)
        
        # 5. Performance comparison table and bar chart
        st.write("### Model Performance Comparison (Metrics)")
        perf_df = pd.DataFrame(st.session_state.performance_test)
        st.table(perf_df.style.format({
            'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}',
            'F1 Score': '{:.4f}', 'MSE': '{:.4f}', 'RMSE': '{:.4f}'
        }))
        
        # Bar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        melted = perf_df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
        fig_bar, ax_bar = plt.subplots(figsize=(8,5))
        sns.barplot(data=melted, x='Metric', y='Score', hue='Model', ax=ax_bar)
        ax_bar.set_ylim(0,1)
        ax_bar.set_title("KNN vs SVM Performance")
        st.pyplot(fig_bar)
