"""
Heart Disease Prediction Web App
Supervised Machine Learning: KNN & SVM
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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, mean_squared_error, confusion_matrix,
                             roc_curve, auc)
from imblearn.over_sampling import SMOTE

# ---------------------------
# Helper functions
# ---------------------------
def load_and_preprocess():
    """Load heart.csv and preprocess it for the Cleveland dataset."""
    if not os.path.exists("heart.csv"):
        st.error("❌ heart.csv file not found! Please upload it to the same folder.")
        return None
    
    df = pd.read_csv("heart.csv")
    st.write(f"Raw data shape: {df.shape}")
    
    # If the CSV has no header (first row is data), assign column names
    # Standard Cleveland dataset has 14 columns (13 features + target)
    if df.shape[1] == 14 and not df.iloc[0, 13] in [0,1,2,3,4, '0','1','2','3','4']:
        # Assume first row is header? Let's check: typical header might be 'age','sex',...
        # Actually, easier: let's just use the UCI column names
        col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        # If first row looks like numbers, then there is no header
        try:
            first_val = df.iloc[0, 0]
            if isinstance(first_val, (int, float)) or str(first_val).replace('.','').isdigit():
                df.columns = col_names
            else:
                # First row might be header; we'll use it as is but rename
                df.columns = col_names
        except:
            df.columns = col_names
    else:
        # Already has column names? We'll rename to standard names
        standard_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        if len(df.columns) == 14:
            df.columns = standard_names
        else:
            st.error(f"Expected 14 columns, but found {df.shape[1]}. Please check your CSV.")
            return None
    
    # Handle missing values (represented as '?')
    df = df.replace('?', np.nan)
    # Convert all columns to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    # Drop rows with any NaN
    before = len(df)
    df = df.dropna()
    st.write(f"Dropped {before - len(df)} rows with missing values.")
    
    # Convert target: original values 1,2,3,4 -> 1 (disease), 0 -> 0 (no disease)
    if 'target' in df.columns:
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    else:
        st.error("No 'target' column found after preprocessing.")
        return None
    
    st.write(f"Final dataset shape: {df.shape}")
    st.write("Target distribution:", df['target'].value_counts().to_dict())
    return df

def train_models(X_train, y_train):
    """Train KNN and SVM models and save them."""
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(probability=True, random_state=42)
    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(knn, "models/knn.joblib")
    joblib.dump(svm, "models/svm.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    return {"KNN": knn, "SVM": svm}

def load_model(model_name):
    path = f"models/{model_name.lower()}.joblib"
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ---------------------------
# Streamlit app
# ---------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Prediction Web App")
st.markdown("Uses **KNN** and **SVM** with SMOTE balancing")

# Load data
df = load_and_preprocess()
if df is None:
    st.stop()

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Dataset Overview", "Train Models", "Predict Heart Disease", "Data Visualization"])

# Initialize session state
if "trained" not in st.session_state:
    st.session_state.trained = False
if "performance_train" not in st.session_state:
    st.session_state.performance_train = []
if "performance_test" not in st.session_state:
    st.session_state.performance_test = []

# ---------------------------
# Menu: Dataset Overview
# ---------------------------
if menu == "Dataset Overview":
    st.subheader("📊 Dataset Overview")
    st.write("First 10 rows:")
    st.dataframe(df.head(10))
    st.write("Last 10 rows:")
    st.dataframe(df.tail(10))
    st.write(f"**Shape:** {df.shape}")
    st.write("**Target distribution:**")
    st.bar_chart(df['target'].value_counts())
    col1, col2 = st.columns(2)
    col1.metric("No Heart Disease", (df['target']==0).sum())
    col2.metric("Heart Disease", (df['target']==1).sum())

# ---------------------------
# Menu: Train Models
# ---------------------------
elif menu == "Train Models":
    st.subheader("⚙️ Train KNN & SVM Models (with SMOTE)")
    X = df.drop('target', axis=1)
    y = df['target']
    
    st.write("Original class distribution:")
    st.write(y.value_counts())
    
    test_size = st.slider("Test size (%)", 10, 40, 20) / 100
    
    if st.button("🚀 Train Models"):
        with st.spinner("Training..."):
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            # SMOTE only on training set
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            st.write("After SMOTE (training set):")
            st.write(pd.Series(y_train_res).value_counts())
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_res)
            X_test_scaled = scaler.transform(X_test)
            
            # Save scaler globally for prediction
            if not os.path.exists("models"):
                os.makedirs("models")
            joblib.dump(scaler, "models/scaler.joblib")
            
            # Train
            knn = KNeighborsClassifier(n_neighbors=5)
            svm = SVC(probability=True, random_state=42)
            knn.fit(X_train_scaled, y_train_res)
            svm.fit(X_train_scaled, y_train_res)
            joblib.dump(knn, "models/knn.joblib")
            joblib.dump(svm, "models/svm.joblib")
            models = {"KNN": knn, "SVM": svm}
            
            # Evaluate
            perf_train = []
            perf_test = []
            for name, model in models.items():
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
                acc_train = accuracy_score(y_train_res, y_train_pred)
                acc_test = accuracy_score(y_test, y_test_pred)
                prec = precision_score(y_test, y_test_pred)
                rec = recall_score(y_test, y_test_pred)
                f1 = f1_score(y_test, y_test_pred)
                mse = mean_squared_error(y_test, y_test_pred)
                rmse = np.sqrt(mse)
                perf_train.append({"Model": name, "Accuracy": acc_train})
                perf_test.append({"Model": name, "Accuracy": acc_test,
                                  "Precision": prec, "Recall": rec,
                                  "F1 Score": f1, "MSE": mse, "RMSE": rmse})
            st.session_state.performance_train = perf_train
            st.session_state.performance_test = perf_test
            st.session_state.trained = True
        st.success("✅ Models trained and saved!")
        st.subheader("Test Set Performance")
        st.dataframe(pd.DataFrame(perf_test))

# ---------------------------
# Menu: Predict Heart Disease
# ---------------------------
elif menu == "Predict Heart Disease":
    st.subheader("🩺 Enter Patient Data")
    if not (os.path.exists("models/knn.joblib") and os.path.exists("models/scaler.joblib")):
        st.warning("Models not trained yet. Please go to 'Train Models' first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 1, 120, 50)
            sex = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
            cp = st.selectbox("Chest Pain Type", [0,1,2,3])
            trestbps = st.number_input("Resting BP", 80, 200, 120)
            chol = st.number_input("Cholesterol", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
            restecg = st.selectbox("Resting ECG", [0,1,2])
        with col2:
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Angina", [0,1])
            oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
            slope = st.selectbox("Slope", [0,1,2])
            ca = st.selectbox("Number of Major Vessels", [0,1,2,3])
            thal = st.selectbox("Thalassemia", [1,2,3])
        
        model_choice = st.radio("Select Model", ["KNN", "SVM"], horizontal=True)
        
        if st.button("Predict"):
            scaler = joblib.load("models/scaler.joblib")
            model = load_model(model_choice)
            if model is None:
                st.error("Model not found. Train first.")
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
    st.subheader("📊 Visualizations")
    if not st.session_state.trained:
        st.warning("Please train models first.")
    else:
        # Correlation heatmap
        st.write("### Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        # Performance comparison
        if st.session_state.performance_test:
            perf_df = pd.DataFrame(st.session_state.performance_test)
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            melted = perf_df.melt(id_vars='Model', value_vars=metrics)
            fig2, ax2 = plt.subplots()
            sns.barplot(data=melted, x='variable', y='value', hue='Model', ax=ax2)
            ax2.set_ylim(0,1)
            st.pyplot(fig2)
