"""
Heart Disease Prediction Web App (Simplified & Robust)
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
# Load data with caching
# ---------------------------
@st.cache_data
def load_data():
    # The Cleveland dataset has 14 columns, no header
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target_raw'
    ]
    
    try:
        # Try reading with no header (the default UCI format)
        df = pd.read_csv("heart.csv", names=column_names, header=None)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        # Fallback: try with header
        df = pd.read_csv("heart.csv")
        if len(df.columns) == 14:
            df.columns = column_names
        else:
            st.error("CSV does not have 14 columns. Please check the file.")
            return None
    
    # Convert target: original values 1,2,3,4 -> 1 (disease), 0 -> 0 (no disease)
    df['target'] = df['target_raw'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop('target_raw', axis=1)
    
    # Replace '?' with NaN and drop missing rows
    df = df.replace('?', np.nan)
    df = df.dropna()
    
    # Ensure all columns are numeric
    df = df.astype(float)
    
    st.write(f"✅ Loaded {len(df)} patient records")
    st.write(f"Target distribution:\n{df['target'].value_counts()}")
    return df

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Prediction Web App")
st.markdown("Uses **KNN** and **SVM** with SMOTE balancing")

df = load_data()
if df is None:
    st.stop()

# Sidebar
menu = st.sidebar.selectbox("Menu", ["Dataset Overview", "Train Models", "Predict Heart Disease", "Data Visualization"])

# Session state
if "trained" not in st.session_state:
    st.session_state.trained = False
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
    st.write(f"**Total samples:** {df.shape[0]}")
    st.write("**Target distribution:**")
    st.bar_chart(df['target'].value_counts())

# ---------------------------
# Menu: Train Models
# ---------------------------
elif menu == "Train Models":
    st.subheader("⚙️ Train KNN & SVM with SMOTE")
    X = df.drop('target', axis=1)
    y = df['target']
    
    st.write("Original class distribution:")
    st.write(y.value_counts())
    
    test_size = st.slider("Test size (%)", 10, 40, 20) / 100
    
    if st.button("🚀 Train Models"):
        with st.spinner("Training in progress..."):
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            # SMOTE
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            st.write("After SMOTE (training set):")
            st.write(pd.Series(y_train_res).value_counts())
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_res)
            X_test_scaled = scaler.transform(X_test)
            
            # Save scaler and models
            os.makedirs("models", exist_ok=True)
            joblib.dump(scaler, "models/scaler.joblib")
            
            knn = KNeighborsClassifier(n_neighbors=5)
            svm = SVC(probability=True, random_state=42)
            knn.fit(X_train_scaled, y_train_res)
            svm.fit(X_train_scaled, y_train_res)
            joblib.dump(knn, "models/knn.joblib")
            joblib.dump(svm, "models/svm.joblib")
            
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
        st.success("✅ Models trained and saved!")
        st.subheader("Test Set Performance")
        st.dataframe(pd.DataFrame(results))

# ---------------------------
# Menu: Predict Heart Disease
# ---------------------------
elif menu == "Predict Heart Disease":
    st.subheader("🩺 Enter Patient Data")
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
    if not st.session_state.trained:
        st.warning("Please train models first.")
    else:
        # Correlation heatmap
        st.write("### Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        # Performance bar chart
        if st.session_state.performance_test:
            perf_df = pd.DataFrame(st.session_state.performance_test)
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            melted = perf_df.melt(id_vars='Model', value_vars=metrics, var_name='Metric', value_name='Score')
            fig2, ax2 = plt.subplots()
            sns.barplot(data=melted, x='Metric', y='Score', hue='Model', ax=ax2)
            ax2.set_ylim(0,1)
            ax2.set_title("Model Performance Comparison")
            st.pyplot(fig2)
