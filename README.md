🖥️ How to Use the App

1. Dataset Overview
   - Check the first/last 10 rows of the cleaned dataset.
   - See the target distribution (No Disease vs Disease) as both a bar chart and a table.

2. Train Models
   - Before training, the app shows the target distribution.
   - Adjust the test size slider (percentage of data to keep for testing).
   - Click "Train Models".
   - The app will:
      -  Split the data (stratified)
      -  Apply SMOTE to balance the training set
      -  Scale all features
      -  Train KNN and SVM
      -  Save the models and scaler in the models/ folder
      -  Display a performance comparison table (Accuracy, Precision, Recall, F1, MSE, RMSE)

3. Predict Heart Disease
   - Enter patient data using the input fields (age, sex, chest pain type, blood pressure, cholesterol, etc.).
   - Choose the model you want to use (KNN or SVM).
   - Click "Predict", you will see:
   - Heart Disease Detected or No Heart Disease
   - Confidence percentage (probability)

4. Data Visualization
   - Target Distribution: same table as before.
   - Correlation Matrix: shows how features are correlated with each other and with the target.
   - Confusion Matrices: side‑by‑side for KNN and SVM (only visible after training).
   - ROC Curves: with AUC values for both models.
   - Performance Table & Bar Chart – compare accuracy, precision, recall, and F1‑score.

📊 Dataset Description (Cleveland)
   - The dataset contains 303 patient records with 13 clinical features:
      -  age : Age in years
      -  sex : 1 = male, 0 = female
      -  cp : Chest pain type (0–3)
      -  trestbps : Resting blood pressure (mm Hg)
      -  chol : Serum cholesterol (mg/dl)
      -  fbs : Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
      -  restecg : Resting ECG results (0,1,2)
      -  thalach : Maximum heart rate achieved
      -  exang : Exercise induced angina (1 = yes, 0 = no)
      -  oldpeak : ST depression induced by exercise
      -  slope : Slope of peak exercise ST segment (0,1,2)
      -  ca : Number of major vessels (0–3)
      -  thal : Thalassemia (1,2,3)
      -  target : Diagnosis (0 = no disease, 1 = disease)
   
🤝 Contributing
   - Feel free to open issues or submit pull requests if you find bugs or want to improve the app.

📝 License
   - This project is for educational purposes, part of an AI assignment.

🙏 Acknowledgements
   - UCI Supervised Machine Learning Repository for the Cleveland Heart Disease dataset.
   - Streamlit for making ML app deployment so easy.
   - Scikit‑learn and imbalanced‑learn for the machine learning tools.
