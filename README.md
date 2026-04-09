🖥️ How to Use the App

1. Dataset Overview
   -- Check the first/last 10 rows of the cleaned dataset.
   -- See the target distribution (No Disease vs Disease) as both a bar chart and a table.

2. Train Models
   -- Before training, the app shows the target distribution.
   -- Adjust the test size slider (percentage of data to keep for testing).
   -- Click "Train Models".

   The app will:
      -- Split the data
      -- Apply SMOTE to balance the training set
      -- Scale all features
      -- Train KNN and SVM
   -- Save the models and scaler in the models/folder
   -- Display a performance comparison table (Accuracy, Precision, Recall, F1, MSE, RMSE)

3. Predict Heart Disease
   -- Enter patient data using the input fields (age, sex, chest pain type, blood pressure, cholesterol, etc.).
   -- Choose the model you want to use (KNN or SVM).
   -- Click "Predict", you will see:
      -- Heart Disease Detected or No Heart Disease
      -- Confidence percentage (probability)

4. Data Visualization
   -- Target Distribution: same table as in dataset overview.
   -- Correlation Matrix: shows how features are correlated with each other and with the target.
   -- Confusion Matrices: side‑by‑side for KNN and SVM (only visible after training).
   -- ROC Curves: with AUC values for both models.
   -- Performance Table & Bar Chart: compare accuracy, precision, recall, and F1‑score.
