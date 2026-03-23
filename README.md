#  Codveda Machine Learning Internship

This repository contains my complete work during the **Codveda Machine Learning Internship**, covering **data preprocessing, regression, classification, and advanced machine learning models**.

---

#  Internship Overview

During this internship, I implemented multiple machine learning algorithms across different levels:

- Level 1: Basic ML concepts
- Level 2: Intermediate ML models
- Level 3: Advanced ML techniques

Each level builds upon the previous one, forming a complete machine learning pipeline.

---

# 🔹 Level 1 (Basic)

## 📊 Task 1: Data Preprocessing
- Handled missing values using mean imputation
- Standardized numerical features using StandardScaler
- Prepared dataset for machine learning
- Split dataset into training and testing sets

---

## 📈 Task 2: Linear Regression
- Built a regression model to predict house prices
- Evaluated model using:
  - Mean Squared Error (MSE)
  - R² Score (~0.66)
- Interpreted feature importance using coefficients

---

# 🔹 Level 2 (Intermediate)

## 📉 Task 1: Logistic Regression (Churn Prediction)
- Built a binary classification model
- Performed data preprocessing and encoding
- Handled class imbalance using `class_weight='balanced'`
- Improved recall from **0.23 → 0.71**
- Evaluated using:
  - Accuracy (~0.76)
  - Precision & Recall
  - Confusion Matrix
  - ROC Curve (AUC ≈ 0.78)

---

## 🌳 Task 2: Decision Tree
- Built a classification model using the Iris dataset
- Visualized the decision tree
- Achieved **100% accuracy**
- Understood feature-based decision rules

---

# 🔹 Level 3 (Advanced)

## 🌲 Task 1: Random Forest
- Built an ensemble model using multiple decision trees
- Improved model stability and generalization
- Performed feature importance analysis:

| Feature         | Importance |
|----------------|-----------|
| petal_length   | 0.44      |
| petal_width    | 0.42      |
| sepal_length   | 0.10      |
| sepal_width    | 0.02      |

---

## ⚡ Task 2: Support Vector Machine (SVM)
- Trained SVM using:
  - Linear Kernel
  - RBF Kernel
- Visualized decision boundaries
- Compared model performance across kernels
- Achieved high classification accuracy

---

# 🛠️ Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

# 📊 Key Concepts Learned

- Data preprocessing and feature scaling
- Regression vs Classification models
- Handling class imbalance
- Model evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Decision Trees and ensemble learning
- Feature importance analysis
- Support Vector Machines and kernels
- Visualization of ML models

---

# 📁 Project Structure
ml-task1-preprocessing/   
│   
├── house.csv   
├── iris.csv   
├── churn-bigml-80.csv   
├── churn-bigml-20.csv   
├── task1_preprocessing.py   
├── task2_linear_regression.py   
├── task4_logistic_regression.py   
├── task5_decision_tree.py   
├── task6_random_forest.py   
├── task7_svm.py   
├── README.md   
└── requirements.txt  


---

# 🎯 Key Achievements

- Built end-to-end machine learning pipeline
- Implemented multiple ML algorithms from scratch
- Improved model performance using real-world techniques
- Visualized models and decision boundaries
- Developed strong understanding of ML fundamentals and advanced concepts

---
