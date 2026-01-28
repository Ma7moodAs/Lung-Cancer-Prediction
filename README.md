# Lung Cancer Prediction

This project focuses on **binary classification** for lung cancer prediction using **machine learning techniques** applied to health survey data.

##  Project Overview
The goal of this project is to predict whether a patient is likely to have lung cancer based on survey-based features. Special attention was given to **class imbalance** and **medical evaluation metrics**, where false negatives can be costly.

## Methods & Techniques
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature scaling
- Handling class imbalance using **SMOTE**
- Model training and comparison
- Hyperparameter tuning using **RandomizedSearchCV**

##  Models Used
- Support Vector Classifier (SVC)
- Logistic Regression
- Random Forest
- Naive Bayes
- K-Nearest Neighbors (KNN)

## Evaluation Metrics
- Accuracy
- ROC-AUC Score
- Precision, Recall, F1-score

## Results
- Best Accuracy: **94% (SVC)**
- Best ROC-AUC: **0.98 (Random Forest)**
- Strong recall for the positive class, making the model suitable for healthcare-related use cases.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib / Seaborn

## 📁 Project Structure
