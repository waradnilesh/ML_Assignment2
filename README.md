# ML Assignment 2

## Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models using the Adult Income Dataset. The models are compared using various evaluation metrics including Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The goal is to identify the best-performing model for predicting whether an individual's annual income is greater than $50,000.

---

## Dataset Description

Dataset Name: Adult Income Dataset

Source: Kaggle

Problem Type: Binary Classification

Target Variable: income

Target Classes:
- <=50K
- >50K

Dataset Statistics:

- Original Records: 32,561
- Records After Cleaning: 30,162
- Number of Features: 14
- Numerical Features: 6
- Categorical Features: 8

Description:

The Adult Income Dataset contains demographic, educational, and employment-related information about individuals. The objective is to predict whether a person's annual income exceeds $50,000 based on the available features.

---

## GitHub Repository Link

Add your GitHub Repository Link here.

---

## Models Used

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.8475 | 0.9022 | 0.7354 | 0.6052 | 0.6640 | 0.5711 |
| Decision Tree | 0.8087 | 0.7478 | 0.6134 | 0.6265 | 0.6199 | 0.4922 |
| KNN | 0.8270 | 0.8595 | 0.6664 | 0.6105 | 0.6372 | 0.5248 |
| Naive Bayes | 0.5826 | 0.8018 | 0.3678 | 0.9414 | 0.5290 | 0.3643 |
| Random Forest | 0.8488 | 0.9007 | 0.7287 | 0.6258 | 0.6734 | 0.5786 |

---

## Model Observations

### Logistic Regression

Logistic Regression achieved strong overall performance with high accuracy and the highest AUC score among all models. It provides a good baseline for binary classification problems.

### Decision Tree

Decision Tree produced reasonable results and is easy to interpret. However, its overall performance was lower than Logistic Regression and Random Forest, indicating possible overfitting.

### KNN

KNN achieved moderate performance and generated better results than Decision Tree in terms of overall accuracy. However, it was not the top-performing model for this dataset.

### Naive Bayes

Naive Bayes achieved the highest recall score but suffered from low precision and overall accuracy. The model predicted many positive cases but produced a large number of false positives.

### Random Forest

Random Forest achieved the highest F1 Score and MCC Score while also maintaining excellent accuracy and AUC values. The model demonstrated the best balance between precision and recall.

---

## Overall Winner

### Best Model: Random Forest

Reason:

Random Forest achieved the highest F1 Score (0.6734) and the highest MCC Score (0.5786), making it the most balanced and reliable model for the Adult Income Dataset. It provided strong overall classification performance while maintaining good precision and recall.

---

## Streamlit Application

The Streamlit application provides:

- CSV file upload option
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix display
- Classification report display
- Prediction results preview

Streamlit App Link:

Add your deployed Streamlit URL here.

---

## Project Files

- app.py
- 2025AC05851_TRAINING_MODEL_ASSIGNMENT2.py
- adult.csv
- test_data.csv
- requirements.txt
- README.md

---

## Author

Nilesh Warad

M.Tech (AIML/DSE)

BITS Pilani WILP
