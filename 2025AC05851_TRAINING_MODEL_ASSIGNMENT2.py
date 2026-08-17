# ==========================================================
# MACHINE LEARNING ASSIGNMENT 2
# DATASET: ADULT INCOME DATASET
# TASK: BINARY CLASSIFICATION
# TARGET: income <=50K or >50K
# ==========================================================


# ==========================================================
# SECTION 1: IMPORT LIBRARIES
# ==========================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ==========================================================
# SECTION 2: CREATE REQUIRED FOLDERS
# ==========================================================

os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ==========================================================
# SECTION 3: DEFINE COLUMN NAMES
# ==========================================================
# adult.csv has no header row.
# Column names are taken from adult_names.csv.

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]


# ==========================================================
# SECTION 4: LOAD DATASET
# ==========================================================

df = pd.read_csv(
    "adult.csv",
    header=None,
    names=columns,
    skipinitialspace=True,
    na_values="?"
)

print("Dataset loaded successfully")
print("Dataset shape before cleaning:", df.shape)
print(df.head())


# ==========================================================
# SECTION 5: BASIC DATA UNDERSTANDING
# ==========================================================

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nTarget class distribution before cleaning:")
print(df["income"].value_counts())


# ==========================================================
# SECTION 6: DATA CLEANING
# ==========================================================

# Remove rows with missing values.
# In Adult dataset, unknown values are represented by '?',
# which were converted to NaN using na_values="?".

df = df.dropna()

# Remove unwanted spaces and dots from income labels if present.
df["income"] = df["income"].astype(str).str.strip()
df["income"] = df["income"].str.replace(".", "", regex=False)

print("\nDataset shape after cleaning:", df.shape)

print("\nTarget class distribution after cleaning:")
print(df["income"].value_counts())


# ==========================================================
# SECTION 7: ENCODE TARGET VARIABLE
# ==========================================================
# <=50K becomes 0
# >50K becomes 1

df["income"] = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

print("\nTarget after encoding:")
print(df["income"].value_counts())


# ==========================================================
# SECTION 8: DEFINE FEATURES AND TARGET
# ==========================================================

X = df.drop("income", axis=1)
y = df["income"]

numeric_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]


# ==========================================================
# SECTION 9: TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ==========================================================
# SECTION 10: SAVE TEST DATA
# ==========================================================
# This file is required for GitHub and Streamlit testing.

test_data = X_test.copy()
test_data["income"] = y_test.values

# Convert numeric target back to readable labels for the app.
test_data["income"] = test_data["income"].map({
    0: "<=50K",
    1: ">50K"
})

test_data.to_csv("test_data.csv", index=False)

print("\ntest_data.csv saved successfully")


# ==========================================================
# SECTION 11: PREPROCESSING PIPELINE
# ==========================================================
# Numeric columns are scaled.
# Categorical columns are converted into numbers using OneHotEncoding.

try:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", one_hot_encoder, categorical_features)
    ]
)


# ==========================================================
# SECTION 12: DEFINE MODELS
# ==========================================================
# Assignment requires:
# 1. Logistic Regression
# 2. Decision Tree
# 3. KNN
# 4. Naive Bayes
# 5. Random Forest

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ==========================================================
# SECTION 13: FUNCTION TO CALCULATE METRICS
# ==========================================================

def calculate_metrics(model_name, y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "ML Model Name": model_name,
        "Accuracy": round(accuracy, 4),
        "AUC": round(auc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    }


# ==========================================================
# SECTION 14: TRAIN ALL MODELS AND EVALUATE
# ==========================================================

results = []

for model_name, classifier in models.items():
    print("\n==================================================")
    print("Training model:", model_name)
    print("==================================================")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    model_metrics = calculate_metrics(
        model_name,
        y_test,
        y_pred,
        y_prob
    )

    results.append(model_metrics)

    print("\nMetrics:")
    print(model_metrics)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save each trained model
    file_name = model_name.lower().replace(" ", "_") + ".joblib"
    model_path = os.path.join("model", file_name)
    joblib.dump(pipeline, model_path)

    print("Saved model:", model_path)


# ==========================================================
# SECTION 15: CREATE COMPARISON TABLE
# ==========================================================

results_df = pd.DataFrame(results)

print("\n==================================================")
print("MODEL COMPARISON TABLE")
print("==================================================")
print(results_df)

results_df.to_csv("outputs/model_comparison.csv", index=False)

print("\nmodel_comparison.csv saved inside outputs folder")


# ==========================================================
# SECTION 16: FIND OVERALL WINNER
# ==========================================================
# Here we select winner based on F1 score.
# You may also choose based on AUC or MCC.

winner_row = results_df.sort_values(by="F1", ascending=False).iloc[0]
winner_model = winner_row["ML Model Name"]

print("\n==================================================")
print("OVERALL WINNER")
print("==================================================")
print("Best model based on F1 Score:", winner_model)


# ==========================================================
# SECTION 17: SAVE MODEL INFORMATION
# ==========================================================

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib"
}

with open("model/model_files.json", "w") as f:
    json.dump(model_files, f, indent=4)

print("\nmodel_files.json saved successfully")


# ==========================================================
# SECTION 18: WRITE SIMPLE OBSERVATIONS
# ==========================================================

observations = []

for _, row in results_df.iterrows():
    model_name = row["ML Model Name"]

    if model_name == winner_model:
        observation = f"{model_name} gave the best performance based on F1 score."
    elif model_name == "Logistic Regression":
        observation = "Logistic Regression is a simple baseline model and works well for binary classification."
    elif model_name == "Decision Tree":
        observation = "Decision Tree is easy to understand but may overfit on training data."
    elif model_name == "KNN":
        observation = "KNN performance depends on distance calculation and may be slower for large datasets."
    elif model_name == "Naive Bayes":
        observation = "Naive Bayes is fast but assumes independence between features."
    elif model_name == "Random Forest":
        observation = "Random Forest combines multiple trees and usually gives stable performance."
    else:
        observation = "Model performance was evaluated using classification metrics."

    observations.append({
        "ML Model Name": model_name,
        "Observation about model performance": observation
    })

observations_df = pd.DataFrame(observations)
observations_df.to_csv("outputs/model_observations.csv", index=False)

print("\nmodel_observations.csv saved inside outputs folder")


# ==========================================================
# SECTION 19: FINAL MESSAGE
# ==========================================================

print("\n==================================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("==================================================")
print("Generated files:")
print("1. test_data.csv")
print("2. outputs/model_comparison.csv")
print("3. outputs/model_observations.csv")
print("4. Saved models inside model folder")
print("5. model/model_files.json")