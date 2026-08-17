# ==========================================================
# STREAMLIT APP FOR MACHINE LEARNING ASSIGNMENT 2
# DATASET: ADULT INCOME DATASET
# ==========================================================


# ==========================================================
# SECTION 1: IMPORT LIBRARIES
# ==========================================================

import json
import joblib
import pandas as pd
import streamlit as st

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
# SECTION 2: PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Adult Income Classification App",
    layout="wide"
)

st.title("Adult Income Classification App")
st.write("This app predicts whether income is <=50K or >50K using different ML classification models.")


# ==========================================================
# SECTION 3: COLUMN NAMES
# ==========================================================

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
# SECTION 4: LOAD MODEL FILE INFORMATION
# ==========================================================

with open("model/model_files.json", "r") as f:
    model_files = json.load(f)


# ==========================================================
# SECTION 5: FILE UPLOAD OPTION
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Try reading uploaded file normally.
    data = pd.read_csv(uploaded_file)

    # If uploaded file has no proper column names, assign Adult dataset column names.
    if "income" not in data.columns:
        uploaded_file.seek(0)
        data = pd.read_csv(
            uploaded_file,
            header=None,
            names=columns,
            skipinitialspace=True,
            na_values="?"
        )

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    st.write("Dataset shape:", data.shape)


    # ======================================================
    # SECTION 6: CLEAN UPLOADED DATA
    # ======================================================

    data = data.dropna()

    data["income"] = data["income"].astype(str).str.strip()
    data["income"] = data["income"].str.replace(".", "", regex=False)

    data["income"] = data["income"].map({
        "<=50K": 0,
        ">50K": 1
    })

    data = data.dropna()

    X = data.drop("income", axis=1)
    y = data["income"]


    # ======================================================
    # SECTION 7: MODEL SELECTION DROPDOWN
    # ======================================================

    selected_model = st.selectbox(
        "Select ML Model",
        list(model_files.keys())
    )

    model_path = model_files[selected_model]
    model = joblib.load(model_path)


    # ======================================================
    # SECTION 8: PREDICTION
    # ======================================================

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]


    # ======================================================
    # SECTION 9: DISPLAY EVALUATION METRICS
    # ======================================================

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy, 4))
    col1.metric("AUC", round(auc, 4))

    col2.metric("Precision", round(precision, 4))
    col2.metric("Recall", round(recall, 4))

    col3.metric("F1 Score", round(f1, 4))
    col3.metric("MCC Score", round(mcc, 4))


    # ======================================================
    # SECTION 10: CONFUSION MATRIX
    # ======================================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual <=50K", "Actual >50K"],
        columns=["Predicted <=50K", "Predicted >50K"]
    )

    st.dataframe(cm_df)


    # ======================================================
    # SECTION 11: CLASSIFICATION REPORT
    # ======================================================

    st.subheader("Classification Report")

    report = classification_report(
        y,
        y_pred,
        target_names=["<=50K", ">50K"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)


    # ======================================================
    # SECTION 12: SHOW PREDICTIONS
    # ======================================================

    st.subheader("Prediction Results")

    output_data = X.copy()
    output_data["Actual Income"] = y.map({
        0: "<=50K",
        1: ">50K"
    })

    output_data["Predicted Income"] = pd.Series(y_pred).map({
        0: "<=50K",
        1: ">50K"
    })

    st.dataframe(output_data.head(20))

else:
    st.info("Please upload test_data.csv to begin.")