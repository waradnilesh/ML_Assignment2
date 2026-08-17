# ==========================================================
# STREAMLIT APP FOR MACHINE LEARNING ASSIGNMENT 2
# DATASET: ADULT INCOME DATASET
# APPROACH: NO JOBLIB FILES, MODELS TRAIN INSIDE STREAMLIT
# ==========================================================

# ==========================================================
# SECTION 1: IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import streamlit as st

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
# SECTION 2: PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Adult Income Classification App",
    layout="wide"
)

st.title("Adult Income Classification App")
st.write(
    "This Streamlit app trains multiple machine learning classification models "
    "on the Adult Income dataset and evaluates the selected model on uploaded test data."
)


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
# SECTION 4: FUNCTION TO CLEAN DATA
# ==========================================================

def clean_adult_data(data):
    data = data.copy()

    data = data.dropna()

    data["income"] = data["income"].astype(str).str.strip()
    data["income"] = data["income"].str.replace(".", "", regex=False)

    data["income"] = data["income"].map({
        "<=50K": 0,
        ">50K": 1,
        "0": 0,
        "1": 1
    })

    data = data.dropna()

    return data


# ==========================================================
# SECTION 5: FUNCTION TO LOAD TRAINING DATA
# ==========================================================

def load_training_data():
    df = pd.read_csv(
        "adult.csv",
        header=None,
        names=columns,
        skipinitialspace=True,
        na_values="?"
    )

    df = clean_adult_data(df)

    X = df.drop("income", axis=1)
    y = df["income"]

    return X, y


# ==========================================================
# SECTION 6: FUNCTION TO CREATE PREPROCESSOR
# ==========================================================

def create_preprocessor():
    try:
        one_hot_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        one_hot_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", one_hot_encoder, categorical_features)
        ]
    )

    return preprocessor


# ==========================================================
# SECTION 7: TRAIN MODELS INSIDE STREAMLIT
# ==========================================================

@st.cache_resource
def train_all_models():
    X, y = load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    trained_models = {}

    for model_name, classifier in models.items():

        preprocessor = create_preprocessor()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        pipeline.fit(X_train, y_train)

        trained_models[model_name] = pipeline

    return trained_models


# ==========================================================
# SECTION 8: TRAIN MODELS WHEN APP STARTS
# ==========================================================

with st.spinner("Training models from adult.csv. Please wait..."):
    trained_models = train_all_models()

st.success("Models trained successfully inside Streamlit app.")


# ==========================================================
# SECTION 9: FILE UPLOAD OPTION
# ==========================================================

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_data = pd.read_csv(uploaded_file)

    if "income" not in uploaded_data.columns:
        uploaded_file.seek(0)
        uploaded_data = pd.read_csv(
            uploaded_file,
            header=None,
            names=columns,
            skipinitialspace=True,
            na_values="?"
        )

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(uploaded_data.head())

    st.write("Dataset shape:", uploaded_data.shape)

    test_data = clean_adult_data(uploaded_data)

    X_test = test_data.drop("income", axis=1)
    y_test = test_data["income"]


    # ======================================================
    # SECTION 10: MODEL SELECTION DROPDOWN
    # ======================================================

    st.subheader("Select Machine Learning Model")

    selected_model_name = st.selectbox(
        "Choose a model",
        list(trained_models.keys())
    )

    selected_model = trained_models[selected_model_name]


    # ======================================================
    # SECTION 11: PREDICTION
    # ======================================================

    y_pred = selected_model.predict(X_test)

    if hasattr(selected_model, "predict_proba"):
        y_prob = selected_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred


    # ======================================================
    # SECTION 12: EVALUATION METRICS
    # ======================================================

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy, 4))
    col1.metric("AUC Score", round(auc, 4))

    col2.metric("Precision", round(precision, 4))
    col2.metric("Recall", round(recall, 4))

    col3.metric("F1 Score", round(f1, 4))
    col3.metric("MCC Score", round(mcc, 4))


    # ======================================================
    # SECTION 13: CONFUSION MATRIX
    # ======================================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual <=50K", "Actual >50K"],
        columns=["Predicted <=50K", "Predicted >50K"]
    )

    st.dataframe(cm_df)


    # ======================================================
    # SECTION 14: CLASSIFICATION REPORT
    # ======================================================

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["<=50K", ">50K"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)


    # ======================================================
    # SECTION 15: PREDICTION RESULTS
    # ======================================================

    st.subheader("Prediction Results")

    output_data = X_test.copy()

    output_data["Actual Income"] = y_test.map({
        0: "<=50K",
        1: ">50K"
    })

    output_data["Predicted Income"] = pd.Series(y_pred, index=X_test.index).map({
        0: "<=50K",
        1: ">50K"
    })

    st.dataframe(output_data.head(20))


else:
    st.info("Please upload test_data.csv to evaluate the models.")