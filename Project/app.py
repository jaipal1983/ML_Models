import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

st.write("Current working directory:", os.getcwd())
st.write("Files in model folder:", os.listdir("model"))
# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Model Evaluation App",
    layout="centered"
)

st.title("ML Model Evaluation – Streamlit App")

# ===============================
# LOAD MODELS
# ===============================
models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR,"logistic_regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_DIR,"decision_tree.pkl")),
    "KNN": joblib.load(os.path.join(MODEL_DIR,"knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(MODEL_DIR,"naive_bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(MODEL_DIR,"random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(MODEL_DIR,"xgboost.pkl")),
}

scaler = joblib.load(os.path.join(MODEL_DIR,"scaler.pkl"))

# ===============================
# MODEL SELECTION
# ===============================
st.subheader( "Select Model")
model_name = st.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

# ===============================
# DATASET UPLOAD (TEST DATA ONLY)
# ===============================
st.subheader("Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader(
    "Upload CSV file (must contain booking_status)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "booking_status" not in data.columns:
        st.error("CSV must contain 'booking_status' column")
        st.stop()

    X = data.drop("booking_status", axis=1)
    y_true = data["booking_status"]

    # Apply scaling only where needed
    if model_name in ["Logistic Regression", "KNN"]:
        X = scaler.transform(X)

    # ===============================
    # PREDICTION
    # ===============================
    y_pred = model.predict(X)

    # Some models support predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = None

    # ===============================
    # METRICS DISPLAY
    # ===============================
    st.subheader("Evaluation Metrics")

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    st.write(f"**Accuracy:** {acc*100:.2f}%")
    st.write(f"**MCC:** {mcc:.2f}")

    if auc is not None:
        st.write(f"**AUC:** {auc:.2f}")

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.write(cm)

    # ===============================
    # CLASSIFICATION REPORT
    # ===============================
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
