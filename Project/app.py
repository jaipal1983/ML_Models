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

# ===============================
# PATH SETUP
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Model Evaluation App",
    layout="centered"
)

st.title("ML Model Evaluation - Hotel Reservations")

# ===============================
# LOAD MODEL BUNDLES
# ===============================
models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
	"Decision Tree": joblib.load(os.path.join(MODEL_DIR,"decision_tree.pkl")),
    "KNN": joblib.load(os.path.join(MODEL_DIR,"knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(MODEL_DIR,"naive_bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(MODEL_DIR,"random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(MODEL_DIR,"xgboost.pkl")),
}

# ===============================
# MODEL SELECTION
# ===============================
st.subheader("Select Model")
model_name = st.selectbox("Choose a model", list(models.keys()))

bundle = models[model_name]
model = bundle["model"]
scaler = bundle.get("scaler", None)
features = bundle.get("features", None)

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

    # -------------------------------
    # VALIDATION
    # -------------------------------
    if "booking_status" not in data.columns:
        st.error("CSV must contain 'booking_status' column")
        st.stop()

    X = data.drop("booking_status", axis=1)
    y_true = data["booking_status"]

    # -------------------------------
    # FEATURE ALIGNMENT (CRITICAL)
    # -------------------------------
    if features is not None:
        missing = set(features) - set(X.columns)
        extra = set(X.columns) - set(features)

        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        X = X[features]  # enforce exact order

    # -------------------------------
    # SCALING (ONLY IF REQUIRED)
    # -------------------------------
    if scaler is not None:
        X = scaler.transform(X)

    # ===============================
    # PREDICTION
    # ===============================
    y_pred = model.predict(X)

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
    st.write(f"**MCC:** {mcc:.3f}")

    if auc is not None:
        st.write(f"**AUC:** {auc:.3f}")

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0 (Not Canceled)", "Predicted 1 (canceled)"]
        )
    )

    # ===============================
    # CLASSIFICATION REPORT
    # ===============================
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
