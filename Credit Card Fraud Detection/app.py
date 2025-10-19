# app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Fraud Detection (Upload CSV, model predicts fraud or not)")

# --- CONFIG: modellər və fayl adları (sənə uyğunlaşdır)
model_files = {
    "Random Forest": "random_forest_optimal_model.pkl",
    "XGBoost": "xgboost_optimal_model.pkl",
    "LightGBM": "lightgbm_optimal_model.pkl"
}
# (optional) scaler faylı əgər train zamanı scaler fit edib saxlamısan
scaler_filename = "scaler.pkl"  # əgər yoxdursa app xəbərdarlıq verəcək

# Model seçimi
model_choice = st.selectbox("Select model:", list(model_files.keys()))

# Try load model package
model_package = None
if os.path.exists(model_files[model_choice]):
    with open(model_files[model_choice], "rb") as f:
        model_package = pickle.load(f)
    model = model_package.get("model")
    model_threshold = model_package.get("threshold", 0.5)
else:
    st.error(f"Model file {model_files[model_choice]} not found. Put it in the app folder.")
    st.stop()

# Try load scaler (optional)
scaler = None
if os.path.exists(scaler_filename):
    try:
        with open(scaler_filename, "rb") as f:
            scaler = pickle.load(f)
        st.info("Scaler loaded and will be applied to input features.")
    except Exception as e:
        st.warning(f"Scaler file found but could not be loaded: {e}")
else:
    st.warning("No scaler.pkl found. Input will be used as-is. If your model expects scaled features, predictions may be wrong.")

st.write(f"Using model `{model_choice}` with default threshold = {model_threshold}. You can override with slider below.")
use_slider = st.checkbox("Override threshold with slider", value=False)
if use_slider:
    threshold = st.slider("Threshold:", 0.0, 1.0, float(model_threshold), 0.01)
else:
    threshold = float(model_threshold)

# Upload CSV
uploaded = st.file_uploader("Upload CSV file with features (columns must match training features).", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to get predictions. CSV may include 'Class' column (optional) for evaluation.")
    st.stop()

df = pd.read_csv(uploaded)
st.write("Input preview (first rows):")
st.dataframe(df.head())

# --- Determine expected feature columns
# Replace this list with exact features you used when training
# For your case earlier: Time, V1..V28, Amount
feature_cols = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

# Check columns
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"The uploaded CSV is missing these required feature columns: {missing}")
    st.stop()

X_input = df[feature_cols].copy()

# Apply scaler if available
if scaler is not None:
    try:
        X_proc = scaler.transform(X_input)
        # If scaler returns numpy array, convert to DataFrame to keep columns
        if isinstance(X_proc, np.ndarray):
            X_proc = pd.DataFrame(X_proc, columns=feature_cols)
    except Exception as e:
        st.error(f"Failed to apply scaler: {e}")
        st.stop()
else:
    X_proc = X_input  # raw features

# Make proba and prediction
try:
    y_prob = model.predict_proba(X_proc)[:, 1]
except Exception as e:
    st.error(f"Model.predict_proba failed: {e}")
    st.stop()

y_pred = (y_prob >= threshold).astype(int)

# Append to df and show
df_result = df.copy()
df_result["Probability"] = y_prob
df_result["Prediction"] = y_pred
st.write("### Predictions")
st.dataframe(df_result.head())

# Download option
csv = df_result.to_csv(index=False).encode('utf-8')
st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime='text/csv')

# If Class present, evaluate
if "Class" in df.columns or "y_true" in df.columns:
    y_true = df.get("Class") if "Class" in df.columns else df["y_true"]
    # ensure ints
    y_true = y_true.astype(int).values
    cm = confusion_matrix(y_true, y_pred)
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.write("### Classification Report")
    cr = classification_report(y_true, y_pred, digits=4, output_dict=True)
    st.json(cr)

    # ROC if possible
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        st.write(f"ROC-AUC: {roc_auc:.4f}")
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax2.plot([0,1],[0,1], 'k--')
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Could not compute ROC: {e}")
else:
    st.info("No label (Class/y_true) column found in the uploaded CSV — showing predictions only.")
