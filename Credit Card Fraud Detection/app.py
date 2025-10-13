import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("best_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Top 8 features for input
top_features = ['V14','V10','V12','V3','V4','V17','V11','V16','V2','V9']

# All features in the model
all_features = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                'V21','V22','V23','V24','V25','V26','V27','V28','Time','Amount']

st.title("Fraud Detection App (Top 8 Features)")

# User input for top features
input_data = {}
for feature in top_features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Fill missing features with 0
for feature in all_features:
    if feature not in input_data:
        input_data[feature] = 0.0

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    fraud_proba = model.predict_proba(input_df)[:, 1][0]
    fraud_percent = fraud_proba * 100

    # Output label with probability
    if fraud_proba >= 0.5:
        st.write(f"This is fraud ({fraud_percent:.2f}%)")
    else:
        st.write(f"Not fraud ({fraud_percent:.2f}%)")









