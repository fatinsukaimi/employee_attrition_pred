import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load the pre-trained models and preprocessor
hybrid_model = joblib.load("hybrid_model.pkl")
nn_model = load_model("nn_model.keras")
preprocessor = joblib.load("preprocessor.pkl")

# Title and Description
st.title("Employee Attrition Prediction")
st.write("This application predicts employee attrition using a hybrid Neural Network and XGBoost model.")

# User Input Form for Important Features
overtime = st.selectbox("OverTime (Yes=1, No=0)", [0, 1])
environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4)
relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, step=100)
years_with_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, step=1)

# Predict Button
if st.button("Predict"):
    # Combine inputs into an array
    input_features = np.array([[
        overtime, environment_satisfaction, relationship_satisfaction,
        monthly_income, years_with_manager
    ]])

    # Preprocess inputs
    input_processed = preprocessor.transform(input_features)

    # NN Predictions
    nn_preds = nn_model.predict(input_processed)

    # Combine NN predictions for hybrid model
    hybrid_input = np.column_stack((input_processed, nn_preds))

    # Hybrid model predictions
    prediction = hybrid_model.predict(hybrid_input)
    attrition_probability = hybrid_model.predict_proba(hybrid_input)[:, 1]

    # Display Results
    st.write("### Prediction Result")
    st.write(f"Will the employee leave the company? {'Yes' if prediction[0] == 1 else 'No'}")
    st.write(f"Probability of Attrition: {attrition_probability[0]:.2f}")
