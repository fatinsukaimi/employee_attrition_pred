import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models and preprocessor
nn_model = load_model("nn_model.keras")
hybrid_model = joblib.load("hybrid_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Title
st.title("Employee Attrition Prediction")

# Sidebar Inputs
st.sidebar.header("Employee Features")

# Reset Prediction Button
if "reset_prediction" not in st.session_state:
    st.session_state.reset_prediction = False

if st.sidebar.button("Reset Prediction"):
    st.session_state.reset_prediction = True
else:
    st.session_state.reset_prediction = False

# Helper function to clean and convert numeric inputs
def clean_and_convert_input(input_value):
    try:
        cleaned_value = input_value.replace(',', '').replace(' ', '')
        return float(cleaned_value)
    except ValueError:
        st.error(f"Invalid input: {input_value}. Please enter a valid number.")
        return None

# Inputs
overtime = st.sidebar.selectbox("OverTime (Yes/No)", ["Yes", "No"])
environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)
relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
monthly_income_input = st.sidebar.text_input("Monthly Income (e.g., 5000)", value="5000")
monthly_income = clean_and_convert_input(monthly_income_input)
years_with_curr_manager = st.sidebar.slider("Years with Current Manager", 0, 20, 5)

# Process and Predict Button
if st.button("Predict"):
    st.session_state.reset_prediction = False
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            "OverTime": [1 if overtime == "Yes" else 0],
            "EnvironmentSatisfaction": [environment_satisfaction],
            "RelationshipSatisfaction": [relationship_satisfaction],
            "MonthlyIncome": [monthly_income],
            "YearsWithCurrManager": [years_with_curr_manager],
        })

        # Ensure numeric and categorical types
        numeric_columns = preprocessor.transformers[0][2]
        input_data[numeric_columns] = input_data[numeric_columns].astype('float64')

        categorical_columns = preprocessor.transformers[1][2]
        input_data[categorical_columns] = input_data[categorical_columns].astype(str)

        # Preprocess
        input_array = preprocessor.transform(input_data)

        # Predict using Neural Network
        nn_predictions = nn_model.predict(input_array).flatten()

        # Create hybrid features
        input_hybrid = np.column_stack((input_array, nn_predictions))

        # Predict using Hybrid NN-XGBoost
        hybrid_predictions = hybrid_model.predict(input_hybrid)

        # Display predictions
        st.subheader("Prediction Results")
        if not st.session_state.reset_prediction:
            prediction = "Yes" if hybrid_predictions[0] == 1 else "No"
            st.write(f"Will the employee leave? **{prediction}**")

    except Exception as e:
        st.error(f"Error during processing: {e}")
