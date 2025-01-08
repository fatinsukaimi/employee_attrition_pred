import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load the pre-trained models and preprocessor
try:
    hybrid_model = joblib.load("hybrid_model.pkl")
    nn_model = load_model("nn_model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
except Exception as e:
    st.error(f"Error loading models or preprocessor: {e}")

# Title and Description
st.title("Employee Attrition Prediction")
st.write("This application predicts employee attrition using a hybrid Neural Network and XGBoost model.")

# Input Form
overtime = st.selectbox("OverTime (Yes=1, No=0)", [0, 1])
environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4)
relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4)
monthly_income = st.text_input("Monthly Income (e.g., 1000)", "1000")
years_with_manager = st.slider("Years With Current Manager", 0, 20, 0)

# Predict and Reset Buttons Side-by-Side
col1, col2 = st.columns(2)

def clean_input(value):
    """Helper function to clean and convert numeric inputs."""
    try:
        return float(value.replace(",", "").strip())
    except ValueError:
        return None

if col1.button("Predict"):
    try:
        # Clean inputs
        monthly_income_cleaned = clean_input(monthly_income)

        if monthly_income_cleaned is None:
            st.error("Invalid Monthly Income. Please enter a valid number.")
            st.stop()

        # Combine inputs into a DataFrame
        input_features = pd.DataFrame([[
            overtime,
            environment_satisfaction,
            relationship_satisfaction,
            monthly_income_cleaned,
            years_with_manager
        ]], columns=[
            "OverTime", "EnvironmentSatisfaction", "RelationshipSatisfaction",
            "MonthlyIncome", "YearsWithCurrManager"
        ])

        # Debug: Display input DataFrame before processing
        st.write("### Input DataFrame (Before Processing):")
        st.write(input_features)

        # Ensure column alignment with the preprocessor
        expected_columns = [name for transformer in preprocessor.transformers_ for name in transformer[2]]
        for col in expected_columns:
            if col not in input_features.columns:
                input_features[col] = 0  # Fill missing columns with default values

        # Convert all columns to numeric to avoid dtype issues
        input_features = input_features.apply(pd.to_numeric, errors="coerce")

        # Debug: Display input DataFrame after adding missing columns
        st.write("### Input DataFrame (After Processing):")
        st.write(input_features)

        # Check for NaN or invalid data
        if input_features.isnull().any().any():
            st.error("Input contains NaN or invalid values after processing. Please check your inputs.")
            st.stop()

        # Preprocess inputs
        input_processed = preprocessor.transform(input_features)

        # Debug: Display preprocessed inputs
        st.write("### Preprocessed Input:")
        st.write(input_processed)

        # NN Predictions
        nn_preds = nn_model.predict(input_processed)

        # Debug: Display NN predictions
        st.write("### Neural Network Predictions:")
        st.write(nn_preds)

        # Combine NN predictions for hybrid model
        hybrid_input = np.column_stack((input_processed, nn_preds))

        # Debug: Display hybrid input
        st.write("### Hybrid Model Input:")
        st.write(hybrid_input)

        # Hybrid model predictions
        prediction = hybrid_model.predict(hybrid_input)
        attrition_probability = hybrid_model.predict_proba(hybrid_input)[:, 1]

        # Display Results
        st.write("### Prediction Result")
        st.write(f"Will the employee leave the company? {'Yes' if prediction[0] == 1 else 'No'}")
        st.write(f"Probability of Attrition: {attrition_probability[0]:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

if col2.button("Reset"):
    st.experimental_rerun()
