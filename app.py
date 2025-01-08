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

# Define session state to handle reset
if "reset" not in st.session_state:
    st.session_state.reset = False

# User Input Form for Important Features
if st.session_state.reset:
    overtime = 0
    environment_satisfaction = 1
    relationship_satisfaction = 1
    monthly_income = 1000
    years_with_manager = 0
    st.session_state.reset = False
else:
    overtime = st.selectbox("OverTime (Yes=1, No=0)", [0, 1])
    environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4)
    relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4)
    monthly_income = st.text_input("Monthly Income", value="1000")  # Use text input to handle commas
    years_with_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, step=1)

# Add Predict and Reset buttons
col1, col2 = st.columns(2)
with col1:
    predict = st.button("Predict")
with col2:
    reset = st.button("Reset")

# Handle Reset Button
if reset:
    st.session_state.reset = True
    st.experimental_rerun()

# Handle Predict Button
if predict:
    try:
        # Remove commas from MonthlyIncome and convert to float
        monthly_income = float(monthly_income.replace(",", ""))
    except ValueError:
        st.error("Please enter a valid number for Monthly Income.")
        st.stop()

    # Create a DataFrame for input features
    input_features_df = pd.DataFrame([[
        overtime,
        environment_satisfaction,
        relationship_satisfaction,
        monthly_income,
        years_with_manager
    ]], columns=["OverTime", "EnvironmentSatisfaction", "RelationshipSatisfaction", "MonthlyIncome", "YearsWithCurrManager"])

    # Debug: Print the input DataFrame
    st.write("Input DataFrame Before Processing:", input_features_df)

    # Map categorical values to match the preprocessor's expectations
    input_features_df["OverTime"] = input_features_df["OverTime"].map({1: "Yes", 0: "No"}).astype(str)

    # Ensure numeric types for all relevant features
    input_features_df["EnvironmentSatisfaction"] = input_features_df["EnvironmentSatisfaction"].astype(float)
    input_features_df["RelationshipSatisfaction"] = input_features_df["RelationshipSatisfaction"].astype(float)
    input_features_df["MonthlyIncome"] = input_features_df["MonthlyIncome"].astype(float)
    input_features_df["YearsWithCurrManager"] = input_features_df["YearsWithCurrManager"].astype(float)

    # Add missing columns with default values as required by the preprocessor
    expected_columns = [name for transformer in preprocessor.transformers_ for name in transformer[2]]
    for col in expected_columns:
        if col not in input_features_df.columns:
            input_features_df[col] = 0  # Default value for missing columns

    # Debug: Show input after adding missing columns
    st.write("Input DataFrame After Adding Missing Columns:", input_features_df)

    # Preprocess inputs
    try:
        input_processed = preprocessor.transform(input_features_df)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

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
