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
        cleaned_value = str(input_value).replace(',', '').replace(' ', '')
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

# Default values for missing columns
default_values = {
    "Age": 30,
    "DailyRate": 800,
    "DistanceFromHome": 10,
    "Education": 3,
    "HourlyRate": 50,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobSatisfaction": 3,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 2,
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 2,
    "BusinessTravel": 1,
    "MaritalStatus": 1,
    "Gender": 1,
    "Department": 1,
    "EducationField": 1,
    "JobRole": 1,
}

# Prepare input data with all expected columns
input_data = pd.DataFrame({
    "OverTime": [1 if overtime == "Yes" else 0],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "RelationshipSatisfaction": [relationship_satisfaction],
    "MonthlyIncome": [monthly_income],
    "YearsWithCurrManager": [years_with_curr_manager],
    **{col: [default_values[col]] for col in default_values.keys()},
})

# Ensure all numeric columns are of type float64
for col in input_data.columns:
    try:
        input_data[col] = pd.to_numeric(input_data[col], errors="coerce")
    except Exception as e:
        st.error(f"Error converting column {col} to numeric: {e}")
        st.stop()

# Debug: Print data types before preprocessing
st.write("Input DataFrame:", input_data)
st.write("Data Types:", input_data.dtypes)

# Process and Predict Button
if st.button("Predict"):
    st.session_state.reset_prediction = False
    try:
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
