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

# Reset functionality
if "reset" not in st.session_state:
    st.session_state.reset = False

if st.sidebar.button("Reset Inputs"):
    st.session_state.reset = True
else:
    st.session_state.reset = False

# Helper function to clean and convert numeric inputs
def clean_and_convert_input(input_value):
    try:
        cleaned_value = input_value.replace(',', '').replace(' ', '')
        return float(cleaned_value)
    except ValueError:
        st.error(f"Invalid input: {input_value}. Please enter a valid number.")
        return None

# Inputs
age = st.sidebar.slider(
    "Age", 18, 65, 18 if st.session_state.reset else 30
)
monthly_income_input = st.sidebar.text_input(
    "Monthly Income (e.g., 5000)", value="" if st.session_state.reset else "5000"
)
monthly_income = clean_and_convert_input(monthly_income_input)
monthly_rate_input = st.sidebar.text_input(
    "Monthly Rate (e.g., 15000)", value="" if st.session_state.reset else "15000"
)
monthly_rate = clean_and_convert_input(monthly_rate_input)
overtime = st.sidebar.selectbox(
    "OverTime (Yes/No)", ["Yes", "No"], index=0 if st.session_state.reset else ["Yes", "No"].index("Yes")
)
environment_satisfaction = st.sidebar.slider(
    "Environment Satisfaction (1-4)", 1, 4, 1 if st.session_state.reset else 3
)
relationship_satisfaction = st.sidebar.slider(
    "Relationship Satisfaction (1-4)", 1, 4, 1 if st.session_state.reset else 3
)
percent_salary_hike = st.sidebar.slider(
    "Percent Salary Hike (%)", 0, 50, 0 if st.session_state.reset else 10
)
years_with_curr_manager = st.sidebar.slider(
    "Years with Current Manager", 0, 20, 0 if st.session_state.reset else 5
)
job_involvement = st.sidebar.slider(
    "Job Involvement (1-4)", 1, 4, 1 if st.session_state.reset else 3
)
years_at_company = st.sidebar.slider(
    "Years at Company", 0, 40, 0 if st.session_state.reset else 5
)
job_satisfaction = st.sidebar.slider(
    "Job Satisfaction (1-4)", 1, 4, 1 if st.session_state.reset else 3
)
marital_status = st.sidebar.selectbox(
    "Marital Status", ["Single", "Married", "Divorced"], index=0 if st.session_state.reset else ["Single", "Married", "Divorced"].index("Single")
)
stock_option_level = st.sidebar.slider(
    "Stock Option Level (0-3)", 0, 3, 0 if st.session_state.reset else 0
)
hourly_rate = st.sidebar.number_input(
    "Hourly Rate (e.g., 40)", min_value=10, max_value=100, value=10 if st.session_state.reset else 40
)
daily_rate = st.sidebar.number_input(
    "Daily Rate (e.g., 800)", min_value=100, max_value=2000, value=100 if st.session_state.reset else 800
)
performance_rating = st.sidebar.slider(
    "Performance Rating (1-4)", 1, 4, 1 if st.session_state.reset else 3
)
years_in_current_role = st.sidebar.slider(
    "Years in Current Role", 0, 20, 0 if st.session_state.reset else 5
)
training_times_last_year = st.sidebar.slider(
    "Training Times Last Year", 0, 10, 0 if st.session_state.reset else 3
)
business_travel = st.sidebar.selectbox(
    "Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], index=0 if st.session_state.reset else ["Travel_Rarely", "Travel_Frequently", "Non-Travel"].index("Travel_Rarely")
)
distance_from_home = st.sidebar.number_input(
    "Distance from Home (e.g., 10)", min_value=0, max_value=50, value=0 if st.session_state.reset else 10
)
education_field = st.sidebar.selectbox(
    "Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"], index=0 if st.session_state.reset else ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"].index("Life Sciences")
)
years_since_last_promotion = st.sidebar.slider(
    "Years Since Last Promotion", 0, 20, 0 if st.session_state.reset else 1
)
total_working_years = st.sidebar.slider(
    "Total Working Years", 0, 40, 0 if st.session_state.reset else 10
)
num_companies_worked = st.sidebar.slider(
    "Number of Companies Worked", 0, 20, 0 if st.session_state.reset else 2
)
job_role = st.sidebar.selectbox(
    "Job Role", ["Sales Executive", "Manager", "Research Scientist", "Laboratory Technician", "Other"], index=0 if st.session_state.reset else ["Sales Executive", "Manager", "Research Scientist", "Laboratory Technician", "Other"].index("Sales Executive")
)
job_level = st.sidebar.slider(
    "Job Level (1-5)", 1, 5, 1 if st.session_state.reset else 2
)
work_life_balance = st.sidebar.slider(
    "Work-Life Balance (1-4)", 1, 4, 1 if st.session_state.reset else 3
)
gender = st.sidebar.selectbox(
    "Gender", ["Male", "Female"], index=0 if st.session_state.reset else ["Male", "Female"].index("Male")
)
department = st.sidebar.selectbox(
    "Department", ["Sales", "Research & Development", "Human Resources"], index=0 if st.session_state.reset else ["Sales", "Research & Development", "Human Resources"].index("Sales")
)
education = st.sidebar.slider(
    "Education Level (1-5)", 1, 5, 1 if st.session_state.reset else 3
)

# Process and Predict Button
if st.button("Predict"):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            "Age": [age],
            "MonthlyIncome": [monthly_income],
            "MonthlyRate": [monthly_rate],
            "OverTime": [1 if overtime == "Yes" else 0],
            "EnvironmentSatisfaction": [environment_satisfaction],
            "RelationshipSatisfaction": [relationship_satisfaction],
            "PercentSalaryHike": [percent_salary_hike],
            "YearsWithCurrManager": [years_with_curr_manager],
            "JobInvolvement": [job_involvement],
            "YearsAtCompany": [years_at_company],
            "JobSatisfaction": [job_satisfaction],
            "MaritalStatus": [marital_status],
            "StockOptionLevel": [stock_option_level],
            "HourlyRate": [hourly_rate],
            "DailyRate": [daily_rate],
            "PerformanceRating": [performance_rating],
            "YearsInCurrentRole": [years_in_current_role],
            "TrainingTimesLastYear": [training_times_last_year],
            "BusinessTravel": [business_travel],
            "DistanceFromHome": [distance_from_home],
            "EducationField": [education_field],
            "YearsSinceLastPromotion": [years_since_last_promotion],
            "TotalWorkingYears": [total_working_years],
            "NumCompaniesWorked": [num_companies_worked],
            "JobRole": [job_role],
            "JobLevel": [job_level],
            "WorkLifeBalance": [work_life_balance],
            "Gender": [gender],
            "Department": [department],
            "Education": [education],
        })

        # Preprocess input
        numeric_columns = preprocessor.transformers[0][2]
        input_data[numeric_columns] = input_data[numeric_columns].astype("float64")
        input_array = preprocessor.transform(input_data)

        # Predict using neural network
        nn_predictions = nn_model.predict(input_array).flatten()

        # Hybrid features
        input_hybrid = np.column_stack((input_array, nn_predictions))

        # Predict using hybrid model
        hybrid_predictions = hybrid_model.predict(input_hybrid)

        # Display results
        st.subheader("Prediction Results")
        prediction = "Yes" if hybrid_predictions[0] == 1 else "No"
        st.write(f"Will the employee leave? **{prediction}**")

    except Exception as e:
        st.error(f"Error during processing: {e}")
