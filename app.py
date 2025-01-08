if st.button("Predict"):
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

    # Clean numeric data (remove commas, if any)
    input_features_df["MonthlyIncome"] = input_features_df["MonthlyIncome"].replace({',': ''}, regex=True).astype(float)
    input_features_df["EnvironmentSatisfaction"] = input_features_df["EnvironmentSatisfaction"].astype(float)
    input_features_df["RelationshipSatisfaction"] = input_features_df["RelationshipSatisfaction"].astype(float)
    input_features_df["YearsWithCurrManager"] = input_features_df["YearsWithCurrManager"].astype(float)

    # Ensure all expected columns are present in the input DataFrame
    expected_columns = [name for transformer in preprocessor.transformers_ for name in transformer[2]]
    for col in expected_columns:
        if col not in input_features_df.columns:
            input_features_df[col] = 0  # Add missing columns with default values

    # Debug: Check columns after adding missing ones
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
