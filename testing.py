import streamlit as st
import pandas as pd
import joblib

# Load the trained RandomForest model
model = joblib.load('random_forest_model.pkl')

# The feature names from the trained model
training_columns = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'BusinessTravel_Travel_Frequently',
    'BusinessTravel_Travel_Rarely', 'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical',
    'EducationField_Other', 'EducationField_Technical Degree', 'JobRole_Human Resources',
    'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director',
    'JobRole_Research Director', 'JobRole_Research Scientist', 'JobRole_Sales Executive',
    'JobRole_Sales Representative', 'MaritalStatus_Married', 'MaritalStatus_Single'
]

# Streamlit form to get user inputs
st.title("Employee Churn Prediction")

# Input form for the user
input_data = {
    'Age': st.number_input('Age', min_value=18, max_value=100),
    'DailyRate': st.number_input('DailyRate'),
    'DistanceFromHome': st.number_input('DistanceFromHome'),
    'Education': st.number_input('Education'),
    'EnvironmentSatisfaction': st.number_input('EnvironmentSatisfaction'),
    'Gender': st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female'),
    'HourlyRate': st.number_input('HourlyRate'),
    'JobInvolvement': st.number_input('JobInvolvement'),
    'JobLevel': st.number_input('JobLevel'),
    'JobSatisfaction': st.number_input('JobSatisfaction'),
    'MonthlyIncome': st.number_input('MonthlyIncome'),
    'MonthlyRate': st.number_input('MonthlyRate'),
    'NumCompaniesWorked': st.number_input('NumCompaniesWorked'),
    'OverTime': st.selectbox('OverTime', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes'),
    'PercentSalaryHike': st.number_input('PercentSalaryHike'),
    'PerformanceRating': st.number_input('PerformanceRating'),
    'RelationshipSatisfaction': st.number_input('RelationshipSatisfaction'),
    'StockOptionLevel': st.number_input('StockOptionLevel'),
    'TotalWorkingYears': st.number_input('TotalWorkingYears'),
    'TrainingTimesLastYear': st.number_input('TrainingTimesLastYear'),
    'WorkLifeBalance': st.number_input('WorkLifeBalance'),
    'YearsAtCompany': st.number_input('YearsAtCompany'),
    'YearsInCurrentRole': st.number_input('YearsInCurrentRole'),
    'YearsSinceLastPromotion': st.number_input('YearsSinceLastPromotion'),
    'YearsWithCurrManager': st.number_input('YearsWithCurrManager'),
    'BusinessTravel_Travel_Frequently': st.selectbox('BusinessTravel_Travel_Frequently', options=[0, 1]),
    'BusinessTravel_Travel_Rarely': st.selectbox('BusinessTravel_Travel_Rarely', options=[0, 1]),
    'Department': st.selectbox('Department', options=['Research & Development', 'Sales']),
    'EducationField': st.selectbox('EducationField', options=[
        'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'
    ]),
    'JobRole': st.selectbox('JobRole', options=[
        'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director',
        'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'
    ]),
    'MaritalStatus': st.selectbox('Marital Status', options=['Married', 'Single']),
}

# Button to trigger prediction
submit_button = st.button("Predict Employee Churn")

if submit_button:
    # Convert the input data into DataFrame
    input_data_df = pd.DataFrame([input_data])

    # Convert categorical columns to one-hot encoding
    input_data_df = pd.get_dummies(input_data_df, columns=['Department', 'EducationField', 'JobRole', 'MaritalStatus'])

    # Ensure the input data has the same columns as the training data (exclude the 'Attrition' column)
    input_data_df = input_data_df.reindex(columns=training_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_data_df)

    # Display the result in Streamlit
    if prediction == 0:
        st.write("The employee is likely to **stay** with the company (No Churn).")
    else:
        st.write("The employee is likely to **leave** the company (Churn).")
