import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

predictor = PredictPipeline()

st.title("SoulMonitor — Churn Prediction")

age = st.number_input("Age")
salary = st.number_input("Salary")
tenure = st.number_input("Tenure")
balance = st.number_input("Balance")
num_products = st.number_input("Number of Products")
has_cr_card = st.checkbox("Has Credit Card")
is_active_member = st.checkbox("Is Active Member")
estimated_salary = st.number_input("Estimated Salary")

if st.button("Predict"):
    data = CustomData(
        Age=age,
        Salary=salary,
        Tenure=tenure,
        Balance=balance,
        NumOfProducts=num_products,
        HasCrCard=has_cr_card,
        IsActiveMember=is_active_member,
        EstimatedSalary=estimated_salary
    )
    df = data.get_data_as_data_frame()
    prediction = predictor.predict(df)
    st.success("Churn: Yes" if int(prediction.item()) == 1 else "Churn: No")
