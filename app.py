import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# 🔧 Setup
predictor = PredictPipeline()
st.set_page_config(page_title="SoulMonitor", page_icon="🧠")

# 🧠 Title + Affirmation
st.title("SoulMonitor — Churn Prediction with Soul 💡")
st.markdown(
    "💬 *“Clarity is power. Systems create freedom. Soul fuels everything.”*")
st.markdown("🔧 Built with love, logic, and low latency by **Amit Chougule**")

# 📊 Inputs
age = st.number_input("Age")
salary = st.number_input("Salary")
tenure = st.number_input("Tenure")
balance = st.number_input("Balance")
num_products = st.number_input("Number of Products")
has_cr_card = st.checkbox("Has Credit Card")
is_active_member = st.checkbox("Is Active Member")
estimated_salary = st.number_input("Estimated Salary")

# 🔍 Prediction
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

# 📘 Sidebar Branding
st.sidebar.markdown("## About SoulMonitor")
st.sidebar.info("""
SoulMonitor is a modular churn prediction app built by Amit Chougule.

🔹 MLOps with soul
🔹 Streamlit UI for clarity
🔹 MIT Licensed & recruiter-ready
""")

# 🖋️ Footer Signature
st.markdown("---")
st.markdown(
    "**Built by Amit Chougule — AI/ML Architect & SoulFuel Cart Founder**")
st.markdown("💡 Remix everything. Deploy with soul. Live with swagger.")
st.markdown(
    "[GitHub](https://github.com/amitchouguleack)
