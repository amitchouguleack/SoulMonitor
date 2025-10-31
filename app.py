import gradio as gr
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

predictor = PredictPipeline()


def predict_churn(age, salary, tenure, balance, num_products, has_cr_card, is_active_member, estimated_salary):
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
    return "Yes" if int(prediction.item()) == 1 else "No"


demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Salary"),
        gr.Number(label="Tenure"),
        gr.Number(label="Balance"),
        gr.Number(label="Number of Products"),
        gr.Checkbox(label="Has Credit Card"),
        gr.Checkbox(label="Is Active Member"),
        gr.Number(label="Estimated Salary")
    ],
    outputs=gr.Text(label="Churn Prediction"),
    title="SoulMonitor",
    description="MLOps with soul — predict churn with affirmation-powered clarity 💡"
)

demo.launch()
