from fastapi import FastAPI
import uvicorn
import pandas as pd
from pydantic import BaseModel
import sys
import os
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException
from src.logger import logging
from src.pipeline.validation_pipeline import CustomDataModel

# 🧠 Initiate SoulMonitor API
app = FastAPI(title="SoulMonitor",
              description="MLOps with soul — drift detection, predictions, and clarity.", version="1.0")
predictor = PredictPipeline()


@app.get("/")
def home():
    logging.info("Received a GET request at / endpoint.")
    return {
        "message": "Welcome to SoulMonitor — MLOps with soul and swagger 💥",
        "source_code": "https://github.com/amthouduglacek/SoulMonitor",
        "affirmation": "You're not just deploying models — you're deploying clarity 🧘"
    }


@app.post("/")
def wrong_method_post():
    logging.warning("POST request sent to GET-only endpoint.")
    return {
        "error": "Oops! This endpoint only accepts GET. Try /predict with POST instead.",
        "affirmation": "Mistakes are just reroutes to clarity ✨"
    }


@app.post("/predict")
async def predict_custom_data(custom_data: CustomDataModel):
    try:
        logging.info("Prediction request received at /predict.")

        # Convert Pydantic model to dict
        custom_data_dict = custom_data.dict()

        # Create CustomData instance
        custom_data_instance = CustomData(**custom_data_dict)

        # Convert to DataFrame
        custom_data_df = custom_data_instance.get_data_as_data_frame()

        # Make prediction
        preds = predictor.predict(custom_data_df)
        prediction_result = int(preds.item())

        response = {
            "Churn Prediction": "Yes" if prediction_result == 1 else "No",
            "affirmation": "Prediction complete — clarity delivered 💡"
        }
        logging.info("Prediction successful.")
        return response

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return {
            "error": str(e),
            "affirmation": "Even errors teach us something 🧘"
        }


@app.get("/predict")
def wrong_method_get():
    logging.warning("GET request sent to POST-only endpoint.")
    return {
        "error": "This endpoint only accepts POST. Try sending data for prediction.",
        "affirmation": "Direction matters — POST is the path to clarity 🚀"
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=4040)
