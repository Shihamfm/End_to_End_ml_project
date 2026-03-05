import sys
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from src.exception import CustomException
from src.logger import logging

app = FastAPI(title="Credit-card-fraud-detection", version="1.0.0")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        logging.info('Accessing the web index.html')
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        raise CustomException (e, sys)


@app.get("/predictdata", response_class=HTMLResponse)
async def predict_form(request: Request):
    try:
        logging.info('Request the features for prediction')
        return templates.TemplateResponse("home.html", {"request": request})
    except Exception as e:
        raise CustomException(e, sys)


@app.post("/predictdata", response_class=HTMLResponse)
async def predict_datapoint(
    request: Request,
    Time: float = Form(...),
    V1: float = Form(...),
    V2: float = Form(...),
    V3: float = Form(...),
    V4: float = Form(...),
    V5: float = Form(...),
    V6: float = Form(...),
    V7: float = Form(...),
    V8: float = Form(...),
    V9: float = Form(...),
    V10: float = Form(...),
    V11: float = Form(...),
    V12: float = Form(...),
    V13: float = Form(...),
    V14: float = Form(...),
    V15: float = Form(...),
    V16: float = Form(...),
    V17: float = Form(...),
    V18: float = Form(...),
    V19: float = Form(...),
    V20: float = Form(...),
    V21: float = Form(...),
    V22: float = Form(...),
    V23: float = Form(...),
    V24: float = Form(...),
    V25: float = Form(...),
    V26: float = Form(...),
    V27: float = Form(...),
    V28: float = Form(...),
    Amount: float = Form(...)
    ):

    data = CustomData(
        Time= Time,
        V1 = V1,
        V2 = V2,
        V3 = V3,
        V4 = V4,
        V5 = V5,
        V6 = V6,
        V7 = V7,
        V8 = V8,
        V9 = V9,
        V10 = V10,
        V11 = V11,
        V12 = V12,
        V13 = V13,
        V14 = V14,
        V15 = V15,
        V16 = V16,
        V17 = V17,
        V18 = V18,
        V19 = V19,
        V20 = V20,
        V21 = V21,
        V22 = V22,
        V23 = V23,
        V24 = V24,
        V25 = V25,
        V26 = V26,
        V27 = V27,
        V28 = V28,
        Amount =  Amount,
        )

    pred_df = data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline = PredictPipeline()
    print("Mid Prediction")
    results = predict_pipeline.predict(pred_df)
    print("After Prediction")

    return templates.TemplateResponse(
        "home.html", {"request": request, "results": results[0]}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)