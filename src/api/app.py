# -*- coding: utf-8 -*-
"""FastAPI сервис скоринга PD-модели: POST /predict принимает JSON с признаками клиента."""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.build_features import add_features

MODEL_PATH = os.path.join("models", "model.joblib")

app = FastAPI(
    title="PD Scoring API",
    description="Скоринговая модель вероятности дефолта клиента",
    version="1.0.0",
)

pipeline = None


class ClientFeatures(BaseModel):
    LIMIT_BAL: float = Field(..., example=200000.0, ge=0)
    SEX: int = Field(..., example=2)
    EDUCATION: int = Field(..., example=2)
    MARRIAGE: int = Field(..., example=1)
    AGE: int = Field(..., example=30)
    PAY_0: int = Field(..., example=0)
    PAY_2: int = Field(..., example=0)
    PAY_3: int = Field(..., example=0)
    PAY_4: int = Field(..., example=0)
    PAY_5: int = Field(..., example=0)
    PAY_6: int = Field(..., example=0)
    BILL_AMT1: float = Field(..., example=10000.0)
    BILL_AMT2: float = Field(..., example=9000.0)
    BILL_AMT3: float = Field(..., example=8000.0)
    BILL_AMT4: float = Field(..., example=7000.0)
    BILL_AMT5: float = Field(..., example=6000.0)
    BILL_AMT6: float = Field(..., example=5000.0)
    PAY_AMT1: float = Field(..., example=1200.0)
    PAY_AMT2: float = Field(..., example=1200.0)
    PAY_AMT3: float = Field(..., example=1200.0)
    PAY_AMT4: float = Field(..., example=1200.0)
    PAY_AMT5: float = Field(..., example=1200.0)
    PAY_AMT6: float = Field(..., example=1200.0)


@app.on_event("startup")
def load_model():
    global pipeline
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        print("Модель загружена:", MODEL_PATH)
    else:
        pipeline = None
        print("Модель не найдена, запустите этап train (dvc repro)")


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict")
def predict(client: ClientFeatures):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Модель не загружена, выполните dvc repro")
    raw = pd.DataFrame([client.model_dump()])
    features = add_features(raw)
    proba_default = float(pipeline.predict_proba(features)[0, 1])
    return {
        "default_class": int(proba_default >= 0.5),
        "probability": round(proba_default, 4),
        "probability_no_default": round(1 - proba_default, 4),
    }
