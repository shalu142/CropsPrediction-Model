from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path

app = FastAPI(title="Yield Sense API 🌾")

# -------------------------
# LOAD MODELS
# -------------------------

crop_models = {
    "Rice": {
        "yield": joblib.load("rice_yield_model.pkl"),
        "prod": joblib.load("rice_prod_model.pkl")
    },
    "Wheat": {
        "yield": joblib.load("wheat_yield_model.pkl"),
        "prod": joblib.load("wheat_prod_model.pkl")
    },
    "Maize": {
        "yield": joblib.load("maize_yield_model.pkl"),
        "prod": joblib.load("maize_prod_model.pkl")
    },
    "Cotton": {
        "yield": joblib.load("cotton_yield_model.pkl"),
        "prod": joblib.load("cotton_prod_model.pkl")
    },
    "Sugarcane": {
        "yield": joblib.load("sugarcane_yield_model.pkl"),
        "prod": joblib.load("sugarcane_prod_model.pkl")
    }
}

with open("crop_yield_stats.json") as f:
    crop_stats = json.load(f)

# -------------------------
# REQUEST SCHEMAS
# -------------------------

class YieldRequest(BaseModel):
    crop: str
    field_size: float

class RecommendRequest(BaseModel):
    soil_ph: float
    soil_moisture: int
    soil_fertility: int
    crop_code: int
    ndvi: float
    temperature: float
    humidity: float
    has_image: int

# -------------------------
# UTILS
# -------------------------

def build_input(crop, field_size):
    if crop == "Rice":
        return (
            pd.DataFrame({"RICE AREA (1000 ha)": [field_size], "RICE PRODUCTION (1000 tons)": [0]}),
            pd.DataFrame({"RICE AREA (1000 ha)": [field_size], "RICE YIELD (Kg per ha)": [0]})
        )

    if crop == "Wheat":
        return (
            pd.DataFrame({"WHEAT AREA (1000 ha)": [field_size], "WHEAT PRODUCTION (1000 tons)": [0]}),
            pd.DataFrame({"WHEAT AREA (1000 ha)": [field_size], "WHEAT YIELD (Kg per ha)": [0]})
        )

    if crop == "Maize":
        return (
            pd.DataFrame({"MAIZE AREA (1000 ha)": [field_size], "MAIZE PRODUCTION (1000 tons)": [0]}),
            pd.DataFrame({"MAIZE AREA (1000 ha)": [field_size], "MAIZE YIELD (Kg per ha)": [0]})
        )

    if crop == "Cotton":
        return (
            pd.DataFrame({"COTTON AREA (1000 ha)": [field_size], "COTTON PRODUCTION (1000 tons)": [0]}),
            pd.DataFrame({"COTTON AREA (1000 ha)": [field_size], "COTTON YIELD (Kg per ha)": [0]})
        )

    if crop == "Sugarcane":
        return (
            pd.DataFrame({"SUGARCANE AREA (1000 ha)": [field_size], "SUGARCANE PRODUCTION (1000 tons)": [0]}),
            pd.DataFrame({"SUGARCANE AREA (1000 ha)": [field_size], "SUGARCANE YIELD (Kg per ha)": [0]})
        )

# -------------------------
# ROUTES
# -------------------------

@app.get("/")
def home():
    return {"status": "Yield Sense API running 🚀"}

@app.post("/predict-yield")
def predict_yield(req: YieldRequest):

    models = crop_models.get(req.crop)
    if not models:
        return {"error": "Crop not supported"}

    input_yield, input_prod = build_input(req.crop, req.field_size)

    yield_per_ha = models["yield"].predict(input_yield)[0]

    input_prod.iloc[0, -1] = yield_per_ha
    production = models["prod"].predict(input_prod)[0]

    total_yield = yield_per_ha * req.field_size

    return {
        "yield_per_ha": round(yield_per_ha, 2),
        "total_yield": round(total_yield, 2),
        "production": round(production, 2),
        "stats": crop_stats.get(req.crop, {})
    }

@app.post("/recommend")
def recommend(req: RecommendRequest):

    recs = []

    if req.soil_ph < 5.5:
        recs.append("Apply lime to increase soil pH")
    elif req.soil_ph > 7.5:
        recs.append("Apply sulfur to reduce soil alkalinity")

    if req.soil_moisture == 0:
        recs.append("Irrigation needed")
    elif req.soil_moisture == 2:
        recs.append("Drain excess water")

    if req.soil_fertility == 0:
        recs.append("Apply NPK fertilizer")

    if req.ndvi < 0.05:
        recs.append("Low vegetation health — inspect crops")

    if req.crop_code == 0:
        recs.append("Rice: manage water depth properly")

    return {"recommendations": recs}
