# -*- coding: utf-8 -*-

# =========================
# main.py  (FastAPI)
# =========================

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
import os

from inference import predict_price

# -------------------------
# App Setup
# -------------------------
app = FastAPI(
    title="House Price Prediction API",
    description="Predicts house prices using an XGBoost model.",
    version="1.0.0",
)

# Mount static files (frontend)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# -------------------------
# Request / Response Schemas
# -------------------------
class HouseInput(BaseModel):
    area: int = Field(..., ge=300, le=10000, example=3000, description="Area in sq ft")
    bedrooms: int = Field(..., ge=1, le=5, example=3)
    bathrooms: int = Field(..., ge=1, le=4, example=2)
    stories: int = Field(..., ge=1, le=4, example=2)
    parking: int = Field(..., ge=0, le=3, example=1)
    mainroad: Literal["yes", "no"] = Field(..., example="yes")
    guestroom: Literal["yes", "no"] = Field(..., example="no")
    basement: Literal["yes", "no"] = Field(..., example="yes")
    hotwaterheating: Literal["yes", "no"] = Field(..., example="no")
    airconditioning: Literal["yes", "no"] = Field(..., example="yes")
    prefarea: Literal["yes", "no"] = Field(..., example="yes")
    furnishingstatus: Literal["unfurnished", "semi-furnished", "furnished"] = Field(
        ..., example="semi-furnished"
    )


class PredictionResponse(BaseModel):
    predicted_price: float
    formatted_price: str


# -------------------------
# Endpoints
# -------------------------
@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the HTML frontend."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "House Price Prediction API"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(house: HouseInput):
    """
    Predict the price of a house given its features.

    Returns the predicted price in INR.
    """
    input_dict = house.model_dump()
    price = predict_price(input_dict)
    return PredictionResponse(
        predicted_price=round(float(price), 2),
        formatted_price=f"₹ {price:,.0f}",
    )
