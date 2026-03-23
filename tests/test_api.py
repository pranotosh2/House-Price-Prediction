# -*- coding: utf-8 -*-
"""
tests/test_api.py
Unit tests for the FastAPI House Price Prediction API.
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

# Import app from root main.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "area": 3000,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "parking": 1,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "yes",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "prefarea": "yes",
    "furnishingstatus": "semi-furnished",
}


def test_health():
    """GET /health should return 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_valid():
    """POST /predict with valid payload returns a positive predicted_price."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0
    assert "formatted_price" in data
    assert "₹" in data["formatted_price"]


def test_predict_invalid():
    """POST /predict with missing required fields returns 422 Unprocessable Entity."""
    response = client.post("/predict", json={"area": 3000})
    assert response.status_code == 422
