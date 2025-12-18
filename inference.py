# -*- coding: utf-8 -*-

# =========================
# inference.py
# =========================

import numpy as np
import pandas as pd
import joblib

# -------------------------
# Load model and features
# -------------------------
model = joblib.load("xgb_house_model.pkl")
model_features = joblib.load("model_features.pkl")

def preprocess_input(input_dict):
    """
    Converts raw user input into model-ready format
    """

    df = pd.DataFrame([input_dict])

    # Binary encoding
    binary_cols = [
        'mainroad', 'guestroom', 'basement',
        'hotwaterheating', 'airconditioning', 'prefarea'
    ]

    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # Furnishing status encoding
    df['furnishingstatus'] = df['furnishingstatus'].map({
        'unfurnished': 0,
        'semi-furnished': 1,
        'furnished': 2
    })

    # Feature engineering
    df['log_area'] = np.log(df['area'])
    df.drop(columns=['area'], inplace=True)

    # Ensure correct column order
    df = df[model_features]

    return df


def predict_price(input_dict):
    processed_data = preprocess_input(input_dict)
    log_price = model.predict(processed_data)[0]
    price = np.exp(log_price)
    return price


# -------------------------
# Test Prediction
# -------------------------
if __name__ == "__main__":
    sample_input = {
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
        "furnishingstatus": "semi-furnished"
    }

    predicted_price = predict_price(sample_input)
    print("Predicted House Price: ₹", round(predicted_price, 2))
