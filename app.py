# -*- coding: utf-8 -*-

# =========================
# app.py
# =========================

import streamlit as st
from inference import predict_price

st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

st.title("🏠 House Price Prediction App")
st.write("Enter house details to estimate the price")

# -------------------------
# User Inputs
# -------------------------
area = st.number_input("Area (sq ft)", min_value=300, max_value=10000, step=100)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4])
stories = st.selectbox("Stories", [1, 2, 3, 4])
parking = st.selectbox("Parking Spaces", [0, 1, 2, 3])

mainroad = st.selectbox("Main Road Access", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["unfurnished", "semi-furnished", "furnished"]
)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Price"):
    user_input = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }

    price = predict_price(user_input)

    st.success(f"💰 Estimated House Price: ₹ {price:,.0f}")
