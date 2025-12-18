🏠 House Price Prediction using Machine Learning
📌 Project Description

This project focuses on predicting house prices using machine learning regression techniques.
It covers the complete end-to-end ML workflow including data analysis, feature engineering, model training, hyperparameter tuning, and deployment using Streamlit.

🚀 Objectives

Analyze housing data and identify key price-driving features

Apply feature engineering and transformations

Train multiple regression models

Optimize performance using hyperparameter tuning

Deploy the final model as a web application

📊 Dataset Overview

Target Variable: price

Total Features: Numerical + Categorical

Dataset Type: Supervised Regression

Important Features

Area of house

Number of bedrooms & bathrooms

Parking availability

Furnishing status

Air conditioning, basement, guestroom, etc.

🔧 Feature Engineering

Binary categorical variables encoded as 0 / 1

Furnishing status encoded numerically

Log transformation applied to:

price

area

Removal of irrelevant and redundant features

🧠 Models Implemented

Linear Regression

Ridge Regression (Regularization)

XGBoost Regressor (Final Model)

⚙️ Hyperparameter Tuning

GridSearchCV was used to find the best parameters for XGBoost.

param_grid = {
  'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
  'learning_rate'   : [0.001, 0.01, 0.1, 1],
  'max_depth'       : [3, 5, 8, 10],
  'alpha'           : [1, 10, 100],
  'n_estimators'    : [10, 50, 100]
}


Evaluation Metrics

RMSE (Root Mean Squared Error)

R² Score

📈 Model Performance

The tuned XGBoost model achieved the best performance in terms of RMSE and R², making it suitable for deployment.

💾 Model Saving

The trained model is saved using pickle:

xgb_house_model.pkl

🌐 Deployment using Streamlit
Run the Application
streamlit run app.py

Application Features

User-friendly input interface

Real-time house price prediction

Model loaded from .pkl file

🏗 Deployment Architecture
User Input → Streamlit UI → Feature Processing → XGBoost Model → Prediction Output