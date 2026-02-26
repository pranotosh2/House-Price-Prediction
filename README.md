## 🏠 House Price Prediction using Machine Learning
📌 Project Description

This project focuses on predicting house prices using machine learning regression techniques.
It covers the complete end-to-end ML workflow including data analysis, feature engineering, model training, hyperparameter tuning, and deploy on DockerHub.

# 🚀 Objectives

* Analyze housing data and identify key price-driving features

* Apply feature engineering and transformations

* Train multiple regression models

* Optimize performance using hyperparameter tuning

* Deploy the final model using Docker and DockerHub

# 📊 Dataset Overview

Target Variable: price
Total Features: Numerical + Categorical
Dataset Type: Supervised Regression
Important Features
Area of house
Number of bedrooms & bathrooms
Parking availability
Furnishing status
Air conditioning, basement, guestroom, etc.

# 🔧 Feature Engineering

Binary categorical variables encoded as 0 / 1
Furnishing status encoded numerically
Log transformation applied to: price and area
Removal of irrelevant and redundant features

# 🧠 Models Implemented

Linear Regression
Ridge Regression (Regularization)
XGBoost Regressor (Final Model)

# ⚙️ Hyperparameter Tuning

GridSearchCV was used to find the best parameters for XGBoost.

```python
param_grid = {
  'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
  'learning_rate'   : [0.001, 0.01, 0.1, 1],
  'max_depth'       : [3, 5, 8, 10],
  'alpha'           : [1, 10, 100],
  'n_estimators'    : [10, 50, 100]
}
```
# Evaluation Metrics

* RMSE: 0.2517315686480639
* R2 Score: 0.6718152233407819

# 📈 Model Performance

The tuned XGBoost model achieved the best performance in terms of RMSE and R², making it suitable for deployment.

# Run Using Docker Hub
* The prebuilt Docker image is available on Docker Hub: [Docker Hub Image](https://hub.docker.com/r/pranotosh/house-price-prediction)
* Pull the image : `docker pull pranotosh/house-price-prediction`
* Run the container : `docker run -p 8501:8501 pranotosh/house-price-prediction`
* Open in Browse : `http://localhost:8501`

