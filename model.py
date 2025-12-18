# =========================
# main_model.py
# =========================

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge

# -------------------------
# 1. Load Dataset
# -------------------------
df = pd.read_csv("C:/Users/manda/Downloads/house_price_prediction/Housing.csv")

# -------------------------
# 2. Encoding Categorical Variables
# -------------------------
binary_cols = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'prefarea'
]

for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

df['furnishingstatus'] = df['furnishingstatus'].map({
    'unfurnished': 0,
    'semi-furnished': 1,
    'furnished': 2
})

# -------------------------
# 3. Feature Engineering
# -------------------------
df['log_price'] = np.log(df['price'])
df['log_area'] = np.log(df['area'])

df.drop(columns=['price', 'area'], inplace=True)

# -------------------------
# 4. Train-Test Split
# -------------------------
X = df.drop('log_price', axis=1)
y = df['log_price']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 5. Baseline Models
# -------------------------
lr = LinearRegression()
lr.fit(x_train, y_train)

ridge = Ridge(alpha=1)
ridge.fit(x_train, y_train)

# -------------------------
# 6. XGBoost + Grid Search
# -------------------------
param_grid = {
    'colsample_bytree': [0.3, 0.5, 0.7],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 8],
    'alpha': [1, 10],
    'n_estimators': [50, 100]
}

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_

# -------------------------
# 7. Evaluation
# -------------------------
y_pred = best_model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Best XGBoost Model Performance")
print("RMSE:", rmse)
print("R2 Score:", r2)

# -------------------------
# 8. Save Model
# -------------------------
joblib.dump(best_model, "xgb_house_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("Model saved successfully!")
