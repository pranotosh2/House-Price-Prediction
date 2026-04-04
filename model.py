if __name__ == "__main__":
    # -------------------------
    # 1. Load Dataset
    # -------------------------
    df = pd.read_csv("Housing.csv")

    # -------------------------
    # 2. Encoding
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
    # 4. Split
    # -------------------------
    X = df.drop('log_price', axis=1)
    y = df['log_price']

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # 5. Models
    # -------------------------
    lr = LinearRegression().fit(x_train, y_train)
    ridge = Ridge(alpha=1).fit(x_train, y_train)

    # -------------------------
    # 6. XGBoost Grid Search
    # -------------------------
    param_grid = {
        'colsample_bytree': [0.3, 0.5],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
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
        cv=3,              # ⚡ reduced for faster CI
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    # -------------------------
    # 7. Save
    # -------------------------
    joblib.dump(best_model, "xgb_house_model.pkl")
    joblib.dump(X.columns.tolist(), "model_features.pkl")

    print("Model saved successfully!")
