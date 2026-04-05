# 🏠 House Price Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)

A production-ready **House Price Prediction** web application powered by an **XGBoost** regression model, served via a **FastAPI** REST API with a sleek dark-mode frontend — fully containerised with Docker and automated via GitHub Actions CI/CD.
* The model already deployed on Render : [Model link](https://house-price-prediction-1-kv9d.onrender.com)
---
## ✨ Features

- 🤖 **XGBoost model** with GridSearchCV hyperparameter tuning
- ⚡ **FastAPI** REST backend — `/predict`, `/health`, interactive `/docs`
- 🎨 **Premium dark-mode** HTML/JS frontend (no frameworks, no reload)
- 🐳 **Docker** ready — single command to run
- 🔁 **GitHub Actions** CI/CD — lint, test, build & push to GHCR automatically

---

## 📊 Dataset & Model

| Item | Detail |
|---|---|
| Dataset | `Housing.csv` — 545 samples, 13 features |
| Target | `price` (log-transformed during training) |
| Final Model | XGBoost Regressor (tuned via GridSearchCV) |
| RMSE | 0.2517 |
| R² Score | 0.6718 |

**Key features:** area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus

---

## 🚀 Quick Start

### 1. Conda Environment (recommended)

```bash
conda env create -f environment.yml
conda activate house-price-prediction
uvicorn main:app --reload --port 8000
```

### 2. Pip

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open **http://127.0.0.1:8000** in your browser.  
Interactive API docs → **http://127.0.0.1:8000/docs**

### 3. Docker

```bash
docker build -t house-price-prediction .
docker run -p 8000:8000 house-price-prediction
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | HTML frontend |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict house price |
| `GET` | `/docs` | Swagger UI |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area": 3000, "bedrooms": 3, "bathrooms": 2, "stories": 2,
    "parking": 1, "mainroad": "yes", "guestroom": "no",
    "basement": "yes", "hotwaterheating": "no",
    "airconditioning": "yes", "prefarea": "yes",
    "furnishingstatus": "semi-furnished"
  }'
```

### Example Response

```json
{
  "predicted_price": 5823941.5,
  "formatted_price": "₹ 5,823,942"
}
```

---

## 🧪 Running Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

| Test | What it checks |
|---|---|
| `test_health` | `GET /health` returns 200 |
| `test_predict_valid` | Valid payload returns `predicted_price > 0` |
| `test_predict_invalid` | Missing fields return HTTP 422 |

---

## 🔁 CI/CD Pipeline (GitHub Actions)

```
Push / PR to main
       │
       ▼
  ┌─────────────────────┐
  │  🧪 test job        │
  │  flake8 lint        │
  │  pytest tests/      │
  └─────────┬───────────┘
            │ pass + push to main
            ▼
  ┌─────────────────────────────┐
  │  🐳 docker-build job        │
  │  Build Docker image         │
  │  Push → ghcr.io (latest +   │
  │         sha-<commit>)       │
  └─────────────────────────────┘
```

---

## 📁 Project Structure

```
House-Price-Prediction/
├── .github/workflows/ci-cd.yml   # GitHub Actions CI/CD
├── static/index.html             # Dark-mode HTML/JS frontend
├── tests/
│   └── test_api.py               # Pytest test suite
├── main.py                       # FastAPI app (entry point)
├── inference.py                  # Preprocessing + prediction
├── model.py                      # Model training script
├── environment.yml               # Conda environment
├── requirements.txt              # Pip dependencies
├── Dockerfile                    # Container definition
├── xgb_house_model.pkl           # Trained XGBoost model
├── model_features.pkl            # Feature column order
└── Housing.csv                   # Raw dataset
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost, scikit-learn, pandas, numpy |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Container | Docker |
| CI/CD | GitHub Actions → GHCR |
| Environment | Conda (Python 3.10) |
