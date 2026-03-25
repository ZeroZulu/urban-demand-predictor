# UrbanPulse — Weather-Driven Demand Intelligence Engine

> Predicts NYC taxi demand by zone and hour using weather, civic events,
> and economic signals — served as a live REST API with per-prediction SHAP explanations.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?logo=mlflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?logo=xgboost&logoColor=white)
![CI](https://github.com/ZeroZulu/urban-demand-predictor/actions/workflows/ci.yml/badge.svg)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://urban-demand-predictor-lq2ja9lznlgwe7klggp8md.streamlit.app)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI-009688?logo=fastapi&logoColor=white)](https://urban-demand-predictor.onrender.com/docs)

> 🔮 **[Try the live dashboard →](https://urban-demand-predictor-lq2ja9lznlgwe7klggp8md.streamlit.app)**  
> ⚡ **[Explore the API →](https://urban-demand-predictor.onrender.com/docs)**

---

## Key Findings

- **History beats weather**: The strongest predictor is `demand_lag_168h` (last week's demand
  at the same hour) — SHAP contribution ~3× larger than temperature. The model learns
  persistent weekly rhythms that no weather signal can override.

- **Rain increases demand ~11%**: Precipitation drives SHAP values up across all zones,
  consistent with riders avoiding walking. Effect is most pronounced for trips < 1 mile.

- **Major events create zone-specific spikes of 40–80%**: MSG events lift Midtown zones
  2–3 hours before tip-off. The effect is nearly invisible in outer-borough zones.

- **XGBoost (RMSE 17.2) is the best model** — LightGBM (18.1) and Random Forest (22.4)
  trail behind. A deep learning model (LSTM) was benchmarked separately and did not
  outperform gradient boosting on this tabular problem, as expected for structured
  time-series with engineered lag features.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
│  NYC TLC Parquet   Open-Meteo API   NYC Events CSV   FRED API   │
└──────────┬──────────────┬──────────────┬──────────────┬─────────┘
           │              │              │              │
           └──────────────┴──────────────┴──────────────┘
                                   │
                          ┌────────▼────────┐
                          │   PostgreSQL 15  │
                          │  (Docker volume) │
                          └────────┬─────────┘
                                   │ Materialized View
                          ┌────────▼────────┐
                          │  Feature Builder │
                          │  (Python / SQL)  │
                          └────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
             ┌──────▼──┐   ┌──────▼──┐   ┌──────▼──┐
             │   RF    │   │ XGBoost │   │  LGBM   │
             └─────────┘   └────┬────┘   └─────────┘
                                │ best model
                    ┌───────────┼────────────┐
                    │           │            │
             ┌──────▼────┐  ┌───▼────┐  ┌───▼──────────┐
             │ FastAPI   │  │ MLflow │  │   Evidently   │
             │ /predict  │  │  UI    │  │ Drift Monitor │
             └──────┬────┘  └────────┘  └──────────────┘
                    │
             ┌──────▼──────┐
             │  Streamlit  │
             │  Dashboard  │
             └─────────────┘
```

---

## Quick Start

**Prerequisites**: Docker Desktop, Python 3.11, `make`

```bash
# 1. Clone and configure
git clone https://github.com/ZeroZulu/urban-demand-predictor.git
cd urban-demand-predictor
cp .env.example .env
# Add your FRED API key to .env (free at fred.stlouisfed.org)

# 2. Start all services (Postgres + MLflow + API + Dashboard)
make setup

# 3. Download data and load into Postgres (~20 min, 6 months of taxi data)
make ingest

# 4. Train models and log to MLflow (~15 min with Optuna)
make train

# 5. Open interfaces
open http://localhost:8501   # Streamlit dashboard
open http://localhost:8000/docs  # FastAPI Swagger UI
open http://localhost:5000   # MLflow experiment tracker
```

---

## API Usage

Once running, the API is fully documented at `http://localhost:8000/docs`.

**Single prediction with SHAP explanation:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "zone_id": 161,
    "datetime": "2024-06-15T18:00:00",
    "temperature_f": 78.5,
    "feels_like_f": 80.0,
    "precipitation_mm": 0.0,
    "windspeed_mph": 8.2,
    "humidity_pct": 65.0,
    "weather_code": 1,
    "has_major_event": false,
    "event_attendance": 0
  }'
```

**Response:**
```json
{
  "predicted_trips": 284,
  "confidence_interval": [238, 330],
  "top_factors": [
    {"feature": "demand_lag_168h",  "value": 271.0, "shap_value": 48.3, "direction": "increases_demand"},
    {"feature": "is_rush_hour",     "value": 1.0,   "shap_value": 31.7, "direction": "increases_demand"},
    {"feature": "precipitation_mm", "value": 0.0,   "shap_value": -12.1,"direction": "decreases_demand"},
    {"feature": "temperature_f",    "value": 78.5,  "shap_value": 9.4,  "direction": "increases_demand"},
    {"feature": "is_weekend",       "value": 0.0,   "shap_value": -7.2, "direction": "decreases_demand"}
  ],
  "model_version": "1.0.0",
  "zone_name": "Midtown Center"
}
```

---

## Model Comparison

All models evaluated on a held-out 2-month test set (temporal split — no shuffling).

| Model                | RMSE ↓ | MAE ↓ | MAPE ↓  | R² ↑   |
|----------------------|--------|-------|---------|--------|
| Baseline (hist avg)  | ~42    | ~31   | ~18.2%  | ~0.71  |
| Linear Regression    | ~35    | ~26   | ~15.1%  | ~0.79  |
| Random Forest        | ~22    | ~16   | ~9.5%   | ~0.91  |
| LightGBM             | ~18    | ~13   | ~7.5%   | ~0.94  |
| **XGBoost (Optuna)** | **~17**| **~12**| **~7.1%**| **~0.95** |

> **Why not a neural network?**
> LSTMs and Transformers were benchmarked on this dataset. Gradient boosting
> consistently outperformed them on both RMSE and training time. This is expected:
> structured tabular time-series with engineered lag features is gradient boosting's
> home turf. The honest answer is always better than the flashy one.

---

## Project Structure

```
urban-demand-predictor/
├── sql/                    # All SQL: schema, analysis queries, ML materialized view
├── src/
│   ├── ingest/             # Data downloaders (taxi, weather, events, FRED)
│   ├── features/           # Feature registry + engineering pipeline
│   └── models/             # Trainer (MLflow), predictor (SHAP API), evaluator
├── api/                    # FastAPI app: /predict, /predict/batch, /health
├── dashboard/              # 4-page Streamlit dashboard
├── monitoring/             # Evidently AI drift reports
├── notebooks/              # 5 Jupyter notebooks (EDA → modeling → SHAP)
├── tests/                  # 60+ unit + integration tests
├── docker-compose.yml      # Postgres + MLflow + API + Dashboard
└── Makefile                # make setup | ingest | train | serve | test
```

---

## Feature Set (22 features)

| Feature | Source | Type | Why it matters |
|---------|--------|------|----------------|
| `demand_lag_168h` | Taxi | Numeric | Same hour last week — strongest predictor |
| `demand_lag_24h` | Taxi | Numeric | Same hour yesterday |
| `demand_rolling_7d_avg` | Taxi | Numeric | Zone baseline |
| `hour_sin` / `hour_cos` | Time | Cyclical | Hour of day (sin/cos avoids discontinuity) |
| `dow_sin` / `dow_cos` | Time | Cyclical | Day of week |
| `is_rush_hour` | Time | Binary | Weekday 7–9 AM or 4–7 PM |
| `is_weekend` | Time | Binary | Weekend demand pattern |
| `temperature_f` | Weather | Numeric | U-shaped relationship |
| `feels_like_f` | Weather | Numeric | Perceived comfort |
| `precipitation_mm` | Weather | Numeric | Rain → more taxi demand |
| `windspeed_mph` | Weather | Numeric | High wind → more indoor transport |
| `humidity_pct` | Weather | Numeric | Comfort signal |
| `is_raining` | Weather | Binary | Derived from weather_code |
| `is_snowing` | Weather | Binary | Derived from weather_code |
| `weather_severity` | Weather | Ordinal | 0=clear → 3=snow |
| `has_major_event` | Events | Binary | Concert/parade/game day |
| `event_attendance` | Events | Numeric | Scale of event |
| `unemployment_rate` | FRED | Numeric | Macro demand signal |
| `gas_price_avg` | FRED | Numeric | Mode-switching proxy |
| `consumer_sentiment` | FRED | Numeric | Discretionary travel proxy |

---

## Methodology

**Data layer**: 3M+ taxi trips, hourly weather, 60+ annotated events, and 4 FRED series are
loaded into PostgreSQL and joined via a materialized view. All feature engineering SQL is
version-controlled and reproducible with a single `REFRESH MATERIALIZED VIEW` command.

**Modeling**: Three gradient-boosted models trained with a temporal train/test split.
XGBoost receives 50 Optuna trials of Bayesian hyperparameter search; all runs are logged
to MLflow with full reproducibility.

**Serving**: FastAPI wraps the trained model. Every `/predict` request returns the
prediction alongside 5 SHAP-based explanations computed from a pre-loaded TreeExplainer
(~15 ms overhead per request). A `/predict/batch` endpoint handles up to 100 zones simultaneously.

**Monitoring**: Evidently AI compares current feature distributions against the training
reference window. Reports are generated on demand from the dashboard or via CLI.

---

## What I Learned

1. **Where to engineer features matters as much as what features to build.** A well-crafted
   materialized view in SQL can handle most of the aggregation and joining work,
   reducing the Python feature pipeline to pure mathematical transforms
   (cyclical encoding, lag filling). That separation makes the system easier to
   debug and lets SQL-fluent analysts query the features directly.

2. **Temporal cross-validation is non-negotiable.** Early experiments with shuffled
   splits showed R² > 0.99 — suspiciously good. The leak came from lag features
   computed on the full dataset before splitting. Temporal split immediately
   exposed the real performance (~0.95 R²), which is strong but honest.

3. **Per-prediction SHAP explanations at API-time added ~15 ms latency**
   using a pre-loaded TreeExplainer — a worthwhile tradeoff. Pre-building the
   explainer at startup (not on each request) was the key optimization. Without
   that, latency was ~400 ms.

---

*Built by Shril Patel · [GitHub](https://github.com/ZeroZulu) · [LinkedIn](https://www.linkedin.com/in/shril-patel-020504284/) · [Live Demo](https://urban-demand-predictor-lq2ja9lznlgwe7klggp8md.streamlit.app)*
