"""
UrbanPulse FastAPI Application
Prediction service for NYC taxi demand forecasting.

Docs:   http://localhost:8000/docs
Redoc:  http://localhost:8000/redoc
Health: http://localhost:8000/health
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import health, predict
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model artifacts once at startup — not on every request.
    Raises a clear error if models haven't been trained yet.
    """
    try:
        from src.models.predictor import load_explainer, load_feature_names, load_model
        load_model()
        load_feature_names()
        load_explainer()
        logger.info("Model artifacts loaded successfully.")
    except FileNotFoundError as e:
        logger.warning(
            f"Model not found: {e}. "
            "The /predict endpoint will return 503 until `make train` is run."
        )
    yield
    logger.info("Shutting down UrbanPulse API.")


app = FastAPI(
    title="UrbanPulse — Urban Demand Predictor",
    description=(
        "Predicts NYC taxi demand by zone and hour using weather, civic events, "
        "and economic signals. Returns predictions with SHAP-based explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "Shril Patel",
        "url": "https://github.com/ZeroZulu/urban-demand-predictor",
    },
)

# ── CORS (for Streamlit dashboard) ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://dashboard:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(predict.router)


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "service":  "UrbanPulse Demand Predictor",
        "version":  "1.0.0",
        "docs":     "/docs",
        "health":   "/health",
        "predict":  "/predict",
    })
