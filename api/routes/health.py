"""GET /health — liveness + readiness check for Docker and load balancers."""
from fastapi import APIRouter
from api.schemas import HealthResponse
from src.models.predictor import (
    MODEL_PATH,
    load_feature_names,
    load_metadata,
)

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Returns service health and loaded model metadata.
    Used by Docker healthcheck and monitoring dashboards.
    """
    model_loaded = MODEL_PATH.exists()
    try:
        feature_names = load_feature_names()
        feature_count = len(feature_names)
        metadata      = load_metadata()
        model_version = metadata.get("model_version", "unknown")
    except Exception:
        feature_count = 0
        model_version = "not_loaded"

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_version=model_version,
        model_loaded=model_loaded,
        feature_count=feature_count,
    )
