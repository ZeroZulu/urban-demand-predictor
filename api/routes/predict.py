"""
POST /predict  — single prediction with SHAP explanation
POST /predict/batch — batch predictions
"""
import math
from datetime import datetime
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import (
    BatchPredictionRequest, BatchPredictionResponse,
    PredictionRequest, PredictionResponse, SHAPFactor,
)
from src.models.predictor import predict_with_explanation, predict_batch
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Predictions"])

ZONE_NAMES = {
    161: "Midtown Center",      162: "Midtown East",
    163: "Midtown North",       230: "Times Sq / Theatre District",
    186: "Penn Station / MSG",  132: "JFK Airport",
    138: "LaGuardia Airport",   234: "Union Square",
    107: "Gramercy",            170: "Murray Hill",
    236: "UES North",           237: "UES South",
    142: "Lincoln Square E",
}


def _safe_float(v) -> float:
    """Replace NaN/inf with 0.0 so JSON serialization never fails."""
    if v is None:
        return 0.0
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return round(f, 4)
    except (TypeError, ValueError):
        return 0.0


def _request_to_row(req: PredictionRequest) -> dict:
    dt: datetime = req.datetime
    return {
        "hour_of_day":        dt.hour,
        "day_of_week":        dt.weekday(),
        "month":              dt.month,
        "is_weekend":         int(dt.weekday() >= 5),
        "temperature_f":      req.temperature_f,
        "feels_like_f":       req.feels_like_f,
        "precipitation_mm":   req.precipitation_mm,
        "windspeed_mph":      req.windspeed_mph,
        "humidity_pct":       req.humidity_pct,
        "is_raining":         int(51 <= req.weather_code <= 67),
        "is_snowing":         int(71 <= req.weather_code <= 77),
        "has_major_event":    int(req.has_major_event),
        "event_attendance":   req.event_attendance,
        "demand_lag_24h":     None,
        "demand_lag_168h":    None,
        "trip_count":         None,
        "unemployment_rate":  None,
        "gas_price_avg":      None,
        "consumer_sentiment": None,
    }


def _build_response(result: dict, zone_id: int) -> PredictionResponse:
    """Build PredictionResponse, sanitizing all floats."""
    predicted = max(0, int(result.get("predicted_trips", 0)))
    ci = result.get("confidence_interval", [0, 0])
    ci_safe = [max(0, int(ci[0])), max(0, int(ci[1]))]

    factors = []
    for f in result.get("top_factors", []):
        sv = _safe_float(f.get("shap_value", 0))
        factors.append(SHAPFactor(
            feature=str(f.get("feature", "unknown")),
            value=_safe_float(f.get("value", 0)),
            shap_value=sv,
            direction="increases_demand" if sv >= 0 else "decreases_demand",
        ))

    return PredictionResponse(
        predicted_trips=predicted,
        confidence_interval=ci_safe,
        top_factors=factors,
        model_version=str(result.get("model_version", "1.0.0")),
        zone_name=ZONE_NAMES.get(zone_id),
    )


@router.post("/predict", response_model=PredictionResponse,
             summary="Predict hourly taxi demand for a single zone")
async def predict(req: PredictionRequest) -> PredictionResponse:
    try:
        row    = _request_to_row(req)
        result = predict_with_explanation(row, top_n_shap=5)
        return _build_response(result, req.zone_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}. Run `python train.py` first.")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@router.post("/predict/batch", response_model=BatchPredictionResponse,
             summary="Predict demand for multiple zones / times")
async def predict_batch_endpoint(body: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        rows    = [_request_to_row(r) for r in body.requests]
        results = predict_batch(rows, top_n_shap=5)
        responses = [_build_response(r, body.requests[i].zone_id)
                     for i, r in enumerate(results)]
        return BatchPredictionResponse(predictions=responses, count=len(responses))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))