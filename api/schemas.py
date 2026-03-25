"""
Pydantic schemas for the UrbanPulse prediction API.
Uses typing.List/Optional for Pydantic v2 compatibility.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    zone_id:          int   = Field(..., ge=1, le=263)
    datetime:         datetime
    temperature_f:    float = Field(..., ge=-40, le=130)
    feels_like_f:     float = Field(..., ge=-50, le=140)
    precipitation_mm: float = Field(0.0, ge=0, le=500)
    windspeed_mph:    float = Field(0.0, ge=0, le=200)
    humidity_pct:     float = Field(50.0, ge=0, le=100)
    weather_code:     int   = Field(0, ge=0, le=99)
    has_major_event:  bool  = Field(False)
    event_attendance: int   = Field(0, ge=0)

    @field_validator("datetime")
    @classmethod
    def snap_to_hour(cls, v: datetime) -> datetime:
        return v.replace(minute=0, second=0, microsecond=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "zone_id": 161,
                "datetime": "2024-06-15T18:00:00",
                "temperature_f": 78.5,
                "feels_like_f": 80.0,
                "precipitation_mm": 0.0,
                "windspeed_mph": 8.2,
                "humidity_pct": 65.0,
                "weather_code": 1,
                "has_major_event": False,
                "event_attendance": 0,
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest] = Field(..., min_length=1, max_length=100)


class SHAPFactor(BaseModel):
    feature:    str
    value:      float
    shap_value: float
    direction:  str


class PredictionResponse(BaseModel):
    predicted_trips:     int
    confidence_interval: List[int]
    top_factors:         List[SHAPFactor]
    model_version:       str
    zone_name:           Optional[str] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count:       int


class HealthResponse(BaseModel):
    status:        str
    model_version: str
    model_loaded:  bool
    feature_count: int
