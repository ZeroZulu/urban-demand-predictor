"""
Shared pytest fixtures for UrbanPulse test suite.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_row() -> pd.DataFrame:
    """A single row as it comes from the ml_features materialized view."""
    return pd.DataFrame([{
        "hour_bucket":        "2024-06-15 17:00:00",
        "zone_id":            161,
        "trip_count":         280,
        "hour_of_day":        17,
        "day_of_week":        5,          # Saturday
        "month":              6,
        "is_weekend":         1,
        "temperature_f":      78.5,
        "feels_like_f":       80.0,
        "precipitation_mm":   0.0,
        "windspeed_mph":      8.2,
        "humidity_pct":       65.0,
        "weather_code":       1,
        "is_raining":         0,
        "is_snowing":         0,
        "has_major_event":    0,
        "event_attendance":   0,
        "demand_lag_24h":     260.0,
        "demand_lag_168h":    270.0,
        "unemployment_rate":  3.8,
        "gas_price_avg":      3.65,
        "consumer_sentiment": 76.0,
    }])


@pytest.fixture
def sample_raw_row_rainy() -> pd.DataFrame:
    """Row with heavy rain conditions."""
    return pd.DataFrame([{
        "hour_bucket":        "2024-03-10 08:00:00",
        "zone_id":            162,
        "trip_count":         320,
        "hour_of_day":        8,
        "day_of_week":        0,          # Monday
        "month":              3,
        "is_weekend":         0,
        "temperature_f":      48.0,
        "feels_like_f":       42.0,
        "precipitation_mm":   12.5,
        "windspeed_mph":      18.0,
        "humidity_pct":       90.0,
        "weather_code":       63,
        "is_raining":         1,
        "is_snowing":         0,
        "has_major_event":    0,
        "event_attendance":   0,
        "demand_lag_24h":     290.0,
        "demand_lag_168h":    285.0,
        "unemployment_rate":  3.9,
        "gas_price_avg":      3.70,
        "consumer_sentiment": 75.0,
    }])


@pytest.fixture
def sample_features_df(sample_raw_row) -> pd.DataFrame:
    """Pre-built feature DataFrame."""
    from src.features.builder import build_features
    return build_features(sample_raw_row)


@pytest.fixture
def prediction_payload() -> dict:
    """Valid prediction request dict for API tests."""
    return {
        "zone_id":          161,
        "datetime":         "2024-06-15T18:00:00",
        "temperature_f":    78.5,
        "feels_like_f":     80.0,
        "precipitation_mm": 0.0,
        "windspeed_mph":    8.2,
        "humidity_pct":     65.0,
        "weather_code":     1,
        "has_major_event":  False,
        "event_attendance": 0,
    }
