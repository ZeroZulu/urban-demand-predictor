"""
Unit tests for feature engineering pipeline.
Tests build_features() output shape, value bounds, and data integrity.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.builder import build_features
from src.features.registry import FEATURE_REGISTRY, MODEL_FEATURES


class TestBuildFeatures:
    """Tests for the main feature engineering function."""

    def test_output_is_dataframe(self, sample_raw_row):
        result = build_features(sample_raw_row)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape(self, sample_raw_row):
        result = build_features(sample_raw_row)
        assert result.shape[0] == 1
        assert result.shape[1] == len(MODEL_FEATURES)

    def test_no_nulls_in_output(self, sample_raw_row):
        result = build_features(sample_raw_row)
        null_cols = result.columns[result.isnull().any()].tolist()
        assert not null_cols, f"Unexpected NaN columns: {null_cols}"

    def test_cyclical_features_bounded(self, sample_raw_row):
        result = build_features(sample_raw_row)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                    "month_sin", "month_cos"]:
            val = result[col].iloc[0]
            assert -1.0 <= val <= 1.0, f"{col} = {val} is out of [-1, 1]"

    def test_binary_features_are_0_or_1(self, sample_raw_row):
        result = build_features(sample_raw_row)
        for col in ["is_weekend", "is_rush_hour", "is_raining", "is_snowing",
                    "has_major_event"]:
            if col in result.columns:
                val = result[col].iloc[0]
                assert val in (0, 1), f"{col} = {val} is not binary"

    def test_weather_severity_range(self, sample_raw_row):
        result = build_features(sample_raw_row)
        val = result["weather_severity"].iloc[0]
        assert 0 <= val <= 3, f"weather_severity = {val} out of [0, 3]"

    def test_rush_hour_on_weekday_peak(self):
        """Weekday at 8 AM should be rush hour."""
        row = pd.DataFrame([{
            "hour_of_day": 8, "day_of_week": 1, "month": 6,
            "is_weekend": 0, "temperature_f": 70.0, "feels_like_f": 70.0,
            "precipitation_mm": 0.0, "windspeed_mph": 5.0, "humidity_pct": 60.0,
            "is_raining": 0, "is_snowing": 0, "has_major_event": 0,
            "event_attendance": 0, "demand_lag_24h": 200.0,
            "demand_lag_168h": 205.0, "trip_count": 200,
            "unemployment_rate": 3.8, "gas_price_avg": 3.6, "consumer_sentiment": 76.0,
        }])
        result = build_features(row)
        assert result["is_rush_hour"].iloc[0] == 1

    def test_not_rush_hour_on_weekend(self):
        """Weekend at 8 AM should NOT be rush hour."""
        row = pd.DataFrame([{
            "hour_of_day": 8, "day_of_week": 5, "month": 6,
            "is_weekend": 1, "temperature_f": 70.0, "feels_like_f": 70.0,
            "precipitation_mm": 0.0, "windspeed_mph": 5.0, "humidity_pct": 60.0,
            "is_raining": 0, "is_snowing": 0, "has_major_event": 0,
            "event_attendance": 0, "demand_lag_24h": 200.0,
            "demand_lag_168h": 205.0, "trip_count": 200,
            "unemployment_rate": 3.8, "gas_price_avg": 3.6, "consumer_sentiment": 76.0,
        }])
        result = build_features(row)
        assert result["is_rush_hour"].iloc[0] == 0

    def test_rain_detection(self, sample_raw_row_rainy):
        """Rows with rain weather_code should have is_raining=1."""
        result = build_features(sample_raw_row_rainy)
        assert result["is_raining"].iloc[0] == 1
        assert result["is_snowing"].iloc[0] == 0

    def test_weather_severity_higher_for_rain(self, sample_raw_row, sample_raw_row_rainy):
        dry_result  = build_features(sample_raw_row)
        rain_result = build_features(sample_raw_row_rainy)
        assert rain_result["weather_severity"].iloc[0] > dry_result["weather_severity"].iloc[0]

    def test_columns_match_registry(self, sample_raw_row):
        """Output columns must exactly match MODEL_FEATURES."""
        result = build_features(sample_raw_row)
        assert set(result.columns) == set(MODEL_FEATURES), (
            f"Extra: {set(result.columns) - set(MODEL_FEATURES)}\n"
            f"Missing: {set(MODEL_FEATURES) - set(result.columns)}"
        )

    def test_handles_null_lag_features(self, sample_raw_row):
        """Null lag features (cold-start zones) should be filled gracefully."""
        row = sample_raw_row.copy()
        row["demand_lag_24h"]  = None
        row["demand_lag_168h"] = None
        result = build_features(row)
        assert result["demand_lag_24h"].iloc[0] is not None
        assert not pd.isna(result["demand_lag_24h"].iloc[0])

    def test_batch_processing(self, sample_raw_row, sample_raw_row_rainy):
        """build_features should handle multi-row DataFrames."""
        batch = pd.concat([sample_raw_row, sample_raw_row_rainy], ignore_index=True)
        result = build_features(batch)
        assert result.shape[0] == 2
        assert result.isnull().sum().sum() == 0


class TestFeatureRegistry:
    """Tests for the feature registry definitions."""

    def test_all_model_features_have_entries(self):
        registry_names = {f.name for f in FEATURE_REGISTRY}
        for feat in MODEL_FEATURES:
            assert feat in registry_names, f"'{feat}' in MODEL_FEATURES but not in registry"

    def test_no_duplicate_feature_names(self):
        names = [f.name for f in FEATURE_REGISTRY]
        assert len(names) == len(set(names)), "Duplicate feature names in registry"
