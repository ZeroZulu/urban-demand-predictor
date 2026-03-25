"""
Unit tests for data ingestion modules.
Validates cleaning logic, column mapping, and filtering — no network calls.
"""
import numpy as np
import pandas as pd
import pytest

from src.ingest.taxi import clean_taxi_df, COLUMN_MAP


class TestTaxiCleaning:

    @pytest.fixture
    def raw_taxi_df(self):
        """Simulated raw NYC TLC parquet output."""
        return pd.DataFrame({
            "tpep_pickup_datetime":  pd.to_datetime(["2024-01-15 08:30:00",
                                                      "2024-01-15 09:00:00",
                                                      "2024-01-15 10:00:00",
                                                      "2024-01-15 11:00:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2024-01-15 08:45:00",
                                                      "2024-01-15 09:20:00",
                                                      "2024-01-15 10:15:00",
                                                      "2024-01-15 11:30:00"]),
            "PULocationID":   [161, 162, 999, 0],    # 999 and 0 are invalid
            "DOLocationID":   [163, 161, 160, 161],
            "passenger_count": [1, 2, 1, 1],
            "trip_distance":   [1.5, 3.2, 2.0, 1.0],
            "fare_amount":     [9.5, 15.0, 8.0, -5.0],  # -5 is invalid
            "total_amount":    [12.0, 18.5, 10.5, 0.0],
            "payment_type":    [1, 2, 1, 1],
        })

    def test_column_renaming(self, raw_taxi_df):
        result = clean_taxi_df(raw_taxi_df)
        for old_col, new_col in COLUMN_MAP.items():
            if old_col in raw_taxi_df.columns:
                assert new_col in result.columns, f"Missing renamed column: {new_col}"

    def test_invalid_zones_removed(self, raw_taxi_df):
        result = clean_taxi_df(raw_taxi_df)
        assert all(result["pickup_zone"].between(1, 263))

    def test_negative_fare_removed(self, raw_taxi_df):
        result = clean_taxi_df(raw_taxi_df)
        assert all(result["fare_amount"] >= 0)

    def test_output_has_fewer_rows_than_input(self, raw_taxi_df):
        result = clean_taxi_df(raw_taxi_df)
        assert len(result) < len(raw_taxi_df)

    def test_pickup_dt_is_datetime(self, raw_taxi_df):
        result = clean_taxi_df(raw_taxi_df)
        assert pd.api.types.is_datetime64_any_dtype(result["pickup_dt"])

    def test_zone_columns_are_int(self, raw_taxi_df):
        result = clean_taxi_df(raw_taxi_df)
        assert result["pickup_zone"].dtype in (np.int16, np.int32, np.int64)

    def test_no_null_pickup_datetimes(self, raw_taxi_df):
        result = clean_taxi_df(raw_taxi_df)
        assert result["pickup_dt"].notna().all()


class TestWeatherIngestion:

    def test_weather_response_parsing(self):
        """Simulate parsing an Open-Meteo API response."""
        import inspect
        import unittest.mock as mock
        from src.ingest.weather import fetch_weather

        # Discover the actual signature so the test is robust to either version
        sig = inspect.signature(fetch_weather)
        params = list(sig.parameters.keys())

        fake_response = {
            "hourly": {
                "time":                 ["2024-01-01T00:00", "2024-01-01T01:00"],
                "temperature_2m":       [32.0, 31.5],
                "apparent_temperature": [28.0, 27.5],
                "precipitation":        [0.0, 0.2],
                "windspeed_10m":        [8.0, 9.5],
                "relativehumidity_2m":  [75.0, 78.0],
                "weathercode":          [0, 61],
            }
        }

        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status.return_value = None

        with mock.patch("requests.get", return_value=mock_resp):
            # Call with positional args that match whichever signature is present
            if params[0] in ("lat", "latitude"):
                # Signature: fetch_weather(lat, lon, start, end, variables, ...)
                df = fetch_weather(
                    40.71, -74.01, "2024-01-01", "2024-01-01",
                    ["temperature_2m", "apparent_temperature",
                     "precipitation", "windspeed_10m",
                     "relativehumidity_2m", "weathercode"],
                )
            else:
                # Signature: fetch_weather(start_date, end_date)
                df = fetch_weather("2024-01-01", "2024-01-01")

        assert len(df) == 2
        assert "obs_dt" in df.columns
        # Accept either "temperature_f" (renamed) or "temperature_2m" (raw)
        assert any(c in df.columns for c in ("temperature_f", "temperature_2m"))
        assert pd.api.types.is_datetime64_any_dtype(df["obs_dt"])


class TestEventsIngestion:

    def test_events_dataframe_has_required_columns(self):
        from src.ingest.events import build_events_dataframe
        df = build_events_dataframe()
        required = ["event_date", "event_name", "event_type",
                    "borough", "est_attendance", "is_major"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_events_dataframe_not_empty(self):
        from src.ingest.events import build_events_dataframe
        df = build_events_dataframe()
        assert len(df) > 0

    def test_is_major_is_boolean(self):
        from src.ingest.events import build_events_dataframe
        df = build_events_dataframe()
        assert df["is_major"].dtype == bool

    def test_no_duplicate_events(self):
        from src.ingest.events import build_events_dataframe
        df = build_events_dataframe()
        dupes = df.duplicated(subset=["event_date", "event_name"])
        assert not dupes.any(), f"Found {dupes.sum()} duplicate events"

    def test_est_attendance_non_negative(self):
        from src.ingest.events import build_events_dataframe
        df = build_events_dataframe()
        assert (df["est_attendance"] >= 0).all()
