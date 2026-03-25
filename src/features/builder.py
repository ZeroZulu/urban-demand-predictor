"""
Feature engineering pipeline.

Converts the raw ml_features materialized view output into the
model-ready feature matrix defined by FEATURE_REGISTRY.

All transforms are pure functions so they can be unit-tested
and called identically during training and inference.
"""
import numpy as np
import pandas as pd

from src.features.registry import MODEL_FEATURES


# ── Cyclical encoding helpers ─────────────────────────────────

def cyclical_encode(series: pd.Series, period: float):
    """Return (sin, cos) pair for a periodic numeric series."""
    radians = 2 * np.pi * series / period
    return np.sin(radians), np.cos(radians)


# ── Main feature builder ──────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the ml_features view DataFrame into a model-ready feature matrix.

    Args:
        df: Raw DataFrame from the ml_features materialized view (or a
            single-row dict converted to a DataFrame for inference).

    Returns:
        DataFrame with exactly the columns listed in MODEL_FEATURES,
        no NaN values, in the same row order as the input.
    """
    feat = pd.DataFrame(index=df.index)

    # ── Time: cyclical encodings ──────────────────────────────
    feat["hour_sin"], feat["hour_cos"] = cyclical_encode(df["hour_of_day"], 24)
    feat["dow_sin"],  feat["dow_cos"]  = cyclical_encode(df["day_of_week"], 7)
    feat["month_sin"], feat["month_cos"] = cyclical_encode(df["month"], 12)

    # ── Time: binary flags ────────────────────────────────────
    feat["is_weekend"] = df["is_weekend"].fillna(0).astype(int)
    feat["is_rush_hour"] = (
        (~feat["is_weekend"].astype(bool)) &
        (df["hour_of_day"].isin(list(range(7, 10)) + list(range(16, 20))))
    ).astype(int)

    # ── Weather ───────────────────────────────────────────────
    for col in ["temperature_f", "feels_like_f", "precipitation_mm",
                "windspeed_mph", "humidity_pct"]:
        feat[col] = df[col].fillna(df[col].median() if len(df) > 1 else 60.0)

    feat["is_raining"] = df["is_raining"].fillna(0).astype(int)
    feat["is_snowing"] = df["is_snowing"].fillna(0).astype(int)

    feat["weather_severity"] = (
        feat["is_snowing"] * 3 +
        feat["is_raining"] * 2 +
        (feat["windspeed_mph"] > 20).astype(int)
    ).clip(0, 3)

    # ── Events ────────────────────────────────────────────────
    feat["has_major_event"]  = df["has_major_event"].fillna(0).astype(int)
    feat["event_attendance"] = df["event_attendance"].fillna(0).astype(float)

    # ── Lag features ─────────────────────────────────────────
    # At inference time, lag features arrive as 0.0 (no history available).
    # Use a realistic fallback (200 trips/hr) so the model is not anchored to zero.
    is_inference = len(df) == 1

    if is_inference:
        fallback = pd.Series(200.0, index=df.index)
    else:
        fallback = (
            df["trip_count"].rolling(24, min_periods=1).mean()
            if "trip_count" in df.columns
            else pd.Series(200.0, index=df.index)
        )

    feat["demand_lag_24h"]  = df["demand_lag_24h"].replace(0.0, np.nan).fillna(fallback)
    feat["demand_lag_168h"] = df["demand_lag_168h"].replace(0.0, np.nan).fillna(fallback)

    if not is_inference and "trip_count" in df.columns:
        feat["demand_rolling_7d_avg"] = (
            df["trip_count"].rolling(window=168, min_periods=1).mean()
        )
    else:
        feat["demand_rolling_7d_avg"] = fallback

    # ── Economic ──────────────────────────────────────────────
    for col in ["unemployment_rate", "gas_price_avg", "consumer_sentiment"]:
        series = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
        feat[col] = (
            series
            .ffill()
            .bfill()
            .fillna(series.median() if len(series.dropna()) > 0 else 0.0)
        )

    # ── Return only model features, in registry order ─────────
    return feat[MODEL_FEATURES].reset_index(drop=True)
