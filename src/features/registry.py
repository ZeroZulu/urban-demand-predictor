"""
Centralized feature catalog.
Every feature used in the model is defined here with metadata.
"""
from dataclasses import dataclass, field
from typing import Literal

FeatureType = Literal["numeric", "binary", "cyclical", "categorical"]


@dataclass
class Feature:
    name: str
    source: str
    dtype: FeatureType
    description: str
    in_model: bool = True
    nullable: bool = False


FEATURE_REGISTRY: list[Feature] = [
    # ── Time features ─────────────────────────────────────────
    Feature("hour_sin",          "time",    "cyclical", "Sin encoding of hour-of-day"),
    Feature("hour_cos",          "time",    "cyclical", "Cos encoding of hour-of-day"),
    Feature("dow_sin",           "time",    "cyclical", "Sin encoding of day-of-week"),
    Feature("dow_cos",           "time",    "cyclical", "Cos encoding of day-of-week"),
    Feature("month_sin",         "time",    "cyclical", "Sin encoding of month"),
    Feature("month_cos",         "time",    "cyclical", "Cos encoding of month"),
    Feature("is_weekend",        "time",    "binary",   "1 if Saturday or Sunday"),
    Feature("is_rush_hour",      "time",    "binary",   "1 if 7–9 AM or 4–7 PM on a weekday"),

    # ── Weather features ──────────────────────────────────────
    Feature("temperature_f",     "weather", "numeric",  "Actual temperature in °F"),
    Feature("feels_like_f",      "weather", "numeric",  "Apparent temperature in °F"),
    Feature("precipitation_mm",  "weather", "numeric",  "Hourly precipitation in mm"),
    Feature("windspeed_mph",     "weather", "numeric",  "Wind speed in mph"),
    Feature("humidity_pct",      "weather", "numeric",  "Relative humidity %"),
    Feature("is_raining",        "weather", "binary",   "1 if weather_code indicates rain"),
    Feature("is_snowing",        "weather", "binary",   "1 if weather_code indicates snow"),
    Feature("weather_severity",  "weather", "numeric",  "Ordinal: 0=clear, 1=cloudy, 2=rain, 3=snow"),

    # ── Event features ────────────────────────────────────────
    Feature("has_major_event",   "events",  "binary",   "1 if a major event occurs in NYC today"),
    Feature("event_attendance",  "events",  "numeric",  "Estimated event attendance (0 if none)"),

    # ── Lag / historical demand ───────────────────────────────
    Feature("demand_lag_24h",        "taxi", "numeric", "Trip count same zone 24h ago",   nullable=True),
    Feature("demand_lag_168h",       "taxi", "numeric", "Trip count same zone 168h ago",  nullable=True),
    Feature("demand_rolling_7d_avg", "taxi", "numeric", "7-day rolling hourly avg for zone"),

    # ── Economic indicators ───────────────────────────────────
    Feature("unemployment_rate", "fred",    "numeric",  "US unemployment rate (monthly)"),
    Feature("gas_price_avg",     "fred",    "numeric",  "US avg gas price per gallon"),
    Feature("consumer_sentiment","fred",    "numeric",  "U-Mich consumer sentiment index"),
]

MODEL_FEATURES = [f.name for f in FEATURE_REGISTRY if f.in_model]
