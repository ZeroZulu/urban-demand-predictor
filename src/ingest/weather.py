"""
Open-Meteo historical weather data ingestion for NYC.
Free API — no key required.

Usage:
    python -m src.ingest.weather
"""
import requests
import pandas as pd
import yaml
from src.utils.db import get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(lat: float, lon: float, start: str, end: str,
                  variables: list, timezone: str = "America/New_York") -> pd.DataFrame:
    """Fetch hourly weather observations from Open-Meteo archive."""
    params = {
        "latitude":          lat,
        "longitude":         lon,
        "start_date":        start,
        "end_date":          end,
        "hourly":            ",".join(variables),
        "temperature_unit":  "fahrenheit",
        "windspeed_unit":    "mph",
        "timezone":          timezone,
    }
    logger.info(f"Fetching weather {start} → {end} …")
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()

    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["obs_dt"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df = df.rename(columns={
        "temperature_2m":      "temperature_f",
        "apparent_temperature": "feels_like_f",
        "precipitation":       "precipitation_mm",
        "windspeed_10m":       "windspeed_mph",
        "relativehumidity_2m": "humidity_pct",
        "weathercode":         "weather_code",
    })
    return df


def load_weather_to_db(df: pd.DataFrame) -> None:
    """Upsert weather rows into weather_hourly table."""
    engine = get_engine()
    # Use INSERT ... ON CONFLICT DO NOTHING to be idempotent
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy import Table, MetaData

    meta = MetaData()
    meta.reflect(engine, only=["weather_hourly"])
    tbl = meta.tables["weather_hourly"]

    with engine.begin() as conn:
        for batch_start in range(0, len(df), 5000):
            batch = df.iloc[batch_start:batch_start + 5000]
            stmt = insert(tbl).values(batch.to_dict(orient="records"))
            stmt = stmt.on_conflict_do_nothing(index_elements=["obs_dt"])
            conn.execute(stmt)

    logger.info(f"Loaded {len(df):,} weather rows to database.")


def run():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    wc = cfg["data"]["weather"]
    df = fetch_weather(
        lat=wc["latitude"],
        lon=wc["longitude"],
        start=wc["start_date"],
        end=wc["end_date"],
        variables=wc["hourly_vars"],
        timezone=wc["timezone"],
    )
    load_weather_to_db(df)
    logger.info("✅ Weather ingestion complete.")


if __name__ == "__main__":
    run()
