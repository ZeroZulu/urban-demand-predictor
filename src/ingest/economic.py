"""
FRED (Federal Reserve Economic Data) indicator ingestion.
Requires FRED_API_KEY in environment / .env file.

Usage:
    python -m src.ingest.economic
"""
import os
import requests
import pandas as pd
from datetime import date
from dotenv import load_dotenv
from src.utils.db import get_engine
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "unemployment_rate":  "UNRATE",
    "consumer_sentiment": "UMCSENT",
    "gas_price_avg":      "GASREGCOVW",
    "cpi":                "CPIAUCSL",
}


def fetch_series(series_id: str, start: str, end: str, api_key: str) -> pd.Series:
    """Fetch a FRED series and return as a date-indexed Series."""
    params = {
        "series_id":      series_id,
        "observation_start": start,
        "observation_end":   end,
        "api_key":        api_key,
        "file_type":      "json",
        "frequency":      "m",  # monthly
    }
    resp = requests.get(FRED_BASE, params=params, timeout=30)
    resp.raise_for_status()

    obs = resp.json().get("observations", [])
    records = {
        pd.Timestamp(o["date"]): float(o["value"])
        for o in obs
        if o["value"] != "."
    }
    return pd.Series(records, name=series_id)


def run():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logger.warning("FRED_API_KEY not set — skipping economic data ingestion.")
        logger.warning("Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    start, end = "2024-01-01", str(date.today())
    frames = {}
    for col_name, series_id in SERIES.items():
        logger.info(f"Fetching FRED series {series_id} …")
        frames[col_name] = fetch_series(series_id, start, end, api_key)

    df = pd.DataFrame(frames)
    df.index.name = "month_start"
    df = df.reset_index()
    df["month_start"] = df["month_start"].dt.date

    engine = get_engine()
    df.to_sql(
        "economic_indicators", engine,
        if_exists="replace", index=False, method="multi",
    )
    logger.info(f"Loaded {len(df)} months of economic indicators.")
    logger.info("✅ Economic ingestion complete.")


if __name__ == "__main__":
    run()
