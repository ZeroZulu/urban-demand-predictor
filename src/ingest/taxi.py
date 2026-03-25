"""
NYC Yellow Taxi trip data downloader and PostgreSQL loader.

Downloads Parquet files from NYC TLC open data portal and bulk-loads
them into the taxi_trips table.

Usage:
    python -m src.ingest.taxi
"""
import os
import requests
import pandas as pd
from pathlib import Path
from sqlalchemy import text
import yaml

from src.utils.db import get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)
RAW_DIR = Path("data/raw")
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

COLUMN_MAP = {
    "tpep_pickup_datetime":  "pickup_dt",
    "tpep_dropoff_datetime": "dropoff_dt",
    "PULocationID":          "pickup_zone",
    "DOLocationID":          "dropoff_zone",
    "passenger_count":       "passenger_cnt",
    "trip_distance":         "trip_distance",
    "fare_amount":           "fare_amount",
    "total_amount":          "total_amount",
    "payment_type":          "payment_type",
}


def download_month(year_month: str) -> Path:
    """Download a single month of taxi parquet data."""
    filename = f"yellow_tripdata_{year_month}.parquet"
    dest = RAW_DIR / filename
    if dest.exists():
        logger.info(f"Already downloaded: {filename}")
        return dest

    url = f"{BASE_URL}/{filename}"
    logger.info(f"Downloading {url} …")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    logger.info(f"Saved → {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def load_parquet_to_db(path: Path) -> int:
    """Load a parquet file into the taxi_trips table. Returns rows inserted."""
    logger.info(f"Loading {path.name} into PostgreSQL …")
    df = pd.read_parquet(path, columns=list(COLUMN_MAP.keys()))
    df = df.rename(columns=COLUMN_MAP)

    # Basic cleaning
    df = df.dropna(subset=["pickup_dt", "dropoff_dt", "pickup_zone"])
    df = df[df["fare_amount"] > 0]
    df = df[df["trip_distance"] > 0]
    df["pickup_dt"]  = pd.to_datetime(df["pickup_dt"])
    df["dropoff_dt"] = pd.to_datetime(df["dropoff_dt"])
    df["pickup_zone"]  = df["pickup_zone"].astype(int)
    df["dropoff_zone"] = df["dropoff_zone"].fillna(0).astype(int)
    df["passenger_cnt"] = df["passenger_cnt"].fillna(1).astype(int)
    df["payment_type"]  = df["payment_type"].fillna(0).astype(int)

    engine = get_engine()
    df.to_sql(
        "taxi_trips", engine,
        if_exists="append", index=False,
        method="multi", chunksize=10_000,
    )
    logger.info(f"Inserted {len(df):,} rows from {path.name}")
    return len(df)


def run():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    months = cfg["data"]["taxi"]["months"]
    total = 0
    for ym in months:
        path = download_month(ym)
        total += load_parquet_to_db(path)
    logger.info(f"✅ Taxi ingestion complete — {total:,} total rows loaded.")


if __name__ == "__main__":
    run()

VALID_ZONE_RANGE = (1, 263)


def clean_taxi_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and apply quality filters to a raw TLC DataFrame."""
    df = df.rename(columns=COLUMN_MAP)
    keep = list(COLUMN_MAP.values())
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df[
        df["pickup_dt"].notna()
        & df["dropoff_dt"].notna()
        & df["pickup_zone"].between(*VALID_ZONE_RANGE)
        & df["fare_amount"].between(0, 500)
        & df["trip_distance"].between(0, 100)
    ]
    df["pickup_zone"]   = df["pickup_zone"].astype("int16")
    df["dropoff_zone"]  = df["dropoff_zone"].fillna(0).astype("int16")
    df["passenger_cnt"] = df["passenger_cnt"].fillna(1).clip(0, 9).astype("int8")
    df["payment_type"]  = df["payment_type"].fillna(0).astype("int8")
    return df


def ingest_months(months: list) -> None:
    """Download and load a list of year-month strings (e.g. ['2024-01'])."""
    for month in months:
        path = download_month(month)
        load_parquet_to_db(path)
    logger.info("Taxi ingestion complete.")
