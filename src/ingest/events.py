"""
NYC Events data processor.
Loads a curated events CSV from data/external/events.csv into PostgreSQL.

The CSV should have columns:
  event_date, event_name, event_type, borough, est_attendance, is_major

A sample events file is included at data/external/sample_events.csv.

Usage:
    python -m src.ingest.events
"""
import pandas as pd
from pathlib import Path
from src.utils.db import get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)
EVENTS_PATH = Path("data/external/events.csv")
SAMPLE_PATH = Path("data/external/sample_events.csv")


def run():
    path = EVENTS_PATH if EVENTS_PATH.exists() else SAMPLE_PATH
    if not path.exists():
        logger.warning(f"Events file not found at {path}. Skipping.")
        return

    df = pd.read_csv(path, parse_dates=["event_date"])
    df["is_major"] = df.get("is_major", False).fillna(False).astype(bool)
    df["est_attendance"] = df.get("est_attendance", 0).fillna(0).astype(int)

    engine = get_engine()
    df.to_sql("events", engine, if_exists="replace", index=False, method="multi")
    logger.info(f"Loaded {len(df)} events from {path}")
    logger.info("✅ Events ingestion complete.")


if __name__ == "__main__":
    run()


# ── Curated event data (embedded so tests work without any CSV on disk) ────────
_CURATED_EVENTS_CSV = """event_date,event_name,event_type,borough,est_attendance,is_major
2024-01-01,New Year's Day,holiday,Manhattan,500000,true
2024-03-17,St. Patrick's Day Parade,parade,Manhattan,150000,true
2024-05-27,Memorial Day,holiday,All,0,false
2024-06-29,NYC Pride Parade,parade,Manhattan,100000,true
2024-07-04,Independence Day,holiday,All,500000,true
2024-11-04,NYC Marathon,sports,All,55000,true
2024-11-28,Macy's Thanksgiving Parade,parade,Manhattan,350000,true
2024-12-31,New Year's Eve,holiday,Manhattan,1000000,true
"""

_MSG_EVENTS_CSV = """event_date,event_name,event_type,borough,est_attendance,is_major
2024-01-10,Knicks vs Lakers,sports,Manhattan,19500,false
2024-02-07,Knicks vs Celtics,sports,Manhattan,19500,false
2024-05-15,Knicks Playoffs Game 1,sports,Manhattan,19500,true
"""


import io as _io


def build_events_dataframe() -> pd.DataFrame:
    """
    Build a combined events DataFrame from embedded curated data and any
    user-supplied CSV at data/external/events.csv.
    No file on disk is required — the embedded data is always available.
    """
    import io
    dfs = [
        pd.read_csv(io.StringIO(_CURATED_EVENTS_CSV), parse_dates=["event_date"]),
        pd.read_csv(io.StringIO(_MSG_EVENTS_CSV),     parse_dates=["event_date"]),
    ]

    user_path = Path("data/external/events.csv")
    if user_path.exists():
        try:
            dfs.append(pd.read_csv(user_path, parse_dates=["event_date"]))
        except Exception as exc:
            logger.warning(f"Could not read {user_path}: {exc}")

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["event_date", "event_name"])
    df["is_major"]        = df["is_major"].fillna(False).astype(bool)
    df["est_attendance"]  = df["est_attendance"].fillna(0).astype(int)
    df = df.sort_values("event_date").reset_index(drop=True)
    return df
