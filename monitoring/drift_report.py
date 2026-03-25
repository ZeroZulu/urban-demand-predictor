"""
UrbanPulse — Model Drift Monitor
Generates an Evidently AI report comparing early 2024 vs late 2024 data.

Usage:
    python -m monitoring.drift_report
"""
import argparse
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.utils.db import get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("outputs/drift_reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns available directly in ml_features (no lag computation needed)
FEATURE_COLS = [
    "zone_id", "hour_of_day", "day_of_week", "month", "is_weekend",
    "temperature_f", "precipitation_mm", "windspeed_mph", "humidity_pct",
    "is_raining", "is_snowing", "has_major_event", "event_attendance",
]


def load_period(start: str, end: str) -> pd.DataFrame:
    engine = get_engine()
    cols = ", ".join(FEATURE_COLS + ["trip_count"])
    sql = f"""
        SELECT {cols}
        FROM ml_features
        WHERE hour_bucket >= '{start}'
          AND hour_bucket <  '{end}'
        ORDER BY hour_bucket
    """
    df = pd.read_sql(sql, engine)
    logger.info(f"Loaded {len(df):,} rows ({start} → {end})")
    return df


def generate_drift_report(ref_weeks: int = 4, cur_weeks: int = 1) -> Path:
    """
    Compare Jan–Mar 2024 (reference) vs Apr–Jun 2024 (current).
    Uses raw ml_features columns — no lag feature computation needed.
    """
    logger.info("Loading reference data (Jan–Mar 2024)...")
    ref_df = load_period("2024-01-01", "2024-04-01")

    logger.info("Loading current data (Apr–Jun 2024)...")
    cur_df = load_period("2024-04-01", "2024-07-01")

    if ref_df.empty or cur_df.empty:
        logger.warning("Not enough data. Check that ml_features is populated.")
        return None

    logger.info(f"Reference: {len(ref_df):,} rows | Current: {len(cur_df):,} rows")

    # Fill any nulls
    for df in [ref_df, cur_df]:
        for col in FEATURE_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median()
                                         if df[col].dtype != object else 0)

    column_mapping = ColumnMapping(
        target="trip_count",
        numerical_features=[c for c in FEATURE_COLS
                            if c not in ("zone_id",)],
        categorical_features=["zone_id"],
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=ref_df,
        current_data=cur_df,
        column_mapping=column_mapping,
    )

    output_path = OUTPUT_DIR / "drift_report_latest.html"
    report.save_html(str(output_path))
    logger.info(f"Drift report saved → {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-weeks", type=int, default=4)
    parser.add_argument("--cur-weeks", type=int, default=1)
    args = parser.parse_args()
    generate_drift_report(ref_weeks=args.ref_weeks, cur_weeks=args.cur_weeks)