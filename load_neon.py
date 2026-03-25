"""
Load a 500k-row sample into Neon (fits 512MB free tier).
Takes every 6th row from Jan 2024 = ~500k representative trips.
"""
import os, sys, requests
from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

NEON_URL = os.environ.get("NEON_URL")
if not NEON_URL:
    print("ERROR: NEON_URL not set.")
    sys.exit(1)

conn = psycopg2.connect(NEON_URL)
conn.autocommit = False
cur = conn.cursor()
print("Connected to Neon.")

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

# ── 1. Sample taxi trips (every 6th row from Jan 2024 = ~500k rows) ───────────
print("\nStep 1: Loading taxi sample (~500k rows from Jan 2024)...")
path = Path("data/raw/yellow_tripdata_2024-01.parquet")
if not path.exists():
    print("ERROR: data/raw/yellow_tripdata_2024-01.parquet not found")
    sys.exit(1)

df = pd.read_parquet(path, columns=list(COLUMN_MAP.keys()))
df = df.rename(columns=COLUMN_MAP)
df = df.dropna(subset=["pickup_dt","dropoff_dt","pickup_zone"])
df = df[df["fare_amount"].between(0,500) & df["trip_distance"].between(0,100)]
df["pickup_zone"]   = df["pickup_zone"].astype(int)
df["dropoff_zone"]  = df["dropoff_zone"].fillna(0).astype(int)
df["passenger_cnt"] = df["passenger_cnt"].fillna(1).clip(0,9).astype(int)
df["payment_type"]  = df["payment_type"].fillna(0).astype(int)

# Take every 6th row for a representative sample
df = df.iloc[::6].reset_index(drop=True)
print(f"  Sample size: {len(df):,} rows")

records = [
    (row.pickup_dt, row.dropoff_dt, row.pickup_zone, row.dropoff_zone,
     row.passenger_cnt, row.trip_distance, row.fare_amount,
     row.total_amount, row.payment_type)
    for row in df.itertuples(index=False)
]

sql = """INSERT INTO taxi_trips
         (pickup_dt,dropoff_dt,pickup_zone,dropoff_zone,passenger_cnt,
          trip_distance,fare_amount,total_amount,payment_type)
         VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"""

BATCH = 1000
for i in range(0, len(records), BATCH):
    execute_batch(cur, sql, records[i:i+BATCH], page_size=BATCH)
    conn.commit()
    if i % 50000 == 0:
        print(f"  {i:,} / {len(records):,} rows inserted...")

print(f"  Done: {len(records):,} rows loaded")

# ── 2. Weather ────────────────────────────────────────────────────────────────
print("\nStep 2: Fetching weather (Jan 2024)...")
params = {
    "latitude": 40.7128, "longitude": -74.0060,
    "start_date": "2024-01-01", "end_date": "2024-01-31",
    "hourly": "temperature_2m,apparent_temperature,precipitation,windspeed_10m,relativehumidity_2m,weathercode",
    "temperature_unit": "fahrenheit", "windspeed_unit": "mph",
    "timezone": "America/New_York",
}
resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
resp.raise_for_status()
h = resp.json()["hourly"]
weather_records = list(zip(
    pd.to_datetime(h["time"]).tolist(),
    h["temperature_2m"], h["apparent_temperature"],
    h["precipitation"], h["windspeed_10m"],
    h["relativehumidity_2m"], h["weathercode"]
))
wsql = """INSERT INTO weather_hourly
          (obs_dt,temperature_f,feels_like_f,precipitation_mm,
           windspeed_mph,humidity_pct,weather_code)
          VALUES (%s,%s,%s,%s,%s,%s,%s)
          ON CONFLICT (obs_dt) DO NOTHING"""
execute_batch(cur, wsql, weather_records, page_size=500)
conn.commit()
print(f"  Inserted {len(weather_records):,} weather rows")

# ── 3. Events ─────────────────────────────────────────────────────────────────
print("\nStep 3: Loading events...")
sys.path.insert(0, '.')
from src.ingest.events import build_events_dataframe
edf = build_events_dataframe()
cur.execute("TRUNCATE TABLE events RESTART IDENTITY")
for _, row in edf.iterrows():
    cur.execute(
        "INSERT INTO events (event_date,event_name,event_type,borough,est_attendance,is_major) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        (row.event_date, row.event_name, row.event_type,
         row.borough, int(row.est_attendance), bool(row.is_major))
    )
conn.commit()
print(f"  Inserted {len(edf)} events")

# ── 4. Economic ───────────────────────────────────────────────────────────────
print("\nStep 4: Loading economic data...")
try:
    from sqlalchemy import create_engine
    local_url = os.environ.get("DATABASE_URL","postgresql://analyst:localdev@localhost:5432/urban_demand")
    local_eng = create_engine(local_url)
    econ = pd.read_sql("SELECT * FROM economic_indicators WHERE month_start >= '2024-01-01'", local_eng)
    for _, row in econ.iterrows():
        cur.execute(
            "INSERT INTO economic_indicators (month_start,unemployment_rate,consumer_sentiment,gas_price_avg,cpi) "
            "VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (row.month_start, row.get("unemployment_rate"), row.get("consumer_sentiment"),
             row.get("gas_price_avg"), row.get("cpi"))
        )
    conn.commit()
    print(f"  Inserted {len(econ)} economic rows")
except Exception as e:
    print(f"  Skipped economic: {e}")

# ── 5. Materialized view ──────────────────────────────────────────────────────
print("\nStep 5: Creating ml_features view...")
conn.autocommit = True
cur.execute("DROP MATERIALIZED VIEW IF EXISTS ml_features")
cur.execute("""
CREATE MATERIALIZED VIEW ml_features AS
SELECT
    date_trunc('hour', t.pickup_dt)                  AS hour_bucket,
    t.pickup_zone                                     AS zone_id,
    COUNT(*)                                          AS trip_count,
    EXTRACT(HOUR  FROM t.pickup_dt)::INTEGER          AS hour_of_day,
    EXTRACT(DOW   FROM t.pickup_dt)::INTEGER          AS day_of_week,
    EXTRACT(MONTH FROM t.pickup_dt)::INTEGER          AS month,
    CASE WHEN EXTRACT(DOW FROM t.pickup_dt) IN (0,6) THEN 1 ELSE 0 END AS is_weekend,
    w.temperature_f, w.feels_like_f, w.precipitation_mm,
    w.windspeed_mph, w.humidity_pct, w.weather_code,
    COALESCE(w.is_raining::INT, 0) AS is_raining,
    COALESCE(w.is_snowing::INT, 0) AS is_snowing,
    COALESCE(e.is_major::INT, 0)   AS has_major_event,
    COALESCE(e.est_attendance, 0)  AS event_attendance,
    ec.unemployment_rate, ec.gas_price_avg, ec.consumer_sentiment
FROM taxi_trips t
LEFT JOIN weather_hourly w ON date_trunc('hour', t.pickup_dt) = w.obs_dt
LEFT JOIN events e         ON DATE(t.pickup_dt) = e.event_date
LEFT JOIN economic_indicators ec ON date_trunc('month', t.pickup_dt) = ec.month_start
GROUP BY
    date_trunc('hour', t.pickup_dt), t.pickup_zone,
    EXTRACT(HOUR FROM t.pickup_dt), EXTRACT(DOW FROM t.pickup_dt),
    EXTRACT(MONTH FROM t.pickup_dt),
    w.temperature_f, w.feels_like_f, w.precipitation_mm,
    w.windspeed_mph, w.humidity_pct, w.weather_code,
    w.is_raining, w.is_snowing, e.is_major, e.est_attendance,
    ec.unemployment_rate, ec.gas_price_avg, ec.consumer_sentiment
""")
cur.execute("CREATE INDEX idx_ml_neon ON ml_features(zone_id, hour_bucket)")
cur.execute("SELECT COUNT(*) FROM ml_features")
count = cur.fetchone()[0]
print(f"  ml_features: {count:,} rows")
cur.close()
conn.close()
print("\nNeon loading complete! Ready to deploy.")
