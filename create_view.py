import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL", "postgresql://analyst:localdev@localhost:5432/urban_demand")
url = db_url.replace("postgresql://", "")
user_pass, rest = url.split("@")
user, password = user_pass.split(":")
host_port, dbname = rest.split("/")
host, port = host_port.split(":")

conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)
conn.autocommit = True
cur = conn.cursor()

print("Dropping old view if exists...")
cur.execute("DROP MATERIALIZED VIEW IF EXISTS ml_features")

print("Creating ml_features materialized view (takes 3-5 min on 28M rows)...")
cur.execute("""
CREATE MATERIALIZED VIEW ml_features AS
SELECT
    date_trunc('hour', t.pickup_dt)                  AS hour_bucket,
    t.pickup_zone                                     AS zone_id,
    COUNT(*)                                          AS trip_count,
    EXTRACT(HOUR  FROM t.pickup_dt)::INTEGER          AS hour_of_day,
    EXTRACT(DOW   FROM t.pickup_dt)::INTEGER          AS day_of_week,
    EXTRACT(MONTH FROM t.pickup_dt)::INTEGER          AS month,
    CASE WHEN EXTRACT(DOW FROM t.pickup_dt) IN (0,6)
         THEN 1 ELSE 0 END                            AS is_weekend,
    w.temperature_f,
    w.feels_like_f,
    w.precipitation_mm,
    w.windspeed_mph,
    w.humidity_pct,
    w.weather_code,
    COALESCE(w.is_raining::INT, 0)                    AS is_raining,
    COALESCE(w.is_snowing::INT, 0)                    AS is_snowing,
    COALESCE(e.is_major::INT, 0)                      AS has_major_event,
    COALESCE(e.est_attendance, 0)                     AS event_attendance,
    ec.unemployment_rate,
    ec.gas_price_avg,
    ec.consumer_sentiment
FROM taxi_trips t
LEFT JOIN weather_hourly w
    ON date_trunc('hour', t.pickup_dt) = w.obs_dt
LEFT JOIN events e
    ON DATE(t.pickup_dt) = e.event_date
LEFT JOIN economic_indicators ec
    ON date_trunc('month', t.pickup_dt) = ec.month_start
GROUP BY
    date_trunc('hour', t.pickup_dt),
    t.pickup_zone,
    EXTRACT(HOUR  FROM t.pickup_dt),
    EXTRACT(DOW   FROM t.pickup_dt),
    EXTRACT(MONTH FROM t.pickup_dt),
    w.temperature_f,
    w.feels_like_f,
    w.precipitation_mm,
    w.windspeed_mph,
    w.humidity_pct,
    w.weather_code,
    w.is_raining,
    w.is_snowing,
    e.is_major,
    e.est_attendance,
    ec.unemployment_rate,
    ec.gas_price_avg,
    ec.consumer_sentiment
""")

print("Creating index...")
cur.execute("CREATE INDEX idx_ml_features_zone_dt ON ml_features(zone_id, hour_bucket)")

cur.execute("SELECT COUNT(*) FROM ml_features")
count = cur.fetchone()[0]
print(f"\nDone! ml_features has {count:,} rows — ready to train.")

cur.close()
conn.close()