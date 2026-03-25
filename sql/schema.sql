-- ============================================================
-- UrbanPulse — Database Schema
-- PostgreSQL 15
-- ============================================================

-- ── Raw tables ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS taxi_trips (
    trip_id         SERIAL PRIMARY KEY,
    pickup_dt       TIMESTAMP NOT NULL,
    dropoff_dt      TIMESTAMP NOT NULL,
    pickup_zone     INTEGER NOT NULL,
    dropoff_zone    INTEGER,
    passenger_cnt   INTEGER,
    trip_distance   FLOAT,
    fare_amount     FLOAT,
    total_amount    FLOAT,
    payment_type    SMALLINT
);

CREATE TABLE IF NOT EXISTS weather_hourly (
    obs_dt              TIMESTAMP PRIMARY KEY,
    temperature_f       FLOAT,
    feels_like_f        FLOAT,
    precipitation_mm    FLOAT,
    windspeed_mph       FLOAT,
    humidity_pct        FLOAT,
    weather_code        INTEGER,
    is_raining          BOOLEAN GENERATED ALWAYS AS (weather_code BETWEEN 51 AND 67) STORED,
    is_snowing          BOOLEAN GENERATED ALWAYS AS (weather_code BETWEEN 71 AND 77) STORED
);

CREATE TABLE IF NOT EXISTS events (
    event_id            SERIAL PRIMARY KEY,
    event_date          DATE NOT NULL,
    event_name          TEXT,
    event_type          TEXT,
    borough             TEXT,
    est_attendance      INTEGER,
    is_major            BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS economic_indicators (
    month_start         DATE PRIMARY KEY,
    unemployment_rate   FLOAT,
    consumer_sentiment  FLOAT,
    gas_price_avg       FLOAT,
    cpi                 FLOAT
);

CREATE TABLE IF NOT EXISTS taxi_zones (
    zone_id             INTEGER PRIMARY KEY,
    zone_name           TEXT,
    borough             TEXT,
    service_zone        TEXT
);

-- ── Indexes ──────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_taxi_pickup_dt   ON taxi_trips(pickup_dt);
CREATE INDEX IF NOT EXISTS idx_taxi_pickup_zone ON taxi_trips(pickup_zone);
CREATE INDEX IF NOT EXISTS idx_taxi_dt_zone     ON taxi_trips(pickup_dt, pickup_zone);
CREATE INDEX IF NOT EXISTS idx_weather_dt       ON weather_hourly(obs_dt);
CREATE INDEX IF NOT EXISTS idx_events_date      ON events(event_date);
