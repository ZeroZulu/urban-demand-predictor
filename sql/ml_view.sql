-- ============================================================
-- UrbanPulse — Materialized View for ML Feature Set
-- Run AFTER loading all raw tables.
-- Refresh with: REFRESH MATERIALIZED VIEW ml_features;
-- ============================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS ml_features AS
SELECT
    date_trunc('hour', t.pickup_dt)             AS hour_bucket,
    t.pickup_zone                               AS zone_id,
    COUNT(*)                                    AS trip_count,

    -- ── Time features ────────────────────────────────────────
    EXTRACT(HOUR  FROM t.pickup_dt)::INTEGER    AS hour_of_day,
    EXTRACT(DOW   FROM t.pickup_dt)::INTEGER    AS day_of_week,
    EXTRACT(MONTH FROM t.pickup_dt)::INTEGER    AS month,
    CASE WHEN EXTRACT(DOW FROM t.pickup_dt) IN (0, 6)
         THEN 1 ELSE 0 END                      AS is_weekend,

    -- ── Weather features ─────────────────────────────────────
    w.temperature_f,
    w.feels_like_f,
    w.precipitation_mm,
    w.windspeed_mph,
    w.humidity_pct,
    w.weather_code,
    w.is_raining::INTEGER                       AS is_raining,
    w.is_snowing::INTEGER                       AS is_snowing,

    -- ── Event features ───────────────────────────────────────
    COALESCE(e.is_major::INTEGER, 0)            AS has_major_event,
    COALESCE(e.est_attendance, 0)               AS event_attendance,

    -- ── Economic features ────────────────────────────────────
    ec.unemployment_rate,
    ec.gas_price_avg,
    ec.consumer_sentiment,

    -- ── Lag features (partial — enhanced further in Python) ──
    LAG(COUNT(*), 24) OVER (
        PARTITION BY t.pickup_zone
        ORDER BY date_trunc('hour', t.pickup_dt)
    )                                           AS demand_lag_24h,

    LAG(COUNT(*), 168) OVER (
        PARTITION BY t.pickup_zone
        ORDER BY date_trunc('hour', t.pickup_dt)
    )                                           AS demand_lag_168h

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
    w.temperature_f, w.feels_like_f, w.precipitation_mm,
    w.windspeed_mph, w.humidity_pct, w.weather_code,
    w.is_raining, w.is_snowing,
    e.is_major, e.est_attendance,
    ec.unemployment_rate, ec.gas_price_avg, ec.consumer_sentiment;

CREATE INDEX IF NOT EXISTS idx_ml_features_zone_dt
    ON ml_features(zone_id, hour_bucket);
