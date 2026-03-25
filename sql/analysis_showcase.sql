-- ============================================================
-- UrbanPulse — SQL Analysis Showcase
-- 10 progressively complex queries demonstrating SQL proficiency
-- Run in: notebooks/02_eda_sql_analysis.ipynb
-- ============================================================


-- ── Query 1: Hourly demand overview ──────────────────────────────────────────
-- Basic aggregation + ordering — foundation for all subsequent analysis
SELECT
    date_trunc('hour', pickup_dt)           AS hour_bucket,
    pickup_zone,
    COUNT(*)                                 AS trip_count,
    ROUND(AVG(fare_amount)::NUMERIC, 2)      AS avg_fare,
    ROUND(AVG(trip_distance)::NUMERIC, 2)    AS avg_distance_mi,
    ROUND(SUM(total_amount)::NUMERIC, 2)     AS total_revenue
FROM taxi_trips
GROUP BY 1, 2
ORDER BY 1, trip_count DESC;


-- ── Query 2: Weather impact on demand ────────────────────────────────────────
-- Multi-table JOIN + CASE classification + PERCENTILE_CONT
SELECT
    CASE
        WHEN w.weather_code IN (0, 1)              THEN 'Clear'
        WHEN w.weather_code IN (2, 3)              THEN 'Partly Cloudy'
        WHEN w.weather_code BETWEEN 51 AND 55      THEN 'Drizzle'
        WHEN w.weather_code BETWEEN 61 AND 67      THEN 'Rain'
        WHEN w.weather_code BETWEEN 71 AND 77      THEN 'Snow'
        WHEN w.weather_code BETWEEN 95 AND 99      THEN 'Thunderstorm'
        ELSE 'Other'
    END                                             AS weather_category,
    COUNT(t.trip_id)                                AS total_trips,
    ROUND(AVG(t.fare_amount)::NUMERIC, 2)           AS avg_fare,
    ROUND(AVG(t.trip_distance)::NUMERIC, 2)         AS avg_distance,
    PERCENTILE_CONT(0.5)
        WITHIN GROUP (ORDER BY t.fare_amount)       AS median_fare,
    ROUND(100.0 * COUNT(t.trip_id)
          / SUM(COUNT(t.trip_id)) OVER ()
          , 2)                                      AS pct_of_all_trips
FROM taxi_trips t
JOIN weather_hourly w
    ON date_trunc('hour', t.pickup_dt) = w.obs_dt
GROUP BY weather_category
ORDER BY total_trips DESC;


-- ── Query 3: Event-driven demand deviation ────────────────────────────────────
-- CTE + window functions + LEFT JOIN + date arithmetic
WITH daily_demand AS (
    SELECT
        DATE(pickup_dt)   AS trip_date,
        COUNT(*)           AS daily_trips
    FROM taxi_trips
    GROUP BY 1
),
windowed AS (
    SELECT
        trip_date,
        daily_trips,
        ROUND(
            AVG(daily_trips) OVER (
                ORDER BY trip_date
                ROWS BETWEEN 7 PRECEDING AND 7 FOLLOWING
            )::NUMERIC, 0
        ) AS rolling_avg_14d
    FROM daily_demand
)
SELECT
    w.trip_date,
    w.daily_trips,
    w.rolling_avg_14d,
    e.event_name,
    e.event_type,
    e.est_attendance,
    ROUND(
        (w.daily_trips - w.rolling_avg_14d)
        / NULLIF(w.rolling_avg_14d, 0) * 100
    , 1)                                AS pct_deviation_from_baseline
FROM windowed w
LEFT JOIN events e ON w.trip_date = e.event_date
WHERE e.event_name IS NOT NULL
ORDER BY pct_deviation_from_baseline DESC
LIMIT 25;


-- ── Query 4: Zone performance ranking ─────────────────────────────────────────
-- Window function ranking (DENSE_RANK), JOIN with zone lookup
SELECT
    z.zone_name,
    z.borough,
    COUNT(t.trip_id)                                AS trip_count,
    ROUND(SUM(t.total_amount)::NUMERIC, 2)          AS total_revenue,
    ROUND(AVG(t.fare_amount)::NUMERIC, 2)           AS avg_fare,
    ROUND(AVG(t.trip_distance)::NUMERIC, 2)         AS avg_distance,
    DENSE_RANK() OVER (ORDER BY COUNT(t.trip_id) DESC)       AS demand_rank,
    DENSE_RANK() OVER (ORDER BY SUM(t.total_amount) DESC)    AS revenue_rank
FROM taxi_trips t
JOIN taxi_zones z ON t.pickup_zone = z.zone_id
GROUP BY z.zone_name, z.borough
ORDER BY demand_rank
LIMIT 20;


-- ── Query 5: Hour × Day-of-Week heatmap ───────────────────────────────────────
-- Basis for the heatmap visualization in the EDA notebook
SELECT
    EXTRACT(DOW FROM pickup_dt)::INTEGER    AS day_of_week,
    TO_CHAR(pickup_dt, 'Dy')               AS day_name,
    EXTRACT(HOUR FROM pickup_dt)::INTEGER   AS hour_of_day,
    COUNT(*)                                AS trip_count,
    ROUND(AVG(fare_amount)::NUMERIC, 2)     AS avg_fare,
    ROUND(AVG(trip_distance)::NUMERIC, 2)   AS avg_distance
FROM taxi_trips
GROUP BY 1, 2, 3
ORDER BY 1, 3;


-- ── Query 6: Event before/during/after demand with LAG/LEAD ──────────────────
-- Advanced window functions: LAG, LEAD + percent-change calculation
WITH event_days AS (
    SELECT event_date, event_name, est_attendance
    FROM events
    WHERE is_major = TRUE
),
daily AS (
    SELECT DATE(pickup_dt) AS trip_date, COUNT(*) AS trips
    FROM taxi_trips
    GROUP BY 1
)
SELECT
    e.event_name,
    e.est_attendance,
    LAG(d.trips, 1)  OVER (ORDER BY e.event_date)  AS trips_day_before,
    d.trips                                          AS trips_event_day,
    LEAD(d.trips, 1) OVER (ORDER BY e.event_date)  AS trips_day_after,
    ROUND(
        (d.trips - LAG(d.trips, 1) OVER (ORDER BY e.event_date))::NUMERIC
        / NULLIF(LAG(d.trips, 1) OVER (ORDER BY e.event_date), 0) * 100
    , 1)                                             AS pct_change_vs_prior_day,
    ROUND(
        (LEAD(d.trips, 1) OVER (ORDER BY e.event_date) - d.trips)::NUMERIC
        / NULLIF(d.trips, 0) * 100
    , 1)                                             AS pct_change_next_day
FROM event_days e
JOIN daily d ON d.trip_date = e.event_date
ORDER BY e.event_date;


-- ── Query 7: Precipitation threshold analysis ─────────────────────────────────
-- CASE bucketing, conditional aggregation, multiple GROUP BY
SELECT
    CASE
        WHEN w.precipitation_mm = 0             THEN '0 mm (Dry)'
        WHEN w.precipitation_mm < 2             THEN '0.1–2 mm (Light)'
        WHEN w.precipitation_mm < 10            THEN '2–10 mm (Moderate)'
        ELSE '10+ mm (Heavy)'
    END                                          AS precipitation_bucket,
    COUNT(t.trip_id)                             AS total_trips,
    ROUND(AVG(t.fare_amount)::NUMERIC, 2)        AS avg_fare,
    ROUND(
        100.0 * COUNT(t.trip_id)
        / SUM(COUNT(t.trip_id)) OVER ()
    , 2)                                         AS pct_of_trips,
    ROUND(AVG(t.trip_distance)::NUMERIC, 2)      AS avg_distance
FROM taxi_trips t
JOIN weather_hourly w
    ON date_trunc('hour', t.pickup_dt) = w.obs_dt
GROUP BY precipitation_bucket
ORDER BY MIN(w.precipitation_mm);


-- ── Query 8: Running total and month-over-month growth ───────────────────────
-- SUM() OVER, LAG across months, growth calculation
WITH monthly AS (
    SELECT
        date_trunc('month', pickup_dt)          AS month,
        COUNT(*)                                 AS monthly_trips,
        ROUND(SUM(total_amount)::NUMERIC, 2)     AS monthly_revenue
    FROM taxi_trips
    GROUP BY 1
)
SELECT
    month,
    monthly_trips,
    monthly_revenue,
    SUM(monthly_trips)    OVER (ORDER BY month) AS cumulative_trips,
    SUM(monthly_revenue)  OVER (ORDER BY month) AS cumulative_revenue,
    LAG(monthly_trips)    OVER (ORDER BY month) AS prev_month_trips,
    ROUND(
        (monthly_trips - LAG(monthly_trips) OVER (ORDER BY month))::NUMERIC
        / NULLIF(LAG(monthly_trips) OVER (ORDER BY month), 0) * 100
    , 1)                                         AS mom_trip_growth_pct
FROM monthly
ORDER BY month;


-- ── Query 9: Top origin-destination pairs ─────────────────────────────────────
-- Self-join on zone lookup, aggregate flow analysis
SELECT
    pz.zone_name        AS pickup_zone_name,
    dz.zone_name        AS dropoff_zone_name,
    pz.borough          AS pickup_borough,
    dz.borough          AS dropoff_borough,
    COUNT(*)            AS trip_count,
    ROUND(AVG(t.trip_distance)::NUMERIC, 2) AS avg_distance,
    ROUND(AVG(t.fare_amount)::NUMERIC, 2)   AS avg_fare
FROM taxi_trips t
JOIN taxi_zones pz ON t.pickup_zone  = pz.zone_id
JOIN taxi_zones dz ON t.dropoff_zone = dz.zone_id
WHERE t.pickup_zone != t.dropoff_zone
GROUP BY pz.zone_name, dz.zone_name, pz.borough, dz.borough
HAVING COUNT(*) > 100
ORDER BY trip_count DESC
LIMIT 30;


-- ── Query 10: Full multi-source feature join ──────────────────────────────────
-- 4-table join + multiple aggregations — mirrors the materialized view logic
SELECT
    date_trunc('hour', t.pickup_dt)             AS hour_bucket,
    t.pickup_zone                                AS zone_id,
    z.zone_name,
    z.borough,
    COUNT(t.trip_id)                             AS trip_count,
    ROUND(SUM(t.total_amount)::NUMERIC, 2)       AS revenue,
    w.temperature_f,
    w.precipitation_mm,
    COALESCE(e.event_name, 'None')               AS event_name,
    COALESCE(e.is_major::TEXT, 'false')          AS is_major_event,
    ec.unemployment_rate,
    ec.gas_price_avg
FROM taxi_trips t
LEFT JOIN taxi_zones z         ON t.pickup_zone = z.zone_id
LEFT JOIN weather_hourly w     ON date_trunc('hour', t.pickup_dt) = w.obs_dt
LEFT JOIN events e             ON DATE(t.pickup_dt) = e.event_date
LEFT JOIN economic_indicators ec
    ON date_trunc('month', t.pickup_dt) = ec.month_start
GROUP BY
    date_trunc('hour', t.pickup_dt), t.pickup_zone,
    z.zone_name, z.borough,
    w.temperature_f, w.precipitation_mm,
    e.event_name, e.is_major,
    ec.unemployment_rate, ec.gas_price_avg
ORDER BY hour_bucket, trip_count DESC;
