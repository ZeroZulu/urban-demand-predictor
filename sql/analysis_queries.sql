-- ============================================================
-- UrbanPulse — SQL Analysis Showcase
-- 10 progressively complex queries demonstrating SQL proficiency
-- ============================================================


-- ── Query 1: Hourly demand by zone ───────────────────────────
-- Basic aggregation with GROUP BY and ORDER BY
SELECT
    date_trunc('hour', pickup_dt)        AS hour_bucket,
    pickup_zone,
    COUNT(*)                             AS trip_count,
    ROUND(AVG(fare_amount)::NUMERIC, 2)  AS avg_fare,
    ROUND(AVG(trip_distance)::NUMERIC, 2) AS avg_distance
FROM taxi_trips
GROUP BY 1, 2
ORDER BY 1, trip_count DESC;


-- ── Query 2: Weather impact on demand ────────────────────────
-- Multi-table JOIN, CASE expression, PERCENTILE_CONT
SELECT
    CASE
        WHEN w.weather_code IN (0, 1)            THEN 'Clear'
        WHEN w.weather_code IN (2, 3)            THEN 'Partly Cloudy'
        WHEN w.weather_code BETWEEN 51 AND 67   THEN 'Rain'
        WHEN w.weather_code BETWEEN 71 AND 77   THEN 'Snow'
        ELSE 'Other'
    END                                           AS weather_category,
    COUNT(t.trip_id)                              AS total_trips,
    ROUND(AVG(t.fare_amount)::NUMERIC, 2)         AS avg_fare,
    PERCENTILE_CONT(0.5) WITHIN GROUP
        (ORDER BY t.trip_distance)                AS median_distance_miles,
    ROUND(
        100.0 * COUNT(t.trip_id) /
        SUM(COUNT(t.trip_id)) OVER (), 2
    )                                             AS pct_of_all_trips
FROM taxi_trips t
JOIN weather_hourly w
    ON date_trunc('hour', t.pickup_dt) = w.obs_dt
GROUP BY 1
ORDER BY total_trips DESC;


-- ── Query 3: Event impact vs rolling baseline ─────────────────
-- CTE, window functions, LEFT JOIN, date arithmetic
WITH daily_demand AS (
    SELECT
        DATE(pickup_dt) AS trip_date,
        COUNT(*)        AS daily_trips
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
    e.est_attendance,
    ROUND(
        (w.daily_trips - w.rolling_avg_14d)
        / NULLIF(w.rolling_avg_14d, 0) * 100, 1
    ) AS pct_deviation_from_baseline
FROM windowed w
LEFT JOIN events e ON w.trip_date = e.event_date
WHERE e.event_name IS NOT NULL
ORDER BY pct_deviation_from_baseline DESC
LIMIT 20;


-- ── Query 4: Zone revenue leaderboard ─────────────────────────
-- DENSE_RANK, multiple window functions, JOIN
SELECT
    z.zone_name,
    z.borough,
    COUNT(t.trip_id)                            AS trip_count,
    ROUND(SUM(t.total_amount)::NUMERIC, 2)      AS total_revenue,
    ROUND(AVG(t.fare_amount)::NUMERIC, 2)       AS avg_fare,
    DENSE_RANK() OVER (ORDER BY COUNT(t.trip_id) DESC)        AS demand_rank,
    DENSE_RANK() OVER (ORDER BY SUM(t.total_amount) DESC)     AS revenue_rank
FROM taxi_trips t
JOIN taxi_zones z ON t.pickup_zone = z.zone_id
GROUP BY z.zone_name, z.borough
ORDER BY demand_rank
LIMIT 30;


-- ── Query 5: Hour-of-day × day-of-week demand heatmap ─────────
-- EXTRACT, GROUP BY multiple time parts
SELECT
    EXTRACT(DOW  FROM pickup_dt)   AS day_of_week,   -- 0=Sun
    EXTRACT(HOUR FROM pickup_dt)   AS hour_of_day,
    COUNT(*)                       AS avg_trips,
    ROUND(AVG(fare_amount)::NUMERIC, 2) AS avg_fare
FROM taxi_trips
GROUP BY 1, 2
ORDER BY 1, 2;


-- ── Query 6: Before / during / after major events ─────────────
-- LAG, LEAD, nested CTE, complex CASE
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
        (d.trips::FLOAT - LAG(d.trips, 1) OVER (ORDER BY e.event_date))
        / NULLIF(LAG(d.trips, 1) OVER (ORDER BY e.event_date), 0) * 100
    , 1)                                             AS pct_change_vs_prior_day
FROM event_days e
JOIN daily d ON d.trip_date = e.event_date
ORDER BY e.event_date;


-- ── Query 7: Monthly revenue with month-over-month growth ─────
-- DATE_TRUNC, LAG for period comparison
WITH monthly AS (
    SELECT
        date_trunc('month', pickup_dt)         AS month,
        COUNT(*)                               AS trips,
        ROUND(SUM(total_amount)::NUMERIC, 2)   AS revenue,
        ROUND(AVG(fare_amount)::NUMERIC, 2)    AS avg_fare
    FROM taxi_trips
    GROUP BY 1
)
SELECT
    month,
    trips,
    revenue,
    avg_fare,
    LAG(revenue) OVER (ORDER BY month)        AS prev_month_revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY month))
        / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) * 100
    , 1)                                      AS revenue_mom_growth_pct
FROM monthly
ORDER BY month;


-- ── Query 8: Top zones during rain vs. clear weather ──────────
-- Conditional aggregation (FILTER clause), subquery
SELECT
    z.zone_name,
    z.borough,
    COUNT(t.trip_id) FILTER (WHERE w.is_raining)     AS rainy_trips,
    COUNT(t.trip_id) FILTER (WHERE NOT w.is_raining) AS clear_trips,
    ROUND(
        100.0 * COUNT(t.trip_id) FILTER (WHERE w.is_raining)
        / NULLIF(COUNT(t.trip_id), 0), 1
    )                                                 AS rainy_trip_pct
FROM taxi_trips t
JOIN weather_hourly w ON date_trunc('hour', t.pickup_dt) = w.obs_dt
JOIN taxi_zones z     ON t.pickup_zone = z.zone_id
GROUP BY z.zone_name, z.borough
HAVING COUNT(t.trip_id) > 1000
ORDER BY rainy_trip_pct DESC
LIMIT 20;


-- ── Query 9: Running total of trips per zone ─────────────────
-- SUM window function (cumulative), ROW_NUMBER
SELECT
    date_trunc('week', pickup_dt)  AS week_start,
    pickup_zone,
    COUNT(*)                       AS weekly_trips,
    SUM(COUNT(*)) OVER (
        PARTITION BY pickup_zone
        ORDER BY date_trunc('week', pickup_dt)
    )                              AS cumulative_trips,
    ROW_NUMBER() OVER (
        PARTITION BY pickup_zone
        ORDER BY date_trunc('week', pickup_dt)
    )                              AS week_number
FROM taxi_trips
GROUP BY 1, 2
ORDER BY pickup_zone, week_start;


-- ── Query 10: Multi-source join — full context per hour ───────
-- 4-table join, coalesce, complex aggregations
SELECT
    date_trunc('hour', t.pickup_dt)               AS hour_bucket,
    COUNT(t.trip_id)                              AS trip_count,
    ROUND(AVG(t.fare_amount)::NUMERIC, 2)         AS avg_fare,
    w.temperature_f,
    w.precipitation_mm,
    w.windspeed_mph,
    CASE WHEN w.is_raining THEN 'Rain'
         WHEN w.is_snowing THEN 'Snow'
         ELSE 'Dry' END                           AS weather_type,
    COALESCE(e.event_name, 'None')                AS event,
    COALESCE(e.est_attendance, 0)                 AS event_attendance,
    ec.unemployment_rate,
    ec.gas_price_avg
FROM taxi_trips t
LEFT JOIN weather_hourly w
    ON date_trunc('hour', t.pickup_dt) = w.obs_dt
LEFT JOIN events e
    ON DATE(t.pickup_dt) = e.event_date
LEFT JOIN economic_indicators ec
    ON date_trunc('month', t.pickup_dt) = ec.month_start
GROUP BY 1, w.temperature_f, w.precipitation_mm, w.windspeed_mph,
         w.is_raining, w.is_snowing, e.event_name, e.est_attendance,
         ec.unemployment_rate, ec.gas_price_avg
ORDER BY 1;
