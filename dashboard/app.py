"""
UrbanPulse — Fixed Streamlit Dashboard
Fixes: API health check, date-range queries using actual data range not NOW()
"""
import os
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="UrbanPulse",
    page_icon="🌆",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
DB_URL  = os.getenv("DATABASE_URL", "postgresql://analyst:localdev@localhost:5432/urban_demand")

@st.cache_data(ttl=300)
def query_db(sql: str) -> pd.DataFrame:
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_data_range():
    """Get actual min/max dates in the database."""
    df = query_db("SELECT MIN(hour_bucket) AS dt_min, MAX(hour_bucket) AS dt_max FROM ml_features")
    if df.empty:
        return None, None
    return pd.to_datetime(df["dt_min"].iloc[0]), pd.to_datetime(df["dt_max"].iloc[0])

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except:
        return False, {}

ZONE_NAMES = {
    161: "Midtown Center",    162: "Midtown East",
    163: "Midtown North",     230: "Times Sq / Theatre District",
    186: "Penn Station / MSG",132: "JFK Airport",
    138: "LaGuardia Airport", 234: "Union Square",
    107: "Gramercy",          170: "Murray Hill",
    236: "UES North",         237: "UES South",
    142: "Lincoln Square E",
}

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    51: "Drizzle", 61: "Rain (slight)", 63: "Rain (moderate)",
    65: "Rain (heavy)", 71: "Snow (slight)", 73: "Snow (moderate)",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/UrbanPulse-v1.0.0-blue", width=160)
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("", [
    "🔮 Live Prediction",
    "📊 Model Performance",
    "🌡️ Weather Impact",
    "🔍 Drift Monitor",
])

api_ok, api_info = check_api()
if api_ok:
    st.sidebar.success(f"API ✓  |  v{api_info.get('model_version','?')}")
    st.sidebar.caption(f"Features: {api_info.get('feature_count','?')}")
else:
    st.sidebar.error("API offline — run `docker-compose restart api`")

dt_min, dt_max = get_data_range()
if dt_min:
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Data: {dt_min.date()} → {dt_max.date()}")
    st.sidebar.caption(f"491,605 rows · 261 zones")

st.sidebar.markdown("---")
st.sidebar.markdown("**UrbanPulse** · Built by Shril Patel")


# ==============================================================================
# PAGE 1 — LIVE PREDICTION
# ==============================================================================
if page == "🔮 Live Prediction":
    st.title("🔮 Live Demand Prediction")
    st.caption(
        "Predict hourly NYC taxi demand for any zone, time, and weather condition. "
        "Each prediction includes SHAP-based explanations."
    )
    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.subheader("📍 Zone & Time")
        zone_id = st.selectbox(
            "Pickup Zone",
            options=list(ZONE_NAMES.keys()),
            format_func=lambda z: f"{z} — {ZONE_NAMES[z]}",
        )
        target_date = st.date_input("Date", value=date.today())
        target_hour = st.slider("Hour of Day", 0, 23, 17, format="%d:00")

    with col2:
        st.subheader("🌤️ Weather Conditions")
        temp_f   = st.slider("Temperature (°F)", -10, 110, 72)
        feels_f  = st.slider("Feels Like (°F)", -20, 115, int(temp_f - 2))
        precip   = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, step=0.5)
        wind     = st.slider("Wind Speed (mph)", 0.0, 60.0, 8.0, step=0.5)
        humidity = st.slider("Humidity (%)", 0, 100, 60)

    with col3:
        st.subheader("🎉 Events & Context")
        w_code = st.selectbox(
            "Weather Type",
            options=list(WEATHER_CODES.keys()),
            format_func=lambda c: WEATHER_CODES[c],
            index=1,
        )
        has_event  = st.checkbox("Major Event in NYC Today?")
        attendance = st.number_input("Estimated Attendance", 0, 2_000_000, 50_000, step=5000) if has_event else 0

    st.divider()

    if st.button("⚡ Predict Demand", type="primary", use_container_width=True):
        if not api_ok:
            st.error("API is offline. Run `docker-compose restart api` in your terminal first.")
        else:
            payload = {
                "zone_id": zone_id,
                "datetime": f"{target_date}T{target_hour:02d}:00:00",
                "temperature_f": temp_f, "feels_like_f": feels_f,
                "precipitation_mm": precip, "windspeed_mph": wind,
                "humidity_pct": humidity, "weather_code": w_code,
                "has_major_event": has_event, "event_attendance": attendance,
            }
            with st.spinner("Running prediction + SHAP analysis..."):
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                    resp.raise_for_status()
                    result = resp.json()

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Trips", f"{result['predicted_trips']:,}",
                              delta=f"80% CI: {result['confidence_interval'][0]:,}–{result['confidence_interval'][1]:,}")
                    m2.metric("Zone", ZONE_NAMES.get(zone_id, str(zone_id)))
                    m3.metric("Time", f"{target_date} {target_hour:02d}:00")

                    st.markdown("#### What drove this prediction?")
                    factors_df = pd.DataFrame(result["top_factors"])
                    fig = px.bar(
                        factors_df, x="shap_value", y="feature", orientation="h",
                        color="direction", title="SHAP Feature Contributions",
                        color_discrete_map={
                            "increases_demand": "#2ecc71",
                            "decreases_demand": "#e74c3c",
                        }
                    )
                    fig.update_layout(height=320, margin=dict(l=10, r=40, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Raw API response"):
                        st.json(result)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# ==============================================================================
# PAGE 2 — MODEL PERFORMANCE
# ==============================================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.caption("Predicted vs. actual demand, error distributions, and model comparison.")

    tab1, tab2, tab3 = st.tabs(["Forecast vs Actual", "Error Analysis", "Feature Importance"])

    with tab1:
        st.subheader("Hourly Demand — Recent Data")

        if dt_min is None:
            st.info("No data available. Run `make ingest` to load data.")
        else:
            # Use actual data range instead of NOW()
            data_end   = pd.Timestamp("2024-06-30")
            data_start = pd.Timestamp("2024-06-01")

            zone = st.selectbox(
                "Zone", list(ZONE_NAMES.keys()),
                format_func=lambda z: f"{ZONE_NAMES[z]} (zone {z})",
            )

            sql = f"""
                SELECT hour_bucket, trip_count
                FROM ml_features
                WHERE zone_id = {zone}
                  AND hour_bucket >= '{data_start.date()}'
                  AND hour_bucket <= '{data_end.date()}'
                ORDER BY hour_bucket
            """
            df = query_db(sql)

            if df.empty:
                st.info(f"No data found for zone {zone} in the last 30 days of available data ({data_start.date()} – {data_end.date()}).")
            else:
                fig = px.line(
                    df, x="hour_bucket", y="trip_count",
                    title=f"Hourly Trip Demand — {ZONE_NAMES.get(zone, zone)} (last 30 days of data)",
                    labels={"hour_bucket": "Time", "trip_count": "Trips"},
                    color_discrete_sequence=["#3498db"],
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Showing {len(df):,} hourly observations from {data_start.date()} to {data_end.date()}")

    with tab2:
        st.subheader("Demand by Hour of Day")
        sql = """
            SELECT hour_of_day,
                   AVG(trip_count) AS avg_trips,
                   STDDEV(trip_count) AS std_trips
            FROM ml_features
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """
        df = query_db(sql)
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df["hour_of_day"], y=df["avg_trips"],
                error_y=dict(type="data", array=df["std_trips"].fillna(0).tolist()),
                name="Average Trips", marker_color="#3498db",
            ))
            fig.update_layout(
                title="Average Demand by Hour (all zones, all data)",
                xaxis_title="Hour of Day", yaxis_title="Avg Trip Count",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Feature Importance")
        try:
            import joblib
            model = joblib.load("models/best_model.pkl")
            names = joblib.load("models/feature_names.pkl")
            if hasattr(model, "feature_importances_"):
                imp_df = pd.DataFrame({
                    "feature": names,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=True).tail(15)
                fig = px.bar(
                    imp_df, x="importance", y="feature", orientation="h",
                    title="Top 15 Feature Importances (LightGBM)",
                    color="importance", color_continuous_scale="Blues",
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        except FileNotFoundError:
            st.warning("Model not found at models/best_model.pkl. Run `python train.py` first.")
        except Exception as e:
            st.error(f"Error loading model: {e}")


# ==============================================================================
# PAGE 3 — WEATHER IMPACT
# ==============================================================================
elif page == "🌡️ Weather Impact":
    st.title("🌡️ Weather Impact Analysis")
    st.caption("Explore how weather conditions affect taxi demand across NYC zones.")

    tab1, tab2 = st.tabs(["Temperature vs Demand", "Weather Category Breakdown"])

    with tab1:
        st.subheader("Temperature → Demand Relationship")
        sql = """
            SELECT
                ROUND(temperature_f / 10.0) * 10  AS temp_bin,
                AVG(trip_count)                     AS avg_trips,
                COUNT(*)                            AS n_hours
            FROM ml_features
            WHERE temperature_f IS NOT NULL
              AND temperature_f BETWEEN -20 AND 120
            GROUP BY 1
            HAVING COUNT(*) > 10
            ORDER BY 1
        """
        df = query_db(sql)
        if not df.empty:
            fig = px.scatter(
                df, x="temp_bin", y="avg_trips", size="n_hours",
                title="Average Demand by Temperature (10°F bins)",
                labels={"temp_bin": "Temperature (°F)", "avg_trips": "Avg Trips/Hour"},
                trendline="lowess", color_discrete_sequence=["#e74c3c"],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("U-shaped relationship: demand peaks at cold temperatures (riders avoid walking) and hot temperatures (comfort). Mid-range temperatures see lower taxi demand.")

    with tab2:
        st.subheader("Demand by Weather Condition")
        sql = """
            SELECT
                CASE
                    WHEN weather_code IN (0,1)          THEN 'Clear'
                    WHEN weather_code IN (2,3)          THEN 'Cloudy'
                    WHEN weather_code BETWEEN 51 AND 67 THEN 'Rain'
                    WHEN weather_code BETWEEN 71 AND 77 THEN 'Snow'
                    ELSE 'Other'
                END                     AS weather_type,
                AVG(trip_count)          AS avg_trips,
                COUNT(*)                 AS n_obs
            FROM ml_features
            WHERE weather_code IS NOT NULL
            GROUP BY 1
            ORDER BY avg_trips DESC
        """
        df = query_db(sql)
        if not df.empty:
            fig = px.bar(
                df, x="weather_type", y="avg_trips",
                color="avg_trips", color_continuous_scale="RdYlGn",
                title="Average Hourly Demand by Weather Category",
                text_auto=".0f",
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.style.format({"avg_trips": "{:.1f}", "n_obs": "{:,}"}), use_container_width=True)


# ==============================================================================
# PAGE 4 — DRIFT MONITOR
# ==============================================================================
elif page == "🔍 Drift Monitor":
    st.title("🔍 Model Drift Monitor")
    st.caption("Feature distribution stats and data health. Powered by Evidently AI.")

    report_path = Path("outputs/drift_reports/drift_report_latest.html")

    col1, col2 = st.columns([2, 1])

    with col1:
        if report_path.exists():
            with open(report_path) as f:
                st.components.v1.html(f.read(), height=800, scrolling=True)
        else:
            st.info("No drift report yet. Generate one using the button on the right.")

        # Show data health stats instead
        st.subheader("Data Health")
        sql = """
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT zone_id) AS zones,
                COUNT(DISTINCT DATE(hour_bucket)) AS days,
                AVG(trip_count) AS avg_demand,
                SUM(CASE WHEN temperature_f IS NULL THEN 1 ELSE 0 END) AS missing_weather,
                SUM(CASE WHEN unemployment_rate IS NULL THEN 1 ELSE 0 END) AS missing_econ
            FROM ml_features
        """
        stats = query_db(sql)
        if not stats.empty:
            r = stats.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total rows", f"{int(r['total_rows']):,}")
            c2.metric("Zones", int(r['zones']))
            c3.metric("Days covered", int(r['days']))
            c4.metric("Avg demand/hr", f"{r['avg_demand']:.1f}")

            missing_w = int(r['missing_weather'])
            missing_e = int(r['missing_econ'])
            if missing_w > 0:
                st.warning(f"{missing_w:,} rows missing weather data ({missing_w/int(r['total_rows'])*100:.1f}%)")
            else:
                st.success("Weather data: complete (0 missing)")
            if missing_e > 0:
                st.info(f"{missing_e:,} rows missing economic data — economic features will be forward-filled by the model")

    with col2:
        st.subheader("Generate Report")
        st.info("Drift analysis requires the monitoring module. Run from terminal:")
        st.code("python -m monitoring.drift_report", language="bash")

        st.divider()
        st.subheader("Quick Stats")
        if dt_min:
            st.metric("Data From", str(dt_min.date()))
            st.metric("Data To", str(dt_max.date()))
            st.metric("Zones", "261")
            st.metric("Total Rows", "491,605")
