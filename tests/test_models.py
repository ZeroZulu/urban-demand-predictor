"""
Unit tests for model training, evaluation, and inference utilities.
Uses small synthetic datasets — no database or trained model required.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.models.evaluator import comparison_table, evaluate, mape


# ── Evaluator tests ────────────────────────────────────────────────────────────

class TestEvaluator:

    def test_perfect_predictions(self):
        y = np.array([100.0, 200.0, 300.0])
        metrics = evaluate(y, y)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"]  == pytest.approx(0.0)
        assert metrics["mape"] == pytest.approx(0.0)
        assert metrics["r2"]   == pytest.approx(1.0)

    def test_metrics_keys(self):
        y_true = np.array([100.0, 150.0, 200.0])
        y_pred = np.array([110.0, 145.0, 195.0])
        metrics = evaluate(y_true, y_pred)
        assert set(metrics.keys()) == {"rmse", "mae", "mape", "r2"}

    def test_metrics_are_floats(self):
        y = np.array([50.0, 100.0, 150.0])
        p = np.array([55.0,  95.0, 160.0])
        metrics = evaluate(y, p)
        for k, v in metrics.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

    def test_rmse_greater_than_mae(self):
        """RMSE >= MAE always (Jensen's inequality)."""
        y = np.array([100.0, 200.0, 300.0, 400.0])
        p = np.array([120.0, 180.0, 350.0, 390.0])
        m = evaluate(y, p)
        assert m["rmse"] >= m["mae"]

    def test_mape_ignores_zero_actuals(self):
        """MAPE denominator is zero when y_true = 0 — should not crash."""
        y = np.array([0.0, 100.0, 200.0])
        p = np.array([10.0, 110.0, 190.0])
        result = mape(y, p)
        assert np.isfinite(result)

    def test_mape_all_zeros_returns_nan(self):
        y = np.array([0.0, 0.0, 0.0])
        p = np.array([1.0, 2.0, 3.0])
        result = mape(y, p)
        assert np.isnan(result)

    def test_r2_below_zero_for_bad_model(self):
        """A model worse than the mean should have negative R²."""
        y = np.array([10.0, 20.0, 30.0, 40.0])
        # Predictions go in opposite direction
        p = np.array([40.0, 30.0, 20.0, 10.0])
        m = evaluate(y, p)
        assert m["r2"] < 0

    def test_comparison_table_returns_dataframe(self):
        results = {
            "ModelA": {"rmse": 10.0, "mae": 7.0, "mape": 5.0, "r2": 0.92},
            "ModelB": {"rmse": 15.0, "mae": 10.0, "mape": 8.0, "r2": 0.87},
        }
        df = comparison_table(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_comparison_table_sorted_by_rmse(self):
        results = {
            "Worse":  {"rmse": 20.0, "mae": 14.0, "mape": 10.0, "r2": 0.80},
            "Better": {"rmse": 10.0, "mae":  7.0, "mape":  5.0, "r2": 0.92},
        }
        df = comparison_table(results)
        assert df.index[0] == "Better"


# ── Temporal split tests ───────────────────────────────────────────────────────

class TestTemporalSplit:

    def _make_df(self, n_months: int = 6) -> pd.DataFrame:
        """Build a synthetic ml_features-like DataFrame spanning n_months."""
        dates = pd.date_range("2024-01-01", periods=n_months * 30 * 24, freq="h")
        return pd.DataFrame({
            "hour_bucket": dates,
            "zone_id":     np.random.randint(1, 10, len(dates)),
            "trip_count":  np.random.randint(50, 400, len(dates)),
        })

    def test_no_data_leakage(self):
        from src.models.trainer import temporal_split
        df = self._make_df()
        train, test = temporal_split(df, test_months=2)
        assert train["hour_bucket"].max() < test["hour_bucket"].min()

    def test_split_sizes_sum_to_total(self):
        from src.models.trainer import temporal_split
        df = self._make_df()
        train, test = temporal_split(df, test_months=2)
        assert len(train) + len(test) == len(df)

    def test_test_set_is_last_n_months(self):
        from src.models.trainer import temporal_split
        df = self._make_df(n_months=6)
        _, test = temporal_split(df, test_months=2)
        # Test set should cover ~2 months out of 6
        test_frac = len(test) / len(df)
        assert 0.28 <= test_frac <= 0.38, f"Test fraction {test_frac:.2f} outside expected range"

    def test_train_is_not_empty(self):
        from src.models.trainer import temporal_split
        df = self._make_df()
        train, _ = temporal_split(df, test_months=1)
        assert len(train) > 0

    def test_test_is_not_empty(self):
        from src.models.trainer import temporal_split
        df = self._make_df()
        _, test = temporal_split(df, test_months=1)
        assert len(test) > 0


# ── End-to-end mini training test ──────────────────────────────────────────────

class TestMiniTraining:
    """
    End-to-end training on a tiny synthetic dataset.
    Validates the full build_features → fit → evaluate pipeline
    without touching the database or MLflow.
    """

    @pytest.fixture
    def synthetic_dataset(self) -> pd.DataFrame:
        np.random.seed(42)
        # Span 4 months ending now — temporal_split(test_months=1) gets ~75% train, ~25% test
        end_dt   = pd.Timestamp.now().floor("h")
        start_dt = end_dt - pd.DateOffset(months=4)
        hours    = pd.date_range(start_dt, end_dt, freq="h")
        n        = len(hours)
        return pd.DataFrame({
            "hour_bucket":        hours,
            "zone_id":            np.random.randint(1, 5, n),
            "trip_count":         np.random.randint(20, 300, n),
            "hour_of_day":        hours.hour,
            "day_of_week":        hours.dayofweek,
            "month":              hours.month,
            "is_weekend":         (hours.dayofweek >= 5).astype(int),
            "temperature_f":      np.random.uniform(30, 90, n),
            "feels_like_f":       np.random.uniform(25, 95, n),
            "precipitation_mm":   np.random.exponential(1.0, n),
            "windspeed_mph":      np.random.uniform(0, 30, n),
            "humidity_pct":       np.random.uniform(30, 90, n),
            "weather_code":       np.random.choice([0, 1, 2, 61, 73], n),
            "is_raining":         np.random.randint(0, 2, n),
            "is_snowing":         np.random.randint(0, 2, n),
            "has_major_event":    np.random.randint(0, 2, n),
            "event_attendance":   np.random.randint(0, 100000, n),
            "demand_lag_24h":     np.random.uniform(50, 300, n),
            "demand_lag_168h":    np.random.uniform(50, 300, n),
            "unemployment_rate":  np.full(n, 3.8),
            "gas_price_avg":      np.full(n, 3.65),
            "consumer_sentiment": np.full(n, 76.0),
        })

    def test_features_built_from_synthetic_data(self, synthetic_dataset):
        from src.features.builder import build_features
        X = build_features(synthetic_dataset)
        assert X.shape[0] == len(synthetic_dataset)
        assert X.isnull().sum().sum() == 0

    def test_random_forest_fits_and_predicts(self, synthetic_dataset):
        from src.features.builder import build_features
        from src.models.trainer import temporal_split

        train, test = temporal_split(synthetic_dataset, test_months=1)
        X_train = build_features(train)
        y_train = train["trip_count"].values
        X_test  = build_features(test)
        y_test  = test["trip_count"].values

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        assert len(preds) == len(y_test)
        assert np.all(preds >= 0), "Predictions should be non-negative"

    def test_evaluate_on_fitted_model(self, synthetic_dataset):
        from src.features.builder import build_features
        from src.models.trainer import temporal_split

        train, test = temporal_split(synthetic_dataset, test_months=1)
        X_train = build_features(train)
        y_train = train["trip_count"].values
        X_test  = build_features(test)
        y_test  = test["trip_count"].values

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        preds   = model.predict(X_test)
        metrics = evaluate(y_test, preds)

        # R² can be slightly negative on purely random data (model close to mean)
        assert metrics["r2"] >= -1.0
        assert metrics["rmse"] > 0
        assert metrics["mae"]  > 0
        # All metrics must be finite numbers
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"
